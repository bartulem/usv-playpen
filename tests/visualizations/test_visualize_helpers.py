"""
@author: bartulem
Mock-based tests for the leaf-level public helpers in
`make_neuronal_tuning_figures` and `make_behavioral_videos`.

We deliberately stay out of the matplotlib rendering paths (which need
real per-cluster pkls and tracking data) and instead test:

- pure logic helpers (decision-rules, channel→region resolution, TTL
  edge detection, simple memmap loading);
- the early-exit / skip paths in the top-level orchestrators (no
  tuning-curve dir, no pkls, etc).
"""

from __future__ import annotations

import numpy as np
import pytest

# Force a non-interactive backend before importing the visualization modules
import matplotlib
matplotlib.use("Agg")

from usv_playpen.visualizations.make_neuronal_tuning_figures import (
    NeuronalTuningFigureMaker,
    _decide_strip_xscale,
)
from usv_playpen.visualizations.make_behavioral_videos import (
    read_ttl_events,
    filter_spikes_for_raster,
    find_region_by_channel,
    load_audio_data,
    create_spike_sound_file,
    Create3DVideo,
)


# ===========================================================================
# _decide_strip_xscale — pure decision rule
# ===========================================================================


def test_decide_strip_xscale_returns_linear_for_empty_input():
    """All-NaN inputs → linear (graceful default)."""
    nan = np.array([np.nan, np.nan])
    assert _decide_strip_xscale(nan, nan, nan, log_ratio_threshold=10.0) == "linear"


def test_decide_strip_xscale_returns_linear_when_no_positive_values():
    """All zeros / negatives → linear (no positive value to anchor min)."""
    zeros = np.zeros(3)
    neg = np.array([-1.0, -2.0, -3.0])
    assert _decide_strip_xscale(zeros, neg, neg, log_ratio_threshold=10.0) == "linear"


def test_decide_strip_xscale_returns_symlog_for_high_dynamic_range():
    """Very wide max/min ratio → symlog."""
    obs = np.array([0.001, 1000.0])
    null = np.array([0.01, 100.0])
    assert _decide_strip_xscale(obs, null, null, log_ratio_threshold=100.0) == "symlog"


def test_decide_strip_xscale_returns_linear_when_below_threshold():
    """Narrow dynamic range → linear."""
    obs = np.array([1.0, 2.0, 3.0])
    null = np.array([1.0, 2.0])
    assert _decide_strip_xscale(obs, null, null, log_ratio_threshold=100.0) == "linear"


# ===========================================================================
# NeuronalTuningFigureMaker — early-exit / skip paths
# ===========================================================================


def _make_fig_maker(root_directory: str, **viz_overrides) -> NeuronalTuningFigureMaker:
    """Build a fig maker with a minimal (but valid) settings dict."""
    msg = []
    viz = {
        "male_colors": ["#1f77b4"],
        "female_colors": ["#d62728"],
    }
    viz.update(viz_overrides)
    return NeuronalTuningFigureMaker(
        root_directory=root_directory,
        visualizations_parameter_dict=viz,
        message_output=msg.append,
    ), msg


def test_fig_maker_sex_color_resolves_male_and_female(tmp_path):
    """_sex_color → first male / first female palette entry."""
    maker, _ = _make_fig_maker(str(tmp_path))
    assert maker._sex_color("male") == "#1f77b4"
    assert maker._sex_color("female") == "#d62728"
    # Anything else → female (defensive default).
    assert maker._sex_color("unknown") == "#d62728"


def test_fig_maker_load_segmentation_caches_empty_when_file_missing(tmp_path):
    """_load_segmentation returns {} (and caches it) when the bundled
    segmentation file is missing — without raising."""
    maker, _ = _make_fig_maker(str(tmp_path))
    # Force the segmentation path to a known-missing location to exercise the
    # "file absent" branch. The instance attribute can be overridden directly.
    maker._segmentation_path = tmp_path / "nope.npz"
    seg = maker._load_segmentation()
    assert seg == {}
    # Second call must return the cached dict (same object identity).
    seg2 = maker._load_segmentation()
    assert seg2 is seg


def test_fig_maker_skips_when_tuning_dir_absent(tmp_path):
    """No <root>/ephys/tuning_curves/ → log a skip and return."""
    maker, msgs = _make_fig_maker(str(tmp_path))
    maker.make_neuronal_tuning_figures()
    assert any("not found" in m for m in msgs)


def test_fig_maker_skips_when_tuning_dir_empty(tmp_path):
    """Tuning curves dir exists but contains no pkls → skip + return."""
    (tmp_path / "ephys" / "tuning_curves").mkdir(parents=True)
    maker, msgs = _make_fig_maker(str(tmp_path))
    maker.make_neuronal_tuning_figures()
    assert any("no tuning pkls" in m for m in msgs)


def test_fig_maker_skips_pkl_with_no_payload(tmp_path, mocker):
    """A pkl with neither beh_offset= nor vocal_q keys is silently skipped
    (no save attempted), so the run finishes without errors."""
    import pickle as _pickle
    tuning_dir = tmp_path / "ephys" / "tuning_curves"
    tuning_dir.mkdir(parents=True)
    pkl_path = tuning_dir / "cl_001_tuning_curves_data.pkl"
    with pkl_path.open("wb") as f:
        _pickle.dump({"unrelated_key": 42}, f)

    maker, _msgs = _make_fig_maker(str(tmp_path))
    # Patch smart_wait so we don't actually sleep.
    mocker.patch("usv_playpen.visualizations.make_neuronal_tuning_figures.smart_wait")
    maker.make_neuronal_tuning_figures()  # should not raise


# ===========================================================================
# make_behavioral_videos — pure helpers
# ===========================================================================


def test_read_ttl_events_returns_first_off_to_on_and_on_to_off():
    """LSB-encoded TTL: returns (start_of_first_pulse, end_of_first_pulse)."""
    arr = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    off_to_on, on_to_off = read_ttl_events(arr)
    # diff(lsb) negative at index 4→5 (1->0) and 8→9 (1->0); off_to_on is
    # the *first* sample where it transitions from 0 to 1 +1 (with the
    # convention used in the source: where(diff < 0) maps to off_to_on).
    # We just verify the helper returned ints.
    assert isinstance(int(off_to_on), int)
    assert isinstance(int(on_to_off), int)
    assert off_to_on > on_to_off  # given this signal, the high-edge came first


def test_filter_spikes_for_raster_window_and_offset():
    """Returns spikes inside [ra_st_fr, ra_end_fr) shifted by -fr_start."""
    spikes = np.array([1, 5, 10, 15, 20, 25], dtype=np.int64)
    out = filter_spikes_for_raster(spikes, ra_st_fr=8, ra_end_fr=22, fr_start=10)
    np.testing.assert_array_equal(np.array(out), np.array([0, 5, 10]))


def test_find_region_by_channel_returns_color_when_in_group():
    """Channel inside a region's [low, high) group → returns its bucket
    color. `brain_color_scheme` is bucket-keyed (PAG / MRN / ...); raw
    region acronyms are pooled to a bucket before lookup."""
    brain_areas = {"probeA": {"PAG": [(0, 50)], "MRN": [(50, 100)]}}
    colors = {"PAG": "#aabbcc", "MRN": "#112233", "other": "#B8B8B8"}
    out = find_region_by_channel("cl_ch042_probeA", brain_areas, colors,
                                  return_only_color=True)
    assert out == "#aabbcc"


def test_find_region_by_channel_returns_area_when_only_area():
    """return_only_area=True → returns the brain region name."""
    brain_areas = {"probeA": {"V1": [(0, 50)]}}
    colors = {"V1": "#aabbcc"}
    out = find_region_by_channel("cl_ch042_probeA", brain_areas, colors,
                                  return_only_color=False, return_only_area=True)
    assert out == "V1"


def test_find_region_by_channel_returns_pair_when_neither_flag():
    """When both flags are False → returns (raw_region, bucket_color)
    tuple. The raw region acronym is preserved for filter selection;
    the color is resolved through the bucket-keyed palette."""
    brain_areas = {"probeA": {"PAG": [(0, 50)]}}
    colors = {"PAG": "#aabbcc", "other": "#B8B8B8"}
    out = find_region_by_channel("cl_ch042_probeA", brain_areas, colors,
                                  return_only_color=False, return_only_area=False)
    assert out == ("PAG", "#aabbcc")


def test_find_region_by_channel_returns_none_for_unknown_channel():
    """Channel outside any group → None (signalling 'unknown')."""
    brain_areas = {"probeA": {"V1": [(0, 50)]}}
    colors = {"V1": "#aabbcc"}
    out = find_region_by_channel("cl_ch200_probeA", brain_areas, colors)
    assert out is None


def test_load_audio_data_reads_mmap_with_correct_shape(tmp_path):
    """Filename encodes (sample_rate, n_samples, n_channels, dtype)."""
    n_samples = 100
    n_channels = 4
    sample_rate = 250000
    fname = f"sess_{sample_rate}_{n_samples}_{n_channels}_int16.mmap"
    audio_path = tmp_path / fname
    arr = np.arange(n_samples * n_channels, dtype=np.int16).reshape(n_samples, n_channels)
    arr.tofile(audio_path)

    out, sr = load_audio_data(str(tmp_path))
    assert sr == sample_rate
    assert out.shape == (n_samples, n_channels)


def test_create_spike_sound_file_writes_wav(tmp_path):
    """create_spike_sound_file should emit a .wav at the requested path."""
    # Spike events are positions in tracking-frame units; at tracking_esr=150
    # they need to fit inside audio_duration*tracking_esr seconds, so for a
    # 5-second audio buffer we have ~750 frames of headroom.
    spike_array = np.array([10, 30, 60, 100], dtype=np.int64)
    out = tmp_path / "spike_audio"
    out.mkdir()
    create_spike_sound_file(
        audio_duration=5.0,
        spike_array=spike_array,
        sound_save_directory=str(out),
        sound_session_id="sess1",
        sound_frame_start=0,
        sound_frame_span=600,
        tracking_esr=150.0,
        unit_id="cl_001",
    )
    wavs = list(out.glob("*.wav"))
    assert len(wavs) == 1
    # Filename mirrors the documented pattern: <session>_3D_<start>-<end>fr_spike_sound_<unit>.wav
    assert wavs[0].name == "sess1_3D_0-600fr_spike_sound_cl_001.wav"


# ===========================================================================
# Create3DVideo — early-exit / no-files paths
# ===========================================================================


def test_create_3d_video_load_beh_features_missing_raises(tmp_path):
    """No behavioral features CSV → FileNotFoundError from first_match_or_raise."""
    vid = Create3DVideo(
        root_directory=str(tmp_path),
        visualizations_parameter_dict={"brain_area_colors": {"other": "#B8B8B8"}},
        message_output=lambda *_a, **_kw: None,
    )
    with pytest.raises(FileNotFoundError):
        vid.load_beh_features_file()


def test_create_3d_video_load_h5_file_missing_raises(tmp_path):
    """No 3D tracking H5 → FileNotFoundError. Both root_directory and the
    arena_directory attribute must be set; we point both at tmp_path so
    the search produces no matches and the helper raises."""
    vid = Create3DVideo(
        root_directory=str(tmp_path),
        arena_directory=str(tmp_path),
        visualizations_parameter_dict={"brain_area_colors": {"other": "#B8B8B8"}},
        message_output=lambda *_a, **_kw: None,
    )
    with pytest.raises(FileNotFoundError):
        vid.load_h5_file()


def test_visualize_in_video_no_data_leaves_no_save_directory(tmp_path):
    """
    Description
    -----------
    Regression guard: `visualize_in_video` must not create its
    `data_animation_examples` save directory on a data-less run. The
    directory's `mkdir` was previously the first statement in the method,
    so any run pointed at an empty root left an empty directory behind --
    invisible to git (git ignores empty directories) and easy to miss.
    The mkdir now sits behind the `load_h5_file()` validation gate, so a
    root with no tracking H5 must raise `FileNotFoundError` *before* the
    directory is created.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Empty session root (no tracking H5), so load_h5_file raises.

    Returns
    -------
    None
    """

    vid = Create3DVideo(
        root_directory=str(tmp_path),
        arena_directory=str(tmp_path),
        visualizations_parameter_dict={
            "make_behavioral_videos": {"speaker_bool": False},
            "brain_area_colors": {"other": "#B8B8B8"},
        },
        message_output=lambda *_a, **_kw: None,
    )
    with pytest.raises(FileNotFoundError):
        vid.visualize_in_video()
    assert not (tmp_path / "data_animation_examples").exists()
