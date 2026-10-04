"""
@author: bartulem
Tests for visualizations/make_vocal_pose_figures.py.

Coverage:
- the trail fade (decay, floor, taper to nothing at the trail's end)
- colour helpers (hex round trip, blending toward a ground colour)
- the male-side camera rule and the peak-microphone channel mapping
- the candidate-window finder on a synthetic session (male-only windows pass,
  windows touching a female call, a noise detection or a cut call do not)
- the still maker end to end on the synthetic session (PNG + SVG written under tmp_path)
- the video maker end to end when ffmpeg is present (frames pooled, clip encoded)
- the view-picker page (data and title embedded)
"""

from __future__ import annotations

import copy
import itertools
import json
import pathlib
import shutil

import matplotlib

matplotlib.use("Agg")
import h5py
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import pytest
import soundfile as sf

import usv_playpen
from usv_playpen.visualizations.make_behavioral_videos import (
    MOUSE_NODE_CONNECTIONS,
    MOUSE_NODE_POLYGONS,
)
from usv_playpen.visualizations.make_vocal_pose_figures import (
    VocalPoseStillMaker,
    VocalPoseVideoMaker,
    blend_colors,
    find_vocal_pose_windows,
    hex_to_rgb,
    male_side_azimuth,
    peak_microphone,
    plot_vocal_pose_window_candidates,
    rgb_to_hex,
    trail_strength,
    vocal_pose_view_picker_html,
)

_NODES = ["Nose", "Ear_R", "Ear_L", "TTI", "TailTip", "Head", "Trunk", "Tail_0", "Tail_1", "Tail_2",
          "Shoulder_left", "Shoulder_right", "Haunch_left", "Haunch_right", "Neck"]
_FPS = 30.0
_SESSION_SECONDS = 12.0
_WAV_RATE = 250000


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Description
    -----------
    Close every figure after each test so the open-figure warning cannot trip warnings-as-errors.

    Parameters
    ----------

    Returns
    -------
    None
    """

    yield
    plt.close("all")


def _mouse_pose(centre_xy: np.ndarray, heading: float, scale: float = 1.0) -> np.ndarray:
    """
    Description
    -----------
    One plausible mouse pose (all 15 nodes) at a floor position, facing a heading.

    Parameters
    ----------
    centre_xy (np.ndarray)
        Body centre on the floor (m).
    heading (float)
        Direction the nose points (radians).
    scale (float)
        Body-size multiplier.

    Returns
    -------
    pose (np.ndarray)
        (15, 3) node coordinates (m).
    """

    forward = np.array([np.cos(heading), np.sin(heading)])
    left = np.array([-np.sin(heading), np.cos(heading)])
    layout = {
        "Nose": (0.045, 0.0, 0.025), "Head": (0.03, 0.0, 0.03), "Ear_L": (0.03, 0.01, 0.03), "Ear_R": (0.03, -0.01, 0.03),
        "Neck": (0.02, 0.0, 0.03), "Shoulder_left": (0.015, 0.012, 0.025), "Shoulder_right": (0.015, -0.012, 0.025),
        "Trunk": (0.0, 0.0, 0.03), "Haunch_left": (-0.02, 0.013, 0.025), "Haunch_right": (-0.02, -0.013, 0.025),
        "TTI": (-0.035, 0.0, 0.02), "Tail_0": (-0.05, 0.0, 0.01), "Tail_1": (-0.07, 0.0, 0.005), "Tail_2": (-0.09, 0.0, 0.003),
        "TailTip": (-0.11, 0.0, 0.002),
    }
    pose = np.zeros((len(_NODES), 3))
    for index, name in enumerate(_NODES):
        along, across, height = layout[name]
        pose[index, :2] = centre_xy + scale * (along * forward + across * left)
        pose[index, 2] = scale * height
    return pose


def _write_session(root: pathlib.Path) -> pathlib.Path:
    """
    Description
    -----------
    Write a synthetic session in the on-disk shape the module reads: the mouse points3d .h5 (two
    animals, 15 nodes, 30 fps, 12 s, experimental code of a male-female courtship pair), a USV summary
    (male USVs at 3-5 s, a female squeak at 7 s, a noise detection at 8 s, male USVs again at 9-11 s)
    and one microphone's video-cropped wav with a tone burst at each call.

    Parameters
    ----------
    root (pathlib.Path)
        Directory to create the session in.

    Returns
    -------
    session (pathlib.Path)
        The session directory.
    """

    session = root / "20240101_120000"
    (session / "video" / "20240101120000").mkdir(parents=True)
    (session / "audio" / "cropped_to_video").mkdir(parents=True)
    n_frames = int(_SESSION_SECONDS * _FPS)
    times = np.arange(n_frames) / _FPS
    tracks = np.zeros((n_frames, 2, len(_NODES), 3))
    for frame, t in enumerate(times):
        female_centre = np.array([0.02 * t, 0.0])
        tracks[frame, 1] = _mouse_pose(female_centre, 0.0)
        tracks[frame, 0] = _mouse_pose(female_centre + np.array([-0.02, -0.03]), 0.0)
    with h5py.File(session / "video" / "20240101120000" / "20240101120000_points3d_translated_rotated_metric.h5", "w") as h5_file:
        h5_file.create_dataset("tracks", data=tracks)
        h5_file.create_dataset("track_names", data=np.array([b"100_1", b"200_2"]))
        h5_file.create_dataset("node_names", data=np.array([name.encode("utf-8") for name in _NODES]))
        h5_file.create_dataset("recording_frame_rate", data=_FPS)
        h5_file.create_dataset("experimental_code", data=b"BCL2MGFGd")
    # each animal's sex is read from the session metadata, matched to its track name
    (session / f"{session.name}_metadata.yaml").write_text(
        "Subjects:\n- subject_id: '100_1'\n  sex: male\n- subject_id: '200_2'\n  sex: female\n")
    rows = []
    for start in np.concatenate([np.arange(3.0, 5.0, 0.2), np.arange(9.0, 11.0, 0.2)]):
        rows.append({"start": start, "stop": start + 0.08, "duration": 0.08, "peak_amp_ch": 0, "emitter": "100_1", "noise": False, "usv": True,
                     "squeak": False, "freq_bandwidth_hz": 20000.0})
    rows.append({"start": 7.0, "stop": 7.2, "duration": 0.2, "peak_amp_ch": 0, "emitter": "200_2", "noise": False, "usv": False, "squeak": True,
                 "freq_bandwidth_hz": 60000.0})
    rows.append({"start": 8.0, "stop": 8.05, "duration": 0.05, "peak_amp_ch": 0, "emitter": None, "noise": True, "usv": None, "squeak": None,
                 "freq_bandwidth_hz": 5000.0})
    pls.DataFrame(rows).with_columns(pls.col("emitter").cast(pls.Utf8)).write_csv(session / "audio" / "20240101_120000_usv_summary.csv")
    audio_times = np.arange(int(_SESSION_SECONDS * _WAV_RATE)) / _WAV_RATE
    audio = np.zeros(audio_times.size, dtype=np.float32)
    for row in rows:
        inside = (audio_times >= row["start"]) & (audio_times < row["stop"])
        audio[inside] = 0.5 * np.sin(2 * np.pi * 60000.0 * audio_times[inside]).astype(np.float32)
    sf.write(session / "audio" / "cropped_to_video" / "m_240101120000_ch01_cropped_to_video.wav", audio, _WAV_RATE)
    return session


def _settings(save_directory: pathlib.Path) -> dict:
    """
    Description
    -----------
    The shipped visualizations settings with the figure directory redirected to a temporary
    directory and the vocal-pose knobs set for the small synthetic session.

    Parameters
    ----------
    save_directory (pathlib.Path)
        Where figures are written.

    Returns
    -------
    settings (dict)
        A deep copy of the shipped settings with the overrides applied.
    """

    with (pathlib.Path(usv_playpen.__file__).parent / "_parameter_settings" / "visualizations_settings.json").open("r") as settings_file:
        settings = copy.deepcopy(json.load(settings_file))
    settings["figures"]["save_directory"] = str(save_directory)
    settings["figures"]["timestamp_in_name"] = False
    block = settings["vocal_pose_figures"]
    block["window"]["end_time"] = 5.0
    block["window"]["history_seconds"] = 2.0
    block["spectrogram"]["freq_range_khz"] = [30.0, 110.0]
    block["output"]["dpi"] = 50
    block["video"]["dpi"] = 40
    block["video"]["duration_seconds"] = 0.3
    block["video"]["workers"] = 1
    # The synthetic animals fill the small panel, so the key goes beside the spectrogram by default.
    block["layout"]["key_position"] = "spectrogram"
    return settings


class TestTrailStrength:
    """Shape of the fade that the still's trail, time bar and the video's trail all share."""

    def test_decays_from_strongest_toward_floor(self):
        """The newest silhouette carries the full strength and old ones settle at the floor."""

        trail = {"strongest": 0.6, "fade_seconds": 0.1, "floor": 0.05, "taper_fraction": 0.5}
        strength = trail_strength(np.array([0.0, 0.1, 10.0]), trail, 0.0)
        assert strength[0] == pytest.approx(0.6)
        assert strength[1] == pytest.approx(0.05 + 0.55 * np.exp(-1.0))
        assert strength[2] == pytest.approx(0.05)
        assert np.all(np.diff(strength) <= 0)

    def test_tapers_to_nothing_at_the_trail_end(self):
        """With an end set, the strength is untouched before the taper and exactly zero at the end."""

        trail = {"strongest": 0.6, "fade_seconds": 0.1, "floor": 0.05, "taper_fraction": 0.5}
        assert trail_strength(0.4, trail, 1.0) == pytest.approx(trail_strength(0.4, trail, 0.0))
        assert trail_strength(0.75, trail, 1.0) == pytest.approx(0.5 * trail_strength(0.75, trail, 0.0))
        assert trail_strength(1.0, trail, 1.0) == pytest.approx(0.0)
        assert trail_strength(2.0, trail, 1.0) == pytest.approx(0.0)


class TestColorHelpers:
    """Hex colours in, hex colours out, blended toward whatever the trail fades into."""

    def test_hex_round_trip(self):
        """A hex colour survives a trip through RGB."""

        assert rgb_to_hex(hex_to_rgb("#9AC0CD")) == "#9AC0CD"

    def test_blend_endpoints(self):
        """Strength 1 keeps the colour, 0 gives the ground colour, and the middle sits between."""

        assert blend_colors("#FF6347", "#FFFFFF", 1.0) == "#FF6347"
        assert blend_colors("#FF6347", "#FFFFFF", 0.0) == "#FFFFFF"
        assert blend_colors("#000000", "#EEEEEE", 0.5) == "#777777"


class TestGeometryHelpers:
    """The camera rule and the microphone index mapping."""

    def test_male_side_azimuth_points_from_female_to_male(self):
        """With the male straight along +x from the female, the camera azimuth is 0; along +y it is 90."""

        names, sex_of = ["m", "f"], {"m": "male", "f": "female"}
        pose = np.stack([_mouse_pose(np.array([0.1, 0.0]), 0.0), _mouse_pose(np.array([0.0, 0.0]), 0.0)])
        assert male_side_azimuth(pose, names, _NODES, sex_of) == pytest.approx(0.0)
        pose = np.stack([_mouse_pose(np.array([0.0, 0.1]), 0.0), _mouse_pose(np.array([0.0, 0.0]), 0.0)])
        assert male_side_azimuth(pose, names, _NODES, sex_of) == pytest.approx(90.0)

    def test_peak_microphone_maps_the_24_channel_index(self):
        """Channels 0-11 are device m 1-12 and 12-23 device s 1-12; the majority wins."""

        assert peak_microphone(pls.DataFrame({"peak_amp_ch": [2.0, 2.0, 15.0]})) == "m03"
        assert peak_microphone(pls.DataFrame({"peak_amp_ch": [15.0, 15.0, 2.0]})) == "s04"

    def test_skeleton_lists_name_only_known_nodes(self):
        """Every node the shared skeleton draws exists in the pose file's node list."""

        for pair in (*MOUSE_NODE_CONNECTIONS, *MOUSE_NODE_POLYGONS):
            for name in pair.split("-"):
                assert name in _NODES


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
class TestOnSyntheticSession:
    """The window finder, still, video and picker on one small on-disk session."""

    def test_find_windows_keeps_only_clean_male_windows(self, tmp_path):
        """Windows inside the male-only stretches qualify; any touching the squeak, the noise detection or a cut call do not."""

        session = _write_session(tmp_path)
        settings = _settings(tmp_path / "figures")
        windows = find_vocal_pose_windows(str(session), settings, message_output=lambda *_: None)
        assert windows.height > 0
        for row in windows.iter_rows(named=True):
            assert row["n_calls"] > 0
            assert not (row["start"] < 7.2 and row["end"] > 7.0), "a window overlapping the female squeak qualified"
            assert not (row["start"] < 8.05 and row["end"] > 8.0), "a window overlapping the noise detection qualified"
            assert row["gap_max_cm"] <= settings["vocal_pose_figures"]["candidates"]["max_gap_cm"]
        assert windows["score"].to_numpy()[0] == windows["score"].max()
        fig = plot_vocal_pose_window_candidates(str(session), windows, settings, message_output=lambda *_: None)
        # The sheet shows the best windows that do not overlap one another, so fewer than are listed.
        assert 1 <= len(fig.axes) <= min(settings["vocal_pose_figures"]["candidates"]["n_shown"], windows.height)
        starts = sorted(float(axis.get_title(loc="left").split()[1]) for axis in fig.axes)
        assert all(later - earlier >= settings["vocal_pose_figures"]["candidates"]["window_seconds"] for earlier, later in itertools.pairwise(starts))

    def test_find_windows_needs_pure_usvs_whatever_the_flag_typing(self, tmp_path):
        """A male call detect-usv-squeaks left unscored (null booleans, not noise) is not a pure USV,
        so no window holding it qualifies; with the flags written as text the clean windows are found
        unchanged."""

        session = _write_session(tmp_path)
        settings = _settings(tmp_path / "figures")
        summary_path = session / "audio" / "20240101_120000_usv_summary.csv"
        summary = pls.read_csv(summary_path)
        clean = find_vocal_pose_windows(str(session), settings, message_output=lambda *_: None)
        as_text = summary.with_columns(pls.col("usv").cast(pls.Utf8), pls.col("squeak").cast(pls.Utf8))
        as_text.write_csv(summary_path)
        assert find_vocal_pose_windows(str(session), settings, message_output=lambda *_: None).height == clean.height
        unscored = summary.with_columns(
            pls.when(pls.col("start") == pls.col("start").min()).then(None).otherwise(pls.col("usv")).alias("usv"),
            pls.when(pls.col("start") == pls.col("start").min()).then(None).otherwise(pls.col("squeak")).alias("squeak"),
        )
        unscored.write_csv(summary_path)
        first_call = float(summary["start"].min())
        for row in find_vocal_pose_windows(str(session), settings, message_output=lambda *_: None).iter_rows(named=True):
            assert not (row["start"] <= first_call < row["end"]), "a window holding an unscored call qualified"

    def test_still_writes_png_and_svg(self, tmp_path):
        """The still maker renders the synthetic session and writes both files where the figures block points."""

        session = _write_session(tmp_path)
        settings = _settings(tmp_path / "figures")
        settings["vocal_pose_figures"]["output"]["save_svg"] = True
        messages = []
        paths = VocalPoseStillMaker(str(session), settings, message_output=messages.append).make_vocal_pose_still()
        assert paths["png"].is_file()
        assert paths["svg"].is_file()
        assert paths["png"].parent == tmp_path / "figures"
        assert "vocal_pose_20240101_120000_3.00-5.00s" in paths["png"].name
        assert any("camera azimuth" in message for message in messages)
        assert b"<svg" in paths["svg"].read_bytes()[:400]

    def test_still_with_fixed_camera_and_no_key(self, tmp_path):
        """A numeric azimuth, no colour key, no alignment and no scale mark are honoured."""

        session = _write_session(tmp_path)
        settings = _settings(tmp_path / "figures")
        block = settings["vocal_pose_figures"]
        block["camera"]["azimuth"] = 30.0
        block["layout"]["color_key"] = False
        block["layout"]["align_pose"] = False
        block["scale_mark"]["length_cm"] = 0.0
        paths = VocalPoseStillMaker(str(session), settings, message_output=lambda *_: None).make_vocal_pose_still()
        assert paths["png"].is_file()
        assert "svg" not in paths

    def test_key_beside_animals_renders_or_names_the_way_out(self, tmp_path):
        """The key beside the animals renders when there is room, and a key too wide to fit fails with a message naming the setting to change."""

        session = _write_session(tmp_path)
        settings = _settings(tmp_path / "figures")
        settings["vocal_pose_figures"]["layout"]["key_position"] = "pose"
        paths = VocalPoseStillMaker(str(session), settings, message_output=lambda *_: None).make_vocal_pose_still()
        assert paths["png"].is_file()
        settings["vocal_pose_figures"]["layout"]["key_wide_inches"] = 10.0
        with pytest.raises(RuntimeError, match="key_position"):
            VocalPoseStillMaker(str(session), settings, message_output=lambda *_: None).make_vocal_pose_still()

    @pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")
    def test_video_writes_clip(self, tmp_path):
        """The video maker renders a short clip with the spectrogram strip below and encodes it."""

        session = _write_session(tmp_path)
        settings = _settings(tmp_path / "figures")
        settings["vocal_pose_figures"]["video"]["spectrogram"]["layout"] = "below"
        path = VocalPoseVideoMaker(str(session), settings, message_output=lambda *_: None).make_vocal_pose_video()
        assert path.is_file()
        assert path.suffix == ".mp4"
        assert path.stat().st_size > 0
        assert not (path.parent / f"{path.stem}_frames").exists()

    def test_view_picker_embeds_the_frame(self, tmp_path):
        """The picker page carries the title, the session and the pose data."""

        session = _write_session(tmp_path)
        settings = _settings(tmp_path / "figures")
        html = vocal_pose_view_picker_html(str(session), settings, title="Pick the view")
        assert "Pick the view" in html
        assert "20240101_120000" in html
        assert '"presets"' in html
        assert "Male side" in html
