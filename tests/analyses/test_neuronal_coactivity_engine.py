"""
@author: bartulem
Tests for ``usv_playpen.analyses.neuronal_coactivity_engine``.

Covers the deterministic building blocks (binary-search snippet counts,
the three coactivity metrics, the joint circular shift, sliding-window
sweep) against hand-computed values, and the five significance routines
(bootstrap / circular-shuffle / chained-circular-shuffle / label-
permutation / onset-sampling). The significance tests assert that the
``seed`` parameter makes results reproducible — locking in the N2 seed
plumbing so coactivity significance is no longer dependent on global
RNG entropy.
"""

from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pandas as pd
import polars as pls
import pytest

from usv_playpen.analyses import neuronal_coactivity_engine as engine


def _neural_data():
    """
    Description
    -----------
    Three neurons with irregular (seeded-random) spike trains spanning
    a 100 s session, sorted as the engine requires. Irregular rates
    keep per-window spike counts variable across trials so the
    correlation metrics stay finite (no all-NaN degenerate matrices).

    Returns
    -------
    dict[str, numpy.ndarray]
        Spike-time arrays keyed by neuron id.
    """

    rng = np.random.default_rng(0)
    return {
        "n0": np.sort(rng.uniform(0.0, 100.0, 40)),
        "n1": np.sort(rng.uniform(0.0, 100.0, 30)),
        "n2": np.sort(rng.uniform(0.0, 100.0, 50)),
    }


def test_extract_snippet_matrix_counts_by_binary_search():
    """
    Description
    -----------
    Each cell counts the spikes falling in ``[onset, onset + window]``
    via the searchsorted left/right pair. For neuron ``a`` with spikes
    ``[0.1, 0.5, 0.9, 1.5]`` and window 0.5, onset 0 captures
    ``{0.1, 0.5}`` (2) and onset 1.0 captures ``{1.5}`` (1).
    """

    neural_data = {
        "a": np.array([0.1, 0.5, 0.9, 1.5]),
        "b": np.array([0.2, 0.6]),
    }
    matrix = engine.extract_snippet_matrix(np.array([0.0, 1.0]), neural_data, 0.5)
    np.testing.assert_array_equal(matrix, [[2, 1], [1, 0]])


def test_extract_snippet_matrix_no_trials_returns_empty_columns():
    """
    Description
    -----------
    With zero onsets the function short-circuits to a
    ``(n_neurons, 0)`` zero matrix.
    """

    matrix = engine.extract_snippet_matrix(np.array([]), _neural_data(), 1.0)
    assert matrix.shape == (3, 0)


def test_compute_coactivity_metrics_too_few_neurons_is_nan():
    """
    Description
    -----------
    Pairwise correlation is undefined for a single neuron, so all three
    metrics are NaN when the matrix has fewer than two rows.
    """

    metrics = engine.compute_coactivity_metrics(np.array([[1.0, 2.0, 3.0]]))
    assert all(np.isnan(metrics[k]) for k in ("r_sc", "similarity", "pop_corr"))


def test_compute_coactivity_metrics_perfectly_correlated_rows():
    """
    Description
    -----------
    Two perfectly correlated neurons (row 2 = 2 × row 1) give a mean
    pairwise spike-count correlation of exactly 1, and the metric dict
    exposes the three expected keys.
    """

    matrix = np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    metrics = engine.compute_coactivity_metrics(matrix)
    assert set(metrics) == {"r_sc", "similarity", "pop_corr"}
    assert metrics["r_sc"] == pytest.approx(1.0)


def test_apply_circular_shift_wraps_and_resorts():
    """
    Description
    -----------
    Every spike is shifted by a constant and wrapped modulo the session
    duration, then re-sorted: ``[1, 5, 9] + 3 (mod 10) = [4, 8, 2]``
    sorts to ``[2, 4, 8]``.
    """

    shifted = engine.apply_circular_shift({"a": np.array([1.0, 5.0, 9.0])}, 3.0, 10.0)
    np.testing.assert_array_equal(shifted["a"], [2.0, 4.0, 8.0])


def test_apply_circular_shift_matches_sort_reference():
    """The O(log N) rotation must equal the reference np.sort((spikes+shift)%dur)
    for arbitrary sorted inputs and shifts, including no-wrap, half, near-full, and
    over-duration (modulo) shifts."""
    rng = np.random.default_rng(0)
    dur = 100.0
    spikes = np.sort(rng.uniform(0.0, dur, size=500))
    for shift in (0.0, 7.3, dur / 2.0, dur - 1e-6, 250.0):
        ref = np.sort((spikes + shift) % dur)
        got = engine.apply_circular_shift({"a": spikes}, shift, dur)["a"]
        np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_compute_sliding_coactivity_shapes_and_time_bins():
    """
    Description
    -----------
    The sliding sweep returns ``time_bins`` at multiples of the step
    plus per-bin ``r_sc`` / ``similarity`` arrays, all of length
    ``n_steps``.
    """

    out = engine.compute_sliding_coactivity(
        np.array([10.0, 30.0, 50.0]), _neural_data(),
        window_s=5.0, step_s=2.0, n_steps=4,
    )
    np.testing.assert_array_equal(out["time_bins"], [0.0, 2.0, 4.0, 6.0])
    assert out["r_sc"].shape == (4,)
    assert out["similarity"].shape == (4,)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_bootstrap_distribution_is_seed_reproducible():
    """
    Description
    -----------
    Two bootstrap runs with the same ``seed`` produce identical
    per-metric distributions, while a different seed diverges — the N2
    reproducibility guarantee.
    """

    matrix = engine.extract_snippet_matrix(
        np.array([10.0, 30.0, 50.0, 70.0, 90.0]), _neural_data(), 5.0,
    )
    a = engine.bootstrap_coactivity_distribution(matrix, n_target=5, n_iterations=16, seed=42)
    b = engine.bootstrap_coactivity_distribution(matrix, n_target=5, n_iterations=16, seed=42)
    c = engine.bootstrap_coactivity_distribution(matrix, n_target=5, n_iterations=16, seed=43)
    for key in ("r_sc", "similarity", "pop_corr"):
        np.testing.assert_array_equal(a[key], b[key])
    assert not np.array_equal(a["r_sc"], c["r_sc"])


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_circular_shuffle_is_seed_reproducible():
    """
    Description
    -----------
    The joint circular shuffle yields identical null distributions for
    equal seeds and length-``n_shuffles`` arrays for every metric.
    """

    onsets = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    kwargs = dict(total_duration=100.0, window_s=5.0, n_shuffles=12)
    a = engine.perform_circular_shuffle(onsets, _neural_data(), seed=7, **kwargs)
    b = engine.perform_circular_shuffle(onsets, _neural_data(), seed=7, **kwargs)
    c = engine.perform_circular_shuffle(onsets, _neural_data(), seed=8, **kwargs)
    for key in ("r_sc", "similarity", "pop_corr"):
        np.testing.assert_array_equal(a[key], b[key])
        assert a[key].shape == (12,)
    # A different seed must produce a different null (at least one metric),
    # otherwise the seed is not actually driving the shuffle.
    assert any(
        not np.array_equal(a[key], c[key])
        for key in ("r_sc", "similarity", "pop_corr")
    )


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
def test_chained_circular_shuffle_is_seed_reproducible():
    """
    Description
    -----------
    The multi-session chained shuffle concatenates per-session shuffled
    matrices and is reproducible under a fixed ``seed``.
    """

    onsets = np.array([10.0, 30.0, 50.0, 70.0, 90.0])
    nd = _neural_data()
    kwargs = dict(
        session_onsets=[onsets, onsets],
        session_neural_data=[nd, nd],
        session_durations=[100.0, 100.0],
        window_s=5.0,
        n_shuffles=10,
    )
    a = engine.perform_chained_circular_shuffle(seed=3, **kwargs)
    b = engine.perform_chained_circular_shuffle(seed=3, **kwargs)
    c = engine.perform_chained_circular_shuffle(seed=4, **kwargs)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
    # A different seed must diverge on at least one metric.
    assert any(not np.array_equal(a[key], c[key]) for key in a)


@pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:invalid value encountered:RuntimeWarning")
@pytest.mark.filterwarnings("ignore:Degrees of freedom <= 0:RuntimeWarning")
def test_label_permutation_test_structure_and_reproducibility():
    """
    Description
    -----------
    The label-permutation test returns, per metric, the observed delta,
    the null array, the two p-values and a z-score; the null is
    reproducible under a fixed ``seed``.
    """

    nd = _neural_data()
    counts_a = engine.extract_snippet_matrix(np.array([10.0, 30.0, 50.0]), nd, 5.0)
    counts_b = engine.extract_snippet_matrix(np.array([20.0, 40.0, 60.0, 80.0]), nd, 5.0)
    a = engine.perform_label_permutation_test(counts_a, counts_b, n_permutations=20, seed=11)
    b = engine.perform_label_permutation_test(counts_a, counts_b, n_permutations=20, seed=11)
    c = engine.perform_label_permutation_test(counts_a, counts_b, n_permutations=20, seed=12)
    for key in ("r_sc", "similarity", "pop_corr"):
        entry = a[key]
        assert set(entry) >= {
            "observed_delta", "null", "p_a_gt_b", "p_two_tailed", "z_score",
        }
        assert entry["null"].shape == (20,)
        assert 0.0 <= entry["p_a_gt_b"] <= 1.0
        np.testing.assert_array_equal(entry["null"], b[key]["null"])
    # A different seed must produce a different null on at least one metric.
    assert any(
        not np.array_equal(a[key]["null"], c[key]["null"])
        for key in ("r_sc", "similarity", "pop_corr")
    )


def _sessions_for_sampling():
    """
    Description
    -----------
    Two sessions, each exposing a ``group`` table with a ``start``
    column — the structure :func:`sample_onsets_across_sessions`
    consumes.

    Returns
    -------
    list[dict]
        Session dicts pooling 10 candidate onsets.
    """

    return [
        {"group": pd.DataFrame({"start": [1.0, 2.0, 3.0, 4.0, 5.0]})},
        {"group": pd.DataFrame({"start": [6.0, 7.0, 8.0, 9.0, 10.0]})},
    ]


def test_sample_onsets_across_sessions_is_seed_reproducible():
    """
    Description
    -----------
    Sampling the same total count with the same ``seed`` redistributes
    the identical onsets back into the same per-session buckets, and
    each bucket is returned sorted.
    """

    sessions = _sessions_for_sampling()
    a = engine.sample_onsets_across_sessions(sessions, "group", n_total=4, seed=5)
    b = engine.sample_onsets_across_sessions(sessions, "group", n_total=4, seed=5)
    assert len(a) == 2
    for arr_a, arr_b in zip(a, b):
        np.testing.assert_array_equal(arr_a, arr_b)
        np.testing.assert_array_equal(arr_a, np.sort(arr_a))
    assert sum(arr.size for arr in a) == 4
    # A different seed must redistribute the onsets differently. The pool is
    # small, so guard against a coincidental match by requiring at least one
    # of several alternative seeds to diverge.
    a_flat = np.concatenate([np.sort(arr) for arr in a])
    assert any(
        not np.array_equal(
            a_flat,
            np.concatenate([
                np.sort(arr)
                for arr in engine.sample_onsets_across_sessions(
                    sessions, "group", n_total=4, seed=alt_seed
                )
            ]),
        )
        for alt_seed in (6, 7, 8, 9)
    )


def test_sample_onsets_across_sessions_rejects_oversized_request():
    """
    Description
    -----------
    Requesting more onsets than the pooled total raises ``ValueError``
    naming the available and target counts.
    """

    with pytest.raises(ValueError, match="less than target N"):
        engine.sample_onsets_across_sessions(_sessions_for_sampling(), "group", n_total=99)


def _write_tone_mmap(
    tmp_path,
    *,
    sampling_rate=250000,
    n_channels=4,
    n_samples=250000,
    loud_channel=2,
    f0=60000.0,
    loud_amp=10000,
    quiet_amp=200,
    tone_lo_s=0.10,
    tone_hi_s=0.16,
):
    """
    Description
    -----------
    Writes a synthetic concatenated int16 audio memmap under ``<tmp_path>/audio``
    whose filename encodes the ``_<sr>_<n_samples>_<n_ch>_int16.mmap`` metadata the
    helper parses. The loud channel carries a pure ``f0`` tone of int16 amplitude
    ``loud_amp`` ONLY within ``[tone_lo_s, tone_hi_s)`` (silence elsewhere on it);
    every other channel carries a quieter ``f0`` tone (amplitude ``quiet_amp``)
    throughout. This lets the test exercise channel selection, the onset-anchored
    window, and end-of-session clamping.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        pytest temporary directory (the session root).
    sampling_rate, n_channels, n_samples : int
        Audio memmap geometry.
    loud_channel : int
        Channel that carries the loud, time-limited tone.
    f0 : float
        Tone frequency (Hz).
    loud_amp, quiet_amp : int
        int16 amplitudes of the loud and quiet tones.
    tone_lo_s, tone_hi_s : float
        Start/stop (s) of the loud tone on ``loud_channel``.

    Returns
    -------
    session_root (pathlib.Path)
        ``tmp_path`` (the directory containing ``audio/``).
    """

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    t = np.arange(n_samples) / sampling_rate
    arr = np.zeros((n_samples, n_channels), dtype=np.int16)
    quiet = (quiet_amp * np.sin(2 * np.pi * f0 * t)).astype(np.int16)
    for ch in range(n_channels):
        arr[:, ch] = quiet
    s_lo, s_hi = round(tone_lo_s * sampling_rate), round(tone_hi_s * sampling_rate)
    loud = (loud_amp * np.sin(2 * np.pi * f0 * t)).astype(np.int16)
    arr[:, loud_channel] = 0
    arr[s_lo:s_hi, loud_channel] = loud[s_lo:s_hi]
    name = f"sess_concatenated_audio_{sampling_rate}_{n_samples}_{n_channels}_int16.mmap"
    arr.tofile(audio_dir / name)
    return tmp_path


def test_extract_snippet_acoustics_tone(tmp_path):
    """
    Description
    -----------
    The amplitude / frequency features are correct for a known pure tone: RMS
    equals ``A/sqrt(2)`` of the int16->float waveform, the peak frequency lands on
    the ``f0`` bin, and the mean frequency / bandwidth are sensible. Also verifies
    that the helper reads the SUPPLIED loudest channel (not another), that the
    window is anchored at the onset, and that an out-of-tone window reads silence.
    """

    sampling_rate, f0, loud_amp, quiet_amp = 250000, 60000.0, 10000, 200
    root = _write_tone_mmap(
        tmp_path, sampling_rate=sampling_rate, loud_channel=2, f0=f0,
        loud_amp=loud_amp, quiet_amp=quiet_amp, tone_lo_s=0.10, tone_hi_s=0.16,
    )

    # (0) loud channel, window inside the tone; (1) same onset but the quiet
    # channel 0; (2) loud channel but a silent stretch (window past the tone).
    onsets = np.array([0.10, 0.10, 0.50])
    peak_channels = np.array([2.0, 0.0, 2.0])
    out = engine.extract_snippet_acoustics(str(root), onsets, peak_channels, 0.030)

    bin_hz = sampling_rate / 2048
    # (0) loud-channel tone. The energy-weighted (linear-power) features track a pure
    # tone tightly: RMS is exact, peak + mean frequency land on f0, and the bandwidth
    # is only a few bins wide.
    assert out["rms"][0] == pytest.approx((loud_amp / 32767) / np.sqrt(2), rel=2e-2)
    assert abs(out["peak_freq_hz"][0] - f0) <= bin_hz
    assert abs(out["mean_freq_hz"][0] - f0) <= 5 * bin_hz
    assert 0.0 < out["freq_bandwidth_hz"][0] <= 10 * bin_hz
    # (1) quiet channel selected -> much smaller RMS (channel indexing matters)
    assert out["rms"][1] == pytest.approx((quiet_amp / 32767) / np.sqrt(2), rel=2e-2)
    assert out["rms"][1] < out["rms"][0]
    # (2) onset-anchored window lands in silence on the loud channel
    assert out["rms"][2] < 1e-3
    assert np.isnan(out["peak_freq_hz"][2])


def test_extract_snippet_acoustics_clamps_and_empty(tmp_path):
    """
    Description
    -----------
    A window running past the session end is clamped (no crash; RMS still finite,
    frequency features NaN when the clamped snippet is shorter than one STFT
    window), and an empty onset list returns empty arrays without touching the
    audio file.
    """

    sampling_rate, n_samples = 250000, 250000
    root = _write_tone_mmap(tmp_path, sampling_rate=sampling_rate, n_samples=n_samples)

    # onset 5 ms before the end -> 30 ms window clamps to ~5 ms (< nperseg=2048)
    near_end = (n_samples / sampling_rate) - 0.005
    out = engine.extract_snippet_acoustics(
        str(root), np.array([near_end]), np.array([2.0]), 0.030,
    )
    assert np.isfinite(out["rms"][0])
    assert np.isnan(out["peak_freq_hz"][0])

    empty = engine.extract_snippet_acoustics(str(root), np.array([]), np.array([]), 0.030)
    assert all(empty[key].shape == (0,) for key in empty)


def test_cohens_d_known_values():
    """
    Description
    -----------
    ``cohens_d`` matches the analytic pooled-SD standardized mean difference, is 0
    for identical samples, drops non-finite entries first, and returns NaN when a
    sample is undersized.
    """

    x = np.array([2.0, 4.0, 6.0, 8.0])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    pooled = np.sqrt(x.var(ddof=1))   # equal spread; mean difference is 1.0
    assert engine.cohens_d(x, y) == pytest.approx(1.0 / pooled)
    assert engine.cohens_d(x, x) == pytest.approx(0.0)
    assert engine.cohens_d(np.array([np.nan, 2.0, 4.0, 6.0, 8.0]), y) == pytest.approx(1.0 / pooled)
    assert np.isnan(engine.cohens_d(np.array([1.0]), y))


def test_bootstrap_vs_null_stats():
    """
    Description
    -----------
    Right-tailed ``+1``-corrected p-value and the null-standardized Z-score match
    hand-computed values, and a zero-spread null yields Z = 0.0.
    """

    boot = np.array([3.0, 3.0, 3.0])            # mean 3
    null = np.array([0.0, 1.0, 2.0, 3.0, 4.0])  # mean 2, population std sqrt(2)
    boot_mean, null_mean, p_val, z = engine.bootstrap_vs_null_stats(boot, null)
    assert boot_mean == pytest.approx(3.0)
    assert null_mean == pytest.approx(2.0)
    assert p_val == pytest.approx(3 / 6)        # {3, 4} >= 3 -> (2 + 1) / (5 + 1)
    assert z == pytest.approx(1.0 / np.sqrt(2.0))
    _, _, _, z_flat = engine.bootstrap_vs_null_stats(boot, np.array([2.0, 2.0, 2.0]))
    assert z_flat == 0.0


def test_filter_units_by_catalog():
    """
    Description
    -----------
    The three-criteria filter (``cluster_group`` + ``somatic`` + ``brain_area``)
    keeps only matching catalog rows; stems without a catalog row are dropped, and an
    empty ``brain_areas`` set disables the area filter.
    """

    catalog = {
        ("m", "d", "u_good_pag"):    {"cluster_group": "good", "somatic": "True",  "brain_area": "PAG"},
        ("m", "d", "u_mua_pag"):     {"cluster_group": "mua",  "somatic": "True",  "brain_area": "PAG"},
        ("m", "d", "u_good_nonsom"): {"cluster_group": "good", "somatic": "False", "brain_area": "PAG"},
        ("m", "d", "u_good_other"):  {"cluster_group": "good", "somatic": "True",  "brain_area": "BLA"},
    }
    stems = {stem for _, _, stem in catalog} | {"u_absent"}
    kept = engine.filter_units_by_catalog(
        "m", "d", stems, catalog,
        cluster_group="good", require_somatic=True, brain_areas={"PAG"},
    )
    assert kept == {"u_good_pag"}
    kept_any_area = engine.filter_units_by_catalog(
        "m", "d", stems, catalog,
        cluster_group="good", require_somatic=True, brain_areas=set(),
    )
    assert kept_any_area == {"u_good_pag", "u_good_other"}


def test_load_unit_catalog(tmp_path):
    """
    Description
    -----------
    The catalog loads into a ``(mouse_id, rec_date, unit_id) -> row`` lookup with
    every field read as a string (so numeric-looking unit ids stay strings and match
    file stems).
    """

    csv_path = tmp_path / "unit_catalog.csv"
    csv_path.write_text(
        "mouse_id,rec_date,unit_id,cluster_group,somatic,brain_area\n"
        "158112_0,20241107,42,good,True,PAG\n"
    )
    catalog = engine.load_unit_catalog(csv_path)
    row = catalog[("158112_0", "20241107", "42")]
    assert row["cluster_group"] == "good"
    assert row["brain_area"] == "PAG"
    assert row["unit_id"] == "42"


def test_extract_snippet_acoustics_channel_length_mismatch_raises(tmp_path):
    """
    Description
    -----------
    A ``peak_channels`` array whose length disagrees with ``onsets`` raises a
    ``ValueError`` naming both lengths, BEFORE the audio memmap is ever located —
    so the guard fires on a session root that does not even contain an audio file.
    The message is asserted to mention the mismatching counts.
    """

    onsets = np.array([0.10, 0.20, 0.30])
    peak_channels = np.array([2.0, 0.0])  # one short
    with pytest.raises(ValueError, match=r"peak_channels length \(2\).*onsets length \(3\)"):
        engine.extract_snippet_acoustics(str(tmp_path), onsets, peak_channels, 0.030)


def test_extract_snippet_acoustics_unparseable_mmap_name_raises(tmp_path):
    """
    Description
    -----------
    When an ``*_int16.mmap*`` file exists but its name lacks the
    ``_<sr>_<n_samples>_<n_ch>_int16.mmap`` metadata segment, the metadata regex
    returns ``None`` and the helper raises a ``ValueError`` quoting the offending
    file name. Exercises the malformed-filename branch.
    """

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    # A file matching the glob ('*_int16.mmap*') but NOT the metadata regex.
    bad_name = "session_audio_int16.mmap"
    (audio_dir / bad_name).write_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match=r"Could not parse.*int16\.mmap.*segment"):
        engine.extract_snippet_acoustics(
            str(tmp_path), np.array([0.10]), np.array([0.0]), 0.030,
        )


def test_extract_snippet_acoustics_negative_onset_skipped(tmp_path):
    """
    Description
    -----------
    A negative onset yields a start sample ``s0 < 0``, so the trial is skipped
    (``continue``) and every feature stays NaN — the out-of-range guard. A valid
    onset in the same call still produces a finite RMS, proving the skip is
    per-trial rather than aborting the whole sweep.
    """

    root = _write_tone_mmap(tmp_path, loud_channel=2, tone_lo_s=0.10, tone_hi_s=0.16)
    onsets = np.array([-0.05, 0.10])
    peak_channels = np.array([2.0, 2.0])
    out = engine.extract_snippet_acoustics(str(root), onsets, peak_channels, 0.030)
    assert np.isnan(out["rms"][0])
    assert np.isnan(out["peak_freq_hz"][0])
    assert np.isfinite(out["rms"][1])


def test_extract_snippet_acoustics_empty_band_skips_frequency_features(tmp_path):
    """
    Description
    -----------
    Choosing a pass band entirely above the Nyquist frequency leaves the in-band
    power array empty, so the ``power.size == 0`` guard fires and the frequency
    features stay NaN while ``rms`` (computed before the STFT) is still finite.
    Exercises the empty-band branch of the trial loop.
    """

    sampling_rate = 250000
    root = _write_tone_mmap(
        tmp_path, sampling_rate=sampling_rate, loud_channel=2,
        tone_lo_s=0.10, tone_hi_s=0.16,
    )
    out = engine.extract_snippet_acoustics(
        str(root), np.array([0.10]), np.array([2.0]), 0.030,
        min_freq=sampling_rate, max_freq=sampling_rate * 2.0,
    )
    assert np.isfinite(out["rms"][0])
    assert np.isnan(out["mean_freq_hz"][0])
    assert np.isnan(out["peak_freq_hz"][0])


def _write_session_dir(
    session_dir,
    *,
    unit_stems,
    n_neural_samples=2000,
    emitter="mouse_focal",
    other_emitter="mouse_other",
    n_frames=300,
    frame_rate=120.0,
    category_column="qlvm_supercategory",
    rows=None,
    with_peak_amp_ch=True,
):
    """
    Description
    -----------
    Materialises one session directory on disk with exactly the three artefacts
    :func:`load_animal_sessions` (and, indirectly, :func:`compute_group_acoustics`)
    consume: one ``<stem>.npy`` spike-train file per unit stem (each a
    ``(1, n_neural_samples)`` float array so ``np.load(...)[0, :]`` works), one
    ``*_translated_rotated_metric.h5`` tracking file exposing ``track_names``,
    ``recording_frame_rate`` and a ``tracks`` array (only its leading dimension is
    read), and one ``*_usv_summary.csv`` of per-call rows. The focal mouse is the
    FIRST entry of ``track_names`` so the loader's ``emitter`` filter keeps only
    its calls.

    Parameters
    ----------
    session_dir (pathlib.Path)
        Directory to create and populate (the per-session root).
    unit_stems : list[str]
        File stems for the ``<stem>.npy`` good-unit files (e.g. ``"u_a_good"``);
        these are also the catalog ``unit_id`` keys.
    n_neural_samples : int
        Length of each unit's spike-train vector.
    emitter, other_emitter : str
        Focal and non-focal mouse track names (focal is listed first).
    n_frames : int
        Number of tracking frames (drives ``total_duration``).
    frame_rate : float
        Recording frame rate written to the H5.
    category_column : str
        ``usv_summary`` column used to split calls into the two groups.
    rows : list[dict] or None
        Explicit ``usv_summary`` rows; when None a default focal/other split is
        written.
    with_peak_amp_ch : bool
        When True a ``peak_amp_ch`` column is included in the summary CSV.

    Returns
    -------
    session_dir (pathlib.Path)
        The populated directory.
    """

    session_dir.mkdir(parents=True, exist_ok=True)
    for idx, stem in enumerate(unit_stems):
        # The file stem must NOT include the '_good' suffix here because the loader
        # globs '**/*_good.npy' and keys by f.stem (which then carries '_good').
        train = np.arange(n_neural_samples, dtype=np.float64) + float(idx)
        np.save(session_dir / f"{stem}.npy", train.reshape(1, -1))

    track_path = session_dir / "sess_translated_rotated_metric.h5"
    with h5py.File(name=track_path, mode="w") as track_file:
        track_file.create_dataset(
            "track_names",
            data=np.array([emitter.encode("utf-8"), other_emitter.encode("utf-8")]),
        )
        track_file.create_dataset("recording_frame_rate", data=np.float64(frame_rate))
        track_file.create_dataset("tracks", data=np.zeros((n_frames, 2, 3), dtype=np.float64))

    if rows is None:
        rows = [
            {"emitter": emitter,       "start": 1.0, category_column: "USV", "peak_amp_ch": 2},
            {"emitter": emitter,       "start": 2.0, category_column: "USV", "peak_amp_ch": 0},
            {"emitter": emitter,       "start": 3.0, category_column: "WHISTLE", "peak_amp_ch": 1},
            {"emitter": other_emitter, "start": 4.0, category_column: "USV", "peak_amp_ch": 2},
        ]
    frame = pls.DataFrame(rows)
    if not with_peak_amp_ch and "peak_amp_ch" in frame.columns:
        frame = frame.drop("peak_amp_ch")
    frame.write_csv(session_dir / "sess_usv_summary.csv")
    return session_dir


def test_load_animal_sessions_empty_session_names_returns_empty():
    """
    Description
    -----------
    With no session names the ``by_date`` index is empty, so the loader short-
    circuits and returns ``[]`` without touching the filesystem (``data_root`` is a
    throwaway path that is never read).
    """

    out = engine.load_animal_sessions(
        "mouse_focal",
        [],
        data_root=pathlib.Path("/nonexistent"),
        catalog={},
        category_column="qlvm_supercategory",
        group_a_ids=["USV"],
        group_b_ids=["WHISTLE"],
        cluster_group="good",
        require_somatic=False,
        brain_areas=set(),
    )
    assert out == []


def test_load_animal_sessions_picks_richest_day_and_builds_entries(tmp_path):
    """
    Description
    -----------
    End-to-end happy path across two recording days for one focal mouse. Day
    ``20240102`` has two catalog-passing units versus one on ``20240101``, so the
    loader picks the richer day, intersects the per-session unit sets, reads the
    tracking H5 (frame rate + frame count) and the ``usv_summary`` CSV (focal-only
    calls split into the two category groups), and returns one entry per session of
    the chosen day. A custom ``message_output`` sink captures the two diagnostic
    lines (day pick + common-unit count), confirming the logging branch runs.
    """

    animal_id = "mouse_focal"
    category_column = "qlvm_supercategory"
    # Day 1: one session, one good unit. Day 2: one session, two good units.
    day1 = tmp_path / "20240101_run0"
    day2 = tmp_path / "20240102_run0"
    _write_session_dir(day1, unit_stems=["u_a_good"], frame_rate=120.0, n_frames=240)
    _write_session_dir(day2, unit_stems=["u_a_good", "u_b_good"], frame_rate=150.0, n_frames=300)

    catalog = {
        (animal_id, "20240101", "u_a_good"): {
            "cluster_group": "good", "somatic": "True", "brain_area": "PAG",
        },
        (animal_id, "20240102", "u_a_good"): {
            "cluster_group": "good", "somatic": "True", "brain_area": "PAG",
        },
        (animal_id, "20240102", "u_b_good"): {
            "cluster_group": "good", "somatic": "True", "brain_area": "PAG",
        },
    }

    messages: list[str] = []
    out = engine.load_animal_sessions(
        animal_id,
        ["20240101_run0", "20240102_run0"],
        data_root=tmp_path,
        catalog=catalog,
        category_column=category_column,
        group_a_ids=["USV"],
        group_b_ids=["WHISTLE"],
        cluster_group="good",
        require_somatic=True,
        brain_areas={"PAG"},
        message_output=messages.append,
    )

    # Richer day chosen -> exactly its one session, with both common units loaded.
    assert len(out) == 1
    entry = out[0]
    assert entry["session_id"] == "20240102_run0"
    assert entry["session_root"] == str(day2)
    assert entry["fs"] == pytest.approx(150.0)
    assert entry["total_duration"] == pytest.approx(300 / 150.0)
    assert set(entry["neural_data"]) == {"u_a_good", "u_b_good"}
    # Focal-only calls split by category; group A = USV (2 focal rows), B = WHISTLE (1).
    assert entry["group_a_df"].height == 2
    assert entry["group_b_df"].height == 1
    # Both diagnostic lines were emitted and name the chosen day.
    assert any("20240102" in line and "picked day" in line for line in messages)
    assert any("common filtered units" in line for line in messages)


def test_compute_group_acoustics_reads_peak_amp_ch(tmp_path, monkeypatch):
    """
    Description
    -----------
    :func:`compute_group_acoustics` pulls onsets from the group dataframe's
    ``start`` column and per-call channels from ``peak_amp_ch`` when present, then
    forwards them to :func:`extract_snippet_acoustics`. The acoustics extractor is
    monkeypatched to a recorder so the test asserts the exact onsets/channels/root
    passed through (the ``peak_amp_ch`` branch), without needing a real audio file.
    """

    group_df = pls.DataFrame({"start": [0.1, 0.2, 0.3], "peak_amp_ch": [2, 0, 1]})
    session = {"session_id": "sess0", "session_root": "/some/root", "group_a_df": group_df}

    captured = {}

    def _fake_extract(session_root, onsets, peak_channels, window_s):
        captured["session_root"] = session_root
        captured["onsets"] = onsets
        captured["peak_channels"] = peak_channels
        captured["window_s"] = window_s
        return {"rms": np.zeros(onsets.shape[0])}

    monkeypatch.setattr(engine, "extract_snippet_acoustics", _fake_extract)
    out = engine.compute_group_acoustics(session, "group_a_df", 0.030)
    assert captured["session_root"] == "/some/root"
    np.testing.assert_array_equal(captured["onsets"], [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(captured["peak_channels"], [2, 0, 1])
    assert captured["window_s"] == pytest.approx(0.030)
    assert "rms" in out


def test_compute_group_acoustics_missing_peak_amp_ch_falls_back_to_channel_zero(tmp_path, monkeypatch):
    """
    Description
    -----------
    When the group dataframe predates the ``peak_amp_ch`` column the helper logs a
    fallback diagnostic and passes an all-zero channel vector (channel 0) to
    :func:`extract_snippet_acoustics`. Verifies both the zero channels and that the
    custom ``message_output`` sink received the fallback line naming the session id.
    """

    group_df = pls.DataFrame({"start": [0.5, 0.6]})
    session = {"session_id": "sess_no_ch", "session_root": "/r", "group_b_df": group_df}

    captured = {}

    def _fake_extract(session_root, onsets, peak_channels, window_s):
        captured["peak_channels"] = peak_channels
        return {"rms": np.zeros(onsets.shape[0])}

    monkeypatch.setattr(engine, "extract_snippet_acoustics", _fake_extract)
    messages: list[str] = []
    engine.compute_group_acoustics(session, "group_b_df", 0.040, message_output=messages.append)
    np.testing.assert_array_equal(captured["peak_channels"], [0.0, 0.0])
    assert any("no 'peak_amp_ch' column" in line and "sess_no_ch" in line for line in messages)
