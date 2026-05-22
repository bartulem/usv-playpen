"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.spikeinterface_helpers``.

Two owned reimplementations of SpikeInterface fork patches are checked:
the closest-channel sparsity built around the phy peak channel, and the
somatic / non-somatic single-channel waveform classifier.
"""

from __future__ import annotations

import numpy as np
import pytest

from spikeinterface.core import ChannelSparsity

from usv_playpen.neuropixels.spikeinterface_helpers import (
    _closest_channel_mask,
    sparsity_around_phy_peak,
    is_somatic,
    get_exp_decay,
    get_spread,
    compute_amplitude_cv,
    compute_sd_ratio,
)


def _linear_probe(n_channels=10, pitch_um=20.0):
    """
    Description
    -----------
    Build channel coordinates for a single straight column of contacts
    spaced ``pitch_um`` apart along y (x fixed at 0) — the simplest
    layout in which "closest channels" has an obvious answer.

    Parameters
    ----------
    n_channels : int, default 10
        Number of contacts.
    pitch_um : float, default 20.0
        Spacing between adjacent contacts.

    Returns
    -------
    numpy.ndarray
        ``(n_channels, 2)`` array of channel coordinates.
    """
    ys = np.arange(n_channels, dtype=float) * pitch_um
    return np.column_stack([np.zeros(n_channels), ys])


class _StubRecording:
    """
    Description
    -----------
    Minimal duck-typed stand-in for a SpikeInterface ``BaseRecording``
    exposing only what :func:`sparsity_around_phy_peak` consumes —
    ``get_channel_locations()`` and ``channel_ids`` — so the helper can
    be tested without building a real recording.
    """

    def __init__(self, channel_locations, channel_ids):
        self._channel_locations = channel_locations
        self.channel_ids = channel_ids

    def get_channel_locations(self):
        return self._channel_locations


class _StubSorting:
    """
    Description
    -----------
    Minimal duck-typed stand-in for a SpikeInterface ``BaseSorting``
    exposing only what :func:`sparsity_around_phy_peak` consumes — the
    ``ch`` peak-channel property via ``get_property('ch')`` and
    ``unit_ids`` — so the helper can be tested without building a real
    sorting.
    """

    def __init__(self, peak_channels, unit_ids):
        self._peak_channels = peak_channels
        self.unit_ids = unit_ids

    def get_property(self, key):
        assert key == "ch"
        return self._peak_channels


def test_closest_channel_mask_selects_peak_and_nearest_neighbours():
    """
    Description
    -----------
    On a linear probe the ``num_channels`` closest channels to a peak
    channel are the peak channel itself plus its nearest neighbours on
    either side. With ``num_channels = 3`` an interior peak channel
    yields {peak-1, peak, peak+1}, and a peak channel at the probe edge
    yields {peak, peak+1, peak+2}.
    """
    channel_locations = _linear_probe(n_channels=10)
    peak_channels = np.array([5, 0])
    mask = _closest_channel_mask(channel_locations, peak_channels, num_channels=3)

    assert mask.shape == (2, 10)
    assert mask[0].sum() == 3 and mask[1].sum() == 3
    np.testing.assert_array_equal(np.flatnonzero(mask[0]), [4, 5, 6])
    np.testing.assert_array_equal(np.flatnonzero(mask[1]), [0, 1, 2])


def test_closest_channel_mask_always_includes_the_peak_channel():
    """
    Description
    -----------
    The peak channel is at distance zero from itself, so it must appear
    in every unit's sparse set regardless of ``num_channels``.
    """
    channel_locations = _linear_probe(n_channels=8)
    peak_channels = np.array([0, 3, 7])
    mask = _closest_channel_mask(channel_locations, peak_channels, num_channels=2)

    for unit_ind, peak in enumerate(peak_channels):
        assert mask[unit_ind, peak]


def test_closest_channel_mask_caps_at_total_channel_count():
    """
    Description
    -----------
    Requesting more channels than the probe has must simply select every
    channel rather than raising or producing a malformed mask.
    """
    channel_locations = _linear_probe(n_channels=5)
    peak_channels = np.array([2])
    mask = _closest_channel_mask(channel_locations, peak_channels, num_channels=99)

    assert mask.shape == (1, 5)
    assert mask.all()


def test_sparsity_around_phy_peak_wraps_mask_in_channel_sparsity():
    """
    Description
    -----------
    :func:`sparsity_around_phy_peak` must return a stock
    :class:`spikeinterface.core.ChannelSparsity` whose mask equals the
    pure-array result of :func:`_closest_channel_mask`, with the
    analyzer's ``unit_ids`` and ``channel_ids`` (non-trivial id values)
    threaded through to the per-unit channel-id mapping.
    """
    channel_locations = _linear_probe(n_channels=10)
    peak_channels = np.array([5, 0])
    unit_ids = np.array([100, 101])
    channel_ids = np.arange(10, 20)
    recording = _StubRecording(channel_locations, channel_ids)
    sorting = _StubSorting(peak_channels, unit_ids)

    sparsity = sparsity_around_phy_peak(recording, sorting, num_channels=3)
    expected_mask = _closest_channel_mask(channel_locations, peak_channels, num_channels=3)

    assert isinstance(sparsity, ChannelSparsity)
    np.testing.assert_array_equal(sparsity.mask, expected_mask)
    np.testing.assert_array_equal(sparsity.unit_id_to_channel_ids[100], [14, 15, 16])
    np.testing.assert_array_equal(sparsity.unit_id_to_channel_ids[101], [10, 11, 12])


def test_is_somatic_true_when_trough_leads_peak():
    """
    Description
    -----------
    The canonical somatic waveform — negative trough first, positive
    repolarisation peak after — must be classified somatic (``True``).
    """
    template = np.zeros(60)
    template[20] = -5.0
    template[40] = 3.0
    assert is_somatic(template) is True


def test_is_somatic_false_when_peak_leads_trough():
    """
    Description
    -----------
    A waveform whose positive peak both exceeds its negative trough and
    occurs before it in time is the inverted / non-somatic signature and
    must be classified non-somatic (``False``).
    """
    template = np.zeros(60)
    template[20] = 3.0
    template[40] = -5.0
    assert is_somatic(template) is False


def test_is_somatic_on_realistic_biphasic_waveforms():
    """
    Description
    -----------
    On smooth Gaussian-bump biphasic waveforms (baseline noise absent),
    a trough-then-peak shape classifies somatic and the time-reversed
    peak-then-trough shape classifies non-somatic.
    """
    t = np.arange(120)

    def _bump(centre, amp, width=6.0):
        return amp * np.exp(-0.5 * ((t - centre) / width) ** 2)

    somatic = _bump(40, -6.0) + _bump(70, 3.0)
    non_somatic = somatic[::-1].copy()

    assert is_somatic(somatic) is True
    assert is_somatic(non_somatic) is False


_FS = 30000.0


def test_get_exp_decay_recovers_known_decay_constant():
    """
    Description
    -----------
    With per-channel peak amplitudes following ``A * exp(-decay * d)``
    over channel distance ``d``, :func:`get_exp_decay` must recover the
    known decay constant from the curve fit.
    """
    n_channels = 20
    ys = np.arange(n_channels) * 30.0
    channel_locations = np.column_stack([np.zeros(n_channels), ys])
    decay_true = 0.01
    amplitudes = 100.0 * np.exp(-decay_true * ys)
    template = np.vstack([np.zeros(n_channels), amplitudes])

    decay = get_exp_decay(template, channel_locations,
                          exp_peak_function="ptp", min_r2_exp_decay=0.5)
    assert decay == pytest.approx(decay_true, rel=0.05)


def test_get_spread_is_depth_extent_above_threshold():
    """
    Description
    -----------
    Spread is the depth extent of channels whose (normalised) ptp
    amplitude exceeds ``spread_threshold``. With smoothing disabled and
    normalised amplitudes ``[0.1, 0.2, 0.8, 1.0, 0.3, 0.1]`` at depths
    ``0..100`` µm, only depths 40/60/80 clear a threshold of ``0.2``, so
    the spread is ``80 - 40 = 40`` µm.
    """
    depths = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    channel_locations = np.column_stack([np.zeros(6), depths])
    amplitudes = np.array([1.0, 2.0, 8.0, 10.0, 3.0, 1.0])
    template = np.vstack([np.zeros(6), amplitudes])

    spread = get_spread(template, channel_locations, _FS,
                        depth_direction="y", spread_threshold=0.2,
                        spread_smooth_um=0, column_range=None)
    assert spread == pytest.approx(40.0)


def test_compute_amplitude_cv_bins_temporally_and_returns_median_and_range():
    """
    Description
    -----------
    With 50 spikes laid out as 10 temporal bins of 5 spikes each, and
    every bin carrying the amplitudes ``[8, 9, 10, 11, 12]``, each bin's
    CV is ``std([8,9,10,11,12]) / mean(all) = sqrt(2) / 10``. The median
    over the 10 identical bins is that value, and the inter-percentile
    range is zero.
    """
    n_samples_total, fs = 10000, 1000.0
    sample_indices = np.concatenate([
        bin_index * 1000 + np.array([100, 200, 300, 400, 500]) for bin_index in range(10)
    ])
    amplitudes = np.tile([8.0, 9.0, 10.0, 11.0, 12.0], 10)

    cv_median, cv_range = compute_amplitude_cv(
        amplitudes, sample_indices, n_samples_total, fs,
        average_num_spikes_per_bin=5, percentiles=(5, 95), min_num_bins=10)

    assert cv_median == pytest.approx(np.sqrt(2.0) / 10.0)
    assert cv_range == pytest.approx(0.0)


def test_compute_amplitude_cv_is_nan_below_min_num_bins():
    """
    Description
    -----------
    When the temporal binning yields fewer than ``min_num_bins`` bins,
    both outputs are ``NaN``.
    """
    sample_indices = np.array([100, 200, 300, 400, 500])
    amplitudes = np.array([8.0, 9.0, 10.0, 11.0, 12.0])
    cv_median, cv_range = compute_amplitude_cv(
        amplitudes, sample_indices, n_samples_total=10000, sampling_frequency=1000.0,
        average_num_spikes_per_bin=5, min_num_bins=10)
    assert np.isnan(cv_median) and np.isnan(cv_range)


def test_compute_sd_ratio_is_amplitude_sd_over_noise_sd():
    """
    Description
    -----------
    With drift correction, the amplitude SD estimate is
    ``std(diff(amplitudes)) / sqrt(2)``. For amplitudes alternating
    ``0, 2`` the diffs are all ``±2`` so that estimate is ``sqrt(2)``;
    with ``noise_level = sqrt(2)`` and no template correction the SD
    ratio is exactly ``1.0``. Spikes are spaced well beyond the censored
    period, so none are removed.
    """
    amplitudes = np.array([0.0, 2.0] * 10 + [0.0])  # 21 spikes -> 20 diffs all ±2
    sample_indices = np.arange(21) * 10             # 10-sample spacing >> censored period

    sd_ratio = compute_sd_ratio(
        amplitudes, sample_indices, noise_level=np.sqrt(2.0),
        template_best_channel=np.zeros(5), n_spikes_full=21, n_samples_total=10000,
        sampling_frequency=1000.0, correct_for_template_itself=False)
    assert sd_ratio == pytest.approx(1.0)


def test_compute_sd_ratio_template_correction_subtracts_template_variance():
    """
    Description
    -----------
    With ``correct_for_template_itself`` the channel-noise variance has
    the unit template's own contribution subtracted:
    ``std_noise = sqrt(noise**2 - template_variance)``. With a constant
    template ``[1,1,1,1]``, ``n_spikes_full = 10`` and
    ``n_samples_total = 1000``, ``p = 4*10/1000 = 0.04`` and
    ``template_variance = 0.04 - 0.04**2 = 0.0384``; choosing
    ``noise_level = sqrt(1.0384)`` makes ``std_noise = 1.0`` exactly, so
    the SD ratio equals the amplitude SD estimate (``sqrt(2)``).
    """
    amplitudes = np.array([0.0, 2.0] * 10 + [0.0])
    sample_indices = np.arange(21) * 10

    sd_ratio = compute_sd_ratio(
        amplitudes, sample_indices, noise_level=np.sqrt(1.0384),
        template_best_channel=np.ones(4), n_spikes_full=10, n_samples_total=1000,
        sampling_frequency=1000.0, correct_for_template_itself=True)
    assert sd_ratio == pytest.approx(np.sqrt(2.0))
