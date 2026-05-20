"""
@author: bartulem
Owned reimplementations of the SpikeInterface internals the Neuropixels
spike-quality-metrics pipeline depends on.

The pipeline historically ran against a patched fork of SpikeInterface.
Pinning stock ``spikeinterface`` and re-implementing the relevant pieces
here removes the fork dependency while keeping the exact behaviour the
pipeline was validated against. Everything below is faithful to the
fork; the rest of the pipeline uses stock SpikeInterface unchanged.

Four groups of functions live here:

1. Phy-peak sparsity — :func:`sparsity_around_phy_peak`
   The fork patched ``ChannelSparsity.from_closest_channels`` so that the
   sparse channel set of each unit is centred on the **phy-reported peak
   channel** (the ``ch`` unit property written by phy / Kilosort
   curation) rather than on the extremum channel SpikeInterface
   recomputes from the templates. The two can disagree — most often
   after manual curation — and the rest of the pipeline keys everything
   off the phy peak channel, so the sparse set must agree with it. This
   reproduces the patch against a stock
   :class:`spikeinterface.core.ChannelSparsity`.

2. Somatic classifier — :func:`is_somatic`
   The fork hijacked ``template_metrics.get_num_positive_peaks`` so that,
   instead of counting positive peaks, it returned a boolean somatic /
   non-somatic classification of the waveform. :func:`is_somatic` is
   that classification as a standalone function on a single 1D template.

3. Multi-channel template metrics — :func:`get_exp_decay`,
   :func:`get_spread`, with the helpers :func:`transform_column_range`
   and :func:`sort_template_and_locations`. These remain owned because
   they do not consume the peak/trough detector and the stock SI
   0.104.3 signatures take a different parameter set. The
   single-channel template metrics (waveform duration, peak/trough
   ratio, half-width, repolarization slope, recovery slope) and the
   underlying prominence-based detector now come directly from stock
   :mod:`spikeinterface.metrics.template.metrics` — see
   :meth:`usv_playpen.analyses.npx_spike_quality_metrics.SpikeQualityMetricsExtractor._compute_template_metrics`.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from spikeinterface.core import ChannelSparsity
from spikeinterface.curation.curation_tools import find_duplicated_spikes


def _closest_channel_mask(channel_locations: np.ndarray,
                          peak_channels: np.ndarray,
                          num_channels: int) -> np.ndarray:
    """
    Description
    -----------
    Build the boolean ``(n_units, n_channels)`` sparsity mask that, for
    each unit, selects the ``num_channels`` channels physically closest
    to that unit's peak channel.

    For every unit the Euclidean distances from its peak channel to all
    channels are sorted ascending; the first ``num_channels`` entries
    (the peak channel itself is at distance zero and is therefore always
    included) are marked ``True``. This is the pure-array core shared by
    :func:`sparsity_around_phy_peak`, kept separate so it can be tested
    without constructing a SpikeInterface ``SortingAnalyzer``.

    Parameters
    ----------
    channel_locations : numpy.ndarray
        ``(n_channels, n_dims)`` array of channel coordinates (2D probe
        plane or 3D), in consistent spatial units.
    peak_channels : numpy.ndarray
        ``(n_units,)`` array of integer channel indices, one per unit,
        giving each unit's peak channel (a row index into
        ``channel_locations``).
    num_channels : int
        Number of channels to retain per unit, counting from the peak
        channel outward.

    Returns
    -------
    numpy.ndarray
        ``(n_units, n_channels)`` boolean mask, ``True`` where a channel
        belongs to a unit's sparse set.
    """
    # Full channel-by-channel Euclidean distance matrix.
    distances = np.linalg.norm(
        channel_locations[:, np.newaxis] - channel_locations[np.newaxis, :], axis=2
    )

    n_units = peak_channels.shape[0]
    n_channels = channel_locations.shape[0]
    mask = np.zeros((n_units, n_channels), dtype="bool")

    for unit_ind in range(n_units):
        # Channels ordered by distance from this unit's peak channel.
        chan_inds = np.argsort(distances[peak_channels[unit_ind]])
        chan_inds = chan_inds[:num_channels]
        mask[unit_ind, chan_inds] = True

    return mask


def sparsity_around_phy_peak(recording, sorting, num_channels: int) -> ChannelSparsity:
    """
    Description
    -----------
    Construct a :class:`spikeinterface.core.ChannelSparsity` in which
    each unit's sparse channel set is the ``num_channels`` channels
    physically closest to that unit's **phy-reported peak channel**.

    This is the owned reimplementation of the fork's patched
    ``ChannelSparsity.from_closest_channels``. The peak channel of each
    unit is read from the ``ch`` unit property (the channel index phy /
    Kilosort assigned to the unit), rather than recomputed from the
    templates as stock SpikeInterface does — guaranteeing the sparse set
    is centred on the same channel the rest of the pipeline uses.

    The helper takes the ``recording`` and ``sorting`` separately,
    rather than a ``SortingAnalyzer``, because the pipeline needs the
    sparsity *before* the analyzer is created: ``create_sorting_analyzer``
    accepts the sparsity as a constructor argument. Channel coordinates
    and ``channel_ids`` come from the recording; the phy ``ch`` property
    and ``unit_ids`` come from the sorting.

    Parameters
    ----------
    recording : spikeinterface.core.BaseRecording
        Recording for the session. Must expose channel coordinates via
        ``get_channel_locations()``; ``channel_ids`` is taken from it.
    sorting : spikeinterface.core.BaseSorting
        Phy-curated sorting. Must carry a per-unit integer peak-channel
        property named ``ch`` (accessible via ``get_property('ch')``);
        ``unit_ids`` is taken from it.
    num_channels : int
        Number of channels to retain per unit, counting from the peak
        channel outward.

    Returns
    -------
    spikeinterface.core.ChannelSparsity
        Sparsity object whose mask selects, per unit, the
        ``num_channels`` channels closest to the unit's phy peak
        channel.
    """
    channel_locations = recording.get_channel_locations()
    peak_channels = np.asarray(sorting.get_property("ch"))

    mask = _closest_channel_mask(channel_locations, peak_channels, num_channels)

    return ChannelSparsity(mask, sorting.unit_ids, recording.channel_ids)


def is_somatic(template_single: np.ndarray) -> bool:
    """
    Description
    -----------
    Classify a single-channel template waveform as somatic or
    non-somatic.

    This is the owned reimplementation of the fork's hijacked
    ``template_metrics.get_num_positive_peaks``. A waveform whose
    positive peak both exceeds its negative trough and occurs **before**
    that trough in time is treated as non-somatic (the inverted /
    axonal-passing-fibre signature) and the function returns ``False``.
    Every other waveform — including the canonical somatic shape, where
    the negative trough leads — returns ``True``.

    The logic and variable names are kept identical to the fork patch.

    Parameters
    ----------
    template_single : numpy.ndarray
        1D template waveform on a single channel (typically the unit's
        peak channel).

    Returns
    -------
    bool
        ``True`` if the waveform is classified as somatic, ``False`` if
        non-somatic.
    """
    peak_height = max(template_single)
    trough_height = min(template_single)

    if (peak_height > trough_height) and (np.argmax(template_single) < np.argmin(template_single)):
        return False
    else:
        return True


def transform_column_range(template, channel_locations, column_range, depth_direction="y"):
    """
    Transform template and channel locations based on column range.
    """
    column_dim = 0 if depth_direction == "y" else 1
    if column_range is None:
        template_column_range = template
        channel_locations_column_range = channel_locations
    else:
        max_channel_x = channel_locations[np.argmax(np.ptp(template, axis=0)), 0]
        column_mask = np.abs(channel_locations[:, column_dim] - max_channel_x) <= column_range
        template_column_range = template[:, column_mask]
        channel_locations_column_range = channel_locations[column_mask]
    return template_column_range, channel_locations_column_range


def sort_template_and_locations(template, channel_locations, depth_direction="y"):
    """
    Sort template and locations.
    """
    depth_dim = 1 if depth_direction == "y" else 0
    sort_indices = np.argsort(channel_locations[:, depth_dim])
    return template[:, sort_indices], channel_locations[sort_indices, :]


def get_exp_decay(template, channel_locations, sampling_frequency=None, **kwargs):
    """
    Compute the exponential decay of the template amplitude over distance in units um/s.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - exp_peak_function: the function to use to compute the peak amplitude for the exp decay ("ptp" or "min")
        - min_r2_exp_decay: the minimum r2 to accept the exp decay fit

    Returns
    -------
    exp_decay_value : float
        The exponential decay of the template amplitude
    """
    def exp_decay(x, decay, amp0, offset):
        return amp0 * np.exp(-decay * x) + offset

    assert "exp_peak_function" in kwargs, "exp_peak_function must be given as kwarg"
    exp_peak_function = kwargs["exp_peak_function"]
    assert "min_r2_exp_decay" in kwargs, "min_r2_exp_decay must be given as kwarg"
    min_r2_exp_decay = kwargs["min_r2_exp_decay"]
    # exp decay fit
    if exp_peak_function == "ptp":
        fun = np.ptp
    elif exp_peak_function == "min":
        fun = np.min
    peak_amplitudes = np.abs(fun(template, axis=0))
    max_channel_location = channel_locations[np.argmax(peak_amplitudes)]
    channel_distances = np.array([np.linalg.norm(cl - max_channel_location) for cl in channel_locations])
    distances_sort_indices = np.argsort(channel_distances)

    # longdouble is float128 when the platform supports it, otherwise it is float64
    channel_distances_sorted = channel_distances[distances_sort_indices].astype(np.longdouble)
    peak_amplitudes_sorted = peak_amplitudes[distances_sort_indices].astype(np.longdouble)

    try:
        amp0 = peak_amplitudes_sorted[0]
        offset0 = np.min(peak_amplitudes_sorted)

        popt, _ = curve_fit(
            exp_decay,
            channel_distances_sorted,
            peak_amplitudes_sorted,
            bounds=([1e-5, amp0 - 0.5 * amp0, 0], [2, amp0 + 0.5 * amp0, 2 * offset0]),
            p0=[1e-3, peak_amplitudes_sorted[0], offset0],
        )
        r2 = r2_score(peak_amplitudes_sorted, exp_decay(channel_distances_sorted, *popt))
        exp_decay_value = popt[0]

        if r2 < min_r2_exp_decay:
            exp_decay_value = np.nan
    except Exception:
        # Catch failures in the scipy.optimize.curve_fit call /
        # the surrounding r2 computation; fall back to NaN so the
        # caller's downstream aggregation isn't aborted by a
        # single problematic unit. Narrowed from a bare ``except:``
        # which also swallowed KeyboardInterrupt and SystemExit.
        exp_decay_value = np.nan

    return exp_decay_value


def get_spread(template, channel_locations, sampling_frequency, **kwargs) -> float:
    """
    Compute the spread of the template amplitude over distance in units um/s.

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        The sampling frequency of the template
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute velocity above and below ("x", "y", or "z")
        - spread_threshold: the threshold to compute the spread
        - column_range: the range in um in the x-direction to consider channels for velocity

    Returns
    -------
    spread : float
        Spread of the template amplitude
    """
    assert "depth_direction" in kwargs, "depth_direction must be given as kwarg"
    depth_direction = kwargs["depth_direction"]
    assert "spread_threshold" in kwargs, "spread_threshold must be given as kwarg"
    spread_threshold = kwargs["spread_threshold"]
    assert "spread_smooth_um" in kwargs, "spread_smooth_um must be given as kwarg"
    spread_smooth_um = kwargs["spread_smooth_um"]
    assert "column_range" in kwargs, "column_range must be given as kwarg"
    column_range = kwargs["column_range"]

    depth_dim = 1 if depth_direction == "y" else 0
    template, channel_locations = transform_column_range(template, channel_locations, column_range)
    template, channel_locations = sort_template_and_locations(template, channel_locations, depth_direction)

    MM = np.ptp(template, 0)
    channel_depths = channel_locations[:, depth_dim]

    if spread_smooth_um is not None and spread_smooth_um > 0:
        spread_sigma = spread_smooth_um / np.median(np.diff(np.unique(channel_depths)))
        MM = gaussian_filter1d(MM, spread_sigma)

    MM = MM / np.max(MM)

    channel_locations_above_threshold = channel_locations[MM > spread_threshold]
    channel_depth_above_threshold = channel_locations_above_threshold[:, depth_dim]

    spread = np.ptp(channel_depth_above_threshold)

    return spread


def compute_amplitude_cv(amplitudes: np.ndarray, sample_indices: np.ndarray,
                         n_samples_total: int, sampling_frequency: float,
                         average_num_spikes_per_bin: int = 50,
                         percentiles: tuple = (5, 95), min_num_bins: int = 10) -> tuple:
    """
    Description
    -----------
    Coefficient of variation of spike amplitudes within temporal bins,
    for a single unit — the owned reimplementation of SpikeInterface's
    ``compute_amplitude_cv_metrics``.

    SpikeInterface's version reads the per-spike amplitudes of every
    spike from the ``spike_amplitudes`` extension (a full recording
    sweep). This port instead takes a per-spike amplitude array directly
    — in the pipeline, the amplitudes of the random-spike subsample,
    derived from the windowed waveform extraction. Because the temporal
    bin size is derived from the (subsample) firing rate, each bin still
    contains ``average_num_spikes_per_bin`` amplitudes on average, so the
    per-bin CV is computed on the same effective sample size as the full
    version; the bins are simply wider.

    The recording is treated as a single segment (the concatenated
    Neuropixels session). Logic mirrors the SpikeInterface function: the
    temporal bin size is ``average_num_spikes_per_bin / firing_rate``
    seconds; within each bin the CV is ``std(amplitudes) / |mean(all
    amplitudes)|``; the median and the inter-percentile range of those
    per-bin CVs are returned. When fewer than ``min_num_bins`` bins are
    produced, both outputs are ``NaN``.

    Parameters
    ----------
    amplitudes : numpy.ndarray
        ``(n_spikes,)`` per-spike amplitudes for one unit, ordered by
        spike time.
    sample_indices : numpy.ndarray
        ``(n_spikes,)`` sample indices of those spikes, sorted ascending
        and parallel to ``amplitudes``.
    n_samples_total : int
        Total number of samples in the (single-segment) recording.
    sampling_frequency : float
        Recording sampling frequency in Hz.
    average_num_spikes_per_bin : int, default 50
        Target average number of spikes per temporal bin; sets the bin
        size via the unit's firing rate.
    percentiles : tuple, default (5, 95)
        Lower/upper percentiles whose difference defines the CV "range".
    min_num_bins : int, default 10
        Minimum number of temporal bins required; below this both
        outputs are ``NaN``.

    Returns
    -------
    tuple[float, float]
        ``(amplitude_cv_median, amplitude_cv_range)`` — the median and
        inter-percentile range of the per-bin coefficients of variation.
    """
    num_spikes = amplitudes.size
    if num_spikes == 0:
        return np.nan, np.nan

    total_duration = n_samples_total / sampling_frequency
    firing_rate = num_spikes / total_duration
    temporal_bin_size_samples = int((average_num_spikes_per_bin / firing_rate) * sampling_frequency)
    if temporal_bin_size_samples < 1:
        return np.nan, np.nan

    sample_bin_edges = np.arange(0, n_samples_total + 1, temporal_bin_size_samples)
    amp_mean = np.abs(np.mean(amplitudes))
    bounds = np.searchsorted(sample_indices, sample_bin_edges, side="left")

    amp_spreads = []
    for i0, i1 in zip(bounds[:-1], bounds[1:]):
        amp_spreads.append(np.std(amplitudes[i0:i1]) / amp_mean)

    if len(amp_spreads) < min_num_bins:
        return np.nan, np.nan

    amplitude_cv_median = np.median(amp_spreads)
    amplitude_cv_range = np.percentile(amp_spreads, percentiles[1]) - np.percentile(amp_spreads, percentiles[0])
    return amplitude_cv_median, amplitude_cv_range


def compute_sd_ratio(amplitudes: np.ndarray, sample_indices: np.ndarray,
                     noise_level: float, template_best_channel: np.ndarray,
                     n_spikes_full: int, n_samples_total: int, sampling_frequency: float,
                     censored_period_ms: float = 4.0, correct_for_drift: bool = True,
                     correct_for_template_itself: bool = True) -> float:
    """
    Description
    -----------
    Ratio of the standard deviation of a unit's spike amplitudes to the
    standard deviation of the noise on its best channel, for a single
    unit — the owned reimplementation of SpikeInterface's
    ``compute_sd_ratio``.

    SpikeInterface's version reads every spike's amplitude from the
    ``spike_amplitudes`` extension (a full recording sweep). This port
    takes a per-spike amplitude array directly — in the pipeline, the
    random-spike subsample's amplitudes from the windowed waveform
    extraction. Logic mirrors the SpikeInterface function:

    * near-duplicate spikes within ``censored_period_ms`` are removed
      (via SpikeInterface's ``find_duplicated_spikes``,
      ``keep_first_iterative``) so bursts do not inflate the spread;
    * with ``correct_for_drift`` the amplitude SD is estimated
      drift-robustly as ``std(diff(amplitudes)) / sqrt(2)``, otherwise
      as the plain ``std``;
    * with ``correct_for_template_itself`` the contribution of the
      unit's own template to the channel's measured noise is estimated
      and subtracted, using the unit's **full** spike count.

    Parameters
    ----------
    amplitudes : numpy.ndarray
        ``(n_spikes,)`` per-spike amplitudes for one unit, ordered by
        spike time.
    sample_indices : numpy.ndarray
        ``(n_spikes,)`` sample indices of those spikes, sorted ascending
        and parallel to ``amplitudes``.
    noise_level : float
        Standard deviation of the noise on the unit's best channel.
    template_best_channel : numpy.ndarray
        The unit's template waveform on its best channel (1D); used only
        for the ``correct_for_template_itself`` correction.
    n_spikes_full : int
        The unit's total spike count (not the subsample size) — the
        template-itself correction needs the true spike count.
    n_samples_total : int
        Total number of samples in the (single-segment) recording.
    sampling_frequency : float
        Recording sampling frequency in Hz.
    censored_period_ms : float, default 4.0
        Spikes within this period of a kept spike are censored before
        the SD is computed.
    correct_for_drift : bool, default True
        Use the drift-robust ``std(diff) / sqrt(2)`` estimator instead
        of the plain standard deviation.
    correct_for_template_itself : bool, default True
        Subtract the unit template's own contribution from the channel
        noise estimate.

    Returns
    -------
    float
        The SD ratio for the unit; ``NaN`` when no spikes remain after
        censoring, ``0.0`` when exactly one remains.
    """
    if amplitudes.size == 0:
        return np.nan

    censored_period = int(round(censored_period_ms * 1e-3 * sampling_frequency))
    censored_indices = find_duplicated_spikes(
        sample_indices, censored_period, method="keep_first_iterative"
    )
    spk_amp = np.delete(amplitudes, censored_indices)

    if len(spk_amp) == 0:
        return np.nan
    if len(spk_amp) == 1:
        return 0.0

    if correct_for_drift:
        unit_std = np.std(np.diff(spk_amp)) / np.sqrt(2)
    else:
        unit_std = np.std(spk_amp)

    std_noise = noise_level
    if correct_for_template_itself:
        # the unit's own template inflates the measured channel noise;
        # estimate and remove that contribution
        p = len(template_best_channel) * n_spikes_full / n_samples_total
        template_variance = (
            p * np.mean(template_best_channel ** 2) - p ** 2 * np.mean(template_best_channel) ** 2
        )
        std_noise = np.sqrt(std_noise ** 2 - template_variance)

    return unit_std / std_noise
