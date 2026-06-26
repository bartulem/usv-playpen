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

2. Somatic classifier — :func:`classify_somatic`
   A waveform peak/trough shape rule, grounded in Deligkaris et al. (2016):
   axonal / dendritic (passing-fibre) units show a large, narrow positive
   peak BEFORE the main trough, whereas somatic units are trough-dominated.
   It computes the waveform-shape features (peak-before / peak-after /
   trough sizes and widths and the derived ratios) from the trough + peak
   indices stock SI already detects, and applies a two-condition decision.
   It replaces the fork's one-line classifier, which flagged a unit non-
   somatic purely when its most-positive sample preceded its most-negative
   one — a noise-sensitive sample-order test with no magnitude or width gate.

3. Multi-channel template metrics — :func:`get_exp_decay`,
   :func:`get_spread`, with the helpers :func:`transform_column_range`
   and :func:`sort_template_and_locations`. These remain owned because
   they do not consume the peak/trough detector and the stock SI
   0.104.3 signatures take a different parameter set. The
   single-channel template metrics (waveform duration, peak/trough
   ratio, half-width, repolarization slope, recovery slope) and the
   underlying prominence-based detector now come directly from stock
   :mod:`spikeinterface.metrics.template.metrics` — see
   :meth:`usv_playpen.neuropixels.spike_quality_metrics.SpikeQualityMetricsExtractor._compute_template_metrics`.

4. Single-unit amplitude-spread metrics — :func:`compute_amplitude_cv`,
   :func:`compute_sd_ratio`. These are owned single-unit ports of
   SpikeInterface's ``compute_amplitude_cv_metrics`` and
   ``compute_sd_ratio``. Stock SI reads every spike's amplitude from the
   ``spike_amplitudes`` extension (a full recording sweep); these ports
   instead take a per-spike amplitude array directly — in the pipeline,
   the amplitudes of the random-spike subsample derived from the
   windowed waveform extraction — so the metrics can be computed without
   materialising the ``spike_amplitudes`` extension the pipeline
   deliberately avoids.
"""

from __future__ import annotations

import warnings
from itertools import pairwise

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import peak_widths
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


# Non-somatic decision thresholds (waveform peak/trough shape rule, after Deligkaris
# et al. 2016). `standard_spike_width` is the 61-sample reference window the two width
# thresholds scale against.
DEFAULT_SOMATIC_PARAMS = {
    'min_trough_to_peak2_ratio': 5.0,
    'min_width_first_peak': 4,
    'min_width_main_trough': 5,
    'max_peak1_to_peak2_ratio': 3.0,
    'max_main_peak_to_trough_ratio': 0.8,
    'standard_spike_width': 61,
}


def classify_somatic(
    template_single: np.ndarray,
    peaks_info: dict,
    *,
    params: dict | None = None,
) -> dict:
    """
    Description
    -----------
    Classify a single-channel template waveform as somatic or non-somatic and
    return the waveform-shape features the decision rests on.

    A waveform peak/trough shape rule grounded in Deligkaris et al. (2016): axonal /
    dendritic (passing-fibre) units show a large, narrow positive peak BEFORE the main
    trough, whereas somatic units are trough-dominated. It replaces the fork's one-line
    classifier, which flagged a unit non-somatic purely when its most-positive sample
    preceded its most-negative one — a noise-sensitive sample-order test with no
    magnitude or width gate.

    The main trough and the largest peaks before / after it are taken from
    ``peaks_info`` (already computed by stock SI's ``get_trough_and_peak_idx`` for
    the other single-channel template metrics, so the detection is consistent across
    the catalog). Peak / trough widths are measured in SAMPLES at half-prominence
    (:func:`scipy.signal.peak_widths`); the two width thresholds scale with the
    template length relative to a 61-sample standard. A unit is NON-somatic iff::

        ( trough_to_peak2_ratio   < min_trough_to_peak2_ratio  AND
          main_peak_before_width  < min_width_first_peak        AND
          main_trough_width       < min_width_main_trough       AND
          peak1_to_peak2_ratio    > max_peak1_to_peak2_ratio )
        OR  main_peak_to_trough_ratio > max_main_peak_to_trough_ratio

    Ratios are ``abs(numerator / denominator)`` (zero denominator -> ``inf``; a zero
    / absent numerator -> ``0.0``). All sizes are absolute amplitudes; a ``None``
    peak / trough index (the stock detector's "not found" sentinel) yields size 0 and
    width 0.

    Parameters
    ----------
    template_single : numpy.ndarray
        1D template waveform on the unit's peak (extremum) channel. Raw ADC units are
        fine — every feature is a ratio or a sample count.
    peaks_info : dict
        Output of ``get_trough_and_peak_idx``; uses ``trough_index``,
        ``peak_before_index`` and ``peak_after_index`` (each an int or ``None``).
    params : dict, optional
        Threshold overrides merged onto :data:`DEFAULT_SOMATIC_PARAMS`. Keys:
        ``min_trough_to_peak2_ratio``, ``min_width_first_peak``,
        ``min_width_main_trough``, ``max_peak1_to_peak2_ratio``,
        ``max_main_peak_to_trough_ratio``, ``standard_spike_width``.

    Returns
    -------
    dict
        ``{'somatic': bool, 'main_trough_size', 'main_peak_before_size',
        'main_peak_after_size', 'main_peak_before_width', 'main_trough_width',
        'peak1_to_peak2_ratio', 'trough_to_peak2_ratio',
        'main_peak_to_trough_ratio'}``.
    """

    resolved_params = {**DEFAULT_SOMATIC_PARAMS, **(params or {})}
    template_single = np.asarray(template_single, dtype=np.float64)
    n_samples = template_single.shape[0]

    trough_index = peaks_info['trough_index']
    peak_before_index = peaks_info['peak_before_index']
    peak_after_index = peaks_info['peak_after_index']

    def _size(index):
        # Absolute amplitude at `index`; a None index (the detector's
        # "not found" sentinel) yields size 0.0, mirroring _width.
        return abs(float(template_single[index])) if index is not None else 0.0

    def _width(index, signal):
        # Half-prominence width in samples; `index` must be a local maximum of
        # `signal` (the trough is a maximum of the negated waveform).
        if index is None:
            return 0.0
        try:
            with warnings.catch_warnings():
                # a degenerate (zero-prominence / edge) peak warns and returns 0 width
                warnings.simplefilter('ignore')
                return float(peak_widths(signal, [int(index)], rel_height=0.5)[0][0])
        except Exception:  # a degenerate edge peak just yields width 0
            return 0.0

    def _ratio(numerator, denominator):
        if denominator == 0:
            return float('inf')
        if numerator == 0:
            return 0.0
        return abs(numerator / denominator)

    main_trough_size = _size(trough_index)
    main_peak_before_size = _size(peak_before_index)
    main_peak_after_size = _size(peak_after_index)
    main_peak_before_width = _width(peak_before_index, template_single)
    main_trough_width = _width(trough_index, -template_single)

    peak1_to_peak2_ratio = _ratio(main_peak_before_size, main_peak_after_size)
    trough_to_peak2_ratio = _ratio(main_trough_size, main_peak_before_size)
    main_peak_to_trough_ratio = _ratio(
        max(main_peak_before_size, main_peak_after_size), main_trough_size
    )

    width_scale = n_samples / resolved_params['standard_spike_width']
    min_width_first_peak = max(2.0, round(resolved_params['min_width_first_peak'] * width_scale))
    min_width_main_trough = max(3.0, round(resolved_params['min_width_main_trough'] * width_scale))

    is_non_somatic = (
        (trough_to_peak2_ratio < resolved_params['min_trough_to_peak2_ratio'])
        and (main_peak_before_width < min_width_first_peak)
        and (main_trough_width < min_width_main_trough)
        and (peak1_to_peak2_ratio > resolved_params['max_peak1_to_peak2_ratio'])
    ) or (main_peak_to_trough_ratio > resolved_params['max_main_peak_to_trough_ratio'])

    return {
        'somatic': not is_non_somatic,
        'main_trough_size': main_trough_size,
        'main_peak_before_size': main_peak_before_size,
        'main_peak_after_size': main_peak_after_size,
        'main_peak_before_width': main_peak_before_width,
        'main_trough_width': main_trough_width,
        'peak1_to_peak2_ratio': peak1_to_peak2_ratio,
        'trough_to_peak2_ratio': trough_to_peak2_ratio,
        'main_peak_to_trough_ratio': main_peak_to_trough_ratio,
    }


def transform_column_range(template, channel_locations, column_range, depth_direction="y"):
    """
    Description
    -----------
    Restrict a template and its channel locations to the channels lying
    within ``+/- column_range`` (in um) of the peak channel along the
    LATERAL axis — the axis perpendicular to depth. This narrows the
    channel set to a single probe column before the spread is measured
    along depth, so the spread is not contaminated by channels on
    laterally distant columns.

    The peak channel is the channel with the largest peak-to-peak
    template amplitude. When ``column_range`` is ``None`` the template
    and locations are returned unchanged.

    Parameters
    ----------
    template : numpy.ndarray
        ``(num_samples, num_channels)`` template waveform.
    channel_locations : numpy.ndarray
        ``(num_channels, 2)`` channel coordinates.
    column_range : float or None
        Half-width (um) of the lateral band, centred on the peak
        channel, that channels must fall within to be kept. ``None``
        keeps every channel.
    depth_direction : str, default "y"
        Which axis is depth; this selects the LATERAL (column) axis as
        the OTHER one. Note this is the inverse axis sense of
        :func:`sort_template_and_locations` / :func:`get_spread`, which
        use ``depth_direction`` to pick the depth axis itself.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(template_column_range, channel_locations_column_range)`` — the
        template (columns sub-selected on its channel axis) and channel
        locations restricted to the lateral band.
    """
    # `column_dim` is the LATERAL axis (perpendicular to depth): for
    # depth_direction == "y" depth is axis 1, so the lateral axis is 0.
    # This is the inverse of `depth_dim` in sort_template_and_locations /
    # get_spread, which select the depth axis instead.
    column_dim = 0 if depth_direction == "y" else 1
    if column_range is None:
        template_column_range = template
        channel_locations_column_range = channel_locations
    else:
        # Lateral coordinate of the peak (max peak-to-peak) channel; the
        # lateral band that `column_mask` keeps is centred on this value.
        max_channel_x = channel_locations[np.argmax(np.ptp(template, axis=0)), 0]
        column_mask = np.abs(channel_locations[:, column_dim] - max_channel_x) <= column_range
        template_column_range = template[:, column_mask]
        channel_locations_column_range = channel_locations[column_mask]
    return template_column_range, channel_locations_column_range


def sort_template_and_locations(template, channel_locations, depth_direction="y"):
    """
    Description
    -----------
    Reorder a template's channels and the matching channel locations so
    that channels run ascending along DEPTH. The depth axis is the one
    selected by ``depth_direction`` (the opposite axis sense from
    :func:`transform_column_range` directly above, which uses
    ``depth_direction`` to pick the lateral axis). The template is
    reindexed on its channel axis (axis 1) in lockstep with the rows of
    ``channel_locations`` so the two stay aligned.

    Parameters
    ----------
    template : numpy.ndarray
        ``(num_samples, num_channels)`` template waveform.
    channel_locations : numpy.ndarray
        ``(num_channels, 2)`` channel coordinates.
    depth_direction : str, default "y"
        Which axis is depth; channels are sorted ascending along this
        axis. ``"y"`` selects axis 1, otherwise axis 0.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(template_sorted, channel_locations_sorted)`` — the template
        with its channel axis reordered by ascending depth and the
        channel locations reordered to match.
    """
    depth_dim = 1 if depth_direction == "y" else 0
    sort_indices = np.argsort(channel_locations[:, depth_dim])
    return template[:, sort_indices], channel_locations[sort_indices, :]


def get_exp_decay(template, channel_locations, sampling_frequency=None, **kwargs):  # noqa: ARG001 - name kept for SpikeInterface signature compatibility
    """
    Compute the exponential decay constant of the template amplitude over distance (units: 1/um).

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        Unused; retained for signature compatibility with SpikeInterface.
    **kwargs: Required kwargs:
        - exp_peak_function: the function to use to compute the peak amplitude for the exp decay ("ptp" or "min")
        - min_r2_exp_decay: the minimum r2 to accept the exp decay fit

    Returns
    -------
    exp_decay_value : float
        The exponential decay of the template amplitude
    """
    def exp_decay(x, decay, amp0, offset):
        # x: distance (um) from the peak channel; amp0: peak-channel
        # amplitude; offset: far-field baseline amplitude; decay: the
        # 1/um decay constant returned as the metric (popt[0]).
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
    else:
        # Match stock SI: any unrecognised value falls back to peak-to-peak
        # rather than leaving `fun` unbound (which would raise an uncaught
        # UnboundLocalError before the try/except NaN fallback below).
        fun = np.ptp
    peak_amplitudes = np.abs(fun(template, axis=0))
    max_channel_location = channel_locations[np.argmax(peak_amplitudes)]
    # Vectorized row-wise Euclidean distance from the peak channel to every channel
    # (identical to the per-row np.linalg.norm list comprehension, without the
    # per-channel Python overhead -- this runs once per unit).
    channel_distances = np.linalg.norm(channel_locations - max_channel_location, axis=1)
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


def get_spread(template, channel_locations, sampling_frequency, **kwargs) -> float:  # noqa: ARG001 - name kept for SpikeInterface signature compatibility
    """
    Compute the spread (depth extent) of the template amplitude over distance (units: um).

    Parameters
    ----------
    template: numpy.ndarray
        The template waveform (num_samples, num_channels)
    channel_locations: numpy.ndarray
        The channel locations (num_channels, 2)
    sampling_frequency : float
        Unused; retained for signature compatibility with SpikeInterface.
    **kwargs: Required kwargs:
        - depth_direction: the direction to compute the spread above and below ("x", "y", or "z")
        - spread_threshold: the threshold to compute the spread
        - column_range: the range in um in the x-direction to consider channels for the spread

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

    # MM: per-channel peak-to-peak amplitude profile across depth — the
    # quantity the spread is measured on (smoothed, normalised, thresholded
    # below).
    MM = np.ptp(template, 0)
    channel_depths = channel_locations[:, depth_dim]

    if spread_smooth_um is not None and spread_smooth_um > 0:
        spread_sigma = spread_smooth_um / np.median(np.diff(np.unique(channel_depths)))
        MM = gaussian_filter1d(MM, spread_sigma)

    MM = MM / np.max(MM)

    channel_locations_above_threshold = channel_locations[spread_threshold < MM]
    channel_depth_above_threshold = channel_locations_above_threshold[:, depth_dim]

    return np.ptp(channel_depth_above_threshold)


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
    for i0, i1 in pairwise(bounds):
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

    censored_period = round(float(censored_period_ms * 1e-3 * sampling_frequency))
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
