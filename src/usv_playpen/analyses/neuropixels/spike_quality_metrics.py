"""
@author: bartulem
Spike quality-metrics extraction for one Neuropixels 2.0 session.

This module ports the per-session half of the
``si_quality_metrics_Neuropixels2.0`` notebook (cells "load data" through
"save catalog", plus the channel-order-per-shank cell) into an owned,
testable class. It depends on a pinned stock ``spikeinterface==0.104.3``
rather than the patched fork the notebook historically used; the two
fork modifications the pipeline relied on are reimplemented in
:mod:`usv_playpen.analyses.neuropixels.spikeinterface_helpers` and the 3D
monopolar source triangulation in
:mod:`usv_playpen.analyses.neuropixels.monopolar_triangulation`.

Two-pass design
---------------
A multi-hour concatenated NP2.0 session on a network mount is hundreds
of GB. The original workflow streamed it several times — once for
``waveforms``, again for ``spike_amplitudes``, again to project all
spikes for PCA — taking hours. This pipeline reads the recording **once**;
both passes are run by :meth:`SpikeQualityMetricsExtractor.run`:

* **Core pass** (:meth:`compute_metrics`) — recording-free. The
  spike-train quality metrics (``num_spikes``, ``firing_rate``,
  ``isi_violation`` etc.) are computed by SpikeInterface on a
  no-waveforms analyzer.
* **Recording-dependent pass**
  (:meth:`compute_recording_dependent_metrics`) — one sequential read.
  SpikeInterface computes the ``waveforms`` extension for a uniform
  per-unit random subsample of spikes (this is the single recording
  read — a network mount handles the sequential chunk stream well).
  ``templates``, ``noise_levels`` and ``principal_components`` then
  compute off the ``waveforms`` extension with no further bulk read.
  From those: the template metrics, the somatic classification and the
  unit locations are computed from the ``templates`` extension; ``snr``
  and the PCA metrics run through stock ``compute_quality_metrics``; and
  the amplitude metrics and ``sd_ratio`` are computed directly from the
  ``waveforms`` extension (see :meth:`_compute_amplitude_metrics`) —
  which is what lets ``spike_amplitudes`` (a second whole-recording
  stream) be dropped. Using SpikeInterface's own templates — rather
  than reconstructing them from Kilosort's ``templates.npy`` — keeps the
  template metrics faithful to the reference workflow.

Catalog
-------
All sessions' units are written into one global ``unit_catalog.csv``
(under the EPHYS directory). The write is idempotent: any rows already
present for this session — matched on ``mouse_id`` + ``rec_date`` +
probe — are dropped before this session's fresh rows are appended, so
re-processing a session updates its rows in place rather than
duplicating them (the bug in the original notebook's blind-append).
"""

from __future__ import annotations

import glob
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import spikeinterface.full as si
from spikeinterface.core.template_tools import get_dense_templates_array
from spikeinterface.metrics.quality.misc_metrics import amplitude_cutoff
# Stock 0.104.3 prominence-based peak/trough detector + the template
# metrics that consume its ``peaks_info`` dict. Adopted in place of the
# fork's naive argmax-after-trough detector — see :meth:`_compute_template_metrics`.
from spikeinterface.metrics.template.metrics import (
    get_trough_and_peak_idx,
    get_peak_to_trough_duration,
    get_waveform_ratios,
    get_half_widths,
    get_repolarization_slope,
    get_recovery_slope,
)

from usv_playpen.analyses.neuropixels.histology_ibl_alignment_export import read_ap_meta, parse_imro_table
from usv_playpen.analyses.neuropixels.monopolar_triangulation import solve_monopolar_triangulation_3d
from usv_playpen.analyses.neuropixels.spikeinterface_helpers import (
    sparsity_around_phy_peak,
    is_somatic,
    get_exp_decay,
    get_spread,
    compute_amplitude_cv,
    compute_sd_ratio,
)

# Default SpikeInterface job kwargs for the single `waveforms` recording
# stream. Deliberately omits the notebook's `total_memory='24G'`:
# `ensure_chunk_size` precedence is chunk_size > chunk_memory >
# total_memory > chunk_duration, so passing total_memory silently
# overrides chunk_duration. A modest chunk_duration keeps the stream to a
# bounded per-worker memory footprint; `n_jobs` is the main throughput
# knob and is raised in the analyses_settings.json job_kwargs.
DEFAULT_JOB_KWARGS = {
    'n_jobs': 2,
    'chunk_duration': '1s',
    'progress_bar': True,
}

# SpikeInterface extensions for the recording-dependent pass. ``waveforms``
# is the single sequential recording read — it extracts snippets for only
# the ``random_spikes`` subsample, but SpikeInterface streams the recording
# in chunks to do so (sequential reads, which a network mount handles well;
# random per-spike reads do not). ``templates``, ``noise_levels`` and
# ``principal_components`` then compute off the ``waveforms`` extension
# without touching the recording in bulk. ``spike_amplitudes`` is
# deliberately NOT computed — it would re-stream the whole recording for
# every spike; the amplitude metrics are computed from the ``waveforms``
# extension instead (see :meth:`_compute_amplitude_metrics`).
EXTENSION_PARAMS = {
    'random_spikes': {'method': 'uniform', 'max_spikes_per_unit': 1000, 'seed': 0},
    'waveforms': {'ms_before': 0.7, 'ms_after': 2.1},
    'templates': {},
    # seeded so the noise-level chunk sample (and therefore `snr` /
    # `noise_level`) is reproducible run-to-run; `seed` is nested under
    # `random_slices_kwargs` because stock 0.104.3's get_noise_levels
    # deprecated the top-level form
    'noise_levels': {'random_slices_kwargs': {'seed': 0}},
    'principal_components': {},
}

# Quality metrics split by what they need (the union is the notebook's
# cell-2 metric_names). The spike-train metrics need only the sorting and
# are computed in the recording-free core pass; the recording-dependent
# metrics need the waveforms / templates / PCA extensions and are
# computed in the second pass, after the single recording read.
SPIKE_TRAIN_METRIC_NAMES = [
    'num_spikes', 'firing_rate', 'firing_range', 'presence_ratio',
    'isi_violation', 'rp_violation', 'synchrony',
]
# Recording-dependent quality metrics, split by how they are computed.
# SI_RECORDING_METRIC_NAMES run through stock compute_quality_metrics on
# the injected windowed waveforms / PCA — no recording sweep.
# WINDOWED_AMPLITUDE_METRIC_NAMES would otherwise need the
# spike_amplitudes extension's full per-spike recording sweep, so they
# are computed directly from the windowed waveforms (see
# :meth:`SpikeQualityMetricsExtractor._compute_amplitude_metrics`).
# 'mahalanobis' is stock 0.104.3's renamed metric producing both the
# 'isolation_distance' and 'l_ratio' columns.
SI_RECORDING_METRIC_NAMES = ['snr', 'd_prime', 'mahalanobis', 'nearest_neighbor', 'silhouette']
WINDOWED_AMPLITUDE_METRIC_NAMES = ['amplitude_cutoff', 'amplitude_cv', 'amplitude_median', 'sd_ratio']
RECORDING_DEPENDENT_METRIC_NAMES = SI_RECORDING_METRIC_NAMES + WINDOWED_AMPLITUDE_METRIC_NAMES

# Per-metric parameters (from the notebook's cell 2). Stock spikeinterface
# 0.104.3 raises a KeyError if metric_params carries a key that is not one
# of the metrics actually requested, so QM_PARAMS only holds params for
# metrics in SPIKE_TRAIN_METRIC_NAMES + RECORDING_DEPENDENT_METRIC_NAMES
# (the notebook's fork additionally listed sliding_rp_violation, drift,
# nn_isolation and nn_noise_overlap params, which were never computed),
# and each pass is passed only the subset of keys for its own metrics.
QM_PARAMS = {
    'num_spikes': {},
    'firing_rate': {},
    'presence_ratio': {'bin_duration_s': 60, 'mean_fr_ratio_thresh': 0.0},
    'snr': {'peak_sign': 'both', 'peak_mode': 'at_index'},
    'isi_violation': {'isi_threshold_ms': 1.5, 'min_isi_ms': 0},
    'rp_violation': {'refractory_period_ms': 1.0, 'censored_period_ms': 0.0},
    'synchrony': {},
    'firing_range': {'bin_size_s': 5, 'percentiles': (5, 95)},
    'amplitude_cv': {'average_num_spikes_per_bin': 50, 'percentiles': (5, 95), 'min_num_bins': 10, 'amplitude_extension': 'spike_amplitudes'},
    'amplitude_cutoff': {'peak_sign': 'both', 'num_histogram_bins': 100, 'histogram_smoothing_value': 3, 'amplitudes_bins_min_ratio': 5},
    'amplitude_median': {'peak_sign': 'both'},
    'sd_ratio': {'censored_period_ms': 4.0, 'correct_for_drift': True, 'correct_for_template_itself': True},
    'nearest_neighbor': {'max_spikes': 10000, 'n_neighbors': 5},
    'silhouette': {'method': ('simplified',)},
    'd_prime': {},
}

# Parameters consumed by the template-metric functions. ``recovery_window_ms``
# is read by stock SI's ``get_recovery_slope``; the rest are required by
# the owned multi-channel metrics ``get_exp_decay`` (``exp_peak_function``,
# ``min_r2_exp_decay``) and ``get_spread`` (``depth_direction``,
# ``spread_threshold``, ``spread_smooth_um``, ``column_range``). The dict
# is passed wholesale to every metric function; each consumes the keys it
# needs and ignores the rest.
TEMPLATE_METRIC_PARAMS = {
    'recovery_window_ms': 0.7,
    'depth_direction': 'y',
    'exp_peak_function': 'ptp',
    'min_r2_exp_decay': 0.5,
    'spread_threshold': 0.2,
    'spread_smooth_um': 20,
    'column_range': None,
}

# Column order of the per-session unit catalog.
CATALOG_COLUMNS = [
    'rec_date', 'mouse_id', 'rec_sessions', 'probe_sn', 'hs_sn', 'kilosort_version',
    'phy_curated', 'unit_id', 'cluster_group', 'somatic', 'spiking_profile',
    'loc_ap', 'loc_ml', 'loc_dv', 'closest_ch', 'brain_area', 'firing_rate',
    'noise_level', 'waveform_duration', 'peak_trough_ratio', 'fwhm',
    'repolarization_slope', 'recovery_slope', 'exp_decay', 'spread',
    'amplitude_cutoff', 'amplitude_cv_median', 'amplitude_cv_range', 'amplitude_median',
    'firing_range', 'firing_rate_si', 'isi_violations_ratio', 'isi_violations_count',
    'num_spikes', 'presence_ratio', 'rp_contamination', 'sd_ratio', 'snr',
    'sync_spike_2', 'sync_spike_4', 'sync_spike_8', 'd_prime', 'isolation_distance',
    'l_ratio', 'silhouette', 'nn_hit_rate', 'nn_miss_rate',
]


class SpikeQualityMetricsExtractor:
    """
    Description
    -----------
    Orchestrates spike quality-metrics extraction for one Neuropixels 2.0
    session, for a single ``(mouse_id, session_date, probe_id,
    hemisphere)`` tuple.

    Construction resolves and caches all required paths and parses the
    ``.ap.meta`` file; no recording data is read until :meth:`load`. The
    public step methods are designed to run in sequence (see
    :meth:`run`), each storing its result on the instance for the next
    step to consume:

    * :meth:`load` — read the recording and sorting, build the analyzer,
      and reconstruct the Kilosort templates in the shared channel space.
    * :meth:`compute_metrics` — sweep-free core pass (template metrics +
      spike-train quality metrics).
    * :meth:`compute_recording_dependent_metrics` — one cached recording
      sweep (amplitude / PCA / ``sd_ratio`` / ``snr`` metrics + noise
      levels).
    * :meth:`compute_unit_locations` — monopolar triangulation against
      the IBL channel locations.
    * :meth:`write_channel_order_per_shank` — the per-shank channel order
      JSON.
    * :meth:`build_session_catalog` — assemble and write the per-session
      catalog.

    Parameters
    ----------
    os_cup_loc : str or pathlib.Path
        Root mount point of the storage server (e.g. ``/mnt/falkner/
        Bartul``).
    mouse_id : str
        Animal identifier, including the tail tag (e.g. ``"158112_0"``).
        Used both to locate the histology directory
        (``{os_cup_loc}/{histology_dirname}/{mouse_id}/...``) and as the
        value written into the catalog's ``mouse_id`` column, so the
        histology directory must be named with the tail-tagged id.
    session_date : str
        Recording date in ``YYYYMMDD`` format (e.g. ``"20241107"``).
    probe_id : str
        SpikeGLX probe label (e.g. ``"imec0"``). Combined with
        ``session_date`` to form the EPHYS subdirectory name.
    hemisphere : str
        ``"L"`` or ``"R"``. Selects the IBL-aligned ``channel_locations.
        json`` (under ``ibl_{hemisphere}H``) and the sign convention used
        when converting channel coordinates to anatomical space.
    kilosort_version : str or int, default ``"4"``
        Version suffix of the Kilosort subdirectory under the EPHYS
        directory (``kilosort4/`` by default).
    phy_curated : bool, default True
        Whether the sorting was manually curated in phy. Recorded in the
        catalog's ``phy_curated`` column.
    num_channels_sparsity : int, default 7
        Number of channels per unit in the phy-peak-centred sparsity.
    shank_width_microns : float, default 70
        Shank width used to convert channel ``lateral`` offsets into the
        anatomical coordinate frame in :meth:`compute_unit_locations`.
    shank_spacing_microns : float, default 250
        Centre-to-centre spacing between adjacent shanks. Used to strip
        the inter-shank offset from each channel's ``lateral`` value
        (via ``lateral % shank_spacing_microns``) so the
        :meth:`compute_unit_locations` transform always sees a
        within-shank offset, regardless of whether ``channel_locations.
        json`` stores ``lateral`` within-shank or with the full
        multi-shank offset baked in.
    histology_dirname : str, default ``"histology"``
        Top-level directory name under ``os_cup_loc`` holding per-animal
        histology output.
    ephys_dirname : str, default ``"EPHYS"``
        Top-level directory name under ``os_cup_loc`` holding per-session
        EPHYS output.
    unit_subset : int or sequence of int or None, default None
        When ``None``, every non-noise unit is processed. An ``int`` N
        processes the first N units (handy for fast validation runs); a
        sequence of unit ids processes exactly those units.
    analyzer_folder : str or pathlib.Path or None, default None
        When ``None``, the ``SortingAnalyzer`` is held in memory. A path
        writes a binary-folder analyzer there (with ``overwrite=True``) —
        which caches the recording-dependent pass's expensive extensions
        on disk so a re-run does not repeat the sweep.
    job_kwargs : dict or None, default None
        SpikeInterface job kwargs for the compute calls. When ``None``,
        :data:`DEFAULT_JOB_KWARGS` is used.
    """

    def __init__(
        self,
        os_cup_loc: str | os.PathLike,
        mouse_id: str,
        session_date: str,
        probe_id: str,
        hemisphere: str,
        kilosort_version: str | int = '4',
        phy_curated: bool = True,
        num_channels_sparsity: int = 7,
        shank_width_microns: float = 70,
        shank_spacing_microns: float = 250,
        histology_dirname: str = 'histology',
        ephys_dirname: str = 'EPHYS',
        unit_subset: int | list | None = None,
        analyzer_folder: str | os.PathLike | None = None,
        job_kwargs: dict | None = None,
    ) -> None:
        if hemisphere not in ('L', 'R'):
            raise ValueError(f"hemisphere must be 'L' or 'R', got {hemisphere!r}")

        self.os_cup_loc = Path(os_cup_loc)
        self.mouse_id = mouse_id
        self.session_date = str(session_date)
        self.probe_id = probe_id
        self.hemisphere = hemisphere
        self.kilosort_version = str(kilosort_version)
        self.phy_curated = phy_curated
        self.num_channels_sparsity = num_channels_sparsity
        self.shank_width_microns = shank_width_microns
        self.shank_spacing_microns = shank_spacing_microns
        self.unit_subset = unit_subset
        self.analyzer_folder = Path(analyzer_folder) if analyzer_folder is not None else None
        self.job_kwargs = dict(job_kwargs) if job_kwargs is not None else dict(DEFAULT_JOB_KWARGS)

        self.ephys_path = self.os_cup_loc / ephys_dirname / f"{self.session_date}_{self.probe_id}"
        self.ks_path = self.ephys_path / f"kilosort{self.kilosort_version}"
        self.channel_locations_file = (
            self.os_cup_loc / histology_dirname / self.mouse_id / self.session_date
            / f"ibl_{self.hemisphere}H" / 'channel_locations.json'
        )

        meta_candidates = sorted(self.ephys_path.glob('concatenated_*.ap.meta'))
        if len(meta_candidates) == 0:
            raise FileNotFoundError(
                f"No concatenated_*.ap.meta found under {self.ephys_path}."
            )
        if len(meta_candidates) > 1:
            raise RuntimeError(
                f"Multiple concatenated .ap.meta files under {self.ephys_path}; "
                f"expected exactly one."
            )
        self.meta_path = meta_candidates[0]
        self.meta = read_ap_meta(self.meta_path)
        self.imro_rows = parse_imro_table(self.meta['imroTbl'])
        # snsGeomMap is the authoritative source for physical channel
        # positions and shank assignment — Kilosort / SpikeInterface
        # both read it for `channel_positions.npy` and
        # `channel_shanks.npy`. The IMRO table's shank column is
        # unreliable for NP 2.0 multi-shank probes (e.g. probe type
        # 2013), so any code that needs a per-channel shank label
        # must key off `geom_rows`, not `imro_rows`.
        self.geom_rows = parse_imro_table(self.meta['snsGeomMap'])

        # Populated by the step methods.
        self.recording = None
        self.sorting = None
        self.analyzer = None
        self.dense_templates = None
        self.spike_train_metrics = None
        self.recording_dependent_metrics = None
        self.quality_metrics = None
        self.template_metrics = None
        self.unit_locations = None
        self.session_catalog = None
        self.catalog_path = None

    def load(self) -> None:
        """
        Description
        -----------
        Read the SpikeGLX recording and the phy-curated Kilosort 4
        sorting, median-center the recording, optionally slice the
        sorting to ``unit_subset``, build the phy-peak-centred channel
        sparsity, and create the ``SortingAnalyzer``.

        The analyzer is created with ``return_in_uV=False`` — templates
        stay in raw int16 ADC units, matching the convention the
        downstream metrics were validated against — but no
        recording-traversing extension is computed here. Sets
        :attr:`recording`, :attr:`sorting` and :attr:`analyzer`.
        """
        recording_raw = si.read_spikeglx(
            folder_path=str(self.ephys_path),
            stream_id=f"{self.probe_id}.ap",
            use_names_as_ids=True,
            load_sync_channel=False,
        )
        self.recording = si.center(recording_raw, mode='median')

        sorting = si.read_phy(
            folder_path=str(self.ks_path),
            exclude_cluster_groups='noise',
            load_all_cluster_properties=True,
        )
        if self.unit_subset is not None:
            if isinstance(self.unit_subset, int):
                sorting = sorting.select_units(sorting.unit_ids[:self.unit_subset])
            else:
                sorting = sorting.select_units(np.asarray(self.unit_subset))
        self.sorting = sorting

        sparsity = sparsity_around_phy_peak(self.recording, self.sorting, self.num_channels_sparsity)

        if self.analyzer_folder is None:
            self.analyzer = si.create_sorting_analyzer(
                sorting=self.sorting,
                recording=self.recording,
                return_in_uV=False,
                sparsity=sparsity,
                format='memory',
            )
        else:
            self.analyzer = si.create_sorting_analyzer(
                sorting=self.sorting,
                recording=self.recording,
                return_in_uV=False,
                sparsity=sparsity,
                format='binary_folder',
                folder=str(self.analyzer_folder),
                overwrite=True,
            )

    def compute_metrics(self) -> None:
        """
        Description
        -----------
        Core pass: the spike-train quality metrics (recording-free).

        The :data:`SPIKE_TRAIN_METRIC_NAMES` metrics need only the
        sorting and are computed by SpikeInterface on a no-waveforms
        analyzer — only ``random_spikes`` is computed first, which does
        not touch the recording. The template metrics are computed later
        (in :meth:`compute_recording_dependent_metrics`) from
        SpikeInterface's own ``templates`` extension. Sets
        :attr:`spike_train_metrics` (a ``pandas.DataFrame`` indexed by
        unit id).
        """
        self.analyzer.compute(['random_spikes'])
        self.spike_train_metrics = si.compute_quality_metrics(
            self.analyzer,
            metric_names=SPIKE_TRAIN_METRIC_NAMES,
            metric_params={k: v for k, v in QM_PARAMS.items() if k in SPIKE_TRAIN_METRIC_NAMES},
            **self.job_kwargs,
        )

    def _compute_template_metrics(self) -> None:
        """
        Description
        -----------
        Compute the per-unit template metrics from SpikeInterface's own
        ``templates`` extension (the average of the windowed waveforms),
        directly — not through SI's ``template_metrics`` analyzer
        extension, whose output units silently depend on the analyzer's
        ``return_in_uV`` flag.

        Trough and post-trough peak indices come from stock
        :func:`spikeinterface.metrics.template.metrics.get_trough_and_peak_idx`
        — the prominence-based ``scipy.signal.find_peaks`` detector with
        a multi-tier fallback. This replaces the fork's naive
        ``argmax(template[trough_idx:])``, which was fragile on units
        whose post-trough region is flat or multi-modal (tiny template
        differences flipped the detected peak to a late slow rebound).
        The single-channel metrics that depend on those indices —
        ``peak_to_valley`` (``waveform_duration``), ``peak_trough_ratio``,
        ``half_width`` (``fwhm``), ``repolarization_slope``,
        ``recovery_slope`` — are computed by the matching stock functions
        consuming the ``peaks_info`` dict. ``peak_trough_ratio`` is
        ``peak_after_to_trough_ratio`` from stock's
        :func:`get_waveform_ratios` (the magnitude of the post-trough
        peak relative to the trough; sign-positive, where the fork's
        version was signed-negative). The somatic classification stays
        on :func:`is_somatic` since stock SI carries no equivalent.
        Multi-channel metrics (``exp_decay``, ``spread``) and their
        helpers remain owned: they do not consume ``peaks_info`` and the
        stock signatures differ in ways the existing parameter set does
        not cover.

        All single-channel metrics are computed on the template at the
        unit's extremum channel — the most-negative channel, matching
        SpikeInterface's ``peak_sign='neg'`` extremum. Multi-channel
        metrics are computed on the unit's sparse template and its
        channel coordinates. Requires :attr:`dense_templates` to have
        been set by :meth:`compute_recording_dependent_metrics`. Sets
        :attr:`template_metrics`.
        """
        channel_locations = self.analyzer.get_channel_locations()
        sampling_frequency = self.analyzer.sampling_frequency
        sparsity_mask = self.analyzer.sparsity.mask

        template_metrics = {}
        for unit_index, unit_id in enumerate(self.analyzer.unit_ids):
            template_all = self.dense_templates[unit_index]

            extremum_channel = int(np.argmin(template_all.min(axis=0)))
            template_single = template_all[:, extremum_channel]
            peaks_info = get_trough_and_peak_idx(template_single, sampling_frequency)
            # Stock SI uses `-1` as the "no extremum found" sentinel when
            # the search window collapses (e.g. trough too close to the
            # template edge for a post-trough peak to fit). Its downstream
            # metric functions guard with `is None`, not `< 0`, so the
            # sentinel slips through and produces nonsense — most
            # notably a *negative* `waveform_duration` = `(-1 - trough)/fs`.
            # Coerce the sentinel to None here so the downstream
            # `is None` branches fire and the metric returns NaN.
            for _idx_key in ('trough_index', 'peak_before_index', 'peak_after_index'):
                if peaks_info[_idx_key] is not None and peaks_info[_idx_key] < 0:
                    peaks_info[_idx_key] = None

            unit_template_metrics = {
                'peak_to_valley': get_peak_to_trough_duration(
                    peaks_info, sampling_frequency, **TEMPLATE_METRIC_PARAMS),
                'peak_trough_ratio': get_waveform_ratios(
                    template_single, peaks_info, **TEMPLATE_METRIC_PARAMS)['peak_after_to_trough_ratio'],
                # get_half_widths returns (trough_hw, peak_hw); the catalog's
                # `fwhm` is the trough half-width (the fork's get_half_width
                # measured the trough)
                'half_width': get_half_widths(
                    template_single, sampling_frequency, peaks_info, **TEMPLATE_METRIC_PARAMS)[0],
                'repolarization_slope': get_repolarization_slope(
                    template_single, sampling_frequency, peaks_info, **TEMPLATE_METRIC_PARAMS),
                'recovery_slope': get_recovery_slope(
                    template_single, sampling_frequency, peaks_info, **TEMPLATE_METRIC_PARAMS),
                'somatic': is_somatic(template_single),
            }

            template_multi = template_all[:, sparsity_mask[unit_index]]
            channel_locations_sparse = channel_locations[sparsity_mask[unit_index]]
            unit_template_metrics['exp_decay'] = get_exp_decay(
                template_multi, channel_locations_sparse, **TEMPLATE_METRIC_PARAMS)
            unit_template_metrics['spread'] = get_spread(
                template_multi, channel_locations_sparse, sampling_frequency, **TEMPLATE_METRIC_PARAMS)

            template_metrics[unit_id] = unit_template_metrics

        self.template_metrics = template_metrics

    def compute_recording_dependent_metrics(self) -> None:
        """
        Description
        -----------
        Recording-dependent pass — one sequential read of the recording.

        SpikeInterface computes the :data:`EXTENSION_PARAMS` extensions on
        the analyzer: ``waveforms`` (snippets for a uniform per-unit
        random subsample of spikes — the single recording read), then
        ``templates``, ``noise_levels`` and ``principal_components``,
        which all compute off the ``waveforms`` extension with no further
        bulk read. From those: the template metrics and the somatic
        classification are computed from the ``templates`` extension
        (:meth:`_compute_template_metrics`); the
        :data:`SI_RECORDING_METRIC_NAMES` quality metrics (``snr`` and the
        PCA metrics) run through stock ``compute_quality_metrics``; and
        the :data:`WINDOWED_AMPLITUDE_METRIC_NAMES` (the amplitude metrics
        and ``sd_ratio``) are computed directly from the ``waveforms``
        extension (:meth:`_compute_amplitude_metrics`) — which is what
        lets ``spike_amplitudes`` (a second whole-recording stream) be
        dropped.

        Sets :attr:`dense_templates`, :attr:`template_metrics`,
        :attr:`recording_dependent_metrics` and the merged
        :attr:`quality_metrics`.

        The whole pass runs under a ``warnings.catch_warnings()`` block:
        the marginal tail of any full-probe sort (units with very few
        spikes, or units spatially isolated on edge channels) makes
        SpikeInterface, scikit-learn and NumPy emit benign
        ``RuntimeWarning`` / ``UserWarning`` noise — degenerate PCA
        variance, "no other units in the vicinity", a variance over a
        single-sample slice. The affected metrics come out ``NaN`` for
        those units by design, so the warnings carry no actionable
        information and are silenced here.
        """
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            self.analyzer.compute(EXTENSION_PARAMS, **self.job_kwargs)

            # SpikeInterface's own templates (the average of the windowed
            # waveforms) — used by the template metrics, the amplitude
            # metrics' template correction, and the unit locations
            self.dense_templates = get_dense_templates_array(self.analyzer, return_in_uV=False)
            self._compute_template_metrics()

            si_metrics = si.compute_quality_metrics(
                self.analyzer,
                metric_names=SI_RECORDING_METRIC_NAMES,
                metric_params={k: v for k, v in QM_PARAMS.items() if k in SI_RECORDING_METRIC_NAMES},
            )
            amplitude_metrics = self._compute_amplitude_metrics()

        self.recording_dependent_metrics = pd.concat([si_metrics, amplitude_metrics], axis=1)
        quality_metrics = pd.concat(
            [self.spike_train_metrics, self.recording_dependent_metrics], axis=1
        )
        # compute_quality_metrics auto-includes num_spikes / firing_rate
        # regardless of metric_names, so they collide with the core pass's
        # spike-train columns; keep the first (core-pass) occurrence
        self.quality_metrics = quality_metrics.loc[:, ~quality_metrics.columns.duplicated()]

    def _compute_amplitude_metrics(self) -> pd.DataFrame:
        """
        Description
        -----------
        Compute the amplitude metrics and ``sd_ratio`` directly from the
        analyzer's ``waveforms`` extension, avoiding SpikeInterface's
        ``spike_amplitudes`` extension (which would re-stream the whole
        recording for every spike).

        Each random-subsample spike's amplitude is the trace value at the
        spike sample on the unit's extremum channel — the most extreme
        channel of the unit's mean waveform. From the per-unit amplitude
        arrays: ``amplitude_median`` is their median; ``amplitude_cutoff``
        is SpikeInterface's standalone histogram-based estimator (fed
        positive amplitudes); ``amplitude_cv_median`` /
        ``amplitude_cv_range`` and ``sd_ratio`` are the owned
        reimplementations (:func:`compute_amplitude_cv`,
        :func:`compute_sd_ratio`). The noise level ``sd_ratio`` needs is
        read from the ``noise_levels`` extension, the unit's full spike
        count from the sorting, and the best-channel template from the
        reconstructed Kilosort templates.

        Requires the ``analyzer.compute`` step of
        :meth:`compute_recording_dependent_metrics` to have run first —
        the ``waveforms``, ``random_spikes`` and ``noise_levels``
        extensions are consumed here.

        Returns
        -------
        pandas.DataFrame
            Indexed by unit id, with columns ``amplitude_cutoff``,
            ``amplitude_cv_median``, ``amplitude_cv_range``,
            ``amplitude_median`` and ``sd_ratio``.
        """
        sampling_frequency = self.analyzer.sampling_frequency
        n_samples_total = self.recording.get_num_samples()
        noise_levels = self.analyzer.get_extension('noise_levels').get_data()

        waveforms_extension = self.analyzer.get_extension('waveforms')
        all_waveforms = waveforms_extension.get_data()
        nbefore = waveforms_extension.nbefore
        random_spikes = self.analyzer.get_extension('random_spikes').get_random_spikes()

        amplitude_cv_params = QM_PARAMS['amplitude_cv']
        amplitude_cutoff_params = QM_PARAMS['amplitude_cutoff']
        sd_ratio_params = QM_PARAMS['sd_ratio']

        rows = {}
        for unit_index, unit_id in enumerate(self.analyzer.unit_ids):
            unit_mask = random_spikes['unit_index'] == unit_index
            waveforms = all_waveforms[unit_mask]
            sample_indices = random_spikes['sample_index'][unit_mask]
            if waveforms.shape[0] == 0:
                rows[unit_id] = {
                    'amplitude_cutoff': np.nan, 'amplitude_cv_median': np.nan,
                    'amplitude_cv_range': np.nan, 'amplitude_median': np.nan, 'sd_ratio': np.nan,
                }
                continue

            # extremum channel within the unit's sparse set, then the per-spike
            # amplitude as the trace value at the spike sample on that channel
            mean_waveform = waveforms.mean(axis=0)
            extremum_channel = int(np.argmax(np.abs(mean_waveform).max(axis=0)))
            amplitudes = waveforms[:, nbefore, extremum_channel].astype(np.float64)

            amplitude_median = float(np.median(amplitudes))
            amplitudes_for_cutoff = -amplitudes if amplitude_median < 0 else amplitudes
            amplitude_cutoff_value = amplitude_cutoff(
                amplitudes_for_cutoff,
                num_histogram_bins=amplitude_cutoff_params['num_histogram_bins'],
                histogram_smoothing_value=amplitude_cutoff_params['histogram_smoothing_value'],
                amplitudes_bins_min_ratio=amplitude_cutoff_params['amplitudes_bins_min_ratio'],
            )
            amplitude_cv_median, amplitude_cv_range = compute_amplitude_cv(
                amplitudes, sample_indices, n_samples_total, sampling_frequency,
                average_num_spikes_per_bin=amplitude_cv_params['average_num_spikes_per_bin'],
                percentiles=amplitude_cv_params['percentiles'],
                min_num_bins=amplitude_cv_params['min_num_bins'],
            )

            best_channel = int(self.analyzer.sparsity.unit_id_to_channel_indices[unit_id][extremum_channel])
            template_best_channel = self.dense_templates[unit_index][:, best_channel]
            n_spikes_full = self.sorting.get_unit_spike_train(unit_id).size
            sd_ratio_value = compute_sd_ratio(
                amplitudes, sample_indices, noise_levels[best_channel], template_best_channel,
                n_spikes_full, n_samples_total, sampling_frequency,
                censored_period_ms=sd_ratio_params['censored_period_ms'],
                correct_for_drift=sd_ratio_params['correct_for_drift'],
                correct_for_template_itself=sd_ratio_params['correct_for_template_itself'],
            )

            rows[unit_id] = {
                'amplitude_cutoff': amplitude_cutoff_value,
                'amplitude_cv_median': amplitude_cv_median,
                'amplitude_cv_range': amplitude_cv_range,
                'amplitude_median': amplitude_median,
                'sd_ratio': sd_ratio_value,
            }

        return pd.DataFrame.from_dict(rows, orient='index')


    def compute_unit_locations(self) -> None:
        """
        Description
        -----------
        Estimate each unit's 3D anatomical location by monopolar
        triangulation against the IBL-aligned ``channel_locations.json``,
        with the candidate channel set **restricted to the shank on
        which the unit's Kilosort template peaks**, and with IBL
        anatomy looked up by **physical electrode position** rather
        than by raw channel index.

        The IBL alignment GUI writes its per-channel brain coordinates
        keyed by an index ordering of its own (geometric, shank-major,
        axially sorted within a shank), which for NP 2.0 4-shank probes
        does not match the IMRO-driven channel index that Kilosort and
        SpikeInterface use. Joining ``channel_locations[f"channel_{ks_ch}"]``
        on the KS channel index therefore returns the anatomy of a
        completely different physical electrode. To avoid that we
        build a position-keyed lookup ``(lateral, axial) -> (x, y, z,
        brain_region)`` from the IBL JSON and look up each KS
        channel's anatomy via its physical ``(lateral, axial)`` from
        ``channel_positions.npy`` — both IBL and KS publish the same
        ``(lateral, axial)`` for the same electrode, so the join is
        unambiguous.

        After the position-keyed lookup the per-channel ``(x, y, z)``
        is shifted into the anatomical frame just as before: the
        within-shank ``lateral`` offset is folded into the AP axis
        using ``shank_width_microns``, with the sign set by
        ``hemisphere`` (R adds, L subtracts). The ``lateral`` is
        reduced modulo ``shank_spacing_microns`` first so the fold
        always receives a within-shank offset.

        The candidate set is then intersected with the channels that
        sit on the *template-peak shank* (determined per unit from
        :attr:`dense_templates` and ``channel_shanks.npy``). This is
        needed because in the IBL-aligned space every channel of a
        probe shares a single ML and the inter-shank offset lives
        entirely in AP — without the constraint a unit whose template
        has even small "ghost" amplitudes on far shanks can have its
        triangulated centroid pulled AP-ward, and ``closest_channel``
        ends up on the wrong shank.

        The unit's per-channel peak-to-peak amplitudes — read from
        SpikeInterface's ``templates`` extension
        (:attr:`dense_templates`) — are then fed to
        :func:`solve_monopolar_triangulation_3d` with the
        ``"minimize_with_log_penality"`` optimiser. The closest channel
        to the estimated location supplies the unit's brain region
        (again by physical position).

        Requires :meth:`compute_recording_dependent_metrics` to have run
        first. Sets :attr:`unit_locations` (a ``dict`` keyed by unit id,
        each value a ``dict`` with ``location``, ``closest_channel`` and
        ``brain_region``).
        """
        with open(self.channel_locations_file, 'r') as channel_locs_file:
            channel_locations = json.load(channel_locs_file)

        # Position-keyed IBL lookups: (lateral_int, axial_int) ->
        # (x, y, z) and (lateral_int, axial_int) -> brain_region. Used
        # in place of `channel_locations[f"channel_{ks_ch}"]` to avoid
        # the IBL-vs-KS channel-index mismatch.
        pos_to_xyz: dict[tuple[int, int], tuple[float, float, float]] = {}
        pos_to_region: dict[tuple[int, int], str] = {}
        for key, entry in channel_locations.items():
            if not key.startswith('channel_'):
                continue
            pos = (int(entry['lateral']), int(entry['axial']))
            pos_to_xyz[pos] = (
                float(entry['x']), float(entry['y']), float(entry['z']),
            )
            pos_to_region[pos] = entry['brain_region']

        # KS physical (lateral, axial) per channel id, plus shank id —
        # the canonical source for what each Kilosort channel index
        # actually corresponds to on the probe.
        channel_positions = np.load(self.ks_path / 'channel_positions.npy')
        channel_shanks = np.load(self.ks_path / 'channel_shanks.npy').astype(int)

        sparsity = self.analyzer.sparsity

        unit_locations = {}
        for unit_index, unit_id in enumerate(self.analyzer.unit_ids):
            chan_inds_sparse = sparsity.unit_id_to_channel_indices[unit_id]

            # Identify the unit's template peak channel over ALL contacts
            # (not just the sparse set) so the shank assignment can't be
            # mis-anchored by a sparsity choice that already strayed.
            full_template = self.dense_templates[unit_index]
            template_peak_ch = int(np.ptp(full_template, axis=0).argmax())
            peak_shank = int(channel_shanks[template_peak_ch])

            # Restrict the sparse candidate set to the template-peak shank.
            # The peak channel is by construction in `chan_inds_sparse`, so
            # the intersection is non-empty; the explicit guard below is
            # defensive against a degenerate sparsity result.
            same_shank_mask = channel_shanks[chan_inds_sparse] == peak_shank
            chan_inds = chan_inds_sparse[same_shank_mask]
            if chan_inds.size == 0:
                chan_inds = np.array([template_peak_ch], dtype=chan_inds_sparse.dtype)

            temp_chan_locs = np.zeros((chan_inds.size, 3))
            for channel_idx, channel in enumerate(chan_inds):
                physical_pos = (
                    int(channel_positions[channel, 0]),
                    int(channel_positions[channel, 1]),
                )
                x, y, z = pos_to_xyz[physical_pos]
                # strip any inter-shank offset so only the within-shank position remains
                within_shank_lateral = physical_pos[0] % self.shank_spacing_microns
                if self.hemisphere == 'R':
                    temp_chan_locs[channel_idx, :] = [
                        x,
                        y + self.shank_width_microns - within_shank_lateral,
                        z,
                    ]
                else:
                    temp_chan_locs[channel_idx, :] = [
                        x,
                        y - self.shank_width_microns + within_shank_lateral,
                        z,
                    ]

            wf = self.dense_templates[unit_index][:, chan_inds]
            wf_data = np.ptp(a=wf, axis=0)

            unit_location = np.array(
                solve_monopolar_triangulation_3d(wf_data, temp_chan_locs, 1000, 'minimize_with_log_penality')
            )

            distances = np.linalg.norm(temp_chan_locs - unit_location, axis=1)
            min_distance_index = np.argmin(distances)
            closest_channel_number = chan_inds[min_distance_index]
            closest_physical_pos = (
                int(channel_positions[closest_channel_number, 0]),
                int(channel_positions[closest_channel_number, 1]),
            )

            unit_locations[unit_id] = {
                'location': unit_location,
                'closest_channel': closest_channel_number,
                'brain_region': pos_to_region[closest_physical_pos],
            }

        self.unit_locations = unit_locations

    def write_channel_order_per_shank(self, output_dir: str | os.PathLike | None = None) -> Path:
        """
        Description
        -----------
        Order each shank's channels from the probe tip outward and write
        the result as ``channel_order_per_shank.json``.

        Channels are grouped by shank using the Kilosort
        ``channel_shanks.npy`` array (the 1-indexed physical shank
        per raw channel id, as derived by SpikeInterface from the
        SpikeGLX meta and validated against the absolute lateral in
        ``channel_positions.npy``). Within each shank, channels are
        sorted by axial position ascending — i.e. from the probe tip
        outward.

        The IMRO table and the snsGeomMap each carry a per-channel
        "shank" field, but for NP 2.0 4-shank probes (e.g. probe
        type 2013) those fields can disagree with the absolute lateral
        SpikeInterface ultimately stores in ``channel_positions.npy``.
        Reading directly from the Kilosort sidecar arrays keeps this
        artifact consistent with the spike-sorted data and the IBL
        ``channel_locations.json`` ``lateral``/``axial`` fields.

        Output keys are 0-indexed (``shank_0`` .. ``shank_3``);
        ``channel_shanks.npy`` is 1-indexed, so the per-shank filter
        uses ``ks_shank == json_shank + 1``.

        Parameters
        ----------
        output_dir : str or pathlib.Path or None, default None
            Directory to write into. When ``None``, writes into the
            session's EPHYS directory.

        Returns
        -------
        pathlib.Path
            Path to the written ``channel_order_per_shank.json``.
        """
        channel_positions = np.load(self.ks_path / 'channel_positions.npy')
        channel_shanks = (
            np.load(self.ks_path / 'channel_shanks.npy').astype(int)
        )
        info_dict: dict[str, list[int]] = {}
        for json_shank in range(4):
            target_shank = json_shank + 1
            members: list[tuple[int, float]] = []
            for channel_id in range(channel_positions.shape[0]):
                if int(channel_shanks[channel_id]) != target_shank:
                    continue
                members.append(
                    (channel_id, float(channel_positions[channel_id, 1]))
                )
            members.sort(key=lambda pair: pair[1])
            info_dict[f"shank_{json_shank}"] = [c for c, _ in members]

        out_dir = Path(output_dir) if output_dir is not None else self.ephys_path
        out_path = out_dir / 'channel_order_per_shank.json'
        with open(out_path, 'w') as json_output_file:
            json.dump(info_dict, json_output_file, indent=4)
        return out_path

    def _compute_cross_session_firing_rates(self, unit_file_names: dict) -> dict:
        """
        Description
        -----------
        For every unit, find which recording sessions it appears in and
        compute its per-session firing rate, returning the recording
        session list, headstage serial number(s), probe serial number
        and median firing rate.

        Each session's ``changepoints_info`` entry points at a raw data
        directory; a unit "appears" in a session when its
        ``<unit_id>.npy`` spike-time file is present in that session's
        ``ephys/<probe_id>/cluster_data`` directory. The per-session
        firing rate is the unit's spike count divided by that session's
        ``total_video_time_least``, and the catalog firing rate is the
        median across the sessions the unit appears in.

        Parameters
        ----------
        unit_file_names : dict
            Maps each analyzer unit id to its ``<unit_id>.npy`` cluster
            data file name.

        Returns
        -------
        dict
            Keyed by analyzer unit id; each value is a ``dict`` with
            ``rec_sessions`` (list of raw session directory names),
            ``hs_sn`` (headstage serial — a single value when constant
            across sessions, otherwise a list), ``probe_sn`` (probe
            serial number) and ``median_fr`` (median firing rate).
        """
        changepoints_path = next(self.ephys_path.glob('changepoints_info*.json'))
        with open(changepoints_path, 'r') as changepoints_json_file:
            changepoints_info = json.load(changepoints_json_file)

        session_ids = list(changepoints_info.keys())
        probe_sn = changepoints_info[session_ids[0]]['imec_probe_sn']

        # Cache each session's video time and its cluster-data file listing.
        session_video_time = {}
        session_cluster_files = {}
        for session_id in session_ids:
            root_directory = changepoints_info[session_id]['root_directory']
            frame_count_path = sorted(
                glob.glob(f"{root_directory}{os.sep}video{os.sep}*_camera_frame_count_dict.json")
            )[0]
            with open(frame_count_path, 'r') as frame_count_infile:
                session_video_time[session_id] = json.load(frame_count_infile)['total_video_time_least']
            cluster_data_dir = f"{root_directory}{os.sep}ephys{os.sep}{self.probe_id}{os.sep}cluster_data"
            session_cluster_files[session_id] = {
                os.path.basename(x_name) for x_name in glob.glob(f"{cluster_data_dir}{os.sep}*.npy")
            }

        result = {}
        for unit_id, unit_file_name in unit_file_names.items():
            rec_sessions = []
            hs_ids = []
            firing_rate_dict = {}
            for session_id in session_ids:
                if unit_file_name in session_cluster_files[session_id]:
                    root_directory = changepoints_info[session_id]['root_directory']
                    rec_sessions.append(root_directory.split(os.sep)[-1])
                    hs_ids.append(changepoints_info[session_id]['headstage_sn'])

                    cluster_data_dir = f"{root_directory}{os.sep}ephys{os.sep}{self.probe_id}{os.sep}cluster_data"
                    loaded_spike_times = np.load(f"{cluster_data_dir}{os.sep}{unit_file_name}")
                    firing_rate_dict[session_id] = round(
                        loaded_spike_times.shape[1] / session_video_time[session_id], 3
                    )

            median_fr = round(np.median(np.array(list(firing_rate_dict.values()))), 3) if firing_rate_dict else np.nan

            hs_sn = hs_ids
            if len(rec_sessions) > 0 and all(hs_sn_item == hs_ids[0] for hs_sn_item in hs_ids):
                hs_sn = hs_ids[0]

            result[unit_id] = {
                'rec_sessions': rec_sessions,
                'hs_sn': hs_sn,
                'probe_sn': probe_sn,
                'median_fr': median_fr,
            }

        return result

    def _resolve_catalog_path(self, catalog_path: str | os.PathLike | None) -> Path:
        """
        Description
        -----------
        Resolve the global ``unit_catalog.csv`` path, applying the
        default location when none is given. The default is a single
        ``unit_catalog.csv`` in the parent of the session's EPHYS
        directory, shared across every session and probe.

        Parameters
        ----------
        catalog_path : str or pathlib.Path or None
            Explicit catalog path, or ``None`` to use the default
            location.

        Returns
        -------
        pathlib.Path
            The resolved catalog path.
        """
        return (
            Path(catalog_path) if catalog_path is not None
            else self.ephys_path.parent / 'unit_catalog.csv'
        )

    def is_session_in_catalog(self, catalog_path: str | os.PathLike | None = None) -> bool:
        """
        Description
        -----------
        Report whether this session + probe already has rows in the
        global ``unit_catalog.csv``. Rows are matched on the same
        identity :meth:`build_session_catalog` uses for its in-place
        update — ``rec_date`` + ``mouse_id`` + a ``unit_id`` prefixed
        with ``"{probe_id}_"`` — so a ``True`` here means a re-run would
        overwrite, not duplicate.

        Parameters
        ----------
        catalog_path : str or pathlib.Path or None, default None
            Path of the global ``unit_catalog.csv``. When ``None``,
            resolved via :meth:`_resolve_catalog_path`.

        Returns
        -------
        bool
            ``True`` when at least one row for this session + probe is
            already present, ``False`` otherwise (including when the
            catalog file does not yet exist).
        """
        catalog_path = self._resolve_catalog_path(catalog_path)
        if not catalog_path.exists():
            return False
        existing_catalog = pd.read_csv(catalog_path)
        this_session = (
            (existing_catalog['rec_date'] == int(self.session_date))
            & (existing_catalog['mouse_id'].astype(str) == str(self.mouse_id))
            & (existing_catalog['unit_id'].astype(str).str.startswith(f"{self.probe_id}_"))
        )
        return bool(this_session.any())

    def build_session_catalog(self, catalog_path: str | os.PathLike | None = None, write: bool = True) -> pd.DataFrame:
        """
        Description
        -----------
        Assemble this session's per-unit rows from the computed quality
        metrics, template metrics, unit locations and cross-session
        firing rates, and merge them into the global ``unit_catalog.csv``.

        Each unit's catalog id is ``<probe_id>_cl<cluster_id:04d>_ch<peak
        _channel:03d>_<cluster_group>``; the cluster group is looked up
        per cluster id from ``cluster_info.tsv``. Units that do not
        appear in any recording session (no cross-session spike-time
        file) are skipped, matching the notebook. Requires both
        :meth:`compute_metrics` and
        :meth:`compute_recording_dependent_metrics` to have run (the
        merged :attr:`quality_metrics` and the ``noise_levels`` extension
        are both consumed here).

        The write is idempotent: any rows already in the global catalog
        for this session (matched on ``mouse_id`` + ``rec_date`` +
        probe) are dropped before this session's fresh rows are
        appended, so re-processing a session updates it in place.

        Parameters
        ----------
        catalog_path : str or pathlib.Path or None, default None
            Path of the global catalog CSV. When ``None``, defaults to
            ``unit_catalog.csv`` in the EPHYS directory (the parent of
            this session's EPHYS subdirectory).
        write : bool, default True
            When ``True``, merge this session's rows into the global
            catalog on disk and set :attr:`catalog_path`.

        Returns
        -------
        pandas.DataFrame
            This session's rows, with columns :data:`CATALOG_COLUMNS`.
            Also stored on :attr:`session_catalog`. The global catalog
            on disk additionally contains every other session's rows.
        """
        cluster_info_df = pd.read_csv(self.ks_path / 'cluster_info.tsv', sep='\t')
        group_by_cluster_id = dict(zip(cluster_info_df['cluster_id'], cluster_info_df['group']))

        noise_levels = self.analyzer.get_extension('noise_levels').get_data()
        peak_channels = self.sorting.get_property('ch')
        rec_date = int(self.session_date)

        # Build the <unit_id>.npy file name for every unit, then resolve
        # the cross-session firing-rate information in one pass.
        unit_file_names = {}
        unit_id_strings = {}
        for unit_index, unit_id in enumerate(self.analyzer.unit_ids):
            cluster_group = group_by_cluster_id[unit_id]
            one_peak_ch = int(peak_channels[unit_index])
            unit_id_string = f"{self.probe_id}_cl{int(unit_id):04d}_ch{one_peak_ch:03d}_{cluster_group}"
            unit_id_strings[unit_id] = unit_id_string
            unit_file_names[unit_id] = f"{unit_id_string}.npy"

        cross_session = self._compute_cross_session_firing_rates(unit_file_names)

        rows = []
        for unit_index, unit_id in enumerate(self.analyzer.unit_ids):
            cross_session_info = cross_session[unit_id]
            if len(cross_session_info['rec_sessions']) == 0:
                continue

            cluster_group = group_by_cluster_id[unit_id]
            one_peak_ch = int(peak_channels[unit_index])
            template_metrics = self.template_metrics[unit_id]
            unit_location = self.unit_locations[unit_id]
            quality_metrics = self.quality_metrics.loc[unit_id]

            rows.append([
                rec_date,
                self.mouse_id,
                cross_session_info['rec_sessions'],
                cross_session_info['probe_sn'],
                cross_session_info['hs_sn'],
                self.kilosort_version,
                self.phy_curated,
                unit_id_strings[unit_id],
                cluster_group,
                template_metrics['somatic'],
                np.nan,
                unit_location['location'][1],
                unit_location['location'][0],
                unit_location['location'][2],
                unit_location['closest_channel'],
                unit_location['brain_region'],
                cross_session_info['median_fr'],
                noise_levels[one_peak_ch],
                template_metrics['peak_to_valley'],
                template_metrics['peak_trough_ratio'],
                template_metrics['half_width'],
                template_metrics['repolarization_slope'],
                template_metrics['recovery_slope'],
                template_metrics['exp_decay'],
                template_metrics['spread'],
                quality_metrics['amplitude_cutoff'],
                quality_metrics['amplitude_cv_median'],
                quality_metrics['amplitude_cv_range'],
                quality_metrics['amplitude_median'],
                quality_metrics['firing_range'],
                quality_metrics['firing_rate'],
                quality_metrics['isi_violations_ratio'],
                quality_metrics['isi_violations_count'],
                quality_metrics['num_spikes'],
                quality_metrics['presence_ratio'],
                quality_metrics['rp_contamination'],
                quality_metrics['sd_ratio'],
                quality_metrics['snr'],
                quality_metrics['sync_spike_2'],
                quality_metrics['sync_spike_4'],
                quality_metrics['sync_spike_8'],
                quality_metrics['d_prime'],
                quality_metrics['isolation_distance'],
                quality_metrics['l_ratio'],
                quality_metrics['silhouette'],
                quality_metrics['nn_hit_rate'],
                quality_metrics['nn_miss_rate'],
            ])

        session_catalog = pd.DataFrame(rows, columns=CATALOG_COLUMNS)

        if write:
            catalog_path = self._resolve_catalog_path(catalog_path)
            if catalog_path.exists():
                existing_catalog = pd.read_csv(catalog_path)
                # drop any rows already present for this session so a
                # re-run updates them in place rather than duplicating
                this_session = (
                    (existing_catalog['rec_date'] == rec_date)
                    & (existing_catalog['mouse_id'].astype(str) == str(self.mouse_id))
                    & (existing_catalog['unit_id'].astype(str).str.startswith(f"{self.probe_id}_"))
                )
                global_catalog = pd.concat(
                    [existing_catalog[~this_session], session_catalog], ignore_index=True
                )
            else:
                global_catalog = session_catalog
            global_catalog.to_csv(catalog_path, index=False)
            self.catalog_path = catalog_path

        self.session_catalog = session_catalog
        return session_catalog

    def run(
        self,
        output_dir: str | os.PathLike | None = None,
        catalog_path: str | os.PathLike | None = None,
        write: bool = True,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """
        Description
        -----------
        Run the full per-session pipeline end to end: :meth:`load`, the
        sweep-free core pass (:meth:`compute_metrics`), the
        recording-dependent pass (:meth:`compute_recording_dependent_metrics`),
        :meth:`compute_unit_locations`,
        :meth:`write_channel_order_per_shank` and
        :meth:`build_session_catalog`.

        Parameters
        ----------
        output_dir : str or pathlib.Path or None, default None
            Directory for the ``channel_order_per_shank.json``. When
            ``None``, it goes into the session's EPHYS directory.
        catalog_path : str or pathlib.Path or None, default None
            Path of the global ``unit_catalog.csv``. When ``None``,
            defaults to the EPHYS directory.
        write : bool, default True
            Passed through to :meth:`build_session_catalog`. When
            ``False``, the channel-order JSON is still written (it has no
            in-memory consumer) but the global catalog is not updated.
        overwrite : bool, default False
            When ``False`` and this session + probe already has rows in
            the global catalog (see :meth:`is_session_in_catalog`), the
            whole pipeline is skipped — the slow recording-dependent
            pass is not repeated — and the existing rows are returned.
            Set ``True`` to recompute and overwrite those rows in place
            regardless. Ignored when ``write`` is ``False`` (there is no
            catalog to check against).

        Returns
        -------
        pandas.DataFrame
            This session's catalog rows — freshly computed, or, when
            skipped, the rows already present in the global catalog.
        """
        if write and not overwrite and self.is_session_in_catalog(catalog_path):
            catalog_path = self._resolve_catalog_path(catalog_path)
            existing_catalog = pd.read_csv(catalog_path)
            this_session = (
                (existing_catalog['rec_date'] == int(self.session_date))
                & (existing_catalog['mouse_id'].astype(str) == str(self.mouse_id))
                & (existing_catalog['unit_id'].astype(str).str.startswith(f"{self.probe_id}_"))
            )
            session_catalog = existing_catalog[this_session].reset_index(drop=True)
            print(
                f"{self.session_date} {self.probe_id} already in {catalog_path} "
                f"({session_catalog.shape[0]} units); skipping — pass overwrite=True to recompute."
            )
            self.catalog_path = catalog_path
            self.session_catalog = session_catalog
            return session_catalog
        self.load()
        self.compute_metrics()
        self.compute_recording_dependent_metrics()
        self.compute_unit_locations()
        self.write_channel_order_per_shank(output_dir=output_dir)
        return self.build_session_catalog(catalog_path=catalog_path, write=write)
