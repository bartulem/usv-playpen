"""
@author: bartulem
Tests for ``usv_playpen.neuropixels.spike_quality_metrics``.

The orchestrator is mostly integration glue over real session data, so
these tests cover the pure-logic pieces that are both self-contained and
historically bug-prone: the metric-name / catalog-column bookkeeping and
the per-shank channel ordering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from spikeinterface.metrics.quality.quality_metrics import get_quality_metric_list

from usv_playpen.neuropixels.spike_quality_metrics import (
    CATALOG_COLUMNS,
    QM_PARAMS,
    RECORDING_DEPENDENT_METRIC_NAMES,
    SPIKE_TRAIN_METRIC_NAMES,
    SpikeQualityMetricsExtractor,
)


def _new_extractor():
    """Build a bare extractor instance, bypassing ``__init__`` and its disk I/O."""
    instance = SpikeQualityMetricsExtractor.__new__(SpikeQualityMetricsExtractor)
    instance.somatic_classifier = None
    return instance


def test_resolve_catalog_path_explicit_and_default():
    """
    Description
    -----------
    An explicit ``catalog_path`` is returned verbatim (as a ``Path``);
    ``None`` falls back to ``unit_catalog.csv`` beside the EPHYS root
    (the parent of the session's ephys directory).
    """

    extractor = _new_extractor()
    extractor.ephys_path = Path("/data/EPHYS/20240101_imec0")
    assert extractor._resolve_catalog_path("/x/custom.csv") == Path("/x/custom.csv")
    assert extractor._resolve_catalog_path(None) == Path("/data/EPHYS/unit_catalog.csv")


def test_metric_name_split_is_valid_and_disjoint():
    """
    Description
    -----------
    The spike-train and recording-dependent metric-name lists must be
    disjoint, every name must be a real stock-``spikeinterface==0.104.3``
    quality metric (this guards against version drift such as
    ``isolation_distance``/``l_ratio`` having merged into ``mahalanobis``),
    and :data:`QM_PARAMS` must only carry params for metrics that are
    actually requested — a stale params key raises a ``KeyError`` in
    ``compute_quality_metrics``.
    """
    spike_train = set(SPIKE_TRAIN_METRIC_NAMES)
    recording_dependent = set(RECORDING_DEPENDENT_METRIC_NAMES)
    valid_metric_names = set(get_quality_metric_list())

    assert spike_train.isdisjoint(recording_dependent)
    assert (spike_train | recording_dependent) <= valid_metric_names
    assert set(QM_PARAMS) <= (spike_train | recording_dependent)


def test_catalog_has_55_unique_columns():
    """
    Description
    -----------
    The per-session catalog schema must have exactly 55 columns with no
    duplicates — the row builder in ``build_session_catalog`` appends 55
    values positionally, so a schema drift would silently misalign every
    column. (47 base metrics + the 8 waveform-shape features.)
    """
    assert len(CATALOG_COLUMNS) == 55
    assert len(set(CATALOG_COLUMNS)) == 55


def test_write_channel_order_per_shank_orders_by_electrode_position(tmp_path):
    """
    Description
    -----------
    :meth:`write_channel_order_per_shank` must group channels by their
    Kilosort ``channel_shanks.npy`` value (1-indexed physical shank)
    and order each shank's channels by axial position
    (``channel_positions.npy[:, 1]``) ascending — tip outward —
    leaving shanks with no channels empty. The output JSON uses
    0-indexed ``shank_N`` keys, so the filter is
    ``ks_shank == json_shank + 1``.
    """
    extractor = _new_extractor()
    # Synthetic Kilosort sidecar files. 5 channels: 3 on shank 1 at
    # axials 75/30/135, 2 on shank 2 at axials 120/45.
    ks_dir = tmp_path / "kilosort4"
    ks_dir.mkdir()
    np.save(ks_dir / "channel_positions.npy", np.array([
        [27.0,  75.0],
        [27.0,  30.0],
        [277.0, 120.0],
        [277.0, 45.0],
        [27.0,  135.0],
    ]))
    np.save(ks_dir / "channel_shanks.npy", np.array([1, 1, 2, 2, 1]))
    extractor.ks_path = ks_dir

    out_path = extractor.write_channel_order_per_shank(output_dir=tmp_path)
    assert out_path == tmp_path / 'channel_order_per_shank.json'

    info = json.loads(out_path.read_text())
    assert info['shank_0'] == [1, 0, 4]   # axials 30, 75, 135 (tip outward)
    assert info['shank_1'] == [3, 2]      # axials 45, 120
    assert info['shank_2'] == []
    assert info['shank_3'] == []


def _write_stub_catalog(path):
    """
    Write a minimal global ``unit_catalog.csv`` covering two sessions and
    two probes — only the three columns the catalog-presence check reads
    (``rec_date``, ``mouse_id``, ``unit_id``). The 20241107 session has
    both ``imec0`` and ``imec1`` rows; the 20250909 session has ``imec0``
    only.
    """
    pd.DataFrame(
        {
            'rec_date': [20241107, 20241107, 20241107, 20250909],
            'mouse_id': ['158112_0', '158112_0', '158112_0', '158112_0'],
            'unit_id': ['imec0_0', 'imec0_1', 'imec1_0', 'imec0_0'],
        }
    ).to_csv(path, index=False)


def test_is_session_in_catalog_false_when_catalog_missing(tmp_path):
    """
    Description
    -----------
    :meth:`is_session_in_catalog` must return ``False`` — not raise —
    when the global catalog file does not exist yet, so the very first
    run of a session is never mistaken for an already-catalogued one.
    """
    extractor = _new_extractor()
    extractor.session_date = '20241107'
    extractor.mouse_id = '158112_0'
    extractor.probe_id = 'imec0'

    assert extractor.is_session_in_catalog(tmp_path / 'unit_catalog.csv') is False


def test_is_session_in_catalog_matches_on_rec_date_mouse_probe(tmp_path):
    """
    Description
    -----------
    :meth:`is_session_in_catalog` must report a session + probe as
    present only when ``rec_date``, ``mouse_id`` and the ``{probe_id}_``
    ``unit_id`` prefix all match — the exact identity
    :meth:`build_session_catalog` drops on — so the catalog check and the
    in-place catalog update can never disagree about what is already
    there.
    """
    catalog_path = tmp_path / 'unit_catalog.csv'
    _write_stub_catalog(catalog_path)

    extractor = _new_extractor()
    extractor.mouse_id = '158112_0'

    # present: both probes of the 20241107 session are catalogued
    extractor.session_date, extractor.probe_id = '20241107', 'imec0'
    assert extractor.is_session_in_catalog(catalog_path) is True
    extractor.session_date, extractor.probe_id = '20241107', 'imec1'
    assert extractor.is_session_in_catalog(catalog_path) is True

    # absent: 20250909 has only imec0 rows, so its imec1 is not catalogued
    extractor.session_date, extractor.probe_id = '20250909', 'imec1'
    assert extractor.is_session_in_catalog(catalog_path) is False

    # absent: a different mouse never matches, even on a catalogued date
    extractor.session_date, extractor.probe_id = '20241107', 'imec0'
    extractor.mouse_id = '999999_0'
    assert extractor.is_session_in_catalog(catalog_path) is False


def test_run_skips_pipeline_when_session_already_catalogued(tmp_path):
    """
    Description
    -----------
    With ``overwrite=False`` (the default) and the session + probe
    already in the catalog, :meth:`run` must return the existing rows
    without touching the recording — proven here by the bare extractor
    carrying none of the attributes :meth:`load` needs, so any attempt to
    compute would raise ``AttributeError``. The returned frame,
    :attr:`session_catalog` and :attr:`catalog_path` must all reflect the
    catalogued rows for this session + probe.
    """
    catalog_path = tmp_path / 'unit_catalog.csv'
    _write_stub_catalog(catalog_path)

    extractor = _new_extractor()
    extractor.session_date = '20241107'
    extractor.mouse_id = '158112_0'
    extractor.probe_id = 'imec0'

    session_catalog = extractor.run(catalog_path=catalog_path)

    assert list(session_catalog['unit_id']) == ['imec0_0', 'imec0_1']
    assert extractor.catalog_path == catalog_path
    assert list(extractor.session_catalog['unit_id']) == ['imec0_0', 'imec0_1']


def test_run_overwrite_true_bypasses_the_catalog_skip(tmp_path):
    """
    Description
    -----------
    ``overwrite=True`` must bypass the catalog-presence check even when
    the session + probe is already catalogued — :meth:`run` then proceeds
    straight into :meth:`load`, which on this bare extractor (no
    ``ephys_path`` and friends) raises ``AttributeError``. Reaching that
    error is the proof the skip gate was not taken.
    """
    catalog_path = tmp_path / 'unit_catalog.csv'
    _write_stub_catalog(catalog_path)

    extractor = _new_extractor()
    extractor.session_date = '20241107'
    extractor.mouse_id = '158112_0'
    extractor.probe_id = 'imec0'

    with pytest.raises(AttributeError):
        extractor.run(catalog_path=catalog_path, overwrite=True)


# Stub SpikeInterface objects — duck-typed stand-ins exposing only the
# handful of attributes / methods the recording-free metric methods read,
# so the template / amplitude / unit-location / catalog logic can be
# exercised without a real SortingAnalyzer or recording stream.


class _StubExtension:
    """
    Description
    -----------
    Minimal analyzer-extension stand-in. ``get_data`` returns the array
    handed in; ``waveforms`` extensions also carry an ``nbefore`` index
    and the ``random_spikes`` extension exposes ``get_random_spikes``.
    """

    def __init__(self, data=None, nbefore=None, random_spikes=None):
        self._data = data
        self.nbefore = nbefore
        self._random_spikes = random_spikes

    def get_data(self):
        return self._data

    def get_random_spikes(self):
        return self._random_spikes


class _StubSparsity:
    """Stand-in for ``ChannelSparsity`` exposing only ``mask`` and the
    per-unit channel-index mapping the metric methods consult."""

    def __init__(self, mask=None, unit_id_to_channel_indices=None):
        self.mask = mask
        self.unit_id_to_channel_indices = unit_id_to_channel_indices


class _StubAnalyzer:
    """Duck-typed ``SortingAnalyzer`` carrying the unit ids, sampling
    frequency, channel locations, sparsity and a name→extension map."""

    def __init__(self, unit_ids, sampling_frequency=None,
                 channel_locations=None, sparsity=None, extensions=None):
        self.unit_ids = unit_ids
        self.sampling_frequency = sampling_frequency
        self._channel_locations = channel_locations
        self.sparsity = sparsity
        self._extensions = extensions or {}

    def get_channel_locations(self):
        return self._channel_locations

    def get_extension(self, name):
        return self._extensions[name]


class _StubSorting:
    """Stand-in for ``BaseSorting`` exposing per-unit spike trains and the
    ``ch`` peak-channel property."""

    def __init__(self, spike_trains=None, ch=None):
        self._spike_trains = spike_trains or {}
        self._ch = ch

    def get_unit_spike_train(self, unit_id):
        return self._spike_trains[unit_id]

    def get_property(self, key):
        assert key == 'ch'
        return self._ch


class _StubRecording:
    """Stand-in for ``BaseRecording`` exposing only the total sample count."""

    def __init__(self, n_samples):
        self._n = n_samples

    def get_num_samples(self):
        return self._n


def _biphasic(amp, n_samples=60, trough=20, peak=35):
    """
    Description
    -----------
    Build a single-channel action-potential-shaped waveform: a negative
    trough at sample ``trough`` and a smaller positive rebound at
    ``peak``, scaled by ``amp``.

    Returns
    -------
    numpy.ndarray
        Length-``n_samples`` waveform.
    """

    t = np.arange(n_samples)
    trough_bump = -1.0 * np.exp(-0.5 * ((t - trough) / 2.5) ** 2)
    peak_bump = 0.4 * np.exp(-0.5 * ((t - peak) / 3.0) ** 2)
    return amp * (trough_bump + peak_bump)


def _make_meta(ephys_path):
    """Write a minimal ``concatenated_*.ap.meta`` with the keys the
    constructor parses (imSampRate / probe type / imroTbl / snsGeomMap)."""
    ephys_path.mkdir(parents=True, exist_ok=True)
    (ephys_path / "concatenated_run.ap.meta").write_text(
        "imSampRate=30000.0\n"
        "imAiRangeMax=0.62\n"
        "imMaxInt=2048\n"
        "imDatPrb_type=2013\n"
        "~imroTbl=(2013,384)(0 3 1 0 576)(1 3 1 0 577)\n"
        "~snsGeomMap=(0,0)(0:27:0:1)(0:59:0:1)\n"
    )


def test_constructor_resolves_paths_and_parses_meta(tmp_path):
    """
    Description
    -----------
    Construction resolves the EPHYS / Kilosort / channel-locations paths
    and parses the single ``concatenated_*.ap.meta`` into ``meta`` — all
    without reading any recording data.
    """

    _make_meta(tmp_path / "EPHYS" / "20240101_imec0")
    extractor = SpikeQualityMetricsExtractor(
        os_cup_loc=tmp_path, mouse_id="M1", session_date="20240101",
        probe_id="imec0", hemisphere="R",
    )
    assert extractor.ks_path == tmp_path / "EPHYS" / "20240101_imec0" / "kilosort4"
    assert extractor.channel_locations_file.name == "channel_locations.json"
    assert extractor.meta["imSampRate"] == "30000.0"


def test_constructor_rejects_invalid_hemisphere(tmp_path):
    """
    Description
    -----------
    The hemisphere must be ``'L'`` or ``'R'``; any other value raises
    ``ValueError`` before any path resolution or file access.
    """

    with pytest.raises(ValueError, match="hemisphere must be 'L' or 'R'"):
        SpikeQualityMetricsExtractor(
            os_cup_loc=tmp_path, mouse_id="M1", session_date="20240101",
            probe_id="imec0", hemisphere="X",
        )


def test_constructor_raises_when_no_meta_found(tmp_path):
    """
    Description
    -----------
    With no ``concatenated_*.ap.meta`` under the EPHYS directory the
    constructor raises ``FileNotFoundError``.
    """

    (tmp_path / "EPHYS" / "20240101_imec0").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match=r"No concatenated_.*ap.meta"):
        SpikeQualityMetricsExtractor(
            os_cup_loc=tmp_path, mouse_id="M1", session_date="20240101",
            probe_id="imec0", hemisphere="R",
        )


def test_constructor_raises_on_multiple_meta(tmp_path):
    """
    Description
    -----------
    Two ``concatenated_*.ap.meta`` files are ambiguous and raise
    ``RuntimeError``.
    """

    ephys = tmp_path / "EPHYS" / "20240101_imec0"
    ephys.mkdir(parents=True)
    for name in ("concatenated_a.ap.meta", "concatenated_b.ap.meta"):
        (ephys / name).write_text("imSampRate=30000.0\nimroTbl=(2013,384)\nsnsGeomMap=(0,0)\n")
    with pytest.raises(RuntimeError, match="Multiple concatenated"):
        SpikeQualityMetricsExtractor(
            os_cup_loc=tmp_path, mouse_id="M1", session_date="20240101",
            probe_id="imec0", hemisphere="R",
        )


def test_compute_template_metrics_populates_all_keys():
    """
    Description
    -----------
    :meth:`_compute_template_metrics` computes, per unit, the single-
    channel metrics (peak-to-valley, peak/trough ratio, half-width,
    repolarization / recovery slopes, somatic flag) on the extremum
    channel and the multi-channel metrics (exp_decay, spread) on the
    sparse template. With a clean biphasic template on channel 0 every
    key is produced and ``somatic`` is a bool.
    """

    extractor = _new_extractor()
    n_channels = 4
    amps = np.array([1.0, 0.7, 0.4, 0.2])
    template = np.stack([_biphasic(a) for a in amps], axis=1)   # (60, 4)
    extractor.dense_templates = template[None, :, :]            # (1, 60, 4)
    channel_locations = np.column_stack([np.zeros(n_channels), np.arange(n_channels) * 30.0])
    extractor.analyzer = _StubAnalyzer(
        unit_ids=np.array([0]),
        sampling_frequency=30000.0,
        channel_locations=channel_locations,
        sparsity=_StubSparsity(mask=np.ones((1, n_channels), dtype=bool)),
    )

    extractor._compute_template_metrics()
    metrics = extractor.template_metrics[0]
    expected = {
        'peak_to_valley', 'peak_trough_ratio', 'half_width',
        'repolarization_slope', 'recovery_slope', 'somatic',
        'exp_decay', 'spread',
        'main_trough_size', 'main_peak_before_size', 'main_peak_after_size',
        'main_peak_before_width', 'main_trough_width', 'peak1_to_peak2_ratio',
        'trough_to_peak2_ratio', 'main_peak_to_trough_ratio',
    }
    assert set(metrics) == expected
    assert isinstance(metrics['somatic'], bool)


def test_compute_amplitude_metrics_returns_per_unit_frame():
    """
    Description
    -----------
    :meth:`_compute_amplitude_metrics` reads each unit's windowed
    waveforms, takes the per-spike amplitude at the extremum channel, and
    returns a frame of ``amplitude_cutoff`` / ``amplitude_cv_*`` /
    ``amplitude_median`` / ``sd_ratio``. A unit with no random spikes
    yields an all-NaN row.
    """

    extractor = _new_extractor()
    n_sparse, n_samples, nbefore = 3, 60, 20
    n_spikes = 40
    # Unit 0's waveforms: a biphasic on sparse channel 1 (the extremum).
    waveforms = np.zeros((n_spikes, n_samples, n_sparse))
    rng = np.random.default_rng(0)
    for s in range(n_spikes):
        waveforms[s, :, 1] = _biphasic(1.0 + 0.05 * rng.standard_normal())
    random_spikes = np.zeros(n_spikes, dtype=[('unit_index', 'i8'), ('sample_index', 'i8')])
    random_spikes['unit_index'] = 0
    random_spikes['sample_index'] = np.linspace(0, 2_900_000, n_spikes).astype(np.int64)

    extractor.analyzer = _StubAnalyzer(
        unit_ids=np.array([0, 1]),
        sampling_frequency=30000.0,
        sparsity=_StubSparsity(
            unit_id_to_channel_indices={0: np.array([0, 1, 2]), 1: np.array([0, 1, 2])},
        ),
        extensions={
            'noise_levels': _StubExtension(data=np.array([1.0, 1.0, 1.0, 1.0])),
            'waveforms': _StubExtension(data=waveforms, nbefore=nbefore),
            'random_spikes': _StubExtension(random_spikes=random_spikes),
        },
    )
    extractor.recording = _StubRecording(n_samples=3_000_000)
    extractor.dense_templates = np.zeros((2, n_samples, 4))
    extractor.dense_templates[0, :, 1] = _biphasic(1.0)
    extractor.sorting = _StubSorting(spike_trains={0: np.arange(200)})

    frame = extractor._compute_amplitude_metrics()
    assert list(frame.columns) == [
        'amplitude_cutoff', 'amplitude_cv_median', 'amplitude_cv_range',
        'amplitude_median', 'sd_ratio',
    ]
    assert list(frame.index) == [0, 1]
    assert np.isfinite(frame.loc[0, 'amplitude_median'])
    assert frame.loc[1].isna().all()   # empty-waveforms unit


@pytest.mark.parametrize("hemisphere", ["R", "L"])
def test_compute_unit_locations_locks_region_to_peak_shank(tmp_path, hemisphere):
    """
    Description
    -----------
    :meth:`compute_unit_locations` joins IBL anatomy by physical position
    (ignoring non-``channel_`` keys such as ``origin``), restricts the
    candidate channels to the unit's template-peak shank, triangulates,
    and assigns the closest channel's brain region. A template peaking on
    shank 1 (channels 2/3) yields a closest channel on that shank and its
    region, for both the right- and left-hemisphere AP-fold signs.
    """

    extractor = _new_extractor()
    extractor.hemisphere = hemisphere
    extractor.shank_spacing_microns = 250
    extractor.shank_width_microns = 70

    ks = tmp_path / "kilosort4"
    ks.mkdir()
    np.save(ks / "channel_positions.npy",
            np.array([[0, 0], [0, 20], [250, 0], [250, 20]], dtype=np.int64))
    np.save(ks / "channel_shanks.npy", np.array([0, 0, 1, 1], dtype=np.int64))
    extractor.ks_path = ks

    cl = tmp_path / "channel_locations.json"
    cl.write_text(json.dumps({
        "channel_0": {"lateral": 0, "axial": 0, "x": 0.0, "y": 0.0, "z": 0.0, "brain_region": "PAG"},
        "channel_1": {"lateral": 0, "axial": 20, "x": 0.0, "y": 20.0, "z": 0.0, "brain_region": "PAG"},
        "channel_2": {"lateral": 250, "axial": 0, "x": 10.0, "y": 0.0, "z": 0.0, "brain_region": "VISp"},
        "channel_3": {"lateral": 250, "axial": 20, "x": 10.0, "y": 20.0, "z": 0.0, "brain_region": "VISp"},
        "origin": {"x": 0.0, "y": 0.0, "z": 0.0},
    }))
    extractor.channel_locations_file = cl

    template = np.zeros((60, 4))
    template[:, 2] = _biphasic(5.0)    # peak amplitude on channel 2 (shank 1)
    template[:, 3] = _biphasic(2.0)
    extractor.dense_templates = template[None, :, :]
    extractor.analyzer = _StubAnalyzer(
        unit_ids=np.array([0]),
        sparsity=_StubSparsity(unit_id_to_channel_indices={0: np.array([0, 1, 2, 3])}),
    )

    extractor.compute_unit_locations()
    loc = extractor.unit_locations[0]
    assert loc['location'].shape == (3,)
    assert int(loc['closest_channel']) in (2, 3)
    assert loc['brain_region'] == "VISp"


def test_compute_cross_session_firing_rates(tmp_path):
    """
    Description
    -----------
    :meth:`_compute_cross_session_firing_rates` reads ``changepoints_info``,
    finds the sessions a unit's cluster-data ``.npy`` appears in, and
    returns the per-session recording list, probe / headstage serials and
    the median firing rate (spike count / session video time).
    """

    extractor = _new_extractor()
    extractor.probe_id = 'imec0'
    ephys = tmp_path / "EPHYS" / "20240101_imec0"
    ephys.mkdir(parents=True)
    extractor.ephys_path = ephys

    root = tmp_path / "raw" / "session_one"
    (root / "video").mkdir(parents=True)
    (root / "video" / "x_camera_frame_count_dict.json").write_text(
        json.dumps({"total_video_time_least": 100.0})
    )
    cluster_data = root / "ephys" / "imec0" / "cluster_data"
    cluster_data.mkdir(parents=True)
    np.save(cluster_data / "imec0_cl0000_ch005_good.npy", np.zeros((1, 250)))

    (ephys / "changepoints_info_x.json").write_text(json.dumps({
        "sess1": {
            "root_directory": str(root),
            "imec_probe_sn": "PSN123",
            "headstage_sn": "HS9",
        }
    }))

    result = extractor._compute_cross_session_firing_rates(
        {0: "imec0_cl0000_ch005_good.npy"}
    )
    assert result[0]['rec_sessions'] == ["session_one"]
    assert result[0]['probe_sn'] == "PSN123"
    assert result[0]['hs_sn'] == "HS9"
    assert result[0]['median_fr'] == 2.5   # 250 spikes / 100 s


def _full_quality_metrics_frame(unit_ids):
    """Build a quality-metrics DataFrame with every column
    :meth:`build_session_catalog` reads, indexed by ``unit_ids``."""
    columns = [
        'amplitude_cutoff', 'amplitude_cv_median', 'amplitude_cv_range',
        'amplitude_median', 'firing_range', 'firing_rate', 'isi_violations_ratio',
        'isi_violations_count', 'num_spikes', 'presence_ratio', 'rp_contamination',
        'sd_ratio', 'snr', 'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
        'd_prime', 'isolation_distance', 'l_ratio', 'silhouette',
        'nn_hit_rate', 'nn_miss_rate',
    ]
    data = {c: [float(i) for i in range(len(unit_ids))] for c in columns}
    return pd.DataFrame(data, index=list(unit_ids))


def _template_metrics_for(unit_ids):
    """Build a per-unit template-metrics dict with the keys
    :meth:`build_session_catalog` reads."""
    keys = ('peak_to_valley', 'peak_trough_ratio', 'half_width',
            'repolarization_slope', 'recovery_slope', 'exp_decay', 'spread',
            'main_trough_size', 'main_peak_before_size', 'main_peak_after_size',
            'main_peak_before_width', 'main_trough_width', 'peak1_to_peak2_ratio',
            'trough_to_peak2_ratio', 'main_peak_to_trough_ratio')
    return {u: {**dict.fromkeys(keys, 0.1), 'somatic': True} for u in unit_ids}


def test_build_session_catalog_assembles_rows_and_merges_idempotently(tmp_path):
    """
    Description
    -----------
    :meth:`build_session_catalog` assembles one row per unit (with the
    55-column schema), writes the global catalog, and on a second call
    replaces this session's rows rather than duplicating them. Units that
    appear in no recording session are skipped.
    """

    extractor = _new_extractor()
    extractor.session_date = '20240101'
    extractor.mouse_id = 'M1'
    extractor.probe_id = 'imec0'
    extractor.kilosort_version = '4'
    extractor.phy_curated = True

    ks = tmp_path / "kilosort4"
    ks.mkdir()
    pd.DataFrame({'cluster_id': [0, 1], 'group': ['good', 'mua']}).to_csv(
        ks / 'cluster_info.tsv', sep='\t', index=False
    )
    extractor.ks_path = ks

    unit_ids = np.array([0, 1])
    extractor.analyzer = _StubAnalyzer(
        unit_ids=unit_ids,
        extensions={'noise_levels': _StubExtension(data=np.full(10, 0.5))},
    )
    extractor.sorting = _StubSorting(ch=np.array([5, 6]))
    extractor.template_metrics = _template_metrics_for(unit_ids)
    extractor.unit_locations = {
        u: {'location': np.array([1.0, 2.0, 3.0]), 'closest_channel': 5, 'brain_region': 'PAG'}
        for u in unit_ids
    }
    extractor.quality_metrics = _full_quality_metrics_frame(unit_ids)

    # Unit 0 appears in a session; unit 1 appears in none (gets skipped).
    extractor._compute_cross_session_firing_rates = lambda _unit_file_names: {
        0: {'rec_sessions': ['s1'], 'probe_sn': 'P', 'hs_sn': 'H', 'median_fr': 1.5},
        1: {'rec_sessions': [], 'probe_sn': 'P', 'hs_sn': 'H', 'median_fr': np.nan},
    }

    catalog_path = tmp_path / 'unit_catalog.csv'
    session_catalog = extractor.build_session_catalog(catalog_path=catalog_path, write=True)
    assert list(session_catalog.columns) == CATALOG_COLUMNS
    assert session_catalog.shape[0] == 1                      # unit 1 skipped
    assert session_catalog.loc[0, 'unit_id'] == 'imec0_cl0000_ch005_good'
    assert extractor.catalog_path == catalog_path

    # Idempotent re-merge: the same session's rows are replaced, not duplicated.
    extractor.build_session_catalog(catalog_path=catalog_path, write=True)
    on_disk = pd.read_csv(catalog_path)
    assert (on_disk['unit_id'] == 'imec0_cl0000_ch005_good').sum() == 1
