"""
@author: bartulem
Tests for ``usv_playpen.analyses.npx_spike_quality_metrics``.

The orchestrator is mostly integration glue over real session data, so
these tests cover the pure-logic pieces that are both self-contained and
historically bug-prone: the metric-name / catalog-column bookkeeping and
the per-shank channel ordering.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from usv_playpen.analyses.npx_spike_quality_metrics import (
    SpikeQualityMetricsExtractor,
    SPIKE_TRAIN_METRIC_NAMES,
    RECORDING_DEPENDENT_METRIC_NAMES,
    QM_PARAMS,
    CATALOG_COLUMNS,
)


def _new_extractor():
    """Build a bare extractor instance, bypassing ``__init__`` and its disk I/O."""
    return SpikeQualityMetricsExtractor.__new__(SpikeQualityMetricsExtractor)


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
    from spikeinterface.metrics.quality.quality_metrics import get_quality_metric_list

    spike_train = set(SPIKE_TRAIN_METRIC_NAMES)
    recording_dependent = set(RECORDING_DEPENDENT_METRIC_NAMES)
    valid_metric_names = set(get_quality_metric_list())

    assert spike_train.isdisjoint(recording_dependent)
    assert (spike_train | recording_dependent) <= valid_metric_names
    assert set(QM_PARAMS) <= (spike_train | recording_dependent)


def test_catalog_has_47_unique_columns():
    """
    Description
    -----------
    The per-session catalog schema must have exactly 47 columns with no
    duplicates — the row builder in ``build_session_catalog`` appends 47
    values positionally, so a schema drift would silently misalign every
    column.
    """
    assert len(CATALOG_COLUMNS) == 47
    assert len(set(CATALOG_COLUMNS)) == 47


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
