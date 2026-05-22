"""
@author: bartulem
Tests for ``usv_playpen.analyses.neuropixels.histology_ibl_alignment_export``.

Two tiers of checks:

1. Self-contained tests that exercise the pure-numpy helpers
   (:func:`ccf_apdvml_to_xyz_mlapdv_um`, :func:`parse_imro_table`,
   :func:`read_ap_meta`, :func:`sample_to_volts_ap`) without touching the
   network or any external atlas. These run anywhere the package can be
   imported.

2. Cross-checks against ``iblatlas`` that guard against silent drift of
   the Allen CCF bregma landmark or the apdvml→mlapdv affine. The
   runtime module deliberately does **not** depend on ``iblatlas`` — see
   the rationale in
   ``src/usv_playpen/analyses/neuropixels/histology_ibl_alignment_export.py``
   for why the constants are pinned inline — but if the upstream atlas
   ever rebases on a future CCFv4 or recalibrates the bregma offset,
   these tests are how we find out about it. The tests are skipped when
   ``iblatlas`` is not installed; install it via the ``test`` dependency
   group (``pip install -e .[test]`` or ``uv sync --group test``) to
   enable them.

The cross-checks use ``AllenAtlas(mock=True)`` so the test suite never
downloads the ~300 MB Allen NRRD volumes; ``mock=True`` still installs
the ``BrainCoordinates`` affine that ``ccf2xyz`` relies on.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_playpen.analyses.neuropixels.histology_ibl_alignment_export import (
    ALLEN_BREGMA_MLAPDV_UM,
    NP2_PROBE_TYPES,
    ccf_apdvml_to_xyz_mlapdv_um,
    parse_imro_table,
    read_ap_meta,
    sample_to_volts_ap,
)


def test_ccf_bregma_maps_to_origin():
    """
    Description
    -----------
    The Allen CCF bregma landmark, expressed in CCF apdvml voxel-origin
    micrometres, must map to the origin of IBL mlapdv-space (bregma is
    by construction at ``[0, 0, 0]`` mlapdv). Verifies the column
    permutation, translation and axis flips all line up.
    """
    apdvml = ALLEN_BREGMA_MLAPDV_UM[[1, 2, 0]].reshape(1, 3)
    out = ccf_apdvml_to_xyz_mlapdv_um(apdvml)
    np.testing.assert_allclose(out, [[0.0, 0.0, 0.0]], atol=0)


def test_ccf_axis_flips_are_correct():
    """
    Description
    -----------
    Moving 100 µm posterior in CCF AP should produce an mlapdv AP of
    -100 µm (AP is negated under the apdvml→mlapdv flip). Moving 100 µm
    deeper in CCF DV should produce an mlapdv DV of -100 µm. Moving
    100 µm lateral-right in CCF ML should produce an mlapdv ML of
    +100 µm (no flip on the ML axis).
    """
    posterior = np.array([ALLEN_BREGMA_MLAPDV_UM[1] + 100,
                          ALLEN_BREGMA_MLAPDV_UM[2],
                          ALLEN_BREGMA_MLAPDV_UM[0]]).reshape(1, 3)
    deeper = np.array([ALLEN_BREGMA_MLAPDV_UM[1],
                       ALLEN_BREGMA_MLAPDV_UM[2] + 100,
                       ALLEN_BREGMA_MLAPDV_UM[0]]).reshape(1, 3)
    lateral_right = np.array([ALLEN_BREGMA_MLAPDV_UM[1],
                              ALLEN_BREGMA_MLAPDV_UM[2],
                              ALLEN_BREGMA_MLAPDV_UM[0] + 100]).reshape(1, 3)
    np.testing.assert_allclose(ccf_apdvml_to_xyz_mlapdv_um(posterior),
                               [[0.0, -100.0, 0.0]], atol=0)
    np.testing.assert_allclose(ccf_apdvml_to_xyz_mlapdv_um(deeper),
                               [[0.0, 0.0, -100.0]], atol=0)
    np.testing.assert_allclose(ccf_apdvml_to_xyz_mlapdv_um(lateral_right),
                               [[100.0, 0.0, 0.0]], atol=0)


def test_ccf_broadcasts_over_leading_dims():
    """
    Description
    -----------
    The transform should broadcast over any leading shape. A
    ``(2, 3, 3)`` input must yield a ``(2, 3, 3)`` output with the same
    per-point conversion applied independently.
    """
    pts = np.random.default_rng(0).uniform(0, 10000, size=(2, 3, 3))
    out = ccf_apdvml_to_xyz_mlapdv_um(pts)
    assert out.shape == pts.shape
    flat_in = pts.reshape(-1, 3)
    flat_out = ccf_apdvml_to_xyz_mlapdv_um(flat_in)
    np.testing.assert_allclose(out.reshape(-1, 3), flat_out)


def test_parse_imro_table_np2_multishank_rows():
    """
    Description
    -----------
    A canonical NP2.0 four-shank IMRO header
    (``(probe_type, n_channels)(chan shank bank refid elecid)...``)
    should parse to a list whose first row is the
    ``[probe_type, n_channels]`` header and subsequent rows are
    5-integer channel descriptors with the shank index in column 1.
    """
    raw = "(2013,384)(0 3 1 0 576)(1 3 1 0 577)(2 2 1 0 578)"
    parsed = parse_imro_table(raw)
    assert parsed[0] == [2013, 384]
    assert parsed[1] == [0, 3, 1, 0, 576]
    assert parsed[2] == [1, 3, 1, 0, 577]
    assert parsed[3] == [2, 2, 1, 0, 578]


def test_parse_imro_table_snsgeommap_colon_format():
    """
    Description
    -----------
    The SpikeGLX ``snsGeomMap`` field uses colon-delimited 4-tuples
    inside each parenthesised group; the parser should pick ``:`` as
    the delimiter and return the four integer fields per row.
    """
    raw = "(0,0)(0:27:0:1)(0:59:0:1)"
    parsed = parse_imro_table(raw)
    assert parsed == [[0, 0], [0, 27, 0, 1], [0, 59, 0, 1]]


def test_parse_imro_table_empty_string_returns_empty_list():
    """
    Description
    -----------
    A falsy ``data_string`` short-circuits before the regex pass and
    returns an empty list — the documented behaviour of the parser.
    """
    assert parse_imro_table("") == []


def test_read_ap_meta_strips_tildes_and_parses_canonical_keys(tmp_path):
    """
    Description
    -----------
    SpikeGLX prefixes multi-line / structured metadata keys with ``~``
    (e.g. ``~imroTbl``, ``~snsGeomMap``). :func:`read_ap_meta` must
    drop those tildes so callers can use the canonical IBL/spikeglx
    key names (``meta['imroTbl']`` rather than ``meta['~imroTbl']``).
    Surrounding whitespace must be stripped and lines without ``=``
    must be skipped silently.
    """
    meta_path = tmp_path / "fake.ap.meta"
    meta_path.write_text(
        "imSampRate=30000.0\n"
        "imAiRangeMax=0.62\n"
        "imMaxInt=2048\n"
        "imDatPrb_type=2013\n"
        "~imroTbl=(2013,384)(0 3 1 0 576)\n"
        "garbage line without equals\n"
        "  ~snsGeomMap  =  (0,0)(0:27:0:1)  \n",
        encoding="utf-8",
    )
    meta = read_ap_meta(meta_path)
    assert meta["imSampRate"] == "30000.0"
    assert meta["imAiRangeMax"] == "0.62"
    assert meta["imMaxInt"] == "2048"
    assert meta["imDatPrb_type"] == "2013"
    assert meta["imroTbl"].startswith("(2013,384)")
    assert meta["snsGeomMap"].startswith("(0,0)")
    assert "~imroTbl" not in meta
    assert "~snsGeomMap" not in meta


def test_sample_to_volts_ap_np2_known_constants():
    """
    Description
    -----------
    For a Neuropixels 2.0 probe (``imDatPrb_type=2013``) with
    ``imAiRangeMax=0.62`` and ``imMaxInt=2048`` the int16→volt scaling
    must reduce to ``0.62 / 2048 / 80`` exactly. This is the value used
    by the IBL pipeline for the project's reference session.
    """
    meta = {
        "imDatPrb_type": "2013",
        "imAiRangeMax": "0.62",
        "imMaxInt": "2048",
    }
    np.testing.assert_allclose(sample_to_volts_ap(meta), 0.62 / 2048 / 80)


@pytest.mark.parametrize("bad_probe_type", [0, 1020, 1100, 1300, 9999])
def test_sample_to_volts_ap_raises_for_non_np2(bad_probe_type):
    """
    Description
    -----------
    The module supports only Neuropixels 2.0 probes (see
    :data:`NP2_PROBE_TYPES`). Calling
    :func:`sample_to_volts_ap` with any other probe type — including
    Neuropixels 1.0 (``imDatPrb_type=0``) — must raise
    ``NotImplementedError`` so a downstream caller cannot silently
    apply the wrong gain formula.
    """
    meta = {
        "imDatPrb_type": str(bad_probe_type),
        "imAiRangeMax": "0.62",
        "imMaxInt": "2048",
    }
    assert bad_probe_type not in NP2_PROBE_TYPES
    with pytest.raises(NotImplementedError):
        sample_to_volts_ap(meta)


# ----------------------------------------------------------------------
# Cross-checks against iblatlas — skipped when the package is missing.
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def iblatlas_module():
    """
    Description
    -----------
    Lazy import of ``iblatlas.atlas`` shared across the iblatlas
    cross-check tests in this module. When the package is not
    installed, the entire fixture (and therefore every dependent test)
    is skipped with a clear message; the rest of the suite still runs.
    """
    return pytest.importorskip(
        "iblatlas.atlas",
        reason="iblatlas not installed; install via the 'test' dependency "
               "group to enable Allen CCF cross-checks.",
    )


@pytest.fixture(scope="module")
def mock_allen_atlas(iblatlas_module):
    """
    Description
    -----------
    A mocked :class:`iblatlas.atlas.AllenAtlas` instantiated with
    ``mock=True`` so no NRRD volumes are downloaded. The
    ``BrainCoordinates`` affine — which is the only part of
    ``AllenAtlas`` that :meth:`ccf2xyz` consults — is fully initialised
    in mock mode, so the cross-checks below exercise the same affine
    that runs in production usage of ``iblatlas``.
    """
    return iblatlas_module.AllenAtlas(res_um=25, mock=True)


def test_bregma_constant_matches_iblatlas(iblatlas_module):
    """
    Description
    -----------
    The pinned :data:`ALLEN_BREGMA_MLAPDV_UM` constant in the runtime
    module must exactly equal the bregma landmark in
    ``iblatlas.atlas.ALLEN_CCF_LANDMARKS_MLAPDV_UM``. If iblatlas ever
    rebases on a future CCFv4 or recalibrates the landmark, this test
    fails loudly and prompts a manual update of the pinned constant.
    """
    np.testing.assert_array_equal(
        ALLEN_BREGMA_MLAPDV_UM,
        iblatlas_module.ALLEN_CCF_LANDMARKS_MLAPDV_UM["bregma"].astype(np.float64),
    )


def test_ccf_apdvml_to_xyz_mlapdv_um_matches_iblatlas_on_random_batch(mock_allen_atlas):
    """
    Description
    -----------
    On a 1024-point pseudorandom batch drawn from the interior of the
    Allen CCF 25 µm volume, the inline affine
    :func:`ccf_apdvml_to_xyz_mlapdv_um` must agree with
    ``AllenAtlas(25).ccf2xyz(..., ccf_order='apdvml') * 1e6`` to
    floating-point precision (1e-6 µm absolute, 1e-9 relative). If
    iblatlas changes the affine, the implementation, or the resolution
    convention, this assertion fails before any downstream IBL ALF
    export silently disagrees with the reference pipeline.
    """
    rng = np.random.default_rng(20250912)
    points_apdvml_um = rng.uniform(
        low=[0.0, 0.0, 0.0],
        high=[13200.0, 8000.0, 11400.0],
        size=(1024, 3),
    )

    ours = ccf_apdvml_to_xyz_mlapdv_um(points_apdvml_um)
    theirs = mock_allen_atlas.ccf2xyz(points_apdvml_um, ccf_order="apdvml") * 1e6

    np.testing.assert_allclose(ours, theirs, atol=1e-6, rtol=1e-9)
