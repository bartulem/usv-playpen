"""
@author: bartulem

Tests for processing/qlvm_categories: the content-ridge category build (ranks,
fields, content change, periodic basins, floor-first joins, session-bootstrap
consensus) and the assignment of a session's calls from its torus coordinates.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pls
import pytest
from click.testing import CliRunner

from usv_playpen import os_utils
from usv_playpen.processing import qlvm_categories as qc

# A small grid and few resamples keep the synthetic builds fast.
_CFG = {
    "properties": ["p_a", "p_b"],
    "grid_resolution": 40,
    "field_sigma": 2.0,
    "change_span": 2,
    "marker_sigma": 1.0,
    "marker_distance": 2,
    "size_floor": 0.05,
    "n_categories": 2,
    "n_bootstraps": 6,
    "bootstrap_seed": 3,
    "uncertain_agreement": 0.6,
    "category_order": [],
    "category_descriptions": ["low", "high"],
    "n_jobs": 1,
}


def _corpus(tmp_path, n_sessions=8, calls_per_session=150, seed=0):
    """Write a synthetic corpus: calls spread uniformly over the torus, whose two
    properties jump between the left (x < 0.5) and the right half, so the content
    change has two ridges (x = 0 and x = 0.5) and two categories. Returns the
    positions and properties table paths."""
    rng = np.random.default_rng(seed)
    n_calls = n_sessions * calls_per_session
    x, y = rng.random(n_calls), rng.random(n_calls)
    right = x >= 0.5
    spec_id = [f"2023010{session % 10}_1{session:05d}_{row}" for session in range(n_sessions) for row in range(calls_per_session)]
    session_id = [spec.rsplit("_", 1)[0] for spec in spec_id]
    positions = tmp_path / "positions.csv"
    properties = tmp_path / "properties.csv"
    pls.DataFrame({"spec_id": spec_id, "x": x, "y": y}).write_csv(positions)
    pls.DataFrame({
        "spec_id": spec_id, "session_id": session_id,
        "p_a": np.where(right, 10.0, 1.0) + rng.normal(0, 0.5, n_calls),
        "p_b": np.where(right, -5.0, 5.0) + rng.normal(0, 0.5, n_calls),
    }).write_csv(properties)
    return positions, properties


def _build(tmp_path, cfg=None, name="categories"):
    positions, properties = _corpus(tmp_path)
    out_dir = tmp_path / name
    qc.QLVMCategoryBuilder(
        positions_file=str(positions), properties_file=str(properties), output_directory=str(out_dir),
        input_parameter_dict={"build_qlvm_categories": _CFG if cfg is None else cfg}, message_output=lambda *_a, **_kw: None,
    ).build()
    return out_dir


@pytest.fixture(autouse=True)
def _no_wait(mocker):
    """The build and the assignment pause one second for the GUI; tests skip it."""
    mocker.patch("usv_playpen.processing.qlvm_categories.smart_wait")


def test_property_ranks_are_centred_average_ranks():
    """Ranks are (average rank - 0.5) / n per column, ties sharing their mean rank,
    and a non-finite value stops the ranking."""
    values = np.array([[3.0, 1.0], [1.0, 1.0], [2.0, 5.0], [2.0, 1.0]])
    ranks = qc.property_ranks(values)
    np.testing.assert_allclose(ranks[:, 0], (np.array([4.0, 1.0, 2.5, 2.5]) - 0.5) / 4)
    np.testing.assert_allclose(ranks[:, 1], (np.array([2.0, 2.0, 4.0, 2.0]) - 0.5) / 4)
    with pytest.raises(ValueError, match="not finite"):
        qc.property_ranks(np.array([[1.0], [np.nan]]))


def test_content_fields_and_change_on_a_step():
    """A constant field has no content change; a step in x between columns 9/10 and
    19/20 of a 20-pixel torus gives change only within span of the two edges, equal to
    the step height, and the field of weighted calls is their weighted mean."""
    resolution = 20
    rows = np.repeat(np.arange(resolution), resolution)
    cols = np.tile(np.arange(resolution), resolution)
    flat = qc.content_fields(rows, cols, np.ones((rows.size, 1)), np.ones(rows.size), 1.5, resolution)
    np.testing.assert_allclose(flat, 1.0)
    assert np.allclose(qc.content_change(flat, 2), 0.0)
    step = np.zeros((1, resolution, resolution))
    step[0, :, 10:] = 1.0
    change = qc.content_change(step, 2)
    np.testing.assert_allclose(change[:, 10], 1.0)
    np.testing.assert_allclose(change[:, 5], 0.0)
    np.testing.assert_allclose(change[:, 0], 1.0)
    weighted = qc.content_fields(np.array([0, 0]), np.array([0, 0]), np.array([[0.0], [1.0]]), np.array([1.0, 3.0]), 1.0, resolution)
    assert weighted[0, 0, 0] == pytest.approx(0.75)


def test_join_basins_joins_small_regions_first_then_the_weakest_ridge():
    """Four vertical stripe basins on a torus: with a floor, the smallest region joins
    its lower-ridge neighbour first even when another ridge is lower; without one, the
    lowest ridge joins first; an unreachable floor gives None."""
    resolution = 8
    basins = np.repeat(np.repeat(np.arange(1, 5)[None, :], 2, axis=1), resolution, axis=0)
    change = np.zeros((resolution, resolution))
    # Ridge heights at the boundaries 1|2, 2|3, 3|4 and 4|1 (periodic): 3, 1, 2, 5.
    for column, height in ((2, 3.0), (4, 1.0), (6, 2.0), (0, 5.0)):
        change[:, column] = height
    sizes = np.array([0.0, 40.0, 40.0, 40.0, 2.0])
    lookup = qc.join_basins(basins, change, sizes, 0.05, 3)
    # Basin 4 is under the floor: it joins basin 3 (ridge 2 < ridge 5), although 2|3 is lower.
    assert lookup[4] == lookup[3]
    assert len({lookup[1], lookup[2], lookup[3]}) == 3
    lookup = qc.join_basins(basins, change, np.array([0.0, 40.0, 40.0, 40.0, 40.0]), 0.0, 3)
    assert lookup[2] == lookup[3]
    assert len({lookup[1], lookup[2], lookup[4]}) == 3
    # A floor of 34 % (41.5 of 122 calls) leaves no 3 regions that all hold it.
    assert qc.join_basins(basins, change, sizes, 0.34, 3) is None


def test_align_to_reference_maps_by_overlap():
    """A permuted copy of the reference is mapped back onto it."""
    reference = np.repeat(np.arange(3)[None, :], 6, axis=0).repeat(2, axis=1)
    permuted = np.array([2, 0, 1])[reference]
    np.testing.assert_array_equal(qc.align_to_reference(reference, permuted, 3), reference)


def test_session_bootstrap_weights_count_the_draws():
    """Each call is weighted by how often its session was drawn by the seeded draw."""
    sessions = np.array(["a", "a", "b", "c"])
    pool = np.array(["a", "b", "c"])
    drawn = np.random.default_rng(5).choice(pool, size=3, replace=True)
    expected = np.array([float((drawn == session).sum()) for session in sessions])
    np.testing.assert_array_equal(qc.session_bootstrap_weights(sessions, pool, 5), expected)


def test_build_writes_a_category_directory_split_along_the_ridges(tmp_path):
    """The two halves of the synthetic torus become the two categories (R-1 the
    larger), the files hold consistent grids, call labels and nomenclature, and the
    build does not depend on the number of workers."""
    out_dir = _build(tmp_path)
    with np.load(out_dir / qc.CATEGORY_GRIDS_NAME) as grids:
        label_grid = grids["label_grid"]
        agreement = grids["agreement"]
        assert set(np.unique(label_grid).tolist()) == {1, 2}
        assert grids["reference_grid"].shape == (40, 40)
    assert np.all((agreement > 0) & (agreement <= 1))
    left, right = label_grid[:, 4:16], label_grid[:, 24:36]
    assert len(np.unique(left)) == 1
    assert len(np.unique(right)) == 1
    assert left[0, 0] != right[0, 0]
    nomenclature = json.loads((out_dir / qc.CATEGORY_NOMENCLATURE_NAME).read_text())
    assert [category["name"] for category in nomenclature["categories"]] == ["R-1", "R-2"]
    assert [category["description"] for category in nomenclature["categories"]] == ["low", "high"]
    assert nomenclature["categories"][0]["n_calls"] >= nomenclature["categories"][1]["n_calls"]
    assert nomenclature["n_votes"] == 6
    calls = pls.read_csv(out_dir / qc.CATEGORY_CALL_LABELS_NAME)
    assert calls.height == 1200
    assert calls["category"].value_counts().sort("category")["count"].to_list() == [
        category["n_calls"] for category in nomenclature["categories"]
    ]
    bundle = qc.load_category_bundle(str(out_dir))
    assigned = qc.assign_categories(calls.select(["x", "y"]).to_numpy(), bundle)
    np.testing.assert_array_equal(assigned["category"], calls["category"].to_numpy())
    np.testing.assert_array_equal(assigned["uncertain"], calls["uncertain"].to_numpy())
    parallel_dir = _build(tmp_path, {**_CFG, "n_jobs": 2}, "parallel")
    with np.load(out_dir / qc.CATEGORY_GRIDS_NAME) as first, np.load(parallel_dir / qc.CATEGORY_GRIDS_NAME) as second:
        np.testing.assert_array_equal(first["label_grid"], second["label_grid"])
        np.testing.assert_array_equal(first["agreement"], second["agreement"])


def test_build_refuses_bad_settings_and_unplaced_calls(tmp_path):
    """Invalid settings are listed together; a corpus call without a position stops the build."""
    with pytest.raises(ValueError, match=r"size_floor must lie.*\n.*category_descriptions"):
        _build(tmp_path, {**_CFG, "size_floor": 0.6, "category_descriptions": ["one"]})
    positions, properties = _corpus(tmp_path)
    pls.read_csv(positions).head(100).write_csv(positions)
    with pytest.raises(ValueError, match="have no position"):
        qc.QLVMCategoryBuilder(
            positions_file=str(positions), properties_file=str(properties), output_directory=str(tmp_path / "x"),
            input_parameter_dict={"build_qlvm_categories": _CFG}, message_output=lambda *_a, **_kw: None,
        ).build()


def test_assign_categories_follows_the_pixel_rule_and_threshold():
    """Each coordinate takes its pixel's category and agreement; below the threshold
    it is uncertain; a NaN coordinate is not placed."""
    label_grid = np.array([[1, 2], [3, 4]])
    agreement = np.array([[1.0, 0.5], [0.6, 0.59]])
    bundle = {"label_grid": label_grid, "agreement": agreement, "uncertain_agreement": 0.6}
    labels = qc.assign_categories(np.array([[0.1, 0.1], [0.7, 0.2], [0.2, 0.9], [0.9, 0.9], [np.nan, 0.5]]), bundle)
    assert labels["category"].tolist() == [1, 2, 3, 4, 0]
    assert labels["uncertain"].tolist() == [False, True, False, True, False]
    assert labels["placed"].tolist() == [True, True, True, True, False]
    assert np.isnan(labels["agreement"][4])


def test_assigner_merges_category_columns_into_the_summary(tmp_path):
    """The session's calls with P1 / P2 get P_category and nothing else: calls without
    coordinates get a null, an earlier P_category is replaced, earlier
    P_category_agreement / P_category_uncertain columns are removed (the agreement stays
    computable from the bundle), every other column is kept, and the summary comes out in
    canonical order."""
    out_dir = _build(tmp_path)
    root = tmp_path / "20240101_100000"
    (root / "audio").mkdir(parents=True)
    summary_path = root / "audio" / "20240101_100000_usv_summary.csv"
    pls.DataFrame({
        "usv_id": ["0000", "0001", "0002"],
        "start": [0.1, 0.2, 0.3],
        "qlvm_m1": [0.1, None, 0.9],
        "qlvm_m2": [0.5, None, 0.5],
        "qlvm_m_category": [9, 9, 9],
    }).write_csv(summary_path)
    qc.QLVMCategoryAssigner(
        root_directory=str(root),
        input_parameter_dict={"assign_qlvm_categories": {"category_directory": str(out_dir), "coordinate_prefix": "qlvm_m"}},
        message_output=lambda *_a, **_kw: None,
    ).assign_and_merge()
    df = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    assert df.columns == ["usv_id", "start", "qlvm_m1", "qlvm_m2", "qlvm_m_category"]
    bundle = qc.load_category_bundle(str(out_dir))
    expected = qc.assign_categories(np.array([[0.1, 0.5], [0.9, 0.5]]), bundle)
    assert df["qlvm_m_category"].to_list() == [int(expected["category"][0]), None, int(expected["category"][1])]

    # The regular map's prefix, with the confidence columns an earlier version wrote and
    # columns out of canonical order: only qlvm_category is written, in canonical place.
    pls.DataFrame({
        "usv_id": ["0000", "0001", "0002"],
        "qlvm1": [0.1, None, 0.9],
        "qlvm2": [0.5, None, 0.5],
        "qlvm_category_agreement": [0.1, 0.1, 0.1],
        "qlvm_category_uncertain": [True, True, True],
        "start": [0.1, 0.2, 0.3],
        "qlvm_squeak1": [None, 0.3, None],
    }).write_csv(summary_path)
    qc.QLVMCategoryAssigner(
        root_directory=str(root),
        input_parameter_dict={"assign_qlvm_categories": {"category_directory": str(out_dir), "coordinate_prefix": "qlvm"}},
        message_output=lambda *_a, **_kw: None,
    ).assign_and_merge()
    df = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    assert df.columns == ["usv_id", "start", "qlvm1", "qlvm2", "qlvm_category", "qlvm_squeak1"]
    assert df["qlvm_category"].to_list() == [int(expected["category"][0]), None, int(expected["category"][1])]
    with pytest.raises(ValueError, match="needs a category_directory"):
        qc.QLVMCategoryAssigner(
            root_directory=str(root),
            input_parameter_dict={"assign_qlvm_categories": {"category_directory": str(out_dir), "coordinate_prefix": ""}},
            message_output=lambda *_a, **_kw: None,
        ).assign_and_merge()
    with pytest.raises(ValueError, match=r"no \['qlvm_z1', 'qlvm_z2'\] column"):
        qc.QLVMCategoryAssigner(
            root_directory=str(root),
            input_parameter_dict={"assign_qlvm_categories": {"category_directory": str(out_dir), "coordinate_prefix": "qlvm_z"}},
            message_output=lambda *_a, **_kw: None,
        ).assign_and_merge()


def test_assigner_defaults_to_the_production_bundle_on_the_regular_map(tmp_path, monkeypatch):
    """With a spectrograms_root and an empty category_directory / coordinate_prefix, the
    assigner labels with os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY on the regular map (qlvm):
    the bundle every QLVM figure loads, so the summaries' qlvm_category and the drawn
    boundaries are one partition."""
    out_dir = _build(tmp_path)
    monkeypatch.setattr(os_utils, "QLVM_CATEGORY_BUNDLE_DIRECTORY", str(out_dir))
    root = tmp_path / "20240101_100000"
    (root / "audio").mkdir(parents=True)
    summary_path = root / "audio" / "20240101_100000_usv_summary.csv"
    pls.DataFrame({"usv_id": ["0000", "0001"], "start": [0.1, 0.2], "qlvm1": [0.1, 0.9], "qlvm2": [0.5, 0.5]}).write_csv(summary_path)
    settings = {
        "spectrograms_root": "/mnt/falkner/Bartul/spectrograms",
        "generate_masks": {"sam2_model_dir": "x", "sam2_model_path": "x", "yolo_weights": "x"},
        "infer_qlvm_latents": {"model_cells": {"qlvm": "/c"}},
        "infer_qlvm_squeak_latents": {"model_cell_directory": "/s"},
        "detect_usv_squeaks": {"squeak_model_path": "x"},
        "detect_usv_noise": {"noise_model_path": "x"},
        "assign_qlvm_categories": {"category_directory": "", "coordinate_prefix": ""},
    }
    qc.QLVMCategoryAssigner(
        root_directory=str(root), input_parameter_dict=settings, message_output=lambda *_a, **_kw: None,
    ).assign_and_merge()
    assert settings["assign_qlvm_categories"] == {"category_directory": str(out_dir), "coordinate_prefix": "qlvm"}
    expected = qc.assign_categories(np.array([[0.1, 0.5], [0.9, 0.5]]), qc.load_category_bundle(str(out_dir)))
    df = pls.read_csv(summary_path, schema_overrides={"usv_id": pls.String})
    assert df["qlvm_category"].to_list() == expected["category"].tolist()


def test_category_clis_route_their_settings(tmp_path, mocker):
    """build-qlvm-categories splits --properties into a list and scopes its flags to its
    block; assign-qlvm-categories passes the session and its block on."""
    runner = CliRunner()
    builder = mocker.patch("usv_playpen.processing.qlvm_categories.QLVMCategoryBuilder")
    settings = mocker.patch("usv_playpen.processing.qlvm_categories.modify_settings_json_for_cli",
                            return_value={"build_qlvm_categories": {}})
    positions, properties = _corpus(tmp_path)
    result = runner.invoke(qc.build_qlvm_categories_cli, [
        "--positions-file", str(positions), "--properties-file", str(properties), "--output-directory", str(tmp_path / "o"),
        "--properties", "p_a, p_b", "--n-categories", "3",
    ])
    assert result.exit_code == 0, result.output
    assert settings.call_args.kwargs["block"] == "build_qlvm_categories"
    assert settings.call_args.kwargs["ctx"].params["properties"] == ["p_a", "p_b"]
    assert sorted(settings.call_args.kwargs["provided_params"]) == ["n_categories", "properties"]
    builder.return_value.build.assert_called_once()
    assigner = mocker.patch("usv_playpen.processing.qlvm_categories.QLVMCategoryAssigner")
    settings.return_value = {"assign_qlvm_categories": {}}
    result = runner.invoke(qc.assign_qlvm_categories_cli, [
        "--root-directory", str(tmp_path), "--category-directory", str(tmp_path), "--coordinate-prefix", "qlvm_m",
    ])
    assert result.exit_code == 0, result.output
    assert settings.call_args.kwargs["block"] == "assign_qlvm_categories"
    assert assigner.call_args.kwargs["root_directory"] == str(tmp_path)
    assigner.return_value.assign_and_merge.assert_called_once()


def test_shipped_settings_hold_the_category_blocks():
    """The shipped processing settings carry every key the two classes read."""
    settings = json.loads((qc.pathlib.Path(qc.__file__).parent.parent / "_parameter_settings" / "processing_settings.json").read_text())
    assert set(settings["build_qlvm_categories"]) == set(_CFG)
    assert set(settings["assign_qlvm_categories"]) == {"category_directory", "coordinate_prefix"}


def test_apply_category_order_renumbers_by_size_rank():
    """
    Description
    -----------
    ``category_order`` names, for each output category, the size rank it takes
    (1 = most calls): ``[1, 4, 2, 3]`` keeps the largest first, puts the smallest
    second and shifts the second and third largest to third and fourth. An empty
    order keeps the size order, and a non-permutation raises.

    Returns
    -------
    None
    """

    by_size = np.array([[0, 1], [2, 3]], dtype=np.int16)
    ordered = qc.apply_category_order(by_size, [1, 4, 2, 3], 4)
    assert ordered.tolist() == [[0, 2], [3, 1]]
    assert ordered.dtype == by_size.dtype
    assert qc.apply_category_order(by_size, [], 4).tolist() == by_size.tolist()
    with pytest.raises(ValueError, match="permutation"):
        qc.apply_category_order(by_size, [1, 1, 2, 3], 4)


def test_build_numbers_categories_by_category_order(tmp_path):
    """
    Description
    -----------
    A build with ``category_order`` reversed numbers the categories in that order:
    the category that is first by call count becomes the last, with its call
    count, and the settings check rejects an order that is not a permutation.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        Per-test temporary directory.

    Returns
    -------
    None
    """

    size_order = _build(tmp_path, {**_CFG, "category_order": []}, "size")
    reversed_order = _build(tmp_path, {**_CFG, "category_order": [2, 1]}, "reversed")
    sizes = [category["n_calls"] for category in json.loads((size_order / qc.CATEGORY_NOMENCLATURE_NAME).read_text())["categories"]]
    reversed_sizes = [category["n_calls"] for category in json.loads((reversed_order / qc.CATEGORY_NOMENCLATURE_NAME).read_text())["categories"]]
    assert reversed_sizes == sizes[::-1]
    size_grid = np.load(size_order / qc.CATEGORY_GRIDS_NAME)["label_grid"]
    reversed_grid = np.load(reversed_order / qc.CATEGORY_GRIDS_NAME)["label_grid"]
    assert np.array_equal(reversed_grid, 3 - size_grid)
    with pytest.raises(ValueError, match="category_order must be empty or a permutation"):
        _build(tmp_path, {**_CFG, "category_order": [1, 1]}, "bad")
