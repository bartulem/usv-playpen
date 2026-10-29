"""
@author: bartulem
Unit tests for ``usv_playpen.visualizations.neural_modeling_plots``.

Coverage: the torus image orientation every map is drawn through (``grid_image``, ``grid_centres``),
the held-out category confusion rebuilt from stored per-fold rates and its refusal of a rebuild that does
not reproduce the stored per-USV gain, the confusion and category-scatter panels (on a synthetic
category map), the catalog lookup of brain areas, the settings-hash guard of the figure entry point, and
the registered command line.
"""

from __future__ import annotations

import copy
import json
import pickle

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from click.testing import CliRunner

from usv_playpen.neural_modeling.neural_category_decoding import (
    category_codes,
    category_context,
    category_gain,
)
from usv_playpen.neural_modeling.neural_vocal_decoding import torus_grid
from usv_playpen.visualizations import neural_modeling_plots
from usv_playpen.visualizations.neural_modeling_plots import (
    NEURAL_SETTINGS_PATH,
    category_confusion,
    draw_category_confusion,
    draw_category_scatter,
    grid_centres,
    grid_image,
    neural_modeling_figures_cli,
    plot_neural_modeling_units,
    unit_brain_areas,
)


def _shipped_settings() -> dict:
    """
    Description
    -----------
    A private copy of the shipped neural modeling settings.

    Parameters
    ----------
    None

    Returns
    -------
    settings (dict)
        The shipped settings, deep-copied.
    """

    with NEURAL_SETTINGS_PATH.open() as settings_file:
        return copy.deepcopy(json.load(settings_file))


def _category_unit(n_sessions: int = 3, per_session: int = 150, seed: int = 0) -> tuple[dict, dict, dict]:
    """
    Description
    -----------
    A synthetic unit whose spike count depends on the USV category, decoded by the real categorical
    decoder, in the shapes the figure reads: the assembled ``events``, the stored identity section, and
    the settings.

    Parameters
    ----------
    n_sessions (int)
        Number of sessions.
    per_session (int)
        USVs per session.
    seed (int)
        Random seed.

    Returns
    -------
    events (dict)
        ``category_labels``, ``counts``, ``session_index``, ``vocalization_start`` and ``positions``.
    section (dict)
        The identity section as the dispatcher stores it (``category_gain`` with detail, plus
        ``categories``).
    settings (dict)
        The shipped settings.
    """

    rng = np.random.default_rng(seed)
    n_events = n_sessions * per_session
    labels = rng.integers(1, 5, size=n_events).astype(np.float64)
    rates = np.array([1.0, 2.0, 4.0, 6.0])
    counts = rng.poisson(rates[(labels - 1).astype(int)]).astype(np.float64)
    session_index = np.repeat(np.arange(n_sessions), per_session)
    starts = np.tile(np.sort(rng.uniform(0.0, 1000.0, per_session)), n_sessions)
    positions = rng.uniform(0.0, 1.0, size=(n_events, 2))
    settings = _shipped_settings()
    codes, kept, values = category_codes(labels)
    folds = category_context(codes, session_index[kept], starts[kept],
                             int(settings["data_sufficiency"]["single_session_blocks"]))
    section = category_gain(counts[kept], codes, folds, settings["vocal_decoding"], with_detail=True)
    section["categories"] = values.tolist()
    events = {"category_labels": labels, "counts": counts, "session_index": session_index,
              "vocalization_start": starts, "positions": positions}
    return events, section, settings


class TestGridImage:
    """Every torus map is drawn with acoustic dimension 1 along x and dimension 2 along y."""

    def test_dimension_one_runs_along_columns(self):
        """A field equal to each point's dimension-1 coordinate must vary along columns, not rows."""
        grid, _cell = torus_grid(8, 1.0)
        image = grid_image(grid[:, 0], 8)
        assert np.allclose(image[0], grid_centres(8))
        assert np.allclose(image[:, 3], grid_centres(8)[3])

    def test_a_point_lands_where_its_coordinates_say(self):
        """A single grid point at (0.2, 0.8) must sit in image row of y = 0.8 and column of x = 0.2."""
        grid, _cell = torus_grid(10, 1.0)
        values = np.zeros(grid.shape[0])
        values[np.argmin(np.sum((grid - [0.25, 0.85]) ** 2, axis=1))] = 1.0
        row, column = np.unravel_index(np.argmax(grid_image(values, 10)), (10, 10))
        assert grid_centres(10)[column] == pytest.approx(0.25)
        assert grid_centres(10)[row] == pytest.approx(0.85)

    def test_centres_match_the_torus_grid(self):
        """The contour coordinates must be the torus grid's own cell centres."""
        grid, _cell = torus_grid(12, 1.0)
        assert np.allclose(np.unique(grid[:, 0]), grid_centres(12))


class TestCategoryConfusion:
    """The held-out confusion matrix rebuilt from the stored per-fold rates and amplitudes."""

    def test_rebuild_reproduces_the_tested_gain(self):
        """The count-weighted diagonal must equal the stored gain the permutation test scored."""
        events, section, settings = _category_unit()
        confusion = category_confusion("synthetic", events, section, settings)
        weights = confusion["row_counts"] / confusion["row_counts"].sum()
        assert float(np.sum(np.diag(confusion["matrix"]) * weights)) == pytest.approx(section["gain"], abs=1e-9)
        assert confusion["gain"] == pytest.approx(section["gain"], abs=1e-12)
        assert confusion["matrix"].shape == (4, 4)

    def test_a_tuned_unit_points_at_the_right_category(self):
        """For a unit whose rate rises with category, the extreme categories favour themselves."""
        events, section, settings = _category_unit()
        matrix = category_confusion("synthetic", events, section, settings)["matrix"]
        assert matrix[0, 0] > matrix[0, 3]
        assert matrix[3, 3] > matrix[3, 0]

    def test_a_rebuild_that_does_not_match_the_stored_gain_is_refused(self):
        """Stored rates that are not the run's own must stop the panel rather than draw it."""
        events, section, settings = _category_unit()
        tampered = copy.deepcopy(section)
        tampered["per_fold"][0]["rates"] = [rate * 1.5 for rate in tampered["per_fold"][0]["rates"]]
        with pytest.raises(ValueError, match="does not reproduce"):
            category_confusion("synthetic", events, tampered, settings)

    def test_panel_draws_every_cell(self):
        """The confusion panel must write one value per cell and label its colour bar."""
        events, section, settings = _category_unit()
        confusion = category_confusion("synthetic", events, section, settings)
        figure, axis = plt.subplots()
        draw_category_confusion(axis, confusion)
        assert len(axis.texts) == 16
        assert any("log-likelihood ratio" in inset.get_xlabel() for inset in axis.child_axes)
        plt.close(figure)


class TestCategoryScatter:
    """The USVs on the torus coloured by category, with the category map's borders."""

    def test_draws_borders_and_one_number_per_category(self, monkeypatch):
        """With a synthetic four-quadrant category map, the panel draws the borders and numbers 1-4."""
        resolution = 40
        axis_coordinates = (np.arange(resolution) + 0.5) / resolution
        label_grid = np.ones((resolution, resolution), dtype=np.int16)
        label_grid[:, resolution // 2:] += 1
        label_grid[resolution // 2:, :] += 2
        monkeypatch.setattr(neural_modeling_plots, "load_qlvm_category_bundle",
                            lambda: {"label_grid": label_grid, "axis": axis_coordinates})
        rng = np.random.default_rng(1)
        positions = rng.uniform(0.0, 1.0, size=(2000, 2))
        cells = np.floor(positions * resolution).astype(int)
        labels = label_grid[cells[:, 1], cells[:, 0]].astype(np.float64)
        figure, axis = plt.subplots()
        draw_category_scatter(axis, positions, labels, [1.0, 2.0, 3.0, 4.0], _shipped_settings())
        assert sorted(text.get_text() for text in axis.texts) == ["1", "2", "3", "4"]
        assert len(axis.collections) > 1
        plt.close(figure)


class TestEntryPoint:
    """The catalog lookup, the settings-hash guard and the command line."""

    def test_brain_areas_come_from_the_catalog(self, tmp_path):
        """A unit's area is read from the catalog row its uid is built from; an absent unit is an error."""
        catalog = tmp_path / "unit_catalog.csv"
        catalog.write_text("mouse_id,rec_date,unit_id,brain_area\n"
                           "181316_0,20250919,imec1_cl0401_ch220_good,PAG\n"
                           "181316_0,20250923,imec1_cl0499_ch322_good,VTA\n")
        areas = unit_brain_areas(str(catalog), ["181316_0_20250923_imec1_cl0499_ch322_good"])
        assert areas == {"181316_0_20250923_imec1_cl0499_ch322_good": "VTA"}
        with pytest.raises(ValueError, match="not in the catalog"):
            unit_brain_areas(str(catalog), ["181316_0_20250919_imec0_cl9999_ch000_good"])

    def test_a_result_written_under_other_settings_is_skipped(self, tmp_path, capsys):
        """A result file whose settings hash is not the run's must not be drawn."""
        uid = "181316_0_20250919_imec1_cl0401_ch220_good"
        catalog = tmp_path / "unit_catalog.csv"
        catalog.write_text(f"mouse_id,rec_date,unit_id,brain_area\n181316_0,20250919,{uid[18:]},PAG\n")
        settings = _shipped_settings()
        settings["data_roots"]["catalog_path"] = str(catalog)
        settings_path = tmp_path / "settings.json"
        settings_path.write_text(json.dumps(settings))
        results = tmp_path / "results"
        results.mkdir()
        with (results / f"{uid}.pkl").open("wb") as result_file:
            pickle.dump({"provenance": {"settings_sha256": "0" * 64}}, result_file)
        written = plot_neural_modeling_units(str(results), str(tmp_path / "figures"), str(tmp_path),
                                             None, str(settings_path))
        assert written == []
        assert "SKIPPED" in capsys.readouterr().out

    def test_command_line_is_registered(self):
        """The command must answer --help under its registered name."""
        result = CliRunner().invoke(neural_modeling_figures_cli, ["--help"])
        assert result.exit_code == 0
        assert "neural-modeling-figures" in result.output
