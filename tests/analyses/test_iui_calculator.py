"""
@author: bartulem
Mock-based tests for the InterUSVIntervalCalculator orchestration class.

The class drives a per-mode loop over compute_session_usv_intervals, optionally
invokes the mixture-model sweep + bootstrap LRT, and writes a single HDF5 archive. We
mock every heavy compute helper (compute_session_usv_intervals,
fit_mixture_model_sweep, bootstrap_lrt, write_ivi_h5) so the orchestration can be
exercised end-to-end against synthetic disk fixtures.

Coverage: settings validation and the skip branches of the save step; the archive
written with and without the mixture-model sweep (session-labelled LRT inputs, the
step-up selector); interval pools (fitted vs described-only, per-pool filters); the
tied peak ladder and serial-dependence fits run on fitted pools only; the ladder's
warm-started nulls; per-session interval computation (overlap segments excluded,
chains broken by another animal's vocalization, squeaks timed by their extent); the polars <-> HDF5 round trip of nullable columns; and the
shared USV I/O helpers (emitter-sex resolution, track ids, unique matching, loading
and filtering).
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import polars as pls
import pytest
from scipy.stats import invgauss

import usv_playpen.analyses.compute_inter_usv_interval_distributions as iui_mod
from usv_playpen.analyses._usv_io import (
    emitter_sex_expression,
    extract_animal_sexes,
    load_and_filter_usv_data,
    sex_track_ids,
    unique_match_or_none,
)
from usv_playpen.analyses.compute_inter_usv_interval_distributions import (
    InterUSVIntervalCalculator,
    build_em_initialisation,
    compute_session_usv_intervals,
    fit_mixture_model_sweep,
    fit_serial_dependence,
    fit_tied_peak_ladder_and_lrt,
    validate_interval_pools,
)
from usv_playpen.analyses.usv_interval_archive import _h5_to_polars, _polars_to_h5
from usv_playpen.yaml_utils import extract_animal_sexes as yaml_extract_animal_sexes

# The EM starting-point heuristics the analysis builds from the shipped interval settings,
# passed to the fit functions exactly as the analysis passes them.
with (Path(iui_mod.__file__).parent.parent / "_parameter_settings" / "analyses_settings.json").open() as _settings_file:
    _EM_INIT = build_em_initialisation(json.load(_settings_file)["compute_inter_usv_interval_distributions"])


def test_polars_h5_roundtrip_nullable_int_and_bool(tmp_path):
    """A nullable Int64 and a nullable Boolean column round-trip through
    ``_polars_to_h5`` / ``_h5_to_polars`` as Float64 with the null slot preserved as
    NaN: the writer promotes both to float64 and records "Float64" in the schema, so the
    reader rebuilds a matching float column instead of forcing NaN back to Int64 (which
    raises) or coercing a boolean ``None`` to ``False``. A plain float column is unchanged."""

    df = pls.DataFrame({
        "i": pls.Series([1, None, 3], dtype=pls.Int64),
        "b": pls.Series([True, None, False], dtype=pls.Boolean),
        "f": pls.Series([1.5, 2.5, 3.5], dtype=pls.Float64),
    })
    out = tmp_path / "roundtrip.h5"
    with h5py.File(str(out), "w") as f:
        _polars_to_h5(f, "t", df)
    with h5py.File(str(out), "r") as f:
        back = _h5_to_polars(f["t"])

    assert back["i"].dtype == pls.Float64
    assert back["b"].dtype == pls.Float64
    i_vals = back["i"].to_list()
    b_vals = back["b"].to_list()
    assert i_vals[0] == 1.0
    assert i_vals[2] == 3.0
    assert np.isnan(i_vals[1])
    assert b_vals[0] == 1.0
    assert b_vals[2] == 0.0
    assert np.isnan(b_vals[1])
    assert back["f"].to_list() == [1.5, 2.5, 3.5]


# Fixture: minimal valid input_parameter_dict


def _make_settings(tmp_path, fit_mixture_model=False, fit_tied_model=False, fit_serial_dependence=False):
    """Build the analyses_settings sub-block expected by the class, including the pool spec:
    the male's USVs (fitted) and the female's squeaks (described only)."""
    return {
        "compute_inter_usv_interval_distributions": {
            "session_lists": [],
            "output_directory": str(tmp_path / "out"),
            "exclude_noise_usvs": True,
            "fit_mixture_model": fit_mixture_model,
            "n_components_min": 1,
            "n_components_max": 3,
            "n_repeats": 2,
            "max_modes_reported": 5,
            "random_seed_base": 0,
            "cv_n_folds": 5,
            "cv_n_init": 2,
            "mixture_model_n_init": 3,
            "mixture_model_reg_covar": 1e-4,
            "tau": 0.5,
            "model_class": "gauss",
            "bootstrap_lrt_B": 2,
            "bootstrap_lrt_n_subsample": 50,
            "bootstrap_lrt_n_jobs": 1,
            "bootstrap_lrt_n_init": 1,
            "bootstrap_lrt_alpha": 0.05,
            "bootstrap_lrt_bonferroni": True,
            "interval_pools": [
                {"sex": "male", "vocal_class": "usv", "adjacency": "filtered", "fit": True},
                {"sex": "female", "vocal_class": "squeak", "adjacency": "strict", "fit": False},
            ],
            "min_intervals_for_fitting": 1,
            "fit_tied_model": fit_tied_model,
            "tied_n_background": 2,
            "tied_peak_grid": [1, 2, 3],
            "tied_n_init": 1,
            "tied_n_init_boot": 1,
            "design_bootstrap": 10,
            "design_bootstrap_iid_cap": 400,
            "em_max_iter": 300,
            "em_tol": 1e-5,
            "tied_em_max_iter": 500,
            "tied_em_tol": 1e-6,
            "split_delta": 0.5,
            "mode_grid_size_gauss": 4096,
            "mode_span_scales_gauss": 4.0,
            "mode_grid_size_t": 16384,
            "mode_span_scales_t": 6.0,
            "cv_t_n_init_offset": 2,
            "em_init_kmeans_n_init": 10,
            "em_init_start_nu": 10.0,
            "em_init_background_start_nu": 5.0,
            "em_init_min_start_weight": 0.001,
            "em_init_restart_quantile_range": [0.02, 0.98],
            "em_init_peak_variance_fraction": 0.05,
            "em_init_seeded_peak_quantile_range": [0.15, 0.9],
            "em_init_seeded_background_quantile_range": [0.4, 0.97],
            "em_init_random_peak_quantile_range": [0.05, 0.95],
            "em_init_random_background_quantile_range": [0.35, 0.98],
            "em_init_seed_histogram_bins": 512,
            "em_init_seed_smoothing_sigma": 3.0,
            "em_init_seed_smoothing_half_width": 8,
            "fit_serial_dependence": fit_serial_dependence,
            "serial_dependence_n_knots": 4,
            "serial_dependence_corner_widths": [0.05, 0.15],
            "serial_dependence_bootstrap": 5,
            "serial_dependence_level": 99.0,
            "serial_dependence_bend_bounds_ms": [30.0, 600.0],
            "serial_dependence_grid_percentiles": [1.0, 99.0],
            "serial_dependence_max_iter": 1000,
            "serial_dependence_n_grid": 200,
        }
    }


# __init__


def test_iui_calculator_init_rejects_unknown_kwargs():
    """Unknown keyword arguments (typos) raise TypeError instead of being
    silently stored, so a mistyped settings key is caught at construction."""
    with pytest.raises(TypeError, match=r"unexpected keyword argument"):
        InterUSVIntervalCalculator(foo=1, bar="x")


def test_iui_calculator_init_accepts_expected_kwargs():
    """The documented kwargs are still accepted and exposed as attributes."""
    calc = InterUSVIntervalCalculator(input_parameter_dict={},
                                      message_output=lambda *_a, **_kw: None)
    assert calc.input_parameter_dict == {}


# save_inter_usv_interval_distributions_to_file — validation paths


def test_save_iui_invalid_model_class_raises(tmp_path):
    """model_class outside {'gauss', 't'} → ValueError before any I/O."""
    settings = _make_settings(tmp_path)
    settings["compute_inter_usv_interval_distributions"]["model_class"] = "bogus"
    calc = InterUSVIntervalCalculator(
        input_parameter_dict=settings,
        message_output=lambda *_a, **_kw: None,
    )
    with pytest.raises(ValueError, match="model_class must be"):
        calc.save_inter_usv_interval_distributions_to_file()


def test_save_iui_empty_session_lists_logs_skip(tmp_path):
    """No session_lists configured → log a skip and return cleanly."""
    settings = _make_settings(tmp_path)
    msgs: list[str] = []
    calc = InterUSVIntervalCalculator(
        input_parameter_dict=settings,
        message_output=msgs.append,
    )
    calc.save_inter_usv_interval_distributions_to_file()
    assert any("no session_lists configured" in m for m in msgs)


def test_save_iui_no_output_directory_raises(tmp_path):
    """An empty output_directory string → ValueError."""
    settings = _make_settings(tmp_path)
    settings["compute_inter_usv_interval_distributions"]["session_lists"] = ["/x/y.txt"]
    settings["compute_inter_usv_interval_distributions"]["output_directory"] = ""
    calc = InterUSVIntervalCalculator(
        input_parameter_dict=settings,
        message_output=lambda *_a, **_kw: None,
    )
    with pytest.raises(ValueError, match="output_directory must be set"):
        calc.save_inter_usv_interval_distributions_to_file()


def test_save_iui_zero_resolved_sessions_logs_skip(tmp_path, monkeypatch):
    """A session list that resolves to zero readable sessions → log + return.

    We patch _read_session_lists to return an empty list (the real helper
    would only return [] for a missing or empty file, both of which the
    function still treats as "the user gave us at least one list path"
    upstream). This isolates the second-skip branch."""
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("")  # no sessions

    settings = _make_settings(tmp_path)
    settings["compute_inter_usv_interval_distributions"]["session_lists"] = [str(list_file)]

    # Force _read_session_lists to return [] regardless of file contents
    monkeypatch.setattr(iui_mod, "_read_session_lists",
                        lambda _lists, _msg: [])
    monkeypatch.setattr(iui_mod, "_session_source_map", lambda _lists: {})

    msgs: list[str] = []
    calc = InterUSVIntervalCalculator(
        input_parameter_dict=settings,
        message_output=msgs.append,
    )
    calc.save_inter_usv_interval_distributions_to_file()
    assert any("zero sessions resolved" in m for m in msgs)


# save_inter_usv_interval_distributions_to_file — happy paths (mocked)


def _mock_session_resolution(monkeypatch, tmp_path):
    """Patch _read_session_lists / _session_source_map so they return one
    synthetic session_root, and patch compute_session_usv_intervals to return
    a known interval payload for both modes."""
    sess_root = str(tmp_path / "20260101_120000")
    monkeypatch.setattr(iui_mod, "_read_session_lists",
                        lambda _lists, _msg: [sess_root])
    monkeypatch.setattr(iui_mod, "_session_source_map",
                        lambda _lists: {sess_root: "groupA"})

    def fake_compute(session_root, interval_type, exclude_noise_usvs, vocal_class=None,
                     adjacency="filtered"):
        """
        Description
        -----------
        Stands in for ``compute_session_usv_intervals``: returns a fixed male / female
        interval payload for the one synthetic session and an empty dict for any other.
        The filter arguments are accepted by name, as the orchestrator passes them, and ignored.

        Parameters
        ----------
        session_root (str)
            Session root; only the synthetic one yields intervals.
        interval_type (str)
            ``s2s`` or ``e2s``; sets ``n_dropped_male`` and is echoed back.
        exclude_noise_usvs (bool)
            Noise filter flag (ignored).
        vocal_class (str | None)
            Vocal-class filter (ignored).
        adjacency (str)
            Adjacency rule (ignored).

        Returns
        -------
        payload (dict)
            The synthetic interval payload, or ``{}`` for an unknown session.
        """
        del exclude_noise_usvs, vocal_class, adjacency
        if session_root != sess_root:
            return {}
        return {
            "male":   np.array([0.5, 0.7, 0.9]),
            "female": np.array([0.3, 0.4]),
            "n_dropped_male": 0 if interval_type == "s2s" else 1,
            "n_dropped_female": 0,
            "emitter_male": np.array(["M"] * 3, dtype=object),
            "emitter_female": np.array(["F"] * 2, dtype=object),
            "chain_male": np.ones(3, dtype=np.int64),
            "chain_female": np.ones(2, dtype=np.int64),
            "animal_sex": {"M": "male", "F": "female"},
            "interval_type": interval_type,
        }
    monkeypatch.setattr(iui_mod, "compute_session_usv_intervals", fake_compute)
    return sess_root


def test_save_iui_writes_archive_when_fit_mixture_model_false(tmp_path, mocker, monkeypatch):
    """fit_mixture_model=False → archive contains intervals + drop_counts but NOT
    mixture_model_fits / bootstrap_lrt tables. Verifies write_ivi_h5 was invoked once
    with the expected per_mode payload shape, and that the provenance names both the
    list files and the session directories they resolved to."""
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")

    settings = _make_settings(tmp_path, fit_mixture_model=False)
    settings["compute_inter_usv_interval_distributions"]["session_lists"] = [str(list_file)]

    _mock_session_resolution(monkeypatch, tmp_path)
    write_mock = mocker.patch.object(iui_mod, "write_ivi_h5",
                                     return_value=Path(tmp_path / "out" / "stub.h5"))
    monkeypatch.setattr(iui_mod, "git_sha_for_provenance", lambda _p: "stub")

    calc = InterUSVIntervalCalculator(
        input_parameter_dict=settings,
        message_output=lambda *_a, **_kw: None,
    )
    calc.save_inter_usv_interval_distributions_to_file()

    assert write_mock.call_count == 1
    per_mode = write_mock.call_args.kwargs["per_mode"]
    assert set(per_mode.keys()) == {"s2s", "e2s"}
    # the archive names the session directories it holds, not only the list files
    attrs = write_mock.call_args.kwargs["analysis_attrs"]
    assert attrs["session_roots"] == [str(tmp_path / "20260101_120000")]
    assert attrs["source_lists"] == [str(list_file)]
    # Both modes have intervals + drop_counts; mixture_model_fits / bootstrap_lrt are None
    for mode in per_mode.values():
        assert mode["intervals"].height == 5  # 3 male USV + 2 female squeak per mode
        assert set(mode["intervals"]["vocal_class"].unique()) == {"usv", "squeak"}
        assert mode["pool_summary"].height == 2
        assert mode["mixture_model_fits"] is None
        assert mode["bootstrap_lrt"] is None
        assert mode["bootstrap_lrt_null"] is None
        assert mode["tied_fits"] is None
        assert mode["peak_lrt"] is None


def test_save_iui_writes_archive_when_fit_mixture_model_true(tmp_path, mocker, monkeypatch):
    """fit_mixture_model=True → invokes fit_mixture_model_sweep AND bootstrap_lrt; the resulting
    archive carries the mixture_model_fits + bootstrap_lrt + bootstrap_lrt_null tables.
    Both expensive calls are mocked. Every rung is handed the pool's session labels, the
    step-up selects on the CORRECTED p-value (raw 0.0 would reject, corrected 0.5 keeps), and
    the archived table carries both."""
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")

    settings = _make_settings(tmp_path, fit_mixture_model=True)
    settings["compute_inter_usv_interval_distributions"]["session_lists"] = [str(list_file)]

    _mock_session_resolution(monkeypatch, tmp_path)

    # Synthetic mixture-model sweep result with the per-component columns the archive expects.
    fake_sweep = pls.DataFrame({
        "sex": ["male"], "n_comp": [1], "rep": [0],
        "bic": [1.0], "aic": [1.0], "icl": [1.0], "cv_neg_loglik": [1.0],
        "model_class": ["gauss"],
        "weight_1": [1.0], "logmean_1": [0.0], "logsd_1": [0.5], "nu_1": [float("nan")],
    })
    monkeypatch.setattr(iui_mod, "fit_mixture_model_sweep",
                        lambda **_kw: fake_sweep)

    fake_lrt_res = {
        "K_null": 1, "K_alt": 2, "B": 2, "n_subsample": 5,
        "model_class": "gauss",
        "lr_obs": 1.0, "lr_null": np.array([0.5, 1.5]),
        "p_value": 0.0, "null_mean": 1.0, "null_p95": 1.5, "null_max": 1.5,
        "design_effect": 2.0, "design_effect_raw": 2.0, "lr_corrected": 0.5,
        "p_value_corrected": 0.5,
    }
    lrt_calls: list[dict] = []

    def fake_bootstrap_lrt(**kw):
        """
        Description
        -----------
        Stands in for ``bootstrap_lrt``: records the keyword arguments of every
        invocation and returns one fixed LRT result.

        Parameters
        ----------
        **kw (dict)
            Keyword arguments the orchestrator passes.

        Returns
        -------
        result (dict)
            The fixed ``fake_lrt_res`` payload.
        """
        lrt_calls.append(kw)
        return fake_lrt_res
    monkeypatch.setattr(iui_mod, "bootstrap_lrt", fake_bootstrap_lrt)
    selector_inputs: list[dict] = []

    def fake_select(pair_results, alpha):
        """
        Description
        -----------
        Stands in for ``select_n_components_step_up_lrt``: records the per-pair inputs
        and always selects one component. ``alpha`` arrives by keyword and is ignored.

        Parameters
        ----------
        pair_results (dict)
            Per-(K_null, K_alt) p-values handed to the selector.
        alpha (float)
            Effective significance level (ignored).

        Returns
        -------
        n_components (int)
            Always 1.
        """
        del alpha
        selector_inputs.append(pair_results)
        return 1
    monkeypatch.setattr(iui_mod, "select_n_components_step_up_lrt", fake_select)

    write_mock = mocker.patch.object(iui_mod, "write_ivi_h5",
                                     return_value=Path(tmp_path / "out" / "stub.h5"))
    monkeypatch.setattr(iui_mod, "git_sha_for_provenance", lambda _p: "stub")

    calc = InterUSVIntervalCalculator(
        input_parameter_dict=settings,
        message_output=lambda *_a, **_kw: None,
    )
    calc.save_inter_usv_interval_distributions_to_file()

    assert write_mock.call_count == 1
    per_mode = write_mock.call_args.kwargs["per_mode"]
    for mode in per_mode.values():
        assert mode["mixture_model_fits"] is not None
        assert mode["bootstrap_lrt"] is not None
        assert mode["bootstrap_lrt_null"] is not None
        # Only the fitted pool (the male's USVs) is swept; the female's squeaks are described only.
        assert "K_selected_male" in mode["attrs"]
        assert "K_selected_female" not in mode["attrs"]
        assert mode["bootstrap_lrt"]["p_value_corrected"].to_list()[0] == 0.5
        assert mode["bootstrap_lrt"]["p_value"].to_list()[0] == 0.0
    # the male pool is 3 intervals from one session, and its labels travel with it
    assert lrt_calls
    assert all(list(kw["session_labels"]) == ["20260101_120000"] * 3 for kw in lrt_calls)
    assert all(res["p_value"] == 0.5 for inputs in selector_inputs for res in inputs.values())


@pytest.mark.parametrize(("pools", "match"), [
    ([{"sex": "male", "vocal_class": None, "adjacency": "filtered", "fit": False}],
     "vocal_class must be 'usv' or 'squeak'"),
    ([{"sex": "male", "vocal_class": "usv", "adjacency": "filtered", "fit": False},
      {"sex": "male", "vocal_class": "usv", "adjacency": "filtered", "fit": True}],
     "declared more than once"),
    ([{"sex": "male", "vocal_class": "usv", "adjacency": "filtered", "fit": True},
      {"sex": "male", "vocal_class": "usv", "adjacency": "strict", "fit": True}],
     "share sex 'male' and vocal_class 'usv'"),
    ([{"sex": "male", "vocal_class": "usv", "adjacency": "filtered", "fit": True},
      {"sex": "male", "vocal_class": "squeak", "adjacency": "strict", "fit": True}],
     "share sex 'male'"),
])
def test_save_iui_rejects_pools_the_archive_cannot_hold_before_computing(tmp_path, monkeypatch,
                                                                         pools, match):
    """
    Pool declarations are validated before any session is read: a pool without a named vocal
    class (``vocal_class: null``, which used to crash the archive write after the full run), a
    pool declared twice (whose values silently collapsed onto one key), two fitted pools that
    would write the same tied-test attribute ``selected_n_peak_<sex>_<vocal_class>``, and, with
    ``fit_mixture_model`` on, two fitted pools of one sex, which the unconstrained sweep's
    ``sex``-keyed tables and ``K_selected_<sex>`` attribute cannot tell apart.
    """
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")
    settings = _make_settings(tmp_path, fit_mixture_model=True, fit_tied_model=True)
    block = settings["compute_inter_usv_interval_distributions"]
    block["session_lists"] = [str(list_file)]
    block["interval_pools"] = pools
    computed: list = []
    monkeypatch.setattr(iui_mod, "_read_session_lists", lambda _lists, _msg: ["/dummy/session"])
    monkeypatch.setattr(iui_mod, "_session_source_map", lambda _lists: {})
    monkeypatch.setattr(iui_mod, "compute_session_usv_intervals",
                        lambda **kwargs: computed.append(kwargs) or {})

    with pytest.raises(ValueError, match=match):
        InterUSVIntervalCalculator(input_parameter_dict=settings,
                                   message_output=lambda *_a, **_kw: None
                                   ).save_inter_usv_interval_distributions_to_file()
    assert computed == []


def test_save_iui_two_fitted_pools_of_one_sex_without_the_sweep_keep_both_selections(
        tmp_path, mocker, monkeypatch):
    """
    Two fitted pools of one sex with different vocal classes are legal while the unconstrained
    sweep is off: each pool's tied-test selection lands under its own
    ``selected_n_peak_<sex>_<vocal_class>`` attribute and neither overwrites the other.
    """
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")
    settings = _make_settings(tmp_path, fit_tied_model=True)
    block = settings["compute_inter_usv_interval_distributions"]
    block["session_lists"] = [str(list_file)]
    block["interval_pools"] = [
        {"sex": "male", "vocal_class": "usv", "adjacency": "filtered", "fit": True},
        {"sex": "male", "vocal_class": "squeak", "adjacency": "strict", "fit": True}]
    _mock_session_resolution(monkeypatch, tmp_path)

    def fake_tied(**kwargs):
        """Returns a selection that identifies the pool."""
        identity = kwargs["pool_identity"]
        one = pls.DataFrame([{**identity, "n_peak": 1}])
        return one, one, one, one, (1 if identity["vocal_class"] == "usv" else 3), 0.01

    monkeypatch.setattr(iui_mod, "fit_tied_peak_ladder_and_lrt", fake_tied)
    write_mock = mocker.patch.object(iui_mod, "write_ivi_h5",
                                     return_value=Path(tmp_path / "out" / "stub.h5"))
    monkeypatch.setattr(iui_mod, "git_sha_for_provenance", lambda _p: "stub")

    InterUSVIntervalCalculator(input_parameter_dict=settings,
                               message_output=lambda *_a, **_kw: None
                               ).save_inter_usv_interval_distributions_to_file()

    for mode in write_mock.call_args.kwargs["per_mode"].values():
        assert mode["attrs"]["selected_n_peak_male_usv"] == 1
        assert mode["attrs"]["selected_n_peak_male_squeak"] == 3


def test_save_iui_creates_output_directory(tmp_path, mocker, monkeypatch):
    """The output_directory is created with parents=True, exist_ok=True before
    the archive is written. We verify by pointing at a deeply-nested path."""
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")

    nested_out = tmp_path / "deep" / "nested" / "out"
    settings = _make_settings(tmp_path)
    settings["compute_inter_usv_interval_distributions"]["session_lists"] = [str(list_file)]
    settings["compute_inter_usv_interval_distributions"]["output_directory"] = str(nested_out)

    _mock_session_resolution(monkeypatch, tmp_path)
    mocker.patch.object(iui_mod, "write_ivi_h5",
                        return_value=Path(nested_out / "stub.h5"))
    monkeypatch.setattr(iui_mod, "git_sha_for_provenance", lambda _p: "stub")

    calc = InterUSVIntervalCalculator(
        input_parameter_dict=settings,
        message_output=lambda *_a, **_kw: None,
    )
    calc.save_inter_usv_interval_distributions_to_file()

    assert nested_out.is_dir()


def test_save_iui_runs_tied_model_on_fitted_pools_only(tmp_path, mocker, monkeypatch):
    """fit_tied_model=True -> the tied peak ladder and session-corrected test run on the pool
    marked ``fit`` and not on the described-only pool; their tables and the selected peak count
    reach the archive. The expensive fit is mocked and its call arguments checked."""
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")
    settings = _make_settings(tmp_path, fit_tied_model=True)
    settings["compute_inter_usv_interval_distributions"]["session_lists"] = [str(list_file)]
    _mock_session_resolution(monkeypatch, tmp_path)

    calls: list[dict] = []

    def fake_tied(**kwargs):
        """
        Description
        -----------
        Stands in for ``fit_tied_peak_ladder_and_lrt``: records the keyword arguments
        and returns four one-row tables tagged with the pool identity, two selected
        peaks and a fixed effective alpha.

        Parameters
        ----------
        **kwargs (dict)
            Keyword arguments the orchestrator passes, including ``pool_identity``.

        Returns
        -------
        result (tuple)
            Four one-row DataFrames, the selected peak count 2 and 0.0033.
        """
        calls.append(kwargs)
        identity = kwargs["pool_identity"]
        one = pls.DataFrame([{**identity, "n_peak": 2}])
        return one, one, one, one, 2, 0.0033

    monkeypatch.setattr(iui_mod, "fit_tied_peak_ladder_and_lrt", fake_tied)
    write_mock = mocker.patch.object(iui_mod, "write_ivi_h5",
                                     return_value=Path(tmp_path / "out" / "stub.h5"))
    monkeypatch.setattr(iui_mod, "git_sha_for_provenance", lambda _p: "stub")

    InterUSVIntervalCalculator(input_parameter_dict=settings,
                               message_output=lambda *_a, **_kw: None
                               ).save_inter_usv_interval_distributions_to_file()

    assert len(calls) == 2  # one fitted pool x two interval types
    assert all(c["pool_identity"]["sex"] == "male" for c in calls)
    assert all(c["session_labels"].size == c["values"].size for c in calls)
    per_mode = write_mock.call_args.kwargs["per_mode"]
    for mode in per_mode.values():
        assert mode["tied_fits"] is not None
        assert mode["peak_lrt"] is not None
        assert mode["attrs"]["selected_n_peak_male_usv"] == 2
        fitted = dict(zip(mode["pool_summary"]["sex"], mode["pool_summary"]["fitted"], strict=True))
        assert fitted == {"male": True, "female": False}


def test_save_iui_fits_serial_dependence_on_fitted_pools_only(tmp_path, mocker, monkeypatch):
    """fit_serial_dependence=True → the fit runs on the fitted pool only (the male's USVs, not the
    female's squeaks), is handed that pool's consecutive same-animal pairs with their sessions,
    and its three tables land in every mode's payload."""
    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")
    settings = _make_settings(tmp_path, fit_serial_dependence=True)
    settings["compute_inter_usv_interval_distributions"]["session_lists"] = [str(list_file)]
    _mock_session_resolution(monkeypatch, tmp_path)

    calls: list[dict] = []

    def fake_fit(**kw):
        """
        Description
        -----------
        Stands in for ``fit_serial_dependence``: records the keyword arguments and
        returns the three result tables (curve, summary, bootstrap) tagged with the
        pool identity.

        Parameters
        ----------
        **kw (dict)
            Keyword arguments the orchestrator passes, including ``pool_identity`` and ``current``.

        Returns
        -------
        tables (tuple)
            Three polars DataFrames shaped like the real fit's output.
        """
        calls.append(kw)
        identity = kw["pool_identity"]
        return (pls.DataFrame({**{k: [v] for k, v in identity.items()}, "current_ms": [1.0]}),
                pls.DataFrame([{**identity, "n_pairs": kw["current"].size, "n_sessions": 1,
                                "bend_ms": 110.0, "bend_low_ms": 100.0, "bend_high_ms": 120.0,
                                "slope": 0.5, "flat_level_ms": 90.0, "level": 99.0,
                                "n_bootstrap": 5, "spline_not_converged": 0,
                                "bent_not_converged": 0}]),
                pls.DataFrame({**{k: [v] for k, v in identity.items()}, "b": [0], "bend_ms": [110.0]}))
    monkeypatch.setattr(iui_mod, "fit_serial_dependence", fake_fit)
    write_mock = mocker.patch.object(iui_mod, "write_ivi_h5",
                                     return_value=Path(tmp_path / "out" / "stub.h5"))
    monkeypatch.setattr(iui_mod, "git_sha_for_provenance", lambda _p: "stub")

    InterUSVIntervalCalculator(input_parameter_dict=settings,
                               message_output=lambda *_a, **_kw: None
                               ).save_inter_usv_interval_distributions_to_file()

    # one call per interval type, all on the male pool: 3 intervals of one animal -> 2 pairs
    assert [kw["pool_identity"]["sex"] for kw in calls] == ["male", "male"]
    for kw in calls:
        np.testing.assert_allclose(kw["current"], [0.5, 0.7])
        np.testing.assert_allclose(kw["following"], [0.7, 0.9])
        assert list(kw["pair_sessions"]) == ["20260101_120000"] * 2
    per_mode = write_mock.call_args.kwargs["per_mode"]
    for mode in per_mode.values():
        for table in ("serial_dependence_curves", "serial_dependence_fit", "serial_dependence_bends"):
            assert mode[table] is not None
            assert set(mode[table]["sex"].to_list()) == {"male"}
    assert write_mock.call_args.kwargs["analysis_attrs"]["serial_dependence_level"] == 99.0


def _two_peak_intervals(n_sessions=6, per_session=120, seed=0):
    """
    Draws synthetic end-to-start intervals with the male pool's shape: a dominant narrow
    log-normal peak near 63 ms, a smaller one near 180 ms sharing its log-scale, and a broad
    background near 0.3 s, split evenly across ``n_sessions`` sessions.

    Returns
    -------
    values (np.ndarray)
        Intervals in seconds.
    session_labels (np.ndarray)
        The session each interval came from, aligned with ``values``.
    """

    rng = np.random.default_rng(seed)
    n = n_sessions * per_session
    role = rng.choice(3, size=n, p=[0.65, 0.1, 0.25])
    log_values = np.where(role == 0, rng.normal(np.log(0.063), 0.24, n),
                          np.where(role == 1, rng.normal(np.log(0.18), 0.24, n),
                                   rng.normal(np.log(0.3), 1.1, n)))
    labels = np.repeat(np.arange(n_sessions), per_session).astype(str)
    return np.exp(log_values), labels


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fit_tied_peak_ladder_and_lrt_tables_and_selection():
    """
    Runs the real tied-scale ladder and session-corrected peak-count test on a small
    two-peak pool (tiny B, subsample and design bootstrap so it stays fast) and checks the
    contract of all four tables and the two scalars: the ladder fits every peak count from the
    grid's minimum to one past its maximum, each fit carries n_peak peak rows plus the
    background rows sharing one peak scale, the modes table has at least one mode per fit, the
    LRT table has one row per rung with a finite design effect of at least one and the
    Bonferroni-divided alpha, the null table holds B draws per rung, the pool identity is
    stamped on every row, and the selection is a count the grid can return.
    """

    values, labels = _two_peak_intervals()
    identity = {"sex": "male", "vocal_class": "usv"}
    messages = []
    fits, modes, lrt, null, selected, alpha_eff = fit_tied_peak_ladder_and_lrt(
        values=values, session_labels=labels, pool_identity=identity, peak_grid=[1, 2],
        n_background=1, n_init=1, n_init_boot=1, reg_covar=1e-4, B=19, n_subsample=300,
        alpha=0.01, bonferroni=True, n_design_bootstrap=50, seed=0, n_jobs=1,
        design_bootstrap_iid_cap=400, em_max_iter=500, em_tol=1e-6,
        split_delta=0.5, mode_grid_size=16384, mode_span_scales=6.0, em_init=_EM_INIT,
        message=messages.append)

    assert sorted(fits["n_peak"].unique().to_list()) == [1, 2, 3]
    for n_peak in (1, 2, 3):
        rows = fits.filter(pls.col("n_peak") == n_peak)
        assert rows.height == n_peak + 1
        assert (rows["role"] == "peak").sum() == n_peak
        peaks = rows.filter(pls.col("role") == "peak")
        assert np.allclose(peaks["logscale"].to_numpy(), rows["shared_peak_scale"][0])
        assert np.isclose(rows["weight"].sum(), 1.0)
    assert set(modes["n_peak"].unique().to_list()) == {1, 2, 3}
    assert lrt.height == 2
    assert lrt["n_peak_null"].to_list() == [1, 2]
    assert (lrt["design_effect"] >= 1.0).all()
    assert np.allclose(lrt["alpha_used"].to_numpy(), 0.005)
    assert alpha_eff == pytest.approx(0.005)
    assert null.height == 2 * 19
    for table in (fits, modes, lrt, null):
        assert (table["sex"] == "male").all()
        assert (table["vocal_class"] == "usv").all()
    assert selected in (1, 2, 3)
    assert any("step-up selection" in m for m in messages)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fit_tied_peak_ladder_and_lrt_ladders_each_rung_null_from_the_previous_alternative(
        monkeypatch):
    """
    Rung ``n``'s observed null is seeded by rung ``n - 1``'s observed alternative (the same
    subsample, since the subsample depends only on the pool and the seed), while the first rung
    and any rung whose predecessor is not ``n - 1`` start from scratch. The test function is
    replaced by a recorder so only the hand-over is checked.
    """

    values, labels = _two_peak_intervals()
    starts: list = []
    alternatives: dict = {}

    def fake_lrt(**kwargs):
        """Records the null start and returns a scripted rung result with a unique alternative."""
        starts.append((kwargs["n_peak_null"], kwargs["null_start"]))
        alternatives[kwargs["n_peak_null"]] = object()
        return {"lr_obs": 1.0, "design_effect": 1.0, "design_effect_raw": 1.0, "effective_n": 10.0,
                "lr_corrected": 1.0, "null_p95": 2.0, "p_value": 0.5, "p_value_corrected": 0.5,
                "negative_fraction": 0.0, "B": 2, "n_subsample": 10, "lr_null": np.zeros(2),
                "alt_model": alternatives[kwargs["n_peak_null"]]}

    monkeypatch.setattr(iui_mod, "tied_peak_bootstrap_lrt", fake_lrt)
    fit_tied_peak_ladder_and_lrt(
        values=values, session_labels=labels, pool_identity={"sex": "male"}, peak_grid=[1, 2, 4],
        n_background=1, n_init=1, n_init_boot=1, reg_covar=1e-4, B=2, n_subsample=300,
        alpha=0.05, bonferroni=False, n_design_bootstrap=10, seed=0, n_jobs=1,
        design_bootstrap_iid_cap=400, em_max_iter=500, em_tol=1e-6,
        split_delta=0.5, mode_grid_size=16384, mode_span_scales=6.0, em_init=_EM_INIT,
        message=lambda _m: None)

    assert [n for n, _ in starts] == [1, 2, 4]
    assert starts[0][1] is None
    assert starts[1][1] is alternatives[1]
    assert starts[2][1] is None


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fit_tied_peak_ladder_and_lrt_without_bonferroni_keeps_alpha():
    """
    With ``bonferroni=False`` the per-rung level is the family level itself, and a single-rung
    grid still fits one peak count past the grid (the alternative of its only rung).
    """

    values, labels = _two_peak_intervals(seed=1)
    fits, _modes, lrt, null, selected, alpha_eff = fit_tied_peak_ladder_and_lrt(
        values=values, session_labels=labels, pool_identity={"sex": "female"}, peak_grid=[1],
        n_background=1, n_init=1, n_init_boot=1, reg_covar=1e-4, B=9, n_subsample=200,
        alpha=0.05, bonferroni=False, n_design_bootstrap=20, seed=3, n_jobs=1,
        design_bootstrap_iid_cap=400, em_max_iter=500, em_tol=1e-6,
        split_delta=0.5, mode_grid_size=16384, mode_span_scales=6.0, em_init=_EM_INIT,
        message=lambda _m: None)

    assert alpha_eff == pytest.approx(0.05)
    assert sorted(fits["n_peak"].unique().to_list()) == [1, 2]
    assert lrt.height == 1
    assert null.height == 9
    assert selected in (1, 2)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("model_class", ["gauss", "ig", "t"])
def test_fit_mixture_model_sweep_model_class_branches(model_class):
    """
    The sweep is model-class agnostic, and the calculator tests stub it out, so each model
    class (Gaussian, inverse-Gaussian and the Student-t ladder, where each K starts from the
    K-1 solution) is run here for real: cross-validated likelihood, fit and per-model
    statistics. Checks one row per (key, repeat, K), finite information
    criteria and CV likelihoods, and that a key too short for the CV folds gets NaN CV values
    rather than being dropped from the fits.
    """

    values, _ = _two_peak_intervals(n_sessions=2, per_session=150, seed=2)
    table = fit_mixture_model_sweep(
        intervals_by_key={"male": values, "female": values[:4]}, n_components_min=1,
        n_components_max=2, n_repeats=1, max_modes_reported=2, random_seed_base=0,
        cv_n_folds=5, cv_n_init=1, mixture_model_n_init=1, model_class=model_class,
        tau=0.5, mixture_model_reg_covar=1e-4, em_max_iter=300, em_tol=1e-5, split_delta=0.5,
        mode_grid_size_gauss=4096, mode_span_scales_gauss=4.0, mode_grid_size_t=16384,
        mode_span_scales_t=6.0, cv_t_n_init_offset=2, em_init=_EM_INIT)

    male = table.filter(pls.col("sex") == "male")
    assert male.height == 2
    assert sorted(male["n_comp"].to_list()) == [1, 2]
    assert (table["model_class"] == model_class).all()
    assert np.isfinite(table["bic"].to_numpy()).all()
    male_cv = male["cv_neg_loglik"].to_numpy()
    female_cv = table.filter(pls.col("sex") == "female")["cv_neg_loglik"].to_numpy()
    assert np.isfinite(male_cv).all()
    assert np.isnan(female_cv).all()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_fit_mixture_model_sweep_ig_median_is_the_component_median():
    """
    For an inverse-Gaussian component ``logmean_k`` is the log of the component MEAN ``mu``, and
    the IG is right-skewed, so ``exp(logmean_k)`` is not its median. ``median_sec_k`` must be the
    true median of IG(mu_k, lambda_k) in seconds (``scipy.stats.invgauss(mu / lambda,
    scale=lambda).median()``), and it must lie below the mean.
    """

    rng = np.random.default_rng(0)
    values = np.concatenate([invgauss.rvs(0.07 / 0.2, scale=0.2, size=600, random_state=rng),
                             invgauss.rvs(3.0 / 2.0, scale=2.0, size=300, random_state=rng)])
    table = fit_mixture_model_sweep(
        intervals_by_key={"male": values}, n_components_min=1, n_components_max=2, n_repeats=1,
        max_modes_reported=2, random_seed_base=0, cv_n_folds=5, cv_n_init=1,
        mixture_model_n_init=1, model_class="ig",
        tau=0.5, mixture_model_reg_covar=1e-4, em_max_iter=300, em_tol=1e-5, split_delta=0.5,
        mode_grid_size_gauss=4096, mode_span_scales_gauss=4.0, mode_grid_size_t=16384,
        mode_span_scales_t=6.0, cv_t_n_init_offset=2, em_init=_EM_INIT)

    for row in table.iter_rows(named=True):
        for k in range(1, row["n_comp"] + 1):
            mu, lam = float(np.exp(row[f"logmean_{k}"])), row[f"lambda_{k}"]
            assert row[f"median_sec_{k}"] == pytest.approx(
                float(invgauss(mu / lam, scale=lam).median()), rel=1e-9)
            assert row[f"median_sec_{k}"] < mu


def test_fit_mixture_model_sweep_rejects_unknown_model_class():
    """An unknown ``model_class`` is a ValueError before any fitting."""

    with pytest.raises(ValueError, match="model_class"):
        fit_mixture_model_sweep(intervals_by_key={"male": np.array([0.1, 0.2])}, n_components_min=1,
                                n_components_max=2, n_repeats=1, max_modes_reported=1,
                                random_seed_base=0, model_class="lognormal",
                                cv_n_folds=5, cv_n_init=1, mixture_model_n_init=1,
                                tau=0.5, mixture_model_reg_covar=1e-4, em_max_iter=300, em_tol=1e-5, split_delta=0.5,
                                mode_grid_size_gauss=4096, mode_span_scales_gauss=4.0, mode_grid_size_t=16384,
                                mode_span_scales_t=6.0, cv_t_n_init_offset=2, em_init=_EM_INIT)


def test_compute_session_usv_intervals_rejects_unknown_adjacency():
    """``adjacency`` other than 'filtered' / 'strict' is a ValueError before any I/O."""

    with pytest.raises(ValueError, match="Unknown adjacency"):
        compute_session_usv_intervals(session_root="/whatever", interval_type="e2s",
                                      exclude_noise_usvs=True, adjacency="loose")


def _write_metadata(root: Path, subjects: list[dict]) -> None:
    """Writes a minimal ``<session>_metadata.yaml`` holding only the ``Subjects`` block."""

    lines = ["Subjects:"]
    for subject in subjects:
        lines.append(f"  - subject_id: '{subject['subject_id']}'")
        for key, value in subject.items():
            if key != "subject_id":
                lines.append(f"    {key}: {value}")
    (root / f"{root.name}_metadata.yaml").write_text("\n".join(lines) + "\n")


def test_extract_animal_sexes_reads_and_normalises(tmp_path):
    """Sexes come from the metadata's Subjects block, with padding stripped from the track
    names and the recorded value lower-cased."""

    _write_metadata(tmp_path, [{"subject_id": "A_0", "sex": "Male"}, {"subject_id": "B_1", "sex": "female"}])
    assert extract_animal_sexes(str(tmp_path), [" A_0\x00", "B_1"]) == {"A_0": "male", "B_1": "female"}


def test_extract_animal_sexes_missing_metadata_raises(tmp_path):
    """A session with no metadata file cannot resolve sexes: FileNotFoundError, never a guess."""

    with pytest.raises(FileNotFoundError, match="metadata"):
        extract_animal_sexes(str(tmp_path), ["A_0"])


def test_extract_animal_sexes_track_without_sex_raises(tmp_path):
    """A track whose subject has no recorded sex is a ValueError naming the subjects that do."""

    _write_metadata(tmp_path, [{"subject_id": "A_0", "sex": "male"}, {"subject_id": "B_1"}])
    with pytest.raises(ValueError, match="no subject with a recorded sex"):
        extract_animal_sexes(str(tmp_path), ["A_0", "B_1"])


def test_extract_animal_sexes_without_subjects_raises(tmp_path):
    """A metadata file with no Subjects block is a ValueError, not an empty mapping."""

    (tmp_path / f"{tmp_path.name}_metadata.yaml").write_text("Session:\n  notes: none\n")
    with pytest.raises(ValueError, match="has no Subjects"):
        extract_animal_sexes(str(tmp_path), ["A_0"])


def test_extract_animal_sexes_is_the_shared_yaml_utils_helper():
    """The analyses re-export and the processing / visualization callers use one function."""

    assert extract_animal_sexes is yaml_extract_animal_sexes


def test_emitter_sex_expression_maps_stripped_emitters_and_unassigned():
    """Emitters are stripped before matching, unknown / null emitters are 'unassigned',
    and a same-sex mapping gives both animals the same sex."""

    df = pls.DataFrame({"emitter": ["A_0\x00", " B_1", None, "ghost"]})
    out = df.select(emitter_sex_expression({"A_0": "female", "B_1": "female"}))
    assert out["sex"].to_list() == ["female", "female", "unassigned", "unassigned"]
    assert df.select(emitter_sex_expression({}))["sex"].to_list() == ["unassigned"] * 4


def test_sex_track_ids_names_single_male_and_female_only():
    """male_id / female_id exist only when the session holds exactly one animal of that sex."""

    assert sex_track_ids({"F": "female", "M": "male"}) == ("M", "F")
    assert sex_track_ids({"A": "female", "B": "female"}) == (None, None)
    assert sex_track_ids({"A": "male", "B": "male", "C": "female"}) == (None, "C")


def test_extract_animal_sexes_invalid_sex_raises(tmp_path):
    """A recorded sex other than male / female is a ValueError, not silently mapped."""

    _write_metadata(tmp_path, [{"subject_id": "A_0", "sex": "unknown"}])
    with pytest.raises(ValueError, match="expected male or female"):
        extract_animal_sexes(str(tmp_path), ["A_0"])


def _write_summary(root: Path, with_vocal_flags: bool) -> None:
    """Writes a five-row ``audio/<session>_usv_summary.csv`` (pure usv, pure squeak, pure usv, both
    flags true and an unscored null), with or without the ``usv`` / ``squeak`` booleans and the
    squeak extent: the pure squeak sounds at 2.02-2.08 s inside its 2.0-2.1 s segment, the overlap
    segment's squeak at 4.03-4.15 s, outlasting its 4.0-4.1 s segment."""

    audio = root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    frame = pls.DataFrame({"start": [1.0, 2.0, 3.0, 4.0, 5.0], "stop": [1.1, 2.1, 3.1, 4.1, 5.1],
                           "noise": [False, False, False, False, False]})
    if with_vocal_flags:
        frame = frame.with_columns(
            pls.Series("usv", [True, False, True, True, None], dtype=pls.Boolean),
            pls.Series("squeak", [False, True, False, True, None], dtype=pls.Boolean),
            pls.Series("squeak_start", [None, 2.02, None, 4.03, None], dtype=pls.Float64),
            pls.Series("squeak_end", [None, 2.08, None, 4.15, None], dtype=pls.Float64),
        )
    frame.write_csv(str(audio / f"{root.name}_usv_summary.csv"))


def test_load_and_filter_usv_data_vocal_class_filters(tmp_path):
    """``vocal_class='usv'`` keeps the pure USVs (usv true, squeak false) and ``'squeak'`` the pure
    squeaks; a segment with both flags true (usv true is not enough) and an unscored (null) row
    belong to neither, and None keeps every row. Each row is timed as its vocalization: the pure
    squeak at its squeak extent (2.02-2.08 s), and under None the overlap segment at the union of
    the segment and its squeak (4.0-4.15 s); pure USVs keep their segments."""

    _write_summary(tmp_path, with_vocal_flags=True)
    usv = load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, vocal_class="usv")
    squeak = load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, vocal_class="squeak")
    every = load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, vocal_class=None)
    assert usv["start"].to_list() == [1.0, 3.0]
    assert squeak["start"].to_list() == [2.02]
    assert squeak["stop"].to_list() == [2.08]
    assert squeak["frame_index"].to_list() == [int(np.floor(2.02 * 150.0))]
    assert every.height == 5
    assert every["start"].to_list() == [1.0, 2.02, 3.0, 4.0, 5.0]
    assert every["stop"].to_list() == [1.1, 2.08, 3.1, 4.15, 5.1]


def test_load_and_filter_usv_data_squeaks_need_the_squeak_extent(tmp_path):
    """``'squeak'`` and None time squeaks by their extent, so a summary without
    ``squeak_start`` / ``squeak_end`` raises for them; ``'usv'`` does not need the columns."""

    _write_summary(tmp_path, with_vocal_flags=True)
    summary_path = tmp_path / "audio" / f"{tmp_path.name}_usv_summary.csv"
    pls.read_csv(summary_path).drop("squeak_start", "squeak_end").write_csv(summary_path)
    for vocal_class in ("squeak", None):
        with pytest.raises(KeyError, match="squeak_start"):
            load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, vocal_class=vocal_class)
    usv = load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, vocal_class="usv")
    assert usv["start"].to_list() == [1.0, 3.0]


def test_load_and_filter_usv_data_rejects_unknown_vocal_class(tmp_path):
    """An unknown ``vocal_class`` is a ValueError."""

    _write_summary(tmp_path, with_vocal_flags=True)
    with pytest.raises(ValueError, match="vocal_class"):
        load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, vocal_class="bark")


def test_load_and_filter_usv_data_without_vocal_flags_raises(tmp_path):
    """Asking for a vocal class on a summary the USV / squeak classifier never ran on is a KeyError that
    says which step to run, rather than treating every row as a USV."""

    _write_summary(tmp_path, with_vocal_flags=False)
    with pytest.raises(KeyError, match="detect-usv-squeaks"):
        load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, vocal_class="usv")


def _patch_session_identity(monkeypatch) -> None:
    """Replaces the metadata and sex readers of the interval module with a fixed two-animal
    session (male ``M``, female ``F``, 150 fps), so only the USV summary is read from disk."""

    monkeypatch.setattr(iui_mod, "extract_session_metadata",
                        lambda _root: {"track_names": ["M", "F"], "frame_rate": 150.0})
    monkeypatch.setattr(iui_mod, "extract_animal_sexes", lambda _root, _ids: {"M": "male", "F": "female"})


def test_compute_session_usv_intervals_overlap_is_neither_usv_nor_squeak(tmp_path, monkeypatch):
    """An overlap segment (usv and squeak flags both true) is left out of the squeak sequence: under ``'filtered'`` it is removed
    before pairing (the squeaks either side of it form one interval), under ``'strict'`` it stays
    in the record as a non-target vocalization and breaks the pairs it sits between. It never enters the
    USV sequence either."""

    _patch_session_identity(monkeypatch)
    audio = tmp_path / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    pls.DataFrame({
        "start": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "stop": [1.1, 2.1, 3.1, 4.1, 5.1, 6.1],
        "emitter": ["F", "F", "F", "F", "M", "M"],
        "noise": [False] * 6,
        "usv": [False, True, False, False, True, True],
        "squeak": [True, True, True, True, False, True],
        # each squeak's extent is its segment, so the intervals are the segment intervals
        "squeak_start": [1.0, 2.0, 3.0, 4.0, None, 6.0],
        "squeak_end": [1.1, 2.1, 3.1, 4.1, None, 6.1],
    }).write_csv(str(audio / f"{tmp_path.name}_usv_summary.csv"))

    filtered = compute_session_usv_intervals(str(tmp_path), "s2s", exclude_noise_usvs=True,
                                             vocal_class="squeak", adjacency="filtered")
    strict = compute_session_usv_intervals(str(tmp_path), "s2s", exclude_noise_usvs=True,
                                           vocal_class="squeak", adjacency="strict")
    usv = compute_session_usv_intervals(str(tmp_path), "s2s", exclude_noise_usvs=True,
                                        vocal_class="usv", adjacency="filtered")
    np.testing.assert_allclose(filtered["female"], [2.0, 1.0])
    np.testing.assert_allclose(strict["female"], [1.0])
    assert usv["male"].size == 0
    assert usv["female"].size == 0


def test_compute_session_usv_intervals_chains_break_at_another_animals_vocalization(tmp_path, monkeypatch):
    """The male's intervals carry a chain id that changes wherever his run is broken: his
    intervals 1->2 and 2->3 share vocalization 2 (one chain), the female's USV between his
    vocalizations 3 and 4 records no male interval and starts a new chain for 4->5, and a
    zero interval (5 -> 5 again) is dropped and breaks the chain too. Only neighbours within
    a chain pair."""

    _patch_session_identity(monkeypatch)
    audio = tmp_path / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    pls.DataFrame({
        "start": [1.0, 1.2, 1.5, 1.7, 2.0, 2.4, 2.4, 2.6],
        "stop": [1.1, 1.3, 1.6, 1.8, 2.1, 2.5, 2.5, 2.7],
        "emitter": ["M", "M", "M", "F", "M", "M", "M", "M"],
        "noise": [False] * 8,
        "usv": [True] * 8,
        "squeak": [False] * 8,
        "squeak_start": [None] * 8,
        "squeak_end": [None] * 8,
    }, schema_overrides={"squeak_start": pls.Float64, "squeak_end": pls.Float64}).write_csv(
        str(audio / f"{tmp_path.name}_usv_summary.csv"))

    out = compute_session_usv_intervals(str(tmp_path), "s2s", exclude_noise_usvs=True,
                                        vocal_class="usv", adjacency="filtered")
    np.testing.assert_allclose(out["male"], [0.2, 0.3, 0.4, 0.2])
    np.testing.assert_array_equal(out["chain_male"], [1, 1, 2, 3])
    assert out["n_dropped_male"] == 1
    current, following, _ = iui_mod.consecutive_interval_pairs(
        out["male"], np.array(["s"] * 4), out["emitter_male"], out["chain_male"])
    np.testing.assert_allclose(current, [0.2])
    np.testing.assert_allclose(following, [0.3])


def test_compute_session_usv_intervals_time_squeaks_by_their_extent(tmp_path, monkeypatch):
    """A squeak interval runs between the squeaks themselves (squeak_start / squeak_end), not
    their segments: two pure squeaks whose segments start 1.0 s apart but whose squeaks start at
    1.05 s and 2.20 s give a 1.15 s start-to-start interval and a 1.05 s end-to-start one
    (first squeak ends at 1.15 s)."""

    _patch_session_identity(monkeypatch)
    audio = tmp_path / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    pls.DataFrame({
        "start": [1.0, 2.0],
        "stop": [1.2, 2.3],
        "emitter": ["F", "F"],
        "noise": [False, False],
        "usv": [False, False],
        "squeak": [True, True],
        "squeak_start": [1.05, 2.20],
        "squeak_end": [1.15, 2.25],
    }).write_csv(str(audio / f"{tmp_path.name}_usv_summary.csv"))

    for adjacency in ("filtered", "strict"):
        s2s = compute_session_usv_intervals(str(tmp_path), "s2s", exclude_noise_usvs=True,
                                            vocal_class="squeak", adjacency=adjacency)
        e2s = compute_session_usv_intervals(str(tmp_path), "e2s", exclude_noise_usvs=True,
                                            vocal_class="squeak", adjacency=adjacency)
        np.testing.assert_allclose(s2s["female"], [1.15])
        np.testing.assert_allclose(e2s["female"], [1.05])


def test_unique_match_or_none_rejects_ambiguous_globs(tmp_path):
    """One match is returned, none gives None, and two raise naming both candidates (the
    ``sorted(glob)[0]`` idiom silently took the alphabetically-first file)."""

    assert unique_match_or_none([], label="x") is None
    only = tmp_path / "a.csv"
    assert unique_match_or_none([only], label="x") == only
    with pytest.raises(ValueError, match=r"Expected one USV summary CSV, found 2"):
        unique_match_or_none([tmp_path / "b.csv", only], label="USV summary CSV")


def test_load_and_filter_usv_data_raises_on_two_summaries(tmp_path):
    """A session holding two ``*_usv_summary.csv`` files (e.g. a stale copy beside the current
    one) is ambiguous: the loader used to read the alphabetically-first one without a word, and
    must now raise."""

    _write_summary(tmp_path, with_vocal_flags=True)
    stale = tmp_path / "audio" / "old" / f"{tmp_path.name}_usv_summary.csv"
    stale.parent.mkdir()
    stale.write_text((tmp_path / "audio" / f"{tmp_path.name}_usv_summary.csv").read_text())
    with pytest.raises(ValueError, match="Expected one USV summary CSV"):
        load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, vocal_class="usv")


def test_soft_minimum_rounds_the_broken_stick_corner():
    """
    Description
    -----------
    ``_soft_minimum(x, c, w) = c - w * log(1 + exp((c - x) / w))`` equals ``x`` well below
    the bend, ``c`` well above it, and ``c - w * log(2)`` exactly at it; it never exceeds
    the hard minimum ``min(x, c)``.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    x = np.array([-10.0, -2.0, 0.0, 10.0])
    z = iui_mod._soft_minimum(x, 0.0, 0.05)
    np.testing.assert_allclose(z, [-10.0, -2.0, -0.05 * np.log(2.0), 0.0], atol=1e-12)
    assert np.all(z <= np.minimum(x, 0.0))


def test_median_regression_recovers_a_line_and_flags_the_iteration_cap():
    """
    Description
    -----------
    ``_median_regression`` fits the conditional median: on a line ``y = 1 + 2 x`` with
    symmetric Laplace noise plus a few gross upward outliers (which would pull a mean
    regression up) it recovers intercept 1 and slope 2, reports the sum of absolute
    residuals at those coefficients, and reports convergence; capped at one iteration it
    reports that the solver stopped short.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 1.0, 400)
    y = 1.0 + 2.0 * x + rng.laplace(0.0, 0.05, 400)
    y[:20] += 5.0
    design = np.column_stack([np.ones_like(x), x])
    beta, loss, converged = iui_mod._median_regression(design, y, 1000)
    np.testing.assert_allclose(beta, [1.0, 2.0], atol=0.05)
    assert loss == pytest.approx(float(np.abs(y - design @ beta).sum()))
    assert converged is True
    _beta, _loss, converged_capped = iui_mod._median_regression(design, y, 1)
    assert converged_capped is False


def _bent_line_pairs(n_sessions=6, per_session=150, seed=0):
    """
    Description
    -----------
    Draws consecutive-interval pairs whose log next interval follows a bent median line:
    ``log next = a + 0.8 * softmin(log current, log 0.15 s; 0.05) + Laplace(0, 0.05)``,
    with ``a`` set so the flat level above the bend is 100 ms. Current intervals are
    log-uniform between 20 ms and 1 s, split evenly over ``n_sessions`` sessions.

    Parameters
    ----------
    n_sessions (int)
        Number of sessions.
    per_session (int)
        Pairs per session.
    seed (int)
        Seed of the generator.

    Returns
    -------
    current (np.ndarray)
        Current intervals, in seconds.
    following (np.ndarray)
        Next intervals, in seconds.
    sessions (np.ndarray)
        The session of each pair.
    """

    rng = np.random.default_rng(seed)
    n = n_sessions * per_session
    x = rng.uniform(np.log(0.02), np.log(1.0), n)
    bend, slope = np.log(0.15), 0.8
    intercept = np.log(0.1) - slope * bend
    y = intercept + slope * iui_mod._soft_minimum(x, bend, 0.05) + rng.laplace(0.0, 0.05, n)
    sessions = np.repeat(np.arange(n_sessions), per_session).astype(str)
    return np.exp(x), np.exp(y), sessions


def test_fit_serial_dependence_recovers_the_bend_slope_and_flat_level():
    """
    Description
    -----------
    Runs the real serial-dependence fit on pairs drawn from a known bent median line (bend
    150 ms, slope 0.8 below it, flat at 100 ms above it) and checks its three tables: the
    bend, slope and flat level are recovered, the bend lies inside its bootstrap interval,
    the curves table has ``n_grid`` rows spanning the grid percentiles (taken on the log
    current interval) with the bands
    bracketing the fitted curves, the per-width objectives are keyed ``loss_w_0p05`` /
    ``loss_w_0p15`` with the chosen width the smaller one, the bends table has one row per
    replicate, and the pool identity is stamped on every row.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    current, following, sessions = _bent_line_pairs()
    identity = {"sex": "male", "vocal_class": "usv", "adjacency": "filtered"}
    curves, fit, bends = fit_serial_dependence(
        current=current, following=following, pair_sessions=sessions, pool_identity=identity,
        n_knots=4, corner_widths=[0.05, 0.15], n_bootstrap=8, level=90.0,
        bend_bounds_ms=(30.0, 600.0), grid_percentiles=(5.0, 95.0), max_iter=1000, seed=0,
        n_jobs=1, n_grid=25)

    row = fit.row(0, named=True)
    assert fit.height == 1
    assert row["n_pairs"] == 900
    assert row["n_sessions"] == 6
    assert row["n_knots"] == 4
    assert row["n_bootstrap"] == 8
    assert row["level"] == 90.0
    assert row["bend_ms"] == pytest.approx(150.0, rel=0.05)
    assert row["slope"] == pytest.approx(0.8, abs=0.05)
    assert row["flat_level_ms"] == pytest.approx(100.0, rel=0.05)
    assert row["bend_low_ms"] <= row["bend_high_ms"]
    assert row["corner_width"] == (0.05 if row["loss_w_0p05"] <= row["loss_w_0p15"] else 0.15)
    assert 0 <= row["spline_not_converged"] <= 8
    assert 0 <= row["bent_not_converged"] <= 8

    assert curves.height == 25
    np.testing.assert_allclose(
        curves["current_ms"].to_numpy()[[0, -1]], 1e3 * np.exp(np.percentile(np.log(current), [5.0, 95.0])),
        rtol=1e-9)
    assert np.all(np.diff(curves["current_ms"].to_numpy()) > 0)
    assert (curves["spline_low_ms"] <= curves["spline_high_ms"]).all()
    assert (curves["bent_low_ms"] <= curves["bent_high_ms"]).all()
    # far above the bend the bent line sits at its flat level
    assert curves["bent_ms"][-1] == pytest.approx(row["flat_level_ms"], rel=0.02)

    assert bends.height == 8
    assert bends["b"].to_list() == list(range(8))
    assert ((bends["bend_ms"] >= 30.0 - 1e-6) & (bends["bend_ms"] <= 600.0 + 1e-6)).all()
    for table in (curves, fit, bends):
        for key, value in identity.items():
            assert (table[key] == value).all()


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fit_tied_peak_ladder_and_lrt_every_rung_rejected_selects_past_the_grid(monkeypatch):
    """
    Description
    -----------
    When every rung of the step-up test rejects its null, no peak count in the grid is
    kept and the selection is one past the grid's maximum (here grid ``[1, 2]`` -> 3); the
    selection message reports it. The rung test is replaced by a stand-in that always
    rejects.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        Replaces ``tied_peak_bootstrap_lrt``.

    Returns
    -------
    None
    """

    values, labels = _two_peak_intervals()

    def always_reject(**kwargs):
        """
        Description
        -----------
        Stands in for ``tied_peak_bootstrap_lrt``: returns a rung result whose corrected
        p-value is 0, so the rung rejects at any alpha.

        Parameters
        ----------
        **kwargs (dict)
            Keyword arguments the ladder passes (ignored).

        Returns
        -------
        result (dict)
            A scripted rung result.
        """
        del kwargs
        return {"lr_obs": 50.0, "design_effect": 1.0, "design_effect_raw": 1.0, "effective_n": 10.0,
                "lr_corrected": 50.0, "null_p95": 2.0, "p_value": 0.0, "p_value_corrected": 0.0,
                "negative_fraction": 0.0, "B": 2, "n_subsample": 10, "lr_null": np.zeros(2),
                "alt_model": None}

    monkeypatch.setattr(iui_mod, "tied_peak_bootstrap_lrt", always_reject)
    messages: list[str] = []
    _fits, _modes, lrt, _null, selected, alpha_eff = fit_tied_peak_ladder_and_lrt(
        values=values, session_labels=labels, pool_identity={"sex": "male"}, peak_grid=[1, 2],
        n_background=1, n_init=1, n_init_boot=1, reg_covar=1e-4, B=2, n_subsample=300,
        alpha=0.05, bonferroni=True, n_design_bootstrap=10, seed=0, n_jobs=1,
        design_bootstrap_iid_cap=400, em_max_iter=500, em_tol=1e-6,
        split_delta=0.5, mode_grid_size=16384, mode_span_scales=6.0, em_init=_EM_INIT,
        message=messages.append)

    assert selected == 3
    assert alpha_eff == pytest.approx(0.025)
    assert lrt["rejected"].to_list() == [True, True]
    assert any("3 peak(s) + 1 background" in m for m in messages)


@pytest.mark.parametrize(("spec", "match"), [
    ({"sex": "unknown", "vocal_class": "usv", "adjacency": "filtered", "fit": False},
     r"sex must be 'male' or 'female', got 'unknown' in \('unknown', 'usv', 'filtered'\)"),
    ({"sex": "male", "vocal_class": "usv", "adjacency": "loose", "fit": False},
     r"adjacency must be 'filtered' or 'strict', got 'loose' in \('male', 'usv', 'loose'\)"),
])
def test_validate_interval_pools_rejects_unknown_sex_and_adjacency(spec, match):
    """
    Description
    -----------
    A pool naming a sex other than ``male`` / ``female`` or an adjacency rule other than
    ``filtered`` / ``strict`` is refused with a ValueError naming the bad value and the
    pool identity; a valid pool of the same shape passes.

    Parameters
    ----------
    spec (dict)
        The invalid pool declaration.
    match (str)
        Regular expression the error message must match.

    Returns
    -------
    None
    """

    with pytest.raises(ValueError, match=match):
        validate_interval_pools([spec], fit_tied_model=True, fit_mixture_model=False)
    assert validate_interval_pools(
        [{"sex": "female", "vocal_class": "squeak", "adjacency": "strict", "fit": False}],
        fit_tied_model=True, fit_mixture_model=False) is None


def test_save_iui_skips_sessions_without_intervals_and_empty_pools(tmp_path, mocker, monkeypatch):
    """
    Description
    -----------
    A session whose interval computation returns nothing (an empty payload) contributes to
    no pool, and a session whose payload holds no intervals for a pool's sex contributes no
    rows to it; both are skipped without error. Here one of three sessions yields nothing,
    one yields only male intervals, so the male pool holds 3 + 2 intervals from two
    sessions and the female pool only the 2 intervals of the session that has them.

    Parameters
    ----------
    tmp_path (pathlib.Path)
        pytest-provided temporary directory.
    mocker (pytest_mock.MockerFixture)
        Patches ``write_ivi_h5``.
    monkeypatch (pytest.MonkeyPatch)
        Replaces the session resolution and the per-session interval computation.

    Returns
    -------
    None
    """

    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")
    settings = _make_settings(tmp_path)
    settings["compute_inter_usv_interval_distributions"]["session_lists"] = [str(list_file)]
    full_root = str(tmp_path / "20260101_120000")
    male_only_root = str(tmp_path / "20260102_120000")
    empty_root = str(tmp_path / "20260103_120000")
    monkeypatch.setattr(iui_mod, "_read_session_lists",
                        lambda _lists, _msg: [full_root, male_only_root, empty_root])
    monkeypatch.setattr(iui_mod, "_session_source_map", lambda _lists: {})

    def fake_compute(session_root, interval_type, exclude_noise_usvs, vocal_class=None,
                     adjacency="filtered"):
        """
        Description
        -----------
        Stands in for ``compute_session_usv_intervals``: the full session has three male
        and two female intervals, the male-only session two male intervals and no female
        ones, and the empty session returns an empty payload.

        Parameters
        ----------
        session_root (str)
            Session root selecting the payload.
        interval_type (str)
            Echoed back.
        exclude_noise_usvs (bool)
            Noise filter flag (ignored).
        vocal_class (str | None)
            Vocal-class filter (ignored).
        adjacency (str)
            Adjacency rule (ignored).

        Returns
        -------
        payload (dict)
            The synthetic interval payload, or ``{}`` for the empty session.
        """
        del exclude_noise_usvs, vocal_class, adjacency
        if session_root == empty_root:
            return {}
        female = np.array([0.3, 0.4]) if session_root == full_root else np.array([], dtype=float)
        male = np.array([0.5, 0.7, 0.9]) if session_root == full_root else np.array([0.2, 0.6])
        return {
            "male": male, "female": female,
            "n_dropped_male": 0, "n_dropped_female": 0,
            "emitter_male": np.array(["M"] * male.size, dtype=object),
            "emitter_female": np.array(["F"] * female.size, dtype=object),
            "chain_male": np.ones(male.size, dtype=np.int64),
            "chain_female": np.ones(female.size, dtype=np.int64),
            "animal_sex": {"M": "male", "F": "female"},
            "interval_type": interval_type,
        }

    monkeypatch.setattr(iui_mod, "compute_session_usv_intervals", fake_compute)
    write_mock = mocker.patch.object(iui_mod, "write_ivi_h5",
                                     return_value=Path(tmp_path / "out" / "stub.h5"))
    monkeypatch.setattr(iui_mod, "git_sha_for_provenance", lambda _p: "stub")

    InterUSVIntervalCalculator(input_parameter_dict=settings,
                               message_output=lambda *_a, **_kw: None
                               ).save_inter_usv_interval_distributions_to_file()

    per_mode = write_mock.call_args.kwargs["per_mode"]
    for mode in per_mode.values():
        intervals = mode["intervals"]
        male = intervals.filter(pls.col("sex") == "male")
        female = intervals.filter(pls.col("sex") == "female")
        assert male.height == 5
        assert sorted(male["session_id"].unique().to_list()) == ["20260101_120000", "20260102_120000"]
        assert female.height == 2
        assert female["session_id"].unique().to_list() == ["20260101_120000"]
    assert write_mock.call_args.kwargs["analysis_attrs"]["session_roots"] == [full_root, male_only_root]


def test_save_iui_small_pools_skip_the_tied_fit_and_the_sweep(tmp_path, mocker, monkeypatch):
    """
    Description
    -----------
    A fitted pool with fewer intervals than ``min_intervals_for_fitting`` is archived for
    description only: the tied ladder is not run (a message says so), the pool is not
    marked fitted and no tied tables or selection attributes are written. With
    ``fit_mixture_model`` on, a fitted pool of fewer than 2 intervals leaves the sweep and
    its LRT tables empty (None).

    Parameters
    ----------
    tmp_path (pathlib.Path)
        pytest-provided temporary directory.
    mocker (pytest_mock.MockerFixture)
        Patches ``write_ivi_h5`` and the sweep / tied fit to detect calls.
    monkeypatch (pytest.MonkeyPatch)
        Replaces the session resolution and the per-session interval computation.

    Returns
    -------
    None
    """

    list_file = tmp_path / "sessions.txt"
    list_file.write_text("/dummy/session\n")
    settings = _make_settings(tmp_path, fit_mixture_model=True, fit_tied_model=True)
    block = settings["compute_inter_usv_interval_distributions"]
    block["session_lists"] = [str(list_file)]
    block["min_intervals_for_fitting"] = 10
    sess_root = str(tmp_path / "20260101_120000")
    monkeypatch.setattr(iui_mod, "_read_session_lists", lambda _lists, _msg: [sess_root])
    monkeypatch.setattr(iui_mod, "_session_source_map", lambda _lists: {sess_root: "groupA"})
    monkeypatch.setattr(iui_mod, "compute_session_usv_intervals", lambda **kwargs: {
        "male": np.array([0.5]), "female": np.array([0.3, 0.4]),
        "n_dropped_male": 0, "n_dropped_female": 0,
        "emitter_male": np.array(["M"], dtype=object),
        "emitter_female": np.array(["F", "F"], dtype=object),
        "chain_male": np.ones(1, dtype=np.int64), "chain_female": np.ones(2, dtype=np.int64),
        "animal_sex": {"M": "male", "F": "female"}, "interval_type": kwargs["interval_type"],
    })
    tied_mock = mocker.patch.object(iui_mod, "fit_tied_peak_ladder_and_lrt")
    sweep_mock = mocker.patch.object(iui_mod, "fit_mixture_model_sweep")
    write_mock = mocker.patch.object(iui_mod, "write_ivi_h5",
                                     return_value=Path(tmp_path / "out" / "stub.h5"))
    monkeypatch.setattr(iui_mod, "git_sha_for_provenance", lambda _p: "stub")
    messages: list[str] = []

    InterUSVIntervalCalculator(input_parameter_dict=settings,
                               message_output=messages.append
                               ).save_inter_usv_interval_distributions_to_file()

    assert tied_mock.call_count == 0
    assert sweep_mock.call_count == 0
    assert sum("below min_intervals_for_fitting (10); archived for description only." in m
               for m in messages) == 2
    for mode in write_mock.call_args.kwargs["per_mode"].values():
        assert mode["tied_fits"] is None
        assert mode["peak_lrt"] is None
        assert "selected_n_peak_male_usv" not in mode["attrs"]
        assert mode["mixture_model_fits"] is None
        assert mode["bootstrap_lrt"] is None
        assert mode["bootstrap_lrt_null"] is None
        fitted = dict(zip(mode["pool_summary"]["sex"], mode["pool_summary"]["fitted"], strict=True))
        assert fitted == {"male": False, "female": False}
