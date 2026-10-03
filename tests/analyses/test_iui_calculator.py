"""
@author: bartulem
Mock-based tests for the InterUSVIntervalCalculator orchestration class.

The class drives a per-mode loop over compute_session_usv_intervals, optionally
invokes the mixture-model sweep + bootstrap LRT, and writes a single HDF5 archive. We
mock every heavy compute helper (compute_session_usv_intervals,
fit_mixture_model_sweep, bootstrap_lrt, write_ivi_h5) so the orchestration can be
exercised end-to-end against synthetic disk fixtures.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import polars as pls
import pytest

import usv_playpen.analyses.compute_inter_usv_interval_distributions as iui_mod
from usv_playpen.analyses._usv_io import extract_animal_sexes, load_and_filter_usv_data
from usv_playpen.analyses.compute_inter_usv_interval_distributions import (
    InterUSVIntervalCalculator,
    compute_session_usv_intervals,
    fit_mixture_model_sweep,
    fit_tied_peak_ladder_and_lrt,
)
from usv_playpen.analyses.usv_interval_archive import _polars_to_h5, _h5_to_polars


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

    assert back["i"].dtype == pls.Float64 and back["b"].dtype == pls.Float64
    i_vals = back["i"].to_list()
    b_vals = back["b"].to_list()
    assert i_vals[0] == 1.0 and i_vals[2] == 3.0 and np.isnan(i_vals[1])
    assert b_vals[0] == 1.0 and b_vals[2] == 0.0 and np.isnan(b_vals[1])
    assert back["f"].to_list() == [1.5, 2.5, 3.5]


# Fixture: minimal valid input_parameter_dict


def _make_settings(tmp_path, fit_mixture_model=False, fit_tied_model=False, fit_serial_dependence=False):
    """Build the analyses_settings sub-block expected by the class, including the pool spec:
    the male's ultrasonic calls (fitted) and the female's squeaks (described only)."""
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
                {"sex": "male", "call_type": "usv", "adjacency": "filtered", "fit": True},
                {"sex": "female", "call_type": "squeak", "adjacency": "strict", "fit": False},
            ],
            "min_intervals_for_fitting": 1,
            "fit_tied_model": fit_tied_model,
            "tied_n_background": 2,
            "tied_peak_grid": [1, 2, 3],
            "tied_n_init": 1,
            "tied_n_init_boot": 1,
            "design_bootstrap": 10,
            "fit_serial_dependence": fit_serial_dependence,
            "serial_dependence_n_knots": 4,
            "serial_dependence_corner_widths": [0.05, 0.15],
            "serial_dependence_bootstrap": 5,
            "serial_dependence_level": 99.0,
            "serial_dependence_bend_bounds_ms": [30.0, 600.0],
            "serial_dependence_grid_percentiles": [1.0, 99.0],
            "serial_dependence_max_iter": 1000,
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


def test_save_iui_zero_resolved_sessions_logs_skip(tmp_path, mocker, monkeypatch):
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
                        lambda lists, msg: [])
    monkeypatch.setattr(iui_mod, "_session_source_map", lambda lists: {})

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
                        lambda lists, msg: [sess_root])
    monkeypatch.setattr(iui_mod, "_session_source_map",
                        lambda lists: {sess_root: "groupA"})

    def fake_compute(session_root, interval_type, exclude_noise_usvs, call_type=None,
                     adjacency="filtered"):
        if session_root != sess_root:
            return {}
        return {
            "male":   np.array([0.5, 0.7, 0.9]),
            "female": np.array([0.3, 0.4]),
            "n_dropped_male": 0 if interval_type == "s2s" else 1,
            "n_dropped_female": 0,
            "emitter_male": np.array(["M"] * 3, dtype=object),
            "emitter_female": np.array(["F"] * 2, dtype=object),
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
        assert set(mode["intervals"]["call_type"].unique()) == {"usv", "squeak"}
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
                        lambda **kw: fake_sweep)

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
        lrt_calls.append(kw)
        return fake_lrt_res
    monkeypatch.setattr(iui_mod, "bootstrap_lrt", fake_bootstrap_lrt)
    selector_inputs: list[dict] = []

    def fake_select(pair_results, alpha):
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
    assert lrt_calls and all(list(kw["session_labels"]) == ["20260101_120000"] * 3 for kw in lrt_calls)
    assert all(res["p_value"] == 0.5 for inputs in selector_inputs for res in inputs.values())


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
        fitted = dict(zip(mode["pool_summary"]["sex"], mode["pool_summary"]["fitted"]))
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
            assert mode[table] is not None and set(mode[table]["sex"].to_list()) == {"male"}
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
    identity = {"sex": "male", "call_type": "usv"}
    messages = []
    fits, modes, lrt, null, selected, alpha_eff = fit_tied_peak_ladder_and_lrt(
        values=values, session_labels=labels, pool_identity=identity, peak_grid=[1, 2],
        n_background=1, n_init=1, n_init_boot=1, reg_covar=1e-4, B=19, n_subsample=300,
        alpha=0.01, bonferroni=True, n_design_bootstrap=50, seed=0, n_jobs=1,
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
    assert lrt.height == 2 and lrt["n_peak_null"].to_list() == [1, 2]
    assert (lrt["design_effect"] >= 1.0).all()
    assert np.allclose(lrt["alpha_used"].to_numpy(), 0.005)
    assert alpha_eff == pytest.approx(0.005)
    assert null.height == 2 * 19
    for table in (fits, modes, lrt, null):
        assert (table["sex"] == "male").all() and (table["call_type"] == "usv").all()
    assert selected in (1, 2, 3)
    assert any("step-up selection" in m for m in messages)


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
        message=lambda _m: None)

    assert alpha_eff == pytest.approx(0.05)
    assert sorted(fits["n_peak"].unique().to_list()) == [1, 2]
    assert lrt.height == 1 and null.height == 9
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
        cv_n_folds=5, cv_n_init=1, mixture_model_n_init=1, model_class=model_class)

    male = table.filter(pls.col("sex") == "male")
    assert male.height == 2 and sorted(male["n_comp"].to_list()) == [1, 2]
    assert (table["model_class"] == model_class).all()
    assert np.isfinite(table["bic"].to_numpy()).all()
    male_cv = male["cv_neg_loglik"].to_numpy()
    female_cv = table.filter(pls.col("sex") == "female")["cv_neg_loglik"].to_numpy()
    assert np.isfinite(male_cv).all()
    assert np.isnan(female_cv).all()


def test_fit_mixture_model_sweep_rejects_unknown_model_class():
    """An unknown ``model_class`` is a ValueError before any fitting."""

    with pytest.raises(ValueError, match="model_class"):
        fit_mixture_model_sweep(intervals_by_key={"male": np.array([0.1, 0.2])}, n_components_min=1,
                                n_components_max=2, n_repeats=1, max_modes_reported=1,
                                random_seed_base=0, model_class="lognormal")


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


def test_extract_animal_sexes_invalid_sex_raises(tmp_path):
    """A recorded sex other than male / female is a ValueError, not silently mapped."""

    _write_metadata(tmp_path, [{"subject_id": "A_0", "sex": "unknown"}])
    with pytest.raises(ValueError, match="expected male or female"):
        extract_animal_sexes(str(tmp_path), ["A_0"])


def _write_summary(root: Path, with_vocal_flags: bool) -> None:
    """Writes a five-row ``audio/<session>_usv_summary.csv`` (pure usv, pure squeak, pure usv, both
    flags true and an unscored null), with or without the ``usv`` / ``squeak`` booleans."""

    audio = root / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    frame = pls.DataFrame({"start": [1.0, 2.0, 3.0, 4.0, 5.0], "stop": [1.1, 2.1, 3.1, 4.1, 5.1],
                           "noise": [False, False, False, False, False]})
    if with_vocal_flags:
        frame = frame.with_columns(
            pls.Series("usv", [True, False, True, True, None], dtype=pls.Boolean),
            pls.Series("squeak", [False, True, False, True, None], dtype=pls.Boolean),
        )
    frame.write_csv(str(audio / f"{root.name}_usv_summary.csv"))


def test_load_and_filter_usv_data_call_type_filters(tmp_path):
    """``call_type='usv'`` keeps the pure USVs (usv true, squeak false) and ``'squeak'`` the pure
    squeaks; a segment with both flags true (usv true is not enough) and an unscored (null) row
    belong to neither, and None keeps every row."""

    _write_summary(tmp_path, with_vocal_flags=True)
    usv = load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, call_type="usv")
    squeak = load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, call_type="squeak")
    every = load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, call_type=None)
    assert usv["start"].to_list() == [1.0, 3.0]
    assert squeak["start"].to_list() == [2.0]
    assert every.height == 5


def test_load_and_filter_usv_data_rejects_unknown_call_type(tmp_path):
    """An unknown ``call_type`` is a ValueError."""

    _write_summary(tmp_path, with_vocal_flags=True)
    with pytest.raises(ValueError, match="call_type"):
        load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, call_type="bark")


def test_load_and_filter_usv_data_without_vocal_flags_raises(tmp_path):
    """Asking for a call type on a summary the call classifier never ran on is a KeyError that
    says which step to run, rather than treating every row as a USV."""

    _write_summary(tmp_path, with_vocal_flags=False)
    with pytest.raises(KeyError, match="detect-usv-squeaks"):
        load_and_filter_usv_data(str(tmp_path), frame_rate=150.0, exclude_noise_usvs=True, call_type="usv")


def _patch_session_identity(monkeypatch) -> None:
    """Replaces the metadata and sex readers of the interval module with a fixed two-animal
    session (male ``M``, female ``F``, 150 fps), so only the USV summary is read from disk."""

    monkeypatch.setattr(iui_mod, "extract_session_metadata",
                        lambda _root: {"male_id": "M", "female_id": "F", "frame_rate": 150.0})
    monkeypatch.setattr(iui_mod, "extract_animal_sexes", lambda _root, _ids: {"M": "male", "F": "female"})


def test_compute_session_usv_intervals_both_is_neither_usv_nor_squeak(tmp_path, monkeypatch):
    """A segment with both flags true (usv and squeak) is left out of the squeak sequence: under ``'filtered'`` it is removed
    before pairing (the squeaks either side of it form one interval), under ``'strict'`` it stays
    in the record as a non-target call and breaks the pairs it sits between. It never enters the
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
    }).write_csv(str(audio / f"{tmp_path.name}_usv_summary.csv"))

    filtered = compute_session_usv_intervals(str(tmp_path), "s2s", exclude_noise_usvs=True,
                                             call_type="squeak", adjacency="filtered")
    strict = compute_session_usv_intervals(str(tmp_path), "s2s", exclude_noise_usvs=True,
                                           call_type="squeak", adjacency="strict")
    usv = compute_session_usv_intervals(str(tmp_path), "s2s", exclude_noise_usvs=True,
                                        call_type="usv", adjacency="filtered")
    np.testing.assert_allclose(filtered["female"], [2.0, 1.0])
    np.testing.assert_allclose(strict["female"], [1.0])
    assert usv["male"].size == 0 and usv["female"].size == 0
