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

import numpy as np
import polars as pls
import pytest

import usv_playpen.analyses.compute_inter_usv_interval_distributions as iui_mod
from usv_playpen.analyses.compute_inter_usv_interval_distributions import (
    InterUSVIntervalCalculator,
)


# ---------------------------------------------------------------------------
# Fixture: minimal valid input_parameter_dict
# ---------------------------------------------------------------------------


def _make_settings(tmp_path, fit_mixture_model=False):
    """Build the analyses_settings sub-block expected by the class."""
    return {
        "compute_inter_usv_interval_distributions": {
            "session_lists": [],
            "output_directory": str(tmp_path / "out"),
            "noise_col_id": "cluster",
            "noise_categories": [99],
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
            "bootstrap_lrt_alpha": 0.05,
            "bootstrap_lrt_bonferroni": True,
        }
    }


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# save_inter_usv_interval_distributions_to_file — validation paths
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# save_inter_usv_interval_distributions_to_file — happy paths (mocked)
# ---------------------------------------------------------------------------


def _mock_session_resolution(monkeypatch, tmp_path):
    """Patch _read_session_lists / _session_source_map so they return one
    synthetic session_root, and patch compute_session_usv_intervals to return
    a known interval payload for both modes."""
    sess_root = str(tmp_path / "20260101_120000")
    monkeypatch.setattr(iui_mod, "_read_session_lists",
                        lambda lists, msg: [sess_root])
    monkeypatch.setattr(iui_mod, "_session_source_map",
                        lambda lists: {sess_root: "groupA"})

    def fake_compute(session_root, interval_type, noise_col_id, noise_categories):
        if session_root != sess_root:
            return {}
        return {
            "male":   np.array([0.5, 0.7, 0.9]),
            "female": np.array([0.3, 0.4]),
            "n_dropped_male": 0 if interval_type == "s2s" else 1,
            "n_dropped_female": 0,
            "male_id": "M",
            "female_id": "F",
            "interval_type": interval_type,
        }
    monkeypatch.setattr(iui_mod, "compute_session_usv_intervals", fake_compute)
    return sess_root


def test_save_iui_writes_archive_when_fit_mixture_model_false(tmp_path, mocker, monkeypatch):
    """fit_mixture_model=False → archive contains intervals + drop_counts but NOT
    mixture_model_fits / bootstrap_lrt tables. Verifies write_ivi_h5 was invoked once
    with the expected per_mode payload shape."""
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
    # Both modes have intervals + drop_counts; mixture_model_fits / bootstrap_lrt are None
    for mode in per_mode.values():
        assert mode["intervals"].height == 5  # 3 male + 2 female per mode
        assert mode["mixture_model_fits"] is None
        assert mode["bootstrap_lrt"] is None
        assert mode["bootstrap_lrt_null"] is None


def test_save_iui_writes_archive_when_fit_mixture_model_true(tmp_path, mocker, monkeypatch):
    """fit_mixture_model=True → invokes fit_mixture_model_sweep AND bootstrap_lrt; the resulting
    archive carries the mixture_model_fits + bootstrap_lrt + bootstrap_lrt_null tables.
    Both expensive calls are mocked."""
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
        "p_value": 0.5, "null_mean": 1.0, "null_p95": 1.5, "null_max": 1.5,
    }
    monkeypatch.setattr(iui_mod, "bootstrap_lrt",
                        lambda **kw: fake_lrt_res)
    monkeypatch.setattr(iui_mod, "select_n_components_step_up_lrt",
                        lambda pair_results, alpha: 1)

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
        # Per-mode attrs include the step-up selected K values
        assert "K_selected_male" in mode["attrs"]
        assert "K_selected_female" in mode["attrs"]


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
