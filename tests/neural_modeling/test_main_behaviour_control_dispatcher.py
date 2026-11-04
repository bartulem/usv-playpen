"""
@author: bartulem

Unit tests for ``usv_playpen.neural_modeling.main_behaviour_control_dispatcher``: the command line that
fits the cohort behaviour controls. The cohort assembly and the fits are stubbed (they are tested in
``test_behaviour_control_model``); what is tested here is the routing: which assembler and which fit
each target uses, where the artifact is written, what it records about the run, and which requests are
refused before any slow work starts.
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pytest

from usv_playpen.neural_modeling import (
    main_behaviour_control_dispatcher as control_dispatcher,
)


def _install(monkeypatch, calls: list) -> None:
    """
    Description
    -----------
    Replace the slow parts of the dispatcher with recorders: feature resolution, the session list, the
    history length, both cohort assemblers and the three fits.

    Parameters
    ----------
    monkeypatch (pytest.MonkeyPatch)
        Installs the stubs.
    calls (list)
        Collects ``(name, args)`` for every stub reached, in order.

    Returns
    -------
    None
    """

    def record(name, result):
        """
        Description
        -----------
        A stand-in that records its call and returns ``result``.

        Parameters
        ----------
        name (str)
            The stub's name in ``calls``.
        result (object)
            What the stub returns.

        Returns
        -------
        stub (Callable)
            The recording stand-in.
        """
        def stub(*args, **kwargs):
            """Record the call and return the fixed result."""
            calls.append((name, args, kwargs))
            return result
        return stub

    onset_cohort = {"labels": np.array([1, 0, 0]), "session_ids": ["s0"], "feature_names": ["f"], "skipped": {}}
    content_cohort = {"positions": np.zeros((3, 2)), "session_ids": ["s0"], "feature_names": ["f"],
                      "skipped": {"s9": "0 emitter vocalizations"}}
    per_day = {"20250101": {"n_vocalizations": 3, "n_tiles": 9, "auc": 0.9, "log_score_gain": 0.1,
                            "behaviour": 0.2, "no_behaviour": 0.1}}
    coefficients = {"20250101": {"converged": True}}
    classifier = {"features": ["f"], "coefficients": coefficients, "per_day": per_day,
                  "held_out_auc": 0.9, "held_out_log_score_gain": 0.1}
    monkeypatch.setattr(control_dispatcher, "reduced_model_features", record("features", ["f"]))
    monkeypatch.setattr(control_dispatcher, "session_directories", record("directories", ["/data/s0"]))
    monkeypatch.setattr(control_dispatcher, "behaviour_history_lags", record("lags", 600))
    monkeypatch.setattr(control_dispatcher, "assemble_onset_cohort", record("assemble_onset", onset_cohort))
    monkeypatch.setattr(control_dispatcher, "assemble_cohort", record("assemble", content_cohort))
    monkeypatch.setattr(control_dispatcher, "fit_onset_control_models", record("fit_onset", dict(classifier)))
    monkeypatch.setattr(control_dispatcher, "fit_category_control_models", record("fit_category", dict(classifier)))
    monkeypatch.setattr(control_dispatcher, "fit_control_models",
                        record("fit_manifold", {"features": ["f"], "coefficients": coefficients, "per_day": per_day,
                                                "held_out_score": 0.2, "no_behaviour_score": 0.1}))


class TestFitStage:
    """``--stage fit`` routes each target to its own assembler and fit and writes one artifact."""

    @pytest.mark.parametrize(("target", "assembler", "fit"), [("onset", "assemble_onset", "fit_onset"),
                                                              ("multinomial", "assemble", "fit_category"),
                                                              ("manifold", "assemble", "fit_manifold")])
    def test_each_target_uses_its_own_assembly_and_fit_and_records_the_excluded_days(
            self, tmp_path, monkeypatch, target, assembler, fit):
        """The artifact lands at ``--output``, carries the days left out, and only the target's own
        assembler and fit run."""
        calls = []
        _install(monkeypatch, calls)
        destination = tmp_path / f"{target}.pkl"
        monkeypatch.setattr(sys, "argv", ["dispatcher", "--stage", "fit", "--target", target, "--output",
                                          str(destination), "--exclude-days", "20250102", "20250103"])
        control_dispatcher.main(message_output=lambda *_args, **_kwargs: None)
        names = [name for name, *_rest in calls]
        assert assembler in names
        assert fit in names
        assert {"assemble_onset", "assemble", "fit_onset", "fit_category", "fit_manifold"} & set(names) == {
            assembler, fit}
        with destination.open("rb") as handle:
            artifact = pickle.load(handle)
        assert artifact["excluded_days"] == ["20250102", "20250103"]
        # the excluded days reach the session list before anything is assembled
        assert next(args for name, args, _kwargs in calls if name == "directories")[1] == ["20250102", "20250103"]

    def test_without_output_the_artifact_goes_to_the_settings_path(self, tmp_path, monkeypatch):
        """With no ``--output`` the control is written where the per-unit side reads it from."""
        calls = []
        _install(monkeypatch, calls)
        settings = control_dispatcher.load_settings(None)
        settings["paths"]["behaviour_control_model_path"]["onset"] = str(tmp_path / "onset_control.pkl")
        monkeypatch.setattr(control_dispatcher, "load_settings", lambda _path: settings)
        monkeypatch.setattr(sys, "argv", ["dispatcher", "--stage", "fit", "--target", "onset"])
        control_dispatcher.main(message_output=lambda *_args, **_kwargs: None)
        assert (tmp_path / "onset_control.pkl").exists()


class TestRefusals:
    """Requests the select stage cannot honour are refused before any assembly."""

    def test_select_needs_an_output(self, monkeypatch):
        """The behavioural selection named in the settings must never be overwritten by a re-run."""
        calls = []
        _install(monkeypatch, calls)
        monkeypatch.setattr(sys, "argv", ["dispatcher", "--stage", "select", "--target", "manifold"])
        with pytest.raises(ValueError, match="needs --output"):
            control_dispatcher.main(message_output=lambda *_args, **_kwargs: None)
        assert calls == []

    def test_select_runs_only_the_torus_control(self, tmp_path, monkeypatch):
        """This package's own selection exists only for the torus control."""
        calls = []
        _install(monkeypatch, calls)
        monkeypatch.setattr(sys, "argv", ["dispatcher", "--stage", "select", "--target", "onset",
                                          "--output", str(tmp_path / "x.pkl")])
        with pytest.raises(ValueError, match="torus control's selection"):
            control_dispatcher.main(message_output=lambda *_args, **_kwargs: None)
        assert calls == []
