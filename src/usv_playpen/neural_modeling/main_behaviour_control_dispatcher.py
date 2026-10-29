"""
@author: bartulem
Run the cohort-scale behaviour control: select its features once, then fit it once per recording day.

Both stages run once per cohort, not once per unit. The per-unit nested decoding reads what this
writes and refuses to run on an artifact that disagrees with its settings.

``fit`` fits the control for one target (``--target``, default ``vocalization_settings.vocal_target``):
the torus regression for ``"manifold"``, the behavioural modelling's multinomial logistic regression
for ``"multinomial"``, on the features frozen at ``nested_vocal_decoding.reduced_model_features[target]``,
and writes it to ``behaviour_control_model_path[target]``. Every unit is decoded on both targets, so
both controls must be fitted, one run each.

``select`` runs this package's own forward selection for the torus control. The shipped feature sets
come from the behavioural modelling's selections instead (``data.behaviour_selection_result_path``
points at them), so ``select`` takes an explicit ``--output`` and never writes onto that path. After a
selection the settings' ``reduced_model_features`` must be updated by hand, so the behaviour the
neuron is measured against never changes as a side effect of a re-run.

Usage:
    python -m usv_playpen.neural_modeling.main_behaviour_control_dispatcher --stage fit --target manifold
    python -m usv_playpen.neural_modeling.main_behaviour_control_dispatcher --stage fit --target multinomial
    python -m usv_playpen.neural_modeling.main_behaviour_control_dispatcher --stage select --output <file>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import pickle
import time

import numpy as np

from .behaviour_control_model import (
    assemble_cohort,
    fit_category_control_models,
    fit_control_models,
    select_control_features,
    write_control_model,
)
from .neural_nested_decoding import reduced_model_features, vocal_target


def load_settings(path: str | None) -> dict:
    """
    Description
    -----------
    The run's settings, from the given file or the shipped one.

    Parameters
    ----------
    path (str | None)
        Settings file, or None for the packaged default.

    Returns
    -------
    settings (dict)
        The whole neural-modelling settings dict.
    """

    resolved = (pathlib.Path(path) if path is not None
                else pathlib.Path(__file__).resolve().parent.parent
                / "_parameter_settings" / "neural_modeling_settings.json")
    with resolved.open() as handle:
        return json.load(handle)


def session_directories(settings: dict, excluded_days: list) -> list:
    """
    Description
    -----------
    Every session named by the run's session lists, minus any excluded recording days.

    Parameters
    ----------
    settings (dict)
        The whole neural-modelling settings dict.
    excluded_days (list)
        Eight-character day keys to drop.

    Returns
    -------
    directories (list)
        Full session directory paths.
    """

    directories = []
    for list_file in settings["data"]["session_list_files"]:
        for line in pathlib.Path(list_file).read_text().splitlines():
            entry = line.strip()
            if entry and pathlib.Path(entry).name[:8] not in excluded_days:
                directories.append(entry)
    return directories


def main(message_output=print) -> None:
    """
    Description
    -----------
    Assemble the cohort once, then run the requested stage.

    Parameters
    ----------
    message_output (Callable)
        Where run progress goes; the module convention, so a caller can capture it.

    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("select", "fit"), required=True)
    parser.add_argument("--settings", default=None, help="settings file; default is the shipped one")
    parser.add_argument("--output", default=None, help="destination; default comes from settings")
    parser.add_argument("--design-cache", default=None,
                        help="where the assembled design is written for the workers to memory-map")
    parser.add_argument("--n-workers", type=int, default=6)
    parser.add_argument("--exclude-days", nargs="*", default=[],
                        help="recording days to leave out entirely, as eight-character keys")
    parser.add_argument("--target", choices=("manifold", "multinomial"), default=None,
                        help="which target's control to select or fit; default is "
                             "vocalization_settings.vocal_target")
    arguments = parser.parse_args()

    settings = load_settings(arguments.settings)
    # Every unit is decoded on both targets, so both controls are fitted, one run each; the settings
    # are not edited, so the artifact's settings hash matches the shipped file.
    target = arguments.target if arguments.target is not None else vocal_target(settings)
    if arguments.stage == "select" and target != "manifold":
        msg = "the select stage runs the torus control's selection; set vocal_target to 'manifold'."
        raise ValueError(msg)
    if arguments.stage == "select" and arguments.output is None:
        msg = ("the select stage needs --output: data.behaviour_selection_result_path names the "
               "behavioural modelling's selection, which must not be overwritten.")
        raise ValueError(msg)
    # History length of the behaviour design, in tracking frames (600 = 4 s at 150 fps); recorded in the
    # artifact, so the per-unit nested decoding reads it back from there rather than from settings.
    n_lags = int(settings["nested_vocal_decoding"]["behaviour_control_n_lags"])
    # Resolve the feature set before the (slow) assembly, so a missing or drifted selection artifact
    # fails early. The select stage tolerates a missing file, since it is about to write one, and uses
    # the frozen list as its independent reference.
    features = (reduced_model_features(settings, target) if arguments.stage == "fit"
                else reduced_model_features(settings, target, require_selection_file=False))

    directories = session_directories(settings, list(arguments.exclude_days))
    message_output(f"{len(directories)} sessions listed"
          f"{f', excluding {list(arguments.exclude_days)}' if arguments.exclude_days else ''}",
          flush=True)

    started = time.time()
    cohort = assemble_cohort(directories, settings, n_lags, arguments.n_workers)
    for session_id, reason in cohort["skipped"].items():
        message_output(f"  skipped {session_id}: {reason}", flush=True)
    message_output(f"{len(cohort['session_ids'])} sessions, {cohort['positions'].shape[0]:,} USVs, "
          f"{len(cohort['feature_names'])} features ({time.time() - started:.0f} s)", flush=True)

    if arguments.stage == "select":
        cache = pathlib.Path(arguments.design_cache if arguments.design_cache is not None
                             else "behaviour_control_design.npy")
        np.save(cache, cohort["design"])
        # The frozen list is scored alongside as an independent reference; the gap between the two
        # bounds the selection's optimism.
        selection = select_control_features(cohort, settings, n_lags, arguments.n_workers,
                                            str(cache), external_reference=features)
        destination = pathlib.Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            pickle.dump(selection, handle)
        message_output(f"\nSELECTED {selection['selected_features']}", flush=True)
        for step in selection["per_step"]:
            message_output(f"    {step['feature']:<24} paired gain {step['paired_gain']:+.5f} "
                  f"+- {step['standard_error']:.5f}", flush=True)
        if selection["rejected"] is not None:
            message_output(f"    rejected {selection['rejected']['feature']} at "
                  f"{selection['rejected']['paired_gain']:+.5f} "
                  f"+- {selection['rejected']['standard_error']:.5f}", flush=True)
        message_output(f"  pooled {selection['pooled_score']:+.5f}, no behaviour "
              f"{selection['no_behaviour_score']:+.5f}", flush=True)
        if selection["external_reference"] is not None:
            message_output(f"  independent reference {selection['external_reference']['features']} pooled "
                  f"{selection['external_reference']['pooled']:+.5f} -- the gap to the selected model "
                  f"BOUNDS the selection's optimism", flush=True)
        message_output(f"  -> {destination}", flush=True)
        message_output("  UPDATE nested_vocal_decoding.reduced_model_features to this list "
              "before fitting; the change must be deliberate.", flush=True)
        return

    destination = (arguments.output if arguments.output is not None
                   else settings["nested_vocal_decoding"]["behaviour_control_model_path"][target])
    if target == "multinomial":
        artifact = fit_category_control_models(cohort, features, settings, n_lags)
        write_control_model(artifact, destination)
        message_output(f"\nFITTED {target} {artifact['features']} over {len(artifact['coefficients'])} "
                       f"held-out days", flush=True)
        message_output(f"  macro AUROC {artifact['held_out_auc']:.4f} (chance 0.5)   log-score gain over "
                       f"training frequencies {artifact['held_out_log_score_gain']:+.5f}", flush=True)
        for day, scores in sorted(artifact["per_day"].items()):
            message_output(f"    {day}  n={scores['n_vocalizations']:>6,}  AUROC {scores['auc']:.4f}  "
                           f"{scores['log_score_gain']:+.5f}  "
                           f"{'converged' if artifact['coefficients'][day]['converged'] else 'NOT CONVERGED'}",
                           flush=True)
        message_output(f"  -> {destination}", flush=True)
        return

    artifact = fit_control_models(cohort, features, settings, n_lags)
    write_control_model(artifact, destination)
    message_output(f"\nFITTED {artifact['features']} over {len(artifact['coefficients'])} held-out days",
          flush=True)
    message_output(f"  behaviour {artifact['held_out_score']:+.5f}   no behaviour "
          f"{artifact['no_behaviour_score']:+.5f}   difference "
          f"{artifact['held_out_score'] - artifact['no_behaviour_score']:+.5f}", flush=True)
    for day, scores in sorted(artifact["per_day"].items()):
        message_output(f"    {day}  n={scores['n_vocalizations']:>6,}  "
              f"{scores['behaviour'] - scores['no_behaviour']:+.5f}", flush=True)
    message_output(f"  -> {destination}", flush=True)


if __name__ == "__main__":
    main()
