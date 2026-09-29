"""
@author: bartulem
Run the cohort-scale behaviour control: select its features once, then fit it once per recording day.

Both stages are cohort-level and run RARELY -- once per cohort, not once per unit -- which is the whole
point. The per-unit nested decoding then READS what this writes, and refuses to run on an artifact that
disagrees with its settings.

Order matters. ``select`` writes the feature set, ``fit`` writes the model; the settings'
``reduced_model_features`` must be updated to the selected list DELIBERATELY in between, because that
list is the behaviour the neuron is measured against, and it should never change as a side
effect of a re-run.

Usage:
    python -m usv_playpen.neural_modeling.main_behaviour_control_dispatcher --stage select
    python -m usv_playpen.neural_modeling.main_behaviour_control_dispatcher --stage fit
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
    fit_control_models,
    select_control_features,
    write_control_model,
)
from .neural_nested_decoding import reduced_model_features

N_LAGS = 600


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
    arguments = parser.parse_args()

    settings = load_settings(arguments.settings)
    # Resolve the feature set BEFORE assembling anything. Both stages need it and the assembly costs
    # minutes, so a missing or drifted selection artifact must fail in a second rather than after it.
    # The select stage tolerates a missing file -- it is about to write one -- and uses whatever is
    # frozen as its independent reference.
    features = (reduced_model_features(settings) if arguments.stage == "fit"
                else reduced_model_features(settings, require_selection_file=False))

    directories = session_directories(settings, list(arguments.exclude_days))
    message_output(f"{len(directories)} sessions listed"
          f"{f', excluding {list(arguments.exclude_days)}' if arguments.exclude_days else ''}",
          flush=True)

    started = time.time()
    cohort = assemble_cohort(directories, settings, N_LAGS, arguments.n_workers)
    for session_id, reason in cohort["skipped"].items():
        message_output(f"  skipped {session_id}: {reason}", flush=True)
    message_output(f"{len(cohort['session_ids'])} sessions, {cohort['positions'].shape[0]:,} calls, "
          f"{len(cohort['feature_names'])} features ({time.time() - started:.0f} s)", flush=True)

    if arguments.stage == "select":
        cache = pathlib.Path(arguments.design_cache if arguments.design_cache is not None
                             else "behaviour_control_design.npy")
        np.save(cache, cohort["design"])
        # The frozen list is scored alongside as an INDEPENDENT reference: the selection sees every
        # day, including days units are later scored on, and the gap between the two bounds that
        # optimism.
        selection = select_control_features(cohort, settings, N_LAGS, arguments.n_workers,
                                            str(cache), external_reference=features)
        destination = pathlib.Path(arguments.output if arguments.output is not None
                                   else settings["data"]["behaviour_selection_result_path"])
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
        message_output("  UPDATE nested_vocal_manifold_position_decoding.reduced_model_features to this list "
              "before fitting; the change must be deliberate.", flush=True)
        return

    artifact = fit_control_models(cohort, features, settings, N_LAGS)
    destination = (arguments.output if arguments.output is not None
                   else settings["nested_vocal_manifold_position_decoding"]
                   ["behaviour_control_model_path"])
    write_control_model(artifact, destination)
    message_output(f"\nFITTED {artifact['features']} over {len(artifact['coefficients'])} held-out days",
          flush=True)
    message_output(f"  behaviour {artifact['held_out_score']:+.5f}   no behaviour "
          f"{artifact['no_behaviour_score']:+.5f}   difference "
          f"{artifact['held_out_score'] - artifact['no_behaviour_score']:+.5f}", flush=True)
    for day, scores in sorted(artifact["per_day"].items()):
        message_output(f"    {day}  n={scores['n_calls']:>6,}  "
              f"{scores['behaviour'] - scores['no_behaviour']:+.5f}", flush=True)
    message_output(f"  -> {destination}", flush=True)


if __name__ == "__main__":
    main()
