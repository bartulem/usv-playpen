"""
@author: bartulem
Consolidated HPC Dispatcher for Forward Stepwise Model Selection.

This script acts as the centralized execution engine for identifying the minimally
sufficient behavioral feature sets across all USV modeling frameworks. It replaces
the individual 'run_' selection scripts with a single entry point that manages
path validation, error reporting, and specific argument routing for:

1. Vocal Onset: Selection for binary onset prediction (Logistic/GAM).
2. Vocal Category: Selection for One-vs-Rest classification (Logistic/GAM).
3. Vocal Params: Selection for continuous bout characteristics (Gamma Regression).
4. Multinomial: Selection for the joint USV category probabilities (JAX multinomial
   logistic regression).
5. Continuous: Selection for continuous acoustic-manifold coordinates (JAX; the
   Huber coordinate regression on euclidean, the closed-form torus-embedding
   regression on the torus).

Computational Strategy:
-----------------------
- Forward Stepwise Search: Implements a greedy algorithm that iteratively adds
  features, accepting a step only when the candidate beats the incumbent on the
  PAIRED per-fold improvement: by the 1-Standard-Error (1SE) rule for the onset,
  category, params and multinomial targets, by a fold-bootstrap confidence
  interval on the paired margin for the continuous manifold target.
- Decoupled Orchestration: The dispatcher handles CLI argument parsing and
  filesystem validation, while the heavy algorithmic logic is imported from
  the project's 'model_selection.py' module.
- Traceback Verbosity: Explicitly captures and prints the full stack trace upon
  failure, which is critical for debugging pre-emption or I/O issues on cluster nodes.
- Settings-Driven Knobs: the screening significance level
  (``model_params.selection_p_val``) and the bout-parameter regression target
  (``model_params.model_target_variable``) are read from ``modeling_settings.json``,
  the same file the selectors read, so the CLI cannot silently override them.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from .model_selection import (
    bout_parameter_model_selection,
    continuous_vocal_manifold_model_selection,
    multinomial_vocal_category_model_selection,
    vocal_category_model_selection,
    vocal_onset_model_selection,
)

# The settings JSON ships with the package at
# `<pkg>/_parameter_settings/modeling_settings.json`; module-level so tests can repoint it.
_SETTINGS_PATH = Path(__file__).resolve().parent.parent / '_parameter_settings' / 'modeling_settings.json'


def validate_paths(
        univariate_path: str,
        input_path: str,
        settings_path: str
) -> None:
    """
    Ensures all required data and configuration files are accessible.

    This is a critical guardrail for cluster environments where remote mount
    points (e.g., /mnt/cup/...) may occasionally drop or become unresponsive.

    Parameters
    ----------
    univariate_path : str
        Path to the results of the univariate screening pass.
    input_path : str
        Path to the raw feature data .pkl file.
    settings_path : str
        Path to the modeling_settings.json configuration (auto-resolved
        by the dispatcher relative to this module, not taken as CLI input).

    Returns
    -------
    None

    Raises
    ------
    FileNotFoundError
        If any of the three required paths do not exist on the filesystem.
    """
    paths_to_check = [univariate_path, input_path, settings_path]

    for p in paths_to_check:
        if not Path(p).exists():
            print(f"CRITICAL ERROR: Path not found: {p}")
            msg = f"Missing required file or mount point: {p}"
            raise FileNotFoundError(msg)


def dispatch_model_selection(args: argparse.Namespace) -> int:
    """
    Routes the model selection task to the specific algorithm implementation.

    This function implements the multi-line architecture for logical branching,
    ensuring that specific hyperparameters (like target_variable for params)
    are injected only where appropriate. The screening significance level
    ``p_val`` and the params ``target_variable`` are read from the settings
    file (``model_params.selection_p_val`` / ``model_params.model_target_variable``),
    never from the command line, so the selection always runs with the
    configured values.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments containing analysis_type, all
        necessary filesystem paths and the ``anchor`` flag.

    Returns
    -------
    exit_code : int
        0 on success, 1 on an unknown analysis type or a failed selection;
        results are saved to disk by the underlying orchestrator functions.
    """
    print(f"--- USV Model Selection Dispatcher | Task: {args.analysis_type.upper()} ---")
    print(f"[{datetime.now()}] Univariate Source: {Path(args.univariate_path).name}")

    # Auto-resolve the settings JSON relative to this module — it ships
    # with the package at `<pkg>/_parameter_settings/modeling_settings.json`.
    settings_path = str(_SETTINGS_PATH)

    # 1. Mount Point Verification
    validate_paths(
        univariate_path=args.univariate_path,
        input_path=args.input_path,
        settings_path=settings_path
    )

    # The selection knobs come from the same settings file the selectors read.
    with Path(settings_path).open() as settings_file:
        model_params = json.load(settings_file)['model_params']
    p_val = model_params['selection_p_val']
    target_variable = model_params['model_target_variable']
    print(f"Screening significance level (model_params.selection_p_val): {p_val}")

    # 2. Execution Routing with Multi-line Architecture
    try:

        if args.analysis_type == 'onset':

            vocal_onset_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=settings_path,
                output_directory=args.output_dir,
                use_top_rank_as_anchor=args.anchor,
                p_val=p_val
            )

        elif args.analysis_type == 'category':

            vocal_category_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=settings_path,
                output_directory=args.output_dir,
                use_top_rank_as_anchor=args.anchor,
                p_val=p_val
            )

        elif args.analysis_type == 'params':

            print(f"Target Variable for Regression: {target_variable}")

            bout_parameter_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=settings_path,
                output_directory=args.output_dir,
                target_variable=target_variable,
                use_top_rank_as_anchor=args.anchor,
                p_val=p_val
            )

        elif args.analysis_type == 'multinomial':

            multinomial_vocal_category_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=settings_path,
                output_directory=args.output_dir,
                use_top_rank_as_anchor=args.anchor,
                p_val=p_val
            )

        elif args.analysis_type == 'continuous':

            continuous_vocal_manifold_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=settings_path,
                output_directory=args.output_dir,
                use_top_rank_as_anchor=args.anchor,
                p_val=p_val
            )

        else:
            print(f"FATAL: Unknown analysis type: {args.analysis_type}")
            return 1

        print(f"[{datetime.now()}] Model selection process completed successfully.")
        return 0

    except Exception as e:
        # Catch + log the failure here (so callers/tests still see the verbose
        # traceback and 'CRITICAL FAILURE' rather than an opaque crash), but
        # report it via a NON-ZERO exit code so `__main__`'s sys.exit propagates
        # it: the cluster script's `set -e` / `--mail-type=FAIL` only detect a
        # crashed selection run if the process exits non-zero.
        print(f"!!! CRITICAL FAILURE DURING {args.analysis_type.upper()} SELECTION: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidated USV Model Selection Dispatcher")

    # Core Arguments
    parser.add_argument('--analysis_type', required=True,
                        choices=['onset', 'category', 'params', 'multinomial', 'continuous'],
                        help="The type of model selection framework to execute.")

    parser.add_argument('--univariate_path', type=str, required=True,
                        help='Path to univariate pkl results (source for ranking).')

    parser.add_argument('--input_path', type=str, required=True,
                        help='Path to raw modeling input pkl.')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where step-wise results will be saved.')

    # Logic Flags
    parser.add_argument('--anchor', action='store_true',
                        help='If set, initializes Step 0 with the top-ranked univariate feature.')

    args = parser.parse_args()
    sys.exit(dispatch_model_selection(args))
