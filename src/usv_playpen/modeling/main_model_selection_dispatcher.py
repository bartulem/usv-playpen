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
4. Multinomial: Selection for flat USV category probability (JAX/Soft-Hierarchy).
5. Continuous: Selection for continuous UMAP manifold coordinates (JAX/Gaussian).

Computational Strategy:
-----------------------
- Forward Stepwise Search: Implements a greedy algorithm that iteratively adds
  features based on the 1-Standard-Error (1SE) rule to prevent over-fitting.
- Decoupled Orchestration: The dispatcher handles CLI argument parsing and
  filesystem validation, while the heavy algorithmic logic is imported from
  the project's 'model_selection.py' module.
- Traceback Verbosity: Explicitly captures and prints the full stack trace upon
  failure, which is critical for debugging pre-emption or I/O issues on cluster nodes.
"""

import argparse
import traceback
from datetime import datetime
from pathlib import Path

from .model_selection import (
    bout_onset_model_selection,
    vocal_category_model_selection,
    bout_parameter_model_selection,
    multinomial_vocal_category_model_selection,
    continuous_vocal_manifold_model_selection
)


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
        Path to the modeling_settings.json configuration.

    Raises
    ------
    FileNotFoundError
        If any of the three required paths do not exist on the filesystem.
    """
    paths_to_check = [univariate_path, input_path, settings_path]

    for p in paths_to_check:
        if not Path(p).exists():
            print(f"CRITICAL ERROR: Path not found: {p}")
            raise FileNotFoundError(f"Missing required file or mount point: {p}")


def dispatch_model_selection(args: argparse.Namespace) -> None:
    """
    Routes the model selection task to the specific algorithm implementation.

    This function implements the multi-line architecture for logical branching,
    ensuring that specific hyperparameters (like target_variable for params
    or pval for continuous) are injected only where appropriate.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command-line arguments containing analysis_type and
        all necessary filesystem paths and logic flags.

    Returns
    -------
    None
        Execution state and results are saved to disk by the underlying
        orchestrator functions.
    """
    print(f"--- USV Model Selection Dispatcher | Task: {args.analysis_type.upper()} ---")
    print(f"[{datetime.now()}] Univariate Source: {Path(args.univariate_path).name}")

    # 1. Mount Point Verification
    validate_paths(
        univariate_path=args.univariate_path,
        input_path=args.input_path,
        settings_path=args.settings_path
    )

    # 2. Execution Routing with Multi-line Architecture
    try:

        if args.analysis_type == 'onset':

            bout_onset_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=args.settings_path,
                output_directory=args.output_dir,
                use_top_rank_as_anchor=args.anchor,
                p_val=args.pval
            )

        elif args.analysis_type == 'category':

            vocal_category_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=args.settings_path,
                output_directory=args.output_dir,
                use_top_rank_as_anchor=args.anchor,
                p_val=args.pval
            )

        elif args.analysis_type == 'params':

            print(f"Target Variable for Regression: {args.target_variable}")

            bout_parameter_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=args.settings_path,
                output_directory=args.output_dir,
                target_variable=args.target_variable,
                use_top_rank_as_anchor=args.anchor,
                p_val=args.pval
            )

        elif args.analysis_type == 'multinomial':

            multinomial_vocal_category_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=args.settings_path,
                output_directory=args.output_dir,
                use_top_rank_as_anchor=args.anchor,
                p_val=args.pval
            )

        elif args.analysis_type == 'continuous':

            continuous_vocal_manifold_model_selection(
                univariate_results_path=args.univariate_path,
                input_data_path=args.input_path,
                settings_path=args.settings_path,
                output_directory=args.output_dir,
                use_top_rank_as_anchor=args.anchor,
                p_val=args.pval
            )

        else:
            print(f"FATAL: Unknown analysis type: {args.analysis_type}")
            return

        print(f"[{datetime.now()}] Model selection process completed successfully.")

    except Exception as e:
        print(f"!!! CRITICAL FAILURE DURING {args.analysis_type.upper()} SELECTION: {e}")
        traceback.print_exc()


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

    parser.add_argument('--settings_path', type=str, required=True,
                        help='Path to modeling_settings.json.')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where step-wise results will be saved.')

    # Logic Flags
    parser.add_argument('--anchor', action='store_true',
                        help='If set, initializes Step 0 with the top-ranked univariate feature.')

    parser.add_argument('--pval', type=float, default=0.01,
                        help='Significance threshold for initial candidate screening.')

    # Framework Specific
    parser.add_argument('--target_variable', type=str, default='bout_durations',
                        choices=['bout_durations', 'mean_mask_complexity', 'total_mask_complexity'],
                        help='[Params Only] The continuous target variable to model.')

    args = parser.parse_args()
    dispatch_model_selection(args)
