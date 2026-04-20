"""
@author: bartulem
Consolidated HPC Dispatcher for Univariate USV Modeling.

This script serves as the centralized entry point for all univariate behavioral-to-vocal
modeling tasks within the research pipeline. It is designed to be executed as a
distributed job array on an HPC cluster (e.g., SLURM), where each process handles
a single behavioral feature from the 'Feature Zoo'.

The dispatcher manages five distinct analysis frameworks:
1.  Vocal Onset: Binary prediction (Logistic/GAM) of bout initiation.
2.  Vocal Category: One-vs-Rest classification of specific USV types.
3.  Vocal Params: Gamma-regression of continuous bout duration and complexity.
4.  Multinomial: JAX-accelerated flat classification of the 5-6 USV repertoire.
5.  Continuous: Bivariate Gaussian modeling of UMAP manifold coordinates.

Computational & Structural Features:
------------------------------------
- Memory Guarding: Implements a two-phase loading strategy. For JAX/GPU tasks,
  it loads only dictionary keys to perform feature-to-index mapping before
  releasing CPU memory, preventing Out-of-Memory (OOM) errors during heavy
  GPU allocation.
- Atomic Plotting Lock: Uses a file-based signaling mechanism ('.basis_plotted')
  to ensure that basis set verification plots are generated exactly once per
  batch run, preventing race conditions and I/O collisions across cluster nodes.
- Headless Stability: Explicitly configures the Matplotlib 'Agg' backend
  before any pipeline imports to ensure compatibility with non-interactive
  remote compute environments.
- Model Symmetry: Standardizes the 'Actual vs. Null' experimental design across
  all analysis types, ensuring that p-values and information gain metrics
  remain statistically comparable across the entire project.
"""

import argparse
import pickle
import os
import pathlib
import json
import gc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from .modeling_vocal_onsets import VocalOnsetModelingPipeline
from .modeling_vocal_categories_binomial import VocalCategoryModelingPipeline
from .modeling_vocal_bout_parameters import BoutParameterPipeline
from .modeling_vocal_categories_multinomial import MultinomialModelingPipeline, MultinomialModelRunner
from .modeling_usv_manifold_position import ContinuousModelingPipeline, ContinuousModelRunner
from .load_input_files import load_pickle_modeling_data
from .modeling_bases_functions import (raised_cosine, bsplines, identity,
                                      laplacian_pyramid, _normalizecols)

def get_basis_matrix_standardized(
        settings: dict,
        history_frames: int,
        output_dir: str
) -> np.ndarray:
    """
    Constructs the temporal basis matrix and handles atomic verification plotting.

    This function centralizes the dimensionality reduction logic for sklearn-based
    linear models. It projects the raw behavioral history into a lower-dimensional
    basis set (e.g., Raised Cosines or B-Splines) to ensure temporal smoothness
    and reduce the parameter count of the model.

    The function includes a file-locking mechanism to ensure that in a SLURM
    array of 200+ jobs, only the first job generates the visual verification
    plot, preventing file corruption and redundant overhead.

    Parameters
    ----------
    settings : dict
        The 'modeling_settings' dictionary extracted from the project JSON.
    history_frames : int
        The number of temporal lags (columns) in the input feature matrix.
    output_dir : str
        The directory where the 'basis_verification.png' will be saved.

    Returns
    -------
    basis_matrix : np.ndarray or None
        A matrix of shape (history_frames, n_basis_functions). Returns None
        if the current model_type is 'pygam' (which uses internal splines).
    """
    model_cfg = settings['model_params']

    if model_cfg['model_engine'] != 'sklearn':
        return None

    basis_type = model_cfg['model_basis_function']
    w = history_frames
    basis_matrix = None

    if basis_type == 'raised_cosine':
        p = settings['hyperparameters']['basis_functions']['raised_cosine']
        kp = int(np.floor(w * p['kpeaks_proportion']))
        basis_matrix = raised_cosine(
            neye=p['neye'],
            ncos=p['ncos'],
            kpeaks=[0, kp],
            b=p['b'],
            w=w
        )

    elif basis_type == 'bspline':
        p = settings['hyperparameters']['basis_functions']['bspline']
        max_k = max(0, w - p['degree'])
        knots = np.linspace(0, max_k, p['n_splines'] - p['degree'] + 1).astype(int)
        basis_matrix = _normalizecols(
            bsplines(
                width=w,
                positions=knots,
                degree=p['degree']
            )
        )

    elif basis_type == 'laplacian_pyramid':
        p = settings['hyperparameters']['basis_functions']['laplacian_pyramid']
        basis_matrix = _normalizecols(
            laplacian_pyramid(
                width=w,
                levels=p['levels'],
                fwhm=p['fwhm']
            )
        )

    elif basis_type == 'identity':
        basis_matrix = identity(width=w)

    # Atomic Lock: Only the first job in the array generates the verification plot
    lock_file = pathlib.Path(output_dir) / ".basis_plotted"

    if not lock_file.exists() and basis_matrix is not None:
        try:
            lock_file.touch(exist_ok=False)

            plt.figure(figsize=(6, 4))
            plt.plot(basis_matrix)
            plt.title(f"Basis Verification: {basis_type}")
            plt.xlabel("Lags (frames)")
            plt.ylabel("Weight")
            plt.savefig(pathlib.Path(output_dir) / "basis_verification.png", dpi=150)
            plt.close()

        except FileExistsError:
            pass

    return basis_matrix

def dispatch_univariate_job(args: argparse.Namespace) -> None:
    """
    Orchestrates the loading, fitting, and saving of a single behavioral feature.

    This function acts as the execution router. It maps the SLURM_ARRAY_TASK_ID
    to a specific behavioral feature and then selects the appropriate
    modeling pipeline (Onset, Multinomial JAX, or Continuous JAX) based on
    the provided analysis_type.

    It enforces strict memory cleanup between the indexing phase and the
    modeling phase to ensure that GPU-enabled jobs have maximum available
    headroom for JAX's XLA buffer.

    Parameters
    ----------
    args : argparse.Namespace
        The command-line arguments containing:
        - analysis_type: The modeling framework to use.
        - feature_idx: The integer index from the SLURM job array.
        - input_data: Path to the source .pkl data.
        - settings_file: Path to the project configuration .json.
        - output_dir: Destination for the results file.

    Returns
    -------
    None
        Results are persisted to disk as a .pkl file.
    """
    print(f"--- USV Univariate Dispatcher | Task: {args.analysis_type.upper()} ---")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # 1. Load Experimental Configuration
    try:
        with open(args.settings_file, 'r') as f:
            settings = json.load(f)
    except Exception as e:
        print(f"FATAL: Settings load failed: {e}")
        return

    # 2. Semantic Feature Mapping
    try:
        with open(args.input_data, 'rb') as f:
            # We only load keys to save memory on the login/head node
            all_features = sorted(list(pickle.load(f).keys()))

        if args.feature_idx >= len(all_features):
            print(f"FATAL: Index {args.feature_idx} out of bounds.")
            return

        feature_name = all_features[args.feature_idx]
        print(f"[{timestamp}] Mapped Index {args.feature_idx} -> Feature: {feature_name}")

        # Explicitly free memory before starting pipeline initialization
        del all_features
        gc.collect()

    except Exception as e:
        print(f"FATAL: Feature mapping failed: {e}")
        return

    # 3. Execution Routing
    os.makedirs(args.output_dir, exist_ok=True)
    results = None

    try:
        # CATEGORY A: CPU-based Modeling (Onset, Category, Params)
        if args.analysis_type in ['onset', 'category', 'params']:

            data_dict = load_pickle_modeling_data(args.input_data)
            feat_data = data_dict[feature_name]

            if args.analysis_type == 'onset':

                pipeline = VocalOnsetModelingPipeline(modeling_settings_dict=settings)
                basis = get_basis_matrix_standardized(settings, pipeline.history_frames, args.output_dir)

                if settings['model_params']['model_engine'] == 'sklearn':
                    fn, res = pipeline._run_model_for_feature_sklearn(feature_name, feat_data, basis)
                else:
                    fn, res = pipeline._run_model_for_feature_pygam(feature_name, feat_data, None)

            elif args.analysis_type == 'category':

                pipeline = VocalCategoryModelingPipeline(modeling_settings_dict=settings)
                basis = get_basis_matrix_standardized(settings, pipeline.history_frames, args.output_dir)
                fn, res = pipeline._run_modeling_category(feature_name, feat_data, basis)

            elif args.analysis_type == 'params':

                pipeline = BoutParameterPipeline(modeling_settings_dict=settings)
                basis = get_basis_matrix_standardized(settings, pipeline.history_frames, args.output_dir)

                if settings['model_params']['model_engine'] == 'sklearn':
                    fn, res = pipeline._run_model_for_feature_sklearn(feature_name, feat_data, basis)
                else:
                    fn, res = pipeline._run_model_for_feature_pygam(feature_name, feat_data, None)

            results = {fn: res}

        # CATEGORY B: JAX/GPU-based Modeling (Multinomial Flat)
        elif args.analysis_type == 'multinomial':

            pipeline = MultinomialModelingPipeline(modeling_settings_dict=settings)
            runner = MultinomialModelRunner(pipeline_instance=pipeline)

            _, res = runner.run_univariate_training(
                pkl_path=args.input_data,
                feat_name=feature_name
            )

            results = {feature_name: res}

        # CATEGORY C: CONTINUOUS TOPOGRAPHY (UMAP Manifold)
        elif args.analysis_type == 'continuous':

            pipeline = ContinuousModelingPipeline(modeling_settings_dict=settings)
            runner = ContinuousModelRunner(pipeline_instance=pipeline)

            hp = settings['hyperparameters']['jax_linear']['bivariate_gaussian']

            data_blocks = runner.load_univariate_data_blocks(
                pkl_path=args.input_data,
                bin_size=hp['bin_resizing_factor']
            )

            raw_res = runner.run_univariate_training(
                data_blocks=data_blocks,
                feat_name=feature_name
            )

            # Bulletproof extraction: unpack tuple if present, otherwise just take the dict
            res = raw_res[1] if (isinstance(raw_res, tuple) and len(raw_res) == 2) else raw_res
            results = {feature_name: res}

    except Exception as e:
        print(f"FATAL ERROR: Analysis failed for {feature_name}. Error: {e}")
        return

    # 4. Atomic Result Serialization
    if results:
        safe_feat = feature_name.replace('.', '_')
        out_name = f"univariate_{args.feature_idx}_{safe_feat}_{timestamp}.pkl"
        out_path = os.path.join(args.output_dir, out_name)

        try:
            with open(out_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"[{datetime.now()}] Success. Results saved to: {out_path}")
        except Exception as e:
            print(f"FATAL: Saving results failed. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidated USV Modeling Dispatcher")

    parser.add_argument(
        '--analysis_type',
        required=True,
        choices=['onset', 'category', 'params', 'multinomial', 'continuous'],
        help="The type of USV analysis pipeline to execute."
    )

    parser.add_argument(
        '--feature_idx',
        type=int,
        required=True,
        help="Deterministic index of the behavioral feature from the input pickle."
    )

    parser.add_argument(
        '--input_data',
        required=True,
        help="Path to the .pkl file containing aligned feature history."
    )

    parser.add_argument(
        '--settings_file',
        required=True,
        help="Path to the modeling_settings.json configuration."
    )

    parser.add_argument(
        '--output_dir',
        required=True,
        help="Directory where the individual feature pickle will be saved."
    )

    args = parser.parse_args()
    dispatch_univariate_job(args)
