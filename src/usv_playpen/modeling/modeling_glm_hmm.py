"""
@author: bartulem

Pipeline for fitting a GLM-HMM over latent vocal states and selecting the number
of states.

Consumes the same continuous-manifold modeling input pickle the acoustic-manifold
selection uses (``D[feature][session] = {'X', 'Y', 'w', 'supercategory'}``, rows
already in temporal USV-onset order) and turns each session into one observation
sequence: the emission design matrix ``X`` is the horizontal stack of the
configured behavioural features' history windows, and the target is the shared
2-D manifold position ``Y``. It then fits :class:`~usv_playpen.modeling.glm_hmm.GLMHMM`
for a range of state counts, scoring each on the reserved held-out sessions
(Project-1's ``held_out_session_ids``) by held-out log-likelihood, with BIC on the
development set as a reported cross-check, and saves the selected model together
with the per-session Viterbi state paths.

Two emission/engine paths are supported. The acoustic-manifold (von Mises)
emission and the categorical (multinomial) emission both fit the EM-based
:class:`~usv_playpen.modeling.glm_hmm.GLMHMM` with stationary transitions. The
categorical path additionally offers ``transition_mode='input_driven'``, which
fits the reference-coded, direct-marginal
:class:`~usv_playpen.modeling.glm_hmm.InputDrivenGLMHMM` with per-timestep
transition GLMs of the behavioural design -- the Calhoun/Pillow/Murthy-faithful
engine validated on the Coen-2014 fly to reproduce its state-selection without the
state collapse the EM engine suffers on weakly-identified categorical data.
"""

import json
import pickle
from pathlib import Path

import numpy as np

from .glm_hmm import (
    GLMHMM,
    InputDrivenGLMHMM,
    InputDrivenManifoldGLMHMM,
    resolve_emission_cls,
)
from .modeling_metadata import (
    compute_settings_sha256,
    get_git_commit_info,
    get_package_version,
    inject_metadata,
)
from .modeling_utils import held_out_session_ids_from_metadata


def _read_selected_features(model_selection_path: str) -> list:
    """
    Description
    -----------
    Read the acoustic-manifold selection's chosen feature set from its output.

    The GLM-HMM emission uses exactly the behavioural features the manifold
    selection kept (``final_model_features``), so the two stages stay coupled:
    "given the behaviours that predict acoustic position, do latent states use
    them differently?". The feature set is therefore read from the selection
    output rather than configured, so it can never silently drift from the model
    that produced it.

    Accepts either the final step pickle directly, or the selection output
    directory (the highest-numbered ``*_step_*.pkl`` is the final model, the one
    that carries ``final_model_features``).

    Parameters
    ----------
    model_selection_path (str)
        Path to the selection output file or directory.

    Returns
    -------
    features (list)
        The selected behavioural feature names, in selection order.
    """
    path = Path(model_selection_path)
    if path.is_dir():
        step_files = sorted(path.glob('*_step_*.pkl'),
                            key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        if not step_files:
            raise ValueError(
                f"No '*_step_*.pkl' selection files found in {model_selection_path!r}; "
                "point model_selection_path at a manifold model-selection output."
            )
        path = step_files[-1]
    with open(path, 'rb') as handle:
        selection = pickle.load(handle)
    if 'final_model_features' not in selection:
        raise ValueError(
            f"{path} does not carry 'final_model_features'; it is not a finalized "
            "manifold model-selection result (re-run selection to completion)."
        )
    features = list(selection['final_model_features'])
    if not features:
        raise ValueError(
            f"The manifold selection at {path} kept no features "
            "('final_model_features' is empty), so there is nothing to build a "
            "GLM-HMM emission from."
        )
    return features


def _build_session_sequences(raw_data: dict, emission_features: list,
                             session_ids: list, target_key: str,
                             drop_nan_target: bool, history_frames: int) -> list:
    """
    Description
    -----------
    Turn each session into one GLM-HMM observation sequence.

    For every session the configured ``emission_features`` contribute their
    per-event behavioural history windows, each truncated to its most recent
    ``history_frames`` frames (the columns closest to the vocalization onset --
    the stored window is ordered oldest-first, ``onset - history_frames_full ...
    onset - 1``, so the last columns are the most recent) and horizontally stacked
    (feature-major, matching the emission's ``(n_features, n_time_bins, ...)``
    coefficient layout) into the emission design matrix ``X``. Truncating to the
    behaviourally-predictive recent window keeps the emission's per-cue temporal
    filter at full resolution while holding its parameter count (and hence the
    BIC state penalty) in check. The target is read from ``target_key`` of the
    first feature's session entry (identical across features): ``'Y'`` (2-D
    position) for the manifold emission, the categorical label column (e.g.
    ``'supercategory'``) for the multinomial. Rows are already in temporal
    USV-onset order in the input pickle, so no re-sort is needed.

    Parameters
    ----------
    raw_data (dict)
        The input pickle's ``feature -> session -> {'X', 'Y', ...}`` mapping.
    emission_features (list)
        Behavioural feature names whose history windows form ``X`` (in the given
        order, feature-major).
    session_ids (list)
        Sessions to build sequences for (already filtered to a partition).
    target_key (str)
        Per-session key holding the emission target (``'Y'`` for manifold, the
        categorical column for multinomial).
    drop_nan_target (bool)
        When True (categorical target), events whose label is NaN (unlabelled)
        are dropped from the sequence, together with their design-matrix rows.
        The manifold position target has no NaNs, so this is False there.
    history_frames (int)
        Number of most-recent history frames to keep per feature (the emission's
        ``n_time_bins``). Each feature's stored window is sliced to its last
        ``history_frames`` columns; a window already shorter is kept whole.

    Returns
    -------
    sequences (list)
        List of ``(session_id, X_seq, target_seq)`` tuples, one per session that
        is present for every requested feature and non-empty after any NaN drop.
    """
    sequences = []
    for session_id in session_ids:
        if not all(session_id in raw_data[feat] for feat in emission_features):
            continue
        x_blocks = []
        for feat in emission_features:
            x_feat = np.asarray(raw_data[feat][session_id]['X'], dtype=np.float64)
            if x_feat.shape[1] > history_frames:
                x_feat = x_feat[:, -history_frames:]     # most-recent frames (cols closest to onset)
            x_blocks.append(x_feat)
        n_events = x_blocks[0].shape[0]
        if n_events == 0:
            continue
        x_seq = np.hstack(x_blocks)
        target_seq = np.asarray(raw_data[emission_features[0]][session_id][target_key])
        if drop_nan_target:
            keep = ~np.isnan(np.asarray(target_seq, dtype=np.float64))
            if not keep.any():
                continue
            x_seq = x_seq[keep]
            target_seq = target_seq[keep].astype(np.int64)
        sequences.append((session_id, x_seq, target_seq))
    return sequences


def _manifold_emission_factory(n_features: int, n_time_bins: int, *,
                               metric: str, period: float,
                               lambda_smooth: float, l2_reg: float):
    """
    Description
    -----------
    Build a zero-argument factory that returns a fresh manifold (von Mises)
    emission configured for the GLM-HMM's per-state regressor.

    Parameters
    ----------
    n_features (int)
        Number of behavioural features in the stacked design matrix.
    n_time_bins (int)
        History-window length per feature.
    metric (str)
        Manifold geometry (``'torus'`` or ``'euclidean'``).
    period (float)
        Per-axis wrap period.
    lambda_smooth (float)
        Temporal-smoothness penalty for each state's regressor.
    l2_reg (float)
        Ridge penalty for each state's regressor.

    Returns
    -------
    factory (callable)
        A zero-argument callable returning a new ``ManifoldEmission``.
    """
    emission_cls = resolve_emission_cls('manifold')
    regressor_kwargs = dict(n_features=n_features, n_time_bins=n_time_bins,
                            lambda_smooth=lambda_smooth, l2_reg=l2_reg,
                            metric=metric, period=period)
    return lambda: emission_cls(regressor_kwargs)


def _multinomial_emission_factory(n_features: int, n_time_bins: int, *,
                                  lambda_smooth: float, l2_reg: float,
                                  focal_gamma: float, max_iter: int):
    """
    Description
    -----------
    Build a zero-argument factory returning a fresh multinomial (categorical)
    emission for the GLM-HMM's per-state classifier.

    Parameters
    ----------
    n_features (int)
        Number of behavioural features in the stacked design matrix.
    n_time_bins (int)
        History-window length per feature.
    lambda_smooth (float)
        Temporal-smoothness penalty for each state's classifier.
    l2_reg (float)
        Ridge penalty for each state's classifier.
    focal_gamma (float)
        Focal-loss focusing parameter (``0.0`` = plain cross-entropy).
    max_iter (int)
        Optimiser iterations for each state's classifier.

    Returns
    -------
    factory (callable)
        A zero-argument callable returning a new ``MultinomialEmission``.
    """
    emission_cls = resolve_emission_cls('multinomial')
    classifier_kwargs = dict(n_features=n_features, n_time_bins=n_time_bins,
                             lambda_smooth=lambda_smooth, l2_reg=l2_reg,
                             focal_gamma=focal_gamma, max_iter=max_iter)
    return lambda: emission_cls(classifier_kwargs)


def run_glm_hmm_state_selection(input_data_path: str, settings_path: str,
                                output_directory: str,
                                model_selection_path: str) -> dict:
    """
    Description
    -----------
    Fit a GLM-HMM over latent vocal states across a range of state counts and
    select the number of states on the reserved held-out sessions.

    The behavioural features that form the emission design matrix are NOT
    configured -- they are read from the acoustic-manifold selection's output
    (``final_model_features`` at ``model_selection_path``), so the GLM-HMM always
    emits from exactly the features that selection kept and cannot drift from the
    model that produced them. The ``glm_hmm`` settings block supplies the emission
    type, state-count range, and EM controls only. Sessions are partitioned into
    development and held-out using the input pickle's ``held_out_session_ids``
    (Project-1's reserve). The number of states is chosen by **cross-validated
    held-out log-likelihood** over the development sessions (``cv_folds``-fold,
    Calhoun 2019's criterion): each candidate state count is fit on the training
    folds and scored on the held-out fold, summed across folds; the K with the
    highest cross-validated log-likelihood wins. BIC and (if a reserve exists)
    the reserved-set log-likelihood are reported alongside as diagnostics ONLY --
    BIC's parameter-count penalty spuriously prefers too few states when the
    emission's raw parameter count exceeds its smoothed effective degrees of
    freedom. The selected model is refit on all development sessions and decoded
    to give each session a Viterbi state path, and everything is pickled to
    ``output_directory``.

    Parameters
    ----------
    input_data_path (str)
        Path to the continuous-manifold modeling input pickle.
    settings_path (str)
        Path to the modeling settings JSON (must carry a ``glm_hmm`` block).
    output_directory (str)
        Directory the results pickle is written to (created if absent).
    model_selection_path (str)
        Path to the acoustic-manifold model-selection output (the finalized step
        pickle, or its directory) whose ``final_model_features`` become the
        emission design-matrix features.

    Returns
    -------
    results (dict)
        The saved results dict (also written to disk): ``selected_n_states``,
        ``selection_table`` (per-K held-out log-likelihood + BIC), ``log_pi``,
        ``transition_matrix``, ``state_paths`` (session -> Viterbi states), and a
        ``metadata`` block describing the run.
    """
    with open(settings_path, 'r') as settings_file:
        settings = json.load(settings_file)
    glm_hmm_settings = settings['glm_hmm']
    emission_type = glm_hmm_settings['emission_type']
    if emission_type not in ('manifold', 'multinomial'):
        raise ValueError(
            f"run_glm_hmm_state_selection supports emission_type 'manifold' "
            f"(behaviour -> acoustic-manifold position, von Mises density) or "
            f"'multinomial' (behaviour -> USV category, categorical density); got "
            f"{emission_type!r}."
        )
    # Transition dynamics. 'static' fits the EM-based `GLMHMM` (stationary K x K
    # matrix). 'input_driven' fits a direct-marginal engine with Calhoun-faithful
    # per-timestep transition GLMs -- the reference-coded `InputDrivenGLMHMM` for the
    # categorical (multinomial) model, or the product-von-Mises `InputDrivenManifoldGLMHMM`
    # for the continuous (manifold) model. Both were validated to fit without the state
    # collapse the EM engine suffers on weakly-identified data. (The torus requirement
    # for the input-driven manifold engine is checked below, once the metric is known.)
    transition_mode = glm_hmm_settings['transition_mode']
    if transition_mode not in ('static', 'input_driven'):
        message = (f"glm_hmm.transition_mode must be 'static' or 'input_driven'; "
                   f"got {transition_mode!r}.")
        raise ValueError(message)
    emission_features = _read_selected_features(model_selection_path)
    history_frames = int(glm_hmm_settings['history_frames'])
    n_states_min = int(glm_hmm_settings['n_states_min'])
    n_states_max = int(glm_hmm_settings['n_states_max'])
    n_em_iters = int(glm_hmm_settings['n_em_iters'])
    n_lbfgs = int(glm_hmm_settings['n_lbfgs'])
    n_restarts = int(glm_hmm_settings['n_restarts'])
    em_tol = float(glm_hmm_settings['em_tol'])
    transition_pseudocount = float(glm_hmm_settings['transition_pseudocount'])
    lambda_smooth = float(glm_hmm_settings['lambda_smooth'])
    input_driven_lambda_smooth = float(glm_hmm_settings['input_driven_lambda_smooth'])
    l2_reg = float(glm_hmm_settings['l2_reg'])
    random_seed = int(settings['model_validation']['random_seed'])

    with open(input_data_path, 'rb') as input_file:
        raw_data = pickle.load(input_file)
    input_metadata = raw_data.pop('_input_metadata', {})

    # Manifold geometry from the pickle's own record (the extraction stage writes
    # it under analysis_specific), falling back to the settings' vocal_features.
    analysis_specific = input_metadata['analysis_specific'] if 'analysis_specific' in input_metadata else {}
    metric = (analysis_specific['manifold_metric'] if 'manifold_metric' in analysis_specific
              else settings['vocal_features']['usv_manifold_metric'])
    period = (analysis_specific['manifold_period'] if 'manifold_period' in analysis_specific
              else settings['vocal_features']['usv_manifold_period'])
    # The input-driven manifold engine is a product von Mises on the torus; euclidean
    # geometry would need a Gaussian emission that engine does not implement.
    if transition_mode == 'input_driven' and emission_type == 'manifold' and metric != 'torus':
        message = ("glm_hmm.transition_mode='input_driven' with emission_type='manifold' "
                   f"requires a torus manifold (von Mises); got metric={metric!r}.")
        raise ValueError(message)

    # Emission-type-specific target column and per-state emission factory. The
    # manifold emission predicts the 2-D position 'Y' (no NaNs); the multinomial
    # emission predicts a categorical label column (configured `multinomial_target`,
    # e.g. 'supercategory'), dropping events whose label is NaN.
    if emission_type == 'manifold':
        target_key = 'Y'
        drop_nan_target = False
        emission_descriptor = f'manifold({metric})'
    else:
        target_key = glm_hmm_settings['multinomial_target']
        drop_nan_target = True
        engine_tag = 'input-driven' if transition_mode == 'input_driven' else 'static'
        emission_descriptor = f'multinomial({target_key}, {engine_tag})'

    # Session partition: every session present for the emission features, split
    # into development vs the reserved held-out set (Project-1's reserve).
    all_sessions = sorted(set(raw_data[emission_features[0]].keys()))
    held_out_ids = set(held_out_session_ids_from_metadata(input_metadata))
    dev_ids = [s for s in all_sessions if s not in held_out_ids]
    held_ids = [s for s in all_sessions if s in held_out_ids]

    dev_sequences = _build_session_sequences(
        raw_data, emission_features, dev_ids, target_key, drop_nan_target, history_frames)
    held_sequences = _build_session_sequences(
        raw_data, emission_features, held_ids, target_key, drop_nan_target, history_frames)
    if not dev_sequences:
        raise ValueError(
            "GLM-HMM state selection found no usable development sequences for "
            f"emission_features={emission_features}. Check the feature names and "
            "that the input pickle carries their per-session 'X'."
        )

    n_time_bins = dev_sequences[0][1].shape[1] // len(emission_features)
    dev_xy = [(x_seq, target_seq) for _, x_seq, target_seq in dev_sequences]
    held_xy = [(x_seq, target_seq) for _, x_seq, target_seq in held_sequences]
    use_heldout = len(held_xy) > 0

    # Engine dispatch. The 'input_driven' categorical path fits the reference-coded
    # direct-marginal `InputDrivenGLMHMM` (validated on the Coen-2014 fly to reproduce
    # Calhoun's state-selection without collapse); every other path fits the EM-based
    # `GLMHMM` with a per-state emission factory.
    input_driven = transition_mode == 'input_driven'
    factory = None
    n_classes = None
    if input_driven:
        if emission_type == 'multinomial':
            all_labels = np.concatenate([target for _, target in dev_xy + held_xy])
            n_classes = int(np.asarray(all_labels).max()) + 1
        # The input-driven manifold engine reads the torus target directly (no factory).
    elif emission_type == 'manifold':
        factory = _manifold_emission_factory(
            len(emission_features), n_time_bins,
            metric=metric, period=period, lambda_smooth=lambda_smooth, l2_reg=l2_reg,
        )
    else:
        factory = _multinomial_emission_factory(
            len(emission_features), n_time_bins,
            lambda_smooth=lambda_smooth, l2_reg=l2_reg,
            focal_gamma=float(glm_hmm_settings['focal_gamma']),
            max_iter=int(glm_hmm_settings['multinomial_max_iter']),
        )

    def fit_model(n_states: int, train_sequences: list,
                  held_out_for_selection: list | None) -> object:
        """Fit the configured engine for `n_states` (input-driven direct-marginal, or static EM)."""
        if input_driven:
            if emission_type == 'manifold':
                model = InputDrivenManifoldGLMHMM(
                    n_states=n_states, n_features=len(emission_features),
                    n_time_bins=n_time_bins, period=float(period),
                    lambda_smooth=input_driven_lambda_smooth, n_restarts=n_restarts,
                    n_lbfgs=n_lbfgs, random_state=random_seed)
            else:
                model = InputDrivenGLMHMM(
                    n_states=n_states, n_features=len(emission_features),
                    n_time_bins=n_time_bins, n_classes=n_classes,
                    lambda_smooth=input_driven_lambda_smooth, n_restarts=n_restarts,
                    n_lbfgs=n_lbfgs, random_state=random_seed)
            return model.fit(train_sequences, held_out_sequences=held_out_for_selection)
        model = GLMHMM(
            n_states=n_states, emission_factory=factory, n_em_iters=n_em_iters,
            n_restarts=n_restarts, tol=em_tol,
            transition_pseudocount=transition_pseudocount, random_state=random_seed)
        return model.fit(train_sequences)

    print(f"GLM-HMM state selection | emission={emission_descriptor} | "
          f"features={emission_features} | dev sessions={len(dev_xy)} | "
          f"held-out sessions={len(held_xy)} | K in [{n_states_min}, {n_states_max}]",
          flush=True)

    # K is chosen by CROSS-VALIDATED held-out log-likelihood over the development
    # sessions (Calhoun 2019's criterion), NOT BIC. With smoothed high-parameter
    # emissions the raw parameter count sits far above the effective degrees of
    # freedom, so BIC's `n_params * log(n_obs)` penalty spuriously prefers too few
    # states (it once forced K=2 on the manifold emission purely from parameter
    # count); the cross-validated predictive log-likelihood carries no such
    # penalty and rises to the true state count before it falls. BIC and the
    # reserved-set log-likelihood are recorded alongside as DIAGNOSTICS only.
    n_folds = min(int(glm_hmm_settings['cv_folds']), len(dev_xy))
    if n_folds < 2:
        raise ValueError(
            "GLM-HMM state selection needs at least 2 development sessions for "
            f"cross-validation; got {len(dev_xy)}."
        )
    # Round-robin session -> fold assignment (deterministic, balanced).
    fold_of = [i % n_folds for i in range(len(dev_xy))]
    print(f"  Selecting K by {n_folds}-fold cross-validated held-out log-likelihood "
          f"over dev sessions (BIC reported as a diagnostic only).", flush=True)

    selection_table = []
    fitted_models = {}
    for n_states in range(n_states_min, n_states_max + 1):
        # Cross-validated held-out log-likelihood: fit on the training folds, score the
        # held-out fold; the SUM is the selection criterion, and the PER-FOLD list is
        # kept so a downstream paired-margin analysis can judge whether the K-to-K
        # differences clear the fold-to-fold noise (the fold assignment is fixed across
        # K, so fold f scores the same held-out sessions at every K -- the per-fold LLs
        # are paired across K).
        cv_fold_log_likelihoods = []
        for fold in range(n_folds):
            train = [dev_xy[i] for i in range(len(dev_xy)) if fold_of[i] != fold]
            test = [dev_xy[i] for i in range(len(dev_xy)) if fold_of[i] == fold]
            if not train or not test:
                continue
            cv_model = fit_model(n_states, train, None)
            cv_fold_log_likelihoods.append(float(cv_model.log_likelihood(test)))
        cv_log_likelihood = float(sum(cv_fold_log_likelihoods))
        # Full-development fit: the source of the final selected model, plus the
        # (diagnostic) BIC and reserved-set log-likelihood. The reserved held-out set
        # (when present) also selects the input-driven engine's best restart.
        full_model = fit_model(n_states, dev_xy, held_xy if use_heldout else None)
        fitted_models[n_states] = full_model
        reserved_ll = full_model.log_likelihood(held_xy) if use_heldout else float('nan')
        dev_bic = full_model.bic(dev_xy)
        selection_table.append({
            'n_states': n_states,
            'cv_log_likelihood': cv_log_likelihood,
            'cv_log_likelihood_per_fold': cv_fold_log_likelihoods,
            'dev_bic': dev_bic,
            'reserved_held_out_log_likelihood': reserved_ll,
            'dev_log_likelihood': full_model.log_likelihood_,
        })
        print(f"  K={n_states} | CV held-out loglik={cv_log_likelihood:.1f} | "
              f"dev BIC={dev_bic:.1f} (diagnostic)", flush=True)

    # Selection: maximise the cross-validated held-out log-likelihood.
    selected = max(selection_table, key=lambda row: row['cv_log_likelihood'])
    selected_n_states = selected['n_states']
    selected_model = fitted_models[selected_n_states]
    print(f"Selected {selected_n_states} latent vocal states "
          f"(cross-validated held-out log-likelihood).", flush=True)

    # Viterbi state path per development session under the selected model.
    state_paths = {}
    for session_id, x_seq, target_seq in dev_sequences:
        state_paths[session_id] = selected_model.viterbi(x_seq, target_seq)

    # Initial-state and transition summaries. The static `GLMHMM` carries a single
    # stationary matrix; the input-driven engine's transitions vary per event, so a
    # mean-over-dev-events matrix is recorded as a representative summary.
    if input_driven:
        result_log_pi = selected_model.initial_state_distribution()
        result_transition = selected_model.mean_transition_matrix(dev_xy)
    else:
        result_log_pi = np.exp(selected_model.log_pi_)
        result_transition = np.exp(selected_model.log_A_)

    results = {
        'selected_n_states': selected_n_states,
        'selection_criterion': 'cv_log_likelihood',
        'selection_table': selection_table,
        'log_pi': result_log_pi,
        'transition_matrix': result_transition,
        'state_paths': state_paths,
        'metadata': {
            'emission_type': emission_type,
            'transition_mode': transition_mode,
            'emission_features': emission_features,
            'target_key': target_key,
            'history_frames': history_frames,
            'cv_folds': n_folds,
            'manifold_metric': metric,
            'manifold_period': period,
            'n_states_range': [n_states_min, n_states_max],
            'n_em_iters': n_em_iters,
            'n_lbfgs': n_lbfgs,
            'n_restarts': n_restarts,
            'n_classes': n_classes,
            'n_dev_sessions': len(dev_xy),
            'n_held_out_sessions': len(held_xy),
            'held_out_session_ids': held_ids,
        },
    }

    # Provenance: stamp the run-config block with git / package / settings-hash
    # provenance and re-attach the upstream ``_input_metadata`` (popped above when the
    # pickle was read) under its reserved key, so the GLM-HMM artifact is independently
    # provenance-complete -- the same guarantee every other modeling stage's output
    # carries. Injection is skipped for a legacy input pickle that carried no metadata.
    git_info = get_git_commit_info()
    results['metadata']['git_commit'] = git_info['commit']
    results['metadata']['git_dirty'] = git_info['dirty']
    results['metadata']['package_version'] = get_package_version()
    results['metadata']['settings_sha256'] = compute_settings_sha256(settings_path)
    if input_metadata:
        results = inject_metadata(results, _input_metadata=input_metadata)

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename_tag = metric if emission_type == 'manifold' else target_key
    out_path = output_dir / f"glm_hmm_states_{emission_type}_{filename_tag}_K{selected_n_states}.pkl"
    with open(out_path, 'wb') as out_file:
        pickle.dump(results, out_file)
    print(f"Saved GLM-HMM state-selection results to {out_path}", flush=True)
    return results
