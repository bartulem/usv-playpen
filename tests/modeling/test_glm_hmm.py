"""
@author: bartulem
Synthetic-recovery tests for the GLM-HMM engine in
``usv_playpen.modeling.glm_hmm`` and the per-sample von Mises helpers it relies on
in ``usv_playpen.modeling.manifold_metric``.

The engine is validated the way an HMM should be: generate observation sequences
from a KNOWN latent-state chain with well-separated emissions, fit a fresh model,
and assert it recovers the truth -- the emission parameters (up to a state
relabelling), a high Viterbi state-decoding accuracy, and a state-count
preference (BIC / held-out log-likelihood favour the true K over K=1). The pure
HMM machinery is exercised with the X-independent Gaussian emission; the
acoustic-manifold integration (regressor refit + von Mises density + weighted
concentration) is exercised with the torus ManifoldEmission.
"""

import json
import pathlib
import pickle
import warnings

import numpy as np
import pytest

# The modeling import chain pulls optax -> a one-time JAX DeprecationWarning.
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    from usv_playpen.modeling.glm_hmm import (
        GLMHMM,
        GaussianEmission,
        InputDrivenGLMHMM,
        InputDrivenManifoldGLMHMM,
        ManifoldEmission,
        MultinomialEmission,
        resolve_emission_cls,
    )
    from usv_playpen.modeling.jax_multinomial_logistic_regression import (
        SmoothMultinomialLogisticRegression,
    )
    from usv_playpen.modeling.modeling_glm_hmm import (
        run_glm_hmm_state_selection,
        _read_selected_features,
    )
    from usv_playpen.modeling.manifold_metric import (
        macro_von_mises_logscore,
        von_mises_logpdf_per_point,
        signed_diff,
        _fit_von_mises_kappa,
    )

REPO_SETTINGS = pathlib.Path(__file__).resolve().parents[2] / \
    "src" / "usv_playpen" / "_parameter_settings" / "modeling_settings.json"


def _simulate_markov_states(n_seqs, seq_len, transition, rng):
    """Sample `n_seqs` state sequences of length `seq_len` from a Markov chain."""
    n_states = transition.shape[0]
    sequences = []
    for _ in range(n_seqs):
        states = np.empty(seq_len, dtype=int)
        states[0] = rng.integers(n_states)
        for t in range(1, seq_len):
            states[t] = rng.choice(n_states, p=transition[states[t - 1]])
        sequences.append(states)
    return sequences


def _match_by_nearest(true_means, fit_means):
    """Greedy nearest-mean permutation aligning fitted states to true states."""
    order = []
    for k in range(true_means.shape[0]):
        dists = np.linalg.norm(fit_means - true_means[k][None, :], axis=1)
        for used in order:
            dists[used] = np.inf
        order.append(int(np.argmin(dists)))
    return order


# von Mises helpers (the emission-density kernel)
def test_von_mises_logpdf_per_point_mean_matches_pooled_macro():
    """The per-sample log-density averaged over points equals the pooled macro score."""
    rng = np.random.default_rng(0)
    y_true = rng.random((200, 2))
    y_pred = np.mod(y_true + 0.02 * rng.standard_normal((200, 2)), 1.0)
    kappa = _fit_von_mises_kappa(
        signed_diff(y_true, y_pred, metric='torus', period=1.0) * (2 * np.pi))
    per_point = von_mises_logpdf_per_point(y_pred, y_true, metric='torus', period=1.0, kappa=kappa)
    pooled = macro_von_mises_logscore(y_pred, y_true, None, metric='torus', period=1.0, kappa=kappa)
    assert per_point.shape == (200,)
    assert np.isclose(per_point.mean(), pooled, rtol=1e-9, atol=1e-9)


def test_fit_von_mises_kappa_weighted_tracks_the_upweighted_group():
    """Weighting toward a tight group raises the fitted concentration vs a loose group."""
    rng = np.random.default_rng(1)
    tight = 0.02 * rng.standard_normal((300, 2))          # small residual -> high kappa
    loose = 0.5 * rng.standard_normal((300, 2))           # large residual -> low kappa
    r = np.vstack([tight, loose])
    w_tight = np.concatenate([np.ones(300), np.full(300, 1e-3)])[:, None]
    w_loose = np.concatenate([np.full(300, 1e-3), np.ones(300)])[:, None]
    kappa_tight = _fit_von_mises_kappa(r, weights=w_tight)
    kappa_loose = _fit_von_mises_kappa(r, weights=w_loose)
    assert kappa_tight > kappa_loose
    # Upweighting the tight group must beat the unweighted pooled fit too.
    assert kappa_tight > _fit_von_mises_kappa(r)


# HMM machinery (Gaussian emission, X-independent)
def test_gaussian_hmm_recovers_states_and_transitions():
    """A 2-state Gaussian HMM with separated means is recovered: means, Viterbi, A."""
    rng = np.random.default_rng(2)
    true_means = np.array([[0.0, 0.0], [4.0, 4.0]])
    true_var = 0.4
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    state_seqs = _simulate_markov_states(n_seqs=10, seq_len=80, transition=transition, rng=rng)

    sequences = []
    for states in state_seqs:
        y = true_means[states] + np.sqrt(true_var) * rng.standard_normal((states.shape[0], 2))
        X = np.zeros((states.shape[0], 1))               # Gaussian emission ignores X
        sequences.append((X, y))

    model = GLMHMM(n_states=2, emission_factory=lambda: GaussianEmission(n_targets=2),
                   n_em_iters=100, n_restarts=4, tol=1e-5, random_state=0)
    model.fit(sequences)

    fit_means = np.array([em.mean_ for em in model.emissions_])
    order = _match_by_nearest(true_means, fit_means)
    assert np.allclose(fit_means[order], true_means, atol=0.4)

    # Viterbi decoding accuracy (aligned to the recovered labelling).
    inv = np.argsort(order)
    correct = total = 0
    for (X, y), states in zip(sequences, state_seqs):
        decoded = inv[model.viterbi(X, y)]
        correct += int((decoded == states).sum())
        total += states.shape[0]
    assert correct / total > 0.9

    # Recovered transition matrix (reordered) tracks the truth's strong persistence.
    fit_A = np.exp(model.log_A_)[np.ix_(order, order)]
    assert np.allclose(fit_A, transition, atol=0.15)


def test_bic_prefers_true_state_count_over_one():
    """BIC (lower better) favours K=2 over K=1 on 2-state data."""
    rng = np.random.default_rng(3)
    true_means = np.array([[0.0, 0.0], [5.0, 5.0]])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    state_seqs = _simulate_markov_states(6, 80, transition, rng)
    sequences = []
    for states in state_seqs:
        y = true_means[states] + 0.5 * rng.standard_normal((states.shape[0], 2))
        sequences.append((np.zeros((states.shape[0], 1)), y))

    bics = {}
    for k in (1, 2):
        m = GLMHMM(k, lambda: GaussianEmission(n_targets=2),
                   n_em_iters=80, n_restarts=3, random_state=0).fit(sequences)
        bics[k] = m.bic(sequences)
    assert bics[2] < bics[1]


# Manifold (torus vM) emission integration
def test_manifold_emission_separates_two_torus_states():
    """
    A 2-state torus GLM-HMM whose states sit at distinct manifold positions is
    recovered: the regressor + weighted von Mises M-step give each state a
    positive concentration and distinct mean prediction, with high Viterbi
    accuracy.
    """
    rng = np.random.default_rng(4)
    positions = np.array([[0.2, 0.2], [0.75, 0.75]])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    state_seqs = _simulate_markov_states(n_seqs=8, seq_len=70, transition=transition, rng=rng)

    n_features, n_time_bins = 2, 3
    sequences = []
    for states in state_seqs:
        n = states.shape[0]
        X = rng.standard_normal((n, n_features * n_time_bins))
        y = np.mod(positions[states] + 0.03 * rng.standard_normal((n, 2)), 1.0)
        sequences.append((X, y))

    reg_kwargs = dict(n_features=n_features, n_time_bins=n_time_bins,
                      lambda_smooth=1.0, l2_reg=1.0, metric='torus', period=1.0)
    model = GLMHMM(n_states=2, emission_factory=lambda: ManifoldEmission(reg_kwargs),
                   n_em_iters=40, n_restarts=3, tol=1e-4, random_state=0)
    model.fit(sequences)

    # Each state has a positive concentration and a distinct mean position.
    kappas = np.array([em.kappa_ for em in model.emissions_])
    assert np.all(kappas > 0.0)
    X_all = np.concatenate([X for X, _ in sequences], axis=0)
    mean_pos = np.array([em.regressor_.predict(X_all).mean(axis=0) for em in model.emissions_])
    assert np.linalg.norm(mean_pos[0] - mean_pos[1]) > 0.2

    # Viterbi accuracy against the true state chain (best label alignment).
    fit_state_pos = np.array([em.regressor_.predict(X_all).mean(axis=0) for em in model.emissions_])
    order = _match_by_nearest(positions, fit_state_pos)
    inv = np.argsort(order)
    correct = total = 0
    for (X, y), states in zip(sequences, state_seqs):
        decoded = inv[model.viterbi(X, y)]
        correct += int((decoded == states).sum())
        total += states.shape[0]
    assert correct / total > 0.85


def test_resolve_emission_cls_covers_all_three():
    """The registry resolves gaussian/manifold/multinomial; an unknown type raises."""
    assert resolve_emission_cls('gaussian') is GaussianEmission
    assert resolve_emission_cls('manifold') is ManifoldEmission
    assert resolve_emission_cls('multinomial') is MultinomialEmission
    with pytest.raises(ValueError, match="Unknown emission_type"):
        resolve_emission_cls('nonsense')


def test_multinomial_emission_recovers_two_states():
    """
    A 2-state categorical GLM-HMM whose states map behaviour to USV category by
    DIFFERENT rules is recovered via the responsibility-weighted (soft-EM)
    multinomial M-step: high Viterbi accuracy against the true state chain. This
    exercises the new per-sample sample_weight path in the multinomial GLM.
    """
    rng = np.random.default_rng(11)
    n_features, n_time_bins, n_classes = 2, 2, 2
    n_inputs = n_features * n_time_bins
    # Two distinct behaviour -> category rules: each state has its own light
    # X-dependence AND a strong per-state base rate (intercept), i.e. a different
    # "vocal repertoire" -- state 0 favours category 0, state 1 favours category 1.
    W = [0.5 * rng.standard_normal((n_inputs, n_classes)),
         0.5 * rng.standard_normal((n_inputs, n_classes))]
    b = [np.array([3.0, -3.0]), np.array([-3.0, 3.0])]
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    state_seqs = _simulate_markov_states(n_seqs=10, seq_len=80, transition=transition, rng=rng)

    sequences = []
    for states in state_seqs:
        n = states.shape[0]
        X = rng.standard_normal((n, n_inputs))
        # Deterministic argmax label under the active state's rule -> a strong,
        # learnable per-state mapping the classifier can recover.
        y = np.array([int(np.argmax(X[t] @ W[states[t]] + b[states[t]])) for t in range(n)])
        sequences.append((X, y))

    clf_kwargs = dict(n_features=n_features, n_time_bins=n_time_bins,
                      lambda_smooth=1.0, l2_reg=0.1, focal_gamma=0.0, max_iter=200)
    model = GLMHMM(n_states=2, emission_factory=lambda: MultinomialEmission(clf_kwargs),
                   n_em_iters=40, n_restarts=3, tol=1e-4, random_state=0)
    model.fit(sequences)

    # Each state's classifier saw the full label set (consistent columns).
    assert all(em.classifier_ is not None for em in model.emissions_)
    assert all(len(em.classifier_.classes_) == n_classes for em in model.emissions_)

    # The two states learned distinct classifiers (distinct mean class profiles).
    X_all = np.concatenate([X for X, _ in sequences], axis=0)
    state_profiles = np.array([em.classifier_.predict_proba(X_all).mean(axis=0)
                               for em in model.emissions_])
    assert np.linalg.norm(state_profiles[0] - state_profiles[1]) > 0.05

    # Viterbi decoding accuracy against the true chain (best of the two labellings).
    best_acc = 0.0
    for perm in ([0, 1], [1, 0]):
        inv = np.argsort(perm)
        correct = total = 0
        for (X, y), states in zip(sequences, state_seqs):
            decoded = inv[model.viterbi(X, y)]
            correct += int((decoded == states).sum())
            total += states.shape[0]
        best_acc = max(best_acc, correct / total)
    assert best_acc > 0.8


# Pipeline (input pickle -> per-session sequences -> held-out K-selection)
def _write_synthetic_manifold_pickle(path, feature_names, history_frames, rng):
    """
    Build a small continuous-manifold input pickle with a 2-state torus GLM-HMM
    structure: six sessions (one reserved held-out), each a Markov chain over two
    manifold positions, stored in the `D[feature][session]={'X','Y','w',
    'supercategory'}` shape the pipeline consumes.
    """
    positions = np.array([[0.2, 0.2], [0.75, 0.75]])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    session_ids = [f"s{i}" for i in range(6)]
    raw = {feat: {} for feat in feature_names}
    for session_id in session_ids:
        seq_len = int(rng.integers(55, 75))
        states = np.empty(seq_len, dtype=int)
        states[0] = rng.integers(2)
        for t in range(1, seq_len):
            states[t] = rng.choice(2, p=transition[states[t - 1]])
        y = np.mod(positions[states] + 0.03 * rng.standard_normal((seq_len, 2)), 1.0)
        for feat in feature_names:
            raw[feat][session_id] = {
                'X': rng.standard_normal((seq_len, history_frames)),
                'Y': y,
                'w': np.ones(seq_len),
                'supercategory': states.astype(np.float64),
            }
    raw['_input_metadata'] = {
        'analysis_type': 'continuous',
        'analysis_specific': {'manifold_metric': 'torus', 'manifold_period': 1.0},
        'held_out_session_ids': ['s5'],
    }
    with open(path, 'wb') as handle:
        pickle.dump(raw, handle)
    return session_ids


def _write_glm_hmm_settings(path):
    """Shipped modeling settings with the glm_hmm block trimmed for the synthetic run."""
    settings = json.loads(REPO_SETTINGS.read_text())
    settings['glm_hmm']['n_states_min'] = 1
    settings['glm_hmm']['n_states_max'] = 3
    settings['glm_hmm']['n_restarts'] = 2
    settings['glm_hmm']['n_em_iters'] = 25
    settings['glm_hmm']['cv_folds'] = 2
    settings['glm_hmm']['multinomial_max_iter'] = 60
    settings['model_validation']['random_seed'] = 0
    path.write_text(json.dumps(settings))


def _write_model_selection_file(path, feature_names):
    """A minimal finalized manifold-selection pickle carrying `final_model_features`."""
    with open(path, 'wb') as handle:
        pickle.dump({'final_model_features': list(feature_names),
                     'step_idx': len(feature_names)}, handle)


def test_glm_hmm_pipeline_runs_and_selects_states(tmp_path):
    """
    End-to-end: the pipeline builds per-session sequences, holds out the reserved
    session, fits the GLM-HMM over the state-count range, selects a K by held-out
    log-likelihood, and writes a results pickle with a state path per dev session.
    """
    rng = np.random.default_rng(7)
    feature_names = ['featA', 'featB']
    history_frames = 3
    input_pkl = tmp_path / 'manifold_input.pkl'
    _write_synthetic_manifold_pickle(input_pkl, feature_names, history_frames, rng)
    settings_json = tmp_path / 'settings.json'
    _write_glm_hmm_settings(settings_json)
    selection_pkl = tmp_path / 'model_selection_final.pkl'
    _write_model_selection_file(selection_pkl, feature_names)
    out_dir = tmp_path / 'glm_hmm_out'

    results = run_glm_hmm_state_selection(
        input_data_path=str(input_pkl),
        settings_path=str(settings_json),
        output_directory=str(out_dir),
        model_selection_path=str(selection_pkl),
    )

    assert results['selection_criterion'] == 'cv_log_likelihood'
    assert 1 <= results['selected_n_states'] <= 3
    assert [row['n_states'] for row in results['selection_table']] == [1, 2, 3]
    assert all(np.isfinite(row['cv_log_likelihood']) for row in results['selection_table'])
    # A transition matrix of the selected size with rows summing to 1.
    A = results['transition_matrix']
    assert A.shape == (results['selected_n_states'], results['selected_n_states'])
    assert np.allclose(A.sum(axis=1), 1.0)
    # A Viterbi state path for every development session (the 5 non-held-out ones).
    assert set(results['state_paths']) == {'s0', 's1', 's2', 's3', 's4'}
    # The emission features were read from the model-selection output, not settings.
    assert results['metadata']['emission_features'] == feature_names
    # The results pickle was written to disk.
    assert list(out_dir.glob('glm_hmm_states_*.pkl'))


def test_glm_hmm_pipeline_runs_multinomial(tmp_path):
    """
    The pipeline also runs end-to-end with the multinomial emission: it reads the
    categorical target ('supercategory'), holds out the reserved session, selects
    a state count, and writes results whose metadata records the categorical run.
    """
    rng = np.random.default_rng(8)
    feature_names = ['featA', 'featB']
    input_pkl = tmp_path / 'manifold_input.pkl'
    _write_synthetic_manifold_pickle(input_pkl, feature_names, 3, rng)
    settings_json = tmp_path / 'settings.json'
    _write_glm_hmm_settings(settings_json)
    settings = json.loads(settings_json.read_text())
    settings['glm_hmm']['emission_type'] = 'multinomial'
    settings['glm_hmm']['n_states_max'] = 2
    settings_json.write_text(json.dumps(settings))
    selection_pkl = tmp_path / 'model_selection_final.pkl'
    _write_model_selection_file(selection_pkl, feature_names)
    out_dir = tmp_path / 'glm_hmm_out'

    results = run_glm_hmm_state_selection(
        input_data_path=str(input_pkl),
        settings_path=str(settings_json),
        output_directory=str(out_dir),
        model_selection_path=str(selection_pkl),
    )

    assert results['metadata']['emission_type'] == 'multinomial'
    assert results['metadata']['target_key'] == 'supercategory'
    assert 1 <= results['selected_n_states'] <= 2
    assert set(results['state_paths']) == {'s0', 's1', 's2', 's3', 's4'}
    assert list(out_dir.glob('glm_hmm_states_multinomial_supercategory_*.pkl'))


def test_glm_hmm_pipeline_rejects_unknown_emission(tmp_path):
    """An unsupported emission_type is surfaced with a clear error."""
    rng = np.random.default_rng(9)
    feature_names = ['featA']
    input_pkl = tmp_path / 'manifold_input.pkl'
    _write_synthetic_manifold_pickle(input_pkl, feature_names, 3, rng)
    settings_json = tmp_path / 'settings.json'
    _write_glm_hmm_settings(settings_json)
    settings = json.loads(settings_json.read_text())
    settings['glm_hmm']['emission_type'] = 'nonsense'
    settings_json.write_text(json.dumps(settings))
    selection_pkl = tmp_path / 'model_selection_final.pkl'
    _write_model_selection_file(selection_pkl, feature_names)

    with pytest.raises(ValueError, match="emission_type"):
        run_glm_hmm_state_selection(
            input_data_path=str(input_pkl),
            settings_path=str(settings_json),
            output_directory=str(tmp_path / 'out'),
            model_selection_path=str(selection_pkl),
        )


def test_read_selected_features_from_directory_and_error_paths(tmp_path):
    """The reader accepts the selection directory (highest step) and fails loudly."""
    # A directory holding two step pickles; the highest-numbered is the final one.
    _write_model_selection_file(tmp_path / 'ms_step_0.pkl', ['early'])
    _write_model_selection_file(tmp_path / 'ms_step_2.pkl', ['neck', 'nose'])
    assert _read_selected_features(str(tmp_path)) == ['neck', 'nose']

    # A pickle without `final_model_features` (an unfinished selection) is rejected.
    unfinished = tmp_path / 'unfinished.pkl'
    with open(unfinished, 'wb') as handle:
        pickle.dump({'step_idx': 1, 'current_features': ['x']}, handle)
    with pytest.raises(ValueError, match="final_model_features"):
        _read_selected_features(str(unfinished))

    # An empty selection (nothing kept) is rejected -- no emission to build.
    empty = tmp_path / 'empty.pkl'
    _write_model_selection_file(empty, [])
    with pytest.raises(ValueError, match="kept no features"):
        _read_selected_features(str(empty))


# Input-driven-transition engine (direct-marginal, reference-coded)
def _simulate_input_driven_categorical(rng, n_features, n_time_bins, n_classes,
                                       n_seqs, seq_len):
    """
    Two states with DISTINCT behaviour -> USV-category rules (each its own light
    X-dependence plus a strong opposing base rate), emitted along a sticky Markov
    chain. Returns ``(sequences, state_seqs)`` in the engine's ``(X, y)`` shape.
    """
    n_inputs = n_features * n_time_bins
    weights = [0.5 * rng.standard_normal((n_inputs, n_classes)) for _ in range(2)]
    # Each state favours a different category via its base rate (state s -> class s),
    # generalised to any n_classes: the preferred class gets a strong +ve intercept.
    biases = []
    for state in range(2):
        bias = np.full(n_classes, -1.0)
        bias[state] = 3.0
        biases.append(bias)
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    state_seqs = _simulate_markov_states(n_seqs, seq_len, transition, rng)
    sequences = []
    for states in state_seqs:
        n = states.shape[0]
        X = rng.standard_normal((n, n_inputs))
        y = np.array([int(np.argmax(X[t] @ weights[states[t]] + biases[states[t]]))
                      for t in range(n)])
        sequences.append((X, y.astype(np.int64)))
    return sequences, state_seqs


def _viterbi_best_accuracy(model, sequences, state_seqs):
    """Best-of-both-labellings Viterbi decoding accuracy against the true chain."""
    best_acc = 0.0
    for perm in ([0, 1], [1, 0]):
        inv = np.argsort(perm)
        correct = total = 0
        for (X, y), states in zip(sequences, state_seqs):
            decoded = inv[model.viterbi(X, y)]
            correct += int((decoded == states).sum())
            total += states.shape[0]
        best_acc = max(best_acc, correct / total)
    return best_acc


def test_input_driven_glmhmm_recovers_states_without_collapse():
    """The direct-marginal engine holds two distinct states (no EM-style collapse)."""
    rng = np.random.default_rng(21)
    n_features, n_time_bins, n_classes = 2, 2, 2
    sequences, state_seqs = _simulate_input_driven_categorical(
        rng, n_features, n_time_bins, n_classes, n_seqs=10, seq_len=70)
    model = InputDrivenGLMHMM(n_states=2, n_features=n_features, n_time_bins=n_time_bins,
                              n_classes=n_classes, lambda_smooth=0.05, n_restarts=2,
                              n_lbfgs=200, random_state=0).fit(sequences)
    # Both states are actually used -- the failure the direct-marginal fit fixes is a
    # collapse to one state (an all-zero occupancy on the other).
    decoded = np.concatenate([model.viterbi(X, y) for X, y in sequences])
    occupancy = np.bincount(decoded, minlength=2) / decoded.size
    assert occupancy.min() > 0.1
    # And the recovered states match the truth (up to relabelling).
    assert _viterbi_best_accuracy(model, sequences, state_seqs) > 0.8


def test_input_driven_glmhmm_reference_coding_shapes_and_heldout_selection():
    """Fitted params carry the reference-coded shapes; held-out restart selection runs."""
    rng = np.random.default_rng(22)
    n_features, n_time_bins, n_classes, n_states = 2, 2, 3, 2
    sequences, _ = _simulate_input_driven_categorical(
        rng, n_features, n_time_bins, n_classes, n_seqs=8, seq_len=60)
    model = InputDrivenGLMHMM(n_states=n_states, n_features=n_features,
                              n_time_bins=n_time_bins, n_classes=n_classes,
                              lambda_smooth=0.05, n_restarts=2, n_lbfgs=150,
                              random_state=0)
    model.fit(sequences[:6], held_out_sequences=sequences[6:])
    design_width = n_features * n_time_bins
    params = model.params_
    # Reference coding: the emission has C - 1 free classes (class 0 pinned to 0);
    # the transition is K x D x K (the self logit is pinned in the loss).
    assert np.asarray(params['W_emit']).shape == (n_states, design_width, n_classes - 1)
    assert np.asarray(params['b_emit']).shape == (n_states, n_classes - 1)
    assert np.asarray(params['W_trans']).shape == (n_states, design_width, n_states)
    assert np.asarray(params['b_trans']).shape == (n_states, n_states)
    assert np.asarray(params['log_pi']).shape == (n_states,)
    assert np.isfinite(model.log_likelihood(sequences))
    assert np.isfinite(model.bic(sequences))


def test_input_driven_glmhmm_bic_prefers_true_state_count_over_one():
    """BIC (lower better) favours K=2 over K=1 on 2-state input-driven data."""
    rng = np.random.default_rng(23)
    n_features, n_time_bins, n_classes = 2, 2, 2
    sequences, _ = _simulate_input_driven_categorical(
        rng, n_features, n_time_bins, n_classes, n_seqs=8, seq_len=70)
    bics = {}
    for k in (1, 2):
        model = InputDrivenGLMHMM(n_states=k, n_features=n_features,
                                  n_time_bins=n_time_bins, n_classes=n_classes,
                                  lambda_smooth=0.05, n_restarts=2, n_lbfgs=150,
                                  random_state=0).fit(sequences)
        bics[k] = model.bic(sequences)
    assert bics[2] < bics[1]


def test_input_driven_glmhmm_fit_requires_sequences():
    """An empty sequence list is rejected with a clear error."""
    model = InputDrivenGLMHMM(n_states=2, n_features=2, n_time_bins=2, n_classes=2)
    with pytest.raises(ValueError, match="at least one sequence"):
        model.fit([])


def test_glmhmm_input_driven_em_machinery_runs():
    """
    The EM engine's ``transition_mode='input_driven'`` fits per-"from"-state
    transition GLMs and produces valid per-timestep transition distributions (rows
    sum to 1). Direct-marginal :class:`InputDrivenGLMHMM` is the primary engine for
    this model; this guards that the EM-side machinery stays wired and correct.
    """
    rng = np.random.default_rng(24)
    n_features, n_time_bins, n_classes = 2, 2, 2
    sequences, _ = _simulate_input_driven_categorical(
        rng, n_features, n_time_bins, n_classes, n_seqs=8, seq_len=60)
    clf_kwargs = dict(n_features=n_features, n_time_bins=n_time_bins,
                      lambda_smooth=1.0, l2_reg=0.1, focal_gamma=0.0, max_iter=120)
    trans_kwargs = dict(n_features=n_features, n_time_bins=n_time_bins,
                        lambda_smooth=1.0, l2_reg=0.1, focal_gamma=0.0,
                        uniform_class_weights=True, max_iter=120)
    model = GLMHMM(
        n_states=2, emission_factory=lambda: MultinomialEmission(clf_kwargs),
        n_em_iters=20, n_restarts=1, tol=1e-4, random_state=0,
        transition_mode='input_driven',
        transition_factory=lambda: SmoothMultinomialLogisticRegression(**trans_kwargs))
    model.fit(sequences)
    assert model.transition_glms_ is not None
    # Per-timestep transition tensor is a stack of valid distributions.
    log_A_seq = model._seq_log_transition(sequences[0][0])
    transition = np.exp(log_A_seq)
    assert transition.shape[1:] == (2, 2)
    assert np.allclose(transition.sum(axis=2), 1.0, atol=1e-5)
    assert np.isfinite(model.log_likelihood(sequences))


def test_glmhmm_input_driven_requires_transition_factory():
    """Selecting input-driven transitions without a factory fails loudly at construction."""
    with pytest.raises(ValueError, match="requires a transition_factory"):
        GLMHMM(n_states=2, emission_factory=lambda: GaussianEmission(n_targets=1),
               transition_mode='input_driven')


def _simulate_input_driven_torus(rng, n_features, n_time_bins, n_seqs, seq_len):
    """
    Two states at distinct torus positions along a sticky Markov chain. Returns
    ``(sequences, state_seqs)`` in the manifold engine's ``(X, y)`` shape, with ``y``
    a ``(T, 2)`` torus position.
    """
    positions = np.array([[0.2, 0.2], [0.75, 0.75]])
    transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    state_seqs = _simulate_markov_states(n_seqs, seq_len, transition, rng)
    sequences = []
    for states in state_seqs:
        n = states.shape[0]
        X = rng.standard_normal((n, n_features * n_time_bins))
        y = np.mod(positions[states] + 0.03 * rng.standard_normal((n, 2)), 1.0)
        sequences.append((X.astype(np.float32), y.astype(np.float32)))
    return sequences, state_seqs


def test_input_driven_manifold_glmhmm_recovers_torus_states_without_collapse():
    """The direct-marginal manifold engine holds two distinct torus states (no collapse)."""
    rng = np.random.default_rng(31)
    n_features, n_time_bins = 2, 3
    sequences, state_seqs = _simulate_input_driven_torus(
        rng, n_features, n_time_bins, n_seqs=8, seq_len=70)
    model = InputDrivenManifoldGLMHMM(n_states=2, n_features=n_features,
                                      n_time_bins=n_time_bins, period=1.0,
                                      lambda_smooth=0.05, n_restarts=2, n_lbfgs=250,
                                      random_state=0).fit(sequences)
    decoded = np.concatenate([model.viterbi(X, y) for X, y in sequences])
    assert (np.bincount(decoded, minlength=2) / decoded.size).min() > 0.1
    # Each state has a positive von Mises concentration.
    assert np.all(np.exp(np.asarray(model.params_['log_kappa'])) > 0.0)
    assert _viterbi_best_accuracy(model, sequences, state_seqs) > 0.8


def test_input_driven_manifold_glmhmm_shapes_and_bic():
    """Fitted manifold params carry the (K, D, 2) emission shape; log_likelihood/bic finite."""
    rng = np.random.default_rng(32)
    n_features, n_time_bins, n_states = 2, 3, 2
    sequences, _ = _simulate_input_driven_torus(
        rng, n_features, n_time_bins, n_seqs=6, seq_len=60)
    model = InputDrivenManifoldGLMHMM(n_states=n_states, n_features=n_features,
                                      n_time_bins=n_time_bins, period=1.0,
                                      lambda_smooth=0.05, n_restarts=2, n_lbfgs=150,
                                      random_state=0)
    model.fit(sequences[:4], held_out_sequences=sequences[4:])
    design_width = n_features * n_time_bins
    params = model.params_
    assert np.asarray(params['W_emit']).shape == (n_states, design_width, 2)
    assert np.asarray(params['b_emit']).shape == (n_states, 2)
    assert np.asarray(params['log_kappa']).shape == (n_states,)
    assert np.asarray(params['W_trans']).shape == (n_states, design_width, n_states)
    assert np.isfinite(model.log_likelihood(sequences))
    assert np.isfinite(model.bic(sequences))


def test_input_driven_manifold_glmhmm_kappa_mode_validation_and_global_mle():
    """`kappa_mode` is validated, and the default global-MLE mode fits a finite,
    strictly-positive von Mises concentration -- the guard against the free-`log_kappa`
    collapse to the uniform torus density (kappa -> 0)."""
    with pytest.raises(ValueError, match="kappa_mode"):
        InputDrivenManifoldGLMHMM(n_states=2, n_features=2, n_time_bins=3,
                                  kappa_mode='bogus')
    rng = np.random.default_rng(41)
    n_features, n_time_bins = 2, 3
    sequences, _ = _simulate_input_driven_torus(
        rng, n_features, n_time_bins, n_seqs=8, seq_len=70)
    model = InputDrivenManifoldGLMHMM(n_states=2, n_features=n_features,
                                      n_time_bins=n_time_bins, period=1.0,
                                      lambda_smooth=0.05, n_restarts=2, n_lbfgs=200,
                                      random_state=0, kappa_mode='global_mle').fit(sequences)
    kappa = np.exp(np.asarray(model.params_['log_kappa']))
    assert np.all(np.isfinite(kappa))
    assert np.all(kappa > 0.5)          # well clear of the uniform-density collapse
    assert np.isfinite(model.log_likelihood(sequences))


def test_input_driven_manifold_glmhmm_region_reweighting_changes_the_fit():
    """Per-event weights (the inverse-region-frequency balancing hook) are threaded
    into the emission objective: a non-uniform weighting yields a different fit than the
    unweighted one under an identical seed and restart schedule."""
    rng = np.random.default_rng(42)
    n_features, n_time_bins = 2, 3
    sequences, _ = _simulate_input_driven_torus(
        rng, n_features, n_time_bins, n_seqs=8, seq_len=70)
    # Upweight the second half of every sequence's events (a stand-in for region balancing).
    weighted = [(X, y, np.where(np.arange(len(y)) >= len(y) // 2, 4.0, 1.0).astype(np.float32))
                for X, y in sequences]
    shared = dict(n_states=2, n_features=n_features, n_time_bins=n_time_bins, period=1.0,
                  lambda_smooth=0.05, n_restarts=2, n_lbfgs=200, random_state=0)
    unweighted_model = InputDrivenManifoldGLMHMM(**shared).fit(sequences)
    weighted_model = InputDrivenManifoldGLMHMM(**shared).fit(weighted)
    assert not np.allclose(np.asarray(unweighted_model.params_['W_emit']),
                           np.asarray(weighted_model.params_['W_emit']))


def test_input_driven_manifold_glmhmm_macro_score_finite_and_region_gated():
    """`macro_score` returns a finite region-balanced von Mises log-score for a fitted
    model, and NaN when no region clears `min_region_events`."""
    rng = np.random.default_rng(43)
    n_features, n_time_bins = 2, 3
    sequences, _ = _simulate_input_driven_torus(
        rng, n_features, n_time_bins, n_seqs=8, seq_len=70)
    model = InputDrivenManifoldGLMHMM(n_states=2, n_features=n_features,
                                      n_time_bins=n_time_bins, period=1.0,
                                      lambda_smooth=0.05, n_restarts=2, n_lbfgs=200,
                                      random_state=0).fit(sequences)
    # Two supercategory regions, split by the first torus axis (near each true state).
    region_labels = [np.where(y[:, 0] < 0.5, 0.0, 1.0) for _, y in sequences]
    score = model.macro_score(sequences, region_labels, min_region_events=1)
    assert np.isfinite(score)
    # With a per-region event floor no region can clear, the macro average is empty -> NaN
    # (all-NaN labels would instead degrade to the pooled score, so they don't test this gate).
    assert np.isnan(model.macro_score(sequences, region_labels, min_region_events=10_000))


def test_glm_hmm_pipeline_runs_input_driven_multinomial(tmp_path):
    """
    End-to-end with the input-driven engine: the pipeline routes the categorical
    path to InputDrivenGLMHMM, selects a state count, and records the input-driven
    transition mode with a valid initial-state vector and mean-transition summary.
    """
    rng = np.random.default_rng(12)
    feature_names = ['featA', 'featB']
    input_pkl = tmp_path / 'manifold_input.pkl'
    _write_synthetic_manifold_pickle(input_pkl, feature_names, 3, rng)
    settings_json = tmp_path / 'settings.json'
    _write_glm_hmm_settings(settings_json)
    settings = json.loads(settings_json.read_text())
    settings['glm_hmm']['emission_type'] = 'multinomial'
    settings['glm_hmm']['transition_mode'] = 'input_driven'
    settings['glm_hmm']['n_states_max'] = 2
    settings['glm_hmm']['n_lbfgs'] = 120
    settings_json.write_text(json.dumps(settings))
    selection_pkl = tmp_path / 'model_selection_final.pkl'
    _write_model_selection_file(selection_pkl, feature_names)
    out_dir = tmp_path / 'glm_hmm_out'

    results = run_glm_hmm_state_selection(
        input_data_path=str(input_pkl),
        settings_path=str(settings_json),
        output_directory=str(out_dir),
        model_selection_path=str(selection_pkl),
    )

    assert results['metadata']['transition_mode'] == 'input_driven'
    assert results['metadata']['emission_type'] == 'multinomial'
    assert results['metadata']['n_classes'] == 2
    assert 1 <= results['selected_n_states'] <= 2
    # A valid initial-state distribution and a representative mean transition matrix.
    assert np.isclose(results['log_pi'].sum(), 1.0)
    transition = results['transition_matrix']
    assert transition.shape == (results['selected_n_states'], results['selected_n_states'])
    assert np.allclose(transition.sum(axis=1), 1.0, atol=1e-5)
    assert set(results['state_paths']) == {'s0', 's1', 's2', 's3', 's4'}
    assert list(out_dir.glob('glm_hmm_states_multinomial_supercategory_*.pkl'))


def test_glm_hmm_pipeline_runs_input_driven_manifold(tmp_path):
    """
    End-to-end with the input-driven MANIFOLD engine: emission_type='manifold' +
    transition_mode='input_driven' routes the torus path to InputDrivenManifoldGLMHMM,
    selects a state count, and records the input-driven mode with valid summaries.
    """
    rng = np.random.default_rng(13)
    feature_names = ['featA', 'featB']
    input_pkl = tmp_path / 'manifold_input.pkl'
    _write_synthetic_manifold_pickle(input_pkl, feature_names, 3, rng)
    settings_json = tmp_path / 'settings.json'
    _write_glm_hmm_settings(settings_json)
    settings = json.loads(settings_json.read_text())
    settings['glm_hmm']['emission_type'] = 'manifold'            # torus, from the pickle
    settings['glm_hmm']['transition_mode'] = 'input_driven'
    settings['glm_hmm']['n_states_max'] = 2
    settings['glm_hmm']['n_lbfgs'] = 120
    settings_json.write_text(json.dumps(settings))
    selection_pkl = tmp_path / 'model_selection_final.pkl'
    _write_model_selection_file(selection_pkl, feature_names)
    out_dir = tmp_path / 'glm_hmm_out'

    results = run_glm_hmm_state_selection(
        input_data_path=str(input_pkl),
        settings_path=str(settings_json),
        output_directory=str(out_dir),
        model_selection_path=str(selection_pkl),
    )

    assert results['metadata']['transition_mode'] == 'input_driven'
    assert results['metadata']['emission_type'] == 'manifold'
    assert 1 <= results['selected_n_states'] <= 2
    assert np.isclose(results['log_pi'].sum(), 1.0)
    transition = results['transition_matrix']
    assert np.allclose(transition.sum(axis=1), 1.0, atol=1e-5)
    assert set(results['state_paths']) == {'s0', 's1', 's2', 's3', 's4'}
    assert list(out_dir.glob('glm_hmm_states_manifold_torus_*.pkl'))
