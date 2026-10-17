"""
@author: bartulem

GLM-HMM over latent vocal states.

A hidden Markov model whose per-state emission is a behaviour -> vocalization
GLM: each latent state owns its own mapping from the behavioural history window
``X`` to the vocalization it accompanies, so the states are distinct
"behaviour-to-vocalization rules" the animal switches between over a vocal
bout. The Markov chain over states is stationary (a single ``K x K``
transition matrix); the GLM lives in the emissions, following the classic
Ashwood/Pillow GLM-HMM factorization.

The engine is emission-agnostic: forward-backward, Baum-Welch EM (with random
restarts), Viterbi decoding, and held-out / BIC state-count selection all operate
on a per-state emission object that exposes three things -- ``fit`` (a
responsibility-weighted M-step refit), ``log_density`` (the per-sample E-step
log-likelihood), and ``n_params`` (for BIC). Three emissions are provided:

* :class:`GaussianEmission` -- a diagonal Gaussian on the target, independent of
  ``X`` (states differ only in their target distribution). This is the reference
  emission for validating the HMM machinery and a simple baseline.
* :class:`ManifoldEmission` -- a per-state torus/euclidean regression
  (:class:`SmoothTorusManifoldRegression` / :class:`SmoothBivariateRegression`)
  with a von Mises observation density, reusing the Project-2 selection
  machinery. This is the acoustic-manifold GLM-HMM.
* :class:`MultinomialEmission` -- a per-state multinomial logistic regression
  (:class:`SmoothMultinomialLogisticRegression`) with a categorical observation
  density, for the USV-category GLM-HMM. Its responsibility-weighted M-step uses
  the estimator's per-sample ``sample_weight`` channel.

Sequences are per recording session: the USV events of a session, IN TEMPORAL
ONSET ORDER (the modeling input pickle already stores them onset-sorted), are one
observation sequence; transitions are between consecutive vocalizations.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import lax
from jax.scipy.special import i0e as _jax_i0e
from jax.scipy.special import logsumexp as _jax_logsumexp
from scipy.special import i0e as _np_i0e
from sklearn.cluster import KMeans

from .jax_multinomial_logistic_regression import SmoothMultinomialLogisticRegression
from .manifold_metric import (
    _fit_von_mises_kappa,
    macro_von_mises_logscore,
    signed_diff,
    von_mises_logpdf_per_point,
)
from .manifold_torus_regression import resolve_manifold_regressor_cls

# A floor on the log emission density so a single all-but-impossible observation
# cannot send a whole sequence's log-likelihood to -inf and stall the EM (it only
# guards genuinely degenerate densities; normal densities are far above it).
_LOG_DENSITY_FLOOR = -1e6

# The minimum total posterior responsibility a state must retain for its emission to
# be refit in the M-step. A state that has collapsed to (near) zero responsibility
# would otherwise be refit on an all-but-zero ``sample_weight`` vector, which
# degenerates the weighted MLE into a NaN emission that then poisons the whole
# model's log-likelihood. Below this floor the state keeps its previous emission.
_MIN_STATE_RESPONSIBILITY = 1.0


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Description
    -----------
    Numerically-stable log-sum-exp along one axis. A lightweight numpy
    replacement for ``scipy.special.logsumexp`` in the forward-backward inner
    loop, where the per-call validation overhead of the scipy version dominates
    on the many tiny ``(K, K)`` reductions of a long sequence.

    Parameters
    ----------
    a (np.ndarray)
        Input array of log-values.
    axis (int)
        Axis to reduce.

    Returns
    -------
    np.ndarray
        ``log(sum(exp(a), axis))`` with the reduced axis removed; the max is
        subtracted for stability, and an all ``-inf`` slice yields ``-inf``
        rather than a NaN.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    a_max = np.where(np.isfinite(a_max), a_max, 0.0)
    result = np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)) + a_max
    return np.squeeze(result, axis=axis)


def _kmeans_init_responsibilities(y_all: np.ndarray, n_states: int, seed: int) -> np.ndarray:
    """
    Description
    -----------
    Structured initial state responsibilities from k-means on the targets.

    Random per-sample responsibilities average out over many samples, so every
    state's first M-step fits ~the marginal and the EM sits at the degenerate
    "all states identical" fixed point when the emissions are only weakly
    separated. Seeding instead from a k-means clustering of the targets gives each
    state a genuinely distinct starting emission, which the E-step can then
    amplify. Continuous targets are clustered directly, whether they arrive as
    ``(n, d)`` or as a flat ``(n,)`` scalar; a categorical target (flat, with an
    integer or boolean dtype) is one-hot-encoded first so events group by class.
    The categorical branch is selected by dtype rather than by shape, so a
    continuous scalar target is never mistaken for ``n`` distinct classes.

    Parameters
    ----------
    y_all (np.ndarray)
        Pooled targets across sequences: ``(n, d)`` continuous, ``(n,)`` continuous
        (floating dtype), or ``(n,)`` categorical (integer or boolean dtype).
    n_states (int)
        Number of latent states / clusters.
    seed (int)
        Deterministic k-means seed (varied per EM restart).

    Returns
    -------
    responsibilities (np.ndarray)
        ``(n, n_states)`` near-hard responsibilities (a small floor keeps every
        state's weight strictly positive), rows summing to 1.
    """
    y_arr = np.asarray(y_all)
    # A flat target is routed by DTYPE, not by shape. Only a floating (or complex)
    # dtype is read as continuous and clustered on its own value; every other flat
    # dtype -- integer, boolean, string, object -- is a class label and is one-hot
    # encoded so events group by class. Branching on `ndim` alone silently treated
    # a continuous scalar target as categorical and asked for an
    # `(n, n_distinct_values)` one-hot -- for a real cohort that is an `(n, n)`
    # matrix (hundreds of gigabytes) and the process dies to the OOM killer with no
    # traceback, because every sample is its own "class". The test is written as
    # "floating is continuous" rather than "integer is categorical" so that a label
    # array of strings or objects keeps its original, working behaviour.
    if y_arr.ndim == 1 and y_arr.dtype.kind in 'fc':
        features = y_arr.astype(np.float64)[:, None]
    elif y_arr.ndim == 1:
        classes = np.unique(y_arr)
        features = np.zeros((y_arr.shape[0], classes.size), dtype=np.float64)
        features[np.arange(y_arr.shape[0]), np.searchsorted(classes, y_arr)] = 1.0
    else:
        features = y_arr.astype(np.float64)
    labels = KMeans(n_clusters=n_states, n_init=3, random_state=seed).fit_predict(features)
    responsibilities = np.full((y_arr.shape[0], n_states), 1e-3, dtype=np.float64)
    responsibilities[np.arange(y_arr.shape[0]), labels] = 1.0
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities


def _softmax_free_params(classifier) -> int:
    """
    Description
    -----------
    Number of FREE parameters of a fitted full-softmax multinomial-logistic
    classifier (:class:`SmoothMultinomialLogisticRegression`): ``coef_`` is
    ``(C, d)`` and ``intercept_`` ``(C,)``, but adding one vector to every class
    leaves the softmax unchanged, so one class's ``d + 1`` parameters are
    redundant and ``(C - 1) * (d + 1)`` are free.

    Parameters
    ----------
    classifier (object)
        A fitted classifier exposing ``coef_`` ``(C, d)`` and ``intercept_`` ``(C,)``.

    Returns
    -------
    n_free (int)
        ``coef_.size + intercept_.size - (d + 1)``.
    """
    coef = np.asarray(classifier.coef_)
    intercept = np.asarray(classifier.intercept_)
    return int(coef.size + intercept.size - (coef.shape[1] + 1))


class _BaseEmission:
    """
    Description
    -----------
    Abstract per-state emission for the GLM-HMM. A concrete emission maps the
    behavioural design matrix ``X`` and the observed vocalization ``y`` to a
    per-sample log-density, and refits itself under per-sample responsibilities.

    Subclasses must implement :meth:`fit`, :meth:`log_density` and the
    :attr:`n_params` property.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "_BaseEmission":
        """
        Description
        -----------
        Responsibility-weighted M-step refit; returns self.

        Parameters
        ----------
        X (np.ndarray)
            ``(n, d)`` behavioural design matrix (pooled over all sequences).
        y (np.ndarray)
            Observed targets aligned to the rows of ``X``.
        sample_weight (np.ndarray)
            ``(n,)`` non-negative per-sample weights (the state's E-step
            responsibilities).

        Returns
        -------
        self (_BaseEmission)
            The refit emission. The base class raises ``NotImplementedError``.
        """
        raise NotImplementedError

    def log_density(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Per-sample log-density (E-step emission log-likelihood), shape ``(n,)``.

        Parameters
        ----------
        X (np.ndarray)
            ``(n, d)`` behavioural design matrix.
        y (np.ndarray)
            Observed targets aligned to the rows of ``X``.

        Returns
        -------
        log_density (np.ndarray)
            ``(n,)`` per-sample log-density under this emission. The base class raises
            ``NotImplementedError``.
        """
        raise NotImplementedError

    @property
    def n_params(self) -> int:
        """
        Description
        -----------
        Number of free parameters in this emission (for BIC).

        Parameters
        ----------
        None

        Returns
        -------
        n_params (int)
            Free-parameter count. The base class raises ``NotImplementedError``.
        """
        raise NotImplementedError


class GaussianEmission(_BaseEmission):
    """
    Description
    -----------
    Diagonal-covariance Gaussian emission on the target, independent of ``X``.
    Each state is a Gaussian ``N(mean, diag(var))`` over the observed target, so
    states differ purely in where (and how tightly) their vocalizations sit. This
    is the reference emission for validating the HMM machinery and a simple
    (non-GLM) baseline; it does not use the behavioural design matrix.

    Parameters
    ----------
    n_targets (int)
        Dimensionality ``d`` of the target ``y``.
    min_var (float)
        Variance floor so a near-degenerate state cannot collapse to a spike.

    Returns
    -------
    None
    """

    def __init__(self, n_targets: int, min_var: float = 1e-4) -> None:
        """
        Description
        -----------
        Store the target dimensionality and variance floor; the fitted mean and
        variance (``mean_``, ``var_``) start as ``None`` until :meth:`fit` runs.

        Parameters
        ----------
        n_targets (int)
            Dimensionality ``d`` of the target ``y`` (cast to int).
        min_var (float)
            Variance floor so a near-degenerate state cannot collapse to a spike
            (cast to float); defaults to ``1e-4``.

        Returns
        -------
        None
        """
        self.n_targets = int(n_targets)
        self.min_var = float(min_var)
        self.mean_ = None
        self.var_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "GaussianEmission":
        """
        Description
        -----------
        Weighted MLE of the mean and diagonal variance. When the weights sum to zero
        or less (no responsibility mass) the pooled unweighted moments are used so the
        state stays defined; the variance is floored at ``min_var``.

        Parameters
        ----------
        X (np.ndarray)
            Behavioural design matrix; unused (this emission is independent of ``X``)
            and kept only for the shared emission interface.
        y (np.ndarray)
            ``(n, d)`` observed targets.
        sample_weight (np.ndarray)
            ``(n,)`` per-sample weights (the state's responsibilities).

        Returns
        -------
        self (GaussianEmission)
            The fitted emission, with ``mean_`` and ``var_`` of shape ``(d,)``.
        """
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(sample_weight, dtype=np.float64)
        total = float(w.sum())
        if total <= 0.0:
            # No responsibility mass; fall back to the pooled unweighted moments
            # so the state stays defined rather than producing NaNs.
            w = np.ones(y.shape[0], dtype=np.float64)
            total = float(w.sum())
        self.mean_ = (w[:, None] * y).sum(axis=0) / total
        centred = y - self.mean_[None, :]
        self.var_ = np.maximum((w[:, None] * centred ** 2).sum(axis=0) / total, self.min_var)
        return self

    def log_density(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Per-sample diagonal-Gaussian log-density.

        Parameters
        ----------
        X (np.ndarray)
            Behavioural design matrix; unused (kept for the shared emission interface).
        y (np.ndarray)
            ``(n, d)`` observed targets.

        Returns
        -------
        log_density (np.ndarray)
            ``(n,)`` log-density, summed over the ``d`` independent target dimensions.
        """
        y = np.asarray(y, dtype=np.float64)
        centred = y - self.mean_[None, :]
        return (-0.5 * (np.log(2.0 * np.pi * self.var_)[None, :]
                        + centred ** 2 / self.var_[None, :]).sum(axis=1))

    @property
    def n_params(self) -> int:
        """
        Description
        -----------
        ``d`` means + ``d`` variances.

        Parameters
        ----------
        None

        Returns
        -------
        n_params (int)
            ``2 * n_targets``.
        """
        return 2 * self.n_targets


class ManifoldEmission(_BaseEmission):
    """
    Description
    -----------
    Per-state behaviour -> acoustic-manifold-position emission: a smooth
    regression (torus or euclidean, via :func:`resolve_manifold_regressor_cls`)
    predicts the 2-D manifold coordinate from the behavioural history window, and
    the observation density is a product von Mises around that prediction with a
    state-specific concentration ``kappa`` fit on the state's
    responsibility-weighted residuals. This reuses the Project-2 selection
    machinery (the same regressor and the same von Mises density) as the GLM-HMM
    emission.

    Parameters
    ----------
    regressor_kwargs (dict)
        Keyword arguments for the manifold regressor constructor (``n_features``,
        ``n_time_bins``, ``lambda_smooth``, ``l2_reg``, ``metric``, ``period``,
        ...). ``metric`` / ``period`` also set the von Mises geometry.

    Returns
    -------
    None
    """

    def __init__(self, regressor_kwargs: dict) -> None:
        """
        Description
        -----------
        Copy the regressor keyword arguments, read the von Mises geometry
        (``metric``, ``period``) from them, and resolve the regressor class via
        :func:`resolve_manifold_regressor_cls`. The fitted regressor and concentration
        (``regressor_``, ``kappa_``) start as ``None`` until :meth:`fit` runs.

        Parameters
        ----------
        regressor_kwargs (dict)
            Keyword arguments for the manifold regressor constructor; must contain the
            ``'metric'`` and ``'period'`` keys (a ``KeyError`` is raised otherwise).

        Returns
        -------
        None
        """
        self.regressor_kwargs = dict(regressor_kwargs)
        self.metric = self.regressor_kwargs['metric']
        self.period = self.regressor_kwargs['period']
        self._regressor_cls = resolve_manifold_regressor_cls(self.metric)
        self.regressor_ = None
        self.kappa_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "ManifoldEmission":
        """
        Description
        -----------
        Responsibility-weighted M-step: refit the regressor with the per-sample
        responsibilities as ``sample_weight``, then fit the von Mises
        concentration on the (weighted) residuals.

        Parameters
        ----------
        X (np.ndarray)
            ``(n, d)`` behavioural design matrix.
        y (np.ndarray)
            ``(n, 2)`` observed manifold coordinates.
        sample_weight (np.ndarray)
            ``(n,)`` per-sample weights (the state's responsibilities); the same weight
            is applied to both coordinates in the concentration fit.

        Returns
        -------
        self (ManifoldEmission)
            The fitted emission, with a fresh ``regressor_`` and its ``kappa_``.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        w = np.asarray(sample_weight, dtype=np.float64)
        self.regressor_ = self._regressor_cls(**self.regressor_kwargs)
        self.regressor_.fit(X, y, sample_weight=w)
        residual = signed_diff(y, self.regressor_.predict(X),
                               metric=self.metric, period=self.period) * (2.0 * np.pi / self.period)
        # Weight both torus coordinates by the same per-event responsibility.
        self.kappa_ = _fit_von_mises_kappa(residual, weights=w[:, None])
        return self

    def log_density(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Per-sample product-von-Mises log-density around the regressor prediction.

        Parameters
        ----------
        X (np.ndarray)
            ``(n, d)`` behavioural design matrix.
        y (np.ndarray)
            ``(n, 2)`` observed manifold coordinates.

        Returns
        -------
        log_density (np.ndarray)
            ``(n,)`` per-sample log-density (:func:`von_mises_logpdf_per_point` at the
            fitted ``kappa_``).
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        return von_mises_logpdf_per_point(
            self.regressor_.predict(X), y,
            metric=self.metric, period=self.period, kappa=self.kappa_,
        )

    @property
    def n_params(self) -> int:
        """
        Description
        -----------
        Regressor coefficients + intercept + one concentration scalar.

        Parameters
        ----------
        None

        Returns
        -------
        n_params (int)
            ``coef_.size + intercept_.size + 1`` of the fitted regressor.
        """
        coef = np.asarray(self.regressor_.coef_)
        intercept = np.asarray(self.regressor_.intercept_)
        return int(coef.size + intercept.size + 1)


class MultinomialEmission(_BaseEmission):
    """
    Description
    -----------
    Per-state behaviour -> USV-category emission: a smooth multinomial logistic
    regression (:class:`SmoothMultinomialLogisticRegression`) predicts the
    categorical vocalization label from the behavioural history window, and the
    observation density is the predicted probability of the observed class. Each
    latent state owns its own classifier, refit in the M-step with the per-event
    state responsibilities as ``sample_weight`` (soft EM) and
    ``uniform_class_weights=True`` so the responsibilities -- not inverse-frequency
    class balancing -- drive the per-state fit.

    Because the M-step always fits on the full pooled data (every row present,
    only re-weighted), every state's classifier sees the same complete label set,
    so its ``classes_`` and the ``predict_proba`` column order are consistent
    across states -- which the forward-backward E-step relies on.

    A cross-validation training fold can, however, lack a category that its test
    fold carries. ``class_labels`` fixes the full label set before CV: a label in
    that set the classifier never trained on has zero predicted probability (scored
    at the ``log(1e-12)`` floor), and a label outside it raises.

    Parameters
    ----------
    classifier_kwargs (dict)
        Keyword arguments for the classifier constructor (``n_features``,
        ``n_time_bins``, ``lambda_smooth``, ``l2_reg``, ``focal_gamma``,
        ``max_iter``, ...). ``uniform_class_weights=True`` is forced internally.
    class_labels (np.ndarray or None)
        The full label set, fixed before cross-validation; ``None`` accepts only
        the labels the classifier trained on.

    Returns
    -------
    None
    """

    def __init__(self, classifier_kwargs: dict, class_labels: np.ndarray | None = None) -> None:
        """
        Description
        -----------
        Copy the classifier keyword arguments and force
        ``uniform_class_weights=True`` on the copy, and store the fixed label set;
        the fitted classifier (``classifier_``) starts as ``None`` until :meth:`fit`
        runs.

        Parameters
        ----------
        classifier_kwargs (dict)
            Keyword arguments for the :class:`SmoothMultinomialLogisticRegression`
            constructor (any ``uniform_class_weights`` entry is overwritten).
        class_labels (np.ndarray or None)
            Every label the emission may be asked to score (e.g. the labels of all
            development and reserved sessions); a label in it but absent from a
            training fold scores ``log(1e-12)``. ``None`` (default) restricts scoring
            to the classifier's ``classes_``.

        Returns
        -------
        None
        """
        self.classifier_kwargs = dict(classifier_kwargs)
        self.classifier_kwargs['uniform_class_weights'] = True
        self.class_labels = None if class_labels is None else np.unique(np.asarray(class_labels))
        self.classifier_ = None

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> "MultinomialEmission":
        """
        Description
        -----------
        Responsibility-weighted M-step: refit the classifier with the responsibilities.

        Parameters
        ----------
        X (np.ndarray)
            ``(n, d)`` behavioural design matrix.
        y (np.ndarray)
            ``(n,)`` categorical vocalization labels.
        sample_weight (np.ndarray)
            ``(n,)`` per-sample weights (the state's responsibilities).

        Returns
        -------
        self (MultinomialEmission)
            The fitted emission, with a fresh ``classifier_``.
        """
        self.classifier_ = SmoothMultinomialLogisticRegression(**self.classifier_kwargs)
        self.classifier_.fit(np.asarray(X), np.asarray(y),
                             sample_weight=np.asarray(sample_weight, dtype=np.float64))
        return self

    def log_density(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Per-sample categorical log-density: log P(observed class | x) under this state.

        Parameters
        ----------
        X (np.ndarray)
            ``(n, d)`` behavioural design matrix.
        y (np.ndarray)
            ``(n,)`` observed labels. Without ``class_labels`` each must be one of the
            classifier's ``classes_`` (a ``KeyError`` is raised otherwise); with it,
            each must be in ``class_labels`` (a ``ValueError`` is raised otherwise),
            and one the classifier never trained on has probability 0.

        Returns
        -------
        log_density (np.ndarray)
            ``(n,)`` ``log(P(y | x) + 1e-12)`` (the small constant keeps a zero
            probability finite).
        """
        proba = self.classifier_.predict_proba(np.asarray(X))
        column_of = {label: idx for idx, label in enumerate(self.classifier_.classes_)}
        y = np.asarray(y)
        if self.class_labels is None:
            columns = np.array([column_of[value] for value in y])
            return np.log(proba[np.arange(y.shape[0]), columns] + 1e-12)
        outside = np.setdiff1d(np.unique(y), self.class_labels)
        if outside.size:
            message = (f"Labels {outside.tolist()} are outside the fixed label set "
                       f"{self.class_labels.tolist()} of this multinomial emission.")
            raise ValueError(message)
        # A label of the fixed set that the training fold lacked has no column, i.e.
        # zero predicted probability.
        probability = np.zeros(y.shape[0])
        for row, value in enumerate(y):
            if value in column_of:
                probability[row] = proba[row, column_of[value]]
        return np.log(probability + 1e-12)

    @property
    def n_params(self) -> int:
        """
        Description
        -----------
        Free classifier coefficients + intercepts: one class of the softmax is
        redundant, so ``(C - 1) * (d + 1)`` of the ``C * (d + 1)`` entries are free
        (:func:`_softmax_free_params`).

        Parameters
        ----------
        None

        Returns
        -------
        n_params (int)
            ``(C - 1) * (d + 1)`` for the fitted classifier's ``C`` classes.
        """
        return _softmax_free_params(self.classifier_)


def resolve_emission_cls(emission_type: str) -> type:
    """
    Description
    -----------
    Map an ``emission_type`` string onto its emission class. Kept as a small
    registry so the pipeline and tests select emissions by name.

    Parameters
    ----------
    emission_type (str)
        ``'gaussian'``, ``'manifold'``, or ``'multinomial'``.

    Returns
    -------
    cls (type)
        The emission class.
    """
    registry = {'gaussian': GaussianEmission, 'manifold': ManifoldEmission,
                'multinomial': MultinomialEmission}
    if emission_type not in registry:
        raise ValueError(
            f"Unknown emission_type {emission_type!r}; available: {sorted(registry)}."
        )
    return registry[emission_type]


class GLMHMM:
    """
    Description
    -----------
    Hidden Markov model with per-state GLM emissions, fit by Baum-Welch EM.

    The latent state indexes a per-state emission (a behaviour -> vocalization
    rule); the state evolves as a stationary first-order Markov chain. Fitting
    maximises the marginal likelihood of the observation sequences by EM: the
    E-step runs forward-backward to get per-sample state responsibilities and
    per-transition responsibilities; the M-step updates the initial-state vector,
    the transition matrix, and each state's emission via a
    responsibility-weighted refit. EM is run from several random restarts and the
    highest-likelihood fit is kept.

    All the HMM recursions run in log space (``logsumexp``) so long sequences and
    sharp emissions stay numerically stable.

    Parameters
    ----------
    n_states (int)
        Number of latent states ``K``.
    emission_factory (callable)
        Zero-argument callable returning a fresh emission object (one per state);
        e.g. ``lambda: ManifoldEmission(regressor_kwargs)``.
    n_em_iters (int)
        Maximum Baum-Welch iterations per restart.
    n_restarts (int)
        Number of random EM restarts; the best-likelihood fit is kept.
    tol (float)
        Convergence tolerance on the per-iteration increase in total
        log-likelihood.
    transition_pseudocount (float)
        Laplace pseudocount added to the expected transition and initial-state
        counts in the M-step, so an unvisited transition keeps a small mass and
        the chain stays ergodic.
    random_state (int)
        Seed for the restart / initialisation randomness.
    verbose (bool)
        Print per-iteration log-likelihood when True.

    Returns
    -------
    None
    """

    def __init__(self, n_states: int, emission_factory,
                 *, n_em_iters: int = 100, n_restarts: int = 5, tol: float = 1e-4,
                 transition_pseudocount: float = 1.0, transition_mode: str = 'static',
                 transition_factory=None, random_state: int = 0,
                 verbose: bool = False) -> None:
        """
        Description
        -----------
        Store the model configuration, validate the transition mode, and initialise
        the fitted attributes (``log_pi_``, ``log_A_``, ``transition_glms_``,
        ``emissions_``, ``log_likelihood_``, ``n_iter_``, ``converged_``) to ``None``.

        Parameters
        ----------
        n_states (int)
            Number of latent states ``K``.
        emission_factory (callable)
            Zero-argument callable returning a fresh emission object (one per state).
        n_em_iters (int)
            Maximum Baum-Welch iterations per restart; defaults to 100.
        n_restarts (int)
            Number of random EM restarts; defaults to 5.
        tol (float)
            Convergence tolerance on the per-iteration log-likelihood increase;
            defaults to ``1e-4``.
        transition_pseudocount (float)
            Laplace pseudocount on the expected transition / initial-state counts;
            defaults to 1.0.
        transition_mode (str)
            ``'static'`` (default) for the stationary ``K x K`` matrix, or
            ``'input_driven'`` for per-"from"-state transition GLMs of the design row.
            Any other value raises ``ValueError``.
        transition_factory (callable or None)
            Zero-argument callable returning a fresh per-"from"-state transition
            classifier; required (``ValueError`` otherwise) when
            ``transition_mode='input_driven'``, ignored in ``'static'`` mode.
        random_state (int)
            Seed for the restart / initialisation randomness; defaults to 0.
        verbose (bool)
            Print per-iteration log-likelihood when True; defaults to False.

        Returns
        -------
        None
        """
        self.n_states = int(n_states)
        self.emission_factory = emission_factory
        self.n_em_iters = int(n_em_iters)
        self.n_restarts = int(n_restarts)
        self.tol = float(tol)
        self.transition_pseudocount = float(transition_pseudocount)
        # Transition dynamics. ``'static'`` (default) keeps the classic stationary
        # ``K x K`` matrix ``log_A_``. ``'input_driven'`` makes the transition into
        # each timestep a per-"from"-state multinomial-logistic GLM of the same
        # behavioural design row ``X`` that drives the emissions (Calhoun/Pillow/Murthy
        # input-driven transitions): ``P(z_t = j | z_{t-1} = i, x_t)``. In that mode
        # ``transition_factory`` is a zero-argument callable returning a fresh
        # per-"from"-state classifier (e.g. ``lambda: SmoothMultinomialLogisticRegression(...)``),
        # exactly mirroring ``emission_factory``.
        self.transition_mode = str(transition_mode)
        self.transition_factory = transition_factory
        if self.transition_mode not in ('static', 'input_driven'):
            message = (f"Unknown transition_mode {self.transition_mode!r}; "
                       "available: 'static', 'input_driven'.")
            raise ValueError(message)
        if self.transition_mode == 'input_driven' and self.transition_factory is None:
            message = ("transition_mode='input_driven' requires a transition_factory "
                       "(a zero-argument callable returning a fresh transition classifier).")
            raise ValueError(message)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.log_pi_ = None
        self.log_A_ = None
        self.transition_glms_ = None
        self.emissions_ = None
        self.log_likelihood_ = None
        self.n_iter_ = None
        self.converged_ = None

    # Log-space forward-backward.
    def _forward_backward(self, log_B: np.ndarray, log_A_seq: np.ndarray | None = None,
                          return_full_xi: bool = False) -> tuple:
        """
        Description
        -----------
        Run the log-space forward-backward recursion for one sequence, under either
        a stationary transition matrix or a per-timestep (input-driven) one.

        Parameters
        ----------
        log_B (np.ndarray)
            ``(T, K)`` per-timestep, per-state emission log-density.
        log_A_seq (np.ndarray)
            ``(T, K, K)`` per-timestep log transition tensor, where ``log_A_seq[t]``
            is ``log P(z_t = j | z_{t-1} = i, x_t)`` for the step landing at ``t``
            (row ``i`` = from-state, column ``j`` = to-state); ``log_A_seq[0]`` is
            unused. Supplied in ``'input_driven'`` mode. When ``None`` (``'static'``
            mode) the stationary ``self.log_A_`` is used at every transition.
        return_full_xi (bool)
            When True return the full per-transition responsibility tensor
            ``(T - 1, K, K)`` instead of its sum over ``t``; the input-driven
            transition M-step needs the per-timestep ``xi`` (paired with each
            timestep's covariates) to refit the transition GLMs.

        Returns
        -------
        gamma (np.ndarray)
            ``(T, K)`` posterior state responsibilities.
        xi (np.ndarray)
            ``(K, K)`` summed posterior transition responsibilities (``return_full_xi``
            False), or the full ``(T - 1, K, K)`` per-transition tensor (True).
        gamma0 (np.ndarray)
            ``(K,)`` posterior over the first state (for the initial-state update).
        seq_loglik (float)
            The sequence marginal log-likelihood.
        """
        n_time = log_B.shape[0]

        def log_A_into(t: int) -> np.ndarray:
            """
            Description
            -----------
            Log transition matrix used for the step landing at timestep ``t``.

            Parameters
            ----------
            t (int)
                Index of the timestep the transition lands on.

            Returns
            -------
            log_A (np.ndarray)
                ``(K, K)`` log transition matrix: the stationary ``self.log_A_`` when no
                per-timestep tensor was supplied, else ``log_A_seq[t]``.
            """
            return self.log_A_ if log_A_seq is None else log_A_seq[t]

        log_alpha = np.empty((n_time, self.n_states))
        log_alpha[0] = self.log_pi_ + log_B[0]
        for t in range(1, n_time):
            log_alpha[t] = log_B[t] + _logsumexp(log_alpha[t - 1][:, None] + log_A_into(t), axis=0)
        seq_loglik = float(_logsumexp(log_alpha[-1], axis=0))

        log_beta = np.zeros((n_time, self.n_states))
        for t in range(n_time - 2, -1, -1):
            log_beta[t] = _logsumexp(
                log_A_into(t + 1) + (log_B[t + 1] + log_beta[t + 1])[None, :], axis=1)

        log_gamma = log_alpha + log_beta - seq_loglik
        gamma = np.exp(log_gamma)

        if return_full_xi:
            xi = np.empty((max(n_time - 1, 0), self.n_states, self.n_states))
            for t in range(n_time - 1):
                xi[t] = np.exp(log_alpha[t][:, None] + log_A_into(t + 1)
                               + (log_B[t + 1] + log_beta[t + 1])[None, :] - seq_loglik)
        elif log_A_seq is None and n_time > 1:
            # Stationary transitions let the summed responsibilities be written in
            # closed form instead of walked timestep by timestep. The summand is an
            # outer product whose only `t`-dependence sits in the two vectors, so
            #
            #     xi[i, j] = A[i, j] * sum_t exp(log_alpha[t, i]
            #                                    + log_B[t + 1, j] + log_beta[t + 1, j]
            #                                    - seq_loglik)
            #
            # factors into one matrix product once each timestep's row and column
            # maxima are pulled out as a common scalar weight. That weight is O(1)
            # because the two maxima sum to roughly the sequence log-likelihood, so
            # this is as stable as the loop it replaces and returns the same value
            # rather than an approximation.
            from_terms = log_alpha[:-1]
            into_terms = log_B[1:] + log_beta[1:]
            from_max = from_terms.max(axis=1)
            into_max = into_terms.max(axis=1)
            weights = np.exp(from_max + into_max - seq_loglik)
            from_scaled = np.exp(from_terms - from_max[:, None]) * weights[:, None]
            into_scaled = np.exp(into_terms - into_max[:, None])
            xi = np.exp(self.log_A_) * (from_scaled.T @ into_scaled)
        else:
            # Input-driven transitions vary with `t`, so the product above does not
            # factor and the accumulation stays a loop.
            xi = np.zeros((self.n_states, self.n_states))
            for t in range(n_time - 1):
                log_xi = (log_alpha[t][:, None] + log_A_into(t + 1)
                          + (log_B[t + 1] + log_beta[t + 1])[None, :] - seq_loglik)
                xi += np.exp(log_xi)
        return gamma, xi, gamma[0], seq_loglik

    def _emission_log_B(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Stack per-state emission log-densities into a ``(T, K)`` matrix. Non-finite
        entries are replaced by ``_LOG_DENSITY_FLOOR`` and every entry is clipped from
        below at that floor.

        Parameters
        ----------
        X (np.ndarray)
            ``(T, d)`` behavioural design for the sequence.
        y (np.ndarray)
            The sequence's observed targets.

        Returns
        -------
        log_B (np.ndarray)
            ``(T, K)`` floored per-timestep, per-state emission log-density.
        """
        log_B = np.empty((X.shape[0], self.n_states))
        for k in range(self.n_states):
            log_B[:, k] = self.emissions_[k].log_density(X, y)
        # Any non-finite density (from a degenerate / failed emission refit) is pinned
        # to the floor so it cannot propagate NaN through the whole forward-backward;
        # a floored state is simply "impossible here" and loses its responsibility,
        # rather than poisoning every sequence's log-likelihood.
        log_B = np.where(np.isfinite(log_B), log_B, _LOG_DENSITY_FLOOR)
        return np.maximum(log_B, _LOG_DENSITY_FLOOR)

    def _seq_log_transition(self, X_seq: np.ndarray) -> np.ndarray | None:
        """
        Description
        -----------
        Per-timestep log transition tensor for one sequence in ``'input_driven'``
        mode: ``(T, K, K)`` with ``[t, i, j] = log P(z_t = j | z_{t-1} = i, x_t)``,
        the ``i``-th "from-state" transition GLM evaluated on the sequence's design
        rows (``[0]`` is unused -- there is no transition into the first timestep).
        Returns ``None`` in ``'static'`` mode, and also before the transition GLMs
        exist (the first EM iteration falls back to the initial ``self.log_A_``).

        Parameters
        ----------
        X_seq (np.ndarray)
            ``(T, d)`` behavioural design for the sequence.

        Returns
        -------
        log_A_seq (np.ndarray or None)
            ``(T, K, K)`` per-timestep log transition tensor, or None.
        """
        if self.transition_mode != 'input_driven' or self.transition_glms_ is None:
            return None
        X_seq = np.asarray(X_seq)
        log_A_seq = np.empty((X_seq.shape[0], self.n_states, self.n_states))
        for i in range(self.n_states):
            classifier = self.transition_glms_[i]
            if classifier is None:
                # A "from"-state that has never accrued enough responsibility to be
                # fit (its collapse guard kept skipping it) transitions uniformly.
                log_A_seq[:, i, :] = -np.log(self.n_states)
                continue
            proba = classifier.predict_proba(X_seq)
            column_of = {label: idx for idx, label in enumerate(classifier.classes_)}
            ordered = np.array([column_of[j] for j in range(self.n_states)])
            log_A_seq[:, i, :] = np.log(proba[:, ordered] + 1e-12)
        return log_A_seq

    def _fit_transition_glms(self, X_trans: np.ndarray, xi_trans: np.ndarray) -> None:
        """
        Description
        -----------
        Input-driven transition M-step: refit each "from-state" transition GLM to
        maximise the responsibility-weighted expected transition log-likelihood
        ``sum_t sum_j xi[t, i, j] * log P(z=j | x_t)``. Each soft target ``xi[:, i, :]``
        is fit via the estimator's per-sample ``sample_weight`` channel using the
        standard soft-label expansion: every landing covariate row is repeated once
        per candidate to-state ``j`` (label ``j``, weight ``xi[t, i, j]``), so the
        weighted multinomial-logistic MLE matches the soft-EM objective. A
        "from-state" whose total responsibility has collapsed keeps its previous GLM
        (the collapse guard, mirroring the emission M-step).

        Parameters
        ----------
        X_trans (np.ndarray)
            ``(n_transitions, d)`` design rows at each transition's landing timestep,
            pooled across sequences.
        xi_trans (np.ndarray)
            ``(n_transitions, K, K)`` per-transition posterior responsibilities.

        Returns
        -------
        None
        """
        n_trans = X_trans.shape[0]
        to_states = np.tile(np.arange(self.n_states), n_trans)
        X_expanded = np.repeat(X_trans, self.n_states, axis=0)
        for i in range(self.n_states):
            weights = xi_trans[:, i, :].reshape(-1)
            if weights.sum() <= _MIN_STATE_RESPONSIBILITY:
                continue
            self.transition_glms_[i] = self.transition_factory().fit(
                X_expanded, to_states, sample_weight=weights)

    # Expectation-maximization.
    def _em_once(self, sequences: list, X_all: np.ndarray, y_all: np.ndarray,
                 rng: np.random.Generator) -> dict:
        """
        Description
        -----------
        One EM restart: initialise the parameters and emissions from ``rng``,
        then alternate E- and M-steps until the total log-likelihood converges or
        ``n_em_iters`` is reached.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` per-sequence tuples.
        X_all (np.ndarray)
            Pooled design matrix across sequences (row order = concatenation of
            the sequences), reused for the pooled emission M-step.
        y_all (np.ndarray)
            Pooled targets, aligned to ``X_all``.
        rng (np.random.Generator)
            Seeded generator for this restart.

        Returns
        -------
        result (dict)
            ``log_pi``, ``log_A``, ``emissions``, ``log_likelihood``, ``n_iter``,
            ``converged``. ``log_likelihood`` is always that of the returned
            parameters (recomputed after the final M-step when EM stops at
            ``n_em_iters`` without converging).
        """
        n_total = X_all.shape[0]
        # Structured k-means-on-targets init so the states start differentiated
        # (a random per-sample init collapses to one state for weakly-separated
        # emissions); the restart seed varies the k-means start.
        resp0 = _kmeans_init_responsibilities(
            y_all, self.n_states, int(rng.integers(0, 2 ** 31 - 1)))
        emissions = []
        for k in range(self.n_states):
            emissions.append(self.emission_factory().fit(X_all, y_all, resp0[:, k]))
        self.emissions_ = emissions
        # Mild, non-degenerate initial chain: near-uniform initial state, a
        # self-transition-favouring transition matrix (states persist over a bout).
        self.log_pi_ = np.log(np.full(self.n_states, 1.0 / self.n_states))
        stay = 0.8
        A0 = np.full((self.n_states, self.n_states), (1.0 - stay) / max(self.n_states - 1, 1))
        np.fill_diagonal(A0, stay)
        self.log_A_ = np.log(A0)

        # Input-driven transitions: the transition GLMs do not exist yet, so the first
        # E-step falls back to the static ``A0`` above; the transition M-step then fits
        # them from the full per-transition responsibilities. The landing covariates (the
        # design row at each transition's destination, i.e. every sequence's rows after
        # the first) are pooled once here to pair with those responsibilities.
        input_driven = self.transition_mode == 'input_driven'
        if input_driven:
            X_trans = np.concatenate([np.asarray(X)[1:] for X, _ in sequences], axis=0)
            n_trans = X_trans.shape[0]
            # Seed the transition GLMs from the initial (k-means) responsibilities so the
            # input-driven dynamics engage from the very FIRST E-step. Otherwise EM's first
            # iteration runs under the static ``A0`` and the emissions can collapse to a
            # single state before the transitions ever differentiate them. The initial
            # per-transition responsibility is the independence approximation
            # ``xi_init[t, i, j] = resp0[t, i] * resp0[t + 1, j]``.
            xi_init = np.empty((n_trans, self.n_states, self.n_states))
            cursor = 0
            cursor_tr = 0
            for X_seq, _ in sequences:
                n_seq = np.asarray(X_seq).shape[0]
                resp_seq = resp0[cursor:cursor + n_seq]
                if n_seq > 1:
                    xi_init[cursor_tr:cursor_tr + n_seq - 1] = np.einsum(
                        'ti,tj->tij', resp_seq[:-1], resp_seq[1:])
                    cursor_tr += n_seq - 1
                cursor += n_seq
            self.transition_glms_ = [None] * self.n_states
            self._fit_transition_glms(X_trans, xi_init)

        prev_loglik = -np.inf
        converged = False
        n_iter = 0
        for n_iter in range(1, self.n_em_iters + 1):
            # E-step: forward-backward per sequence, accumulate the sufficient
            # statistics (per-sample responsibilities, transition counts, first-
            # state counts, total log-likelihood).
            gamma_all = np.empty((n_total, self.n_states))
            xi_total = np.zeros((self.n_states, self.n_states))
            xi_trans = np.empty((n_trans, self.n_states, self.n_states)) if input_driven else None
            gamma0_total = np.zeros(self.n_states)
            total_loglik = 0.0
            cursor = 0
            cursor_tr = 0
            for X_seq, y_seq in sequences:
                log_B = self._emission_log_B(X_seq, y_seq)
                log_A_seq = self._seq_log_transition(X_seq)
                gamma, xi, gamma0, seq_loglik = self._forward_backward(
                    log_B, log_A_seq, return_full_xi=input_driven)
                n_seq = X_seq.shape[0]
                gamma_all[cursor:cursor + n_seq] = gamma
                cursor += n_seq
                if input_driven:
                    xi_trans[cursor_tr:cursor_tr + (n_seq - 1)] = xi
                    cursor_tr += n_seq - 1
                else:
                    xi_total += xi
                gamma0_total += gamma0
                total_loglik += seq_loglik

            if self.verbose:
                print(f"  EM iter {n_iter}: loglik={total_loglik:.4f}", flush=True)

            if total_loglik - prev_loglik < self.tol and n_iter > 1:
                converged = True
                prev_loglik = total_loglik
                break
            prev_loglik = total_loglik

            # M-step: initial-state vector and transition matrix from the
            # (pseudocount-smoothed) expected counts; each emission refit on the
            # pooled data with its state's responsibilities as sample weights.
            pi = gamma0_total + self.transition_pseudocount
            self.log_pi_ = np.log(pi / pi.sum())
            if input_driven:
                # Refit the per-"from"-state transition GLMs from the full per-transition
                # responsibilities (the static ``log_A_`` is left at its init and unused).
                if self.transition_glms_ is None:
                    self.transition_glms_ = [None] * self.n_states
                self._fit_transition_glms(X_trans, xi_trans)
            else:
                A = xi_total + self.transition_pseudocount
                self.log_A_ = np.log(A / A.sum(axis=1, keepdims=True))
            for k in range(self.n_states):
                # Collapse guard: only refit a state whose responsibility mass is
                # non-trivial; a state that has emptied out is left at its previous
                # emission rather than refit on ~zero weights (which would NaN).
                if gamma_all[:, k].sum() > _MIN_STATE_RESPONSIBILITY:
                    self.emissions_[k].fit(X_all, y_all, gamma_all[:, k])

        # On convergence the loop breaks BEFORE the M-step, so ``prev_loglik`` is the
        # log-likelihood of the parameters returned. When it instead runs out of
        # iterations, its last value predates the final M-step; recompute it under
        # the returned parameters so restarts are compared, and the fit recorded, on
        # the model actually kept.
        if not converged:
            prev_loglik = self.log_likelihood(sequences)

        return {
            'log_pi': self.log_pi_.copy(),
            'log_A': self.log_A_.copy(),
            'transition_glms': self.transition_glms_,
            'emissions': self.emissions_,
            'log_likelihood': prev_loglik,
            'n_iter': n_iter,
            'converged': converged,
        }

    def fit(self, sequences: list) -> "GLMHMM":
        """
        Description
        -----------
        Fit the GLM-HMM to a list of observation sequences by Baum-Welch EM with
        random restarts, keeping the highest-likelihood restart.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` tuples, one per session, with rows in temporal
            (USV-onset) order. ``X`` is ``(T, d)`` behavioural history; ``y`` is
            the observed vocalization (``(T, 2)`` manifold position, or ``(T,)``
            for a categorical target).

        Returns
        -------
        self (GLMHMM)
            The fitted model (best restart), with ``log_pi_``, ``log_A_``,
            ``emissions_``, ``log_likelihood_``, ``n_iter_`` and ``converged_``.
        """
        if not sequences:
            raise ValueError("GLMHMM.fit requires at least one sequence.")
        X_all = np.concatenate([np.asarray(X) for X, _ in sequences], axis=0)
        y_all = np.concatenate([np.asarray(y) for _, y in sequences], axis=0)

        best = None
        for restart in range(self.n_restarts):
            rng = np.random.default_rng(self.random_state + restart)
            result = self._em_once(sequences, X_all, y_all, rng)
            if self.verbose:
                print(f"Restart {restart}: loglik={result['log_likelihood']:.4f} "
                      f"(converged={result['converged']}, iters={result['n_iter']})", flush=True)
            if best is None or result['log_likelihood'] > best['log_likelihood']:
                best = result

        self.log_pi_ = best['log_pi']
        self.log_A_ = best['log_A']
        self.transition_glms_ = best['transition_glms']
        self.emissions_ = best['emissions']
        self.log_likelihood_ = best['log_likelihood']
        self.n_iter_ = best['n_iter']
        self.converged_ = best['converged']
        return self

    def log_likelihood(self, sequences: list) -> float:
        """
        Description
        -----------
        Total marginal log-likelihood of ``sequences`` under the fitted model
        (the forward pass' normaliser, summed over sequences). Used to score a
        held-out session set for state-count selection.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` tuples.

        Returns
        -------
        total (float)
            Summed sequence log-likelihoods.
        """
        total = 0.0
        for X_seq, y_seq in sequences:
            X_arr = np.asarray(X_seq)
            log_B = self._emission_log_B(X_arr, np.asarray(y_seq))
            log_A_seq = self._seq_log_transition(X_arr)
            log_alpha = self.log_pi_ + log_B[0]
            for t in range(1, log_B.shape[0]):
                log_A_t = self.log_A_ if log_A_seq is None else log_A_seq[t]
                log_alpha = log_B[t] + _logsumexp(log_alpha[:, None] + log_A_t, axis=0)
            total += float(_logsumexp(log_alpha, axis=0))
        return total

    def viterbi(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Most-likely latent-state path for one sequence (log-space Viterbi).

        Parameters
        ----------
        X (np.ndarray)
            ``(T, d)`` behavioural history for the sequence.
        y (np.ndarray)
            The sequence's observed vocalizations.

        Returns
        -------
        states (np.ndarray)
            ``(T,)`` integer most-likely state per timestep.
        """
        X = np.asarray(X)
        log_B = self._emission_log_B(X, np.asarray(y))
        log_A_seq = self._seq_log_transition(X)
        n_time = log_B.shape[0]
        delta = np.empty((n_time, self.n_states))
        psi = np.zeros((n_time, self.n_states), dtype=np.int64)
        delta[0] = self.log_pi_ + log_B[0]
        for t in range(1, n_time):
            log_A_t = self.log_A_ if log_A_seq is None else log_A_seq[t]
            scored = delta[t - 1][:, None] + log_A_t
            psi[t] = np.argmax(scored, axis=0)
            delta[t] = log_B[t] + np.max(scored, axis=0)
        states = np.empty(n_time, dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n_time - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    @property
    def n_params(self) -> int:
        """
        Description
        -----------
        Number of FREE parameters of the fitted model: the initial-state vector
        (``K - 1``, it sums to 1), the transitions (``K * (K - 1)`` for the static
        matrix, whose rows sum to 1; for input-driven transitions each fitted
        "from"-state softmax GLM's ``(C - 1) * (d + 1)``, one class being redundant),
        plus every emission's own free-parameter count.

        Parameters
        ----------
        None

        Returns
        -------
        n_params (int)
            The free-parameter count used by :meth:`bic`.
        """
        n_params = self.n_states - 1
        if self.transition_mode == 'input_driven' and self.transition_glms_ is not None:
            n_params += sum(_softmax_free_params(g) for g in self.transition_glms_ if g is not None)
        else:
            n_params += self.n_states * (self.n_states - 1)
        n_params += sum(em.n_params for em in self.emissions_)
        return int(n_params)

    def bic(self, sequences: list) -> float:
        """
        Description
        -----------
        Bayesian information criterion (lower is better) on ``sequences``:
        ``-2 * loglik + n_params * log(n_obs)``, where ``n_params`` is the
        free-parameter count :attr:`n_params` and ``n_obs`` is the total number of
        observed timesteps. Reported as a cross-check on the held-out
        log-likelihood for state-count selection.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` tuples the model was fit on.

        Returns
        -------
        value (float)
            The BIC.
        """
        n_obs = int(sum(np.asarray(X).shape[0] for X, _ in sequences))
        return float(-2.0 * self.log_likelihood(sequences) + self.n_params * np.log(max(n_obs, 1)))


# A log(0) surrogate for the pinned self-transition / padded steps in the JAX engine.
_ID_NEG = -1e30


def _np_log_softmax(a: np.ndarray, axis: int) -> np.ndarray:
    """
    Description
    -----------
    Numerically-stable numpy log-softmax along one axis, used by the numpy Viterbi
    path of :class:`InputDrivenGLMHMM` (the per-slice max is subtracted for stability).

    Parameters
    ----------
    a (np.ndarray)
        Input array of logits.
    axis (int)
        Axis over which to normalise.

    Returns
    -------
    log_probs (np.ndarray)
        ``a - logsumexp(a, axis)``, the same shape as ``a``.
    """
    a_max = np.max(a, axis=axis, keepdims=True)
    return a - (a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True)))


def _id_seq_loglik(params: dict, X: jax.Array, y: jax.Array,
                   mask: jax.Array) -> jax.Array:
    """
    Description
    -----------
    Log-space forward marginal log-likelihood for ONE padded sequence under the
    reference-coded, input-driven GLM-HMM of :class:`InputDrivenGLMHMM`. Emission
    class 0 (e.g. no-song) is the fixed zero baseline (``C - 1`` free emission
    filters); the self-transition logit is pinned to 0 (self is the transition
    reference). The transition into timestep ``t`` is a per-"from"-state softmax of
    the design row ``X[t]``. Padded steps (``mask`` False) get zero emission and an
    identity transition so they do not contribute.

    Parameters
    ----------
    params (dict)
        Parameters ``log_pi`` ``(K,)``, ``W_emit`` ``(K, D, C - 1)``, ``b_emit``
        ``(K, C - 1)``, ``W_trans`` ``(K, D, K)``, ``b_trans`` ``(K, K)``.
    X (jax.Array)
        ``(T, D)`` padded behavioural design for the sequence.
    y (jax.Array)
        ``(T,)`` padded categorical emissions.
    mask (jax.Array)
        ``(T,)`` boolean; True on real timesteps, False on padding.

    Returns
    -------
    seq_loglik (jax.Array)
        Scalar marginal log-likelihood of the (unpadded) sequence.
    """
    n_states = params['b_emit'].shape[0]
    e_nonref = jnp.einsum('td,kdc->tkc', X, params['W_emit']) + params['b_emit'][None]
    e_full = jnp.concatenate(
        [jnp.zeros((X.shape[0], n_states, 1), e_nonref.dtype), e_nonref], axis=2)
    log_emit_full = e_full - _jax_logsumexp(e_full, axis=2, keepdims=True)
    log_emit = jnp.take_along_axis(log_emit_full, y[:, None, None], axis=2)[:, :, 0]
    log_emit = jnp.where(mask[:, None], log_emit, 0.0)

    t_logits = jnp.einsum('td,idj->tij', X, params['W_trans']) + params['b_trans'][None]
    eye = jnp.eye(n_states, dtype=bool)
    t_logits = jnp.where(eye[None], 0.0, t_logits)
    log_trans = t_logits - _jax_logsumexp(t_logits, axis=2, keepdims=True)
    identity_log = jnp.where(eye, 0.0, _ID_NEG)[None]
    log_trans = jnp.where(mask[:, None, None], log_trans, identity_log)

    log_pi = params['log_pi'] - _jax_logsumexp(params['log_pi'])
    a0 = log_pi + log_emit[0]

    def step(a_prev: jax.Array, inp: tuple) -> tuple:
        """
        Description
        -----------
        One forward-recursion step of the ``lax.scan``: combine the previous log-alpha
        with the transition log-matrix landing at this timestep (``logsumexp`` over the
        from-state axis) and add this timestep's emission log-density.

        Parameters
        ----------
        a_prev (jax.Array)
            ``(K,)`` log-alpha of the previous timestep (the scan carry).
        inp (tuple)
            ``(le_t, lt_t)``: the ``(K,)`` emission log-density of this timestep and the
            ``(K, K)`` log transition matrix into it (row = from-state, column = to-state).

        Returns
        -------
        a_t, None (tuple)
            The ``(K,)`` log-alpha of this timestep (the new carry) and ``None`` (no
            per-step output is stacked).
        """
        le_t, lt_t = inp
        a_t = le_t + _jax_logsumexp(a_prev[:, None] + lt_t, axis=0)
        return a_t, None

    a_last, _ = lax.scan(step, a0, (log_emit[1:], log_trans[1:]))
    return _jax_logsumexp(a_last)


def _id_batch_loglik(params: dict, Xp: jax.Array, yp: jax.Array,
                     mask: jax.Array) -> jax.Array:
    """
    Description
    -----------
    Total forward marginal log-likelihood over a padded batch of sequences, by
    vmapping :func:`_id_seq_loglik` across the batch and summing.

    Parameters
    ----------
    params (dict)
        Model parameters (see :func:`_id_seq_loglik`).
    Xp (jax.Array)
        ``(S, Tmax, D)`` padded design tensor.
    yp (jax.Array)
        ``(S, Tmax)`` padded categorical emissions.
    mask (jax.Array)
        ``(S, Tmax)`` boolean real-vs-padding mask.

    Returns
    -------
    total (jax.Array)
        Scalar summed sequence log-likelihood.
    """
    per_seq = jax.vmap(lambda X, y, m: _id_seq_loglik(params, X, y, m))(Xp, yp, mask)
    return jnp.sum(per_seq)


def _id_first_diff(W: jax.Array, n_features: int, n_time: int) -> jax.Array:
    """
    Description
    -----------
    Sum of squared first temporal differences within each per-feature filter block of
    ``W`` -- the first-difference temporal-smoothness penalty. The design width is
    ``n_features * n_time`` (feature-major), so ``W`` is reshaped to isolate each
    feature's ``n_time`` taps before differencing along the tap axis. Summing (not
    averaging) makes the penalty scale with the number of states.

    Parameters
    ----------
    W (jax.Array)
        ``(K, n_features * n_time, out)`` filter weights.
    n_features (int)
        Number of behavioural cues (feature-major blocks).
    n_time (int)
        Number of temporal taps per cue.

    Returns
    -------
    penalty (jax.Array)
        Scalar sum of squared adjacent-tap differences.
    """
    n_states, _, out = W.shape
    Wb = W.reshape(n_states, n_features, n_time, out)
    d1 = Wb[:, :, 1:, :] - Wb[:, :, :-1, :]
    return jnp.sum(d1 ** 2)


def _id_objective(params: dict, Xp: jax.Array, yp: jax.Array, mask: jax.Array,
                  n_bins: jax.Array, r: jax.Array, n_features: int,
                  n_time: int) -> jax.Array:
    """
    Description
    -----------
    Regularised training objective (minimised): the per-bin-mean negative forward
    marginal log-likelihood plus ``r`` times the summed first-difference smoothness of
    the emission and transition filters (see :func:`_id_first_diff`).

    Parameters
    ----------
    params (dict)
        Model parameters (see :func:`_id_seq_loglik`).
    Xp, yp, mask (jax.Array)
        Padded batch design, emissions, and mask (see :func:`_id_batch_loglik`).
    n_bins (jax.Array)
        Total number of real (unpadded) timesteps, for the per-bin mean.
    r (jax.Array)
        First-difference smoothness coefficient.
    n_features (int)
        Number of behavioural cues.
    n_time (int)
        Number of temporal taps per cue.

    Returns
    -------
    value (jax.Array)
        Scalar regularised objective.
    """
    data = -_id_batch_loglik(params, Xp, yp, mask) / n_bins
    smooth = (_id_first_diff(params['W_emit'], n_features, n_time)
              + _id_first_diff(params['W_trans'], n_features, n_time))
    return data + r * smooth


@partial(jax.jit, static_argnums=(4, 7, 8))
def _id_run_lbfgs(init: dict, Xp: jax.Array, yp: jax.Array, mask: jax.Array,
                  n_steps: int, n_bins: jax.Array, r: jax.Array, n_features: int,
                  n_time: int) -> tuple:
    """
    Description
    -----------
    Full-batch L-BFGS (optax, with a zoom line search) on the regularised objective
    :func:`_id_objective`, running the whole ``n_steps``-iteration solve as a single
    ``lax.scan`` so it compiles to one GPU kernel.

    Parameters
    ----------
    init (dict)
        Initial parameters (see :func:`_id_seq_loglik`).
    Xp, yp, mask (jax.Array)
        Padded batch design, emissions, and mask (see :func:`_id_batch_loglik`).
    n_steps (int)
        Number of L-BFGS iterations (static; the ``lax.scan`` length).
    n_bins (jax.Array)
        Total number of real timesteps, for the per-bin mean in the objective.
    r (jax.Array)
        First-difference smoothness coefficient.
    n_features (int)
        Number of behavioural cues (static; used in the smoothness reshape).
    n_time (int)
        Number of temporal taps per cue (static; used in the smoothness reshape).

    Returns
    -------
    params, values (tuple[dict, jax.Array])
        The final parameters and the ``(n_steps,)`` per-iteration objective trace.
    """
    def fun(p: dict) -> jax.Array:
        """
        Description
        -----------
        Closure over the batch and hyperparameters: the regularised objective
        :func:`_id_objective` as a function of the parameters alone, as optax requires.

        Parameters
        ----------
        p (dict)
            Parameter pytree (see :func:`_id_seq_loglik`).

        Returns
        -------
        value (jax.Array)
            Scalar regularised objective.
        """
        return _id_objective(p, Xp, yp, mask, n_bins, r, n_features, n_time)

    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(fun)

    def step(carry: tuple, _: None) -> tuple:
        """
        Description
        -----------
        One L-BFGS iteration of the ``lax.scan``: evaluate the objective value and
        gradient (reusing the line-search cache in the optimiser state), compute the
        optax L-BFGS update with the zoom line search, and apply it.

        Parameters
        ----------
        carry (tuple)
            ``(params, state)``: the current parameter pytree and the optax L-BFGS state.
        _ (None)
            Unused scan input (the scan runs ``length=n_steps`` with no ``xs``).

        Returns
        -------
        (params, state), value (tuple)
            The updated ``(params, state)`` carry and the scalar objective value
            evaluated at the parameters BEFORE this update (stacked into the trace).
        """
        params, state = carry
        value, grad = value_and_grad(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return (params, state), value

    (params, _), values = lax.scan(step, (init, opt.init(init)), None, length=n_steps)
    return params, values


class InputDrivenGLMHMM:
    """
    Description
    -----------
    GLM-HMM with INPUT-DRIVEN transitions and a reference-coded multinomial emission,
    fit by DIRECT maximisation of the regularised marginal log-likelihood (not EM).

    This is the production home of the Calhoun/Pillow/Murthy-faithful engine validated
    on the Coen-2014 fly data (peak-at-K / drop-after, where the EM-based
    :class:`GLMHMM` collapses to a single state on weakly-identified categorical data).
    Both the emission and the transition into each timestep are per-state
    multinomial-logistic GLMs of the same behavioural design row ``X``:

    * Emission: ``P(y_t = c | z_t = k, x_t)`` with class 0 the fixed zero baseline
      (``num_classes - 1`` free filters per state).
    * Transition: ``P(z_t = j | z_{t-1} = i, x_t)`` with the self-transition ``i -> i``
      pinned to 0 (self is the reference).

    All parameters (initial-state logits, emission filters + biases, transition filters
    + biases) are optimised jointly against the exact forward marginal log-likelihood
    via optax L-BFGS with a zoom line search, under a first-difference, sum-scaled
    (K-scaling) temporal smoothness penalty on the filters -- the regularisation form
    the paper found best. Fitting jointly (rather than by alternating EM) is what keeps
    the states from collapsing. The forward pass is a JAX ``lax.scan`` vmapped over
    padded sequences, so it runs as one GPU kernel.

    Parameters
    ----------
    n_states (int)
        Number of latent states ``K``.
    n_features (int)
        Number of behavioural cues ``U`` (feature-major design blocks).
    n_time_bins (int)
        Number of temporal taps ``L`` per cue; the design width is ``U * L``.
    n_classes (int)
        Number of emission categories ``C`` (class 0 is the reference baseline).
    lambda_smooth (float)
        First-difference temporal-smoothness coefficient ``r`` (sum-scaled).
    n_restarts (int)
        Number of random-init restarts; the best by held-out (or training) forward
        log-likelihood is kept.
    n_lbfgs (int)
        L-BFGS iterations per restart.
    random_state (int)
        Base seed for the structured random-sign filter initialisation.
    verbose (bool)
        Print per-restart diagnostics when True.

    Returns
    -------
    None
    """

    def __init__(self, n_states: int, n_features: int, n_time_bins: int, n_classes: int,
                 *, lambda_smooth: float = 0.05, n_restarts: int = 4,
                 n_lbfgs: int = 600, random_state: int = 0, verbose: bool = False) -> None:
        """
        Description
        -----------
        Store the model configuration (cast to the documented types) and initialise
        the fitted attributes ``params_`` and ``log_likelihood_`` to ``None``.

        Parameters
        ----------
        n_states (int)
            Number of latent states ``K``.
        n_features (int)
            Number of behavioural cues ``U``.
        n_time_bins (int)
            Number of temporal taps ``L`` per cue.
        n_classes (int)
            Number of emission categories ``C``.
        lambda_smooth (float)
            First-difference temporal-smoothness coefficient; defaults to 0.05.
        n_restarts (int)
            Number of random-init restarts; defaults to 4.
        n_lbfgs (int)
            L-BFGS iterations per restart; defaults to 600.
        random_state (int)
            Base seed for the filter initialisation; defaults to 0.
        verbose (bool)
            Print per-restart diagnostics when True; defaults to False.

        Returns
        -------
        None
        """
        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.n_time_bins = int(n_time_bins)
        self.n_classes = int(n_classes)
        self.lambda_smooth = float(lambda_smooth)
        self.n_restarts = int(n_restarts)
        self.n_lbfgs = int(n_lbfgs)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        self.params_ = None
        self.log_likelihood_ = None

    def _init_params(self, seed: int) -> dict:
        """
        Description
        -----------
        Structured random-sign decaying-exponential filter init (Calhoun-style): each
        per-cue filter block is ``exp(-arange(L) / L)`` times a random sign in
        ``{-1, 0, +1}``, which breaks state symmetry so the joint optimiser can pull
        the states apart. Reference-coded shapes: ``W_emit`` has ``C - 1`` free
        emission classes, ``W_trans`` is ``K x D x K`` (self logit pinned in the loss).

        Parameters
        ----------
        seed (int)
            Seed for the ``np.random.default_rng`` drawing the signs and biases.

        Returns
        -------
        params (dict)
            float32 JAX arrays ``log_pi`` ``(K,)`` (zeros, a uniform initial state),
            ``W_emit`` ``(K, D, C - 1)``, ``b_emit`` ``(K, C - 1)`` and ``b_trans``
            ``(K, K)`` (normal, sd 0.3), and ``W_trans`` ``(K, D, K)``.
        """
        rng = np.random.default_rng(seed)
        design_width = self.n_features * self.n_time_bins
        decay = np.exp(-np.arange(self.n_time_bins) / self.n_time_bins)

        def structured(lead: tuple) -> np.ndarray:
            """
            Description
            -----------
            Build one structured random-sign filter block: for every leading index and every
            behavioural cue, a random sign drawn from ``{-1, 0, +1}`` (a uniform draw on
            ``[-1, 1]`` rounded to the nearest integer) multiplies the decaying exponential
            ``exp(-arange(L) / L)`` over the ``L`` temporal taps, and the cue blocks are
            flattened feature-major into the design width ``U * L``. Draws from the enclosing
            method's seeded ``rng``.

            Parameters
            ----------
            lead (tuple)
                Leading shape of the block (e.g. ``(n_states,)``).

            Returns
            -------
            block (np.ndarray)
                ``(*lead, n_features * n_time_bins)`` float32 filter block.
            """
            signs = np.round(rng.uniform(-1, 1, size=(*lead, self.n_features, 1)))
            block = signs * decay[None, :]
            return block.reshape((*lead, design_width)).astype(np.float32)

        n_states, n_classes = self.n_states, self.n_classes
        W_emit = np.stack(
            [structured((n_states,)) for _ in range(n_classes - 1)], axis=-1)
        W_trans = np.stack(
            [structured((n_states,)) for _ in range(n_states)], axis=-1)
        b_emit = rng.normal(0, 0.3, (n_states, n_classes - 1))
        b_trans = rng.normal(0, 0.3, (n_states, n_states))
        return {
            'log_pi': jnp.zeros((n_states,), dtype=jnp.float32),
            'W_emit': jnp.asarray(W_emit, dtype=jnp.float32),
            'b_emit': jnp.asarray(b_emit, dtype=jnp.float32),
            'W_trans': jnp.asarray(W_trans, dtype=jnp.float32),
            'b_trans': jnp.asarray(b_trans, dtype=jnp.float32),
        }

    def _pad_stack(self, sequences: list) -> tuple:
        """
        Description
        -----------
        Pad a list of ``(X, y)`` to a common ``Tmax`` -> ``(Xp, yp, mask)`` arrays.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` tuples, ``X`` of shape ``(T, n_features * n_time_bins)``
            and ``y`` of shape ``(T,)`` integer labels.

        Returns
        -------
        Xp, yp, mask (tuple[jax.Array, jax.Array, jax.Array])
            float32 design ``(S, Tmax, D)``, int32 labels ``(S, Tmax)`` and boolean
            real-step mask ``(S, Tmax)``; padded entries are zero / False.
        """
        design_width = self.n_features * self.n_time_bins
        lengths = np.asarray([len(y) for _, y in sequences], dtype=np.int32)
        n_max = int(lengths.max())
        n_seqs = len(sequences)
        Xp = np.zeros((n_seqs, n_max, design_width), dtype=np.float32)
        yp = np.zeros((n_seqs, n_max), dtype=np.int32)
        mask = np.zeros((n_seqs, n_max), dtype=bool)
        for s, (X, y) in enumerate(sequences):
            Xp[s, :len(y)] = X
            yp[s, :len(y)] = y
            mask[s, :len(y)] = True
        return jnp.asarray(Xp), jnp.asarray(yp), jnp.asarray(mask)

    def _forward_ll(self, params: dict, sequences: list) -> float:
        """
        Description
        -----------
        Total forward marginal log-likelihood of ``sequences`` under ``params``.

        Parameters
        ----------
        params (dict)
            Parameter pytree (see :func:`_id_seq_loglik`).
        sequences (list)
            List of ``(X, y)`` tuples.

        Returns
        -------
        loglik (float)
            Summed marginal log-likelihood over the sequences.
        """
        Xp, yp, mask = self._pad_stack(sequences)
        return float(_id_batch_loglik(params, Xp, yp, mask))

    def fit(self, sequences: list,
            held_out_sequences: list | None = None) -> "InputDrivenGLMHMM":
        """
        Description
        -----------
        Fit by direct marginal-likelihood maximisation with random restarts, keeping
        the restart with the best forward log-likelihood on ``held_out_sequences``
        (or on the training sequences themselves when no held-out set is given).

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` tuples (rows in temporal order); ``X`` is
            ``(T, n_features * n_time_bins)`` feature-major behavioural history and
            ``y`` is the ``(T,)`` categorical emission (integers ``0..C-1``).
        held_out_sequences (list)
            Optional validation sequences used to select the best restart (Calhoun's
            "keep the model with the best cross-validated log-likelihood"). When None,
            restarts are ranked on the training likelihood.

        Returns
        -------
        self (InputDrivenGLMHMM)
            The fitted engine, with ``params_`` and ``log_likelihood_``.
        """
        if not sequences:
            message = "InputDrivenGLMHMM.fit requires at least one sequence."
            raise ValueError(message)
        Xp, yp, mask = self._pad_stack(sequences)
        n_bins = int(sum(len(y) for _, y in sequences))
        selection = held_out_sequences if held_out_sequences else sequences

        best_score = -np.inf
        best_params = None
        for restart in range(self.n_restarts):
            init = self._init_params(self.random_state + restart)
            params, values = _id_run_lbfgs(
                init, Xp, yp, mask, self.n_lbfgs, n_bins, self.lambda_smooth,
                self.n_features, self.n_time_bins)
            if not np.isfinite(float(values[-1])):
                continue
            score = self._forward_ll(params, selection)
            if self.verbose:
                print(f"Restart {restart}: obj={float(values[-1]):.4f} "
                      f"selection_ll={score:.1f}", flush=True)
            if score > best_score:
                best_score = score
                best_params = params

        if best_params is None:
            message = "InputDrivenGLMHMM.fit: all restarts diverged."
            raise RuntimeError(message)
        self.params_ = best_params
        self.log_likelihood_ = self._forward_ll(best_params, sequences)
        return self

    def log_likelihood(self, sequences: list) -> float:
        """
        Description
        -----------
        Total forward marginal log-likelihood of ``sequences`` under the fit model.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` tuples.

        Returns
        -------
        loglik (float)
            Summed marginal log-likelihood under ``params_``.
        """
        return self._forward_ll(self.params_, sequences)

    def viterbi(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Most-likely latent-state path for one sequence (numpy log-space Viterbi over
        the reference-coded emissions and the input-driven per-timestep transitions).

        Parameters
        ----------
        X (np.ndarray)
            ``(T, n_features * n_time_bins)`` behavioural design.
        y (np.ndarray)
            The sequence's observed categorical emissions.

        Returns
        -------
        states (np.ndarray)
            ``(T,)`` integer most-likely state per timestep.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        p = {key: np.asarray(value, dtype=np.float64)
             for key, value in self.params_.items()}
        n_states, n_time = self.n_states, X.shape[0]

        e_nonref = np.einsum('td,kdc->tkc', X, p['W_emit']) + p['b_emit'][None]
        e_full = np.concatenate([np.zeros((n_time, n_states, 1)), e_nonref], axis=2)
        log_emit = np.take_along_axis(_np_log_softmax(e_full, axis=2),
                                      y[:, None, None], axis=2)[:, :, 0]
        t_logits = np.einsum('td,idj->tij', X, p['W_trans']) + p['b_trans'][None]
        eye = np.eye(n_states, dtype=bool)
        t_logits = np.where(eye[None], 0.0, t_logits)
        log_trans = _np_log_softmax(t_logits, axis=2)
        log_pi = p['log_pi'] - _logsumexp(p['log_pi'], axis=0)

        delta = np.empty((n_time, n_states))
        psi = np.zeros((n_time, n_states), dtype=np.int64)
        delta[0] = log_pi + log_emit[0]
        for t in range(1, n_time):
            scored = delta[t - 1][:, None] + log_trans[t]
            psi[t] = np.argmax(scored, axis=0)
            delta[t] = log_emit[t] + np.max(scored, axis=0)
        states = np.empty(n_time, dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n_time - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def bic(self, sequences: list) -> float:
        """
        Description
        -----------
        BIC on ``sequences``: ``-2 * loglik + n_params * log(n_obs)`` with the free-parameter count.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` tuples; ``n_obs`` is their total number of timesteps
            (floored at 1 inside the log).

        Returns
        -------
        bic (float)
            The Bayesian information criterion (lower is better); ``n_params`` is the
            free-parameter count :attr:`n_params`.
        """
        n_obs = int(sum(len(y) for _, y in sequences))
        return float(-2.0 * self.log_likelihood(sequences)
                     + self.n_params * np.log(max(n_obs, 1)))

    @property
    def n_params(self) -> int:
        """
        Description
        -----------
        Number of FREE parameters (for BIC), not the number of entries in
        ``params_``: ``log_pi`` is normalised (``K - 1`` free of ``K``); the
        emissions are already reference-coded (class 0 fixed at zero), so
        ``W_emit`` / ``b_emit`` give ``K * (C - 1) * (D + 1)``; the self-transition
        logit is pinned to 0, so the diagonals of ``W_trans`` / ``b_trans`` carry no
        information and ``K * (K - 1) * (D + 1)`` transition parameters are free.

        Parameters
        ----------
        None

        Returns
        -------
        n_params (int)
            ``(K - 1) + K * (C - 1) * (D + 1) + K * (K - 1) * (D + 1)``, with
            ``D = n_features * n_time_bins``.
        """
        n_states = self.n_states
        design_width = self.n_features * self.n_time_bins
        return int((n_states - 1)
                   + n_states * (self.n_classes - 1) * (design_width + 1)
                   + n_states * (n_states - 1) * (design_width + 1))

    def initial_state_distribution(self) -> np.ndarray:
        """
        Description
        -----------
        Fitted initial-state probability vector (the softmax of the ``log_pi``
        logits), a ``(K,)`` summary for the results record.

        Parameters
        ----------
        None

        Returns
        -------
        pi (np.ndarray)
            ``(K,)`` initial-state probabilities summing to 1.
        """
        log_pi = np.asarray(self.params_['log_pi'], dtype=np.float64)
        log_pi = log_pi - _logsumexp(log_pi, axis=0)
        return np.exp(log_pi)

    def mean_transition_matrix(self, sequences: list) -> np.ndarray:
        """
        Description
        -----------
        Mean ``(K, K)`` transition matrix averaged over all events in ``sequences``
        -- a single representative summary of the input-driven transitions, which
        actually vary from event to event with the design row ``X``. Row ``i`` is
        the average ``P(z = j | z_prev = i, x)`` over the events (self-transition
        pinned as reference), so each row sums to 1.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` tuples over which to average the transitions.

        Returns
        -------
        transition (np.ndarray)
            ``(K, K)`` mean transition matrix (rows sum to 1).
        """
        p = {key: np.asarray(value, dtype=np.float64) for key, value in self.params_.items()}
        eye = np.eye(self.n_states, dtype=bool)
        total = np.zeros((self.n_states, self.n_states))
        count = 0
        for X, _ in sequences:
            X = np.asarray(X, dtype=np.float64)
            t_logits = np.einsum('td,idj->tij', X, p['W_trans']) + p['b_trans'][None]
            t_logits = np.where(eye[None], 0.0, t_logits)
            total += np.exp(_np_log_softmax(t_logits, axis=2)).sum(axis=0)
            count += X.shape[0]
        return total / max(count, 1)


# The manifold input-driven engine shares the transition + forward machinery of the
# categorical one; only the emission differs (a per-state product von Mises around a
# per-state angular regressor prediction, instead of a reference-coded softmax). The
# torus target is 2-D. These helpers duplicate the (small) transition/scan block rather
# than refactor the validated categorical functions above, to keep that engine intact.
_IDM_TARGETS = 2


def _idm_seq_loglik(params: dict, X: jax.Array, y: jax.Array, w: jax.Array,
                    mask: jax.Array, period: jax.Array) -> jax.Array:
    """
    Description
    -----------
    Log-space forward marginal log-likelihood for ONE padded sequence under the
    input-driven MANIFOLD GLM-HMM of :class:`InputDrivenManifoldGLMHMM`. Each state
    predicts a per-axis angle ``mu_k(x) = x . W_emit[k] + b_emit[k]`` and emits the
    2-D torus position under a product von Mises with a per-state concentration
    ``exp(log_kappa[k])``; the transition into timestep ``t`` is the same input-driven
    softmax (self-transition pinned) as the categorical engine. Padded steps
    (``mask`` False) get zero emission and an identity transition. Each event's
    emission log-density is scaled by a per-event weight ``w`` (a WEIGHTED / tempered
    likelihood): with inverse-region-frequency weights this region-balances the fit so
    the common central vocalizations do not dominate the regressors (all weights 1.0 recovers
    the plain marginal likelihood). Weights scale the emission uniformly across states,
    so they leave the per-timestep state ranking -- and hence Viterbi decoding -- intact.

    Parameters
    ----------
    params (dict)
        ``log_pi`` ``(K,)``, ``W_emit`` ``(K, D, 2)``, ``b_emit`` ``(K, 2)``,
        ``log_kappa`` ``(K,)``, ``W_trans`` ``(K, D, K)``, ``b_trans`` ``(K, K)``.
    X (jax.Array)
        ``(T, D)`` padded behavioural design.
    y (jax.Array)
        ``(T, 2)`` padded torus positions in ``[0, period)``.
    w (jax.Array)
        ``(T,)`` per-event emission weights (1.0 = unweighted; padded steps ignored).
    mask (jax.Array)
        ``(T,)`` boolean real-vs-padding mask.
    period (jax.Array)
        Torus wrap period per axis.

    Returns
    -------
    seq_loglik (jax.Array)
        Scalar (weighted) marginal log-likelihood of the (unpadded) sequence.
    """
    n_states = params['b_emit'].shape[0]
    two_pi = 2.0 * jnp.pi
    theta = y * (two_pi / period)                                       # (T, 2) angle
    mu = jnp.einsum('td,kdc->tkc', X, params['W_emit']) + params['b_emit'][None]   # (T,K,2)
    kappa = jnp.exp(params['log_kappa'])                                # (K,)
    log_i0 = jnp.log(_jax_i0e(kappa)) + kappa                           # (K,) = log I0(kappa)
    cos_term = jnp.cos(theta[:, None, :] - mu)                          # (T,K,2)
    log_emit = ((kappa[None, :, None] * cos_term).sum(axis=2)
                - _IDM_TARGETS * (jnp.log(two_pi) + log_i0)[None, :])   # (T,K)
    log_emit = log_emit * w[:, None]                                    # per-event region weight
    log_emit = jnp.where(mask[:, None], log_emit, 0.0)

    t_logits = jnp.einsum('td,idj->tij', X, params['W_trans']) + params['b_trans'][None]
    eye = jnp.eye(n_states, dtype=bool)
    t_logits = jnp.where(eye[None], 0.0, t_logits)
    log_trans = t_logits - _jax_logsumexp(t_logits, axis=2, keepdims=True)
    identity_log = jnp.where(eye, 0.0, _ID_NEG)[None]
    log_trans = jnp.where(mask[:, None, None], log_trans, identity_log)

    log_pi = params['log_pi'] - _jax_logsumexp(params['log_pi'])
    a0 = log_pi + log_emit[0]

    def step(a_prev: jax.Array, inp: tuple) -> tuple:
        """
        Description
        -----------
        One forward-recursion step of the ``lax.scan``: combine the previous log-alpha
        with the transition log-matrix landing at this timestep (``logsumexp`` over the
        from-state axis) and add this timestep's emission log-density.

        Parameters
        ----------
        a_prev (jax.Array)
            ``(K,)`` log-alpha of the previous timestep (the scan carry).
        inp (tuple)
            ``(le_t, lt_t)``: the ``(K,)`` emission log-density of this timestep and the
            ``(K, K)`` log transition matrix into it (row = from-state, column = to-state).

        Returns
        -------
        a_t, None (tuple)
            The ``(K,)`` log-alpha of this timestep (the new carry) and ``None`` (no
            per-step output is stacked).
        """
        le_t, lt_t = inp
        a_t = le_t + _jax_logsumexp(a_prev[:, None] + lt_t, axis=0)
        return a_t, None

    a_last, _ = lax.scan(step, a0, (log_emit[1:], log_trans[1:]))
    return _jax_logsumexp(a_last)


def _idm_batch_loglik(params: dict, Xp: jax.Array, yp: jax.Array, wp: jax.Array,
                      mask: jax.Array, period: jax.Array) -> jax.Array:
    """
    Description
    -----------
    Total (weighted) forward marginal log-likelihood over a padded batch (see :func:`_idm_seq_loglik`).

    Parameters
    ----------
    params (dict)
        Parameter pytree (see :func:`_idm_seq_loglik`).
    Xp (jax.Array)
        ``(S, Tmax, D)`` padded design.
    yp (jax.Array)
        ``(S, Tmax, 2)`` padded torus positions.
    wp (jax.Array)
        ``(S, Tmax)`` per-event emission weights.
    mask (jax.Array)
        ``(S, Tmax)`` boolean real-step mask.
    period (jax.Array)
        Torus wrap period per axis.

    Returns
    -------
    loglik (jax.Array)
        Scalar sum of the per-sequence (weighted) marginal log-likelihoods.
    """
    per_seq = jax.vmap(
        lambda X, y, w, m: _idm_seq_loglik(params, X, y, w, m, period))(Xp, yp, wp, mask)
    return jnp.sum(per_seq)


def _idm_objective(params: dict, Xp: jax.Array, yp: jax.Array, wp: jax.Array,
                   mask: jax.Array, n_bins: jax.Array, r: jax.Array, n_features: int,
                   n_time: int, period: jax.Array) -> jax.Array:
    """
    Description
    -----------
    Per-bin-mean negative (weighted) forward marginal + r * SUM first-difference smoothness.

    Parameters
    ----------
    params (dict)
        Parameter pytree (see :func:`_idm_seq_loglik`).
    Xp, yp, wp, mask (jax.Array)
        Padded design, torus targets, per-event weights and real-step mask (see
        :func:`_idm_batch_loglik`).
    n_bins (jax.Array)
        Total number of real events, for the per-bin mean.
    r (jax.Array)
        First-difference smoothness coefficient.
    n_features (int)
        Number of behavioural cues.
    n_time (int)
        Number of temporal taps per cue.
    period (jax.Array)
        Torus wrap period per axis.

    Returns
    -------
    value (jax.Array)
        Scalar regularised objective (smoothness on ``W_emit`` and ``W_trans``).
    """
    data = -_idm_batch_loglik(params, Xp, yp, wp, mask, period) / n_bins
    smooth = (_id_first_diff(params['W_emit'], n_features, n_time)
              + _id_first_diff(params['W_trans'], n_features, n_time))
    return data + r * smooth


@partial(jax.jit, static_argnums=(5, 8, 9))
def _idm_run_lbfgs(init: dict, Xp: jax.Array, yp: jax.Array, wp: jax.Array, mask: jax.Array,
                   n_steps: int, n_bins: jax.Array, r: jax.Array, n_features: int,
                   n_time: int, period: jax.Array) -> tuple:
    """
    Description
    -----------
    Full-batch L-BFGS (zoom line search) on the manifold objective; whole solve in one scan.

    Parameters
    ----------
    init (dict)
        Initial parameters (see :func:`_idm_seq_loglik`), ``log_kappa`` included and
        optimised freely.
    Xp, yp, wp, mask (jax.Array)
        Padded design, torus targets, per-event weights and real-step mask.
    n_steps (int)
        Number of L-BFGS iterations (static; the ``lax.scan`` length).
    n_bins (jax.Array)
        Total number of real events, for the per-bin mean.
    r (jax.Array)
        First-difference smoothness coefficient.
    n_features (int)
        Number of behavioural cues (static).
    n_time (int)
        Number of temporal taps per cue (static).
    period (jax.Array)
        Torus wrap period per axis.

    Returns
    -------
    params, values (tuple[dict, jax.Array])
        The final parameters and the ``(n_steps,)`` per-iteration objective trace.
    """
    def fun(p: dict) -> jax.Array:
        """
        Description
        -----------
        Closure over the batch and hyperparameters: the manifold objective
        :func:`_idm_objective` as a function of the parameters alone.

        Parameters
        ----------
        p (dict)
            Parameter pytree (see :func:`_idm_seq_loglik`).

        Returns
        -------
        value (jax.Array)
            Scalar regularised objective.
        """
        return _idm_objective(p, Xp, yp, wp, mask, n_bins, r, n_features, n_time, period)

    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(fun)

    def step(carry: tuple, _: None) -> tuple:
        """
        Description
        -----------
        One L-BFGS iteration of the ``lax.scan``: evaluate the objective value and
        gradient (reusing the line-search cache in the optimiser state), compute the
        optax L-BFGS update with the zoom line search, and apply it.

        Parameters
        ----------
        carry (tuple)
            ``(params, state)``: the current parameter pytree and the optax L-BFGS state.
        _ (None)
            Unused scan input (the scan runs ``length=n_steps`` with no ``xs``).

        Returns
        -------
        (params, state), value (tuple)
            The updated ``(params, state)`` carry and the scalar objective value
            evaluated at the parameters BEFORE this update (stacked into the trace).
        """
        params, state = carry
        value, grad = value_and_grad(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return (params, state), value

    (params, _), values = lax.scan(step, (init, opt.init(init)), None, length=n_steps)
    return params, values


def _idm_objective_fixed_kappa(params_wo: dict, log_kappa: jax.Array, Xp: jax.Array,
                               yp: jax.Array, wp: jax.Array, mask: jax.Array,
                               n_bins: jax.Array, r: jax.Array, n_features: int,
                               n_time: int, period: jax.Array) -> jax.Array:
    """
    Description
    -----------
    The manifold objective (:func:`_idm_objective`) with the per-axis concentration
    ``log_kappa`` HELD FIXED -- i.e. only the angular regressors (``W_emit``,
    ``b_emit``) and the transition filters (``W_trans``, ``b_trans``, ``log_pi``) are
    exposed to the optimiser. This is the L-BFGS half of the alternating global-kappa
    fit: freely co-optimising ``log_kappa`` with ``W_emit`` lets kappa slide to 0 (a
    flat, near-uniform attractor) before the regressor develops, so the states never
    learn a mapping; freezing kappa keeps the emission gradient on ``W_emit`` sharp
    while the concentration is re-estimated by MLE between rounds (see
    :func:`_idm_global_kappa`).

    Parameters
    ----------
    params_wo (dict)
        The optimised parameters WITHOUT ``log_kappa`` (``log_pi``, ``W_emit``,
        ``b_emit``, ``W_trans``, ``b_trans``).
    log_kappa (jax.Array)
        ``(K,)`` frozen log-concentration merged back into the emission for scoring.
    Xp, yp, wp, mask (jax.Array)
        Padded design ``(S, Tmax, D)``, torus targets ``(S, Tmax, 2)``, per-event
        weights ``(S, Tmax)``, real-step mask.
    n_bins (jax.Array)
        Total real events (per-bin-mean normaliser of the negative log-likelihood).
    r (jax.Array)
        Smoothness weight on the SUM first-difference filter penalty.
    n_features (int)
        Number of base behavioural features (static; the filter block width).
    n_time (int)
        History taps per feature (static).
    period (jax.Array)
        Torus wrap period per axis.

    Returns
    -------
    objective (jax.Array)
        Scalar per-bin-mean negative marginal log-likelihood + ``r`` * filter smoothness.
    """
    params = {**params_wo, 'log_kappa': log_kappa}
    data = -_idm_batch_loglik(params, Xp, yp, wp, mask, period) / n_bins
    smooth = (_id_first_diff(params['W_emit'], n_features, n_time)
              + _id_first_diff(params['W_trans'], n_features, n_time))
    return data + r * smooth


@partial(jax.jit, static_argnums=(6, 9, 10))
def _idm_run_lbfgs_fixed_kappa(init_wo: dict, log_kappa: jax.Array, Xp: jax.Array,
                               yp: jax.Array, wp: jax.Array, mask: jax.Array,
                               n_steps: int, n_bins: jax.Array, r: jax.Array,
                               n_features: int, n_time: int, period: jax.Array) -> tuple:
    """
    Description
    -----------
    Full-batch L-BFGS on :func:`_idm_objective_fixed_kappa` (the frozen-``log_kappa``
    manifold objective); the whole solve runs inside one ``lax.scan``. Mirrors
    :func:`_idm_run_lbfgs` but optimises only ``init_wo`` (the parameters excluding
    ``log_kappa``), returning the optimised sub-dict and the per-step objective trace.

    Parameters
    ----------
    init_wo (dict)
        Initial parameters WITHOUT ``log_kappa`` (``log_pi``, ``W_emit``, ``b_emit``,
        ``W_trans``, ``b_trans``).
    log_kappa (jax.Array)
        ``(K,)`` frozen log-concentration.
    Xp, yp, wp, mask (jax.Array)
        Padded design ``(S, Tmax, D)``, torus targets ``(S, Tmax, 2)``, per-event
        weights ``(S, Tmax)``, real-step mask.
    n_steps (int)
        Number of L-BFGS iterations (static; the ``lax.scan`` length).
    n_bins (jax.Array)
        Total real events (per-bin-mean normaliser).
    r (jax.Array)
        Smoothness weight on the SUM first-difference filter penalty.
    n_features (int)
        Number of base behavioural features (static).
    n_time (int)
        History taps per feature (static).
    period (jax.Array)
        Torus wrap period per axis.

    Returns
    -------
    params, values (tuple[dict, jax.Array])
        The optimised sub-dict (no ``log_kappa``) and the ``(n_steps,)`` objective trace.
    """
    def fun(p: dict) -> jax.Array:
        """
        Description
        -----------
        Closure over ``log_kappa``, the batch and the hyperparameters: the
        frozen-concentration objective :func:`_idm_objective_fixed_kappa` as a
        function of the optimised sub-dict alone.

        Parameters
        ----------
        p (dict)
            Parameter pytree without ``log_kappa``.

        Returns
        -------
        value (jax.Array)
            Scalar regularised objective.
        """
        return _idm_objective_fixed_kappa(
            p, log_kappa, Xp, yp, wp, mask, n_bins, r, n_features, n_time, period)

    opt = optax.lbfgs()
    value_and_grad = optax.value_and_grad_from_state(fun)

    def step(carry: tuple, _: None) -> tuple:
        """
        Description
        -----------
        One L-BFGS iteration of the ``lax.scan``: evaluate the objective value and
        gradient (reusing the line-search cache in the optimiser state), compute the
        optax L-BFGS update with the zoom line search, and apply it.

        Parameters
        ----------
        carry (tuple)
            ``(params, state)``: the current parameter pytree and the optax L-BFGS state.
        _ (None)
            Unused scan input (the scan runs ``length=n_steps`` with no ``xs``).

        Returns
        -------
        (params, state), value (tuple)
            The updated ``(params, state)`` carry and the scalar objective value
            evaluated at the parameters BEFORE this update (stacked into the trace).
        """
        params, state = carry
        value, grad = value_and_grad(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun)
        params = optax.apply_updates(params, updates)
        return (params, state), value

    (params, _), values = lax.scan(step, (init_wo, opt.init(init_wo)), None, length=n_steps)
    return params, values


def _idm_global_kappa(params: dict, Xp: jax.Array, yp: jax.Array, wp: jax.Array,
                      mask: jax.Array, period: float) -> float:
    """
    Description
    -----------
    Responsibility-weighted global von Mises concentration MLE from the current
    per-state angular regressors -- the M-step for ``kappa`` in the alternating
    global-kappa fit. Each real event is assigned soft EMISSION responsibilities
    ``resp[n, k] = softmax_k p(y_n | state k)`` under the current ``kappa``, the
    per-axis angular residual ``r = wrap(theta - mu_k)`` is formed for every (event,
    state, axis), and a SINGLE global ``kappa`` is fit from the pooled, responsibility-
    weighted residuals via :func:`_fit_von_mises_kappa` (the ``i1e/i0e = Rbar`` Rbar
    solve). Sharing one concentration across states (and the two torus axes) isolates
    the scientific question to the state MAPPINGS (``W_emit``) rather than to per-state
    concentration, and -- unlike a freely optimised ``log_kappa`` -- cannot collapse to
    the uniform density. For ``K = 1`` this is exactly the single-regression residual
    MLE (Project-2's von Mises log-score concentration).

    Parameters
    ----------
    params (dict)
        Current engine parameters (``W_emit`` ``(K, D, 2)``, ``b_emit`` ``(K, 2)``,
        ``log_kappa`` ``(K,)`` used only to weight the responsibilities).
    Xp, yp, wp, mask (jax.Array)
        Padded design, torus targets, per-event emission weights, and real-step mask
        (padded steps are dropped). The per-event weights carry into the MLE so the
        concentration is region-balanced consistently with the reweighted fit.
    period (float)
        Torus wrap period per axis.

    Returns
    -------
    kappa (float)
        The pooled responsibility-and-region-weighted global concentration (Rbar MLE).
    """
    two_pi = 2.0 * np.pi
    W = np.asarray(params['W_emit'])
    b = np.asarray(params['b_emit'])
    kappa = np.exp(np.asarray(params['log_kappa']))                     # (K,)
    Xp = np.asarray(Xp)
    yp = np.asarray(yp)
    m = np.asarray(mask).reshape(-1)
    Xf = Xp.reshape(-1, Xp.shape[-1])[m]                                # (N, D)
    Yf = yp.reshape(-1, _IDM_TARGETS)[m]                                # (N, 2)
    wf = np.asarray(wp).reshape(-1)[m]                                  # (N,) region weight
    theta = Yf * (two_pi / period)                                      # (N, 2)
    mu = np.einsum('nd,kdc->nkc', Xf, W) + b[None]                      # (N, K, 2)
    log_i0 = np.log(_np_i0e(kappa)) + kappa                             # (K,)
    cos_term = np.cos(theta[:, None, :] - mu)                           # (N, K, 2)
    log_emit = ((kappa[None, :, None] * cos_term).sum(axis=2)
                - _IDM_TARGETS * (np.log(two_pi) + log_i0)[None])       # (N, K)
    log_emit -= log_emit.max(axis=1, keepdims=True)
    resp = np.exp(log_emit)
    resp /= resp.sum(axis=1, keepdims=True)                            # (N, K)
    residual = np.angle(np.exp(1j * (theta[:, None, :] - mu)))          # (N, K, 2) in (-pi, pi]
    weights = np.broadcast_to((resp * wf[:, None])[:, :, None],
                              residual.shape).reshape(-1)
    return _fit_von_mises_kappa(residual.reshape(-1), weights=weights)


class InputDrivenManifoldGLMHMM:
    """
    Description
    -----------
    GLM-HMM with INPUT-DRIVEN transitions and a per-state torus (product von Mises)
    MANIFOLD emission, fit by direct maximisation of the regularised marginal
    log-likelihood. This is the continuous-vocal-space counterpart of the categorical
    :class:`InputDrivenGLMHMM`: it represents the acoustic output as a 2-D position on
    the QLVM torus (a continuum) rather than as a discrete category, while keeping the
    same Calhoun/Pillow/Murthy-faithful input-driven transitions and direct-marginal
    (non-EM) fit that avoids state collapse.

    Each latent state owns an angular regressor ``mu_k(x) = x . W_emit[k] + b_emit[k]``
    (per axis) and a concentration ``exp(log_kappa[k])``; the emission density of a
    torus position ``y`` is the product von Mises ``prod_axis vM(theta | mu_k, kappa_k)``
    with ``theta = y * 2pi / period``. Transitions into each timestep are per-"from"-state
    input-driven softmaxes of the same design (self-transition as reference). All
    parameters are optimised jointly by optax L-BFGS under a first-difference sum-scaled
    temporal-smoothness penalty on the filters. The forward pass is a JAX ``lax.scan``
    vmapped over padded sequences.

    Parameters
    ----------
    n_states (int)
        Number of latent states ``K``.
    n_features (int)
        Number of behavioural cues ``U`` (feature-major design blocks).
    n_time_bins (int)
        Number of temporal taps ``L`` per cue; the design width is ``U * L``.
    period (float)
        Torus wrap period per axis (the manifold coordinate range).
    lambda_smooth (float)
        First-difference temporal-smoothness coefficient ``r`` (sum-scaled).
    n_restarts (int)
        Number of random-init restarts; the best by held-out (or training) forward
        log-likelihood is kept.
    n_lbfgs (int)
        L-BFGS iterations per restart.
    random_state (int)
        Base seed for the structured random-sign filter initialisation.
    verbose (bool)
        Print per-restart diagnostics when True.

    Returns
    -------
    None
    """

    def __init__(self, n_states: int, n_features: int, n_time_bins: int,
                 *, period: float = 1.0, lambda_smooth: float = 0.05, n_restarts: int = 4,
                 n_lbfgs: int = 600, random_state: int = 0, verbose: bool = False,
                 kappa_mode: str = 'global_mle', n_kappa_rounds: int = 3) -> None:
        """
        Description
        -----------
        Store the model configuration (cast to the documented types), validate
        ``kappa_mode``, and initialise the fitted attributes ``params_`` and
        ``log_likelihood_`` to ``None``.

        Parameters
        ----------
        n_states (int)
            Number of latent states ``K``.
        n_features (int)
            Number of behavioural cues ``U``.
        n_time_bins (int)
            Number of temporal taps ``L`` per cue.
        period (float)
            Torus wrap period per axis; defaults to 1.0.
        lambda_smooth (float)
            First-difference temporal-smoothness coefficient; defaults to 0.05.
        n_restarts (int)
            Number of random-init restarts; defaults to 4.
        n_lbfgs (int)
            L-BFGS iterations per restart (per round in ``'global_mle'`` mode);
            defaults to 600.
        random_state (int)
            Base seed for the filter initialisation; defaults to 0.
        verbose (bool)
            Print per-restart diagnostics when True; defaults to False.
        kappa_mode (str)
            ``'global_mle'`` (default) alternates L-BFGS at frozen concentration with a
            responsibility-weighted global concentration MLE
            (:meth:`_fit_restart_global_kappa`); ``'free'`` co-optimises ``log_kappa`` in
            L-BFGS. Any other value raises ``ValueError``.
        n_kappa_rounds (int)
            Number of L-BFGS / concentration-MLE alternations per restart in
            ``'global_mle'`` mode; defaults to 3.

        Returns
        -------
        None
        """
        self.n_states = int(n_states)
        self.n_features = int(n_features)
        self.n_time_bins = int(n_time_bins)
        self.period = float(period)
        self.lambda_smooth = float(lambda_smooth)
        self.n_restarts = int(n_restarts)
        self.n_lbfgs = int(n_lbfgs)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)
        # 'global_mle' (default) alternates L-BFGS on the regressors/transitions with a
        # responsibility-weighted global von Mises concentration MLE, which prevents the
        # free-``log_kappa`` collapse to the uniform density (see _idm_global_kappa);
        # 'free' co-optimises ``log_kappa`` in L-BFGS (the original behaviour, kept for
        # comparison and the machinery tests).
        if kappa_mode not in ('global_mle', 'free'):
            message = f"kappa_mode must be 'global_mle' or 'free'; got {kappa_mode!r}."
            raise ValueError(message)
        self.kappa_mode = kappa_mode
        self.n_kappa_rounds = int(n_kappa_rounds)
        self.params_ = None
        self.log_likelihood_ = None

    def _init_params(self, seed: int) -> dict:
        """
        Description
        -----------
        Structured random-sign decaying-exponential filter init for the angular
        regressor + transition filters (breaking state symmetry), a moderate initial
        concentration, and a uniform initial state. Shapes: ``W_emit`` ``(K, D, 2)``,
        ``W_trans`` ``(K, D, K)``.

        Parameters
        ----------
        seed (int)
            Seed for the ``np.random.default_rng`` drawing the signs and biases.

        Returns
        -------
        params (dict)
            float32 JAX arrays ``log_pi`` ``(K,)`` (zeros), ``W_emit`` ``(K, D, 2)``,
            ``b_emit`` ``(K, 2)`` (uniform on ``[0, 2 pi)``), ``log_kappa`` ``(K,)``
            (all ``log 2``), ``W_trans`` ``(K, D, K)`` and ``b_trans`` ``(K, K)``
            (normal, sd 0.3).
        """
        rng = np.random.default_rng(seed)
        design_width = self.n_features * self.n_time_bins
        decay = np.exp(-np.arange(self.n_time_bins) / self.n_time_bins)

        def structured(lead: tuple) -> np.ndarray:
            """
            Description
            -----------
            Build one structured random-sign filter block: for every leading index and every
            behavioural cue, a random sign drawn from ``{-1, 0, +1}`` (a uniform draw on
            ``[-1, 1]`` rounded to the nearest integer) multiplies the decaying exponential
            ``exp(-arange(L) / L)`` over the ``L`` temporal taps, and the cue blocks are
            flattened feature-major into the design width ``U * L``. Draws from the enclosing
            method's seeded ``rng``.

            Parameters
            ----------
            lead (tuple)
                Leading shape of the block (e.g. ``(n_states,)``).

            Returns
            -------
            block (np.ndarray)
                ``(*lead, n_features * n_time_bins)`` float32 filter block.
            """
            signs = np.round(rng.uniform(-1, 1, size=(*lead, self.n_features, 1)))
            block = signs * decay[None, :]
            return block.reshape((*lead, design_width)).astype(np.float32)

        n_states = self.n_states
        W_emit = np.stack([structured((n_states,)) for _ in range(_IDM_TARGETS)], axis=-1)
        W_trans = np.stack([structured((n_states,)) for _ in range(n_states)], axis=-1)
        b_emit = rng.uniform(0, 2.0 * np.pi, (n_states, _IDM_TARGETS))
        b_trans = rng.normal(0, 0.3, (n_states, n_states))
        return {
            'log_pi': jnp.zeros((n_states,), dtype=jnp.float32),
            'W_emit': jnp.asarray(W_emit, dtype=jnp.float32),
            'b_emit': jnp.asarray(b_emit, dtype=jnp.float32),
            'log_kappa': jnp.full((n_states,), np.log(2.0), dtype=jnp.float32),
            'W_trans': jnp.asarray(W_trans, dtype=jnp.float32),
            'b_trans': jnp.asarray(b_trans, dtype=jnp.float32),
        }

    def _pad_stack(self, sequences: list) -> tuple:
        """
        Description
        -----------
        Pad a list of sequences to a common ``Tmax``. Each sequence is ``(X, y)`` or
        ``(X, y, w)`` -- ``y`` a ``(T, 2)`` torus position and ``w`` an optional
        ``(T,)`` per-event emission weight (defaults to all-ones when absent). Returns
        the padded ``(Xp, yp, wp, mask)``.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` or ``(X, y, w)`` tuples.

        Returns
        -------
        Xp, yp, wp, mask (tuple[jax.Array, jax.Array, jax.Array, jax.Array])
            float32 design ``(S, Tmax, D)``, float32 torus targets ``(S, Tmax, 2)``,
            float32 weights ``(S, Tmax)`` and boolean real-step mask ``(S, Tmax)``;
            padded entries are zero / False.
        """
        design_width = self.n_features * self.n_time_bins
        lengths = np.asarray([len(seq[1]) for seq in sequences], dtype=np.int32)
        n_max = int(lengths.max())
        n_seqs = len(sequences)
        Xp = np.zeros((n_seqs, n_max, design_width), dtype=np.float32)
        yp = np.zeros((n_seqs, n_max, _IDM_TARGETS), dtype=np.float32)
        wp = np.zeros((n_seqs, n_max), dtype=np.float32)
        mask = np.zeros((n_seqs, n_max), dtype=bool)
        for s, seq in enumerate(sequences):
            X, y = seq[0], seq[1]
            w = seq[2] if len(seq) > 2 else np.ones(len(y), dtype=np.float32)
            n_ev = len(y)
            Xp[s, :n_ev] = X
            yp[s, :n_ev] = y
            wp[s, :n_ev] = w
            mask[s, :n_ev] = True
        return jnp.asarray(Xp), jnp.asarray(yp), jnp.asarray(wp), jnp.asarray(mask)

    def _forward_ll(self, params: dict, sequences: list) -> float:
        """
        Description
        -----------
        Total (weighted) forward marginal log-likelihood of ``sequences`` under ``params``.

        Parameters
        ----------
        params (dict)
            Parameter pytree (see :func:`_idm_seq_loglik`).
        sequences (list)
            List of ``(X, y)`` or ``(X, y, w)`` tuples.

        Returns
        -------
        loglik (float)
            Summed (weighted) marginal log-likelihood over the sequences.
        """
        Xp, yp, wp, mask = self._pad_stack(sequences)
        return float(_idm_batch_loglik(params, Xp, yp, wp, mask, self.period))

    def fit(self, sequences: list, held_out_sequences: list | None = None) -> "InputDrivenManifoldGLMHMM":
        """
        Description
        -----------
        Fit by direct marginal-likelihood maximisation with random restarts, keeping
        the restart with the best forward log-likelihood on ``held_out_sequences`` (or
        the training sequences when none is given).

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` or ``(X, y, w)`` tuples; ``X`` is
            ``(T, n_features * n_time_bins)``, ``y`` is the ``(T, 2)`` torus position,
            and the optional ``w`` is a ``(T,)`` per-event emission weight (defaults to
            all-ones -- e.g. inverse-region-frequency weights to region-balance the fit).
        held_out_sequences (list)
            Optional validation sequences for restart selection.

        Returns
        -------
        self (InputDrivenManifoldGLMHMM)
            The fitted engine, with ``params_`` and ``log_likelihood_``.
        """
        if not sequences:
            message = "InputDrivenManifoldGLMHMM.fit requires at least one sequence."
            raise ValueError(message)
        Xp, yp, wp, mask = self._pad_stack(sequences)
        n_bins = int(sum(len(seq[1]) for seq in sequences))
        selection = held_out_sequences if held_out_sequences else sequences

        best_score = -np.inf
        best_params = None
        for restart in range(self.n_restarts):
            init = self._init_params(self.random_state + restart)
            if self.kappa_mode == 'global_mle':
                params = self._fit_restart_global_kappa(init, Xp, yp, wp, mask, n_bins)
                if params is None:
                    continue
            else:
                params, values = _idm_run_lbfgs(
                    init, Xp, yp, wp, mask, self.n_lbfgs, n_bins, self.lambda_smooth,
                    self.n_features, self.n_time_bins, self.period)
                if not np.isfinite(float(values[-1])):
                    continue
            score = self._forward_ll(params, selection)
            if self.verbose:
                print(f"Restart {restart}: selection_ll={score:.1f}", flush=True)
            if score > best_score:
                best_score = score
                best_params = params

        if best_params is None:
            message = "InputDrivenManifoldGLMHMM.fit: all restarts diverged."
            raise RuntimeError(message)
        self.params_ = best_params
        self.log_likelihood_ = self._forward_ll(best_params, sequences)
        return self

    def _fit_restart_global_kappa(self, init: dict, Xp: jax.Array, yp: jax.Array,
                                  wp: jax.Array, mask: jax.Array,
                                  n_bins: int) -> dict | None:
        """
        Description
        -----------
        One random restart under ``kappa_mode='global_mle'``: alternate
        ``n_kappa_rounds`` times between (i) an L-BFGS solve of the angular regressors +
        transition filters at FROZEN concentration (:func:`_idm_run_lbfgs_fixed_kappa`)
        and (ii) a responsibility-weighted global concentration MLE
        (:func:`_idm_global_kappa`) that replaces ``log_kappa`` for the next round. The
        first round starts from the init's moderate ``log_kappa`` (log 2) so the
        regressor gradient is sharp; subsequent rounds re-estimate the concentration from
        the improved residuals. Returns ``None`` if an L-BFGS solve diverges (non-finite
        objective) so the caller can skip the restart.

        Parameters
        ----------
        init (dict)
            Structured random-restart parameters (includes the starting ``log_kappa``).
        Xp, yp, wp, mask (jax.Array)
            Padded design ``(S, Tmax, D)``, torus targets ``(S, Tmax, 2)``, per-event
            weights ``(S, Tmax)``, real-step mask.
        n_bins (int)
            Total real events (objective normaliser).

        Returns
        -------
        params (dict or None)
            The fitted parameters (with the final global ``log_kappa`` broadcast over the
            states), or ``None`` if a round diverged.
        """
        params_wo = {k: v for k, v in init.items() if k != 'log_kappa'}
        log_kappa = init['log_kappa']
        for _round in range(self.n_kappa_rounds):
            params_wo, values = _idm_run_lbfgs_fixed_kappa(
                params_wo, log_kappa, Xp, yp, wp, mask, self.n_lbfgs, n_bins,
                self.lambda_smooth, self.n_features, self.n_time_bins, self.period)
            if not np.isfinite(float(values[-1])):
                return None
            kappa = _idm_global_kappa(
                {**params_wo, 'log_kappa': log_kappa}, Xp, yp, wp, mask, self.period)
            log_kappa = jnp.full(
                (self.n_states,), float(np.log(max(kappa, 1e-6))), dtype=jnp.float32)
        return {**params_wo, 'log_kappa': log_kappa}

    def log_likelihood(self, sequences: list) -> float:
        """
        Description
        -----------
        Total forward marginal log-likelihood of ``sequences`` under the fit model.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` or ``(X, y, w)`` tuples (any ``w`` weights the emissions).

        Returns
        -------
        loglik (float)
            Summed marginal log-likelihood under ``params_``.
        """
        return self._forward_ll(self.params_, sequences)

    def macro_score(self, sequences: list, region_labels: list,
                    min_region_events: int = 1) -> float:
        """
        Description
        -----------
        Region-balanced (macro) von Mises held-out log-score of the model's per-event
        position predictions -- Project-2's :func:`macro_von_mises_logscore`, so the
        switching regression's held-out quality is read on the SAME scale as the
        single-map baseline (uniform torus ``-2*log(2*pi)`` ~ -3.676; Project-2 single
        map ~ -3.626). Each event's predicted torus position is the Viterbi-decoded
        state's angular regressor ``mu_state(x)`` wrapped back into ``[0, period)``; a
        single global concentration is fit from the residuals and the per-point log-
        density is macro-averaged over the acoustic regions (QLVM categories; equal weight per
        region). Predictions are unweighted -- any fit reweighting only shaped ``W_emit``,
        not this score. With no labelled event at all (no category labels, e.g.
        a pickle extracted without a label column) the score is the pooled one, and
        :func:`macro_von_mises_logscore` prints (once per process) that it fell back.

        Parameters
        ----------
        sequences (list)
            ``(X, y)`` / ``(X, y, w)`` tuples (any ``w`` is ignored for scoring).
        region_labels (list)
            Per-sequence ``(T,)`` QLVM category labels aligned to each sequence's events
            (NaN marks an unlabelled event).
        min_region_events (int)
            Minimum labelled events for a region to enter the macro average.

        Returns
        -------
        score (float)
            The macro von Mises log-score (higher is better); NaN if no region qualifies.
        """
        two_pi = 2.0 * np.pi
        W = np.asarray(self.params_['W_emit'])
        b = np.asarray(self.params_['b_emit'])
        pred_blocks = []
        true_blocks = []
        region_blocks = []
        for seq, regions in zip(sequences, region_labels, strict=True):
            X = np.asarray(seq[0], dtype=np.float64)
            y = np.asarray(seq[1], dtype=np.float64)
            states = np.asarray(self.viterbi(seq[0], seq[1]))
            mu = np.einsum('td,kdc->tkc', X, W) + b[None]              # (T, K, 2) angle
            mu_sel = mu[np.arange(states.shape[0]), states]            # (T, 2) angle
            pred_blocks.append(((mu_sel / two_pi) % 1.0) * self.period)
            true_blocks.append(y)
            region_blocks.append(np.asarray(regions, dtype=np.float64))
        Y_pred = np.vstack(pred_blocks)
        Y_true = np.vstack(true_blocks)
        regions_all = np.concatenate(region_blocks)
        return macro_von_mises_logscore(
            Y_pred, Y_true, regions_all, metric='torus', period=self.period,
            min_region_events=min_region_events)

    def viterbi(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Most-likely latent-state path for one sequence (numpy log-space Viterbi over
        the product-von-Mises emissions and the input-driven per-timestep transitions).

        Parameters
        ----------
        X (np.ndarray)
            ``(T, n_features * n_time_bins)`` behavioural design.
        y (np.ndarray)
            ``(T, 2)`` observed torus positions.

        Returns
        -------
        states (np.ndarray)
            ``(T,)`` integer most-likely state per timestep.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        p = {key: np.asarray(value, dtype=np.float64)
             for key, value in self.params_.items()}
        n_states, n_time = self.n_states, X.shape[0]
        two_pi = 2.0 * np.pi

        theta = y * (two_pi / self.period)
        mu = np.einsum('td,kdc->tkc', X, p['W_emit']) + p['b_emit'][None]
        kappa = np.exp(p['log_kappa'])
        log_i0 = np.log(_np_i0e(kappa)) + kappa
        cos_term = np.cos(theta[:, None, :] - mu)
        log_emit = ((kappa[None, :, None] * cos_term).sum(axis=2)
                    - _IDM_TARGETS * (np.log(two_pi) + log_i0)[None, :])
        t_logits = np.einsum('td,idj->tij', X, p['W_trans']) + p['b_trans'][None]
        eye = np.eye(n_states, dtype=bool)
        t_logits = np.where(eye[None], 0.0, t_logits)
        log_trans = _np_log_softmax(t_logits, axis=2)
        log_pi = p['log_pi'] - _logsumexp(p['log_pi'], axis=0)

        delta = np.empty((n_time, n_states))
        psi = np.zeros((n_time, n_states), dtype=np.int64)
        delta[0] = log_pi + log_emit[0]
        for t in range(1, n_time):
            scored = delta[t - 1][:, None] + log_trans[t]
            psi[t] = np.argmax(scored, axis=0)
            delta[t] = log_emit[t] + np.max(scored, axis=0)
        states = np.empty(n_time, dtype=np.int64)
        states[-1] = int(np.argmax(delta[-1]))
        for t in range(n_time - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def bic(self, sequences: list) -> float:
        """
        Description
        -----------
        BIC on ``sequences``: ``-2 * loglik + n_params * log(n_obs)`` with the free-parameter count.

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` or ``(X, y, w)`` tuples; ``n_obs`` is their total
            number of events (floored at 1 inside the log).

        Returns
        -------
        bic (float)
            The Bayesian information criterion (lower is better); ``n_params`` is the
            free-parameter count :attr:`n_params`.
        """
        n_obs = int(sum(len(seq[1]) for seq in sequences))
        return float(-2.0 * self.log_likelihood(sequences)
                     + self.n_params * np.log(max(n_obs, 1)))

    @property
    def n_params(self) -> int:
        """
        Description
        -----------
        Number of FREE parameters (for BIC), not the number of entries in
        ``params_``: ``log_pi`` is normalised (``K - 1`` free of ``K``); the angular
        regressors ``W_emit`` / ``b_emit`` give ``K * 2 * (D + 1)``; the
        concentration is ONE shared value under ``kappa_mode='global_mle'`` (stored
        broadcast over the ``K`` states) and ``K`` values under ``'free'``; the
        self-transition logit is pinned to 0, so ``K * (K - 1) * (D + 1)``
        transition parameters are free.

        Parameters
        ----------
        None

        Returns
        -------
        n_params (int)
            ``(K - 1) + K * 2 * (D + 1) + n_kappa + K * (K - 1) * (D + 1)``, with
            ``D = n_features * n_time_bins`` and ``n_kappa`` 1 or ``K``.
        """
        n_states = self.n_states
        design_width = self.n_features * self.n_time_bins
        n_kappa = 1 if self.kappa_mode == 'global_mle' else n_states
        return int((n_states - 1)
                   + n_states * _IDM_TARGETS * (design_width + 1)
                   + n_kappa
                   + n_states * (n_states - 1) * (design_width + 1))

    def initial_state_distribution(self) -> np.ndarray:
        """
        Description
        -----------
        Fitted initial-state probability vector (softmax of the ``log_pi`` logits).

        Parameters
        ----------
        None

        Returns
        -------
        pi (np.ndarray)
            ``(K,)`` initial-state probabilities summing to 1.
        """
        log_pi = np.asarray(self.params_['log_pi'], dtype=np.float64)
        log_pi = log_pi - _logsumexp(log_pi, axis=0)
        return np.exp(log_pi)

    def mean_transition_matrix(self, sequences: list) -> np.ndarray:
        """
        Description
        -----------
        Mean ``(K, K)`` input-driven transition matrix over all events (rows sum to 1).

        Parameters
        ----------
        sequences (list)
            List of ``(X, y)`` or ``(X, y, w)`` tuples over which to average the
            transitions (only ``X`` is read).

        Returns
        -------
        transition (np.ndarray)
            ``(K, K)`` mean transition matrix (rows sum to 1).
        """
        p = {key: np.asarray(value, dtype=np.float64) for key, value in self.params_.items()}
        eye = np.eye(self.n_states, dtype=bool)
        total = np.zeros((self.n_states, self.n_states))
        count = 0
        for seq in sequences:
            X = np.asarray(seq[0], dtype=np.float64)
            t_logits = np.einsum('td,idj->tij', X, p['W_trans']) + p['b_trans'][None]
            t_logits = np.where(eye[None], 0.0, t_logits)
            total += np.exp(_np_log_softmax(t_logits, axis=2)).sum(axis=0)
            count += X.shape[0]
        return total / max(count, 1)
