"""
@author: bartulem
Closed-form sin-cos embedding regression for the TORUS manifold.

The coordinate-output :class:`SmoothBivariateRegression` predicts the 2-D
manifold coordinate directly (``X @ W + b``) and optimises a wrap-aware Huber
loss with gradient descent. That model is structurally blind to behaviour ->
position relationships that *wind* around the torus: the wrap-aware loss is
non-convex, so descent from the flat init cannot climb to a wound solution and
the model returns a held-out R^2 that is indistinguishable from "no signal"
even when a strong relationship exists. (Empirically, on a known wound target
it returns R^2 ~ -0.1 against an oracle of +1.0, regardless of restarts.)

This estimator fixes that for ``metric='torus'`` by predicting the **4-D torus
embedding** ``[cos(2*pi*x), sin(2*pi*x), cos(2*pi*y), sin(2*pi*y)]`` instead of
the raw coordinate. The embedding is a smooth, periodic target with no wrap
discontinuity, so any winding becomes an ordinary smooth function and the fit
is a **convex generalised ridge** (L2 + temporal-smoothness penalty) solved in
**closed form** -- which also eliminates the convergence pathology and the
L2-grid-ceiling sensitivity of the iterative coordinate model. Predictions are
decoded back to ``(x, y)`` with ``atan2`` so every downstream metric (the
wrap-aware ``r2_spatial`` bundle in :meth:`evaluate_metrics`) is computed on the
native 2-D coordinates exactly as before.

It is used ONLY for ``metric='torus'``. Euclidean / VAE runs continue to use the
unchanged coordinate :class:`SmoothBivariateRegression`, so their results are
byte-identical -- this class subclasses it and overrides only :meth:`fit` and
:meth:`predict`, inheriting :meth:`evaluate_metrics` (which operates purely on
:meth:`predict` output) untouched.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.linalg import block_diag
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from .jax_bivariate_regression import SmoothBivariateRegression
from .manifold_metric import (
    circular_mean,
    signed_diff,
    torus_embed,
    _validate_metric_period,
)


class SmoothTorusManifoldRegression(SmoothBivariateRegression):
    """
    Description
    -----------
    Generalised-ridge regression from a behavioural-history design matrix to
    the 4-D torus embedding of the manifold position, with a closed-form
    solve, decoded back to 2-D coordinates. A drop-in replacement for
    :class:`SmoothBivariateRegression` whenever ``metric='torus'``; it accepts
    the identical constructor signature (the iterative-optimiser arguments
    ``huber_delta`` / ``learning_rate`` / ``max_iter`` / ``tol`` /
    ``_use_lax_loop`` are accepted for API compatibility but unused, since the
    objective is convex and solved exactly).

    Overrides only :meth:`fit` (the closed-form embedding solve and the
    metric-aware train statistics) and :meth:`predict` (decode the 4-D output
    to 2-D coordinates before the optional manifold snap). Everything else --
    notably :meth:`evaluate_metrics`, which consumes :meth:`predict` output and
    the inherited ``train_mean_`` / ``train_cov_inv_`` / ``_train_kdtree`` --
    is inherited unchanged, so the persisted metric bundle is identical in
    schema to the coordinate model.

    Parameters
    ----------
    See :class:`SmoothBivariateRegression`. ``metric`` must be ``'torus'``.

    Returns
    -------
    None
    """

    def _encode(self, Y: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Map 2-D torus coordinates in ``[0, period)`` to their 4-D canonical
        embedding ``[cos(theta_x), sin(theta_x), cos(theta_y), sin(theta_y)]``
        where ``theta = 2*pi*coord/period``. This is the smooth, wrap-free
        regression target.

        Parameters
        ----------
        Y (np.ndarray)
            ``(n_samples, 2)`` coordinate matrix.

        Returns
        -------
        E (np.ndarray)
            ``(n_samples, 4)`` embedding matrix (float64).
        """

        a = 2.0 * np.pi * np.asarray(Y, dtype=np.float64) / self.period
        return np.column_stack([
            np.cos(a[:, 0]), np.sin(a[:, 0]),
            np.cos(a[:, 1]), np.sin(a[:, 1]),
        ])

    def _decode(self, E: np.ndarray) -> np.ndarray:
        """
        Description
        -----------
        Decode a (generally un-normalised) 4-D embedding prediction back to
        2-D torus coordinates in ``[0, period)`` via ``atan2`` per axis. The
        magnitude of each ``(cos, sin)`` pair is irrelevant to ``atan2``, so
        the unconstrained linear prediction needs no projection onto the unit
        circle before decoding.

        Parameters
        ----------
        E (np.ndarray)
            ``(n_samples, 4)`` predicted embedding.

        Returns
        -------
        Y (np.ndarray)
            ``(n_samples, 2)`` decoded coordinates in ``[0, period)``.
        """

        ang_x = np.arctan2(E[:, 1], E[:, 0])
        ang_y = np.arctan2(E[:, 3], E[:, 2])
        px = (ang_x / (2.0 * np.pi) * self.period) % self.period
        py = (ang_y / (2.0 * np.pi) * self.period) % self.period
        return np.column_stack([px, py]).astype(np.float64)

    def _smoothness_penalty(self) -> np.ndarray:
        """
        Description
        -----------
        Build the temporal-smoothness penalty matrix ``S = D_k^T D_k`` applied
        to each output column of the filter, where ``D_k`` is the
        ``smoothness_derivative_order``-th finite-difference operator along the
        ``n_time_bins`` time axis. For ``n_features > 1`` the penalty is
        block-diagonal (one identical block per feature), matching the
        per-feature temporal-smoothness term of the coordinate model's loss
        (``jnp.diff(W.reshape(n_feats, n_time, .), n=order, axis=1)``).

        Parameters
        ----------

        Returns
        -------
        S (np.ndarray)
            ``(n_features*n_time_bins, n_features*n_time_bins)`` penalty matrix.
        """

        p = int(self.n_time_bins)
        d_k = np.diff(np.eye(p), n=int(self.smoothness_derivative_order), axis=0)
        block = d_k.T @ d_k
        if int(self.n_features) == 1:
            return block
        return block_diag(*([block] * int(self.n_features)))

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> "SmoothTorusManifoldRegression":
        """
        Description
        -----------
        Fit the closed-form generalised ridge from ``X`` to the 4-D torus
        embedding of ``y``. The objective is the inverse-density-weighted
        squared error on the embedding plus an L2 penalty and the
        temporal-smoothness penalty:

            min_W  sum_i w_i ||E_i - (X_i W + b)||^2
                   + l2_reg * ||W||^2 + lambda_smooth * ||D_k W||^2

        solved exactly via the normal equations (the intercept is absorbed by
        weighted-centering ``X`` and ``E``). The same closed-form ``W`` (shape
        ``(n_inputs, 4)``) and ``b`` (shape ``(4,)``) serve all four embedding
        components. Because the problem is convex, ``converged_`` is always
        ``True`` and ``n_iter_`` is ``1``.

        The training-set statistics (kd-tree on the 4-D torus embedding for
        snapping, circular-mean centroid and wrap-aware inverse covariance for
        the Mahalanobis metric) are computed exactly as in
        :class:`SmoothBivariateRegression` so the inherited
        :meth:`evaluate_metrics` and :meth:`predict` snap behaviour are
        unchanged.

        Parameters
        ----------
        X (np.ndarray)
            ``(n_samples, n_features*n_time_bins)`` design matrix.
        y (np.ndarray)
            ``(n_samples, 2)`` torus coordinates in ``[0, period)``.
        sample_weight (np.ndarray, optional)
            ``(n_samples,)`` inverse-density weights; defaults to uniform.
            Normalised to unit mean internally (matching the coordinate model)
            so ``l2_reg`` / ``lambda_smooth`` are decoupled from the weight
            scale.

        Returns
        -------
        self (SmoothTorusManifoldRegression)
            The fitted estimator.
        """

        if self.metric != 'torus':
            raise ValueError(
                f"SmoothTorusManifoldRegression requires metric='torus'; got {self.metric!r}."
            )
        _validate_metric_period(self.metric, self.period)
        fit_start = time.perf_counter()

        X, y = check_X_y(X, y, accept_sparse=False, multi_output=True)
        n_samples, n_inputs = X.shape
        expected_inputs = int(self.n_features) * int(self.n_time_bins)
        if n_inputs != expected_inputs:
            raise ValueError(
                f"Input X has {n_inputs} columns, but init parameters expect "
                f"n_features({self.n_features}) * n_time_bins({self.n_time_bins}) "
                f"= {expected_inputs} columns."
            )
        if y.shape[1] != 2:
            raise ValueError(
                f"Target y must have exactly 2 columns (torus x and y). Found shape {y.shape}."
            )

        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = check_array(sample_weight, ensure_2d=False)
        sample_weight = sample_weight / (np.mean(sample_weight) + 1e-12)

        # Closed-form weighted generalised ridge to the embedding target.
        emb = self._encode(y)
        w = np.asarray(sample_weight, dtype=np.float64)
        x_d = np.asarray(X, dtype=np.float64)
        w_sum = float(np.sum(w))
        x_mean = (w[:, None] * x_d).sum(axis=0) / w_sum
        e_mean = (w[:, None] * emb).sum(axis=0) / w_sum
        x_c = x_d - x_mean
        e_c = emb - e_mean

        penalty_s = self._smoothness_penalty()
        x_cw = x_c * w[:, None]
        a_mat = (
            x_cw.T @ x_c
            + float(self.l2_reg) * np.eye(n_inputs)
            + float(self.lambda_smooth) * penalty_s
        )
        coef = np.linalg.solve(a_mat, x_cw.T @ e_c)
        intercept = e_mean - x_mean @ coef

        self.coef_ = coef.astype(np.float64)
        self.intercept_ = intercept.astype(np.float64)
        self.n_iter_ = 1
        self.converged_ = True
        self.fit_time_ = float(time.perf_counter() - fit_start)

        # Training-set spatial structures: circular-mean centroid + wrap-aware
        # inverse covariance for the Mahalanobis metric (as in the coordinate
        # model's torus branch).
        #
        # Deliberately NO snap kd-tree. The coordinate model snaps its raw 2-D
        # prediction (``X @ W``, which can land far outside ``[0, period)``)
        # onto the nearest observed training point to keep it on the manifold
        # support. The embedding model's decoded prediction is ALWAYS a valid
        # torus coordinate in ``[0, period)``, so that projection is
        # unnecessary -- and, for the weak-signal regime this model exists to
        # detect, snapping a near-centroid prediction to discrete observed
        # points only adds discretisation scatter and can flip a real positive
        # ``r2_spatial`` to negative. ``_train_kdtree = None`` makes the
        # inherited :meth:`evaluate_metrics` and :meth:`predict` operate on the
        # raw decoded coordinates (their ``snap`` branch is skipped).
        y_train_np = np.asarray(y, dtype=np.float64)
        self.Y_train_ = y_train_np
        self._train_kdtree = None

        w_cov = sample_weight / (np.sum(sample_weight) + 1e-12)
        mu_train = circular_mean(y_train_np, metric='torus', period=self.period, weights=w_cov)
        diff = signed_diff(y_train_np, mu_train[None, :], metric='torus', period=self.period)
        cov = (w_cov[:, None] * diff).T @ diff
        self.train_mean_ = mu_train.astype(np.float64)
        self.train_cov_inv_ = np.linalg.pinv(cov)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray, snap: bool = True) -> np.ndarray:
        """
        Description
        -----------
        Predict 2-D torus coordinates. The raw linear prediction is the 4-D
        embedding ``X @ W + b``, decoded to ``(x, y)`` via ``atan2``. When
        ``snap`` is True the decoded prediction is projected onto the nearest
        observed training point (1-NN in the 4-D torus-embedding space, so the
        nearest-neighbour respects wraparound), constraining the output to the
        training manifold's support exactly as in the coordinate model.

        Parameters
        ----------
        X (np.ndarray)
            ``(n_samples, n_features*n_time_bins)`` design matrix.
        snap (bool)
            Whether to snap the decoded prediction to the nearest training
            point. Default True.

        Returns
        -------
        Y_pred (np.ndarray)
            ``(n_samples, 2)`` predicted coordinates (float64).
        """

        check_is_fitted(self, ['coef_', 'intercept_'])
        X = check_array(X)
        raw_embedding = np.dot(X, self.coef_) + self.intercept_
        decoded = self._decode(raw_embedding)

        if snap and getattr(self, '_train_kdtree', None) is not None:
            _, idx = self._train_kdtree.query(torus_embed(decoded, self.period), k=1)
            return self.Y_train_[idx]
        return decoded


def resolve_manifold_regressor_cls(metric: str) -> type:
    """
    Description
    -----------
    Select the manifold-regression estimator class appropriate for the
    requested metric geometry. On ``metric='torus'`` the wound-blind
    coordinate model :class:`SmoothBivariateRegression` is replaced by the
    convex closed-form :class:`SmoothTorusManifoldRegression`, which predicts
    the 4-D sin-cos torus embedding and therefore recovers behaviour ->
    position relationships that *wind* around the torus (the coordinate model
    returns false nulls on those -- see this module's docstring). For every
    other metric (``'euclidean'``, i.e. flat VAE / UMAP manifolds) the unchanged
    coordinate model is returned, so those runs remain byte-identical.

    Both classes expose an identical constructor signature, so the caller can
    instantiate the returned class with exactly the same keyword arguments; the
    iterative-optimiser arguments (``huber_delta`` / ``learning_rate`` /
    ``max_iter`` / ``tol`` / ``_use_lax_loop``) are accepted-but-unused by the
    torus estimator, whose objective is convex and solved in closed form.

    Parameters
    ----------
    metric (str)
        Manifold geometry tag, ``'euclidean'`` or ``'torus'``.

    Returns
    -------
    regressor_cls (type)
        :class:`SmoothTorusManifoldRegression` if ``metric == 'torus'``,
        otherwise :class:`SmoothBivariateRegression`.
    """

    if metric == 'torus':
        return SmoothTorusManifoldRegression
    return SmoothBivariateRegression
