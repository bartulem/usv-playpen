.. _Modeling:

Modeling
==================
This page explains how to use the **vocal-modeling pipelines** in
``usv_playpen.modeling``. Where the :ref:`Analyze` section produces the
per-session behavioral-feature tables, the modeling subsystem asks the
inverse question: *how well, and with what temporal structure, do those
behavioral kinematics predict a mouse's vocal behavior?*

Five prediction targets are supported, each with its own extraction
pipeline:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Pipeline
     - Predicts
     - ``Y(t)`` impulses
   * - ``VocalOnsetModelingPipeline``
     - whether a frame starts a vocal event — a bout or an individual USV (ultrasonic vocalization)
       (set by ``model_target_vocal_type``)
     - bout / USV onsets
   * - ``BoutParameterPipeline``
     - per-bout duration / complexity / intensity
     - bout starts
   * - ``VocalCategoryModelingPipeline``
     - whether a USV is a specific target category vs a pooled "other"
       (binomial, one-vs-rest)
     - per-USV starts
   * - ``MultinomialModelingPipeline``
     - the USV's vocal category across all categories jointly (multinomial)
     - per-USV starts
   * - ``ContinuousModelingPipeline``
     - per-USV 2-D acoustic-manifold (UMAP, Uniform Manifold Approximation and Projection) position
     - per-USV starts

Each target is fit first with **univariate** generalized additive / linear
models (one behavioral feature at a time, to rank predictors), then with a
**forward-stepwise model-selection** routine that greedily stacks features,
and finally — for the continuous manifold target — with a non-linear
**1-D ResNet (a residual-network convolutional neural network, CNN)**. Every stochastic step is seeded from
``model_params.random_seed`` so results are reproducible.

The whole subsystem is configured by a single settings file,
``_parameter_settings/modeling_settings.json``, and is driven either
interactively from the :ref:`modeling-notebook` (``modeling_analyses.ipynb``,
detailed in :doc:`Notebooks`) or, at cohort scale, from the SLURM job scheduler's dispatchers
described in :ref:`modeling-model-selection`.

.. note::

   The modeling pipelines are **not** exposed as a GUI tab. Run them from
   the notebook (interactive, single node) or the dispatchers (HPC). Every
   pipeline reads ``_parameter_settings/modeling_settings.json`` via
   ``modeling_settings_dict=None``; pass an explicit dict to override.

Modeling settings
-----------------
All knobs live in ``_parameter_settings/modeling_settings.json``, organised into
blocks. Each block is shown below as it appears in the file, followed by its keys.

.. note::

   The ``mixture_model_params`` block is not enumerated here: it holds fitted per-sex inter-syllable-interval mixture parameters (``male``/``female`` → ``means``/``sds``) written by the pipeline, not user-facing tuning knobs.

**io** — the cohort and where outputs go.

.. code-block:: json

    "io": {
        "session_list_file": "/mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_intact_partners_sessions_list.txt",
        "save_directory": "/mnt/falkner/Bartul/modeling",
        "csv_separator": ",",
        "camera_sampling_rate": 150
    }

* **session_list_file** — path to the text file that lists the cohort's sessions (one session root per line; see below).
* **save_directory** — directory where every modeling-input pickle, audit, and result is written.
* **csv_separator** — column delimiter of the per-session ``*_behavioral_features.csv`` files (``','``).
* **camera_sampling_rate** — camera frame rate in Hz (``150``); every pipeline uses it to convert ``filter_history`` seconds into a frame count.

The **session-list file** is the single source of truth for the cohort: a plain
text file with **one session-root directory per line**, each a ``Data``-tree
session (``<cup_root>/Data/<YYYYMMDD_HHMMSS>``). Every listed session is loaded
and pooled; blank lines are ignored. For example:

.. code-block:: text

    /mnt/falkner/Bartul/Data/20230119_155302
    /mnt/falkner/Bartul/Data/20230119_162529
    /mnt/falkner/Bartul/Data/20230119_172410
    /mnt/falkner/Bartul/Data/20230124_172125
    /mnt/falkner/Bartul/Data/20230207_141317

The **filename itself carries the cohort label**: ``derive_experimental_condition``
parses it (e.g. ``behavioral_courtship_intact_partners_sessions_list.txt`` →
``intact_partners``; other cohorts read ``male_mute_partner``,
``intact_partners_female``, …) and embeds that label into every output filename,
so artifacts from different cohorts never collide and each one is self-identifying.

**model_params** — the prediction target, history window, model engine,
forward-selection acceptance thresholds, and bout definition. The
cross-validation and held-out-test settings live in their own
``model_validation`` block below.

.. code-block:: json

    "model_params": {
        "filter_history": 4,
        "mixture_model_component_index": 0,
        "mixture_model_z_score": 2.58,
        "model_basis_function": "raised_cosine",
        "model_engine": "pygam",
        "model_predictor_mouse_index": 1,
        "model_target_vocal_type": "bout",
        "model_target_variable": "bout_durations",
        "selection_p_val": 0.01,
        "selection_effect_floor": 0.1,
        "selection_n_bootstrap": 1000,
        "selection_ci_level": 0.99,
        "usv_bout_time": 2,
        "usv_per_bout_floor": 2,
        "onset_target_category": null
    }

* **filter_history** — seconds of behavioral history preceding each event that feed the temporal filter (× ``camera_sampling_rate`` → frames).
* **mixture_model_component_index** / **mixture_model_z_score** — bout grouping: the fitted inter-syllable-interval mixture (``mixture_model_params``) is thresholded at ``mean + z·sd`` of the selected component (component ``0``, ``z = 2.58``) to decide where one bout ends and the next begins.
* **model_basis_function** — temporal-filter basis over the history window: ``'raised_cosine'`` / ``'bspline'`` / ``'laplacian_pyramid'`` (parameters in ``hyperparameters.basis_functions``), or ``'identity'`` (the raw per-frame history, no projection). Only relevant when ``model_engine = 'sklearn'`` — the ``'pygam'`` engine uses its own tensor-product splines instead.
* **model_engine** — univariate model backend: ``'pygam'`` (tensor-product-spline GAM, a generalized additive model) or ``'sklearn'`` (basis-projected linear).
* **model_predictor_mouse_index** — which mouse (``0`` / ``1``) is the **partner**; the **target** — the mouse whose vocal behavior is being predicted — is defined as the other one. Both mice's kinematics enter the predictor set.
* **model_target_vocal_type** — onset target mode, one of ``'bout'`` (clustered bout onsets, both positive and negative pre-event windows kept clean), ``'individual'`` (per-USV onsets), or ``'state'`` (the session is sampled on a regular ``filter_history``-spaced time grid and each sample labelled vocal / silent, with no clean-history requirement); used only by ``VocalOnsetModelingPipeline``.
* **model_target_variable** — for ``BoutParameterPipeline``, which per-bout quantity to regress: ``'bout_durations'`` (first-to-last-USV span, seconds), ``'mean_mask_complexity'`` (per-USV mean spectrogram-mask complexity), or ``'total_mask_complexity'`` (summed over the bout).
* **selection_p_val** — the significance level gating whether a candidate feature is admitted during forward-stepwise model selection (default ``0.01``); on the acoustic-manifold target it is the Benjamini–Hochberg FDR ``q`` used to screen candidates.
* **selection_effect_floor** / **selection_n_bootstrap** / **selection_ci_level** — the acoustic-manifold selection's **fold-grain acceptance gate** (``continuous_vocal_manifold_model_selection``): a feature is kept only when its per-fold paired score margin over the shuffle null (the macro von Mises log-likelihood on the torus, the wrap-aware distance correlation on euclidean) is consistently positive across CV folds. ``selection_effect_floor`` is the relative effect floor a screened feature must clear — a fraction of the top surviving driver's margin (default ``0.1`` = 10%); ``selection_n_bootstrap`` is the number of fold bootstrap resamples (default ``1000``); ``selection_ci_level`` is the bootstrap confidence level whose lower bound must exceed ``0`` for an anchor / forward step to be accepted (default ``0.99``). These three apply to the manifold gate only; the onset / category / bout-parameter selections use ``selection_p_val`` alone.
* **usv_bout_time** — duration (seconds) of the post-onset silence window that defines the **negative (No-USV) events** in ``'bout'`` mode: a candidate silent-epoch onset is kept only if no USV (from any source) starts within ``[t_onset, t_onset + usv_bout_time)`` after it.
* **usv_per_bout_floor** — the minimum number of USVs a positive bout must contain (``'bout'`` mode).
* **onset_target_category** — restrict positive onsets to a single USV category (``'individual'`` mode only); ``null`` pools all categories (see the single-category note under :ref:`Modeling input data <modeling-extract>`).

**model_validation** — the held-out test set and cross-validation splitting.

.. code-block:: json

    "model_validation": {
        "held_out_test_proportion": 0.1,
        "split_strategy": "session",
        "n_cv_folds": 10,
        "cv_validation_proportion": 0.1,
        "random_seed": 0,
        "spatial_cluster_num": 20,
        "session_split_initial_tolerance": 0.05,
        "session_split_max_attempts": 50000,
        "session_split_widen_step": 0.02,
        "session_split_widen_every": 1000
    }

* **held_out_test_proportion** — the fraction of sessions carved off **once** as a final held-out test set that is excluded from every CV fold and the whole feature-selection search, then scored only by the refit final model — an honest last-look estimate untouched by selection. ``0`` disables the carve-out (the entire dataset enters CV); default ``0.1``.
* **split_strategy** — the cross-validation scheme: ``'mixed'`` (stratified shuffle over the pooled data) or ``'session'`` (hold whole sessions out, so no session straddles train and test).
* **n_cv_folds** — the number of outer cross-validation folds each candidate feature is scored across.
* **cv_validation_proportion** — the fraction of data held out per CV fold as that fold's validation set.
* **random_seed** — seeds every stochastic step (splits, permutations, initialisation) for reproducibility.
* **spatial_cluster_num** — the number of spatial clusters used to build the spatial-CV folds for the continuous manifold target.
* **session_split_initial_tolerance** — the starting class-balance tolerance for the ``'session'`` strategy's held-out-set search, before it begins widening (default ``0.05``).
* **session_split_max_attempts** / **session_split_widen_step** / **session_split_widen_every** — tuning for the ``'session'`` strategy's search for balanced held-out session sets (max attempts, plus how much / how often the balance tolerance is relaxed).

**kinematic_features** — which behavioral predictors enter the feature zoo.

.. code-block:: json

    "kinematic_features": {
        "egocentric": ["speed", "neck_elevation", "allo_roll", "allo_pitch",
                       "ego_yaw", "back_pitch", "back_yaw", "tail_curvature"],
        "dyadic_pose": ["nose-nose", "allo_yaw-nose", "nose-allo_yaw",
                        "allo_pitch-nose", "nose-allo_pitch"],
        "dyadic_engagement": ["orofacial-sei"],
        "dyadic_pose_symmetric": false,
        "include_1st_derivatives": false,
        "include_2nd_derivatives": false,
        "abs_features": ["allo_roll", "allo_yaw-nose", "nose-allo_yaw",
                         "allo_yaw-TTI", "TTI-allo_yaw"],
        "smooth_abs_features": {"ego_yaw": 1.0, "back_yaw": 0.5}
    }

* **egocentric** — single-mouse posture / movement features of the predictor mouse.
* **dyadic_pose** — relative-pose features between the two mice (``<self>-<other>`` naming).
* **dyadic_engagement** — social-engagement features (e.g. ``orofacial-sei``).
* **dyadic_pose_symmetric** — if ``true``, include both ``A-B`` and ``B-A`` orientations of each dyadic-pose feature.
* **include_1st_derivatives** / **include_2nd_derivatives** — also add the velocity / acceleration of each feature.
* **abs_features** — feature suffixes folded to their plain absolute value ``|x|`` before pooled z-scoring (for signed angles whose sign is not behaviorally meaningful); read by every pipeline that z-scores across sessions.
* **smooth_abs_features** — a *smooth* magnitude fold, ``sqrt(x² + ε²)``, where the mapped value is the per-feature ``ε`` **in the feature's own units** (degrees for the angles), not a smoothing width in frames and not a Gaussian σ. It is the differentiable counterpart of ``abs_features``: identical to ``|x|`` away from zero, but rounded near it, which avoids the kink that a plain absolute value puts at ``x = 0``. A feature listed in both dicts takes this branch, since ``smooth_abs_features`` has priority (see ``zscore_different_sessions_together``).

**vocal_features** — which vocal predictors enter the zoo, and the acoustic-manifold definition.

.. code-block:: json

    "vocal_features": {
        "usv_predictor_type": "categories_rate",
        "usv_predictor_partner_only": true,
        "usv_predictor_smoothing_sd": 1,
        "usv_category_column_name": "qlvm_supercategory",
        "usv_noise_column": "qlvm_supercategory",
        "usv_noise_categories": [0],
        "usv_manifold_column_names": ["qlvm1", "qlvm2"],
        "usv_manifold_metric": "torus",
        "usv_manifold_period": 1.0,
        "usv_manifold_min_region_events": 20,
        "usv_manifold_selection_score": "macro",
        "usv_manifold_geodesic_metrics": {
            "compute": true,
            "grid_n_per_dim": 40,
            "graph_k": 8,
            "density_exponent": 1.0,
            "decoder_weights_npz_path": "/mnt/falkner/Bartul/spectrograms/qlvm/qmc_decoder_weights.npz"
        }
    }

* **usv_predictor_type** — which vocal-syntax predictor traces to build: ``'pooled_binary'`` (one pooled per-frame USV-event indicator), ``'pooled_rate'`` (one pooled USV-rate trace), ``'categories_rate'`` (one per-category USV-rate trace per ``usv_category_column_name`` category), or ``'all_rate'`` (the pooled rate plus the per-category rates). A falsy value builds no vocal predictors.
* **usv_predictor_partner_only** — if ``true``, ingest only the *partner's* USV signals as predictors (not the target mouse's own vocal history).
* **usv_predictor_smoothing_sd** — Gaussian σ (frames) applied to the USV-rate predictor traces.
* **usv_category_column_name** — the USV-catalog column defining categories (``'vae_supercategory'`` / ``'qlvm_supercategory'`` / ``'vae_category'`` / ``'qlvm_category'``).
* **usv_noise_column** / **usv_noise_categories** — the column and category indices treated as noise and excluded.
* **usv_manifold_column_names** — the two catalog columns giving the 2-D manifold position (the ``ContinuousModelingPipeline`` target).
* **usv_manifold_metric** — ``'euclidean'`` (plane) or ``'torus'`` (wrap-aware) distance on the manifold.
* **usv_manifold_period** — the wrap period for the ``'torus'`` metric.
* **usv_manifold_min_region_events** — the minimum number of labelled events an acoustic region (supercategory) must contain to enter the **macro** (region-balanced) von Mises average and the region-weighted MAE; sparser regions are dropped from those balanced statistics so a single under-sampled corner cannot dominate them (default ``20``). Ignored on euclidean and when no region labels are present.
* **usv_manifold_selection_score** — on the ``'torus'`` metric, which von Mises log-score the forward selection ranks on: ``'macro'`` (default) uses the region-balanced ``vm_logscore``, ``'micro'`` uses the event-weighted ``vm_logscore_pooled`` twin. Both are always logged per candidate, so this only changes which column drives the greedy ranking and the acceptance gate — the candidate pool, the region-reweighted fit, and every other reported metric are identical — making a macro-vs-micro selection comparison a one-key flip. Ignored on euclidean (which always ranks on ``dcor_xy``); an absent key resolves to ``'macro'``.
* **usv_manifold_geodesic_metrics** — the analysis-only *reference-map* geometry for the two torus **geodesic** prediction-error columns (``density_geodesic_mae``, ``pullback_geodesic_mae``), reported per fold alongside the flat-torus MAE on the ``'torus'`` metric (both ``NaN`` on euclidean). ``compute`` toggles the whole block; ``grid_n_per_dim`` sets the resolution of the regular torus grid the all-pairs geodesic distance matrices are precomputed on once (``40`` → a 40×40 node lattice, so per-event errors are cheap snap-to-grid look-ups); ``graph_k`` is the number of wrap-aware nearest neighbours per node in the k-NN graph the shortest paths run over; ``density_exponent`` is the inverse-aggregate-posterior-density exponent ``α`` weighting the density-ratio geodesic (``0`` recovers the flat graph metric, larger values push paths harder through dense regions); ``decoder_weights_npz_path`` is the frozen QLVM ConvTranspose decoder ``.npz`` whose Jacobian defines the pullback metric ``G = JᵀJ`` (an empty or unreadable path degrades ``pullback_geodesic_mae`` to ``NaN`` and the run proceeds).

**diagnostics** — the predictor-collinearity and predictor-timescale audits (rendered in :ref:`Predictor diagnostics <modeling-diagnostics>`).

.. code-block:: json

    "diagnostics": {
        "collinearity_audit": false,
        "collinearity_concern_threshold": 0.7,
        "collinearity_exclude_threshold": 0.85,
        "timescale_audit": false,
        "binary_decision_threshold": 0.5,
        "ece_n_bins": 10,
        "timescale_max_lag_seconds": 10.0,
        "timescale_n_shuffles": 1000,
        "timescale_shuffle_range": [20, 60],
        "timescale_signal_floor_seconds": 0.5,
        "timescale_signal_min_run_seconds": 0.2
    }

* **collinearity_audit** / **timescale_audit** — enable each audit during extraction.
* **collinearity_concern_threshold** / **collinearity_exclude_threshold** — Pearson ``|ρ|`` cutoffs for the collinearity audit: pairs above the concern threshold (default ``0.7``) are flagged for review, pairs above the exclude threshold (default ``0.85``) are treated as effectively redundant.
* **binary_decision_threshold** — probability cutoff for turning a predicted probability into a 0/1 label (a positive label is assigned when ``p >= threshold``); shared by every binary onset / category selector and pipeline.
* **ece_n_bins** — histogram bin count for the Expected Calibration Error diagnostic (default ``10``), shared by every calibration computation (the one adaptive site scales its own bin count to the sample size).
* **timescale_max_lag_seconds** — maximum lag examined for the ACF (autocorrelation function) / cross-correlation horizons.
* **timescale_n_shuffles** / **timescale_shuffle_range** — number of circular-shift surrogates and the ``(min, max)`` shift range (seconds) for the null envelope.
* **timescale_signal_floor_seconds** / **timescale_signal_min_run_seconds** — thresholds for calling a horizon significant (minimum above-null run length).

**hyperparameters** — per-engine model tuning, grouped into four sub-blocks:

* **deep_learning.cnn_continuous** — the 1-D ResNet for the continuous manifold target (architecture, optimiser, spatial-CV, saliency), consumed by ``NeuralContinuousCNNRunner``. The ``block_channels`` list sets the per-block channel widths (and therefore the network depth); ``warmup_fraction`` is the fraction of total steps spent warming the learning rate up before the cosine decay.
* **linear_models.manifold_regression** / **linear_models.multinomial_logistic** — the JAX smooth bivariate regression (continuous manifold position) and multinomial-logistic (vocal categories) models. The multinomial estimator additionally exposes a ``grad_clip_norm`` hyperparameter (global-norm gradient clip, default ``1.0``) that bounds each optimiser step.
* **classical.pygam** / **classical.logistic_regression** / **classical.ridge_regression** — the ``'pygam'`` / ``'sklearn'`` engine models (GAM splines; logistic-CV for binary targets; and, for the bout-parameter regression, an L2-penalized Gamma GLM whose penalty grid / CV come from the ``ridge_regression`` block — matching the pyGAM engine's Gamma likelihood so fit and Gamma-deviance score agree).
* **basis_functions.raised_cosine** / **bspline** / **laplacian_pyramid** — parameters for each ``model_basis_function`` choice.

The regularisation controls (shared by both ``linear_models`` sub-blocks) look like:

.. code-block:: json

    "linear_models": {
        "manifold_regression": {
            "lambda_smooth_fixed": 1.0,
            "l2_reg_fixed": 0.01,
            "smoothness_derivative_order": 1,
            "learning_rate": 0.005,
            "max_iter": 20000,
            "tune_regularization_bool": false,
            "tune_regularization_params": {
                "lambda_smooth_decades_each_side": 0,
                "l2_reg_decades_each_side": 4,
                "inner_cv_folds": 5,
                "inner_cv_use_one_se_rule": false,
                "inner_max_iter": 2500
            }
        }
    }

(The ``multinomial_logistic`` block mirrors this, plus an ``inner_cv_scoring_metric``
key — e.g. ``"auc"`` — that the manifold block does not carry, because a torus
manifold fit is always scored by the macro von Mises log-score ``vm_logscore``.)

* **lambda_smooth_fixed** / **l2_reg_fixed** — the fixed smoothness and L2 penalties. These are the operative values for **both** linear models, whose ``tune_regularization_bool`` now defaults to ``false`` (see the note below).
* **tune_regularization_bool** — if ``true``, run an inner-loop cross-validation to pick ``lambda_smooth`` / ``l2_reg`` per fold (parameters in ``tune_regularization_params``: the search width in decades, inner-CV folds, scoring metric, and the one-standard-error rule). When ``false``, ``tune_regularization_params`` is not read and the two ``*_fixed`` penalties are used directly.

.. note::

   **Regularisation is FIXED, not tuned, for the manifold model.** Empirically,
   on the behaviour→manifold-position problem the prediction score is *flat*
   across the smoothness penalty ``lambda_smooth`` over its entire range, and
   *flat* across the ridge penalty ``l2_reg`` below ~1 (only degrading when
   ``l2_reg`` is pushed much higher). The behavioural signal is real but weak and
   low-dimensional, so a wide range of shrinkage settings all land on the same
   answer — there is no interior optimum for an inner-CV to find. Tuning a flat
   surface makes the one-standard-error rule rail the penalty to a grid edge,
   which is exactly what produced an unstable feature count (a run selecting
   several features flipping to zero on a re-run). ``manifold_regression`` therefore
   ships with ``tune_regularization_bool = false``: ``lambda_smooth`` and ``l2_reg``
   are fixed at their ``*_fixed`` values. ``lambda_smooth`` changes only how smooth
   the interpretable filter *looks* (a constant filter scores the same as a
   recent-lag one), never the selection; the torus smoothness penalty uses
   reflective (Neumann) boundary rows so a higher fixed ``lambda_smooth`` cleans the
   filter's middle without its edges floating free.

   The multinomial-category model also ships with ``tune_regularization_bool =
   false``, but for a different, more cautionary reason. It is an *iterative* GLM
   (not the closed-form manifold solve) whose inner-CV was capped at
   ``inner_max_iter = 2500`` while a *converged* multinomial fit needs
   ~15000-18000 iterations — so its inner-CV was badly under-converged and its
   earlier tuning choices (which railed ``lambda_smooth`` to the grid ceiling) were
   a convergence artifact, not a real optimum. Tuning is therefore disabled here
   too, pending a proper fix: raise ``inner_max_iter`` and investigate why the fit
   needs so many iterations (the outer ``max_iter`` of 20000 is itself only
   marginally sufficient). Until then ``lambda_smooth`` / ``l2_reg`` are fixed at
   their ``*_fixed`` values.

.. note::

   **Frozen von Mises concentration.** On a torus manifold run, set
   ``vocal_features.freeze_selection_kappa = true`` to fit the von Mises
   concentration ``kappa`` once on the development set and reuse it for every
   score (baseline, fitted candidates, held-out, and the acceptance-gate
   bootstrap). The default (``false``) self-refits ``kappa`` per call, which floors
   the score on weak folds (the ``kappa → 0`` clamp) and corrupts the gate's paired
   per-fold margin; a single frozen ``kappa`` keeps every score on one dispersion
   scale so the margins are a proper scoring rule.

**glm_hmm** — the GLM-HMM over latent vocal states (see
:ref:`Latent vocal states <modeling-glm-hmm>` below).

.. code-block:: json

    "glm_hmm": {
        "emission_type": "manifold",
        "transition_mode": "static",
        "history_frames": 75,
        "cv_folds": 5,
        "n_states_min": 2,
        "n_states_max": 6,
        "n_em_iters": 100,
        "n_lbfgs": 500,
        "n_restarts": 5,
        "em_tol": 0.0001,
        "transition_pseudocount": 1.0,
        "lambda_smooth": 100.0,
        "input_driven_lambda_smooth": 0.05,
        "l2_reg": 0.1,
        "multinomial_target": "supercategory",
        "focal_gamma": 0.0,
        "multinomial_max_iter": 2000
    }

* **emission_type** — the per-state observation model: ``'manifold'`` (behaviour → 2-D acoustic-manifold position, von Mises density) or ``'multinomial'`` (behaviour → discrete USV category, categorical density).
* **transition_mode** — ``'static'`` fits the EM-based engine with a stationary ``K × K`` transition matrix; ``'input_driven'`` fits a direct-marginal engine whose transition *into* each time bin is a per-state GLM of the behavioural design (Calhoun/Pillow/Murthy), validated on the Coen-2014 fly to reproduce its state-selection without the state collapse the EM engine suffers on weakly-identified data. Input-driven is defined for both emissions (categorical → reference-coded ``InputDrivenGLMHMM``; manifold → torus product-von-Mises ``InputDrivenManifoldGLMHMM``, which requires a torus manifold).
* **history_frames** — number of most-recent behavioural-history frames kept per feature (the emission's temporal-filter length ``n_time_bins``).
* **cv_folds** — number of cross-validation folds over the development sessions used to select the state count.
* **n_states_min** / **n_states_max** — inclusive range of latent state counts ``K`` swept during selection.
* **n_em_iters** — maximum Baum-Welch EM iterations per restart (``'static'`` engine only).
* **n_lbfgs** — L-BFGS iterations per restart for the ``'input_driven'`` direct-marginal engines.
* **n_restarts** — number of random-initialisation restarts; the best (by held-out, else training, log-likelihood) is kept.
* **em_tol** — relative EM convergence tolerance (``'static'`` engine only).
* **transition_pseudocount** — Laplace pseudocount smoothing the static transition and initial-state expected counts (``'static'`` engine only).
* **lambda_smooth** / **l2_reg** — the temporal-smoothness and ridge penalties on each state's *static-engine* emission GLM.
* **input_driven_lambda_smooth** — the first-difference temporal-smoothness coefficient ``r`` for the *input-driven* engines (a separate key because its scale, ``~0.05``, differs from the static-engine ``lambda_smooth`` by orders of magnitude).
* **multinomial_target** — the categorical label column read as the target when ``emission_type='multinomial'`` (e.g. ``'supercategory'``); events whose label is NaN are dropped.
* **focal_gamma** — focal-loss focusing parameter of the static multinomial emission (``0.0`` = plain cross-entropy).
* **multinomial_max_iter** — optimiser iterations for the static multinomial emission's per-state classifier.

**behavioral_response** — the inverted analysis: does a partner's vocal trace predict a
*behavioural* variable (see :ref:`Behavioral response <modeling-behavioral-response>` below)?

.. code-block:: json

    "behavioral_response": {
        "response_mouse_index": 1,
        "response_feature": "speed",
        "history_seconds": 4.0,
        "target_window_seconds": 0.5,
        "target_gap_seconds": 0.0,
        "vocal_predictor_type": "pooled_rate",
        "vocal_smoothing_sd_frames": 1,
        "likelihood": "gamma",
        "n_shift_draws": 200,
        "shift_null_min_seconds": 20.0
    }

* **response_mouse_index** — which mouse's behaviour is predicted, by **absolute slot index** (``0`` is always the male, ``1`` always the female). Deliberately *not* the relative ``self.`` / ``other.`` role keys used elsewhere: those are defined against ``model_params.model_predictor_mouse_index`` and cannot be read on their own. **This is the only mouse index you set.** ``model_params.model_predictor_mouse_index`` decides whose *calls* are ingested and carries the opposite meaning on the same 0/1 axis; since the partner's calls are by definition the other animal's, it is **derived** as ``1 - response_mouse_index`` on a private copy of the settings (the caller's dict is never mutated, and the shipped value the five vocal pipelines read is untouched). So ``response_mouse_index: 1`` predicts the female from the male's calls, with nothing to keep in step by hand.
* **response_feature** — the behavioural feature used as the regression target (e.g. ``'speed'``). Read from the raw per-session feature table **before** column selection and z-scoring, so it need not belong to the predictor zoo and stays in native units — a Gamma likelihood needs a strictly positive response, and the log link, not standardisation, is what handles its scale.
* **history_seconds** — seconds of behavioural history preceding each anchor. Block-local rather than shared with ``model_params.filter_history``, mirroring how ``glm_hmm`` carries its own ``history_frames``. Because anchors are tiled non-overlapping, this doubles as the anchor stride: no two rows share a history sample, which keeps the rows close to independent.
* **target_window_seconds** — width of the forward window the response is averaged over. Averaging suppresses the frame-to-frame differentiation noise in a single 6.7 ms sample and breaks the near-determinism that would otherwise tie the target to the last frame of its own history.
* **target_gap_seconds** — delay between the end of the history and the start of the target window; ``0.0`` places the target immediately after the history. A non-zero gap is the leakage check: when a baseline predictor is nearly deterministic, misspecification leaves structured residual that a vocal regressor could absorb as a spurious increment.
* **vocal_predictor_type** — which vocal representation forms the block under test: ``'pooled_binary'`` (a single ``usv_event`` indicator), ``'pooled_rate'`` (a single smoothed ``usv_rate`` trace), ``'categories_rate'`` (one ``usv_cat_<n>`` trace per category) or ``'all_rate'`` (both). This is a **block-local override** of ``vocal_features.usv_predictor_type``, applied to a shallow copy so the shared block the five vocal pipelines read is left untouched. There is deliberately no second setting listing the column names: the block is *derived* from whichever traces this produces, because two keys that must agree can silently disagree, and a disagreement would quietly change what is being tested. The resolved partition is written into the artifact's ``analysis_specific`` metadata so the nested comparison never re-derives it. ``'pooled_rate'`` is the default because the first question is whether calling matters at all, not whether she responds differently to different call types — that is a strictly stronger claim, and splitting first would also mean several tests instead of one.
* **vocal_smoothing_sd_frames** — sigma of the Gaussian kernel the call train is convolved with, in **frames**, matching the loader's own frame-based sigma (kept as a float, so a fractional kernel stays expressible). A **block-local override** of ``vocal_features.usv_predictor_smoothing_sd``, for the same reason as the predictor type. It matters more here than elsewhere because this is the only pipeline where the vocal trace is the *block under test* rather than one predictor among many: at the default ``1`` the pooled rate is a near-impulse train that is overwhelmingly zero, so the GAM's value-axis splines fit a badly-conditioned distribution. Widening it trades temporal precision — of which there is spare, the lag axis carrying only ``n_splines_time`` knots across the whole history — for better-posed value splines.
* **likelihood** — ``'gamma'`` (Gamma GAM, log link, native units), ``'lognormal'`` (Gaussian GAM on ``log(y)``, i.e. a lognormal model), or ``'both'``. Defaults to ``'gamma'``: it reuses the machinery ``BoutParameterPipeline`` already validates and keeps ``y`` in native units. Use ``'both'`` to stress-test a result once there is one — it exactly doubles the compute, including the null. The two are scored on different scales and are deliberately **not** made comparable — back-transforming a log-scale fit yields a geometric mean rather than ``E[y]``, the exact fit/score mismatch ``BoutParameterPipeline`` was rewritten to remove — so each arm carries its own baseline and its own null, and only the *increments* are compared across arms.
* **n_shift_draws** — number of shifted refits forming the increment null. Only the full model is refit per draw (the baseline is identical across draws), so the cost is ``n_shift_draws × n_cv_folds`` fits. The smallest attainable p-value is ``1 / (n_shift_draws + 1)``.
* **shift_null_min_seconds** — minimum circular-shift offset. Offsets are drawn from ``[min, T − min]``: the floor keeps the shift past the slowest behavioural autocorrelation in the zoo (``nose-nose``, ~6–8 s), and the mirrored ceiling excludes near-full-length shifts, which wrap almost all the way round and are nearly the identity.

.. _modeling-extract:

Modeling input data
-------------------
Each pipeline converts the per-session loader output into a
**modeling-input pickle** — a nested ``{feature: {session: {event-window
arrays}}}`` dictionary with an embedded ``_input_metadata`` provenance
block — that every downstream runner consumes. The five extraction calls
differ only in *what gets predicted*:

.. code-block:: python

    from usv_playpen.modeling.modeling_vocal_onsets import VocalOnsetModelingPipeline
    from usv_playpen.modeling.modeling_vocal_bout_parameters import BoutParameterPipeline
    from usv_playpen.modeling.modeling_vocal_categories_binomial import (
        VocalCategoryModelingPipeline,
    )
    from usv_playpen.modeling.modeling_vocal_categories_multinomial import (
        MultinomialModelingPipeline,
    )
    from usv_playpen.modeling.modeling_usv_manifold_position import (
        ContinuousModelingPipeline,
    )

    # Vocal-event onsets (bout or individual USV, set by model_target_vocal_type)
    VocalOnsetModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_modeling_input_data()

    # Bout parameters (continuous regression: duration / complexity / intensity)
    BoutParameterPipeline(
        modeling_settings_dict=None
    ).extract_and_save_modeling_input_data()

    # One target USV category vs pooled "other" (binomial, one-vs-rest)
    VocalCategoryModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_category_input_data(target_category=6)

    # Vocal categories across all categories jointly (multinomial)
    MultinomialModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_multinomial_input_data()

    # Continuous manifold position (2-D UMAP regression)
    ContinuousModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_continuous_data()

Every extraction call writes three artifacts to ``io.save_directory``:

- the modeling-input pickle (filename embeds the cohort label and a
  timestamp);
- a paired ``*_collinearity.pkl`` predictor-collinearity audit;
- a paired ``*_timescales.pkl`` predictor-timescale audit.

The two audit artifacts are visualised in the next section before any model
is fit.

Every modeling-input pickle has the same two-part skeleton — one entry per
predictor, each holding per-session event arrays, plus one shared metadata block.
Concretely, part of a ``VocalOnsetModelingPipeline`` pickle:

.. code-block:: text

    {
        "speed": {                                    # a feature-zoo predictor
            "20230119_155302": {                      # a cohort session
                "usv_feature_arr":    <array (41, 600)>,     # 41 positive events x 600 history frames
                "no_usv_feature_arr": <array (380, 600)>     # 380 negative events x 600 frames
            },
            "20230119_162529": {"usv_feature_arr": "...", "no_usv_feature_arr": "..."}
        },
        "nose-nose": {
            "20230119_155302": {"usv_feature_arr": "...", "no_usv_feature_arr": "..."}
        },
        "_input_metadata": { "...": "..." }           # shared provenance (below)
    }

* **top-level keys** (``speed``, ``nose-nose``, …) — one per behavioral / vocal predictor that survived the audits (the *feature zoo*).
* **second-level keys** (``20230119_155302``, …) — under each feature, one per cohort session.
* **innermost dict** — the event-windowed arrays for that feature in that session, each of shape ``(n_events, filter_history_frames)`` (here ``600 = filter_history 4 s × camera_sampling_rate 150``); **this is the only part that differs between pipelines** (see "Individual" below).
* **``_input_metadata``** — a single provenance block, identical in structure across all pipelines.

**Shared — the ``_input_metadata`` block.** Every pickle carries the same
provenance, for example:

.. code-block:: json

    "_input_metadata": {
        "experimental_condition": "intact_partners",
        "session_ids": ["20230119_155302", "..."],
        "n_events_per_session": {"20230119_155302": {"usv": 41, "no_usv": 380}},
        "predictor_idx": 1, "predictor_mouse_sex": "female",
        "target_idx": 0, "target_mouse_sex": "male",
        "feature_zoo_full": ["speed", "..."], "feature_zoo_kept": ["speed", "..."],
        "usv_predictor_type": "categories_rate", "usv_predictor_partner_only": true,
        "filter_history_seconds": 4, "filter_history_frames": 600,
        "ibi_thresholds": {"male": 0.42, "female": 0.55},
        "analysis_specific": { "...": "..." },
        "git_commit": "...", "settings_sha256": "...", "created_utc": "...", "package_version": "..."
    }

* **cohort / scope** — ``experimental_condition`` (the cohort label), ``session_ids``, and ``n_events_per_session`` (how many positive / negative events each session contributed).
* **mouse roles** — ``predictor_idx`` / ``target_idx`` and their sexes.
* **behavioral features** — ``feature_zoo_full`` (everything requested) vs ``feature_zoo_kept`` (what survived the collinearity / timescale audits).
* **vocal-input shape** — ``usv_predictor_type``, ``usv_predictor_partner_only``, ``usv_predictor_smoothing_sd``.
* **temporal frame** — ``filter_history_seconds`` / ``filter_history_frames`` (the history window), and ``ibi_thresholds`` (the per-sex bout-gap thresholds derived from the mixture model).
* **analysis_specific** — the per-pipeline knobs (differs by pipeline; listed just below).
* **run provenance** — ``git_commit`` / ``git_dirty``, ``settings_sha256``, ``created_utc``, ``package_version``, so any result traces back to exact code and settings.

**Individual — the per-session event arrays.** The innermost dict holds the
event-windowed predictors and the ``Y(t)`` each pipeline predicts. Its keys, by
pipeline:

.. code-block:: text

    VocalOnsetModelingPipeline      ->  { "usv_feature_arr", "no_usv_feature_arr" }
    BoutParameterPipeline           ->  { "X", "y", "groups" }
    VocalCategoryModelingPipeline   ->  { "target_feature_arr", "other_feature_arr" }
    MultinomialModelingPipeline     ->  { "X", "y" }
    ContinuousModelingPipeline      ->  { "X", "Y", "w", ["supercategory"], ["category"] }

* **VocalOnsetModelingPipeline** — ``usv_feature_arr`` = positive onset windows, ``no_usv_feature_arr`` = silent-epoch (negative) windows. ``analysis_specific``: ``model_target_vocal_type``, ``usv_bout_time``, ``usv_per_bout_floor``.
* **BoutParameterPipeline** — ``X`` = the bout-onset feature windows, ``y`` = the per-bout regression target (selected by ``model_target_variable``), ``groups`` = the session grouping. ``analysis_specific``: ``target_variable``.
* **VocalCategoryModelingPipeline** — ``target_feature_arr`` = windows for the chosen target category, ``other_feature_arr`` = windows for the pooled "other". ``analysis_specific``: ``target_category``.
* **MultinomialModelingPipeline** — ``X`` = per-USV feature windows, ``y`` = each USV's category label. ``analysis_specific``: ``categories_kept``, ``class_counts``.
* **ContinuousModelingPipeline** — ``X`` = per-USV feature windows, ``Y`` = the 2-D acoustic-manifold position target, ``w`` = inverse-density KDE weights (always present); ``supercategory`` and/or ``category`` (each USV's manifold cluster label) are added when those columns are configured. ``analysis_specific``: ``usv_manifold_column_names``.

.. note::

   **Modeling onsets for a single USV category.** By default
   ``VocalOnsetModelingPipeline`` pools *all* of the target mouse's USVs when
   it derives positive onset events. When overall vocal output is too sparse
   for bout-onset modeling but one category is plentiful — e.g. female
   broadband vocalizations (BBVs) — you can restrict the positive onsets to a
   single category by setting two knobs in ``model_params``:

   - ``model_target_vocal_type = 'individual'`` — each qualifying USV onset
     (rather than a clustered bout onset) becomes a positive event;
   - ``onset_target_category = <int>`` — the category index to keep (e.g.
     ``6`` for BBVs). The *column* this index refers to is the existing
     ``vocal_features.usv_category_column_name``, so any of
     ``vae_supercategory`` / ``qlvm_supercategory`` / ``vae_category`` /
     ``qlvm_category`` can be targeted. Leave it ``null`` (default) to pool all
     categories exactly as before.

   Only the *positive* onsets are filtered: the behavioral / vocal predictors
   and the silent-epoch (No-USV) negative reference are still computed over
   **all** of the mouse's USVs, so the category choice changes only *which*
   onsets count as events — never the predictors or the negatives. The filter
   is honoured in ``'individual'`` mode only; in ``'bout'`` mode it is
   ignored, because the mixture-model inter-syllable-interval threshold used
   for bout grouping is calibrated on the all-USV interval distribution and
   would mis-group a category-sparsified sequence (a warning is printed if the
   setting is combined with a non-individual mode). When active, the chosen
   category column and index are embedded in the ``analysis_tag`` (e.g.
   ``individual_cat_vae_supercategory_6``) and ``_input_metadata``, so VAE (variational autoencoder)-vs-QLVM (in-house quasi-Monte Carlo latent variable model)
   and category-vs-supercategory are unambiguous in every downstream artifact
   name and provenance block.

.. _modeling-diagnostics:

Predictor diagnostics
---------------------
Before committing to model fitting, inspect how the candidate predictors
relate to each other and to the event train. The three diagnostic plots
share feature ordering and per-group colour so a feature can be
cross-referenced by row position and hue across all three:

.. code-block:: python

    from usv_playpen.os_utils import configure_path
    from usv_playpen.visualizations.modeling_plots import (
        plot_timescale_audit_per_feature,
        plot_timescale_audit,
        plot_collinearity_audit,
    )

    timescale_pkl = configure_path("/mnt/falkner/Bartul/modeling/..._timescales.pkl")
    collinearity_pkl = timescale_pkl.replace("_timescales.pkl", "_collinearity.pkl")

    # Per-feature ACF + cross-correlation horizons (run first; ground truth).
    plot_timescale_audit_per_feature(timescale_pkl, save_plot_bool=False)
    # Cohort timescale summary (horizontal bars of the per-feature horizons).
    plot_timescale_audit(timescale_pkl, save_plot_bool=False)
    # Spearman-rho heatmap (left) + variance-inflation-factor bars (right).
    plot_collinearity_audit(collinearity_pkl, save_plot_bool=False)

``plot_timescale_audit_per_feature`` answers, for each predictor: how long
its autocorrelation stays above a circular-shift null (the ACF horizon),
and at what lag its cross-correlation with the event train ``Y(t)`` leaves
that null envelope (the cross-correlation horizon). ``plot_collinearity_audit``
flags predictor pairs whose ``|rho|`` (Spearman correlation) crosses the audit's concern / exclude
thresholds and reports per-feature VIFs.

**The audit artifacts.** Extraction writes both pickles alongside the
modeling-input pickle. Each is a flat dict of **feature-indexed arrays** (not the
``{feature: {session: …}}`` nesting), plus the same ``_input_metadata`` block.

``*_collinearity.pkl`` — how predictors relate to each other and to ``Y(t)``:

.. code-block:: text

    {
        "features":     ["speed", "nose-nose", ...],    # F feature names (row / column order)
        "spearman_rho": <array (F, F)>,                 # feature x feature Spearman correlation (signed, [-1, 1])
        "pearson_rho":  <array (F, F)>,                 # feature x feature Pearson correlation (signed, [-1, 1])
        "vif":          <array (F,)>,                   # per-feature variance-inflation factor
        "condition_number": <float>,                    # design-matrix condition number
        "flagged_pairs": [("ego_yaw", "back_yaw", 0.83, "concern"), ...],  # (feat_i, feat_j, rho, tier) tuples (rho signed); tier in {concern, exclude}
        "concern_threshold": 0.7, "exclude_threshold": 0.85,
        "n_events": <int>, "source_pickle": "...", "created": "...",
        "_input_metadata": {"...": "..."}
    }

* **spearman_rho** / **pearson_rho** / **vif** — the pairwise ``Spearman`` and Pearson (signed) correlation matrices and per-feature VIFs (the ``plot_collinearity_audit`` panels).
* **flagged_pairs** — the feature pairs whose ``|rho|`` crosses a threshold, each as ``(feat_i, feat_j, rho, tier)`` (signed ``rho``) with ``tier`` in ``{"concern", "exclude"}``.
* **condition_number** / **n_events** — the design-matrix condition number and the number of events the audit ran on.

``*_timescales.pkl`` — how far in time each predictor carries information:

.. code-block:: text

    {
        "features":            ["speed", "..."],        # F feature names (axis-0 / rows below)
        "acf_lags_seconds":    "<array (L,)>",          # ACF lag axis
        "acf_median":          "<array (F, L)>",        # per-feature median autocorrelation (+ acf_p25 / acf_p75)
        "acf_null_mean":       "<array (F, L)>",        # circular-shift null envelope (+ acf_null_p0_5 / _p99_5)
        "tau_acf_1_over_e":    "<array (F,)>",          # per-feature ACF horizons (+ tau_acf_0_2, tau_acf_integrated)
        "signal_lags_seconds": "<array (M,)>",          # cross-correlation lag axis
        "rho_signal":          "<array (F, M)>",        # feature x Y(t) cross-correlation (+ per-session mean/sem and null envelope)
        "ibi_thresholds": {"...": "..."}, "configured_filter_history": 4,
        "_input_metadata": {"...": "..."}
    }

* **acf_median** (rows = features, columns = lags) vs **acf_null_*** — each feature's autocorrelation against a circular-shift null; the **ACF horizon** (``tau_acf_*``) is how long it stays above that null.
* **rho_signal** / **signal_lags_seconds** — each feature's cross-correlation with ``Y(t)`` across lags (also stored per-session as ``rho_signal_per_session_mean`` / ``_sem`` against a ``rho_signal_null_mean`` envelope); the **cross-correlation horizon** is the lag at which it leaves the null envelope.
* **ibi_thresholds** / **configured_filter_history** — the bout-gap thresholds and history window recorded for context.

Univariate modeling
-------------------
Univariate fits (one behavioral feature at a time) produce the ranking that
seeds model selection. At cohort scale they are dispatched as a SLURM job
array (one feature per task) via ``main_univariate_dispatcher``, writing one
per-feature pickle each. The ranking is visualised with ``plot_feature_ranking``
(single target) or ``plot_univariate_multinomial_performance`` (multinomial), and
the fitted temporal filters with ``plot_significant_filters``.

After the array finishes, merge the per-feature pickles into a single artifact.
``consolidate_univariate`` asserts metadata equality across every pickle
(guarding against stray files from a different run), hoists the agreed
``_input_metadata`` / ``_run_metadata`` / ``_consolidation_metadata`` blocks to the
top, and emits a self-describing filename:

.. code-block:: python

    from usv_playpen.modeling.consolidate_univariate_results import (
        consolidate as consolidate_univariate,
    )

    consolidate_univariate(
        input_dir="/mnt/falkner/Bartul/modeling/<univariate_dir>",
        delete_individuals_after=False,
    )

The consolidated filename is self-describing, e.g.
``univariate_onsets_bout_male_mute_partner_<ts>.pkl``. Set
``delete_individuals_after=True`` only once you have verified the consolidated
artifact is correct.

The consolidated pickle is keyed by feature, with the hoisted metadata blocks
alongside:

.. code-block:: text

    {
        "speed": {                                   # one key per feature-zoo predictor
            "actual": {"filter_shapes": "<array (n_folds, T)>", "ll": "<array (n_folds,)>", "...": "..."},
            "null":   {"...": "..."},                # same keys — the label-shuffle permutation null
            "split_sizes": {"train": "<array (n_folds,)>", "test": "<array (n_folds,)>"}
        },
        "nose-nose": {"actual": {"...": "..."}, "null": {"...": "..."}, "split_sizes": {"...": "..."}},
        "_input_metadata": {"...": "..."},           # cohort / features / temporal frame (as above)
        "_run_metadata": {"...": "..."},             # model_engine, basis_function, null_strategy, folds, seed
        "_consolidation_metadata": {"...": "..."}     # what was merged, when, from where
    }

* **top-level keys** — one per feature, plus the three ``_*_metadata`` blocks. Each feature holds an ``actual`` and a ``null`` branch of identical shape, plus ``split_sizes`` (per-fold train / test sizes).
* **``actual`` / ``null``** — the per-fold results for the real fit and its label-shuffle permutation null (:ref:`the significance baseline <modeling-model-selection>`). Each holds ``filter_shapes`` of shape ``(n_folds, filter_history_frames)`` (the reconstructed temporal filters) and the per-fold metric arrays ``(n_folds,)``: ``ll`` (log-loss, the significance gate), ``deviance_explained`` (McFadden's D²), ``auc`` (area under the ROC curve), ``score`` (balanced accuracy), ``f1`` (F1 score), ``recall``, ``brier`` (Brier score), ``ece`` (expected calibration error), ``mcc`` (Matthews correlation coefficient), ``confusion_matrix``, and the optimiser diagnostics ``n_iter`` / ``converged`` / ``fit_time`` (plus, for the ``'sklearn'`` engine, ``coefs_projected`` / ``optimal_C``).
* **``_run_metadata``** — how the fits ran: ``model_engine``, ``basis_function``, ``null_strategy``, ``n_outer_folds``, ``split_strategy``, ``random_seed_outer``, the engine hyperparameters, and git / settings provenance. **``_consolidation_metadata``** records the merge audit (how many per-feature files, when, and their paths).
* **multinomial / continuous targets** — the per-fold metrics instead live under an ``actual.folds.metrics`` sub-dict (with ``y_true`` / ``y_pred`` / ``classes`` alongside), rather than as flat top-level arrays.

.. _modeling-model-selection:

Model selection
---------------
Greedy forward-stepwise selection stacks features on top of the univariate
ranking, adding at each step the feature whose contribution most improves
the held-out score, subject to the **fold-grain paired-margin acceptance gate**
described in the note below (a step is kept only when its per-fold score-margin
improvement over the shuffle null has a ``selection_ci_level`` bootstrap CI whose
lower bound exceeds ``0``). ``use_top_rank_as_anchor=True`` seeds step 0 with the
top univariate feature; ``p_val`` is the per-step acceptance threshold.

.. note::

   **Significance baseline for the discrete targets (vocal onsets, binomial
   USV categories).** Every univariate fit is evaluated against a
   *label-shuffle permutation null*: the same estimator is re-fit on a copy of
   the **training** labels permuted within each fold — breaking the
   behaviour→vocalization association while preserving the marginal event rate
   — and then scored against the real (unpermuted) **test** labels, seeded
   reproducibly per fold from ``random_seed``. This replaced the earlier
   pseudo-class controls (resampled No-Bout / Other-USV baselines), which
   tested a weaker question. A feature is admitted to model selection only if
   its mean held-out **log-loss** beats a Bonferroni (multiple-comparison) corrected lower percentile
   of the null log-loss distribution (``q = p_val / n_features``). Log-loss is
   the gate because it is the only *proper* scoring rule among the reported
   metrics: under the null the fitted probabilities sit near chance with a tiny
   feature-monotone residual, so rank / threshold statistics (AUC,
   balanced-accuracy) amplify that residual into spurious ~0 / 1 values and
   must **not** decide significance — they are retained for display only.
   Each fit also reports ``deviance_explained`` (McFadden's D²,
   ``1 − LL / ln 2``, where ``ln 2`` is the chance log-loss of the
   balanced-trained intercept) as a fold- and target-comparable effect size.
   Under H0 the actual and null log-loss coincide, so the screen does not
   inflate false positives.

.. note::

   **Significance and acceptance for the acoustic-manifold target.** The
   continuous-manifold selection scores each fold's out-of-sample predictions
   with a geometry-specific measure — the **macro von Mises log-likelihood**
   (``vm_logscore``, equal-weighting acoustic regions so a feature that rescues a
   rare, badly-predicted region is rewarded) on the torus, and the wrap-aware
   distance correlation (``dcor_xy``) on euclidean — and decides acceptance at
   the **fold** grain. Fold-grain (rather than session-grain) is used because the
   macro average is only stable when every acoustic region is well populated:
   pooling all sessions within a fold guarantees that, whereas scoring one
   session at a time starves most regions below the per-region floor and
   collapses the macro signal. A feature is screened in only when its per-fold
   paired margin over the shuffle null is consistently positive across folds —
   Benjamini–Hochberg-FDR-controlled at ``q = selection_p_val`` and clearing the
   relative ``selection_effect_floor`` (a fraction of the top surviving driver's
   margin) — and an anchor / forward step is admitted only when its per-fold
   improvement over the current model has a ``selection_ci_level`` fold-bootstrap
   CI (``selection_n_bootstrap`` resamples) whose lower bound exceeds ``0``. This
   replaced an earlier fold-level Wilcoxon / one-standard-error gate.

.. note::

   **What the acoustic-manifold selection reports per fold.** The macro von
   Mises log-likelihood is the *objective*, but every candidate at every fold
   also carries a full metric bundle for post-hoc comparison, so the selected
   trajectory can be re-read under a different lens without re-running the
   (day-scale) search:

   * **Two von Mises scores** — the objective ``vm_logscore`` (**macro**: equal
     weight per acoustic region) and its ``vm_logscore_pooled`` (**micro**: equal
     weight per *event*) twin. Both use the identical per-event densities and a
     single internally-fit concentration ``kappa`` and differ only in the
     averaging, so logging both lets a macro-driven selection be checked, at
     every step, against the feature a micro (event-weighted) objective would
     have preferred — at no extra model fit (it only re-averages densities
     already computed). ``vm_logscore_pooled`` is *not* a selection objective; it
     is reported for that comparison only.
   * **Three torus distance geometries** for the (prediction, truth) residual, in
     the spirit of the QLVM paper's Appendix C: the **flat-torus** wrap-aware MAE
     (``euclidean_mae``, the intrinsic metric of the periodic square), the
     **density-ratio graph geodesic** MAE (``density_geodesic_mae``, shortest
     paths on a wrap-aware k-NN grid whose edges are the flat length reweighted
     by inverse aggregate-posterior density, so paths route through dense
     corridors and pay to cross low-density valleys), and the
     **decoder-Jacobian pullback geodesic** MAE (``pullback_geodesic_mae``, the
     Arvanitidis pullback ``G = JᵀJ`` of the frozen QLVM decoder, measuring
     distance in decoded-spectrogram change rather than in raw latent
     coordinates). The two geodesic columns are ``NaN`` on euclidean manifolds
     and when their reference map is disabled or the decoder is unavailable;
     their absolute magnitudes are reweighted path-lengths (arbitrary scale), so
     they are read as a **Δ-vs-baseline**, exactly like the flat MAE.
   * **Region-weighted vs global MAE** — ``euclidean_mae_weighted`` (the macro,
     region-balanced distance twin of ``vm_logscore``) alongside the global
     ``euclidean_mae``, plus ``mahalanobis_mae``, ``r2_spatial``, the per-axis
     MAEs and the pearson / spearman correlations. The fit itself is additionally
     **region-reweighted** (inverse-region-frequency sample weights), so a few
     dominant acoustic regions do not drown the rare ones during estimation.

Run on a single node from the notebook:

.. code-block:: python

    from usv_playpen.modeling.model_selection import (
        vocal_onset_model_selection,
        vocal_category_model_selection,
    )

    vocal_onset_model_selection(
        univariate_results_path="/mnt/falkner/Bartul/modeling/univariate_<...>.pkl",
        input_data_path="/mnt/falkner/Bartul/modeling/modeling_<...>_bout_hist4s.pkl",
        output_directory="/mnt/falkner/Bartul/modeling/model_selection_results/<...>",
        use_top_rank_as_anchor=True,
        p_val=0.01,
    )

Or, for cohort-scale runs, from the HPC dispatchers (the right entry point
for the inner-loop parallelism):

.. code-block:: bash

    python -m usv_playpen.modeling.main_univariate_dispatcher
    python -m usv_playpen.modeling.main_model_selection_dispatcher

The dispatchers read the same ``modeling_settings.json`` and write one
per-feature / per-step pickle each. Consolidate the model-selection steps with
``consolidate_model_selection`` (the same metadata-equality guard and metadata
hoisting as the univariate consolidator above):

.. code-block:: python

    from usv_playpen.modeling.consolidate_model_selection_results import (
        consolidate as consolidate_model_selection,
    )

    consolidate_model_selection(
        input_dir="/mnt/falkner/Bartul/modeling/<selection_dir>", move_to_steps_subdir=False
    )

The consolidated filename is self-describing, e.g.
``model_selection_final_male_intact_partners_onsets_bout_mixed_<ts>.pkl``.

The consolidated pickle is an ordered list of forward-selection steps plus the
hoisted metadata blocks:

.. code-block:: text

    {
        "steps": [
            {                                        # one entry per step (0, 1, 2, ...)
                "step_idx": 0,
                "current_features": ["speed"],       # features already selected before this step
                "baseline_score": 0.68,              # best score of current_features (chance floor at step 0)
                "baseline_folds": "<array (n_folds,)>",  # their PER-FOLD scores, for the paired 1SE test
                "selected_feature": "nose-nose",     # feature accepted this step (None -> final, rejected step)
                "candidates_summary": {              # every feature tested this step -> its per-fold metrics
                    "nose-nose":  {"ll": "<array (n_folds,)>", "auc": "...", "mean_ll": "...", "se_ll": "..."},
                    "back_pitch": {"...": "..."}
                }
            }
        ],
        "_input_metadata": {"...": "..."},
        "_univariate_metadata": {"...": "..."},      # the upstream univariate provenance
        "_run_metadata": {"...": "..."},
        "_consolidation_metadata": {"...": "..."}
    }

* **``steps``** — an ordered list, one entry per forward-selection step. ``step_idx`` is the iteration, ``current_features`` are those already chosen, ``baseline_score`` is their held-out score (the chance floor at step 0), ``baseline_folds`` is the same model's PER-FOLD scores, and ``selected_feature`` is the feature accepted this step (``None`` marks the final, rejected step). ``baseline_folds`` is what the 1SE acceptance test pairs a candidate against: the incumbent and the candidate are nested and scored on the same folds, so the fold-difficulty term they share cancels in the per-fold difference but dominates either score on its own. Checkpoints written before this key existed resume against ``baseline_score`` as a constant, which reproduces the unpaired comparison that wrote them. For the multinomial and manifold selectors, step 0's ``selected_feature`` is the sentinel ``'null_model_free'`` baseline.
* **``candidates_summary``** — under each step, every candidate feature tested that step mapped to its per-fold metrics. For the discrete / regression targets these are flat per-fold arrays (``ll``, ``auc``, ``score``, ``f1``, ``brier``, ``ece``, ``mcc``, ``confusion_matrix``, ``n_iter`` / ``converged`` / ``fit_time``) plus aggregate ``mean_ll`` / ``se_ll``; the multinomial and manifold selectors nest these under a ``folds.metrics`` sub-dict (with ``y_true`` / ``y_pred`` / ``y_probs`` / ``classes`` and the per-fold ``selected_lambda_smooth`` / ``selected_l2_reg`` regularisation choices — equal to the fixed ``*_fixed`` penalties when ``tune_regularization_bool`` is ``false``).
* **last accepted step** — additionally carries ``final_model_features`` (the cumulative selected set) and ``filter_shapes`` (the per-fold refit filters) of the published model.
* **metadata blocks** — ``_input_metadata`` and ``_univariate_metadata`` carry the upstream extraction / univariate provenance, ``_run_metadata`` the selection config, and ``_consolidation_metadata`` the merge audit.

Visualise the trajectory with ``plot_model_selection_results`` (binary / regression),
``plot_multinomial_selection_trajectory`` (multinomial), or
``plot_manifold_selection_trajectory`` (continuous manifold): each reads the
consolidated ``model_selection_final_*.pkl`` and shows the per-step held-out score
gain and the retained-feature filters.

For the acoustic-manifold model the converged filters have a single dedicated
view, ``plot_manifold_filter_atlas`` (torus runs only), reading the same
consolidated artifact — a three-panel figure. The **vocal-space atlas** panel
tiles the torus with canonical USVs decoded through the frozen QLVM decoder
(inferno on black, each icon peak-normalised, the supercategory watershed
boundaries overlaid in white) as the key for the fields. The **filter magnitude**
panel draws the per-feature temporal magnitude ``|W(t)|`` over the history window
(one line per feature, coloured by behavioural category with opacity separating
features that share a category; averaged into display bins so the medium-scale
envelope reads cleanly). The **affinity filmstrips** panel samples each feature's
filter at ``n_time_slices`` instants from ``-history_window_sec`` to onset and
decodes each into the signed ``e(theta).W`` field over the torus (red = a +1 SD
increase in the feature just before onset drives the predicted vocalization toward
that region, blue = away), on a shared diverging scale. The QLVM decoder and
supercategory-boundary ``.npz`` paths default from ``modeling_settings.json``
(``usv_manifold_geodesic_metrics.decoder_weights_npz_path``, with
``arrays_coarse.npz`` taken from the same directory); colours come from
``visualizations_settings.json`` (``sequential_cmap`` / ``diverging_cmap``), and the temporal
filter's smoothness is governed by the per-observation ``lambda_smooth`` prior (see
the note above).

CNN modeling
------------
CNN modeling trains a non-linear 1-D ResNet to predict a USV's continuous
2-D acoustic-manifold position from a window of behavioral kinematics — a
flexible non-linear complement to the interpretable linear pipeline. The
runner loads the modeling-input pickle, stacks the per-feature ``(N, T)``
matrices into the ``(N, F, T)`` tensor the 1-D ResNet consumes, trains over
the spatial-CV folds (tri-strategy: actual / null / null-model-free), and
writes a ``cnn_*_predictions_*.pkl`` artifact:

.. code-block:: python

    from usv_playpen.modeling.jax_neural_network_cnn import NeuralContinuousCNNRunner

    runner = NeuralContinuousCNNRunner(modeling_settings=None)
    data_blocks = runner.load_multivariate_data_blocks(
        pkl_path="/mnt/falkner/Bartul/modeling/modeling_manifold_<...>.pkl"
    )
    runner.run_cnn_training(data_blocks=data_blocks)

The trained-network diagnostics (permutation test, feature importance,
spatial-precision grid, error landscape, regional saliency) are rendered by
``DeepResultsVisualizer`` from the same prediction artifact.

The prediction artifact is organised by fold and by strategy, with the
diagnostics computed once across folds:

.. code-block:: text

    {
        "metadata": {                                # features_list, hyperparameters, manifold config
            "features_list": ["speed", "..."], "manifold_metric": "euclidean", "n_time_bins": 600
        },
        "cross_validation": [                        # one entry per spatial-CV fold
            {
                "Y_true":                 "<array (N, 2)>",   # true manifold positions (this fold's test set)
                "Y_pred_actual":          "<array (N, 2)>",   # the real model's predictions
                "Y_pred_null":            "<array (N, 2)>",   # label-shuffle null model
                "Y_pred_null_model_free": "<array (N, 2)>",   # empirical-density baseline
                "error_actual": 0.14, "error_null": 0.31, "error_null_model_free": 0.33
            }
        ],
        "feature_importance": {                      # permutation importance over the best fold
            "means": {"...": "..."}, "stds": {"...": "..."}, "snrs": {"...": "..."},
            "ranked_features": ["nose-nose", "..."], "best_fold_idx": 3
        },
        "saliency_maps": {"supercategory_0": {"contrastive_saliency": "<array>", "centroid": "...", "radius": "..."}},
        "cluster_geometry": {"...": "..."}           # optional — cluster centroids / radii
    }

* **``metadata``** — the run configuration: ``features_list`` (the ``F`` predictor order), the ``hyperparameters`` block, ``manifold_metric`` / ``manifold_period`` / ``output_encoding``, ``n_time_bins``, ``split_strategy``, and the source-pickle path.
* **``cross_validation``** — a list, one dict per spatial-CV fold. Each holds the fold's test-set ground truth ``Y_true`` ``(N, 2)`` and the three strategies' predictions ``Y_pred_actual`` / ``Y_pred_null`` / ``Y_pred_null_model_free`` (all ``(N, 2)``), plus the scalar wrap-aware ``error_actual`` / ``error_null`` / ``error_null_model_free`` that feed the skill-score and permutation test.
* **``feature_importance``** — permutation importance evaluated on ``best_fold_idx``: per-feature ``means`` / ``stds`` / ``snrs`` (mean Δerror, its spread, and the signal-to-noise ratio), ``ranked_features`` (sorted), and ``significant_features`` (SNR-thresholded).
* **``saliency_maps``** (optional) — one entry per acoustic cluster (keyed ``<segmentation>_<label>``, e.g. ``supercategory_0``), each with a ``contrastive_saliency`` tensor (Input×Gradient over features × time) and the cluster ``centroid`` / ``radius``. **``cluster_geometry``** (optional) records the cluster centroids, radii, and nearest-neighbour distances that place the saliency insets.

.. _modeling-glm-hmm:

Latent vocal states (GLM-HMM)
-----------------------------
A GLM-HMM treats the animal as switching between a small number of latent
"behaviour → vocalization rules": each latent state owns its own GLM mapping the
recent behavioural history to the vocalization, and the animal transitions between
states over a bout of calling. The pipeline reads the behaviours to emit from —
the manifold selection's ``final_model_features`` (``model_selection_path``), so the
GLM-HMM always emits from exactly the features that selection kept — turns each
session into one temporally-ordered observation sequence, fits the model across the
``n_states_min`` … ``n_states_max`` range, and selects the state count by
cross-validated held-out log-likelihood.

Two engines are available, chosen by ``transition_mode``:

* **``'static'``** — the classic Baum-Welch EM engine with a stationary ``K × K``
  transition matrix, for either ``emission_type``.
* **``'input_driven'``** — a direct-marginal engine whose transition into each time
  bin is itself a per-state GLM of the behavioural design (input-driven transitions,
  after Calhoun/Pillow/Murthy). This engine was validated on the Coen-2014 *Drosophila*
  courtship benchmark to reproduce its state-selection curve *without* the state
  collapse the EM engine suffers on weakly-identified categorical data, and is the
  recommended engine for a trustworthy state count. It fits the reference-coded
  categorical model (``emission_type='multinomial'``) or the torus product-von-Mises
  manifold model (``emission_type='manifold'``, torus only) by maximising the
  regularised marginal likelihood with L-BFGS under the ``input_driven_lambda_smooth``
  first-difference filter penalty.

Run on a single node from the notebook:

.. code-block:: python

    from usv_playpen.modeling.modeling_glm_hmm import run_glm_hmm_state_selection

    run_glm_hmm_state_selection(
        input_data_path="/mnt/falkner/Bartul/modeling/modeling_manifold_<...>.pkl",
        settings_path="/mnt/falkner/Bartul/modeling/modeling_settings.json",
        output_directory="/mnt/falkner/Bartul/modeling/glm_hmm_results/<...>",
        model_selection_path="/mnt/falkner/Bartul/modeling/model_selection_results/<...>",
    )

The saved results dict carries ``selected_n_states``, the per-``K`` ``selection_table``
(cross-validated log-likelihood plus a diagnostic BIC), ``log_pi`` and
``transition_matrix`` (a mean over events for the input-driven engines, whose
transitions vary per time bin), the per-session Viterbi ``state_paths``, and a
``metadata`` block recording the emission type, transition mode, and features used.

.. _modeling-behavioral-response:

Behavioral response
-------------------
Every pipeline above predicts a *vocal* target from behavioural history. This one
runs the other way: it predicts a **behavioural** variable and asks whether the
partner's calling changed it.

It answers two questions from one extraction:

1. **Does a male vocal bout, versus comparable silence, change her behaviour?**
2. **Does bout duration matter?**

Anchors are **events, not tiles**. Every row sits either at a male bout offset or
inside an inter-bout gap. A precursor design tiled anchors every 4 s across the
whole session and spent ~70% of its rows on windows containing no calls at all,
diluting the very contrast it was meant to measure — measured on this cohort,
only **30.6%** of tiled anchors carried any call.

The silent comparison is **inter-bout silence**, not globally quiet stretches. A
point inside a gap is a moment where he *could* have called and did not, with
both animals demonstrably still interacting. Globally quiet periods are a
different behavioural regime, and adjusting for the difference would mean
extrapolating between two separated groups rather than comparing within one.

.. note::

   A bout offset qualifies only when the next bout starts at least
   ``post_bout_silence_seconds`` later, which selects on the future: bigger bouts
   are followed sooner by the next bout (measured Spearman ``rho = -0.12``
   between syllable count and gap). A longer requirement therefore discards long
   bouts preferentially and truncates the predictor question 2 asks about. Tying
   the requirement to ``target_window_seconds`` keeps that loss at its minimum.

Covariates are **summaries, not lag histories**: each feature's pre-anchor window
collapses to a mean over the last 0.5 s and a mean over the full 4 s
(``covariate_summary_seconds``). Their job is to hold pre-anchor state fixed, not
to model a temporal filter, and these features are slow — autocorrelation
horizons run from ~0.75 s for ``speed`` to ~6.8 s for ``nose-nose`` — so finer
sub-windows would be collinear rather than informative. The design is ~38 columns
instead of 20 × 600, and the extraction artifact is a few MB instead of 3.5 GB.

The response is kept in **native units, never z-scored**: a Gamma likelihood
needs ``y > 0``, and the effect is then readable in the feature's own units.
Covariates *are* pooled z-scored, so their coefficients stay comparable.

All features at once
~~~~~~~~~~~~~~~~~~~~
``response_features`` lists every feature to extract. The anchors are identical
across features — only the target changes — so one extraction covers all of them
and the expensive part (reading 121 sessions of tracking CSV) is paid once rather
than per feature.

The likelihood is **derived, not configured**. It is a property of the feature's
support rather than a preference: a signed feature simply cannot use a Gamma
likelihood, and a settings key would let someone assert otherwise and silently
discard every non-positive row. Both magnitude folds map onto ``[0, inf)``, so a
folded feature is always Gamma-usable; an unfolded feature is Gamma-usable when
its lower bound is already non-negative.

.. list-table::
   :header-rows: 1

   * - Feature
     - Fold
     - Likelihood
   * - ``speed``, ``neck_elevation``, ``tail_curvature``
     - none
     - gamma
   * - ``allo_roll``
     - abs
     - gamma
   * - ``ego_yaw``, ``back_yaw``
     - smooth_abs
     - gamma
   * - ``allo_pitch``, ``back_pitch``
     - none (signed)
     - **gaussian**

Exact zeros would still be dropped under Gamma, but measured over 9M frames they
are negligible: none at all for ``speed`` and ``tail_curvature``, 0.009% for
``neck_elevation``, 0.005% for ``allo_roll``.

The contrast
~~~~~~~~~~~~
``behavioral_response_contrast`` fits a GLM with session-clustered standard
errors — not a nested predictive comparison. A nested comparison answers "does
the vocal block improve out-of-sample prediction" and reports a ``dD^2``, which
conflates effect size, timing and nonlinearity; on the tiled precursor it came
back at 0.0039, reproducible across every split and uninterpretable. A
coefficient answers the question actually asked, in the feature's own units.

Duration enters as **terciles interacted with** ``vocal``, so each duration band
gets its own step against silence::

    y ~ intercept + b1*vocal_band_0 + b2*vocal_band_1 + b3*vocal_band_2 + covariates

Question 1 is the joint statement of those steps; question 2 is how they differ.
Coding quiet rows as duration zero was rejected: it would assert that "no bout at
all" lies on the same line as "a very short bout", which is the assumption under
test. Terciles rather than a slope because duration is heavily skewed (median
0.43 s with a long tail), so one slope would be dominated by a few long bouts.

**Standard errors are clustered on session.** Rows within a session share an
animal, an arena and a day. On synthetic data with realistic session structure
the clustered error is **3.9x** the naive one, so without it every interval would
be about four times too narrow.

**The same model is refit per time bin.** The predictors never change, only the
target, so the single contrast becomes a time course of when the response appears
— the event-triggered average with pre-anchor state regressed out.

.. code-block:: python

    from usv_playpen.modeling.modeling_behavioral_response import BehavioralResponsePipeline
    from usv_playpen.modeling.behavioral_response_contrast import (
        behavioral_response_contrast,
    )

    BehavioralResponsePipeline(
        modeling_settings_dict=None
    ).extract_and_save_modeling_input_data()

    behavioral_response_contrast(
        input_pickle_path="/mnt/falkner/Bartul/modeling/modeling_behavioral_response_<...>.pkl",
        output_directory="/mnt/falkner/Bartul/modeling/behavioral_response_contrast",
    )

This analysis does **not** use the cluster job array: there is no screen and no
forward selection, so the univariate and model-selection stages have nothing to
consume. Extraction plus contrast is two calls, and the fitting is seconds.

Three failures are made loud rather than silent: a rank-deficient design names
the offending columns instead of raising ``LinAlgError`` from inside statsmodels;
a total loss of rows attributes the loss per covariate, because one non-finite
covariate drops a whole row and 38 columns multiply that risk; and a
``response_features`` entry that is not a known kinematic feature is rejected
before any session is read.

Notebook
--------
The ``modeling_analyses.ipynb`` notebook is the recommended interactive
entry point — it runs the whole workflow above in order from a single
**Parameters** cell. Its detailed walkthrough, knobs, and rendered source
live in :doc:`Notebooks`.
