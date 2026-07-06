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

**model_params** — the prediction target, history window, model engine, and
cross-validation splitting.

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
        "random_seed": 0,
        "spatial_cluster_num": 20,
        "split_strategy": "mixed",
        "split_num": 10,
        "test_proportion": 0.1,
        "session_split_max_attempts": 50000,
        "session_split_widen_step": 0.02,
        "session_split_widen_every": 1000,
        "usv_bout_time": 2,
        "usv_per_bout_floor": 2,
        "onset_target_category": null
    }

* **filter_history** — seconds of behavioral history preceding each event that feed the temporal filter (× ``camera_sampling_rate`` → frames).
* **mixture_model_component_index** / **mixture_model_z_score** — bout grouping: the fitted inter-syllable-interval mixture (``mixture_model_params``) is thresholded at ``mean + z·sd`` of the selected component (component ``0``, ``z = 2.58``) to decide where one bout ends and the next begins.
* **model_basis_function** — temporal-filter basis over the history window (``'raised_cosine'`` / ``'bspline'`` / ``'laplacian_pyramid'``; parameters in ``hyperparameters.basis_functions``). Only relevant when ``model_engine = 'sklearn'`` — the ``'pygam'`` engine uses its own tensor-product splines instead.
* **model_engine** — univariate model backend: ``'pygam'`` (tensor-product-spline GAM, a generalized additive model) or ``'sklearn'`` (basis-projected linear).
* **model_predictor_mouse_index** — which mouse (``0`` / ``1``) is the **partner**; the **target** — the mouse whose vocal behavior is being predicted — is defined as the other one. Both mice's kinematics enter the predictor set.
* **model_target_vocal_type** — onset target mode, ``'bout'`` (clustered) or ``'individual'`` (per-USV); used only by ``VocalOnsetModelingPipeline``.
* **model_target_variable** — for ``BoutParameterPipeline``, which per-bout quantity to regress (``'bout_durations'`` / complexity / intensity).
* **random_seed** — seeds every stochastic step (splits, permutations, initialisation) for reproducibility.
* **spatial_cluster_num** — number of spatial clusters used to build the spatial-CV folds for the continuous manifold target.
* **split_strategy** / **split_num** / **test_proportion** — cross-validation: ``'mixed'`` (stratified shuffle over the pooled data) or ``'session'`` (hold whole sessions out); ``split_num`` folds; ``test_proportion`` held out per fold.
* **session_split_max_attempts** / **session_split_widen_step** / **session_split_widen_every** — tuning for the ``'session'`` strategy's search for balanced held-out session sets (max attempts, plus how much / how often the balance tolerance is relaxed).
* **usv_bout_time** — duration (seconds) of the post-onset silence window that defines the **negative (No-USV) events** in ``'bout'`` mode: a candidate silent-epoch onset is kept only if no USV (from any source) starts within ``[t_onset, t_onset + usv_bout_time)`` after it.
* **usv_per_bout_floor** — the minimum number of USVs a positive bout must contain (``'bout'`` mode).
* **onset_target_category** — restrict positive onsets to a single USV category (``'individual'`` mode only); ``null`` pools all categories (see the single-category note under :ref:`Modeling input data <modeling-extract>`).

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
        "smooth_abs_features": {"ego_yaw": 1.0, "back_yaw": 0.5}
    }

* **egocentric** — single-mouse posture / movement features of the predictor mouse.
* **dyadic_pose** — relative-pose features between the two mice (``<self>-<other>`` naming).
* **dyadic_engagement** — social-engagement features (e.g. ``orofacial-sei``).
* **dyadic_pose_symmetric** — if ``true``, include both ``A-B`` and ``B-A`` orientations of each dyadic-pose feature.
* **include_1st_derivatives** / **include_2nd_derivatives** — also add the velocity / acceleration of each feature.
* **smooth_abs_features** — per-feature Gaussian-smoothing σ (frames) applied to the absolute value of the named features.

**vocal_features** — which vocal predictors enter the zoo, and the acoustic-manifold definition.

.. code-block:: json

    "vocal_features": {
        "usv_predictor_type": "categories_rate",
        "usv_predictor_partner_only": true,
        "usv_predictor_smoothing_sd": 1,
        "usv_category_column_name": "vae_supercategory",
        "usv_noise_column": "vae_supercategory",
        "usv_noise_categories": [0],
        "usv_manifold_column_names": ["vae_umap1", "vae_umap2"],
        "usv_manifold_metric": "euclidean",
        "usv_manifold_period": 1.0
    }

* **usv_predictor_type** — which vocal-syntax predictors to build (e.g. ``'categories_rate'`` = per-category USV rate).
* **usv_predictor_partner_only** — if ``true``, ingest only the *partner's* USV signals as predictors (not the target mouse's own vocal history).
* **usv_predictor_smoothing_sd** — Gaussian σ (frames) applied to the USV-rate predictor traces.
* **usv_category_column_name** — the USV-catalog column defining categories (``'vae_supercategory'`` / ``'qlvm_supercategory'`` / ``'vae_category'`` / ``'qlvm_category'``).
* **usv_noise_column** / **usv_noise_categories** — the column and category indices treated as noise and excluded.
* **usv_manifold_column_names** — the two catalog columns giving the 2-D manifold position (the ``ContinuousModelingPipeline`` target).
* **usv_manifold_metric** — ``'euclidean'`` (plane) or ``'torus'`` (wrap-aware) distance on the manifold.
* **usv_manifold_period** — the wrap period for the ``'torus'`` metric.

**diagnostics** — the predictor-collinearity and predictor-timescale audits (rendered in :ref:`Predictor diagnostics <modeling-diagnostics>`).

.. code-block:: json

    "diagnostics": {
        "collinearity_audit": false,
        "timescale_audit": false,
        "timescale_max_lag_seconds": 10.0,
        "timescale_n_shuffles": 1000,
        "timescale_shuffle_range": [20, 60],
        "timescale_signal_floor_seconds": 0.5,
        "timescale_signal_min_run_seconds": 0.2
    }

* **collinearity_audit** / **timescale_audit** — enable each audit during extraction.
* **timescale_max_lag_seconds** — maximum lag examined for the ACF (autocorrelation function) / cross-correlation horizons.
* **timescale_n_shuffles** / **timescale_shuffle_range** — number of circular-shift surrogates and the ``(min, max)`` shift range (seconds) for the null envelope.
* **timescale_signal_floor_seconds** / **timescale_signal_min_run_seconds** — thresholds for calling a horizon significant (minimum above-null run length).

**hyperparameters** — per-engine model tuning, grouped into four sub-blocks:

* **deep_learning.cnn_continuous** — the 1-D ResNet for the continuous manifold target (architecture, optimiser, spatial-CV, saliency), consumed by ``NeuralContinuousCNNRunner``.
* **jax_linear.bivariate** / **jax_linear.multinomial_logistic** — the JAX smooth bivariate regression (continuous manifold) and multinomial-logistic (vocal categories) models.
* **classical.pygam** / **classical.logistic_regression** / **classical.ridge_regression** — the ``'pygam'`` / ``'sklearn'`` engine models (GAM splines, logistic-CV, ridge).
* **basis_functions.raised_cosine** / **bspline** / **laplacian_pyramid** — parameters for each ``model_basis_function`` choice.

The regularisation controls (shared by both ``jax_linear`` sub-blocks) look like:

.. code-block:: json

    "jax_linear": {
        "bivariate": {
            "lambda_smooth_fixed": 1.0,
            "l2_reg_fixed": 0.01,
            "smoothness_derivative_order": 2,
            "learning_rate": 0.005,
            "max_iter": 20000,
            "tune_regularization_bool": true,
            "tune_regularization_params": {
                "lambda_smooth_decades_each_side": 3,
                "l2_reg_decades_each_side": 4,
                "inner_cv_folds": 5,
                "inner_cv_scoring_metric": "r2_spatial",
                "inner_cv_use_one_se_rule": true,
                "inner_max_iter": 2500
            }
        }
    }

* **lambda_smooth_fixed** / **l2_reg_fixed** — the fixed smoothness and L2 penalties used when regularisation tuning is off.
* **tune_regularization_bool** — if ``true``, run an inner-loop cross-validation to pick ``lambda_smooth`` / ``l2_reg`` (parameters in ``tune_regularization_params``: the search width in decades, inner-CV folds, scoring metric, and the one-standard-error rule).

.. note::

   **Regularisation tuning on the torus manifold.** For the continuous
   manifold target with ``usv_manifold_metric = 'torus'``, this inner-loop
   regularisation CV is unnecessary and is switched off automatically: the
   selection score (wrap-aware distance correlation ``dcor_xy``) is
   regularisation-invariant through the ``atan2`` decode, so the pipeline
   forces ``hyperparameters.jax_linear.bivariate.tune_regularization_bool``
   to ``False`` regardless of its configured value. Leave tuning off and use
   the advised fixed values ``lambda_smooth_fixed = 1.0`` and
   ``l2_reg_fixed = 0.01`` — ``lambda_smooth`` no longer moves the score but
   still shapes the interpretable published filter, so it is not a free
   parameter for visualisation. On euclidean / VAE / UMAP manifolds (where
   the score is ``r2_spatial``) tuning is honoured as configured.

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
    BoutParameterPipeline           ->  { "positive_events", "bout_durations", "bout_syllable_counts" }
    VocalCategoryModelingPipeline   ->  { "target_feature_arr", "other_feature_arr" }
    MultinomialModelingPipeline     ->  { <per-USV feature windows>, "labels" }
    ContinuousModelingPipeline      ->  { <per-USV feature windows>, <2-D manifold position>, "category" }

* **VocalOnsetModelingPipeline** — ``usv_feature_arr`` = positive onset windows, ``no_usv_feature_arr`` = silent-epoch (negative) windows. ``analysis_specific``: ``model_target_vocal_type``, ``usv_count_threshold``.
* **BoutParameterPipeline** — ``positive_events`` = the bout-onset windows; ``bout_durations`` / ``bout_syllable_counts`` = the per-bout regression targets. ``analysis_specific``: ``target_variable``.
* **VocalCategoryModelingPipeline** — ``target_feature_arr`` = windows for the chosen target category, ``other_feature_arr`` = windows for the pooled "other". ``analysis_specific``: ``target_category``.
* **MultinomialModelingPipeline** — per-USV feature windows paired with ``labels`` (each USV's category). ``analysis_specific``: ``categories_kept``, ``class_counts``.
* **ContinuousModelingPipeline** — per-USV feature windows paired with the 2-D acoustic-manifold position target (and each USV's ``category``). ``analysis_specific``: ``usv_manifold_column_names``.

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
        "spearman_rho": <array (F, F)>,                 # feature x feature Spearman correlation
        "vif":          <array (F,)>,                   # per-feature variance-inflation factor
        "rho_signal":   <array (F,)>,                   # each feature's correlation with Y(t)
        "concern":      [["ego_yaw", "back_yaw"]],      # pairs above concern_threshold
        "exclude":      [],                             # pairs above exclude_threshold
        "concern_threshold": 0.7, "exclude_threshold": 0.9,
        "_input_metadata": {"...": "..."}
    }

* **spearman_rho** / **vif** — the pairwise Spearman-``|rho|`` matrix and per-feature VIFs (the two ``plot_collinearity_audit`` panels).
* **rho_signal** — each feature's correlation with the event train ``Y(t)`` (also stored per-session as ``rho_signal_per_session_mean`` / ``_sem``, against a ``rho_signal_null_mean`` baseline).
* **concern** / **exclude** — the feature pairs whose ``|rho|`` crosses each threshold.

``*_timescales.pkl`` — how far in time each predictor carries information:

.. code-block:: text

    {
        "features":            ["speed", "..."],        # F feature names (columns below)
        "acf_lags_seconds":    "<array (L,)>",          # ACF lag axis
        "acf_median":          "<array (L, F)>",        # median autocorrelation per feature (+ acf_p25 / acf_p75)
        "acf_null_mean":       "<array (L, F)>",        # circular-shift null envelope (+ acf_null_p0_5 / _p99_5)
        "tau_acf_1_over_e":    "<array (F,)>",          # per-feature ACF horizons (+ tau_acf_0_2, tau_acf_integrated)
        "signal_lags_seconds": "<array (M,)>",          # cross-correlation lag axis
        "rho_signal":          "<array (M, F)>",        # feature x Y(t) cross-correlation (+ null envelope)
        "ibi_thresholds": {"...": "..."}, "configured_filter_history": 4,
        "_input_metadata": {"...": "..."}
    }

* **acf_median** (rows = lags, columns = features) vs **acf_null_*** — each feature's autocorrelation against a circular-shift null; the **ACF horizon** (``tau_acf_*``) is how long it stays above that null.
* **rho_signal** / **signal_lags_seconds** — each feature's cross-correlation with ``Y(t)`` across lags; the **cross-correlation horizon** is the lag at which it leaves the null envelope.
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
``_input_metadata`` / ``_run_metadata`` / ``_univariate_metadata`` blocks to the
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
the held-out score (one-standard-error rule; see the source for the exact
stopping criterion). ``use_top_rank_as_anchor=True`` seeds step 0 with the
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

* **``steps``** — an ordered list, one entry per forward-selection step. ``step_idx`` is the iteration, ``current_features`` are those already chosen, ``baseline_score`` is their held-out score (the chance floor at step 0), and ``selected_feature`` is the feature accepted this step (``None`` marks the final, rejected step). For the multinomial and manifold selectors, step 0's ``selected_feature`` is the sentinel ``'null_model_free'`` baseline.
* **``candidates_summary``** — under each step, every candidate feature tested that step mapped to its per-fold metrics. For the discrete / regression targets these are flat per-fold arrays (``ll``, ``auc``, ``score``, ``f1``, ``brier``, ``ece``, ``mcc``, ``confusion_matrix``, ``n_iter`` / ``converged`` / ``fit_time``) plus aggregate ``mean_ll`` / ``se_ll``; the multinomial and manifold selectors nest these under a ``folds.metrics`` sub-dict (with ``y_true`` / ``y_pred`` / ``y_probs`` / ``classes`` and the per-fold ``selected_lambda_smooth`` / ``selected_l2_reg`` regularisation choices).
* **last accepted step** — additionally carries ``final_model_features`` (the cumulative selected set) and ``filter_shapes`` (the per-fold refit filters) of the published model.
* **metadata blocks** — ``_input_metadata`` and ``_univariate_metadata`` carry the upstream extraction / univariate provenance, ``_run_metadata`` the selection config, and ``_consolidation_metadata`` the merge audit.

Visualise the trajectory with ``plot_model_selection_results`` (binary / regression),
``plot_multinomial_selection_trajectory`` (multinomial), or
``plot_manifold_selection_trajectory`` (continuous manifold): each reads the
consolidated ``model_selection_final_*.pkl`` and shows the per-step held-out score
gain and the retained-feature filters.

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

Notebook
--------
The ``modeling_analyses.ipynb`` notebook is the recommended interactive
entry point — it runs the whole workflow above in order from a single
**Parameters** cell. Its detailed walkthrough, knobs, and rendered source
live in :doc:`Notebooks`.
