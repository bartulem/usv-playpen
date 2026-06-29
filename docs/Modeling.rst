.. _Modeling:

Modeling
==================
This page explains how to use the **vocal-modeling pipelines** in
``usv_playpen.modeling``. Where the :ref:`Analyze` section produces the
per-session behavioral-feature tables, the modeling subsystem asks the
inverse question: *how well, and with what temporal structure, do those
behavioral kinematics predict a mouse's vocal behavior?*

Four prediction targets are supported, each with its own extraction
pipeline:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Pipeline
     - Predicts
     - ``Y(t)`` impulses
   * - ``VocalOnsetModelingPipeline``
     - whether a frame is the start of a vocal bout
     - bout starts
   * - ``BoutParameterPipeline``
     - per-bout duration / complexity / intensity
     - bout starts
   * - ``MultinomialModelingPipeline``
     - per-USV vocal category (one-vs-rest, multinomial)
     - per-USV starts
   * - ``ContinuousModelingPipeline``
     - per-USV 2-D acoustic-manifold (UMAP) position
     - per-USV starts

Each target is fit first with **univariate** generalized additive / linear
models (one behavioral feature at a time, to rank predictors), then with a
**forward-stepwise model-selection** routine that greedily stacks features,
and finally — for the continuous manifold target — with a non-linear
**1-D ResNet (CNN)** baseline. Every stochastic step is seeded from
``model_params.random_seed`` so results are reproducible.

The whole subsystem is configured by a single settings file,
``_parameter_settings/modeling_settings.json``, and is driven either
interactively from the :ref:`modeling-notebook` (``modeling_analyses.ipynb``,
detailed in :doc:`Notebooks`) or, at cohort scale, from the SLURM dispatchers
described in :ref:`modeling-model-selection`.

.. note::

   The modeling pipelines are **not** exposed as a GUI tab. Run them from
   the notebook (interactive, single node) or the dispatchers (HPC). Every
   pipeline reads ``_parameter_settings/modeling_settings.json`` via
   ``modeling_settings_dict=None``; pass an explicit dict to override.

Settings and inputs
--------------------
All knobs live in ``_parameter_settings/modeling_settings.json``. The most
important blocks are:

- ``io`` — ``session_list_file`` (a text file listing one session root per
  line) and ``save_directory`` (where modeling-input pickles and results
  are written).
- ``model_params`` — ``random_seed``, ``filter_history`` (seconds of
  behavioral history preceding each event), ``model_engine``
  (``'sklearn'`` or ``'pygam'``), ``split_strategy``, ``split_num``, the
  mixture-model inter-bout-interval parameters, and the predictor-mouse index.
- ``kinematic_features`` / ``vocal_features`` — which behavioral and vocal
  predictors enter the feature zoo, the manifold metric
  (``'euclidean'`` / ``'torus'``), and the manifold period.
- ``hyperparameters`` — per-engine model hyperparameters, including the
  ``deep_learning.cnn_continuous`` block consumed by the CNN runner.

.. note::

   **Regularisation tuning on the torus manifold.** For the continuous
   manifold target with ``manifold metric = 'torus'``, the inner-loop
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

The session-list file is the single source of the cohort. List one session
root per line:

.. parsed-literal::

    /mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_mute_female_sessions_list.txt

The cohort label (``male_mute_partner``, ``intact_partners_female``, …) is
derived from this filename by ``derive_experimental_condition`` and is
embedded into every output filename for provenance.

.. _modeling-extract:

1. Extract and save modeling input data
----------------------------------------
Each pipeline converts the per-session loader output into a
**modeling-input pickle** — a nested ``{feature: {session: {event-window
arrays}}}`` dictionary with an embedded ``_input_metadata`` provenance
block — that every downstream runner consumes. The four extraction calls
differ only in *what gets predicted*:

.. code-block:: python

    from usv_playpen.modeling.modeling_vocal_onsets import VocalOnsetModelingPipeline
    from usv_playpen.modeling.modeling_vocal_bout_parameters import BoutParameterPipeline
    from usv_playpen.modeling.modeling_vocal_categories_multinomial import (
        MultinomialModelingPipeline,
    )
    from usv_playpen.modeling.modeling_usv_manifold_position import (
        ContinuousModelingPipeline,
    )

    # Bout onsets (binary: is this frame a bout start?)
    VocalOnsetModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_modeling_input_data()

    # Bout parameters (continuous regression: duration / complexity / intensity)
    BoutParameterPipeline(
        modeling_settings_dict=None
    ).extract_and_save_modeling_input_data()

    # Vocal categories (multinomial, one-vs-rest across USV categories)
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
   is honoured in ``'individual'`` mode only; in ``'bout'`` (and ``'state'``)
   mode it is ignored, because the mixture-model inter-syllable-interval threshold used
   for bout grouping is calibrated on the all-USV interval distribution and
   would mis-group a category-sparsified sequence (a warning is printed if the
   setting is combined with a non-individual mode). When active, the chosen
   category column and index are embedded in the ``analysis_tag`` (e.g.
   ``individual_cat_vae_supercategory_6``) and ``_input_metadata``, so VAE-vs-QLVM
   and category-vs-supercategory are unambiguous in every downstream artifact
   name and provenance block.

.. _modeling-diagnostics:

2. Predictor diagnostics
-------------------------
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
flags predictor pairs whose ``|rho|`` crosses the audit's concern / exclude
thresholds and reports per-feature VIFs.

3. Univariate modeling and ranking
-----------------------------------
Univariate fits (one behavioral feature at a time) produce the ranking that
seeds model selection. At cohort scale they are dispatched as a SLURM job
array (one feature per task) via ``main_univariate_dispatcher``; the
per-feature pickles are then consolidated (see :ref:`modeling-consolidate`).
The ranking is visualised with ``plot_feature_ranking`` (single target) or
``plot_univariate_multinomial_performance`` (multinomial).

.. _modeling-model-selection:

4. Model selection (forward stepwise)
--------------------------------------
Greedy forward-stepwise selection stacks features on top of the univariate
ranking, adding at each step the feature whose contribution most improves
the held-out score (one-standard-error rule; see the source for the exact
stopping criterion). ``use_top_rank_as_anchor=True`` seeds step 0 with the
top univariate feature; ``p_val`` is the per-step acceptance threshold.

.. note::

   **Significance baseline for the discrete targets (bout onsets, binomial
   USV categories).** Every univariate fit is evaluated against a
   *label-shuffle permutation null*: the same estimator is re-fit on a copy of
   the **training** labels permuted within each fold — breaking the
   behaviour→vocalization association while preserving the marginal event rate
   — and then scored against the real (unpermuted) **test** labels, seeded
   reproducibly per fold from ``random_seed``. This replaced the earlier
   pseudo-class controls (resampled No-Bout / Other-USV baselines), which
   tested a weaker question. A feature is admitted to model selection only if
   its mean held-out **log-loss** beats a Bonferroni-corrected lower percentile
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
        bout_onset_model_selection,
        vocal_category_model_selection,
    )

    bout_onset_model_selection(
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
per-feature / per-step pickle each, ready for consolidation.

.. _modeling-consolidate:

5. Consolidate per-feature / per-step pickles
----------------------------------------------
After a SLURM array finishes, merge its outputs into a single artifact. The
consolidators assert metadata equality across every per-feature / per-step
pickle (guarding against stray files from a different run), hoist the agreed
``_input_metadata`` / ``_run_metadata`` / ``_univariate_metadata`` blocks to
the top of the consolidated artifact, and emit a self-describing filename:

.. code-block:: python

    from usv_playpen.modeling.consolidate_univariate_results import (
        consolidate as consolidate_univariate,
    )
    from usv_playpen.modeling.consolidate_model_selection_results import (
        consolidate as consolidate_model_selection,
    )

    consolidate_univariate(
        input_dir="/mnt/falkner/Bartul/modeling/<univariate_dir>",
        delete_individuals_after=False,
    )
    consolidate_model_selection(
        input_dir="/mnt/falkner/Bartul/modeling/<selection_dir>", move_to_steps_subdir=False
    )

The consolidated filenames are self-describing, e.g.:

.. parsed-literal::

    univariate_onsets_bout_male_mute_partner_<ts>.pkl
    model_selection_final_male_intact_partners_onsets_bout_mixed_<ts>.pkl

Set ``delete_individuals_after=True`` only once you have verified the
consolidated artifact is correct.

6. Visualisations
-----------------
The ``usv_playpen.visualizations.modeling_plots`` module renders every
modeling figure. The univariate set (``plot_feature_ranking``,
``plot_significant_filters``, ``plot_significant_filters_grid``,
``plot_raw_feature_difference``) operates on the consolidated univariate
pickle; the selection set (``plot_model_selection_results``,
``plot_multinomial_selection_trajectory``,
``plot_manifold_selection_trajectory``, the multivariate-filter atlases, and
``plot_multinomial_selection_diagnosis``) operates on the consolidated
``selection_*.pkl`` artifacts. Every plotter takes ``save_plot``/
``output_dir`` arguments and otherwise renders inline.

7. CNN deep-learning baseline
------------------------------
A non-linear baseline for the continuous manifold-position regression. The
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

Interactive notebook
---------------------
The ``modeling_analyses.ipynb`` notebook is the recommended interactive
entry point — it runs the whole workflow above in order from a single
**Parameters** cell. Its detailed walkthrough, knobs, and rendered source
live in :doc:`Notebooks`.
