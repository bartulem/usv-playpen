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
     - per-USV 2-D acoustic-manifold position (the QLVM torus)
     - per-USV starts

Each target is fit first with **univariate** generalized additive / linear
models (one behavioral feature at a time, to rank predictors), then with a
**forward-stepwise model-selection** routine that greedily stacks features,
and finally — for the continuous manifold target — with a non-linear
**1-D ResNet (a residual-network convolutional neural network, CNN)**. Every stochastic step is seeded from
``model_validation.random_seed`` so results are reproducible.

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

   The ``mixture_model_params`` block holds the fitted inter-USV-interval mixture of each sex, from which the inter-bout threshold is derived. It is not a tuning knob, and no pipeline writes it: the *Inter-USV interval analyses* notebook (``inter_usv_interval_analyses.ipynb``) fits the mixtures and prints the block in this layout for pasting into the settings. Its four entries are ``male.means`` / ``male.sds`` and ``female.means`` / ``female.sds``: the means and standard deviations of the lognormal components on the natural-log scale of the interval in seconds, in ascending order, so component ``0`` is the shortest-interval peak. The threshold is ``exp(means[i] + z * sds[i])``, with ``i`` = ``model_params.mixture_model_component_index`` and ``z`` = ``model_params.mixture_model_z_score``. One index serves both sexes, so it must be valid for the shorter list (the shipped male mixture has 4 components and the female 3, so ``0``–``2``); a negative or out-of-range index raises in the vocal-event loaders. The vocal-event loaders also accept an inverse-Gaussian block, ``"model_class": "ig"`` with linear-time ``mus`` / ``lambdas`` in place of ``means`` / ``sds``.

   :Options: for each of ``male`` and ``female``, ``means`` a list of real numbers (log seconds) and ``sds`` a list of positive numbers of the same length, with the fitted component order kept
   :Default:
      * ``male`` — ``means`` ``[-2.77117, -1.71942, -1.23503, 2.4513]`` / ``sds`` ``[0.23762, 0.23762, 1.17292, 1.09014]``
      * ``female`` — ``means`` ``[-2.69071, -2.03767, 0.05272]`` / ``sds`` ``[0.23676, 0.83522, 1.34804]``

**io** — the cohort and where outputs go.

.. code-block:: json

    "io": {
        "session_list_file": "/mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_intact_partners_sessions_list.txt",
        "save_directory": "/mnt/falkner/Bartul/modeling",
        "csv_separator": ",",
        "camera_sampling_rate": 150
    }

.. list-table:: ``io`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`session_list_file <mio-session-list-file>`
     - path (see JSON above)
     - path to an existing plain-text session list
   * - :ref:`save_directory <mio-save-directory>`
     - ``/mnt/falkner/Bartul/modeling``
     - path to a writable directory
   * - :ref:`csv_separator <mio-csv-separator>`
     - ``','``
     - a single-character string
   * - :ref:`camera_sampling_rate <mio-camera-sampling-rate>`
     - ``150``
     - Hz > 0

.. _mio-session-list-file:

``session_list_file``
   Path to the text file that lists the cohort's sessions (one session root per line; see below).

   :Options: path (string) to an existing plain-text file holding at least one session root
   :Default: ``/mnt/falkner/Bartul/modeling/input_files/behavioral_courtship_intact_partners_sessions_list.txt``
   :Notes:
      * A missing file or one without any session path raises.
      * The filename should contain ``intact_partners``, ``mute_female``, ``mute_male`` or ``all_partners`` so the cohort label can be derived, otherwise the label is ``unspecified``.

.. _mio-save-directory:

``save_directory``
   Directory where every modeling-input pickle, audit, and result is written.

   :Options: path (string) to a writable directory
   :Default: ``/mnt/falkner/Bartul/modeling``
   :Notes: not validated; the extraction pipelines write into it as given.

.. _mio-csv-separator:

``csv_separator``
   Column delimiter of the per-session ``*_behavioral_features.csv`` files.

   :Options: a single-character string matching the behavioral-features CSVs
   :Default: ``','``
   :Notes: not validated; a wrong separator fails when the CSVs are parsed.

.. _mio-camera-sampling-rate:

``camera_sampling_rate``
   Camera frame rate in Hz; every pipeline uses it to convert ``filter_history`` seconds into a frame count.

   :Options: a positive number, in Hz (not validated)
   :Default: ``150``

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
``male_intact_partner`` or ``female_intact_partner``, by the target mouse's sex; mute cohorts read ``male_mute_partner``
or ``female_mute_partner``, also by the target mouse's sex; a pooled intact + mute cohort whose filename contains ``all_partners`` reads ``male_all_partners`` or ``female_all_partners``; a list whose filename starts with ``ephys`` puts ``ephys_`` in front of any of these, e.g. ``ephys_male_all_partners``; and any other filename ``unspecified``) and embeds that label into every output filename,
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
        "model_target_vocal_type": "bout_onset",
        "model_target_variable": "bout_durations",
        "selection_p_val": 0.01,
        "selection_effect_floor": 0.1,
        "selection_n_bootstrap": 1000,
        "selection_ci_level": 0.99,
        "selection_min_fold_events": 30,
        "selection_sign_flip_exact_max_folds": 22,
        "selection_sign_flip_monte_carlo_draws": 1048576,
        "filter_null_permutations": 20,
        "usv_bout_time": 2,
        "usv_per_bout_floor": 2,
        "onset_target_category": null,
        "onset_target_classes": ["usv"]
    }

.. list-table:: ``model_params`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`filter_history <mp-filter-history>`
     - ``4``
     - seconds > 0
   * - :ref:`mixture_model_component_index <mp-mixture-model>`
     - ``0``
     - ``0``–``3`` (male), ``0``–``2`` (female)
   * - :ref:`mixture_model_z_score <mp-mixture-model>`
     - ``2.58``
     - standard deviations > 0
   * - :ref:`model_basis_function <mp-model-basis-function>`
     - ``'raised_cosine'``
     - ``'raised_cosine'``, ``'bspline'``, ``'laplacian_pyramid'``, ``'identity'``
   * - :ref:`model_engine <mp-model-engine>`
     - ``'pygam'``
     - ``'pygam'``, ``'sklearn'``
   * - :ref:`model_predictor_mouse_index <mp-model-predictor-mouse-index>`
     - ``1``
     - ``0``, ``1``
   * - :ref:`model_target_vocal_type <mp-model-target-vocal-type>`
     - ``'bout_onset'``
     - ``'bout_onset'``, ``'individual'``, ``'state'``, ``'bout_offset'``
   * - :ref:`model_target_variable <mp-model-target-variable>`
     - ``'bout_durations'``
     - ``'bout_durations'``, ``'bout_syllable_counts'``, ``'mean_mask_complexity'``, ``'total_mask_complexity'``
   * - :ref:`selection_p_val <mp-selection-p-val>`
     - ``0.01``
     - number in ``(0, 1)``
   * - :ref:`selection_effect_floor <mp-manifold-gate>`
     - ``0.1``
     - fraction in ``[0, 1]``
   * - :ref:`selection_n_bootstrap <mp-manifold-gate>`
     - ``1000``
     - integer > 0
   * - :ref:`selection_ci_level <mp-manifold-gate>`
     - ``0.99``
     - number in ``(0, 1)``
   * - :ref:`selection_min_fold_events <mp-selection-min-fold-events>`
     - ``30``
     - integer > 0
   * - :ref:`selection_sign_flip_exact_max_folds <mp-sign-flip>`
     - ``22``
     - integer > 0
   * - :ref:`selection_sign_flip_monte_carlo_draws <mp-sign-flip>`
     - ``1048576``
     - integer > 0
   * - :ref:`filter_null_permutations <mp-filter-null-permutations>`
     - ``20``
     - integer >= 0
   * - :ref:`usv_bout_time <mp-usv-bout-time>`
     - ``2``
     - seconds > 0
   * - :ref:`usv_per_bout_floor <mp-usv-per-bout-floor>`
     - ``2``
     - integer >= 1
   * - :ref:`onset_target_category <mp-onset-target-category>`
     - ``null``
     - ``null``, or one category label
   * - :ref:`onset_target_classes <mp-onset-target-classes>`
     - ``["usv"]``
     - list of ``"usv"``, ``"squeak"``, ``"overlap"``

.. _mp-filter-history:

``filter_history``
   Seconds of behavioral history preceding each event that feed the temporal filter.

   :Options: any positive number of seconds (not validated)
   :Default: ``4``
   :Notes: multiplied by ``camera_sampling_rate`` to give the window in frames.

.. _mp-mixture-model:

``mixture_model_component_index`` / ``mixture_model_z_score``
   Bout grouping: where one bout ends and the next begins. The fitted inter-syllable-interval mixture (``mixture_model_params``) is thresholded at ``mean + z·sd`` of the selected component.

   :Options:
      * ``mixture_model_component_index`` — an integer index into the target sex's fitted mixture in ``mixture_model_params``: ``0``–``3`` for male, ``0``–``2`` for female. Components are ordered by mean log interval, so ``0`` is the shortest-interval component; an index past the last component raises ``IndexError``.
      * ``mixture_model_z_score`` — a positive number of standard deviations (not validated)
   :Default: ``0`` and ``2.58`` (component ``0``, ``z = 2.58``)

.. _mp-model-basis-function:

``model_basis_function``
   Temporal-filter basis over the history window.

   :Options:
      * ``'raised_cosine'`` — raised-cosine basis (parameters in ``hyperparameters.basis_functions``)
      * ``'bspline'`` — B-spline basis (parameters in ``hyperparameters.basis_functions``)
      * ``'laplacian_pyramid'`` — Laplacian-pyramid basis (parameters in ``hyperparameters.basis_functions``)
      * ``'identity'`` — the raw per-frame history, no projection
   :Default: ``'raised_cosine'``
   :Notes: only relevant when ``model_engine = 'sklearn'``; the ``'pygam'`` engine uses its own tensor-product splines instead and ignores this setting. Any other value raises when the ``'sklearn'`` engine builds its basis.

.. _mp-model-engine:

``model_engine``
   Univariate model backend.

   :Options:
      * ``'pygam'`` — tensor-product-spline GAM (a generalized additive model)
      * ``'sklearn'`` — basis-projected linear model
   :Default: ``'pygam'``
   :Notes: the multinomial category and acoustic-manifold targets always run the JAX estimators and ignore it. The model selections raise on any other value.

.. _mp-model-predictor-mouse-index:

``model_predictor_mouse_index``
   Which mouse is the **partner**; the **target** — the mouse whose vocal behavior is being predicted — is the other one.

   :Options: ``0`` or ``1``, the slot in the session's ``track_names``
   :Default: ``1``
   :Notes: both mice's kinematics enter the predictor set. Not validated; the target is ``abs(index - 1)``, so other values are meaningless.

.. _mp-model-target-vocal-type:

``model_target_vocal_type``
   Onset target mode; used only by ``VocalOnsetModelingPipeline``.

   :Options:
      * ``'bout_onset'`` — clustered bout onsets, both positive and negative pre-event windows kept clean
      * ``'individual'`` — per-USV onsets
      * ``'state'`` — the session is sampled on a regular ``filter_history``-spaced time grid and each sample labelled vocal / silent by whether a vocalization of the ``onset_target_classes`` occupies it, with no clean-history requirement
      * ``'bout_offset'`` — the END of a bout against an interior USV of a bout that continued; configured by the ``bout_offset`` block, see :ref:`Bout offsets <modeling-bout-offsets>`
   :Default: ``'bout_onset'``
   :Notes: any other value raises ``ValueError``.

.. _mp-model-target-variable:

``model_target_variable``
   For ``BoutParameterPipeline``, which per-bout quantity to regress.

   :Options:
      * ``'bout_durations'`` — first-to-last-USV span, seconds
      * ``'bout_syllable_counts'`` — the number of USVs in the bout
      * ``'mean_mask_complexity'`` — per-USV mean spectrogram-mask complexity
      * ``'total_mask_complexity'`` — spectrogram-mask complexity summed over the bout
   :Default: ``'bout_durations'``
   :Notes: the model-selection dispatcher (``main_model_selection_dispatcher``) reads it from here for the ``params`` selection; it is not a command-line flag. Any other value raises ``KeyError`` at extraction.

.. _mp-selection-p-val:

``selection_p_val``
   The significance level gating whether a candidate feature is admitted during forward-stepwise model selection.

   :Options: a number in ``(0, 1)`` (not validated)
   :Default: ``0.01``
   :Notes:
      * On the onset, category and bout-parameter selections it is a family-wise level, Bonferroni-divided by the number of candidate features before the null percentile is read.
      * On the acoustic-manifold target it is the Benjamini–Hochberg FDR ``q`` used to screen candidates.
      * The model-selection dispatcher passes this value to every selector; it is not a command-line flag.

.. _mp-manifold-gate:

``selection_effect_floor`` / ``selection_n_bootstrap`` / ``selection_ci_level``
   The acoustic-manifold selection's **fold-grain acceptance gate** (``continuous_vocal_manifold_model_selection``): a feature is kept only when its per-fold paired score margin over the shuffle null (the macro von Mises log-likelihood on the torus, the wrap-aware distance correlation on euclidean) is consistently positive across CV folds.

   :Options:
      * ``selection_effect_floor`` — the relative effect floor a screened feature must clear: a fraction in ``[0, 1]`` of the top surviving driver's margin; ``0`` disables the floor (not validated)
      * ``selection_n_bootstrap`` — the number of fold bootstrap resamples, a positive integer (not validated)
      * ``selection_ci_level`` — the bootstrap confidence level, a number in ``(0, 1)``, whose lower bound must exceed ``0`` for an anchor / forward step to be accepted (not validated)
   :Default: ``0.1`` (10%), ``1000`` and ``0.99``
   :Notes:
      * These apply to the manifold gate only; the onset / category / bout-parameter selections use ``selection_p_val`` alone.
      * The screen's p-value is not a bootstrap: it is an exact one-sided sign-flip permutation test over the folds, ``p = (1 + k) / (1 + B)`` over the ``B = 2**manifold_n_cv_folds - 1`` non-identity sign patterns (every pattern enumerated; ``k`` counts those whose margin sum ties or beats the observed one), so it is never ``0`` and its floor is ``1 / 2**manifold_n_cv_folds`` (``9.5e-7`` at the shipped 20 folds, below the Benjamini–Hochberg rank-1 threshold ``q / m``); above ``selection_sign_flip_exact_max_folds`` usable folds a seeded Monte-Carlo set of ``selection_sign_flip_monte_carlo_draws`` patterns replaces the enumeration.
      * All of the gate's knobs are read from the run's own settings file at call time.

.. _mp-selection-min-fold-events:

``selection_min_fold_events``
   The manifold gate scores a fold only when its test set holds at least this many events.

   :Options: a positive integer (not validated)
   :Default: ``30``
   :Notes: a sparser fold is dropped from the paired margin, since too few points give an unstable per-fold score (and an unstable per-fold ``kappa`` when the concentration is not frozen).

.. _mp-sign-flip:

``selection_sign_flip_exact_max_folds`` / ``selection_sign_flip_monte_carlo_draws``
   How the manifold gate's sign-flip p-value is computed.

   :Options:
      * ``selection_sign_flip_exact_max_folds`` — up to this many usable folds, every one of the ``2**n`` sign patterns is enumerated, so the test is exact; a positive integer (memory grows as ``2**n``, so values much above ~25 are impractical; not validated)
      * ``selection_sign_flip_monte_carlo_draws`` — above that fold count, a seeded Monte-Carlo set of this many patterns is drawn instead and the p-value floor becomes ``1 / (1 + draws)``; a positive integer (not validated)
   :Default: ``22`` and ``1048576`` (``2**20``)
   :Notes: ``2**22`` sums are ~32 MB of float64, a fraction of a second. The shipped ``manifold_n_cv_folds`` (20) is always enumerated exactly, so the draw count matters only for a run configured with more than 22 folds.

.. _mp-filter-null-permutations:

``filter_null_permutations``
   How many target-permuted refits the torus and multinomial (QLVM category) model selections store, as the no-signal band of the filter figures.

   :Options: an integer ``>= 0`` (not validated); ``0`` stores none, and the figures then draw no band
   :Default: ``20``
   :Notes:
      * The selections refit the final feature set on the development sessions this many times with the target permuted across the development set (positions with their weights and regions on the torus, category labels for the multinomial model) and store the permuted filters with the held-out block (``heldout['null_weights']``).
      * The filter figures draw each filter against that no-signal band: part of a filter's weight at the two ends of the history window appears without any signal (the residual end weight of the lag basis), so only weight above the band reads as a real effect.
      * No band is stored when the held-out reserve is disabled.

.. _mp-usv-bout-time:

``usv_bout_time``
   Duration (seconds) of the post-onset silence window that defines the **negative (No-USV) events** in ``'bout_onset'`` and ``'individual'`` modes.

   :Options: a positive number of seconds (not validated)
   :Default: ``2``
   :Notes: a candidate silent-epoch onset is kept only if no vocalization of the target mouse, and none assigned to no animal, starts within ``[t_onset, t_onset + usv_bout_time)`` after it.

.. _mp-usv-per-bout-floor:

``usv_per_bout_floor``
   The minimum number of USVs a bout must contain.

   :Options: an integer ``>= 1`` (not validated)
   :Default: ``2``
   :Notes: applies to the positive bouts in ``'bout_onset'`` and ``'bout_offset'`` modes, the bouts of ``BoutParameterPipeline``, and the bouts of the behavioral-response pipeline.

.. _mp-onset-target-category:

``onset_target_category``
   Restrict positive onsets to a single USV category (``'individual'`` mode only).

   :Options:
      * ``null`` — all categories pooled
      * one integer label of ``usv_category_column_name`` — ``1``–``4`` for ``qlvm_category`` (1 is the category with the most USVs); a set value needs a non-null ``usv_category_column_name``
   :Default: ``null``
   :Notes: read only in ``'individual'`` mode, ignored with a warning otherwise. See the single-category note under :ref:`Modeling input data <modeling-extract>`.

.. _mp-onset-target-classes:

``onset_target_classes``
   Which vocal classes are the positive onsets: a non-empty list read from the summary's ``usv`` / ``squeak`` booleans (written by ``detect-usv-squeaks``).

   :Options: a non-empty list of distinct classes:

      * ``"usv"`` — a pure USV (``usv`` true and ``squeak`` false)
      * ``"squeak"`` — a pure squeak (``squeak`` true and ``usv`` false)
      * ``"overlap"`` — a USV and a squeak in one segment (both true)

      Common lists:

      * ``["usv"]`` — pure USVs (the default)
      * ``["usv", "overlap"]`` — every segment holding a USV
      * ``["squeak", "overlap"]`` — every segment holding a squeak
      * ``["overlap"]`` — overlaps alone
      * ``["usv", "squeak", "overlap"]`` — everything
   :Default: ``["usv"]``
   :Notes:
      * **Validation** — anything else raises ``ValueError``. The bout modes (``'bout_onset'``, ``'bout_offset'``) accept ``"usv"`` and ``"overlap"`` only, since squeaks do not come in bouts and the inter-bout threshold is calibrated on USV intervals; ``'individual'`` and ``'state'`` take any list. An unscored segment (null booleans) is in no class.
      * **Timing** — each positive is timed as the event the list asks about (``os_utils.vocal_event_times``): a pure USV by its segment ``start`` / ``stop``, a pure squeak by its own extent ``squeak_start`` / ``squeak_end`` (written by ``detect-usv-squeaks``), and an overlap segment by the squeak extent when the list names ``"squeak"`` but not ``"usv"`` (the squeak typically begins ~40 ms after the segment's USV onset), by the segment when it names ``"usv"`` but not ``"squeak"``, and by the union of the two otherwise (``["overlap"]`` alone, or both classes named); a list that can time a row by the squeak extent raises when a summary has no ``squeak_start`` / ``squeak_end``.
      * **Bouts** — vocalizations of the other classes neither join nor split a bout; the bout-parameter and behavioral-response pipelines group their bouts from the same classes, so a bout is the same in every model.
      * **Silence** — silence is not class-specific: a negative window, and a ``'bout_onset'`` positive's clean history, must both be free of every vocalization of the target mouse (any class) and of every vocalization assigned to no animal, so the two groups are built under one rule whatever the target. A negative window needs exactly the clean history a ``'bout_onset'`` positive needs, ``filter_history`` of silence before its end and nothing more: window ends are tiled every ``filter_history`` from the start of each clean zone, so the first negative after a vocalization ends ``filter_history`` after its stop, the earliest point a positive is clean; a squeak-bearing segment blocks the union of the segment and its squeak extent (``os_utils.vocal_occupancy_times``), which may reach past the segment, so every summary needs ``squeak_start`` / ``squeak_end``.
      * **Order and naming** — noise segments are removed first in every case when ``vocal_features.exclude_noise_usvs`` is ``true``. Applied before ``onset_target_category``; any list other than ``["usv"]`` is embedded in the ``analysis_tag`` (e.g. ``individual_squeak``, ``bout_onset_usv+overlap``).

**model_validation** — the held-out test set and cross-validation splitting.

.. code-block:: json

    "model_validation": {
        "held_out_test_proportion": 0.1,
        "split_strategy": "session",
        "n_cv_folds": 10,
        "manifold_n_cv_folds": 20,
        "cv_validation_proportion": 0.1,
        "random_seed": 0,
        "spatial_cluster_num": 20,
        "session_split_initial_tolerance": 0.05,
        "session_split_max_attempts": 50000,
        "session_split_widen_step": 0.02,
        "session_split_widen_every": 1000,
        "stratification_max_bins": 10,
        "stratification_min_events_per_bin": 20
    }

.. list-table:: ``model_validation`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`held_out_test_proportion <mval-held-out-test-proportion>`
     - ``0.1``
     - number in ``[0, 1)``
   * - :ref:`split_strategy <mval-split-strategy>`
     - ``'session'``
     - ``'session'``, ``'mixed'``
   * - :ref:`n_cv_folds <mval-n-cv-folds>`
     - ``10``
     - integer > 0
   * - :ref:`manifold_n_cv_folds <mval-manifold-n-cv-folds>`
     - ``20``
     - integer > 0 (at least ``11`` in practice)
   * - :ref:`cv_validation_proportion <mval-cv-validation-proportion>`
     - ``0.1``
     - number in ``(0, 1)``
   * - :ref:`random_seed <mval-random-seed>`
     - ``0``
     - any integer
   * - :ref:`spatial_cluster_num <mval-spatial-cluster-num>`
     - ``20``
     - integer > 0
   * - :ref:`session_split_initial_tolerance <mval-session-split-initial-tolerance>`
     - ``0.05``
     - number in ``[0, 1]``
   * - :ref:`session_split_max_attempts <mval-session-split-search>`
     - ``50000``
     - integer > 0
   * - :ref:`session_split_widen_step <mval-session-split-search>`
     - ``0.02``
     - number > 0
   * - :ref:`session_split_widen_every <mval-session-split-search>`
     - ``1000``
     - integer > 0
   * - :ref:`stratification_max_bins <mval-stratification>`
     - ``10``
     - integer >= 2
   * - :ref:`stratification_min_events_per_bin <mval-stratification>`
     - ``20``
     - integer > 0

.. _mval-held-out-test-proportion:

``held_out_test_proportion``
   The fraction of sessions carved off **once** as a final held-out test set that is excluded from every CV fold and the whole feature-selection search, then scored only by the refit final model — an honest last-look estimate untouched by selection.

   :Options: a number in ``[0, 1)`` (other values raise ``ValueError``); ``0`` disables the carve-out (the entire dataset enters CV)
   :Default: ``0.1``

.. _mval-split-strategy:

``split_strategy``
   The cross-validation scheme.

   :Options:
      * ``'session'`` — hold whole sessions out, so no session straddles train and test
      * ``'mixed'`` — stratified shuffle over the pooled data
   :Default: ``'session'``
   :Notes: the category and manifold pipelines raise on any other value.

.. _mval-n-cv-folds:

``n_cv_folds``
   The number of outer cross-validation folds each candidate feature is scored across, in every pipeline except the acoustic-manifold position.

   :Options: a positive integer (not validated)
   :Default: ``10``

.. _mval-manifold-n-cv-folds:

``manifold_n_cv_folds``
   The outer fold count of the acoustic-manifold position pipeline only: the univariate manifold runner, the selection screen that re-tests those folds, and the manifold forward selection.

   :Options: a positive integer (not validated); the screen's p-value floor ``1 / 2**n`` must sit below the Benjamini–Hochberg rank-1 threshold ``q / m``, which needs at least ``11`` folds for 19 candidates at ``q = 0.01``
   :Default: ``20``
   :Notes:
      * The screen tests each feature's per-fold margin with an exact sign-flip permutation over these folds, whose smallest attainable p-value is ``1 / 2**manifold_n_cv_folds``; at 10 folds that floor (``1 / 1024``) sits above the Benjamini–Hochberg rank-1 threshold ``q / m`` (``0.01 / 19``), so no feature could pass, while 20 folds bring it to ``9.5e-7``.
      * A manifold resume refuses a checkpoint written with another fold count.

.. _mval-cv-validation-proportion:

``cv_validation_proportion``
   The fraction of data held out per CV fold as that fold's validation set.

   :Options: a number in ``(0, 1)`` (not validated)
   :Default: ``0.1``
   :Notes: under ``'session'`` it is the fraction of sessions held out.

.. _mval-random-seed:

``random_seed``
   Seeds every stochastic step (splits, permutations, initialisation) for reproducibility.

   :Options: an integer
   :Default: ``0``
   :Notes: ``BoutParameterPipeline`` raises on a non-integer; some helpers read ``null`` as ``0``.

.. _mval-spatial-cluster-num:

``spatial_cluster_num``
   The number of spatial clusters used to build the spatial-CV folds for the continuous manifold target.

   :Options: a positive integer, at most the number of development events (not validated)
   :Default: ``20``

.. _mval-session-split-initial-tolerance:

``session_split_initial_tolerance``
   The starting class-balance tolerance for the ``'session'`` strategy's held-out-set search, before it begins widening.

   :Options: a number in ``[0, 1]`` (not validated)
   :Default: ``0.05``

.. _mval-session-split-search:

``session_split_max_attempts`` / ``session_split_widen_step`` / ``session_split_widen_every``
   Tuning for the ``'session'`` strategy's search for balanced held-out session sets (max attempts, plus how much / how often the balance tolerance is relaxed).

   :Options:
      * ``session_split_max_attempts`` — a positive integer
      * ``session_split_widen_step`` — a positive number
      * ``session_split_widen_every`` — a positive integer

      (none validated)
   :Default: ``50000``, ``0.02`` and ``1000``
   :Notes: these three, like ``session_split_initial_tolerance``, are passed from the run's settings to every splitter (outer folds, the inner-CV tuners and the CNN); none of the splitters carries a default of its own.

.. _mval-stratification:

``stratification_max_bins`` / ``stratification_min_events_per_bin``
   The continuous bout-parameter target is stratified across folds by cutting it into quantile bins: at most ``stratification_max_bins`` bins, and in ``BoutParameterPipeline`` no more than one bin per ``stratification_min_events_per_bin`` events, so every stratum holds enough events to be split.

   :Options:
      * ``stratification_max_bins`` — an integer ``>= 2``
      * ``stratification_min_events_per_bin`` — a positive integer

      (neither validated)
   :Default: ``10`` and ``20``
   :Notes: the bout-parameter selection caps its bins at ``stratification_max_bins`` with one bin per ``n_cv_folds`` events. Two bins is the structural floor (a single stratum stratifies nothing).

**kinematic_features** — which behavioral predictors enter the feature zoo.

.. code-block:: json

    "kinematic_features": {
        "egocentric": ["speed", "neck_elevation", "allo_roll", "allo_pitch",
                       "ego_yaw", "back_pitch", "back_yaw", "tail_curvature"],
        "dyadic_pose": ["head-head", "allo_yaw-head", "head-allo_yaw",
                        "allo_pitch-head", "head-allo_pitch"],
        "dyadic_engagement": [],
        "dyadic_pose_symmetric": false,
        "include_1st_derivatives": false,
        "include_2nd_derivatives": false,
        "abs_features": ["allo_roll", "allo_yaw-nose", "nose-allo_yaw",
                         "allo_yaw-head", "head-allo_yaw",
                         "allo_yaw-TTI", "TTI-allo_yaw"],
        "smooth_abs_features": {"ego_yaw": 1.0, "back_yaw": 0.5}
    }

.. list-table:: ``kinematic_features`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`egocentric <mkf-egocentric>`
     - 8 features (see JSON above)
     - list of single-mouse feature names
   * - :ref:`dyadic_pose <mkf-dyadic-pose>`
     - 5 features (see JSON above)
     - list of dyadic-pose feature names
   * - :ref:`dyadic_engagement <mkf-dyadic-engagement>`
     - ``[]``
     - list from ``orofacial-sei``, ``anogenital-sei``, or empty
   * - :ref:`dyadic_pose_symmetric <mkf-dyadic-pose-symmetric>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`include_1st_derivatives <mkf-derivatives>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`include_2nd_derivatives <mkf-derivatives>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`abs_features <mkf-abs-features>`
     - 7 suffixes (see JSON above)
     - list of feature suffixes, or empty
   * - :ref:`smooth_abs_features <mkf-smooth-abs-features>`
     - ``{"ego_yaw": 1.0, "back_yaw": 0.5}``
     - object of suffix → ``ε`` > 0, or ``{}``

.. _mkf-egocentric:

``egocentric``
   Single-mouse posture / movement features; each name is kept for both mice (``<mouse>.<feature>``).

   :Options: a list of single-mouse feature names present in the behavioral-features CSVs: ``speed``, ``acceleration``, ``neck_elevation``, ``allo_roll``, ``allo_pitch``, ``allo_yaw``, ``ego_yaw``, ``back_pitch``, ``back_yaw``, ``body_dir``, ``tail_curvature``, ``spaceX``, ``spaceY``, ``spaceZ``
   :Default: ``["speed", "neck_elevation", "allo_roll", "allo_pitch", "ego_yaw", "back_pitch", "back_yaw", "tail_curvature"]``
   :Notes: a name absent from a session's CSV is skipped silently.

.. _mkf-dyadic-pose:

``dyadic_pose``
   Relative-pose features between the two mice (``<self>-<other>`` naming).

   :Options: a list of dyadic-pose names present in the CSVs:

      * ``head-head``, ``nose-nose``, ``nose-TTI``, ``TTI-nose``, ``TTI-TTI``
      * the angle pairs ``allo_yaw-<node>`` / ``<node>-allo_yaw`` and ``allo_pitch-<node>`` / ``<node>-allo_pitch``, with ``<node>`` one of ``head``, ``nose``, ``TTI``
   :Default: ``["head-head", "allo_yaw-head", "head-allo_yaw", "allo_pitch-head", "head-allo_pitch"]``
   :Notes:
      * ``head-head`` is the inter-partner distance between the two Head nodes (skull centroids); ``allo_yaw-head`` / ``allo_pitch-head`` are the egocentric yaw / pitch of the partner's Head seen from the predictor mouse's head frame, and ``head-allo_yaw`` / ``head-allo_pitch`` the reverse direction.
      * The Nose- and TTI-anchored equivalents (``allo_yaw-nose``, ``allo_pitch-TTI``, ...) remain available in every behavioral-features CSV.
      * A name absent from the CSV is skipped silently.

.. _mkf-dyadic-engagement:

``dyadic_engagement``
   Social-engagement features (e.g. ``orofacial-sei``).

   :Options: a list from ``orofacial-sei``, ``anogenital-sei``, or empty
   :Default: ``[]``
   :Notes: only the target-toward-partner orientation is kept.

.. _mkf-dyadic-pose-symmetric:

``dyadic_pose_symmetric``
   Whether to include both ``A-B`` and ``B-A`` orientations of each dyadic-pose feature.

   :Options:
      * ``true`` — include both orientations
      * ``false`` — one direction of each angle pair is dropped by the predictor-mouse rule
   :Default: ``false``

.. _mkf-derivatives:

``include_1st_derivatives`` / ``include_2nd_derivatives``
   Also add the velocity / acceleration of each feature.

   :Options: ``true`` or ``false`` each
   :Default: ``false`` and ``false``
   :Notes: ``speed`` and ``acceleration`` never get derivatives.

.. _mkf-abs-features:

``abs_features``
   Feature suffixes folded to their plain absolute value ``|x|`` before pooled z-scoring (for signed angles whose sign is not behaviorally meaningful).

   :Options: a list of feature suffixes (matched on the part after the last ``.``), or empty
   :Default: ``["allo_roll", "allo_yaw-nose", "nose-allo_yaw", "allo_yaw-head", "head-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw"]``
   :Notes:
      * Read by every pipeline that z-scores across sessions; suffixes absent from the data have no effect.
      * The pooled mean and standard deviation come from the development sessions only (the held-out sessions are drawn first, then excluded from the statistics, then z-scored with them), and the manifold pipeline's inverse-density KDE weights are likewise fit on the development targets only.

.. _mkf-smooth-abs-features:

``smooth_abs_features``
   A *smooth* magnitude fold, ``sqrt(x² + ε²)``, where the mapped value is the per-feature ``ε`` **in the feature's own units** (degrees for the angles), not a smoothing width in frames and not a Gaussian σ.

   :Options: an object mapping a feature suffix to a positive ``ε`` in the feature's units, or empty ``{}``
   :Default: ``{"ego_yaw": 1.0, "back_yaw": 0.5}``
   :Notes: it is the differentiable counterpart of ``abs_features``: identical to ``|x|`` away from zero, but rounded near it, which avoids the kink that a plain absolute value puts at ``x = 0``. A feature listed in both dicts takes this branch, since ``smooth_abs_features`` has priority (see ``zscore_different_sessions_together``).

**vocal_features** — which vocal predictors enter the zoo, and the acoustic-manifold definition.

.. code-block:: json

    "vocal_features": {
        "vocal_predictor_type": "categories_rate",
        "vocal_predictor_classes": [["usv"]],
        "vocal_predictor_partner_only": true,
        "vocal_predictor_smoothing_sd": 1,
        "usv_category_column_name": "qlvm_category",
        "exclude_noise_usvs": true,
        "usv_manifold_column_names": ["qlvm1", "qlvm2"],
        "usv_manifold_metric": "torus",
        "usv_manifold_period": 1.0,
        "usv_manifold_weight_clip_percentile": 95.0,
        "usv_manifold_min_region_events": 20,
        "usv_manifold_selection_score": "macro",
        "freeze_selection_kappa": true,
        "usv_manifold_dcor": {
            "n_subsample": 2500,
            "n_repeats": 3,
            "n_repeats_fold_gate": 1
        },
        "usv_manifold_geodesic_metrics": {
            "compute": true,
            "grid_n_per_dim": 40,
            "graph_k": 8,
            "density_exponent": 1.0,
            "pullback_metric": true,
            "pullback_condition_quantile": 0.5
        }
    }

.. list-table:: ``vocal_features`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`vocal_predictor_type <mvf-vocal-predictor-type>`
     - ``'categories_rate'``
     - ``'pooled_binary'``, ``'pooled_rate'``, ``'categories_rate'``, ``'all_rate'``, ``null`` / ``false``
   * - :ref:`vocal_predictor_classes <mvf-vocal-predictor-classes>`
     - ``[["usv"]]``
     - list of class lists from ``"usv"``, ``"squeak"``, ``"overlap"``
   * - :ref:`vocal_predictor_partner_only <mvf-vocal-predictor-partner-only>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`vocal_predictor_smoothing_sd <mvf-vocal-predictor-smoothing-sd>`
     - ``1``
     - frames >= 0, or ``null``
   * - :ref:`usv_category_column_name <mvf-usv-category-column-name>`
     - ``'qlvm_category'``
     - ``'qlvm_category'``, ``null``
   * - :ref:`exclude_noise_usvs <mvf-exclude-noise-usvs>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`usv_manifold_column_names <mvf-usv-manifold-column-names>`
     - ``['qlvm1', 'qlvm2']``
     - two coordinate columns of one map
   * - :ref:`usv_manifold_metric <mvf-usv-manifold-metric>`
     - ``'torus'``
     - ``'torus'``, ``'euclidean'``
   * - :ref:`usv_manifold_period <mvf-usv-manifold-period>`
     - ``1.0``
     - number > 0
   * - :ref:`usv_manifold_weight_clip_percentile <mvf-usv-manifold-weight-clip-percentile>`
     - ``95.0``
     - percentile in ``[0, 100]``
   * - :ref:`usv_manifold_min_region_events <mvf-usv-manifold-min-region-events>`
     - ``20``
     - integer >= 0
   * - :ref:`usv_manifold_selection_score <mvf-usv-manifold-selection-score>`
     - ``'macro'``
     - ``'macro'``, ``'micro'``
   * - :ref:`freeze_selection_kappa <mvf-freeze-selection-kappa>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`usv_manifold_dcor.n_subsample <mvf-usv-manifold-dcor>`
     - ``2500``
     - integer > 0
   * - :ref:`usv_manifold_dcor.n_repeats <mvf-usv-manifold-dcor>`
     - ``3``
     - integer > 0
   * - :ref:`usv_manifold_dcor.n_repeats_fold_gate <mvf-usv-manifold-dcor>`
     - ``1``
     - integer > 0
   * - :ref:`usv_manifold_geodesic_metrics.compute <mvf-usv-manifold-geodesic-metrics>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`usv_manifold_geodesic_metrics.grid_n_per_dim <mvf-usv-manifold-geodesic-metrics>`
     - ``40``
     - integer > 0
   * - :ref:`usv_manifold_geodesic_metrics.graph_k <mvf-usv-manifold-geodesic-metrics>`
     - ``8``
     - integer > 0, below the node count
   * - :ref:`usv_manifold_geodesic_metrics.density_exponent <mvf-usv-manifold-geodesic-metrics>`
     - ``1.0``
     - number >= 0
   * - :ref:`usv_manifold_geodesic_metrics.pullback_metric <mvf-usv-manifold-geodesic-metrics>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`usv_manifold_geodesic_metrics.pullback_condition_quantile <mvf-usv-manifold-geodesic-metrics>`
     - ``0.5``
     - number in ``[0, 1]``

.. _mvf-vocal-predictor-type:

``vocal_predictor_type``
   How each vocal predictor trace is built.

   :Options:
      * ``'pooled_binary'`` — a per-frame event indicator, ``<name>_event``
      * ``'pooled_rate'`` — a smoothed rate, ``<name>_rate``
      * ``'categories_rate'`` — one smoothed rate per ``usv_category_column_name`` category, ``<name>_cat_<k>``
      * ``'all_rate'`` — the pooled rate plus the per-category rates
      * ``null`` / ``false`` — no vocal predictors (a falsy value builds no vocal predictors)
   :Default: ``'categories_rate'`` (over ``qlvm_category``)
   :Notes:
      * **Validation:** any other string raises ``ValueError``.
      * **Squeaks:** pure squeaks carry no QLVM category, so a trace whose class list holds pure squeaks is one ``<name>_rate`` under ``'categories_rate'`` and ``'all_rate'``.
      * **Label column:** ``'categories_rate'`` and ``'all_rate'`` need a label column: with ``usv_category_column_name`` ``null``, or naming a column a session's summary lacks (e.g. a summary whose categories ``assign-qlvm-categories`` has not written yet), they raise a ``ValueError`` naming the setting (before any session is loaded, for the ``null`` case) instead of silently building no category traces. The pooled modes never read a label column.

.. _mvf-vocal-predictor-classes:

``vocal_predictor_classes``
   Which vocalizations the predictor traces hold: a list of class lists, each one trace, from the same three classes as ``onset_target_classes``.

   :Options: a non-empty list of non-empty lists, each holding distinct classes from ``"usv"``, ``"squeak"`` and ``"overlap"``, with no two lists giving the same trace name (all checked, ``ValueError`` otherwise). Examples:

      * ``[["usv"]]`` — gives ``other.usv_cat_<k>`` (or ``other.usv_rate``); the default
      * ``[["squeak", "overlap"]]`` — gives ``other.squeak_rate``
      * ``[["usv", "overlap"], ["squeak"]]`` — gives a USV and a squeak trace side by side
   :Default: ``[["usv"]]``
   :Notes:
      * **Naming:** a trace is named by what it holds: a list holding ``"usv"`` (alone or with ``"overlap"``) is a ``usv`` trace, one holding ``"squeak"`` (alone or with ``"overlap"``) a ``squeak`` trace, ``["overlap"]`` an ``overlap`` trace, and one holding both ``"usv"`` and ``"squeak"`` an ``all_vocalizations`` trace.
      * **Sharing:** lists may share classes (an overlap segment holds a USV and a squeak, so it belongs in both), but two lists that give the same name are refused. The exact lists are recorded in ``_input_metadata`` (``vocal_predictor_classes``), since ``["usv"]`` and ``["usv", "overlap"]`` both give a ``usv`` trace.
      * **Timing:** each trace times its rows as the onset targets do (a squeak by its own extent, an overlap segment by the squeak extent, the segment or their union as the list asks about squeaks, USVs or both).
      * **Own traces:** the target mouse's own pooled traces (any name) are never used as its predictors.
      * **Never null:** never ``null``, even with vocal predictors off (switch them off with ``vocal_predictor_type``).

.. _mvf-vocal-predictor-partner-only:

``vocal_predictor_partner_only``
   Whether to ingest only the *partner's* vocal traces as predictors (not the target mouse's own vocal history).

   :Options:
      * ``true`` — partner's traces only
      * ``false`` — both mice; the target's own pooled traces are still dropped
   :Default: ``true``

.. _mvf-vocal-predictor-smoothing-sd:

``vocal_predictor_smoothing_sd``
   Gaussian σ (frames) applied to the USV-rate predictor traces.

   :Options: a non-negative number of frames; ``0`` or ``null`` leaves the rate traces unsmoothed
   :Default: ``1``
   :Notes: the ``_event`` traces are never smoothed; not range-checked.

.. _mvf-usv-category-column-name:

``usv_category_column_name``
   The USV-catalog column defining categories.

   :Options:
      * ``'qlvm_category'`` — the content-ridge categories R-1 … R-k (4 in production) of the regular map's ``qlvm1`` / ``qlvm2`` torus that ``assign-qlvm-categories`` writes; the only such column the pipeline writes
      * the name of another integer label column present in every summary
      * ``null`` — label-free paths only
   :Default: ``'qlvm_category'``
   :Notes:
      * **One column:** ``'qlvm_category'`` is the summary's only category column (no coarse level, and no category column of the conditional maps: the categories are defined on the regular map, so every map's analyses — ``qlvm`` and the conditional ``qlvm_duration``, ``qlvm_entropy``, ``qlvm_bandwidth`` and ``qlvm_loudness`` alike — label a USV with its ``qlvm_category``). Labels are integers ``1..k``, 1 the category with the most USVs.
      * **null:** ``null`` runs the label-free paths only: every label-dependent path fails clearly on ``null`` — the multinomial and binomial category pipelines, ``onset_target_category`` in ``'individual'`` mode, and the ``'categories_rate'`` / ``'all_rate'`` predictors raise ``QLVM category labels are not available; set vocal_features.usv_category_column_name to an existing label column`` before loading anything — while the continuous manifold pipeline runs without it (no category packet).
      * **Validation:** a column that is set must exist in every summary; a set column missing from a summary raises.

.. _mvf-exclude-noise-usvs:

``exclude_noise_usvs``
   Whether to drop the USV segments ``detect-usv-noise`` flagged as holding no vocalization.

   :Options:
      * ``true`` — drop segments with ``noise`` true
      * ``false`` — keep every detection
   :Default: ``true``
   :Notes: a session whose summary lacks the ``noise`` column raises rather than contributing unfiltered detections.

.. _mvf-usv-manifold-column-names:

``usv_manifold_column_names``
   The two catalog columns giving the 2-D manifold position (the ``ContinuousModelingPipeline`` target).

   :Options: exactly two summary column names (a list of any other length raises ``ValueError``) naming one map's coordinates:

      * ``['qlvm1', 'qlvm2']`` — regular map
      * ``['qlvm_duration1', 'qlvm_duration2']`` — conditional map
      * ``['qlvm_entropy1', 'qlvm_entropy2']`` — conditional map
      * ``['qlvm_bandwidth1', 'qlvm_bandwidth2']`` — conditional map
      * ``['qlvm_loudness1', 'qlvm_loudness2']`` — conditional map
      * ``['qlvm_squeak1', 'qlvm_squeak2']`` — squeak map
   :Default: ``['qlvm1', 'qlvm2']``
   :Notes:
      * **Null coordinates:** USVs the embedding could not place have null coordinates; they are dropped at extraction (with the count printed per session and in total), so the inverse-density KDE, the regressions and the GLM-HMM never see a NaN position.
      * **Squeak map:** the squeak map (``['qlvm_squeak1', 'qlvm_squeak2']``) places the squeak of every squeak-bearing segment, so its onsets are the squeaks' own (``squeak_start``, also for an overlap segment, whose USV begins earlier; a summary without the column raises); every other map keeps the segment onsets.
      * **Run naming:** the columns' map prefix also names the run: the manifold analysis tag is ``manifold_<map>`` (``['qlvm1', 'qlvm2']`` -> ``manifold_qlvm``, ``['qlvm_duration1', 'qlvm_duration2']`` -> ``manifold_qlvm_duration``; ``modeling_utils.manifold_tag_segment``) whether or not a category column is set, and it carries into the modeling input pickle (``modeling_manifold_<map>_<cohort>_<ts>.pkl``), the univariate results, the model-selection step files (``model_selection_continuous_manifold_<map>_<condition>_<split>_step_*.pkl``, named from the input pickle's ``analysis_tag``) and the consolidated artifact (``model_selection_final_<sex>_<condition>_manifold_<map>_<split>_<ts>.pkl``).
      * **Older results:** results written before this naming carry ``manifold_qlvm_category`` (the category column, the same for every map); they are read as before, and the consolidator never appends the category column to a ``manifold_<map>`` tag.

.. _mvf-usv-manifold-metric:

``usv_manifold_metric``
   The distance on the manifold.

   :Options:
      * ``'torus'`` — wrap-aware distance
      * ``'euclidean'`` — plane distance
   :Default: ``'torus'``
   :Notes: any other value raises ``ValueError``.

.. _mvf-usv-manifold-period:

``usv_manifold_period``
   The wrap period for the ``'torus'`` metric.

   :Options: a strictly positive number (a non-positive value raises ``ValueError``, also on euclidean)
   :Default: ``1.0``
   :Notes: ``1.0`` matches the QLVM coordinates, which lie in ``[0, 1)``.

.. _mvf-usv-manifold-weight-clip-percentile:

``usv_manifold_weight_clip_percentile``
   The percentile at which the per-USV inverse-density KDE weights of the manifold extraction are capped (taken over the development rows), so a single isolated USV in an almost empty corner of the manifold cannot dominate the weighted loss; the weights are then rescaled to a development-row mean of 1.

   :Options: a percentile in ``[0, 100]`` (outside it ``numpy.percentile`` raises); ``100`` disables the cap
   :Default: ``95.0``
   :Notes: the value is recorded in ``_input_metadata.analysis_specific.manifold_weight_clip_percentile``.

.. _mvf-usv-manifold-min-region-events:

``usv_manifold_min_region_events``
   The minimum number of labelled events an acoustic region (a QLVM category) must contain to enter the **macro** (region-balanced) von Mises average and the region-weighted MAE; sparser regions are dropped from those balanced statistics so a single under-sampled corner cannot dominate them.

   :Options: a non-negative integer (not validated); ``0`` keeps every region
   :Default: ``20``
   :Notes: ignored on euclidean and when no region labels are present. Without region labels (``usv_category_column_name`` ``null``, or a pickle extracted without them) the torus ``'macro'`` score falls back to the pooled score and the equal-region fit reweighting to uniform weights; the univariate and selection runs print a ``WARNING`` saying so, rather than substituting silently.

.. _mvf-usv-manifold-selection-score:

``usv_manifold_selection_score``
   On the ``'torus'`` metric, which von Mises log-score the forward selection ranks on.

   :Options:
      * ``'macro'`` — the region-balanced ``vm_logscore``
      * ``'micro'`` — the event-weighted ``vm_logscore_pooled`` twin
      * key absent — resolves to ``'macro'``
   :Default: ``'macro'``
   :Notes: any other value raises ``ValueError``. Both scores are always logged per candidate, so this only changes which column drives the greedy ranking and the acceptance gate — the candidate pool, the region-reweighted fit, and every other reported metric are identical — making a macro-vs-micro selection comparison a one-key flip. Ignored on euclidean (which always ranks on ``dcor_xy``).

.. _mvf-freeze-selection-kappa:

``freeze_selection_kappa``
   On a torus manifold run, whether the von Mises concentration ``kappa`` is fitted once or per scored prediction set.

   :Options:
      * ``true`` — fits ``kappa`` once on the development set and reuses it for every score (the univariate screen, baseline, fitted candidates, held-out, and the acceptance-gate bootstrap), so every margin sits on one dispersion scale and a fold predicted worse than the marginal scores below the flat-torus density instead of being clamped to it
      * ``false`` — refits ``kappa`` per scored prediction set (see the *Frozen von Mises concentration* note below)
   :Default: ``true``
   :Notes: read as a boolean. Ignored on euclidean.

.. _mvf-usv-manifold-dcor:

``usv_manifold_dcor``
   The subsampling of the wrap-aware distance correlation ``dcor_xy``, the euclidean manifold selection score (unused on the torus, which scores with the von Mises log-likelihood). Distance correlation is ``O(n²)`` in memory and time, so each evaluation averages ``n_repeats`` random subsamples of ``n_subsample`` events.

   :Options:
      * ``n_subsample`` — a positive integer; the subsample is ``min(n_subsample, n)`` events
      * ``n_repeats`` — a positive integer (``0`` returns ``NaN``)
      * ``n_repeats_fold_gate`` — a positive integer (``0`` returns ``NaN``): the repeat count of the per-fold selection-gate score

      (none is validated)
   :Default: ``2500``, ``3`` and ``1``
   :Notes: a 2500-point distance matrix holds ~6.25e6 entries, small enough to stay fast and large enough for a stable estimate on production folds. The per-fold selection-gate score uses ``n_repeats_fold_gate`` repeats because a single fold's events sit under the subsample cap, so extra repeats would recompute the identical full-sample value.

.. _mvf-usv-manifold-geodesic-metrics:

``usv_manifold_geodesic_metrics``
   The analysis-only *reference-map* geometry for the two torus **geodesic** prediction-error columns (``density_geodesic_mae``, ``pullback_geodesic_mae``), reported per fold alongside the flat-torus MAE on the ``'torus'`` metric (both ``NaN`` on euclidean).

   :Options:
      * ``compute`` — ``true`` / ``false``; toggles the whole block (the whole block may also be absent, which skips the geodesic columns)
      * ``grid_n_per_dim`` — a positive integer (not validated): the resolution of the regular torus grid the all-pairs geodesic distance matrices are precomputed on once (``40`` → a 40×40 node lattice, so per-event errors are cheap snap-to-grid look-ups); the grid has ``grid_n_per_dim**2`` nodes
      * ``graph_k`` — a positive integer below the node count (not validated): the number of wrap-aware nearest neighbours per node in the k-NN graph the shortest paths run over
      * ``density_exponent`` — a non-negative number (not validated): the inverse-aggregate-posterior-density exponent ``α`` weighting the density-ratio geodesic; ``0`` recovers the flat graph metric, larger values push paths harder through dense regions
      * ``pullback_metric`` — ``true`` / ``false``: switches the decoder-Jacobian pullback geodesic on
      * ``pullback_condition_quantile`` — a number in ``[0, 1]`` (outside it raises ``ValueError``): fixes the conditioning value of a conditional map's decoder
   :Default: ``true``, ``40``, ``8``, ``1.0``, ``true`` and ``0.5``
   :Notes:
      **Pullback decoder:** the frozen decoder whose Jacobian defines the pullback metric ``G = JᵀJ`` is not a setting but the production cell of the map the manifold coordinates come from, ``os_utils.qlvm_map_cell_directory(<map>)`` with ``<map>`` the map prefix of the input pickle's recorded ``usv_manifold_column_names`` (``_input_metadata.analysis_specific``; a pickle without that record falls back to the current setting):

      * ``qlvm1`` / ``qlvm2`` -> the regular cell ``.../spectrograms/qlvm/qlvm``
      * ``qlvm_duration`` / ``qlvm_entropy`` / ``qlvm_bandwidth`` / ``qlvm_loudness`` -> their cells ``.../spectrograms/qlvm/<map>`` (``QLVM_PRODUCTION_MODEL_CELLS`` under ``QLVM_MODEL_PACKAGE_ROOT``)
      * ``qlvm_squeak1`` / ``qlvm_squeak2`` -> the squeak cell ``QLVM_SQUEAK_PRODUCTION_CELL`` under ``QLVM_SQUEAK_PACKAGE_ROOT``

      Each is read without torch from its ``checkpoint.tar``. A pullback metric of one map's coordinates under another map's decoder would measure distances on a torus the coordinates do not live on. Manifold columns that name no QLVM map with ``pullback_metric`` ``true`` are a settings error. The cell comes from the code constants so the CLI / GUI re-keying of experimenter folders in the settings can never point it at another folder; a ``decoder_model_cell_directory`` (or the retired ``decoder_weights_npz_path``) key left in older settings is ignored.

      **Conditioning value:** ``pullback_condition_quantile`` (default ``0.5``) fixes the conditioning value of a conditional map's decoder, which decodes a torus position *and* a per-USV conditioning value ``c`` (duration, spectral entropy, bandwidth or loudness): every grid node is decoded at the same ``c``, that quantile of the cell's training corpus's conditioning distribution (read from the cell's ``condition_bins.npz`` bin edges and per-bin training counts, linear inside a bin, then decoded by the cell's own ``condition.decode`` rule: clamped for ``'exact'``, snapped to the decode grid for ``'grid'``; ``processing.qlvm_latents.condition_quantile_value``), so ``G`` is one Riemannian metric of the torus. ``0.5`` measures distances as decoded-spectrogram change for the corpus median USV (production medians: duration ``c`` 0.244, entropy 0.595, bandwidth 0.245, loudness 0.545); the value depends only on the cell, not on which USVs a run holds. The regular and squeak decoders take no conditioning value and ignore it.

      **Degrading:** ``pullback_metric`` ``false`` (or an unreadable decoder) degrades ``pullback_geodesic_mae`` to ``NaN`` and the run proceeds.

**diagnostics** — the predictor-collinearity and predictor-timescale audits (rendered in :ref:`Predictor diagnostics <modeling-diagnostics>`).

.. code-block:: json

    "diagnostics": {
        "collinearity_audit": false,
        "collinearity_concern_threshold": 0.7,
        "collinearity_exclude_threshold": 0.85,
        "timescale_audit": false,
        "binary_decision_threshold": 0.5,
        "ece_n_bins": 10,
        "summary_star_alpha": 0.01,
        "timescale_max_lag_seconds": 10.0,
        "timescale_n_shuffles": 1000,
        "timescale_shuffle_range": [20, 60],
        "timescale_signal_floor_seconds": 0.5,
        "timescale_signal_min_run_seconds": 0.2
    }

.. list-table:: ``diagnostics`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`collinearity_audit <mdg-audits>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`collinearity_concern_threshold <mdg-collinearity-thresholds>`
     - ``0.7``
     - number in ``[0, 1]``
   * - :ref:`collinearity_exclude_threshold <mdg-collinearity-thresholds>`
     - ``0.85``
     - number in ``[0, 1]``
   * - :ref:`timescale_audit <mdg-audits>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`binary_decision_threshold <mdg-binary-decision-threshold>`
     - ``0.5``
     - probability in ``[0, 1]``
   * - :ref:`ece_n_bins <mdg-ece-n-bins>`
     - ``10``
     - integer > 0
   * - :ref:`summary_star_alpha <mdg-summary-star-alpha>`
     - ``0.01``
     - level in ``(0, 1)``
   * - :ref:`timescale_max_lag_seconds <mdg-timescale-max-lag-seconds>`
     - ``10.0``
     - seconds > 0
   * - :ref:`timescale_n_shuffles <mdg-timescale-null>`
     - ``1000``
     - integer > 0
   * - :ref:`timescale_shuffle_range <mdg-timescale-null>`
     - ``[20, 60]``
     - ``[min, max]`` seconds, ``0 < min < max``
   * - :ref:`timescale_signal_floor_seconds <mdg-timescale-signal>`
     - ``0.5``
     - seconds >= 0
   * - :ref:`timescale_signal_min_run_seconds <mdg-timescale-signal>`
     - ``0.2``
     - seconds > 0

.. _mdg-audits:

``collinearity_audit`` / ``timescale_audit``
   Enable each audit during extraction.

   :Options: ``true`` or ``false`` each
   :Default: ``false`` and ``false``

.. _mdg-collinearity-thresholds:

``collinearity_concern_threshold`` / ``collinearity_exclude_threshold``
   Pearson ``|ρ|`` cutoffs for the collinearity audit: pairs at or above the concern threshold are flagged ``concern``, and pairs at or above the exclude threshold are flagged ``exclude`` (effectively redundant).

   :Options: each a number in ``[0, 1]``, the concern threshold below the exclude threshold (neither checked)
   :Default: ``0.7`` and ``0.85``
   :Notes: the audit only reports the pairs; it removes no feature.

.. _mdg-binary-decision-threshold:

``binary_decision_threshold``
   Probability cutoff for turning a predicted probability into a 0/1 label (a positive label is assigned when ``p >= threshold``).

   :Options: a probability in ``[0, 1]`` (not validated)
   :Default: ``0.5``
   :Notes: shared by every binary onset / category selector and pipeline.

.. _mdg-ece-n-bins:

``ece_n_bins``
   Histogram bin count for the Expected Calibration Error diagnostic.

   :Options: a positive integer (not validated)
   :Default: ``10``
   :Notes: shared by every calibration computation (the one adaptive site scales its own bin count to the sample size). It is read from the run's settings at every call site (including the multinomial inner-CV tuner's ``'ece'`` score).

.. _mdg-summary-star-alpha:

``summary_star_alpha``
   The console-only significance marker of the manifold univariate runner's end-of-run summary: a metric is starred ``***`` when its one-sided Wilcoxon signed-rank test over folds against the shuffle null (or the empirical-density draw) gives ``p`` below this level.

   :Options: a level in ``(0, 1)`` (not validated)
   :Default: ``0.01``
   :Notes: it prints only; no stored result depends on it.

.. _mdg-timescale-max-lag-seconds:

``timescale_max_lag_seconds``
   Maximum lag examined for the ACF (autocorrelation function) / cross-correlation horizons.

   :Options: a positive number of seconds (not validated)
   :Default: ``10.0``

.. _mdg-timescale-null:

``timescale_n_shuffles`` / ``timescale_shuffle_range``
   Number of circular-shift surrogates and the ``(min, max)`` shift range (seconds) for the null envelope.

   :Options:
      * ``timescale_n_shuffles`` — a positive integer (not validated)
      * ``timescale_shuffle_range`` — a two-element list ``[min, max]`` of seconds with ``0 < min < max`` (checked, ``ValueError`` otherwise)
   :Default: ``1000`` and ``[20, 60]``

.. _mdg-timescale-signal:

``timescale_signal_floor_seconds`` / ``timescale_signal_min_run_seconds``
   Where the cross-correlation significance marker may sit: crossings of the null envelope at lags below ``timescale_signal_floor_seconds`` are ignored, and the marker is placed at the end of the longest sign-consistent above-null run on the positive lag axis that lasts at least ``timescale_signal_min_run_seconds``.

   :Options:
      * ``timescale_signal_floor_seconds`` — a non-negative number of seconds
      * ``timescale_signal_min_run_seconds`` — a strictly positive number of seconds

      (both checked, ``ValueError`` otherwise)
   :Default: ``0.5`` and ``0.2``

**hyperparameters** — per-engine model tuning, grouped into four sub-blocks:
``deep_learning`` (the CNN), ``linear_models`` (the two JAX linear models),
``classical`` (the ``'pygam'`` / ``'sklearn'`` engines) and ``basis_functions``
(the ``'sklearn'`` engine's temporal bases). Every key is read strictly (a missing
key raises ``KeyError``; no hard-coded fallback), so each block is enumerated in
full below.

**hyperparameters.deep_learning.cnn_continuous** — the 1-D ResNet for the
continuous manifold target, consumed by ``NeuralContinuousCNNRunner`` (see
:ref:`CNN modeling <modeling-cnn>`):

.. code-block:: json

    "cnn_continuous": {
        "use_kde_weights": true,
        "loss_function": "huber",
        "huber_delta": 1.0,
        "learning_rate": 0.0005,
        "block_channels": [32, 64, 128, 256],
        "warmup_fraction": 0.1,
        "weight_decay": 0.001,
        "weight_decay_exclude_output_head": true,
        "cnn_torus_output_encoding": "sin_cos",
        "epochs": 301,
        "batch_size": 32,
        "kernel_size": 15,
        "use_inception_kernels": false,
        "inception_kernel_sizes": [3, 15, 31],
        "dropout_rate": 0.3,
        "se_reduction": 4,
        "se_activation": "sigmoid",
        "hidden_dim": 128,
        "act_func": "relu",
        "bn_momentum": 0.99,
        "bn_eps": 1e-5,
        "grid_size": 25,
        "samples_per_cell": 40,
        "grid_balance_alpha": 0.5,
        "prediction_head_scale_margin": 1.1,
        "use_scheduler": true,
        "patience": 12,
        "null_patience": 3,
        "early_stopping_eval_every_epochs": 5,
        "early_stopping_min_delta": 0.001,
        "validation_session_fraction": 0.15,
        "warp_range": 0.1,
        "n_folds": 10,
        "permutation_iterations": 10,
        "importance_snr_threshold": 3.0,
        "use_kinematic_masking": true,
        "masking_prob": 0.15,
        "masking_length_frames": 60,
        "use_hybrid_flatten": false,
        "saliency": {
            "enable": true,
            "segmentation": "category",
            "alpha": 0.5,
            "radius_mode": "adaptive",
            "center_min_points_per_label": 20,
            "center_max_points_per_label": 30000,
            "center_grid_resolution": 200,
            "center_grid_margin_fraction": 0.05,
            "center_rng_seed": 0
        }
    }

.. list-table:: ``cnn_continuous`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`use_kde_weights <mhp-cnn-use-kde-weights>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`loss_function <mhp-cnn-loss-function>`
     - ``'huber'``
     - ``'huber'``, or any other string (squared error)
   * - :ref:`huber_delta <mhp-cnn-loss-function>`
     - ``1.0``
     - float > 0, manifold units
   * - :ref:`learning_rate <mhp-cnn-learning-rate>`
     - ``0.0005``
     - float > 0
   * - :ref:`use_scheduler <mhp-cnn-learning-rate>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`warmup_fraction <mhp-cnn-learning-rate>`
     - ``0.1``
     - fraction in ``[0, 1)``
   * - :ref:`block_channels <mhp-cnn-block-channels>`
     - ``[32, 64, 128, 256]``
     - non-empty list of integers > 0
   * - :ref:`weight_decay <mhp-cnn-weight-decay>`
     - ``0.001``
     - float >= 0
   * - :ref:`weight_decay_exclude_output_head <mhp-cnn-weight-decay>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`cnn_torus_output_encoding <mhp-cnn-cnn-torus-output-encoding>`
     - ``'sin_cos'``
     - ``'sin_cos'``, ``'raw'``
   * - :ref:`epochs <mhp-cnn-epochs>`
     - ``301``
     - integer >= 1
   * - :ref:`batch_size <mhp-cnn-epochs>`
     - ``32``
     - integer > 0
   * - :ref:`kernel_size <mhp-cnn-kernel-size>`
     - ``15``
     - integer > 0 (frames)
   * - :ref:`use_inception_kernels <mhp-cnn-kernel-size>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`inception_kernel_sizes <mhp-cnn-kernel-size>`
     - ``[3, 15, 31]``
     - non-empty list of integers > 0
   * - :ref:`dropout_rate <mhp-cnn-dropout-rate>`
     - ``0.3``
     - fraction in ``[0, 1)``
   * - :ref:`se_reduction <mhp-cnn-se-reduction>`
     - ``4``
     - integer > 0
   * - :ref:`se_activation <mhp-cnn-se-reduction>`
     - ``'sigmoid'``
     - ``'sigmoid'``, or any other string (hard sigmoid)
   * - :ref:`hidden_dim <mhp-cnn-hidden-dim>`
     - ``128``
     - integer > 0
   * - :ref:`act_func <mhp-cnn-hidden-dim>`
     - ``'relu'``
     - ``'relu'``, or any other string (GELU)
   * - :ref:`bn_momentum <mhp-cnn-bn-momentum>`
     - ``0.99``
     - float in ``(0, 1)``
   * - :ref:`bn_eps <mhp-cnn-bn-momentum>`
     - ``1e-5``
     - small float > 0
   * - :ref:`grid_size <mhp-cnn-grid-size>`
     - ``25``
     - integer > 0
   * - :ref:`samples_per_cell <mhp-cnn-grid-size>`
     - ``40``
     - integer > 0
   * - :ref:`grid_balance_alpha <mhp-cnn-grid-size>`
     - ``0.5``
     - float in ``[0, 1]``
   * - :ref:`prediction_head_scale_margin <mhp-cnn-prediction-head-scale-margin>`
     - ``1.1``
     - float >= 1
   * - :ref:`patience <mhp-cnn-patience>`
     - ``12``
     - integer > 0 (validation checks)
   * - :ref:`null_patience <mhp-cnn-patience>`
     - ``3``
     - integer > 0 (validation checks)
   * - :ref:`early_stopping_eval_every_epochs <mhp-cnn-early-stopping>`
     - ``5``
     - integer > 0
   * - :ref:`early_stopping_min_delta <mhp-cnn-early-stopping>`
     - ``0.001``
     - float >= 0, manifold units
   * - :ref:`validation_session_fraction <mhp-cnn-validation-session-fraction>`
     - ``0.15``
     - fraction in ``(0, 1)``
   * - :ref:`warp_range <mhp-cnn-warp-range>`
     - ``0.1``
     - float in ``[0, 1)``
   * - :ref:`n_folds <mhp-cnn-n-folds>`
     - ``10``
     - integer >= 2
   * - :ref:`permutation_iterations <mhp-cnn-n-folds>`
     - ``10``
     - integer >= 2
   * - :ref:`importance_snr_threshold <mhp-cnn-importance-snr-threshold>`
     - ``3.0``
     - float > 0
   * - :ref:`use_kinematic_masking <mhp-cnn-kinematic-masking>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`masking_prob <mhp-cnn-kinematic-masking>`
     - ``0.15``
     - probability in ``[0, 1]``
   * - :ref:`masking_length_frames <mhp-cnn-kinematic-masking>`
     - ``60``
     - integer > 0 (frames)
   * - :ref:`use_hybrid_flatten <mhp-cnn-use-hybrid-flatten>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`saliency.enable <mhp-cnn-saliency-enable>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`saliency.segmentation <mhp-cnn-saliency-enable>`
     - ``'category'``
     - ``'category'`` only
   * - :ref:`saliency.alpha <mhp-cnn-saliency-alpha>`
     - ``0.5``
     - float > 0
   * - :ref:`saliency.radius_mode <mhp-cnn-saliency-alpha>`
     - ``'adaptive'``
     - ``'adaptive'``, ``'uniform'``
   * - :ref:`saliency.center_min_points_per_label <mhp-cnn-saliency-centers>`
     - ``20``
     - integer > 0, <= max
   * - :ref:`saliency.center_max_points_per_label <mhp-cnn-saliency-centers>`
     - ``30000``
     - integer > 0, >= min
   * - :ref:`saliency.center_grid_resolution <mhp-cnn-saliency-centers>`
     - ``200``
     - integer > 0
   * - :ref:`saliency.center_grid_margin_fraction <mhp-cnn-saliency-centers>`
     - ``0.05``
     - float >= 0
   * - :ref:`saliency.center_rng_seed <mhp-cnn-saliency-centers>`
     - ``0``
     - any integer

.. _mhp-cnn-use-kde-weights:

``use_kde_weights``
   How each epoch's training rows are drawn.

   :Options:
      * ``true`` — samples ``N`` rows with replacement with probability proportional to the pickle's inverse-density KDE weights ``w`` (rare acoustic regions are over-drawn)
      * ``false`` — uses the grid-balanced sampler instead (``grid_size`` / ``samples_per_cell`` below)
   :Default: ``true``

.. _mhp-cnn-loss-function:

``loss_function`` / ``huber_delta``
   The training loss on the wrap-aware residual (per ``usv_manifold_metric``).

   :Options:
      * ``loss_function`` — ``'huber'`` (quadratic below ``huber_delta``, linear beyond it, so satellite USVs cannot dominate the gradient); any other value is plain squared error (not validated)
      * ``huber_delta`` — a positive float in manifold units, read only with ``'huber'`` (not validated)
   :Default: ``'huber'`` and ``1.0``

.. _mhp-cnn-learning-rate:

``learning_rate`` / ``use_scheduler`` / ``warmup_fraction``
   The peak Adam(W) step size and its schedule.

   :Options:
      * ``learning_rate`` — a positive float (not validated)
      * ``use_scheduler`` — ``true``: the rate starts at 0, warms up linearly over ``warmup_fraction`` of the total steps to the peak and then cosine-decays to 0 over the rest; ``false``: it is constant
      * ``warmup_fraction`` — a fraction in ``[0, 1)``, read only when ``use_scheduler`` is ``true`` (not validated)
   :Default: ``0.0005``, ``true`` and ``0.1``

.. _mhp-cnn-block-channels:

``block_channels``
   Output channels of each residual block (the list length is the network depth; every block halves the time axis).

   :Options: a non-empty list of positive integers, one per block, short enough that the halved time axis stays at least one frame (not validated)
   :Default: ``[32, 64, 128, 256]``

.. _mhp-cnn-weight-decay:

``weight_decay`` / ``weight_decay_exclude_output_head``
   AdamW decoupled weight decay, and whether the output head is exempt from it.

   :Options:
      * ``weight_decay`` — a float ``>= 0``; ``0`` turns decay off (not validated)
      * ``weight_decay_exclude_output_head`` — ``true`` exempts the output head (``dense2``), whose collapse to zero would park the torus ``sin_cos`` head at a constant prediction; ``false`` decays it too
   :Default: ``0.001`` and ``true``

.. _mhp-cnn-cnn-torus-output-encoding:

``cnn_torus_output_encoding``
   The output encoding on torus runs only; ignored on euclidean.

   :Options:
      * ``'sin_cos'`` — predicts a per-axis ``(sin, cos)`` pair and trains on the chord distance
      * ``'raw'`` — predicts the two coordinates directly
   :Default: ``'sin_cos'``
   :Notes: any other value raises a ``ValueError`` on a torus run.

.. _mhp-cnn-epochs:

``epochs`` / ``batch_size``
   The epoch cap per fold and strategy (early stopping usually ends training earlier) and the mini-batch size.

   :Options:
      * ``epochs`` — an integer ``>= 1`` (``0`` raises a ``ValueError``)
      * ``batch_size`` — a positive integer no larger than a fold's training rows, or an epoch has no steps (not validated)
   :Default: ``301`` and ``32``

.. _mhp-cnn-kernel-size:

``kernel_size`` / ``use_inception_kernels`` / ``inception_kernel_sizes``
   The depthwise convolution kernel (frames); ``use_inception_kernels`` ``true`` replaces it with parallel depthwise kernels of the listed sizes, so short and long temporal motifs are extracted side by side.

   :Options:
      * ``kernel_size`` — a positive integer (frames; ``'SAME'`` padding, so odd and even both work; not validated)
      * ``use_inception_kernels`` — ``true`` (parallel kernels of ``inception_kernel_sizes``) or ``false`` (the single ``kernel_size`` kernel)
      * ``inception_kernel_sizes`` — a non-empty list of positive integers, read only when ``use_inception_kernels`` is ``true`` (a list too long for the network's reserved random-key budget raises a ``ValueError``)
   :Default: ``15``, ``false`` and ``[3, 15, 31]``

.. _mhp-cnn-dropout-rate:

``dropout_rate``
   Dropout before the dense head.

   :Options: a fraction in ``[0, 1)``; ``0`` turns dropout off (not validated)
   :Default: ``0.3``

.. _mhp-cnn-se-reduction:

``se_reduction`` / ``se_activation``
   The squeeze-and-excitation channel attention of every block: bottleneck width ``channels // se_reduction`` and gate.

   :Options:
      * ``se_reduction`` — a positive integer no larger than the smallest entry of ``block_channels`` (not validated)
      * ``se_activation`` — ``'sigmoid'``, or any other string for a hard sigmoid
   :Default: ``4`` and ``'sigmoid'``

.. _mhp-cnn-hidden-dim:

``hidden_dim`` / ``act_func``
   The dense head's width and activation.

   :Options:
      * ``hidden_dim`` — a positive integer (not validated)
      * ``act_func`` — ``'relu'``, or any other string for GELU
   :Default: ``128`` and ``'relu'``

.. _mhp-cnn-bn-momentum:

``bn_momentum`` / ``bn_eps``
   The batch-normalisation running-statistics momentum and numerical floor.

   :Options:
      * ``bn_momentum`` — a float in ``(0, 1)`` (not validated)
      * ``bn_eps`` — a small positive float (not validated)
   :Default: ``0.99`` and ``1e-5``

.. _mhp-cnn-grid-size:

``grid_size`` / ``samples_per_cell`` / ``grid_balance_alpha``
   The grid-balanced sampler used when ``use_kde_weights`` is ``false`` (it then also sizes the steps per epoch).

   :Options:
      * ``grid_size`` — a positive integer (not validated)
      * ``samples_per_cell`` — a positive integer (not validated)
      * ``grid_balance_alpha`` — a float in ``[0, 1]`` (not validated): ``0`` gives every populated cell the same quota (perfect grid balance), ``1`` draws in proportion to occupancy (no balancing), and ``0.5`` sits between
   :Default: ``25``, ``40`` and ``0.5``
   :Notes: the manifold is tiled ``grid_size × grid_size`` and every populated cell contributes ``ceil(samples_per_cell · count^alpha / samples_per_cell^alpha)`` rows (at least one), with ``alpha = grid_balance_alpha`` (default ``0.5``), so the dense core cannot monopolise the batches without the rare cells being copied hundreds of times. All three are read only when ``use_kde_weights`` is ``false``.

.. _mhp-cnn-prediction-head-scale-margin:

``prediction_head_scale_margin``
   The network's ``tanh`` output head is scaled to each fold's training-target half-range times this margin, so the bounded head can reach the edge of the training support with 10% to spare instead of saturating exactly at the most extreme training USV.

   :Options: a float ``>= 1``; below ``1`` the head cannot reach the training edge (not validated)
   :Default: ``1.1``

.. _mhp-cnn-patience:

``patience`` / ``null_patience``
   Early-stopping patience on the validation error, counted in validation checks (one every ``early_stopping_eval_every_epochs`` epochs).

   :Options:
      * ``patience`` — for the ``actual`` model; a positive integer, counted in validation checks (not validated)
      * ``null_patience`` — for the label-shuffled ``null`` model (which converges to nothing and is stopped sooner); a positive integer, counted in validation checks (not validated)
   :Default: ``12`` and ``3``

.. _mhp-cnn-early-stopping:

``early_stopping_eval_every_epochs`` / ``early_stopping_min_delta``
   How often the validation error is checked and how large a drop counts as an improvement.

   :Options:
      * ``early_stopping_eval_every_epochs`` — the validation error is computed every this many epochs; a positive integer (not validated)
      * ``early_stopping_min_delta`` — a check counts as an improvement (resetting the patience counter and snapshotting the best weights) only when the error falls by more than this below the best so far; a float ``>= 0`` in manifold units (not validated)
   :Default: ``5`` and ``0.001`` (in manifold units)
   :Notes: the minimum delta means a numerically negligible improvement cannot keep a plateaued run alive.

.. _mhp-cnn-validation-session-fraction:

``validation_session_fraction``
   The share of each fold's training sessions (at least one, drawn with a seed derived from ``random_seed``) whose training events are held back to choose the early-stopping epoch and the best weights.

   :Options: a fraction in ``(0, 1)``; the count it gives is clamped to at least one session and at most all but one (not validated)
   :Default: ``0.15``
   :Notes: the network is fit on the rest, and the test fold is scored exactly once with the chosen weights, so its error is not optimistic. The best CV fold (whose weights are applied to the held-out block) is likewise chosen on the validation error. A fold with fewer than two training sessions raises.

.. _mhp-cnn-warp-range:

``warp_range``
   Temporal-warping augmentation: every sequence in a batch is stretched by a factor drawn uniformly from ``[1 - warp_range, 1 + warp_range]`` (centre-anchored).

   :Options: a float in ``[0, 1)``; ``0`` turns warping off (not validated)
   :Default: ``0.1``

.. _mhp-cnn-n-folds:

``n_folds`` / ``permutation_iterations``
   The number of spatial-CV folds, and the permutation repeats per feature in the permutation-importance phase.

   :Options:
      * ``n_folds`` — an integer ``>= 2`` (not validated)
      * ``permutation_iterations`` — an integer ``>= 2``, since a single repeat has zero spread, so its SNR is set to ``0`` and no feature is listed (not validated)
   :Default: ``10`` and ``10``

.. _mhp-cnn-importance-snr-threshold:

``importance_snr_threshold``
   A feature is listed as significant by the permutation-importance phase when the mean error increase its permutation causes, divided by the standard deviation over the ``permutation_iterations`` repeats, exceeds this signal-to-noise ratio.

   :Options: a positive float (not validated)
   :Default: ``3.0``
   :Notes: the threshold defined in the Methods.

.. _mhp-cnn-kinematic-masking:

``use_kinematic_masking`` / ``masking_prob`` / ``masking_length_frames``
   1-D cutout augmentation: with probability ``masking_prob`` each feature channel of a sequence has a contiguous ``masking_length_frames`` chunk zeroed (the z-score mean), so the network cannot lean on a single dominant feature.

   :Options:
      * ``use_kinematic_masking`` — ``true`` or ``false``
      * ``masking_prob`` — a probability in ``[0, 1]`` (not validated)
      * ``masking_length_frames`` — a positive integer, cut to the sequence length (not validated)
   :Default: ``true``, ``0.15`` and ``60``
   :Notes: ``masking_prob`` and ``masking_length_frames`` are both read only when masking is on.

.. _mhp-cnn-use-hybrid-flatten:

``use_hybrid_flatten``
   Whether time-invariant averages reach the dense head beside the strictly-timed motifs.

   :Options:
      * ``true`` — concatenates a global-average-pooled channel vector to the flattened final feature map before the dense head
      * ``false`` — the flattened final feature map alone
   :Default: ``false``

.. _mhp-cnn-saliency-enable:

``saliency.enable`` / ``saliency.segmentation``
   The regional-saliency phase (the ``saliency`` sub-block): ``enable`` toggles it, and ``segmentation`` groups USVs by their QLVM category, which the modeling pickle carries when it was extracted from summaries holding ``qlvm_category`` (the regular map's categories, the label of every map, the conditional ones included).

   :Options:
      * ``enable`` — ``true`` or ``false``
      * ``segmentation`` — ``'category'`` only (any other value raises a ``ValueError``)
   :Default: ``true`` and ``'category'``
   :Notes: a pickle without the requested labels stops the run in the saliency pre-flight check, before Phase 1 (a ``RuntimeError`` saying the labels are unavailable); set ``enable`` to ``false`` for such a pickle.

.. _mhp-cnn-saliency-alpha:

``saliency.alpha`` / ``saliency.radius_mode``
   Each category's inclusion circle.

   :Options:
      * ``alpha`` — scales each category's inclusion circle (radius ``alpha × d_nn / 2``, ``d_nn`` the distance to the nearest other category centre, so circles never overlap for ``alpha <= 1``); a float ``> 0`` (``<= 0`` raises; ``<= 1`` keeps circles from overlapping)
      * ``radius_mode`` — ``'adaptive'`` (one radius per category from its own ``d_nn``) or ``'uniform'`` (one shared radius from the tightest pair); any other value raises
   :Default: ``0.5`` and ``'adaptive'``

.. _mhp-cnn-saliency-centers:

``saliency.center_min_points_per_label`` / ``saliency.center_max_points_per_label`` / ``saliency.center_grid_resolution`` / ``saliency.center_grid_margin_fraction`` / ``saliency.center_rng_seed``
   How each category centre is found: the highest-density point of a 2-D Gaussian KDE of that category's USVs.

   :Options:
      * ``center_grid_resolution`` — the KDE is evaluated on a square grid of this many points per axis; a positive integer (not validated)
      * ``center_grid_margin_fraction`` — the grid is laid over the category's data range padded on every side by this fraction of the larger axis range; a float ``>= 0`` (not validated). On a torus the padded grid is clipped to the period, so the margin acts on a euclidean manifold
      * ``center_min_points_per_label`` — a category with fewer USVs than this gets no centre (too few for a stable density mode); a positive integer with min ``<=`` max (not validated)
      * ``center_max_points_per_label`` — a category with more USVs than this is subsampled to that many to bound the KDE cost; a positive integer with min ``<=`` max (not validated)
      * ``center_rng_seed`` — the seed of the generator for that subsample; any integer (not validated)
   :Default: ``20``, ``30000``, ``200``, ``0.05`` and ``0`` (in the order listed in the heading)
   :Notes: the five values are stored with the derived geometry (``cluster_geometry``).

**hyperparameters.linear_models** — the two JAX linear models,
``manifold_regression`` (continuous manifold position: the Huber coordinate
regression on euclidean, the closed-form torus-embedding regression on the torus)
and ``multinomial_logistic`` (vocal categories). The two blocks share their core
keys; the multinomial block adds the class-balance and focal-loss keys and an
``inner_cv_scoring_metric`` that the manifold block does not carry, because a
manifold fit's inner-CV objective is geometry-determined (the macro von Mises
log-score ``vm_logscore`` on the torus, ``dcor_xy`` on euclidean).

.. code-block:: json

    "linear_models": {
        "manifold_regression": {
            "bin_resizing_factor": 1,
            "lambda_smooth_fixed": 1.0,
            "l2_reg_fixed": 0.01,
            "smoothness_derivative_order": 1,
            "temporal_basis": {
                "type": "bspline",
                "n_splines": 20,
                "spline_order": 3,
                "lambda_smooth_fixed": 1e-8,
                "l2_reg_fixed": 1e-4,
                "l2_lag_alpha": 10,
                "lambda_smooth_decades_each_side": 2
            },
            "huber_delta": 1.0,
            "learning_rate": 0.005,
            "max_iter": 20000,
            "tol": 1e-3,
            "random_state": 0,
            "verbose": false,
            "use_lax_loop": true,
            "tune_regularization_bool": false,
            "tune_regularization_params": {
                "lambda_smooth_decades_each_side": 0,
                "l2_reg_decades_each_side": 4,
                "inner_cv_folds": 5,
                "inner_cv_use_one_se_rule": false,
                "inner_max_iter": 2500
            }
        },
        "multinomial_logistic": {
            "balance_predictions_bool": false,
            "balance_train_bool": false,
            "bin_resizing_factor": 1,
            "lambda_smooth_fixed": 1.0,
            "l2_reg_fixed": 0.01,
            "smoothness_derivative_order": 1,
            "temporal_basis": {
                "type": "bspline",
                "n_splines": 20,
                "spline_order": 3,
                "lambda_smooth_fixed": 1e-5,
                "l2_reg_fixed": 1e-5,
                "l2_lag_alpha": 100,
                "lambda_smooth_decades_each_side": 2
            },
            "learning_rate": 0.005,
            "max_iter": 20000,
            "tol": 1e-3,
            "random_state": 0,
            "verbose": false,
            "use_lax_loop": true,
            "focal_loss_gamma": 0.5,
            "tune_regularization_bool": false,
            "tune_regularization_params": {
                "lambda_smooth_decades_each_side": 0,
                "l2_reg_decades_each_side": 4,
                "inner_cv_folds": 5,
                "inner_cv_scoring_metric": "auc",
                "inner_cv_use_one_se_rule": false,
                "inner_max_iter": 2500
            }
        }
    }

.. list-table:: ``linear_models`` at a glance (defaults are manifold / multinomial where they differ)
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`bin_resizing_factor <mhp-lin-bin-resizing-factor>`
     - ``1``
     - integer ``≥ 1``; only with ``temporal_basis.type`` ``"none"``
   * - :ref:`lambda_smooth_fixed <mhp-lin-fixed-penalties>`
     - ``1.0``
     - float ``>= 0``; per-frame basis only
   * - :ref:`l2_reg_fixed <mhp-lin-fixed-penalties>`
     - ``0.01``
     - float ``>= 0``; per-frame basis only
   * - :ref:`smoothness_derivative_order <mhp-lin-smoothness-derivative-order>`
     - ``1``
     - ``1``, ``2``
   * - :ref:`temporal_basis.type <mhp-lin-temporal-basis>`
     - ``"bspline"``
     - ``"none"``, ``"bspline"``
   * - :ref:`temporal_basis.n_splines <mhp-lin-temporal-basis>`
     - ``20``
     - integer ``≥ spline_order + 1``
   * - :ref:`temporal_basis.spline_order <mhp-lin-temporal-basis>`
     - ``3``
     - positive integer (B-spline degree)
   * - :ref:`temporal_basis.lambda_smooth_fixed <mhp-lin-temporal-basis>`
     - ``1e-8`` / ``1e-5``
     - float ``>= 0``
   * - :ref:`temporal_basis.l2_reg_fixed <mhp-lin-temporal-basis>`
     - ``1e-4`` / ``1e-5``
     - float ``>= 0``
   * - :ref:`temporal_basis.l2_lag_alpha <mhp-lin-temporal-basis>`
     - ``10`` / ``100``
     - float ``>= 0``; ``0`` is a uniform L2
   * - :ref:`temporal_basis.lambda_smooth_decades_each_side <mhp-lin-temporal-basis>`
     - ``2``
     - integer ``>= 0``
   * - :ref:`huber_delta <mhp-lin-huber-delta>`
     - ``1.0`` (manifold only)
     - float > 0, manifold units
   * - :ref:`learning_rate <mhp-lin-optimiser>`
     - ``0.005``
     - float > 0
   * - :ref:`max_iter <mhp-lin-optimiser>`
     - ``20000``
     - integer > 0
   * - :ref:`tol <mhp-lin-optimiser>`
     - ``1e-3``
     - float > 0
   * - :ref:`random_state <mhp-lin-random-state>`
     - ``0``
     - any integer (not ``null``)
   * - :ref:`verbose <mhp-lin-verbose>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`use_lax_loop <mhp-lin-use-lax-loop>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`balance_predictions_bool <mhp-lin-balance-predictions-bool>`
     - ``false`` (multinomial only)
     - ``true``, ``false``
   * - :ref:`balance_train_bool <mhp-lin-balance-train-bool>`
     - ``false`` (multinomial only)
     - ``true``, ``false``
   * - :ref:`focal_loss_gamma <mhp-lin-focal-loss-gamma>`
     - ``0.5`` (multinomial only)
     - float ``>= 0``; ``0`` is off
   * - :ref:`tune_regularization_bool <mhp-lin-tune-regularization-bool>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`tune_regularization_params.lambda_smooth_decades_each_side <mhp-lin-tune-regularization-params>`
     - ``0``
     - integer ``>= 0``
   * - :ref:`tune_regularization_params.l2_reg_decades_each_side <mhp-lin-tune-regularization-params>`
     - ``4``
     - integer ``>= 0``
   * - :ref:`tune_regularization_params.inner_cv_folds <mhp-lin-tune-regularization-params>`
     - ``5``
     - integer ``≥ 2``
   * - :ref:`tune_regularization_params.inner_cv_scoring_metric <mhp-lin-tune-regularization-params>`
     - ``'auc'`` (multinomial only)
     - ``'auc'``, ``'score'``, ``'f1'``, ``'recall'``, ``'mcc'``, ``'ll'``, ``'brier'``, ``'ece'``
   * - :ref:`tune_regularization_params.inner_cv_use_one_se_rule <mhp-lin-tune-regularization-params>`
     - ``false``
     - ``true``, ``false``
   * - :ref:`tune_regularization_params.inner_max_iter <mhp-lin-tune-regularization-params>`
     - ``2500``
     - integer > 0

.. _mhp-lin-bin-resizing-factor:

``bin_resizing_factor``
   Temporal binning of the history before the fit: every ``bin_resizing_factor`` consecutive frames are averaged into one bin (``1`` keeps the full frame resolution).

   :Options: an integer ``≥ 1``; values below ``2`` mean no binning
   :Default: ``1`` in both models
   :Notes: cannot be combined with a ``temporal_basis`` other than ``"none"``: it is read only with ``temporal_basis.type`` ``"none"`` (with ``"bspline"`` any value other than ``1`` raises). A factor larger than the history window (``filter_history × camera_sampling_rate`` frames) raises.

.. _mhp-lin-fixed-penalties:

``lambda_smooth_fixed`` / ``l2_reg_fixed``
   The fixed temporal-smoothness and L2 penalties of the per-frame representation (``temporal_basis.type`` ``"none"``).

   :Options:
      * ``lambda_smooth_fixed`` — a non-negative float
      * ``l2_reg_fixed`` — a non-negative float

      Each must be positive when tuning is on and its ``*_decades_each_side`` is above ``0`` (the log-spaced grid needs a positive centre; validated).
   :Default: ``1.0`` and ``0.01`` in both models
   :Notes: with the shipped ``"bspline"`` basis these two keys are not read: the fit takes ``temporal_basis.lambda_smooth_fixed`` and ``temporal_basis.l2_reg_fixed`` instead (``modeling_bases_functions.resolve_temporal_basis``). Both linear models ship with ``tune_regularization_bool`` ``false`` (see the note below), so whichever pair applies is used as is; in model selection the L2 penalty is additionally rescaled by ``1 / sqrt(n_features)`` of the trial model so larger stacks are not over-regularised.

.. _mhp-lin-smoothness-derivative-order:

``smoothness_derivative_order``
   The finite-difference order of the smoothness penalty along the time axis.

   :Options:
      * ``1`` — penalises squared first differences, zero-cost for a flat filter (the Pillow / Calhoun GLM tradition)
      * ``2`` — penalises squared second differences, zero-cost for any straight line (the GAM / smoothing-spline choice)

      Validated.
   :Default: ``1`` in both models
   :Notes: with order ``2`` on full-resolution frames the multinomial loss and the torus solve add reflective (Neumann) edge rows so the filter's end points are constrained like its interior; order ``1`` is already a first-difference operator and needs none. With ``temporal_basis.type`` ``"bspline"`` this key is not read: the spline penalty is always a second-order difference on neighbouring coefficients.

.. _mhp-lin-temporal-basis:

``temporal_basis``
   How each feature's filter over the history window is represented, the same in the univariate screen and model selection (every sub-key except ``type`` is read only when ``type`` is ``"bspline"``).

   :Options:
      * ``type`` — ``"none"`` or ``"bspline"`` (validated):

        * ``"type": "none"`` — one free weight per frame, penalised by the block's ``lambda_smooth_fixed``, ``l2_reg_fixed`` and ``smoothness_derivative_order`` (with reflective edge rows for order 2).
        * ``"type": "bspline"`` (both models' default) — a spline layer on top of the same models, mirroring the bout-onset GAM's lag axis (see **Spline basis** below).

      * ``n_splines`` — an integer ``≥ spline_order + 1`` (validated)
      * ``spline_order`` — a positive integer, the B-spline degree (``3`` is cubic; not validated beyond the ``n_splines`` check)
      * ``lambda_smooth_fixed`` — a non-negative float, positive when tuning is on with ``lambda_smooth_decades_each_side`` above ``0``
      * ``l2_reg_fixed`` — a non-negative float (not validated)
      * ``l2_lag_alpha`` — a non-negative float, ``0`` for a uniform L2 (only a positive value turns the lag-varying ridge on)
      * ``lambda_smooth_decades_each_side`` — a non-negative integer, read only when tuning is on
   :Default: ``"bspline"``, ``20``, ``3``, ``1e-8`` / ``1e-5``, ``1e-4`` / ``1e-5``, ``10`` / ``100`` and ``2`` (manifold / multinomial where they differ)
   :Notes:
      **Spline basis:** every feature's history is projected onto ``n_splines`` (default 20) cubic B-splines (``spline_order`` 3) with knots spaced evenly from the first to the last frame — exactly pyGAM's basis (``modeling_bases_functions.gam_bspline_basis``) — so the model fits ``n_splines`` coefficients per feature. The projection averages the history under each spline (the basis is divided by the window length, as the GAM averages its per-lag contributions per event), which keeps the inputs on the features' z-score scale.

      **Penalty:** the penalty mirrors pyGAM's: a second-order difference penalty on neighbouring spline coefficients, open boundary, at strength ``temporal_basis.lambda_smooth_fixed``, plus an L2 penalty on the spline coefficients at ``temporal_basis.l2_reg_fixed`` (1e-4 for the torus model, 1e-5 for the multinomial model); with ``tune_regularization_bool`` on, the smoothness strength is tuned over ``lambda_smooth_decades_each_side`` decades and the L2 stays at its fixed value.

      **Why 20 splines and these L2 values:** the 20 splines and the two L2 values come from a basis experiment on the intact-partner male cohort: on the torus, 8 splines with no L2 put 7.3× the mid-window weight at −4 s (and 11× at 0 s) in a fit to permuted positions, because the nearly unpenalised end coefficients absorb noise (the rise follows the window edge when the window is shortened); an L2 of 1e-4 brings that to about 2–3×, recovers planted filters of known shape (bumps at −3 s and −1.5 s, ramps over the last 0.3 s and 1 s) at r ≥ 0.90 with 20 splines, and costs about 9 % of the cross-validated gain, whereas the per-frame torus fit recovers none of them (r −0.18 to 0.36), which is why the torus model also ships in spline mode. For the multinomial model 20 splines gave the highest cross-validated macro-AUC and recovered every planted shape at r ≥ 0.96; its L2 is set with the lag-varying ridge below. A 2–4× end weight remains in every configuration, so the filter figures draw each filter against its permuted null.

      **Lag-varying ridge (l2_lag_alpha):** with ``l2_lag_alpha`` > 0 the L2 grows with each spline's distance from onset, ``l2_reg_fixed × (1 + alpha × lag²)`` (``lag`` as a fraction of the window), the lag-varying ridge of penalised distributed-lag models, implemented by scaling the basis columns (the stored ``basis_coefficients`` belong to the scaled basis). The torus model uses ``alpha = 10``: with a uniform L2 its filters still rose at −4 s, and neither a clamped basis nor much stronger smoothness changed that, whereas the lag-varying ridge brought the far-edge weight of head pitch and neck elevation to their permuted level, kept the recovery of planted shapes (a bump at −3 s included) and the onset rise, and cost no cross-validated gain except for inter-partner distance (−7 %), whose oldest frames carry real information from before the window. The multinomial model uses ``l2_reg_fixed = 1e-5`` with ``alpha = 100``: without them its no-signal fits put a spike at −4 s (inter-partner distance 15.1 against an onset peak of 22.1, ×1e-3), which this setting cuts to 0.76 against 8.8 while costing at most 4 % of the cross-validated macro-AUC (inter-partner distance; under 2 % elsewhere) and keeping the planted-shape recovery at r ≥ 0.89.

      **History window:** the history window stays at 4 s: an 8 s re-extraction showed that the residual far-edge weight of the slow dyadic features is information from before the window (inter-partner distance and head pitch predict ~40 % better with 8 s, with no edge rise in the longer window), while the features that act in the last second gain nothing (neck elevation −14 %); the far-edge weight of a slow feature is read as history beyond the window, not as a timed effect at −4 s.

      **Multinomial smoothness default:** the multinomial default is ``lambda_smooth_fixed`` = 1e-5 with ``lambda_smooth_decades_each_side`` = 2 (1e-7 to 1e-3), set by a penalty pilot: on the top screened feature, cross-validated macro-AUC is flat from no penalty up to 1e-4 and falls beyond it, as the second-order penalty straightens the filters; 1e-5 is the largest strength with no measurable loss and damps the noise at the oldest lags.

      **Why both models ship in spline mode:** the multinomial model ships in spline mode: on the same inputs the two bases select the same anchor and second feature and reach the same cross-validated macro-AUC (0.611 spline vs 0.618 per-frame, overlapping 99 % intervals), but the spline selection ran in a fiftieth of the time and never lost a fold, whereas the per-frame fits at the shipped learning rate occasionally let Adam move a feature's 600 correlated raw-lag weights together and push a few events past the float32 softmax saturation margin (a logit margin of about 17, where ``p_t`` rounds to exactly 1) — on 8 of 10 folds from the second selection step on. Before the focal modulator was clamped (see ``focal_loss_gamma``) one saturated event gave a NaN gradient and killed the fold. On the torus the two bases gave the same selection and scores at an 8-spline, no-L2 setting, but the per-frame filters cannot recover a planted shape (see above), so the torus model ships in spline mode too.

      **Manifold smoothness default:** the manifold (torus) default is ``lambda_smooth_fixed`` = 1e-8 with ``lambda_smooth_decades_each_side`` = 2 (1e-10 to 1e-6), from the same pilot: there every strength costs some macro von Mises log-score, because the signal is a sharp rise over the last ~0.3 s that the penalty flattens, and 1e-8 is the largest strength whose loss is negligible.

      **Stored outputs:** the fitted coefficients are turned back into filters on the frame axis (basis × coefficients) before they are stored, so results, consolidation and plots keep the full-resolution layout. The raw spline coefficients are stored beside them, under ``basis_coefficients`` (per fold, per held-out refit and per selection candidate; ``None`` without a basis). The run metadata records the ``temporal_basis`` block and the ``effective_penalty`` the fit used.

      **Compatibility:** the B-spline basis cannot be combined with ``bin_resizing_factor`` > 1. The penalty strength is not transferable between the two types (the B-spline penalty acts on ``n_splines`` coefficients, not on every frame), nor from pyGAM's ``lam_penalty`` (pyGAM adds it to a deviance summed over all rows, the JAX models to per-event mean losses).

.. _mhp-lin-huber-delta:

``huber_delta``
   ``manifold_regression`` only: the Huber transition point (native manifold units) of the euclidean coordinate regression's loss.

   :Options: a positive float in the manifold's native units (not validated)
   :Default: ``1.0``
   :Notes: read only on the euclidean metric; the torus solve is closed-form squared error and ignores it.

.. _mhp-lin-optimiser:

``learning_rate`` / ``max_iter`` / ``tol``
   The Adam optimiser of the iterative fits (the multinomial model, the euclidean coordinate regression): the initial step size (cosine-decayed to zero over ``max_iter`` steps, with a fixed global-norm gradient clip of ``1.0`` in both estimators), the step cap, and the parameter-change tolerance that ends the loop early.

   :Options:
      * ``learning_rate`` — a positive float
      * ``max_iter`` — a positive integer
      * ``tol`` — a positive float

      None validated.
   :Default: ``0.005``, ``20000`` and ``1e-3`` in both models
   :Notes: the torus solve is closed-form and ignores all three. The shipped multinomial rate (``0.005``) was validated with the spline basis; with the per-frame basis (``temporal_basis.type`` ``"none"``) ``0.001`` converged in a third of the iterations with the same held-out score on the same inputs, so lower it when switching that model to raw lags.

.. _mhp-lin-random-state:

``random_state``
   The base seed of every fit (offset by the fold index, and by the inner-fold index in the inner CV).

   :Options: an integer (not validated; ``null`` fails, since the fold index is added to it)
   :Default: ``0`` in both models

.. _mhp-lin-verbose:

``verbose``
   Print per-fit optimiser progress.

   :Options: ``true`` / ``false``
   :Default: ``false`` in both models

.. _mhp-lin-use-lax-loop:

``use_lax_loop``
   Run the whole descent inside one compiled ``jax.lax.while_loop`` instead of a Python loop.

   :Options:
      * ``true`` — one compiled ``jax.lax.while_loop`` (the fast path)
      * ``false`` — a Python loop
   :Default: ``true`` in both models

.. _mhp-lin-balance-predictions-bool:

``balance_predictions_bool``
   ``multinomial_logistic`` only: whether the scored probabilities are prior-corrected.

   :Options:
      * ``true`` — subtracts the training log class priors from the logits before the softmax (``predict_proba(balanced=True)``), so the scored probabilities carry the learned behaviour-to-category evidence as if every category were equally likely a priori
      * ``false`` — scores the raw softmax
   :Default: ``false``

.. _mhp-lin-balance-train-bool:

``balance_train_bool``
   ``multinomial_logistic`` only: how class imbalance is handled in the training fold.

   :Options:
      * ``true`` — down-samples every training fold to ``min(class_count)`` rows per class and fits with ``focal_gamma = 0`` and uniform class weights (an already balanced batch must not be re-weighted)
      * ``false`` — keeps the natural class prior in the training fold and handles imbalance inside the loss (softened inverse-frequency class weights plus the focal modulation)
   :Default: ``false``
   :Notes: the test fold always keeps the natural prior.

.. _mhp-lin-focal-loss-gamma:

``focal_loss_gamma``
   ``multinomial_logistic`` only: the focal-loss focusing exponent ``(1 - p_t) ** gamma`` that down-weights easy examples (``0`` is plain class-weighted cross-entropy).

   :Options: a non-negative float (not validated); ``0`` turns the focal modulation off
   :Default: ``0.5``
   :Notes: the value is not read when ``balance_train_bool`` is ``true`` (the fit then uses ``0``). The base is clamped at ``max(1 - p_t, 1e-7)``: for ``gamma < 1`` the modulator's derivative is infinite at ``p_t = 1``, which float32 reaches exactly once an event's logit margin exceeds about 17; without the clamp one saturated event turned the whole gradient NaN and ended the fit, with it a saturated event contributes nothing to the gradient and the loss is unchanged everywhere else.

.. _mhp-lin-tune-regularization-bool:

``tune_regularization_bool``
   If ``true``, run an inner-loop cross-validation to pick ``lambda_smooth`` / ``l2_reg`` per fold (parameters in ``tune_regularization_params``: the search width in decades, inner-CV folds, scoring metric, and the one-standard-error rule).

   :Options:
      * ``true`` — tune the two penalties per fold by inner CV
      * ``false`` — ``tune_regularization_params`` is not read and the two ``*_fixed`` penalties are used directly
   :Default: ``false`` in both models (see the note below)
   :Notes: each inner validation fold holds out whole training sessions, as the outer folds do (capped at the number of training sessions; a fold with a single training session falls back to the fixed centres), so a penalty is never scored on events from sessions it was fit on.

.. _mhp-lin-tune-regularization-params:

``tune_regularization_params``
   The inner-CV search; none of its keys is read while ``tune_regularization_bool`` is ``false``.

   :Options:
      * ``lambda_smooth_decades_each_side`` / ``l2_reg_decades_each_side`` — set the log-spaced grid around each ``*_fixed`` centre (``0`` pins that penalty to its centre, ``4`` spans eight decades); non-negative integers (a negative value raises)
      * ``inner_cv_folds`` — the inner fold count; an integer ``≥ 2`` (not validated; capped at the number of training sessions, and a training fold with fewer than ``2 × inner_cv_folds`` events or a single session falls back to the fixed centres)
      * ``inner_cv_scoring_metric`` — multinomial only, the inner objective, which the manifold model derives from its geometry instead; validated when the tuner runs:

        * ``'auc'`` — the macro one-vs-rest AUC (higher is better)
        * ``'score'``, ``'f1'``, ``'recall'``, ``'mcc'`` — higher is better
        * ``'ll'``, ``'brier'``, ``'ece'`` — lower is better; ``'ece'`` uses ``diagnostics.ece_n_bins``

      * ``inner_cv_use_one_se_rule`` — ``true`` picks the smoothest pair within one standard error of the inner argmax rather than the argmax itself; ``true`` / ``false``
      * ``inner_max_iter`` — caps the inner fits' steps; a positive integer (not validated; see the note below on why the multinomial inner CV is under-converged at ``2500``)
   :Default: ``0``, ``4``, ``5``, ``'auc'`` (multinomial only), ``false`` and ``2500`` in both models
   :Notes: with ``temporal_basis.type`` ``"bspline"`` the two decade keys here are not read: the smoothness grid takes ``temporal_basis.lambda_smooth_decades_each_side`` and the L2 is pinned to its fixed value.

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
   filter's middle without its edges floating free (that applies to
   ``smoothness_derivative_order`` ``2``; the shipped order ``1`` is already a
   first-difference operator and has no edge rows to add).

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

   **Frozen von Mises concentration.** On a torus manifold run
   ``vocal_features.freeze_selection_kappa = true`` (shipped) fits the von Mises
   concentration ``kappa`` once on the development set (the marginal dispersion of
   the USV positions around their circular mean) and reuses it for every score
   (the univariate screen, baseline, fitted candidates, held-out, and the
   acceptance-gate bootstrap). ``false`` self-refits ``kappa`` per fold, which
   floors the score on weak folds (the ``kappa → 0`` clamp) and corrupts the
   paired per-fold margin; a single frozen ``kappa`` keeps every score on one
   dispersion scale so the margins are a proper scoring rule. On the intact-partner
   male cohort the self-refit put 5 of 10 session folds at the floor for every
   feature and its null, so the univariate screen passed only 2 of 23 features.

   **The torus screen uses the selection score.** The univariate screen scores
   every fold with the same rule as the greedy gate: the selection score's own
   averaging (``vm_logscore``, the region-balanced macro score, by default; it
   matches the equal-region reweighting of the fit, whose centre is the
   equal-weight average of the regions rather than the centre of the USVs) and
   the frozen ``kappa``. The torus input pickle is therefore loaded before the
   screen.

**hyperparameters.classical** — the ``'pygam'`` / ``'sklearn'`` engine models of
the onset, binomial-category and bout-parameter pipelines (and their model
selections):

.. code-block:: json

    "classical": {
        "pygam": {
            "n_splines_time": 8,
            "n_splines_value": 5,
            "lam_penalty": 0.6,
            "max_iterations": 100,
            "tol_val": 1e-4,
            "distribution": "gamma",
            "link": "log"
        },
        "logistic_regression": {
            "penalty": "l2",
            "cs": [0.0001, 0.001, 0.01, 0.1],
            "cv": 10,
            "solver": "liblinear",
            "max_iter": 10000
        },
        "ridge_regression": {
            "alphas": [0.1, 1.0, 10.0, 100.0],
            "cv": 10
        }
    }

.. list-table:: ``hyperparameters.classical`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`pygam.n_splines_time <mhp-cls-pygam-n-splines>`
     - ``8``
     - integer >= 4
   * - :ref:`pygam.n_splines_value <mhp-cls-pygam-n-splines>`
     - ``5``
     - integer >= 4
   * - :ref:`pygam.lam_penalty <mhp-cls-pygam-lam-penalty>`
     - ``0.6``
     - number >= 0
   * - :ref:`pygam.max_iterations <mhp-cls-pygam-pirls>`
     - ``100``
     - integer >= 1
   * - :ref:`pygam.tol_val <mhp-cls-pygam-pirls>`
     - ``1e-4``
     - number > 0
   * - :ref:`pygam.distribution <mhp-cls-pygam-distribution-link>`
     - ``'gamma'``
     - any string (recorded only)
   * - :ref:`pygam.link <mhp-cls-pygam-distribution-link>`
     - ``'log'``
     - any string (recorded only)
   * - :ref:`logistic_regression.penalty <mhp-cls-lr-penalty-solver>`
     - ``'l2'``
     - ``'l2'``, ``'l1'``
   * - :ref:`logistic_regression.solver <mhp-cls-lr-penalty-solver>`
     - ``'liblinear'``
     - a solver that supports the penalty
   * - :ref:`logistic_regression.cs <mhp-cls-lr-cs>`
     - ``[0.0001, 0.001, 0.01, 0.1]``
     - non-empty list of numbers > 0
   * - :ref:`logistic_regression.cv <mhp-cls-lr-cv>`
     - ``10``
     - integer >= 2
   * - :ref:`logistic_regression.max_iter <mhp-cls-lr-max-iter>`
     - ``10000``
     - integer > 0
   * - :ref:`ridge_regression.alphas <mhp-cls-rr-alphas>`
     - ``[0.1, 1.0, 10.0, 100.0]``
     - non-empty list of numbers >= 0
   * - :ref:`ridge_regression.cv <mhp-cls-rr-cv>`
     - ``10``
     - integer >= 2

.. _mhp-cls-pygam:

``pygam``
   The ``'pygam'`` engine's tensor-product-spline GAM, ``te(value, lag)``.

   :Notes: the whole block is required: a settings file without it raises ``KeyError`` instead of falling back to hard-coded values.

.. _mhp-cls-pygam-n-splines:

``pygam.n_splines_value`` / ``pygam.n_splines_time``
   The B-spline counts on the feature-value and lag axes of ``te(value, lag)``.

   :Options: integers ``>= 4`` (pyGAM requires more splines than the cubic spline order of 3)
   :Default: ``n_splines_value`` ``5`` and ``n_splines_time`` ``8``
   :Notes: the lag basis is the same one ``temporal_basis.type = 'bspline'`` reuses for the JAX models.

.. _mhp-cls-pygam-lam-penalty:

``pygam.lam_penalty``
   ``lam_penalty`` is the **fixed** smoothness penalty strength of the GAM.

   :Options: a number ``>= 0`` (pyGAM raises on a negative value and on ``null``)
   :Default: ``0.6``
   :Notes: the strength is never tuned: pyGAM chooses it by GCV / UBRE only inside ``gridsearch()``, which no pipeline runs.

.. _mhp-cls-pygam-pirls:

``pygam.max_iterations`` / ``pygam.tol_val``
   The PIRLS optimiser's iteration cap and convergence tolerance.

   :Options:
      * ``max_iterations`` — an integer ``>= 1`` (checked by pyGAM)
      * ``tol_val`` — a positive number (not validated; ``0`` or a negative value only makes every fit run to ``max_iterations``)
   :Default: ``100`` and ``1e-4``
   :Notes: a fit whose last deviance change is still above ``tol_val`` is recorded as ``converged = 0``.

.. _mhp-cls-pygam-distribution-link:

``pygam.distribution`` / ``pygam.link``
   Recorded in the run metadata (``pygam_hyperparameters``) and read nowhere else.

   :Options: any string (recorded, not validated), with ``'gamma'`` / ``'log'`` describing the fit actually used
   :Default: ``distribution`` ``'gamma'`` and ``link`` ``'log'``
   :Notes: the bout-parameter GAM is always fitted as a Gamma GAM with a log link and the binary pipelines as ``LogisticGAM``, so editing these two keys changes the record, not the fit.

.. _mhp-cls-logistic-regression:

``logistic_regression``
   The ``'sklearn'`` engine's ``LogisticRegressionCV`` for the binary targets (onsets, binomial categories).

   :Notes: the inner folds hold out whole training sessions (stratified by class); ``cv`` is capped at the number of training sessions, and a training set from a single session, or one whose session folds would leave a class out of a fold, falls back to ``cv`` stratified event folds.

.. _mhp-cls-lr-penalty-solver:

``logistic_regression.penalty`` / ``logistic_regression.solver``
   The penalty and the solver, as in scikit-learn.

   :Options: ``penalty`` ``'l2'`` or ``'l1'``, in a pair the ``solver`` supports:

      * ``'l2'`` — with ``'liblinear'``, ``'lbfgs'``, ``'newton-cg'``, ``'newton-cholesky'``, ``'sag'`` or ``'saga'``
      * ``'l1'`` — with ``'liblinear'`` or ``'saga'``
   :Default: ``'l2'`` and ``'liblinear'``
   :Notes: scikit-learn raises on any other pair; ``'elasticnet'`` is not usable because no ``l1_ratios`` are passed.

.. _mhp-cls-lr-cs:

``logistic_regression.cs``
   The inverse-regularisation grid searched by ``cv``-fold inner cross-validation.

   :Options: a non-empty list of positive numbers (scikit-learn raises on ``0``)
   :Default: ``[0.0001, 0.001, 0.01, 0.1]``

.. _mhp-cls-lr-cv:

``logistic_regression.cv``
   The number of inner cross-validation folds over which ``cs`` is searched.

   :Options: an integer ``>= 2``
   :Default: ``10``
   :Notes: capped at the number of training sessions (see ``logistic_regression`` above for the event-fold fallback).

.. _mhp-cls-lr-max-iter:

``logistic_regression.max_iter``
   The iteration cap.

   :Options: a positive integer
   :Default: ``10000``
   :Notes: a fold that reaches ``max_iter`` is recorded as not converged.

.. _mhp-cls-ridge-regression:

``ridge_regression``
   The ``'sklearn'`` engine's bout-parameter regression, an L2-penalized Gamma GLM (``GammaRegressor``, log link) whose penalty ``alpha`` is chosen from ``alphas`` by a ``cv``-fold ``GridSearchCV`` on the Gamma deviance — the same likelihood the model is scored with, so fit and score agree.

   :Notes: the inner folds hold out whole training sessions (``GroupKFold``, ``cv`` capped at the number of training sessions; a single-session training set falls back to ``cv`` event folds).

.. _mhp-cls-rr-alphas:

``ridge_regression.alphas``
   The grid the penalty ``alpha`` is chosen from.

   :Options: a non-empty list of numbers ``>= 0`` (scikit-learn raises on a negative value; ``0`` is an unpenalized fit)
   :Default: ``[0.1, 1.0, 10.0, 100.0]``

.. _mhp-cls-rr-cv:

``ridge_regression.cv``
   The number of ``GridSearchCV`` folds.

   :Options: an integer ``>= 2``
   :Default: ``10``

**hyperparameters.basis_functions** — the temporal bases of the ``'sklearn'``
engine, selected by ``model_params.model_basis_function`` (the ``'pygam'`` engine
and the JAX models ignore this block; the JAX models' optional spline layer is the
``temporal_basis`` sub-block above). Each basis spans the ``filter_history`` window
and the history is projected onto it before the linear fit; the fitted coefficients
are back-projected onto the frame axis for the stored ``filter_shapes``.

.. code-block:: json

    "basis_functions": {
        "raised_cosine": {
            "neye": 0,
            "ncos": 6,
            "kpeaks_proportion": 0.95,
            "b": 96,
            "plot_bool": true
        },
        "bspline": {
            "n_splines": 32,
            "degree": 3,
            "plot_bool": true
        },
        "laplacian_pyramid": {
            "levels": 9.5,
            "step": 0.5,
            "fwhm": 10,
            "plot_bool": true
        }
    }

.. list-table:: ``hyperparameters.basis_functions`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`raised_cosine.neye <mhp-bf-raised-cosine>`
     - ``0``
     - integer >= 0
   * - :ref:`raised_cosine.ncos <mhp-bf-raised-cosine>`
     - ``6``
     - integer >= 2
   * - :ref:`raised_cosine.kpeaks_proportion <mhp-bf-raised-cosine>`
     - ``0.95``
     - fraction in ``(0, 1]``
   * - :ref:`raised_cosine.b <mhp-bf-raised-cosine>`
     - ``96``
     - number > 0
   * - :ref:`bspline.n_splines <mhp-bf-bspline>`
     - ``32``
     - integer >= ``2 * degree``
   * - :ref:`bspline.degree <mhp-bf-bspline>`
     - ``3``
     - integer ``1``–``5``
   * - :ref:`laplacian_pyramid.levels <mhp-bf-laplacian-pyramid>`
     - ``9.5``
     - number > 0
   * - :ref:`laplacian_pyramid.step <mhp-bf-laplacian-pyramid>`
     - ``0.5``
     - number > 0
   * - :ref:`laplacian_pyramid.fwhm <mhp-bf-laplacian-pyramid>`
     - ``10``
     - frames > 0
   * - :ref:`plot_bool <mhp-bf-plot-bool>` (all three bases)
     - ``true``
     - ``true``, ``false``

.. _mhp-bf-raised-cosine:

``raised_cosine.neye`` / ``raised_cosine.ncos`` / ``raised_cosine.kpeaks_proportion`` / ``raised_cosine.b``
   ``ncos`` raised cosines whose peaks are log-spaced (denser near the event) from the first frame to ``kpeaks_proportion`` of the window, with ``b`` the offset of the log stretch (larger is closer to linear spacing); ``neye`` prepends that many identity (single-frame) columns for dense sampling right before the event.

   :Options:
      * ``neye`` — an integer ``>= 0``
      * ``ncos`` — an integer ``>= 2`` (one cosine leaves the peak spacing undefined and the construction fails)
      * ``kpeaks_proportion`` — a fraction in ``(0, 1]`` of the history window
      * ``b`` — a positive number
   :Default: ``neye`` ``0``, ``ncos`` ``6``, ``kpeaks_proportion`` ``0.95``, ``b`` ``96``
   :Notes: none of the four is validated beyond what the construction itself rejects.

.. _mhp-bf-bspline:

``bspline.n_splines`` / ``bspline.degree``
   B-splines of polynomial ``degree`` on knots spaced evenly from the first frame to ``degree`` frames before the window's end.

   :Options:
      * ``degree`` — an integer from ``1`` to ``5`` (the range SciPy's ``splrep`` accepts)
      * ``n_splines`` — an integer ``>= 2 * degree``, so that at least ``degree + 1`` knots exist (``6`` at the shipped degree)
   :Default: ``n_splines`` ``32``, ``degree`` ``3``
   :Notes: ``n_splines`` sets the knot count, ``n_splines - degree + 1``, and the basis has one column per distinct knot, so the shipped ``32`` / ``3`` gives 30 columns, not 32. A short window that cannot hold that many distinct knots keeps as many as it can, and raises when fewer than ``degree + 1`` remain.

.. _mhp-bf-laplacian-pyramid:

``laplacian_pyramid.levels`` / ``laplacian_pyramid.step`` / ``laplacian_pyramid.fwhm``
   Gaussians at the level indices ``0, step, 2*step, …`` below ``levels``, each whole level doubling the width.

   :Options:
      * ``levels`` — a positive number (fractional values are allowed, since the indices step by ``step``)
      * ``step`` — a positive number
      * ``fwhm`` — a positive number of frames
   :Default: ``levels`` ``9.5``, ``step`` ``0.5``, ``fwhm`` ``10``
   :Notes: the Gaussians at level index ``i`` have a full width at half maximum of ``2**(i - 1) * fwhm`` frames, so ``fwhm`` is the width at index ``1`` and the finest level (index ``0``) is half as wide. Each level tiles the window with centres spaced by half its own width, so the column count grows quickly as the finest level narrows (the shipped values give 4,048 columns over a 600-frame window). A level too wide to place two centres in the window contributes nothing, and a setting that yields no column raises. None is validated beyond the no-column check.

.. _mhp-bf-plot-bool:

``plot_bool``
   Every basis block carries it: whether the dispatcher writes the ``basis_verification.png`` figure of the basis columns (once per job array, guarded by a lock file).

   :Options:
      * ``true`` — write the figure
      * ``false`` — do not write it
   :Default: ``true`` in all three blocks
   :Notes: the ``'identity'`` basis has no block and always writes its plot.

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
        "n_kappa_rounds": 3,
        "em_tol": 0.0001,
        "transition_pseudocount": 1.0,
        "lambda_smooth": 100.0,
        "input_driven_lambda_smooth": 0.05,
        "l2_reg": 0.1,
        "multinomial_target": "category",
        "focal_gamma": 0.0,
        "multinomial_max_iter": 2000,
        "emission_smoothness_derivative_order": 2,
        "emission_huber_delta": 1.0,
        "emission_learning_rate": 0.001,
        "manifold_max_iter": 5000,
        "emission_tol": 0.0001,
        "emission_random_state": 0,
        "init_kmeans_n_init": 3,
        "init_self_transition": 0.8,
        "init_bias_sd": 0.3
    }

.. list-table:: ``glm_hmm`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`emission_type <mgh-emission-type>`
     - ``'manifold'``
     - ``'manifold'``, ``'multinomial'``
   * - :ref:`transition_mode <mgh-transition-mode>`
     - ``'static'``
     - ``'static'``, ``'input_driven'``
   * - :ref:`history_frames <mgh-history-frames>`
     - ``75``
     - integer > 0
   * - :ref:`cv_folds <mgh-cv-folds>`
     - ``5``
     - integer >= 2
   * - :ref:`n_states_min <mgh-n-states>`
     - ``2``
     - integer > 0, <= ``n_states_max``
   * - :ref:`n_states_max <mgh-n-states>`
     - ``6``
     - integer > 0, >= ``n_states_min``
   * - :ref:`n_em_iters <mgh-n-em-iters>`
     - ``100``
     - integer > 0
   * - :ref:`n_lbfgs <mgh-n-lbfgs>`
     - ``500``
     - integer > 0
   * - :ref:`n_restarts <mgh-n-restarts>`
     - ``5``
     - integer > 0
   * - :ref:`n_kappa_rounds <mgh-n-kappa-rounds>`
     - ``3``
     - integer > 0
   * - :ref:`em_tol <mgh-em-tol>`
     - ``0.0001``
     - float > 0
   * - :ref:`transition_pseudocount <mgh-transition-pseudocount>`
     - ``1.0``
     - float >= 0
   * - :ref:`lambda_smooth <mgh-lambda-smooth-l2-reg>`
     - ``100.0``
     - float >= 0
   * - :ref:`input_driven_lambda_smooth <mgh-input-driven-lambda-smooth>`
     - ``0.05``
     - float >= 0
   * - :ref:`l2_reg <mgh-lambda-smooth-l2-reg>`
     - ``0.1``
     - float >= 0
   * - :ref:`multinomial_target <mgh-multinomial-target>`
     - ``'category'``
     - a label packet name
   * - :ref:`focal_gamma <mgh-focal-gamma>`
     - ``0.0``
     - float >= 0
   * - :ref:`multinomial_max_iter <mgh-multinomial-max-iter>`
     - ``2000``
     - integer > 0
   * - :ref:`emission_smoothness_derivative_order <mgh-emission-optimiser>`
     - ``2``
     - ``1``, ``2``
   * - :ref:`emission_huber_delta <mgh-emission-optimiser>`
     - ``1.0``
     - float > 0
   * - :ref:`emission_learning_rate <mgh-emission-optimiser>`
     - ``0.001``
     - float > 0
   * - :ref:`manifold_max_iter <mgh-emission-optimiser>`
     - ``5000``
     - integer > 0
   * - :ref:`emission_tol <mgh-emission-optimiser>`
     - ``0.0001``
     - float > 0
   * - :ref:`emission_random_state <mgh-emission-optimiser>`
     - ``0``
     - any integer
   * - :ref:`init_kmeans_n_init <mgh-init-kmeans-n-init>`
     - ``3``
     - integer > 0
   * - :ref:`init_self_transition <mgh-init-self-transition>`
     - ``0.8``
     - float in ``(0, 1)``
   * - :ref:`init_bias_sd <mgh-init-bias-sd>`
     - ``0.3``
     - float >= 0

.. _mgh-emission-type:

``emission_type``
   The per-state observation model.

   :Options:
      * ``'manifold'`` — behaviour → 2-D acoustic-manifold position, von Mises density
      * ``'multinomial'`` — behaviour → discrete USV category, categorical density
   :Default: ``'manifold'``
   :Notes: any other value raises.

.. _mgh-transition-mode:

``transition_mode``
   How the hidden-state transitions are modelled.

   :Options:
      * ``'static'`` — fits the EM-based engine with a stationary ``K × K`` transition matrix
      * ``'input_driven'`` — fits a direct-marginal engine whose transition *into* each time bin is a per-state GLM of the behavioural design (Calhoun/Pillow/Murthy), validated on the Coen-2014 fly to reproduce its state-selection without the state collapse the EM engine suffers on weakly-identified data
   :Default: ``'static'``
   :Notes: input-driven is defined for both emissions (categorical → reference-coded ``InputDrivenGLMHMM``; manifold → torus product-von-Mises ``InputDrivenManifoldGLMHMM``, which requires a torus manifold). Any other value raises; ``'input_driven'`` with ``emission_type='manifold'`` requires ``vocal_features.usv_manifold_metric`` ``'torus'`` and raises on euclidean.

.. _mgh-history-frames:

``history_frames``
   Number of most-recent behavioural-history frames kept per feature (the emission's temporal-filter length ``n_time_bins``).

   :Options: a positive integer (not validated)
   :Default: ``75``
   :Notes: a value at or above the stored history window's length (``model_params.filter_history`` × ``io.camera_sampling_rate`` at extraction) keeps the whole window.

.. _mgh-cv-folds:

``cv_folds``
   Number of cross-validation folds over the development sessions used to select the state count.

   :Options: an integer ``≥ 2``
   :Default: ``5``
   :Notes: it is capped at the number of development sessions, and a run with fewer than two development folds (including ``cv_folds`` ``1``) raises.

.. _mgh-n-states:

``n_states_min`` / ``n_states_max``
   Inclusive range of latent state counts ``K`` swept during selection.

   :Options: positive integers with ``n_states_min`` ≤ ``n_states_max`` (not validated; a reversed range sweeps no state count)
   :Default: ``2`` and ``6``

.. _mgh-n-em-iters:

``n_em_iters``
   Maximum Baum-Welch EM iterations per restart (``'static'`` engine only).

   :Options: a positive integer (not validated)
   :Default: ``100``

.. _mgh-n-lbfgs:

``n_lbfgs``
   L-BFGS iterations per restart for the ``'input_driven'`` direct-marginal engines.

   :Options: a positive integer (not validated)
   :Default: ``500``

.. _mgh-n-restarts:

``n_restarts``
   Number of random-initialisation restarts.

   :Options: a positive integer (not validated)
   :Default: ``5``
   :Notes: the best by training log-likelihood is kept, so the reserved held-out sessions are scored once and never shape the fitted model.

.. _mgh-n-kappa-rounds:

``n_kappa_rounds``
   The input-driven manifold engine alternates L-BFGS on the regressors and transitions (at a frozen von Mises concentration) with a responsibility-weighted global concentration MLE; this is the number of such alternations per restart.

   :Options: a positive integer (not validated)
   :Default: ``3``
   :Notes: the alternation is what keeps the concentration from collapsing to the uniform torus density. Read only by the input-driven manifold engine.

.. _mgh-em-tol:

``em_tol``
   Relative EM convergence tolerance (``'static'`` engine only).

   :Options: a positive float (not validated)
   :Default: ``0.0001``

.. _mgh-transition-pseudocount:

``transition_pseudocount``
   Laplace pseudocount smoothing the static transition and initial-state expected counts (``'static'`` engine only).

   :Options: a non-negative float (not validated; ``0`` leaves the counts unsmoothed)
   :Default: ``1.0``

.. _mgh-lambda-smooth-l2-reg:

``lambda_smooth`` / ``l2_reg``
   The temporal-smoothness and ridge penalties on each state's *static-engine* emission GLM.

   :Options: non-negative floats (not validated; ``0`` switches the penalty off)
   :Default: ``lambda_smooth`` ``100.0`` and ``l2_reg`` ``0.1``

.. _mgh-input-driven-lambda-smooth:

``input_driven_lambda_smooth``
   The first-difference temporal-smoothness coefficient ``r`` for the *input-driven* engines.

   :Options: a non-negative float (not validated)
   :Default: ``0.05``
   :Notes: a separate key because its scale, ``~0.05``, differs from the static-engine ``lambda_smooth`` by orders of magnitude. Read only by the input-driven engines.

.. _mgh-multinomial-target:

``multinomial_target``
   The categorical label column read as the target when ``emission_type='multinomial'`` (``'category'``, the QLVM category packet); events whose label is NaN are dropped.

   :Options: the name of a label packet in the modeling input pickle, which the manifold extraction writes only as ``'category'`` (and only when ``vocal_features.usv_category_column_name`` is set)
   :Default: ``'category'``
   :Notes: a packet missing from a session raises. Read only when ``emission_type='multinomial'``.

.. _mgh-focal-gamma:

``focal_gamma``
   Focal-loss focusing parameter of the static multinomial emission (``0.0`` = plain cross-entropy).

   :Options: a non-negative float (not validated; ``0.0`` is plain cross-entropy)
   :Default: ``0.0``
   :Notes: read only by the static multinomial emission.

.. _mgh-multinomial-max-iter:

``multinomial_max_iter``
   Optimiser iterations for the static multinomial emission's per-state classifier.

   :Options: a positive integer (not validated)
   :Default: ``2000``

.. _mgh-emission-optimiser:

``emission_smoothness_derivative_order`` / ``emission_huber_delta`` / ``emission_learning_rate`` / ``manifold_max_iter`` / ``emission_tol`` / ``emission_random_state``
   The optimiser of each state's *static-engine* emission.

   :Options:
      * ``emission_smoothness_derivative_order`` — the finite-difference order of the temporal-smoothness penalty: ``1`` or ``2`` (default ``2``, the curvature penalty); the euclidean manifold and multinomial emission regressors raise on any other value; the torus regressor accepts any positive integer below ``history_frames``
      * ``emission_huber_delta`` — the Huber transition point of the manifold emission's residual loss (default ``1.0``, manifold units); a positive float
      * ``emission_learning_rate`` — the initial Adam step size (default ``0.001``, cosine-decayed to zero); a positive float
      * ``manifold_max_iter`` — the manifold emission's step cap (default ``5000``; the multinomial emission's is ``multinomial_max_iter``); a positive integer
      * ``emission_tol`` — the parameter-change convergence tolerance (default ``0.0001``); a positive float
      * ``emission_random_state`` — the weight-initialisation seed (default ``0``); any integer
   :Default: ``2``, ``1.0``, ``0.001``, ``5000``, ``0.0001`` and ``0``
   :Notes: none of the last five is validated. The values are those the emission regressors used before they became settings, and all six are recorded in the result's ``metadata``. On a torus manifold the emission regressor is fitted in closed form, so the Huber, step-size, step-cap and tolerance keys act only on a euclidean manifold.

.. _mgh-init-kmeans-n-init:

``init_kmeans_n_init``
   The number of k-means initialisations (the best by inertia is kept) behind the structured initial state responsibilities of every static-engine EM restart.

   :Options: a positive integer (not validated)
   :Default: ``3``
   :Notes: the k-means runs on the targets, one-hot encoded for a categorical target.

.. _mgh-init-self-transition:

``init_self_transition``
   The diagonal of the static engine's initial transition matrix (default ``0.8``, so states start persistent over a bout); the remaining ``0.2`` is split evenly over the other states.

   :Options: a float strictly inside ``(0, 1)`` (any other value raises)
   :Default: ``0.8``
   :Notes: EM re-estimates the matrix from the first iteration on.

.. _mgh-init-bias-sd:

``init_bias_sd``
   The standard deviation of the normal draw that initialises the input-driven engines' transition biases and the categorical emission's biases at every restart.

   :Options: a non-negative float (not validated)
   :Default: ``0.3``

**bout_offset** — the settings of the ``'bout_offset'`` target of ``VocalOnsetModelingPipeline``
(see :ref:`Bout offsets <modeling-bout-offsets>`); read only when ``model_params.model_target_vocal_type``
is ``'bout_offset'``.

.. code-block:: json

    "bout_offset": {
        "filter_history": 1.0,
        "negative_scheme": "cross_bout",
        "min_singing_after_negative_seconds": 0.5,
        "time_since_bout_onset_tolerance_seconds": 0.1,
        "max_negatives_per_bout": 3
    }

.. list-table:: ``bout_offset`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`filter_history <mbo-filter-history>`
     - ``1.0``
     - seconds >= one frame
   * - :ref:`negative_scheme <mbo-negative-scheme>`
     - ``'cross_bout'``
     - ``'cross_bout'``, ``'within_bout'``
   * - :ref:`min_singing_after_negative_seconds <mbo-min-singing-after-negative-seconds>`
     - ``0.5``
     - seconds >= 0
   * - :ref:`time_since_bout_onset_tolerance_seconds <mbo-time-since-bout-onset-tolerance-seconds>`
     - ``0.1``
     - seconds >= 0
   * - :ref:`max_negatives_per_bout <mbo-max-negatives-per-bout>`
     - ``3``
     - ``null``, or integer > 0

.. _mbo-filter-history:

``filter_history``
   The history window (s) of this target, deliberately its own key.

   :Options: a positive number of seconds, at least one frame (``1 / camera_sampling_rate``), since the window is ``floor(camera_sampling_rate * filter_history)`` frames (not validated)
   :Default: ``1.0``
   :Notes: ``model_params.filter_history`` also sets the silent-tile width, the clean-history criterion of ``'bout_onset'`` mode and the ``'state'`` grid, none of which this target uses, and the offset question lives on a shorter timescale (the kinematic changes that precede a bout end sit in the last ~0.5 s).

.. _mbo-negative-scheme:

``negative_scheme``
   How the negative (an interior USV of a bout that continued) is drawn; see below.

   :Options:
      * ``'cross_bout'`` — one negative per positive, from a different bout, matched on the time since bout onset
      * ``'within_bout'`` — the latest qualifying USV of the same bout
   :Default: ``'cross_bout'``
   :Notes: any other value raises ``ValueError``.

.. _mbo-min-singing-after-negative-seconds:

``min_singing_after_negative_seconds``
   A negative's bout must keep singing for at least this long after the negative's USV, so the negative sits outside the ending itself (the measured ending process spans ~0.3 s).

   :Options: a number of seconds ``>= 0`` (not validated; ``0`` admits every interior USV)
   :Default: ``0.5``
   :Notes: measured from the negative USV's stop to the bout's last stop, under both schemes.

.. _mbo-time-since-bout-onset-tolerance-seconds:

``time_since_bout_onset_tolerance_seconds``
   ``'cross_bout'`` only: how far apart the time since bout onset of a positive and its negative may be.

   :Options: a number of seconds ``>= 0`` (not validated)
   :Default: ``0.1``
   :Notes: the nearest unused candidate within the tolerance is taken; a positive with none is dropped. Ignored under ``'within_bout'``.

.. _mbo-max-negatives-per-bout:

``max_negatives_per_bout``
   ``'cross_bout'`` only: cap on negatives drawn from any one bout (``null`` for none), so that a few long bouts cannot supply most of the negative class.

   :Options:
      * ``null`` — no cap
      * a positive integer — the cap (not validated; ``0`` would admit no negative at all)
   :Default: ``3``
   :Notes: ignored under ``'within_bout'``, which draws one negative per bout by construction.

**behavioral_response** — the inverted analysis: does a partner's vocal trace predict a
*behavioural* variable (see :ref:`Behavioral response <modeling-behavioral-response>` below)?

.. code-block:: json

    "behavioral_response": {
        "response_mouse_index": 1,
        "response_features": ["speed", "neck_elevation", "allo_roll", "allo_pitch",
                              "ego_yaw", "back_pitch", "back_yaw", "tail_curvature"],
        "history_seconds": 4.0,
        "target_window_seconds": 0.5,
        "target_bin_seconds": 0.05,
        "post_bout_silence_seconds": 0.5,
        "deep_silence_margin_seconds": 10.0,
        "covariate_summary_seconds": [0.5, 4.0],
        "covariate_transform": "yeo_johnson",
        "covariate_features": "both",
        "duration_n_bins": 3,
        "report_alpha": 0.05,
        "per_session_min_events_per_arm": 3,
        "matched_divergence": {
            "pre_seconds": 4.0,
            "post_clean_seconds": 4.0,
            "window_seconds": 4.0,
            "pre_window_seconds": 4.0,
            "baseline_start_seconds": 4.0,
            "baseline_end_seconds": 0.5,
            "silence_grid_seconds": 0.5,
            "match_caliper_sd": 0.1
        }
    }

.. list-table:: ``behavioral_response`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`response_mouse_index <mbr-response-mouse-index>`
     - ``1``
     - ``0`` (male), ``1`` (female)
   * - :ref:`response_features <mbr-response-features>`
     - the eight egocentric features
     - names from ``kinematic_features.egocentric``
   * - :ref:`history_seconds <mbr-history-seconds>`
     - ``4.0``
     - seconds >= one frame
   * - :ref:`target_window_seconds <mbr-target-window-seconds>`
     - ``0.5``
     - seconds >= one frame
   * - :ref:`target_bin_seconds <mbr-target-bin-seconds>`
     - ``0.05``
     - seconds >= one frame, <= ``target_window_seconds``
   * - :ref:`post_bout_silence_seconds <mbr-post-bout-silence-seconds>`
     - ``0.5``
     - seconds >= 0
   * - :ref:`deep_silence_margin_seconds <mbr-deep-silence-margin-seconds>`
     - ``10.0``
     - seconds > 0
   * - :ref:`covariate_summary_seconds <mbr-covariate-summary-seconds>`
     - ``[0.5, 4.0]``
     - non-empty list of seconds > 0
   * - :ref:`covariate_transform <mbr-covariate-transform>`
     - ``'yeo_johnson'``
     - ``'yeo_johnson'``, ``'derived'``, ``'normal_scores'``, ``'linear'``
   * - :ref:`covariate_features <mbr-covariate-features>`
     - ``'both'``
     - ``'self'``, ``'partner'``, ``'both'``
   * - :ref:`duration_n_bins <mbr-duration-n-bins>`
     - ``3``
     - integer > 0
   * - :ref:`report_alpha <mbr-report-alpha>`
     - ``0.05``
     - float in ``(0, 1)``
   * - :ref:`per_session_min_events_per_arm <mbr-per-session-min-events-per-arm>`
     - ``3``
     - integer > 0
   * - :ref:`matched_divergence.pre_seconds <mbr-md-pre-seconds>`
     - ``4.0``
     - seconds > 0
   * - :ref:`matched_divergence.post_clean_seconds <mbr-md-post-clean-seconds>`
     - ``4.0``
     - seconds > 0
   * - :ref:`matched_divergence.window_seconds <mbr-md-window>`
     - ``4.0``
     - seconds > 0
   * - :ref:`matched_divergence.pre_window_seconds <mbr-md-window>`
     - ``4.0``
     - seconds > 0
   * - :ref:`matched_divergence.baseline_start_seconds <mbr-md-baseline>`
     - ``4.0``
     - seconds > ``baseline_end_seconds``
   * - :ref:`matched_divergence.baseline_end_seconds <mbr-md-baseline>`
     - ``0.5``
     - seconds > 0, < ``baseline_start_seconds``
   * - :ref:`matched_divergence.silence_grid_seconds <mbr-md-silence-grid-seconds>`
     - ``0.5``
     - seconds > 0
   * - :ref:`matched_divergence.match_caliper_sd <mbr-md-match-caliper-sd>`
     - ``0.1``
     - float > 0

.. _mbr-response-mouse-index:

``response_mouse_index``
   Which mouse's behaviour is predicted, by **absolute slot index** (``0`` is always the male, ``1`` always the female). **This is the only mouse index you set.**

   :Options:
      * ``0`` — predict the male
      * ``1`` — predict the female
   :Default: ``1``
   :Notes:
      * **Why absolute** — deliberately *not* the relative ``self.`` / ``other.`` role keys used elsewhere: those are defined against ``model_params.model_predictor_mouse_index`` and cannot be read on their own.
      * **Derived predictor index** — ``model_params.model_predictor_mouse_index`` decides whose *vocalizations* are ingested and carries the opposite meaning on the same 0/1 axis; since the partner's vocalizations are by definition the other animal's, it is **derived** as ``1 - response_mouse_index`` on a private copy of the settings (the caller's dict is never mutated, and the shipped value the five vocal pipelines read is untouched). So ``response_mouse_index: 1`` predicts the female from the male's vocalizations, with nothing to keep in step by hand.
      * **Validation** — any other value raises.

.. _mbr-response-features:

``response_features``
   The behavioural features used as response variables, every one extracted in the same pass (the anchors are identical across features, only the target changes).

   :Options: a non-empty list of names from ``kinematic_features.egocentric`` (a name outside it raises before any session is read)
   :Default: the eight egocentric features (see JSON above)
   :Notes: each must be one of ``kinematic_features.egocentric``; an unknown name is rejected before any session is read. Read from the raw per-session feature table **before** z-scoring, so the responses stay in native units — the likelihood of each is **derived** from its support (see :ref:`Behavioral response <modeling-behavioral-response>`): a non-negative feature (plain, or folded by ``abs_features`` / ``smooth_abs_features``) is fitted as a Gamma GAM with a log link, a signed one as Gaussian.

.. _mbr-history-seconds:

``history_seconds``
   Seconds of behavioural history preceding each anchor, the window the covariate summaries are taken over.

   :Options: positive seconds spanning at least one frame at ``io.camera_sampling_rate`` (a shorter value raises)
   :Default: ``4.0``
   :Notes: block-local rather than shared with ``model_params.filter_history``, mirroring how ``glm_hmm`` carries its own ``history_frames``. It also sets the stride of the deep-silence anchors (below), so no two of those rows share a history sample.

.. _mbr-target-window-seconds:

``target_window_seconds``
   Width of the forward window the response is averaged over for the headline (``window``) fit.

   :Options: positive seconds spanning at least one frame (a shorter value raises)
   :Default: ``0.5``
   :Notes: averaging suppresses the frame-to-frame differentiation noise in a single 6.7 ms sample and breaks the near-determinism that would otherwise tie the target to the last frame of its own history.

.. _mbr-target-bin-seconds:

``target_bin_seconds``
   Width of the bins the forward window is tiled into for the time-course fit (``0.05`` → ten bins across a ``0.5`` s window).

   :Options: positive seconds spanning at least one frame and no longer than ``target_window_seconds`` (either violation raises)
   :Default: ``0.05``
   :Notes: the window is tiled with rounded bin **edges** rather than a fixed frame count, so at 150 fps the bins alternate between 7 and 8 frames and the last frames of the window are not dropped. Must not exceed ``target_window_seconds``.

.. _mbr-post-bout-silence-seconds:

``post_bout_silence_seconds``
   A male bout offset becomes a vocal anchor only when the next bout starts at least this long after it, so the forward window follows the bout rather than straddling the next one.

   :Options: non-negative seconds (not validated)
   :Default: ``0.5``
   :Notes: it is set equal to ``target_window_seconds`` on purpose: the requirement selects on the future (bigger bouts are followed sooner by the next bout), so anything longer than the window discards long bouts for cleanliness the analysis never uses (see the note in :ref:`Behavioral response <modeling-behavioral-response>`).

.. _mbr-deep-silence-margin-seconds:

``deep_silence_margin_seconds``
   The second, *deep-silence* control: anchors tiled every ``history_seconds`` through the stretches with no male bout within this margin on either side.

   :Options: positive seconds (not validated)
   :Default: ``10.0``
   :Notes: a tile whose window touches any vocalization of the predictor mouse is dropped. Both controls are extracted in the same pass; ``behavioral_response_contrast`` picks one with its ``control`` argument (``'inter_bout_silence'`` or ``'deep_silence'``). Inter-bout silence keeps the animals inside an ongoing interaction but carries over the previous bout's effect; deep silence escapes the carry-over at the cost of a larger imbalance in social distance (measured ``1.09`` SD in ``nose-nose`` against ``0.74`` SD for inter-bout silence, every other covariate under ``0.32`` SD).

.. _mbr-covariate-summary-seconds:

``covariate_summary_seconds``
   The pre-anchor windows each covariate is averaged over, one column per window per feature (``[0.5, 4.0]`` → ``<feature>__mean_0.5s`` and ``<feature>__mean_4s``).

   :Options: a non-empty list of positive seconds, each spanning at least one frame and meant to be no longer than ``history_seconds`` (not validated; a window wider than the history averages frames before it, and a zero-frame window gives an all-NaN column)
   :Default: ``[0.5, 4.0]``
   :Notes: the covariates are summaries rather than lag histories because their job is to hold pre-anchor state fixed, and these features are slow (autocorrelation horizons ~0.75 s for ``speed`` to ~6.8 s for ``nose-nose``), so finer sub-windows would be collinear rather than informative. The responder's own, the vocalizer's and the dyadic features are all stored.

.. _mbr-covariate-transform:

``covariate_transform``
   How the stored (pooled z-scored) covariates enter the contrast's design, applied at **fit time** (read from the settings, not from the artifact, so an artifact written before the key existed is still valid input).

   :Options:
      * ``'yeo_johnson'`` — fits one Yeo-Johnson power parameter per covariate by maximum likelihood on its native-scale values and re-standardises
      * ``'derived'`` — applies the response's own support rule (``log`` for a non-negative covariate, identity for a signed one)
      * ``'normal_scores'`` — replaces each covariate by the Gaussian quantile of its rank
      * ``'linear'`` — uses the stored z-scores as they are
   :Default: ``'yeo_johnson'``
   :Notes: the non-linear options exist because the response is measured to be linear in the dominant covariate's *rank* and strongly concave in its value, a misfit an order of magnitude larger than the effect under test. Any other value raises when the contrast is fitted.

.. _mbr-covariate-features:

``covariate_features``
   Which animal's pre-anchor kinematics adjust the contrast, also a fit-time choice.

   :Options:
      * ``'self'`` — the responder's
      * ``'partner'`` — the vocalizer's
      * ``'both'`` — both animals'
   :Default: ``'both'``
   :Notes: the dyadic columns stay in under every choice. Any other value raises when the contrast is fitted.

.. _mbr-duration-n-bins:

``duration_n_bins``
   The number of equal-count quantile bands bout duration is cut into for the secondary banded design, in which each band is interacted with the vocal indicator so every band gets its own step against silence (``3`` = terciles).

   :Options: a positive integer no larger than the number of distinct bout durations among the vocal rows (more bands than that raises when the contrast is fitted)
   :Default: ``3``
   :Notes: the primary design uses one continuous log-duration slope (``vocal_x_log_duration``) instead and does not read this key. Bands are kept as a check because duration is heavily skewed (median ``0.43`` s with a long tail).

.. _mbr-report-alpha:

``report_alpha``
   The console label of each contrast term: ``SIG`` when its cluster-robust p-value is below this level, ``ns`` otherwise.

   :Options: a float in ``(0, 1)`` (not validated)
   :Default: ``0.05``
   :Notes: it labels the printed line only; the stored result keeps every p-value.

.. _mbr-per-session-min-events-per-arm:

``per_session_min_events_per_arm``
   The unadjusted per-session effect (vocal minus silent mean, on the log scale for a positive feature) is computed for a session only when it holds at least this many usable rows in each arm.

   :Options: a positive integer (not validated)
   :Default: ``3``
   :Notes: so a session with one or two rows on a side cannot contribute an arbitrary difference to the between-session spread.

.. _mbr-matched-divergence:

``matched_divergence``
   The design-based companion analysis, ``MatchedDivergencePipeline.extract_and_save_matched_divergence`` (see :ref:`Matched divergence <modeling-matched-divergence>`), which holds pre-anchor state fixed by matching instead of by covariate adjustment.

   :Notes: every duration is in seconds. Every ``*_seconds`` key takes positive seconds (not validated, except as noted for ``baseline_start_seconds`` / ``baseline_end_seconds`` and ``silence_grid_seconds`` below), and ``match_caliper_sd`` a positive float (not validated). Shipped ``4.0`` for ``pre_seconds``, ``post_clean_seconds``, ``window_seconds``, ``pre_window_seconds`` and ``baseline_start_seconds``, ``0.5`` for ``baseline_end_seconds`` and ``silence_grid_seconds``, and ``0.1`` for ``match_caliper_sd``. In every one of these silence tests, segments flagged as noise are not vocalizations and are ignored when ``vocal_features.exclude_noise_usvs`` is ``true``.

.. _mbr-md-pre-seconds:

``matched_divergence.pre_seconds``
   A candidate silent anchor must lie at least this long after the last vocalization of **any** animal.

   :Options: positive seconds (not validated)
   :Default: ``4.0``

.. _mbr-md-post-clean-seconds:

``matched_divergence.post_clean_seconds``
   No vocalization of any animal may start (or still be ongoing) within this long after an anchor, vocal and silent alike.

   :Options: positive seconds (not validated)
   :Default: ``4.0``
   :Notes: the bout offsets that fail it are dropped.

.. _mbr-md-window:

``matched_divergence.window_seconds`` / ``matched_divergence.pre_window_seconds``
   The stored per-anchor curve spans ``[-pre_window_seconds, +window_seconds]`` around the anchor.

   :Options: positive seconds (not validated)
   :Default: ``4.0`` and ``4.0``

.. _mbr-md-baseline:

``matched_divergence.baseline_start_seconds`` / ``matched_divergence.baseline_end_seconds``
   The matching window is the mean over ``[-baseline_start_seconds, -baseline_end_seconds]`` before the anchor (``start`` must exceed ``end``).

   :Options: positive seconds (not validated, except that ``baseline_start_seconds`` must exceed ``baseline_end_seconds`` in frames, which raises otherwise)
   :Default: ``4.0`` and ``0.5``
   :Notes: it ends ``0.5`` s *before* the anchor because these features are slow (``tau(1/e)`` 0.35–0.54 s), so a window reaching the anchor would already contain the divergence under way.

.. _mbr-md-silence-grid-seconds:

``matched_divergence.silence_grid_seconds``
   Spacing of the grid the candidate silent anchors are drawn from.

   :Options: positive seconds; it must be above ``0``, since it is the step of the anchor grid
   :Default: ``0.5``

.. _mbr-md-match-caliper-sd:

``matched_divergence.match_caliper_sd``
   Every silent anchor whose baseline lies within this many pooled standard deviations of a vocal anchor's is averaged into that anchor's control curve (a vocal anchor has a median of 82 eligible controls).

   :Options: a positive float (not validated)
   :Default: ``0.1``

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
    ).extract_and_save_category_input_data(target_category=3)

    # Vocal categories across all categories jointly (multinomial)
    MultinomialModelingPipeline(
        modeling_settings_dict=None
    ).extract_and_save_multinomial_input_data()

    # Continuous manifold position (2-D QLVM torus regression)
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
        "vocal_predictor_type": "categories_rate", "vocal_predictor_classes": [["usv"]], "vocal_predictor_partner_only": true,
        "filter_history_seconds": 4, "filter_history_frames": 600,
        "ibi_thresholds": {"male": 0.42, "female": 0.55},
        "analysis_specific": { "...": "..." },
        "git_commit": "...", "settings_sha256": "...", "created_utc": "...", "package_version": "..."
    }

* **cohort / scope** — ``experimental_condition`` (the cohort label), ``session_ids``, and ``n_events_per_session`` (how many positive / negative events each session contributed).
* **mouse roles** — ``predictor_idx`` / ``target_idx`` and their sexes.
* **behavioral features** — ``feature_zoo_full`` (everything requested) vs ``feature_zoo_kept`` (what survived the collinearity / timescale audits).
* **vocal-input shape** — ``vocal_predictor_type``, ``vocal_predictor_classes``, ``vocal_predictor_partner_only``, ``vocal_predictor_smoothing_sd``.
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
    ContinuousModelingPipeline      ->  { "X", "Y", "w", ["category"] }

* **VocalOnsetModelingPipeline** — ``usv_feature_arr`` = positive onset windows, ``no_usv_feature_arr`` = silent-epoch (negative) windows. ``analysis_specific``: ``model_target_vocal_type``, ``usv_bout_time``, ``usv_per_bout_floor``, and in ``'bout_offset'`` mode the whole ``bout_offset`` block.
* **BoutParameterPipeline** — ``X`` = the bout-onset feature windows, ``y`` = the per-bout regression target (selected by ``model_target_variable``), ``groups`` = the session grouping. ``analysis_specific``: ``target_variable``.
* **VocalCategoryModelingPipeline** — ``target_feature_arr`` = windows for the chosen target category, ``other_feature_arr`` = windows for the pooled "other". ``analysis_specific``: ``target_category``.
* **MultinomialModelingPipeline** — ``X`` = per-USV feature windows, ``y`` = each USV's category label. ``analysis_specific``: ``categories_kept``, ``class_counts``, ``usv_category_number`` (C, the number of classes the cohort actually holds, which the model is fit with), ``usv_category_column_name``, ``bundle_category_number`` (k, the number of categories of the QLVM category bundle, ``os_utils.load_qlvm_category_bundle``, R-1 … R-k; ``null`` when the label column is not ``qlvm_category``) and ``bundle_categories_missing`` (the bundle labels with no USVs in the cohort, ``[]`` when C = k). When C < k the extraction prints a warning naming the missing categories (e.g. ``missing: R-4``); the model is still fit with the C observed classes.
* **ContinuousModelingPipeline** — ``X`` = per-USV feature windows, ``Y`` = the 2-D acoustic-manifold position target, ``w`` = inverse-density KDE weights (always present); ``category`` (each USV's QLVM category, the regular map's ``qlvm_category`` whatever map the manifold columns come from — the categories are defined on the regular map and label the USV; also the acoustic region of the torus macro score, the equal-region reweighting and model selection) is added when the summaries carry that column. ``analysis_specific``: ``usv_manifold_column_names``, ``manifold_metric``, ``manifold_period``, ``manifold_weight_clip_percentile`` (the ``w`` cap, ``vocal_features.usv_manifold_weight_clip_percentile``) and ``usv_category_column_name``.

.. note::

   **Modeling onsets for a single USV category.** By default
   ``VocalOnsetModelingPipeline`` derives positive onset events from the target
   mouse's vocal segments of the **target classes** (``onset_target_classes``): by
   default ``["usv"]``, pure USVs, so squeaks are neither USV onsets nor bout members
   (a squeak between two USVs neither joins nor splits a bout). To model **squeak
   onsets** — e.g. the female's, which are plentiful in courtship when her USVs
   are not — set ``model_target_vocal_type = 'individual'`` and
   ``onset_target_classes = ["squeak"]`` (or ``["squeak", "overlap"]``); this
   replaces the VAE-era route of targeting squeak category 6. Squeaks are available
   outside the bout modes only, since squeaks do not come in bouts and the per-sex
   inter-bout thresholds are derived from USV intervals; asking for them in a bout
   mode raises. A summary without the ``usv`` / ``squeak`` columns raises.

   A USV and a squeak can fall in the same segment: DAS was trained to detect
   both, and the squeak classifier marks such a segment with ``usv`` and ``squeak``
   both true — the ``overlap`` class. It is a target only when listed: under
   ``["usv"]`` and under ``["squeak"]`` it is dropped whole, while
   ``["usv", "overlap"]`` keeps it as a USV (onset and bout member, at the
   segment's onset) and ``["squeak", "overlap"]`` as a squeak (at the squeak's own
   onset, ``squeak_start``). Comparing the runs shows whether the
   overlap matters for a given question. The predictor side is set independently
   (``vocal_predictor_classes``): e.g. male bout onsets from every segment holding a
   USV, predicted from the female's squeaks only, is ``onset_target_classes =
   ["usv", "overlap"]`` with ``vocal_predictor_classes = [["squeak"]]`` and
   ``vocal_predictor_partner_only = true``.

   Within those target classes, when overall vocal output is too sparse for
   bout-onset modeling but one category is plentiful, you can restrict the
   positive onsets to a single category by setting two knobs in
   ``model_params``:

   - ``model_target_vocal_type = 'individual'`` — each qualifying USV onset
     (rather than a clustered bout onset) becomes a positive event;
   - ``onset_target_category = <int>`` — the category index to keep (e.g.
     ``3``). The *column* this index refers to is the existing
     ``vocal_features.usv_category_column_name``, so any existing label column
     (e.g. ``qlvm_category``) can be targeted. Leave it
     ``null`` (default) to pool all categories exactly as before. With a
     ``null`` ``usv_category_column_name`` a set ``onset_target_category``
     raises a ``ValueError`` before any session is loaded, and a summary that
     lacks the configured column raises when it is read (it no longer falls
     back to all USVs with a warning).

   Only the *positive* onsets are filtered, by target classes and by category
   alike: the vocal predictors follow their own ``vocal_predictor_classes``, and the
   silent-epoch (No-USV) negative reference is still computed over **every**
   vocalization of the mouse, squeaks and overlap segments included (and every
   one assigned to no animal), so these choices change only *which* onsets count
   as events — never the predictors or the negatives. The category filter
   is honoured in ``'individual'`` mode only; in ``'bout_onset'`` mode it is
   ignored, because the mixture-model inter-syllable-interval threshold used
   for bout grouping is calibrated on the all-USV interval distribution and
   would mis-group a category-sparsified sequence (a warning is printed if the
   setting is combined with a non-individual mode). When active, the chosen
   category column and index are embedded in the ``analysis_tag`` (e.g.
   ``individual_cat_qlvm_category_3``) and ``_input_metadata``, so the QLVM (quasi-Monte Carlo latent variable model)
   category column is unambiguous in every downstream artifact
   name and provenance block.

.. _modeling-diagnostics:

.. _modeling-bout-offsets:

Bout offsets
~~~~~~~~~~~~
``model_target_vocal_type: 'bout_offset'`` asks the mirror question of the bout-onset
target: given that he is singing, why does he stop now rather than continue. It
reuses the whole onset machinery — the same bout grouping, the same window slicing,
the same univariate runners, screen, forward selection and held-out evaluation —
and differs only in what a row is.

*Positive*: the offset of the last USV of every bout with at least
``usv_per_bout_floor`` USVs. No clean-history or clean-future requirement is
imposed: the inter-USV threshold of the bout definition already certifies the end
(a silence beyond it is a memoryless restart), and nothing after the offset enters
the design. On the courtship cohort this is every one of the ~16,000 male bouts.

*Negative*: an interior USV's offset, chosen by ``negative_scheme``:

* ``'cross_bout'`` — from ANOTHER bout of the same session, at the same time since
  that bout's onset (within the tolerance), in a bout that kept singing for at least
  ``min_singing_after_negative_seconds`` afterwards; one negative per positive,
  nearest in position, each candidate used once, at most ``max_negatives_per_bout``
  from one bout. Positives without a partner are dropped, so the two groups share the
  same distribution of position in the bout. The bouts that continue are longer by
  construction, so whatever distinguishes long from short bouts — on the cohort,
  mainly how much the female is moving — becomes a difference between the classes.
  That is the answer to "why does he stop sooner", not a confound: bout length is
  the outcome. Measured yield on the cohort: ~14,800 pairs from ~4,600 supplying
  bouts without a cap, ~10,500 pairs with the cap of 3.
* ``'within_bout'`` — from the SAME bout: the latest interior USV at least
  ``min_singing_after_negative_seconds`` before that bout's end. Positive and negative
  then share the bout, the animal, the context and the bout's length, and only the
  approach to the end differs; only bouts long enough to hold such a USV enter
  (~3,000 of ~16,000 on the cohort, all long ones).

Under both schemes the pairing is only the sampling recipe: the loader returns
``positive_events`` and ``negative_events`` exactly as the other modes do, and they
are pooled, balanced and fitted downstream without change. The mode's tag,
``bout_offset``, propagates into every artifact name, and the block is recorded in
``analysis_specific``.

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

Every feature is scored against a **null**: the same model refit with the target of
the training rows shuffled **across the whole training set**, pooled over its sessions
(onset labels, category labels, bout parameters, and on the acoustic manifold the torus
positions with their weights and regions), while the features stay in place and the
test rows keep their true targets. This breaks every pairing between behavior and
vocal output, both the moment-to-moment pairing inside a session and any difference
between sessions (e.g. sessions with a faster partner also holding more bout onsets),
so the null learns nothing and a feature beats it by carrying any relationship to vocal
output that generalizes to held-out sessions. Sessions are still kept whole on either
side of every split (see ``split_strategy``); only the null's target is pooled. The
regularization each fold picks (``LogisticRegressionCV`` for onset and category,
``GridSearchCV`` for bout parameters) is chosen on inner folds that hold out whole training
sessions, as the outer folds do (both classes kept in every inner fold; event folds when a
training set has a single session). The class balancing and the row shuffling draw from a
generator seeded with ``model_validation.random_seed``, never from NumPy's global state. The
held-out score's null and the model selections' held-out nulls follow the same rule.

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
* **``_run_metadata``** — how the fits ran: ``model_engine``, ``basis_function``, ``null_strategy``, ``n_outer_folds``, ``split_strategy``, ``random_seed_outer``, the engine hyperparameters (``pygam_hyperparameters`` / ``sklearn_hyperparameters`` for the CPU engines; ``jax_hyperparameters`` for the multinomial and manifold runs, which records the block's keys, its ``temporal_basis`` sub-block and ``effective_penalty`` — the ``lambda_smooth``, ``l2_reg``, ``smoothness_derivative_order`` and ``smoothness_reflective_edges`` the fit actually used, the block's own on frames and the GAM-mirroring P-spline penalty under a B-spline basis), and git / settings provenance. **``_consolidation_metadata``** records the merge audit (how many per-feature files, when, and their paths).
* **multinomial / continuous targets** — the per-fold metrics instead live under an ``actual.folds.metrics`` sub-dict (with ``y_true`` / ``y_pred`` / ``classes`` alongside), rather than as flat top-level arrays.

.. _modeling-model-selection:

Model selection
---------------
Greedy forward-stepwise selection stacks features on top of the univariate
ranking, adding at each step the feature whose contribution most improves
the held-out score, subject to an acceptance gate that compares the candidate
and the incumbent **paired per fold** (the two models are nested and scored on
the same folds, so the fold-difficulty term they share cancels in the per-fold
difference). The onset, binomial-category, bout-parameter and multinomial
selectors apply the **one-standard-error (1SE) rule** to that paired improvement:
a step is kept only when the mean per-fold improvement exceeds the standard error
of the per-fold differences. The acoustic-manifold selector uses the **fold-grain
paired-margin bootstrap gate** described in the note below instead (a step is kept
only when its per-fold improvement has a ``selection_ci_level`` fold-bootstrap CI
whose lower bound exceeds ``0``). ``use_top_rank_as_anchor=True`` seeds step 0 with
the top univariate feature; ``p_val`` is the significance level of the univariate
screen that admits candidates (Bonferroni-corrected for the discrete targets,
the Benjamini–Hochberg ``q`` on the manifold).

.. note::

   **Significance baseline for the discrete targets (vocal onsets, binomial
   USV categories).** Every univariate fit is evaluated against a
   *label-shuffle permutation null*: the same estimator is re-fit on a copy of
   the **training** labels permuted across the training fold — breaking the
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
   margin), the per-feature p-value being an exact one-sided sign-flip
   permutation test over the folds (floor ``1 / 2**manifold_n_cv_folds``) — and an anchor /
   forward step is admitted only when its per-fold improvement over the current
   model has a ``selection_ci_level`` fold-bootstrap CI (``selection_n_bootstrap``
   resamples) whose lower bound exceeds ``0``. On the torus the current model the
   anchor (or first forward feature) must beat is a training-fold
   **intercept-only** predictor: every test event of a fold is predicted at the
   circular mean of that fold's training targets (weighted like the candidates'
   fit), scored with the same bundle and gated with the same per-fold margin, so
   a model that carries no behavioural information ties it and is rejected. The
   empirical-density draw (``null_model_free``, a random training target per test
   event) is kept in the Step-0 file as the reported chance floor and never gates
   on the torus; it used to be the gate baseline, which any prediction
   concentrated where the repertoire is dense beat without behavioural
   information. On euclidean, where the score is distance correlation (``0`` for
   any constant prediction, positively biased for every other), the bias-matched
   draw stays the gate baseline. This replaced an earlier fold-level Wilcoxon /
   one-standard-error gate.

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
     Arvanitidis pullback ``G = JᵀJ`` of the frozen decoder of the run's QLVM
     map (a conditional map's decoder at the configured
     ``pullback_condition_quantile`` of its conditioning value), measuring
     distance in decoded-spectrogram change rather than in raw latent
     coordinates). The two geodesic columns are ``NaN`` on euclidean manifolds
     and when their reference map is disabled or the decoder is unavailable;
     their absolute magnitudes are reweighted path-lengths (arbitrary scale), so
     they are read as a **Δ-vs-baseline**, exactly like the flat MAE.
   * **Region-balanced (macro) vs pooled distance** — every distance column
     comes in two averages: the pooled per-event mean (``euclidean_mae``,
     ``density_geodesic_mae``, ``pullback_geodesic_mae``) and its
     region-balanced twin (``euclidean_mae_macro``,
     ``density_geodesic_mae_macro``, ``pullback_geodesic_mae_macro``: the
     per-region mean error averaged with equal weight per acoustic region,
     ``manifold_metric.macro_region_mean``). The fit itself is
     **region-reweighted** (inverse-region-frequency sample weights), so a few
     dominant acoustic regions do not drown the rare ones during estimation, and
     the selection score ``vm_logscore`` is macro; the ``*_macro`` columns are the
     distance yardstick consistent with both. The pooled columns charge a
     region-balanced model for resting away from the dominant regions, a charge
     that on an unbalanced map (one region holding 7 % of the vocalizations)
     exceeds what the behavioural features recover, so the trajectory figure's
     error panels read the ``*_macro`` columns. Also reported:
     ``euclidean_mae_weighted`` (the pooled distance weighted by the
     inverse-density sample weights), ``mahalanobis_mae``, ``r2_spatial``, the
     per-axis MAEs and the pearson / spearman correlations.

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
per-feature / per-step pickle each. ``--analysis_type`` must name the pipeline
that built the input pickle (recorded in its ``_input_metadata.analysis_type``):
bout onset and bout offset inputs are ``onset``, bout durations ``params``. A
mismatch stops the univariate task, or the selection (for the univariate file
as well as the input pickle), at load time with exit status 1 and a message
naming the type to rerun with; an artifact that does not record the field is
not checked. Consolidate the model-selection steps with
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

* **``steps``** — an ordered list, one entry per forward-selection step. ``step_idx`` is the iteration, ``current_features`` are those already chosen, ``baseline_score`` is their held-out score (the chance floor at step 0), ``baseline_folds`` is the same model's PER-FOLD scores, and ``selected_feature`` is the feature accepted this step (``None`` marks the final, rejected step). ``baseline_folds`` is what the 1SE acceptance test pairs a candidate against: the incumbent and the candidate are nested and scored on the same folds, so the fold-difficulty term they share cancels in the per-fold difference but dominates either score on its own. Checkpoints written before this key existed resume against ``baseline_score`` as a constant, which reproduces the unpaired comparison that wrote them. For the multinomial and manifold selectors, step 0's ``selected_feature`` is the sentinel ``'null_model_free'``. The manifold Step-0 record also names the strategy the first feature is gated against in ``baseline_strategy`` (``'null_model_intercept'`` on the torus, ``'null_model_free'`` on euclidean), lists that strategy first in its ``candidates_summary``, and stores the empirical-density draw's fold-mean score separately as ``chance_floor_score`` (``chance_floor_strategy`` ``'null_model_free'``); ``baseline_score`` is the gate baseline's score. A resume refuses a manifold checkpoint whose ``_run_metadata.extra_knobs`` does not record the same ``step1_gate_baseline``, ``screen_p_value_test`` (``'exact_sign_flip'``) and ``manifold_n_cv_folds`` and starts fresh.
* **``candidates_summary``** — under each step, every candidate feature tested that step mapped to its per-fold metrics. For the discrete / regression targets these are flat per-fold arrays (``ll``, ``auc``, ``score``, ``f1``, ``brier``, ``ece``, ``mcc``, ``confusion_matrix``, ``n_iter`` / ``converged`` / ``fit_time``) plus aggregate ``mean_ll`` / ``se_ll``; the multinomial and manifold selectors nest these under a ``folds.metrics`` sub-dict (with ``y_true`` / ``y_pred`` / ``y_probs`` / ``classes``, the per-fold ``weights`` — always on the frame axis, ``n_features × filter_history_frames`` per class / output, whatever the ``temporal_basis`` — beside ``basis_coefficients``, the raw spline coefficients when ``temporal_basis.type`` is ``'bspline'`` and ``None`` otherwise, and the per-fold ``selected_lambda_smooth`` / ``selected_l2_reg`` regularisation choices — equal to the fixed ``*_fixed`` penalties when ``tune_regularization_bool`` is ``false``, and under a B-spline basis to ``temporal_basis.lambda_smooth_fixed`` and ``0``).
* **last accepted step** — additionally carries ``final_model_features`` (the cumulative selected set) and ``filter_shapes`` (the per-fold refit filters) of the published model, and ``heldout`` (the development refit scored once on the held-out sessions, ``None`` when the reserve is disabled). In the onset, category and bout-parameter selectors each held-out branch record (``actual`` / ``null``) carries ``heldout_fit_failed`` (bool) and ``heldout_fit_error`` (the exception text, or ``None``): a refit that raised leaves its metrics NaN, sets the flag and is warned about, so a NaN held-out number is never silent. The same two fields are on each held-out record (``heldout['actual']``, ``heldout['null']``) of the onset, binomial-category and bout-parameter univariate results (for bout parameters a failed held-out deviance metric is flagged too). Every modeling pickle is published atomically (written to a temporary sibling, then renamed), so an interrupted write leaves the previous file intact; a failed input-artifact save aborts the extraction.
* **metadata blocks** — ``_input_metadata`` and ``_univariate_metadata`` carry the upstream extraction / univariate provenance, ``_run_metadata`` the selection config, and ``_consolidation_metadata`` the merge audit.

Visualise the trajectory with ``plot_model_selection_results`` (binary / regression),
``plot_multinomial_selection_trajectory`` (multinomial), or
``plot_manifold_selection_trajectory`` (continuous manifold): each reads the
consolidated ``model_selection_final_*.pkl`` and shows the per-step held-out score
gain and the retained-feature filters. Called with no metric arguments, the manifold
plotter draws the score the run selected on (``_run_metadata['selection_metric']``:
``vm_logscore`` on the torus, ``dcor_xy`` on euclidean) and, as percent improvement
over chance, the region-balanced errors of the run's geometry: on the torus the
manifold-position error (``euclidean_mae_macro``) and the acoustic error
(``pullback_geodesic_mae_macro``), on euclidean the manifold-position error alone.

For the acoustic-manifold model the converged filters have a single dedicated
view, ``plot_manifold_filter_atlas`` (torus runs only), reading the same
consolidated artifact — a three-panel figure. The **vocal-space atlas** panel
tiles the torus with real USVs of the run's sessions: each node shows the
vocalization nearest to it in the production spectrogram store, masked by its
SAM masks and at its natural duration in the storage window, never
time-stretched (inferno on black, each tile peak-normalised; on the regular map
the QLVM category boundaries overlaid in white) as the key for the fields. The **filter magnitude**
panel draws the per-feature temporal magnitude ``|W(t)|`` over the history window
(one line per feature, coloured by behavioural category with opacity separating
features that share a category; averaged into display bins so the medium-scale
envelope reads cleanly). The **affinity filmstrips** panel samples each feature's
filter at ``n_time_slices`` instants from ``-history_window_sec`` to onset and
decodes each into the signed ``e(theta).W`` field over the torus (the positive end
of ``figures.diverging_cmap``, brown with the shipped ``BrBG_r`` = a +1 SD increase in
the feature just before onset drives the predicted vocalization toward that region;
the negative end, teal = away), on a shared diverging scale. The atlas tiles come
from the newest production spectrogram store under
``os_utils.SPECTROGRAMS_DIRECTORY`` (``os_utils.resolve_consolidated_h5_path``),
at the positions of the map the run's coordinates come from, ``<map>`` the map
prefix of the artifact's
``_input_metadata.analysis_specific.usv_manifold_column_names`` (the store's
``qlvm/<session>/<map>`` dataset: ``qlvm`` the regular map, the conditional maps
and ``qlvm_squeak`` their own; an artifact without that record predates the
conditional maps and is read as a regular-map run), restricted to the run's
sessions (``_input_metadata.session_ids``) and to embedded rows (status 0); each
node shows the nearest such vocalization (flat-torus distance). The decoder's
canonical reconstructions are time-stretched and are deliberately not shown. On
the regular map its white boundaries are the R-1 … R-k regions of the QLVM
category bundle (``os_utils.load_qlvm_category_bundle``,
``os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY``, the partition the summaries'
``qlvm_category`` was assigned from); the bundle partitions the regular map's
torus only, so a conditional or squeak map's atlas and filmstrips draw no
boundaries. Both come from the code constants, not from arguments or settings
paths (the GUI / CLI experimenter re-keying cannot move them), and the bundle
must be defined on the cell the store's positions were embedded with (the
bundle's map, ``os_utils.QLVM_CATEGORY_MAP``, resolves to the store's recorded
``qlvm_models/<map>`` ``model_id``), otherwise a ``ValueError`` is raised; the
store, the cell and the bundle's ``build_config.json`` identity are printed for
the figure's record. Colours come from
``visualizations_settings.json`` (``sequential_cmap`` / ``diverging_cmap``), and the temporal
filter's smoothness is governed by the per-observation ``lambda_smooth`` prior (see
the note above) — or, under ``temporal_basis.type = 'bspline'``, by the spline layer
itself and its ``temporal_basis.lambda_smooth_fixed``; the stored filters are on the
frame axis in both cases, so the atlas reads them the same way.

.. _modeling-cnn:

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
        "saliency_maps": {"category_1": {"contrastive_saliency": "<array>", "centroid": "...", "radius": "..."}},
        "cluster_geometry": {"...": "..."}           # optional — cluster centroids / radii
    }

* **``metadata``** — the run configuration: ``features_list`` (the ``F`` predictor order), the ``hyperparameters`` block, ``manifold_metric`` / ``manifold_period`` / ``output_encoding``, ``n_time_bins``, ``split_strategy``, and the source-pickle path.
* **``cross_validation``** — a list, one dict per spatial-CV fold. Each holds the fold's test-set ground truth ``Y_true`` ``(N, 2)`` and the three strategies' predictions ``Y_pred_actual`` / ``Y_pred_null`` / ``Y_pred_null_model_free`` (all ``(N, 2)``), plus the scalar wrap-aware ``error_actual`` / ``error_null`` / ``error_null_model_free`` that feed the skill-score and permutation test.
* **``feature_importance``** — permutation importance evaluated on ``best_fold_idx``: per-feature ``means`` / ``stds`` / ``snrs`` (mean Δerror, its spread, and the signal-to-noise ratio), ``ranked_features`` (sorted), and ``significant_features`` (SNR-thresholded).
* **``saliency_maps``** (optional) — one entry per acoustic cluster (keyed ``<segmentation>_<label>``, e.g. ``category_1``), each with a ``contrastive_saliency`` tensor (Input×Gradient over features × time) and the cluster ``centroid`` / ``radius``. **``cluster_geometry``** (optional) records the cluster centroids, radii, and nearest-neighbour distances that place the saliency insets.

.. _modeling-glm-hmm:

Latent vocal states (GLM-HMM)
-----------------------------
A GLM-HMM treats the animal as switching between a small number of latent
"behaviour → vocalization rules": each latent state owns its own GLM mapping the
recent behavioural history to the vocalization, and the animal transitions between
states over a vocal bout. The pipeline reads the behaviours to emit from —
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
  first-difference filter penalty. Its categorical model reads class indices
  ``0 … C-1`` with class 0 as the reference, so the observed ``qlvm_category`` labels
  (``1 … k``) are mapped onto contiguous indices in sorted order; the results'
  ``metadata`` records ``n_classes`` and ``class_labels`` (index ``i`` is label
  ``class_labels[i]``).

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
runs the other way: it predicts a **behavioural** variable and asks whether the partner's vocalizing changed it.

It answers two questions from one extraction:

1. **Does a male vocal bout, versus comparable silence, change her behaviour?**
2. **Does bout duration matter?**

Anchors are **events, not tiles**. Every row sits either at a male bout offset or
inside an inter-bout gap. A precursor design tiled anchors every 4 s across the
whole session and spent ~70% of its rows on windows containing no vocalizations at all,
diluting the very contrast it was meant to measure — measured on this cohort,
only **30.6%** of tiled anchors carried any vocalization.

The silent comparison is **inter-bout silence**, not globally quiet stretches. A
point inside a gap is a moment where he *could* have vocalized and did not, with
both animals demonstrably still interacting. Globally quiet periods are a
different behavioural regime, and adjusting for the difference would mean
extrapolating between two separated groups rather than comparing within one.

No control window — the pre-anchor history or the forward window of an inter-bout or
deep-silence anchor — and no bout-offset forward window may touch any vocalization of
the predictor mouse (or one assigned to no animal): every vocal class counts, a
singleton below ``usv_per_bout_floor`` counts, a squeak-bearing segment blocks the union
of the segment and its squeak extent, and noise segments are dropped first when
``vocal_features.exclude_noise_usvs`` is ``true``. The bout list alone (target classes,
floored) would let such vocalizations into the silent condition.

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

Which animal's covariates enter is chosen at **fit time** through
``behavioral_response.covariate_features``: ``'self'`` adjusts for the responder's
own pre-anchor kinematics, ``'partner'`` for the vocalizer's, ``'both'`` (the default)
for both. The dyadic columns (``nose-nose`` and the two directional angles) stay in
under every choice, since they belong to neither animal. The extraction always
stores all three groups, so one artifact serves every choice, and the contrast
artifact records the choice and the columns kept.

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
``behavioral_response_contrast``, in the same module, fits a GLM with session-clustered standard
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

    from usv_playpen.modeling.modeling_behavioral_response import (
        BehavioralResponsePipeline,
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

.. _modeling-matched-divergence:

Matched divergence
~~~~~~~~~~~~~~~~~~
``MatchedDivergencePipeline`` is the design-based counterpart of the contrast,
configured by the ``behavioral_response.matched_divergence`` block. Where the
contrast holds pre-anchor state fixed with the covariate block, this holds it fixed
by construction: every vocal anchor is paired with the silent anchors whose
pre-anchor level in that same feature matches (within ``match_caliper_sd`` pooled
standard deviations over the ``[-baseline_start_seconds, -baseline_end_seconds]``
window), and the difference between the two mean curves over
``[-pre_window_seconds, +window_seconds]`` is read directly. The two answer the same
question with almost disjoint assumptions — one leans on the functional form of an
adjustment, the other on the choice of what to match — so a result appearing in
both is far harder to explain away than one appearing in either.

Its anchors differ from the contrast's on purpose: cleanliness is judged against **every** vocalization (his singletons, the partner's vocalizations and unassigned ones included),
``post_clean_seconds`` ahead of vocal and silent anchors alike, and silent anchors
are drawn from a ``silence_grid_seconds`` grid at least ``pre_seconds`` after the last vocalization of anyone. Controls are matched, not modelled, and all eligible controls
are averaged into one curve per vocal anchor, so the unit of analysis is the vocal
anchor. Matching is per feature, on that feature's own baseline, so the panels are
different anchor sets and are not comparable with one another.

.. code-block:: python

    from usv_playpen.modeling.modeling_behavioral_response import MatchedDivergencePipeline

    MatchedDivergencePipeline(
        modeling_settings_dict=None
    ).extract_and_save_matched_divergence()

The artifact (``matched_divergence_<cohort>_<ts>.pkl`` in ``io.save_directory``)
holds, per feature, the stacked vocal and control curves (``pairs``), their
sessions and control counts, the anchors rejected for lacking an eligible control,
the pooled scaling, the ``parameters`` block it was built with and the camera frame
rate; ``plot_matched_divergence`` draws it.

.. _modeling-neural-settings:

Neural modeling settings
------------------------
The single-unit PAG analyses (``usv_playpen.neural_modeling``: the kinematic
encoding, the quiet-to-vocal transfer, the vocal gating, the vocal-occurrence and
vocalization-identity decodings, the nested vocal-manifold position decoding and
the cohort behaviour control) read one settings file,
``_parameter_settings/neural_modeling_settings.json``. The dispatchers
(``main_neural_encoding_dispatcher``, ``main_neural_vocal_dispatcher``,
``main_behaviour_control_dispatcher``) take ``--settings`` to point at another copy
(default: the shipped file). Every key is read strictly (a missing key raises
``KeyError``); keys that name a design alternative (``selection_method``,
``calibration_grain``, ``vm_score_mode`` and the like) accept only the value the
code implements and raise on any other, so a declared alternative can never run as
the default in disguise. All durations are in seconds, all frame counts in
tracking frames (150 fps), and all torus coordinates in units of the period.

**data** — which sessions and units the run draws on.

.. code-block:: json

    "data": {
        "session_list_files": ["/mnt/falkner/Bartul/modeling/input_files/ephys_courtship_intact_partners_sessions_list.txt", "/mnt/falkner/Bartul/modeling/input_files/ephys_courtship_mute_female_sessions_list.txt"],
        "behaviour_selection_result_path": "/mnt/falkner/Bartul/modeling/model_selection_results/behaviour_control_selection_ephys_newmap.pkl",
        "brain_areas": ["PAG"],
        "area_mode": "include",
        "pool_areas": false,
        "quality": ["good"],
        "somatic": "somatic"
    }

.. list-table:: ``data`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`session_list_files <nm-data-session-list-files>`
     - two lists, see JSON above
     - list of paths to text files
   * - :ref:`behaviour_selection_result_path <nm-data-behaviour-selection-result-path>`
     - path, see JSON above
     - path to a pickle file
   * - :ref:`brain_areas <nm-data-brain-areas>`
     - ``["PAG"]``
     - list of ``brain_area`` labels
   * - :ref:`area_mode <nm-data-brain-areas>`
     - ``'include'``
     - ``'include'``, ``'exclude'``, ``'all'``
   * - :ref:`pool_areas <nm-data-pool-areas>`
     - ``false``
     - not read
   * - :ref:`quality <nm-data-quality>`
     - ``["good"]``
     - list of ``cluster_group`` labels
   * - :ref:`somatic <nm-data-somatic>`
     - ``'somatic'``
     - ``'somatic'``, ``'non_somatic'``, ``'both'``

.. _nm-data-session-list-files:

``session_list_files``
   Text files listing one session directory per line; the union is the courtship cohort.

   :Options: a list of paths to text files
   :Default: the intact-partner and the mute-female ephys lists (see the JSON above)

.. _nm-data-behaviour-selection-result-path:

``behaviour_selection_result_path``
   The pickle the behaviour-control feature selection (``main_behaviour_control_dispatcher --stage select``) writes.

   :Options: a path to a pickle file
   :Default: ``behaviour_control_selection_ephys_newmap.pkl`` under the modeling results directory (see the JSON above)
   :Notes: the nested decoding reads it back to check its frozen ``reduced_model_features`` against.

.. _nm-data-brain-areas:

``brain_areas`` / ``area_mode``
   The unit catalog's ``brain_area`` filter.

   :Options:
      * ``brain_areas`` — a list of ``brain_area`` labels
      * ``area_mode`` — ``'include'`` keeps the listed areas, ``'exclude'`` drops them, ``'all'`` ignores the list
   :Default: ``["PAG"]`` and ``'include'``

.. _nm-data-pool-areas:

``pool_areas``
   Declared but read by no current code (areas are not pooled or split anywhere).

   :Options: none (the key is not read)
   :Default: ``false``

.. _nm-data-quality:

``quality``
   The allowed ``cluster_group`` labels of the unit catalog.

   :Options: a list of ``cluster_group`` labels
   :Default: ``["good"]``

.. _nm-data-somatic:

``somatic``
   Which units pass the catalog's somatic waveform classification.

   :Options:
      * ``'somatic'`` — somatic units only
      * ``'non_somatic'`` — non-somatic units only
      * ``'both'`` — both
   :Default: ``'somatic'``

**data_roots** — where the run reads its inputs.

.. code-block:: json

    "data_roots": {
        "ephys_root": "/mnt/falkner/Bartul/EPHYS",
        "catalog_path": "/mnt/falkner/Bartul/EPHYS/unit_catalog.csv",
        "data_root": "/mnt/falkner/Bartul/Data"
    }

.. list-table:: ``data_roots`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`ephys_root <nm-data-roots-ephys-root>`
     - path, see JSON above
     - path to a directory (not read here)
   * - :ref:`catalog_path <nm-data-roots-catalog-path>`
     - path, see JSON above
     - path to a CSV file
   * - :ref:`data_root <nm-data-roots-data-root>`
     - path, see JSON above
     - path to a directory

.. _nm-data-roots-ephys-root:

``ephys_root``
   The EPHYS tree.

   :Options: a path to a directory
   :Default: the lab EPHYS tree (see the JSON above)
   :Notes: declared for completeness, the neural modeling code itself reads none of it (the anatomy and catalog tools take it from the analyses settings).

.. _nm-data-roots-catalog-path:

``catalog_path``
   The unit catalog (``unit_catalog.csv``) the cohort is selected from and the recorded animal of each session is resolved from.

   :Options: a path to a CSV file
   :Default: ``unit_catalog.csv`` in the EPHYS tree (see the JSON above)

.. _nm-data-roots-data-root:

``data_root``
   The session data tree (tracking, USV summaries, spikes).

   :Options: a path to a directory
   :Default: the lab session data tree (see the JSON above)
   :Notes: the encoding and vocal dispatchers take it on the command line instead (``--data-root``, required); the behaviour control reads it from here.

**data_sufficiency** — how much data a unit must carry to be analysed. A unit failing the vocal floors is kept and flagged, not dropped, so the denominators the fractions are reported against do not shrink silently.

.. code-block:: json

    "data_sufficiency": {
        "min_courtship_sessions": 2,
        "min_emitter_usvs_per_session": 20,
        "min_vocal_sessions": 2,
        "min_decoding_spikes": 50,
        "single_session_inner_split_blocks": 5
    }

.. list-table:: ``data_sufficiency`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`min_courtship_sessions <nm-data-sufficiency-min-courtship-sessions>`
     - ``2``
     - integer >= 2
   * - :ref:`min_emitter_usvs_per_session <nm-data-sufficiency-min-emitter-usvs-per-session>`
     - ``20``
     - integer >= 0
   * - :ref:`min_vocal_sessions <nm-data-sufficiency-min-vocal-sessions>`
     - ``2``
     - integer >= 2
   * - :ref:`min_decoding_spikes <nm-data-sufficiency-min-decoding-spikes>`
     - ``50``
     - integer >= 0
   * - :ref:`single_session_inner_split_blocks <nm-data-sufficiency-single-session-inner-split-blocks>`
     - ``5``
     - integer > 0

.. _nm-data-sufficiency-min-courtship-sessions:

``min_courtship_sessions``
   Courtship sessions a unit must appear in.

   :Options: an integer; at least ``2``, since the leave-one-session-out rotation needs at least two
   :Default: ``2``

.. _nm-data-sufficiency-min-emitter-usvs-per-session:

``min_emitter_usvs_per_session``
   Focal USVs a session needs before any vocalization-scored analysis (or the behaviour-control cohort) may use it.

   :Options: a non-negative integer
   :Default: ``20``

.. _nm-data-sufficiency-min-vocal-sessions:

``min_vocal_sessions``
   Vocal-eligible sessions a unit needs for the vocal-side leave-one-session-out rotation.

   :Options: an integer; at least ``2`` for the rotation to run
   :Default: ``2``

.. _nm-data-sufficiency-min-decoding-spikes:

``min_decoding_spikes``
   Total spikes across the scored events below which the category decoding reports the unit as not decodable.

   :Options: a non-negative integer
   :Default: ``50``

.. _nm-data-sufficiency-single-session-inner-split-blocks:

``single_session_inner_split_blocks``
   When a held-out fold leaves a single training session, the inner split falls back to this many contiguous blocks within that session.

   :Options: a positive integer
   :Default: ``5``

**kinematic_features** — the behavioural feature zoo, with exactly the meaning of the modeling block of the same name (see the ``kinematic_features`` bullets of *Modeling settings* above); the neural design assembly reuses the modeling column selection and pooled z-scoring, so the features match the behavioural models exactly.

.. code-block:: json

    "kinematic_features": {
        "egocentric": ["speed", "neck_elevation", "allo_roll", "allo_pitch", "ego_yaw", "back_pitch", "back_yaw", "tail_curvature"],
        "dyadic_pose": ["head-head", "allo_yaw-head", "head-allo_yaw", "allo_pitch-head", "head-allo_pitch"],
        "dyadic_engagement": [],
        "dyadic_pose_symmetric": false,
        "include_1st_derivatives": false,
        "include_2nd_derivatives": false,
        "abs_features": ["allo_roll", "allo_yaw-nose", "nose-allo_yaw", "allo_yaw-head", "head-allo_yaw", "allo_yaw-TTI", "TTI-allo_yaw"],
        "smooth_abs_features": {
            "ego_yaw": 1.0,
            "back_yaw": 0.5
        }
    }

.. list-table:: ``kinematic_features`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`egocentric <nm-kinematic-features-zoo>`
     - eight features, see JSON above
     - as in the modeling block
   * - :ref:`dyadic_pose <nm-kinematic-features-zoo>`
     - five features, see JSON above
     - as in the modeling block
   * - :ref:`dyadic_engagement <nm-kinematic-features-zoo>`
     - ``[]``
     - as in the modeling block
   * - :ref:`dyadic_pose_symmetric <nm-kinematic-features-zoo>`
     - ``false``
     - as in the modeling block
   * - :ref:`include_1st_derivatives <nm-kinematic-features-zoo>`
     - ``false``
     - as in the modeling block
   * - :ref:`include_2nd_derivatives <nm-kinematic-features-zoo>`
     - ``false``
     - as in the modeling block
   * - :ref:`abs_features <nm-kinematic-features-zoo>`
     - seven features, see JSON above
     - as in the modeling block
   * - :ref:`smooth_abs_features <nm-kinematic-features-zoo>`
     - ``{"ego_yaw": 1.0, "back_yaw": 0.5}``
     - feature → smoothing ``ε``

.. _nm-kinematic-features-zoo:

``egocentric`` / ``dyadic_pose`` / ``dyadic_engagement`` / ``dyadic_pose_symmetric`` / ``include_1st_derivatives`` / ``include_2nd_derivatives`` / ``abs_features`` / ``smooth_abs_features``
   As in the modeling ``kinematic_features`` block (see *Modeling settings* above).

   :Options: as in the modeling block; ``smooth_abs_features`` maps a feature to its smoothing ``ε`` in the feature's own units
   :Default: see the JSON above

**vocalization_settings** — which vocalizations count, whose, and the window spikes are counted in.

.. code-block:: json

    "vocalization_settings": {
        "vocal_emitter": "self",
        "clean_against": "all",
        "exclude_noise": true,
        "exclude_squeaks": true,
        "squeaks_block_quiet": true,
        "spike_window": {
            "pre_offset_seconds": 0.05,
            "width_seconds": 0.1
        },
        "torus_period": 1.0
    }

.. list-table:: ``vocalization_settings`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`vocal_emitter <nm-vocalization-settings-vocal-emitter>`
     - ``'self'``
     - ``'self'``, ``'partner'``, ``'all'``
   * - :ref:`clean_against <nm-vocalization-settings-clean-against>`
     - ``'all'``
     - ``'all'``, ``'self'``, ``'partner'``
   * - :ref:`exclude_noise <nm-vocalization-settings-exclude-noise>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`exclude_squeaks <nm-vocalization-settings-exclude-squeaks>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`squeaks_block_quiet <nm-vocalization-settings-squeaks-block-quiet>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`spike_window.pre_offset_seconds <nm-vocalization-settings-spike-window>`
     - ``0.05``
     - seconds >= 0
   * - :ref:`spike_window.width_seconds <nm-vocalization-settings-spike-window>`
     - ``0.1``
     - seconds > 0
   * - :ref:`torus_period <nm-vocalization-settings-torus-period>`
     - ``1.0``
     - number > 0

.. _nm-vocalization-settings-vocal-emitter:

``vocal_emitter``
   Whose USVs are the analysed vocalizations.

   :Options:
      * ``'self'`` — the recorded animal's own
      * ``'partner'`` — the partner's
      * ``'all'`` — no emitter filter
   :Default: ``'self'``
   :Notes: one key drives every analysis -- the vocal decoding's events, the kinematic encoding's transfer frames, the gating's vocal frames and the behaviour control's training rows -- so moving to the partner moves them together.

.. _nm-vocalization-settings-clean-against:

``clean_against``
   Whose vocalizations make a frame unclean (non-quiet).

   :Options:
      * ``'all'`` — a quiet frame is one in which the animal was not even hearing a vocalization
      * ``'self'`` — only the recorded animal's own vocalizations
      * ``'partner'`` — only the partner's vocalizations
   :Default: ``'all'``

.. _nm-vocalization-settings-exclude-noise:

``exclude_noise``
   Drop the segments ``detect-usv-noise`` flagged as noise everywhere, including from the quiet guard (a microphone artefact must not veto a silent frame).

   :Options: ``true`` or ``false``
   :Default: ``true``

.. _nm-vocalization-settings-exclude-squeaks:

``exclude_squeaks``
   Drop squeak-bearing segments from the analysed vocalization set (a squeak is out of distribution for the USV map).

   :Options: ``true`` or ``false``
   :Default: ``true``

.. _nm-vocalization-settings-squeaks-block-quiet:

``squeaks_block_quiet``
   Whether a squeak still makes the surrounding frames non-quiet even when excluded from the analysed set.

   :Options: ``true`` or ``false``
   :Default: ``true`` (quiet means the animal made no sound of any kind)
   :Notes: measured exposure (2026-09-30): 3.5% noise and 7.3% squeaks across the behavioural courtship sessions, 0.8–1.5% of focal USVs on the ephys days.

.. _nm-vocalization-settings-spike-window:

``spike_window``
   The per-event spike count window: it opens ``pre_offset_seconds`` before the USV onset and spans ``width_seconds``, so it covers the pre-motor lead and the start of the USV.

   :Options:
      * ``pre_offset_seconds`` — seconds before the USV onset at which the window opens
      * ``width_seconds`` — the window's length in seconds
   :Default: ``0.05`` and ``0.1``

.. _nm-vocalization-settings-torus-period:

``torus_period``
   The QLVM manifold's wrap period, shared by every analysis.

   :Options: a positive number (in units of the manifold's coordinates)
   :Default: ``1.0``
   :Notes: a property of the manifold, so it lives here once rather than in each block. The vocalization-identity decoder and the nested decoding both read it from here.

**kinematic_encoding** — the quiet-period kinematic encoding model (per-frame Bernoulli spike GLM of the behavioural history) and its transfer into vocal periods.

.. code-block:: json

    "kinematic_encoding": {
        "history_pre_seconds": 4.0,
        "clean_post_seconds": 2.0,
        "bin_resize_factor": 1,
        "smoothness_order": 1,
        "include_spike_history": false,
        "spike_history_window_seconds": 0.1,
        "filter_band": {
            "compute": true,
            "n_resamples": 10
        },
        "subsample_negatives": true,
        "negatives_per_positive": 10,
        "calibration": {
            "scheme": "crossfit",
            "n_blocks": 5
        },
        "session_intercepts": {
            "compute": true,
            "indicator_scale": 10.0,
            "max_sweeps": 30,
            "tolerance": 0.0001,
            "newton_max_steps": 25,
            "newton_step_tolerance": 1e-10
        },
        "significance_model": {
            "lambda_group": 0.0,
            "lambda_smooth": 1000.0,
            "lambda_ridge": 0.01
        },
        "feature_selection": {
            "screen_alpha": 0.01,
            "selection_method": "forward_d2",
            "acceptance_rule": "paired_1se",
            "screen_require_positive_slope": true
        },
        "solver": {
            "max_iter": 5000,
            "tol": 1e-6,
            "calibration_steps": 30,
            "null_calibration_bins": 2000
        },
        "transfer": {
            "calibration_grain": "pooled_across_folds",
            "d2_reference_rate": "scored_frames",
            "universe": "bout_window",
            "pre_bout_seconds": 4.0,
            "inter_bout_threshold_seconds": 0.1155,
            "grain": "count",
            "bin_seconds": 0.1,
            "null_block_megabytes": 64
        },
        "subsampling_correction": "log_prior_offset",
        "temporal_basis": {
            "type": "bspline",
            "n_splines": 20,
            "spline_order": 3,
            "l2_lag_alpha": 100.0,
            "lambda_ridge": 0.1,
            "lambda_smooth": 0.0,
            "lambda_group": 0.0,
            "smoothness_order": 2,
            "spike_end_correction": "drop_outside_and_rescale"
        },
        "max_train_total": 70000,
        "chunk_rows": 4000,
        "null_block_megabytes": 32
    }

.. list-table:: ``kinematic_encoding`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`history_pre_seconds <nm-kinematic-encoding-history-pre-seconds>`
     - ``4.0``
     - seconds > 0
   * - :ref:`clean_post_seconds <nm-kinematic-encoding-clean-post-seconds>`
     - ``2.0``
     - seconds >= 0
   * - :ref:`bin_resize_factor <nm-kinematic-encoding-unread-keys>`
     - ``1``
     - not read
   * - :ref:`smoothness_order <nm-kinematic-encoding-smoothness-order>`
     - ``1``
     - integer >= 1
   * - :ref:`include_spike_history <nm-kinematic-encoding-unread-keys>`
     - ``false``
     - not read
   * - :ref:`spike_history_window_seconds <nm-kinematic-encoding-unread-keys>`
     - ``0.1``
     - not read
   * - :ref:`filter_band.compute <nm-kinematic-encoding-filter-band>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`filter_band.n_resamples <nm-kinematic-encoding-filter-band>`
     - ``10``
     - integer > 0
   * - :ref:`subsample_negatives <nm-kinematic-encoding-unread-keys>`
     - ``true``
     - not read
   * - :ref:`negatives_per_positive <nm-kinematic-encoding-negatives-per-positive>`
     - ``10``
     - integer > 0
   * - :ref:`calibration.scheme <nm-kinematic-encoding-calibration>`
     - ``'crossfit'``
     - ``'crossfit'``, ``'in_sample'``
   * - :ref:`calibration.n_blocks <nm-kinematic-encoding-calibration>`
     - ``5``
     - integer > 1
   * - :ref:`session_intercepts.compute <nm-kinematic-encoding-session-intercepts>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`session_intercepts.indicator_scale <nm-kinematic-encoding-session-intercepts>`
     - ``10.0``
     - number > 0
   * - :ref:`session_intercepts.max_sweeps <nm-kinematic-encoding-session-intercepts>`
     - ``30``
     - integer > 0
   * - :ref:`session_intercepts.tolerance <nm-kinematic-encoding-session-intercepts>`
     - ``0.0001``
     - number > 0
   * - :ref:`session_intercepts.newton_max_steps <nm-kinematic-encoding-session-intercepts>`
     - ``25``
     - integer > 0
   * - :ref:`session_intercepts.newton_step_tolerance <nm-kinematic-encoding-session-intercepts>`
     - ``1e-10``
     - number > 0
   * - :ref:`significance_model.lambda_group <nm-kinematic-encoding-significance-model>`
     - ``0.0``
     - number >= 0
   * - :ref:`significance_model.lambda_smooth <nm-kinematic-encoding-significance-model>`
     - ``1000.0``
     - number >= 0
   * - :ref:`significance_model.lambda_ridge <nm-kinematic-encoding-significance-model>`
     - ``0.01``
     - number >= 0
   * - :ref:`feature_selection.screen_alpha <nm-kinematic-encoding-feature-selection>`
     - ``0.01``
     - number in ``(0, 1)``
   * - :ref:`feature_selection.selection_method <nm-kinematic-encoding-feature-selection>`
     - ``'forward_d2'``
     - ``'forward_d2'`` only
   * - :ref:`feature_selection.acceptance_rule <nm-kinematic-encoding-feature-selection>`
     - ``'paired_1se'``
     - ``'paired_1se'`` only
   * - :ref:`feature_selection.screen_require_positive_slope <nm-kinematic-encoding-feature-selection>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`solver.max_iter <nm-kinematic-encoding-solver>`
     - ``5000``
     - integer > 0
   * - :ref:`solver.tol <nm-kinematic-encoding-solver>`
     - ``1e-6``
     - number > 0
   * - :ref:`solver.calibration_steps <nm-kinematic-encoding-solver>`
     - ``30``
     - integer > 0
   * - :ref:`solver.null_calibration_bins <nm-kinematic-encoding-solver>`
     - ``2000``
     - integer >= 0 (``0`` = exact)
   * - :ref:`transfer.calibration_grain <nm-kinematic-encoding-transfer>`
     - ``'pooled_across_folds'``
     - ``'pooled_across_folds'`` only
   * - :ref:`transfer.d2_reference_rate <nm-kinematic-encoding-transfer>`
     - ``'scored_frames'``
     - ``'scored_frames'`` only
   * - :ref:`transfer.universe <nm-kinematic-encoding-transfer>`
     - ``'bout_window'``
     - ``'bout_window'``, ``'usv_span'``
   * - :ref:`transfer.pre_bout_seconds <nm-kinematic-encoding-transfer>`
     - ``4.0``
     - seconds >= 0
   * - :ref:`transfer.inter_bout_threshold_seconds <nm-kinematic-encoding-transfer>`
     - ``0.1155``
     - seconds > 0
   * - :ref:`transfer.grain <nm-kinematic-encoding-transfer>`
     - ``'count'``
     - ``'count'``, ``'frame'``
   * - :ref:`transfer.bin_seconds <nm-kinematic-encoding-transfer>`
     - ``0.1``
     - seconds > 0
   * - :ref:`transfer.null_block_megabytes <nm-kinematic-encoding-transfer>`
     - ``64``
     - MB > 0
   * - :ref:`subsampling_correction <nm-kinematic-encoding-subsampling-correction>`
     - ``'log_prior_offset'``
     - ``'log_prior_offset'`` only
   * - :ref:`temporal_basis.type <nm-kinematic-encoding-temporal-basis>`
     - ``'bspline'``
     - ``'bspline'``, ``'none'``
   * - :ref:`temporal_basis.n_splines <nm-kinematic-encoding-temporal-basis>`
     - ``20``
     - integer > 0
   * - :ref:`temporal_basis.spline_order <nm-kinematic-encoding-temporal-basis>`
     - ``3``
     - integer > 0
   * - :ref:`temporal_basis.l2_lag_alpha <nm-kinematic-encoding-temporal-basis>`
     - ``100.0``
     - number >= 0
   * - :ref:`temporal_basis.lambda_ridge <nm-kinematic-encoding-temporal-basis>`
     - ``0.1``
     - number >= 0
   * - :ref:`temporal_basis.lambda_smooth <nm-kinematic-encoding-temporal-basis>`
     - ``0.0``
     - number >= 0
   * - :ref:`temporal_basis.lambda_group <nm-kinematic-encoding-temporal-basis>`
     - ``0.0``
     - number >= 0
   * - :ref:`temporal_basis.smoothness_order <nm-kinematic-encoding-temporal-basis>`
     - ``2``
     - integer >= 1
   * - :ref:`temporal_basis.spike_end_correction <nm-kinematic-encoding-temporal-basis>`
     - ``'drop_outside_and_rescale'``
     - ``'drop_outside_and_rescale'``, ``'none'``
   * - :ref:`max_train_total <nm-kinematic-encoding-max-train-total>`
     - ``70000``
     - integer > 0
   * - :ref:`chunk_rows <nm-kinematic-encoding-chunk-rows>`
     - ``4000``
     - integer > 0
   * - :ref:`null_block_megabytes <nm-kinematic-encoding-null-block-megabytes>`
     - ``32``
     - MB > 0

.. _nm-kinematic-encoding-history-pre-seconds:

``history_pre_seconds``
   The kinematic history length (``4.0`` → 600 lags at 150 fps).

   :Options: a positive number of seconds
   :Default: ``4.0``
   :Notes: it also defines a quiet frame (no vocalization within this long before it) and the minimum gap of the single-session inner split, deliberately coupled so a quiet anchor's whole history is vocalization-free.

.. _nm-kinematic-encoding-clean-post-seconds:

``clean_post_seconds``
   A quiet frame must also have no vocalization within this long after it.

   :Options: a non-negative number of seconds
   :Default: ``2.0``

.. _nm-kinematic-encoding-unread-keys:

``bin_resize_factor`` / ``include_spike_history`` / ``spike_history_window_seconds`` / ``subsample_negatives``
   Declared but read by no current code.

   :Options: none (the keys are not read)
   :Default: ``1``, ``false``, ``0.1`` and ``true``
   :Notes: the model runs at the tracking frame rate, without spike history, and always subsamples the negatives by ``negatives_per_positive``.

.. _nm-kinematic-encoding-smoothness-order:

``smoothness_order``
   Finite-difference order of the temporal-smoothness penalty on the raw per-frame lag path.

   :Options: a positive integer finite-difference order (``1``: first differences, the GLM-HMM tradition)
   :Default: ``1``
   :Notes: used only when ``temporal_basis.type`` is ``'none'``.

.. _nm-kinematic-encoding-filter-band:

``filter_band``
   The per-lag spread band of the representative filters.

   :Options:
      * ``compute`` — ``true`` or ``false``: toggles the band
      * ``n_resamples`` — the number of refits the band is built from, a positive integer
   :Default: ``true`` and ``10``

.. _nm-kinematic-encoding-negatives-per-positive:

``negatives_per_positive``
   Non-spike frames kept per spike frame in each training session.

   :Options: a positive integer
   :Default: ``10``
   :Notes: the subsampling is undone exactly by the ``subsampling_correction`` offset.

.. _nm-kinematic-encoding-calibration:

``calibration``
   How a frozen model is turned into a score.

   :Options:
      * ``scheme`` — ``'crossfit'`` calibrates on blocks that exclude the scored block, so no score is calibrated on its own frames; ``'in_sample'`` calibrates on the scored frames themselves
      * ``n_blocks`` — the number of contiguous blocks ``'crossfit'`` uses
   :Default: ``'crossfit'`` with ``5`` blocks
   :Notes: ``'in_sample'`` is accepted by the code (``CALIBRATION_SCHEMES`` in ``deviance_metrics``) but was not described here before; any other ``scheme`` raises ``ValueError``.

.. _nm-kinematic-encoding-session-intercepts:

``session_intercepts``
   Per-session baselines, so a session that fires faster cannot be explained by whichever feature also differs in it.

   :Options:
      * ``compute`` — ``true`` or ``false``: toggles them
      * ``indicator_scale`` — on the spline path the session indicators enter as one column group scaled by this factor (``10.0``, so the uniform ridge reaches them at 1/100 of its strength — effectively unpenalised)
      * ``max_sweeps`` — on the raw per-frame path the baselines are backfitted: at most this many alternations of a filter refit and a per-session Newton update of the baselines
      * ``tolerance`` — the backfitting stops when the baselines move less than this
      * ``newton_max_steps`` — each per-session Newton solve takes at most this many steps
      * ``newton_step_tolerance`` — each Newton solve stops once a step is below this
   :Default: ``true``, ``10.0``, ``30``, ``1e-4`` (``0.0001``), ``25`` and ``1e-10``
   :Notes: the per-session Newton update is a one-dimensional convex solve, so ``newton_max_steps`` and ``newton_step_tolerance`` only bound it.

.. _nm-kinematic-encoding-significance-model:

``significance_model``
   The penalties of the raw per-frame-lag fit.

   :Options:
      * ``lambda_group`` — group lasso (``0.0`` = no sparsity)
      * ``lambda_smooth`` — the temporal-smoothness penalty
      * ``lambda_ridge`` — the ridge penalty
   :Default: ``0.0``, ``1000.0`` and ``0.01``

.. _nm-kinematic-encoding-feature-selection:

``feature_selection``
   Features are screened one at a time against the circular-shift null, then added by forward selection.

   :Options:
      * ``screen_alpha`` — the screen's family-wise level; the Bonferroni bar is ``screen_alpha / n_features``
      * ``screen_require_positive_slope`` — ``true`` or ``false``: whether the screen also requires a positive calibration slope
      * ``selection_method`` — ``'forward_d2'`` (forward selection on held-out explained deviance)
      * ``acceptance_rule`` — ``'paired_1se'`` (a step is accepted when its paired gain exceeds one paired standard error)
   :Default: ``0.01``, ``true``, ``'forward_d2'`` and ``'paired_1se'``
   :Notes: both names (``selection_method``, ``acceptance_rule``) accept only the implemented value.

.. _nm-kinematic-encoding-solver:

``solver``
   Solver iteration limits and the calibration of the null draws.

   :Options:
      * ``max_iter`` — maximum iterations of the FISTA group-elastic-net solver
      * ``tol`` — its convergence tolerance
      * ``calibration_steps`` — Newton iterations of each two-parameter calibration refit
      * ``null_calibration_bins`` — equal-count predictor bins for the binned calibration of the null draws; ``0`` runs the exact path (the responses are exact either way)
   :Default: ``5000``, ``1e-6``, ``30`` and ``2000``

.. _nm-kinematic-encoding-transfer:

``transfer``
   The frozen quiet model scored on vocal periods.

   :Options:
      * ``calibration_grain`` — ``'pooled_across_folds'`` concatenates the held-out predictions and calibrates once (averaging per-fold scores let a small fold dominate)
      * ``d2_reference_rate`` — ``'scored_frames'`` scores explained deviance against the vocal frames' own rate
      * ``universe`` — ``'bout_window'`` scores each bout from ``pre_bout_seconds`` before its first USV to its end; ``'usv_span'`` scores only the frames inside the USVs themselves
      * ``pre_bout_seconds`` — the lead before a bout's first USV under ``'bout_window'``
      * ``inter_bout_threshold_seconds`` — bouts are split where the inter-USV gap reaches this
      * ``grain`` — ``'count'`` scores spike counts in ``bin_seconds`` bins rather than per frame; ``'frame'`` scores per frame
      * ``bin_seconds`` — the count bin width
      * ``null_block_megabytes`` — the memory budget for one block of pooled null draws under the ``'frame'`` grain
   :Default: ``'pooled_across_folds'``, ``'scored_frames'``, ``'bout_window'``, ``4.0``, ``0.1155``, ``'count'``, ``0.1`` and ``64``
   :Notes: ``null_block_megabytes`` is a memory bound only (the null scores do not depend on it). ``calibration_grain`` and ``d2_reference_rate`` accept only the implemented value. ``'usv_span'`` is accepted by the code (``TRANSFER_UNIVERSES`` in ``quiet_to_vocal_transfer``) but was not described here before.

.. _nm-kinematic-encoding-subsampling-correction:

``subsampling_correction``
   How the negative subsampling is corrected.

   :Options: ``'log_prior_offset'`` only (the only implemented value): a ``log`` sampling-rate offset in the linear predictor
   :Default: ``'log_prior_offset'``

.. _nm-kinematic-encoding-temporal-basis:

``temporal_basis``
   The lag basis of the shipped fit.

   :Options:
      * ``type`` — ``'bspline'`` (B-splines over the 600 lags) or ``'none'`` (keeps the raw per-frame lags)
      * ``n_splines`` — the number of B-splines
      * ``spline_order`` — the spline order (``3``: cubic)
      * ``l2_lag_alpha`` — makes the ridge grow with a spline's distance from the spike, each column's penalty scaled by ``1 + l2_lag_alpha · f²`` with ``f`` its centre's lag as a fraction of the window, so the far end of the history is shrunk hardest
      * ``lambda_ridge`` — the coefficient ridge
      * ``lambda_smooth`` — the smoothness penalty across neighbouring splines
      * ``lambda_group`` — the group-lasso strength of the basis-path fit
      * ``smoothness_order`` — the finite-difference order of that spline-coefficient smoothness penalty (inert while ``lambda_smooth`` is ``0``)
      * ``spike_end_correction`` — ``'drop_outside_and_rescale'`` repairs what a uniform coefficient ridge does at the spike end of the basis; ``'none'`` leaves it
   :Default: ``'bspline'`` with ``n_splines`` ``20``, cubic (``spline_order`` ``3``), ``l2_lag_alpha`` ``100.0``, ``lambda_ridge`` ``0.1``, ``lambda_smooth`` ``0.0``, ``lambda_group`` ``0.0`` (off: the basis path does no feature sparsity), ``smoothness_order`` ``2`` and ``spike_end_correction`` ``'drop_outside_and_rescale'``

.. _nm-kinematic-encoding-max-train-total:

``max_train_total``
   Cap on the training rows of one fit across sessions, split evenly per session, which bounds memory and time.

   :Options: a positive integer
   :Default: ``70000``

.. _nm-kinematic-encoding-chunk-rows:

``chunk_rows``
   Rows per block when the lagged design is scored in chunks.

   :Options: a positive integer
   :Default: ``4000``
   :Notes: a memory bound only.

.. _nm-kinematic-encoding-null-block-megabytes:

``null_block_megabytes``
   Memory budget, in MB, for the label matrix of one block of circular-shift null draws.

   :Options: a positive number; fractions of a MB are fine
   :Default: ``32``
   :Notes: the batched calibration holds a few such arrays at once. A memory bound only: the null scores do not depend on it.

**vocal_gating** — is a unit's vocal response gated on a behavioural feature (a feature × vocalization interaction in a contemporaneous logistic model, quiet frames guard-banded and the peri-vocal zone dropped)?

.. code-block:: json

    "vocal_gating": {
        "vocal_window": "during_vocalization",
        "require_vocal_tuning": true,
        "vocal_tuning_alpha": 0.01,
        "ridge_frac": 0.02,
        "max_quiet_frames_per_session": 12000,
        "quiet_zeros_per_spike": 40,
        "min_vocal_spikes": null,
        "min_feature_iqr_vocal": null,
        "require_interaction_gt_vocal": true,
        "interaction_gt_vocal_alpha": 0.01,
        "feature_main_alpha": 0.01,
        "content_gating_compute": false,
        "gating_screen_n_shuffles": 2000,
        "screen_alpha": 0.01,
        "sign_ridge_frac": 0.1,
        "borderline_sigma_flag": 5.0,
        "solver": {
            "irls_n_steps": 30,
            "calibration_steps": 40
        }
    }

.. list-table:: ``vocal_gating`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`vocal_window <nm-vocal-gating-vocal-window>`
     - ``'during_vocalization'``
     - ``'during_vocalization'`` (only value)
   * - :ref:`require_vocal_tuning <nm-vocal-gating-vocal-tuning>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`vocal_tuning_alpha <nm-vocal-gating-vocal-tuning>`
     - ``0.01``
     - significance level
   * - :ref:`ridge_frac <nm-vocal-gating-ridge-frac>`
     - ``0.02``
     - fraction of the training rows
   * - :ref:`max_quiet_frames_per_session <nm-vocal-gating-quiet-frames>`
     - ``12000``
     - frames per session
   * - :ref:`quiet_zeros_per_spike <nm-vocal-gating-quiet-frames>`
     - ``40``
     - non-spike quiet frames per spike
   * - :ref:`min_vocal_spikes <nm-vocal-gating-identifiability-floors>`
     - ``null``
     - ``null`` (off), or a spike count
   * - :ref:`min_feature_iqr_vocal <nm-vocal-gating-identifiability-floors>`
     - ``null``
     - ``null`` (off), or an interquartile range
   * - :ref:`require_interaction_gt_vocal <nm-vocal-gating-interaction-gt-vocal>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`interaction_gt_vocal_alpha <nm-vocal-gating-interaction-gt-vocal>`
     - ``0.01``
     - significance level
   * - :ref:`feature_main_alpha <nm-vocal-gating-feature-main-alpha>`
     - ``0.01``
     - significance level
   * - :ref:`content_gating_compute <nm-vocal-gating-content-gating-compute>`
     - ``false``
     - ``false`` (only value)
   * - :ref:`gating_screen_n_shuffles <nm-vocal-gating-screen>`
     - ``2000``
     - number of circular-shift draws
   * - :ref:`screen_alpha <nm-vocal-gating-screen>`
     - ``0.01``
     - significance level (Bonferroni-divided)
   * - :ref:`sign_ridge_frac <nm-vocal-gating-sign-ridge-frac>`
     - ``0.1``
     - fraction of the rows
   * - :ref:`borderline_sigma_flag <nm-vocal-gating-borderline-sigma-flag>`
     - ``5.0``
     - null standard deviations
   * - :ref:`solver.irls_n_steps <nm-vocal-gating-solver>`
     - ``30``
     - Newton/IRLS iterations
   * - :ref:`solver.calibration_steps <nm-vocal-gating-solver>`
     - ``40``
     - calibration-refit iterations

.. _nm-vocal-gating-vocal-window:

``vocal_window``
   Which frames count as vocal frames in the gating model.

   :Options: ``'during_vocalization'`` — the vocal frames are every frame inside a focal USV (the only implemented value)
   :Default: ``'during_vocalization'``

.. _nm-vocal-gating-vocal-tuning:

``require_vocal_tuning`` / ``vocal_tuning_alpha``
   Restrict the gating analysis to vocally tuned units.

   :Options:
      * ``require_vocal_tuning`` — ``true``: only units whose vocal-occurrence or vocalization-identity test passes at ``vocal_tuning_alpha`` are gated; ``false``: no such restriction
      * ``vocal_tuning_alpha`` — the significance level of that vocal-tuning requirement
   :Default: ``true`` and ``0.01``

.. _nm-vocal-gating-ridge-frac:

``ridge_frac``
   The ridge of the gating fits, which scales with the training rows (``ridge_frac × n_train``).

   :Options: a fraction multiplying the number of training rows
   :Default: ``0.02``
   :Notes: a fixed small ridge is ~1000× too weak at ~50k rows.

.. _nm-vocal-gating-quiet-frames:

``max_quiet_frames_per_session`` / ``quiet_zeros_per_spike``
   The quiet frames entering the fit.

   :Options:
      * ``max_quiet_frames_per_session`` — at most this many quiet frames per session
      * ``quiet_zeros_per_spike`` — non-spike quiet frames are subsampled to this many per spike
   :Default: ``12000`` and ``40``
   :Notes: the quiet-only rows of the feature main effect (every quiet spike frame plus ``quiet_zeros_per_spike`` zeros each) are redrawn inside every circular-shift draw from that draw's shifted spike labels, by the same rule as the observed rows, so each draw has the observed class balance.

.. _nm-vocal-gating-identifiability-floors:

``min_vocal_spikes`` / ``min_feature_iqr_vocal``
   Optional identifiability floors.

   :Options:
      * ``min_vocal_spikes`` — a floor on the spikes inside vocal frames; ``null`` disables it
      * ``min_feature_iqr_vocal`` — a floor on the feature's interquartile range during vocalization; ``null`` disables it
   :Default: ``null`` and ``null`` (both disabled)

.. _nm-vocal-gating-interaction-gt-vocal:

``require_interaction_gt_vocal`` / ``interaction_gt_vocal_alpha``
   A gate must also exceed the plain vocal effect.

   :Options:
      * ``require_interaction_gt_vocal`` — ``true``: require the interaction to exceed the plain vocal effect; ``false``: do not
      * ``interaction_gt_vocal_alpha`` — the significance level of that test
   :Default: ``true`` and ``0.01``

.. _nm-vocal-gating-feature-main-alpha:

``feature_main_alpha``
   The level at which the feature's quiet-only main effect is reported as present.

   :Options: a significance level
   :Default: ``0.01``
   :Notes: it labels ``GATE+silentME`` and gates nothing.

.. _nm-vocal-gating-content-gating-compute:

``content_gating_compute``
   The feature × torus-position branch.

   :Options: ``false`` — the branch is deferred, so only ``false`` is implemented
   :Default: ``false``

.. _nm-vocal-gating-screen:

``gating_screen_n_shuffles`` / ``screen_alpha``
   The gating screen's circular-shift draws and its Bonferroni level (``screen_alpha / n_features``).

   :Options:
      * ``gating_screen_n_shuffles`` — the number of circular-shift draws; a draw count whose ``1 / (n + 1)`` floor cannot clear the Bonferroni bar raises
      * ``screen_alpha`` — the family-wise level, divided by the number of features
   :Default: ``2000`` and ``0.01``
   :Notes: each tested unit carries one p for the cohort correction: the smallest per-feature conjunction p (``max(p_interaction, p_interaction_gt_vocal)``, or ``p_interaction`` alone when ``require_interaction_gt_vocal`` is false) times the number of features tested, capped at 1 (``p_unit``). Every tested unit carries it, survivor or not; only a unit with no tested feature has a NaN ``p_unit``, so the cohort Benjamini–Hochberg runs over all tested units.

.. _nm-vocal-gating-sign-ridge-frac:

``sign_ridge_frac``
   The interaction's SIGN is read from a strongly ridged fit (``sign_ridge_frac × n``) while its magnitude comes from the deviance.

   :Options: a fraction multiplying the number of rows
   :Default: ``0.1``
   :Notes: an unridged interaction coefficient quasi-separates, hence the separate ridged fit for the sign.

.. _nm-vocal-gating-borderline-sigma-flag:

``borderline_sigma_flag``
   A passing gate whose margin over its bar is under this many null standard deviations is flagged ``borderline``.

   :Options: a number of null standard deviations
   :Default: ``5.0``

.. _nm-vocal-gating-solver:

``solver``
   Iteration counts of the gating fits.

   :Options:
      * ``irls_n_steps`` — Newton/IRLS iterations of each logistic fit
      * ``calibration_steps`` — iterations of each calibration refit
   :Default: ``irls_n_steps`` ``30``, ``calibration_steps`` ``40``

**vocal_decoding** — the vocal-occurrence axis (does the unit fire before a USV?) and the vocalization-identity axis (does its count say WHICH USV, as a torus position or a category?).

.. code-block:: json

    "vocal_decoding": {
        "require_clean_prevocal_bool": true,
        "usv_manifold_column_names": ["qlvm1", "qlvm2"],
        "max_quiet_tiles_per_session": 5000,
        "vocal_occurrence": {
            "ridge_frac": 0.02,
            "sigma_floor": 0.05,
            "solver": {
                "irls_n_steps": 30,
                "calibration_steps": 40
            }
        },
        "geodesic_metrics": {
            "compute": true,
            "grid_n_per_dim": 40,
            "graph_k": 8,
            "density_exponent": 1.0,
            "decoder_weights_npz_path": "/mnt/falkner/Bartul/spectrograms/qlvm/qmc_decoder_weights.npz"
        },
        "tuning_surface": {
            "tuning_basis": "fourier",
            "fourier_k": 2,
            "bump_grid_n": 10,
            "bump_sigma": 0.1,
            "min_basis_occupancy": 5.0,
            "decode_grid_min_occupancy": 1.0,
            "grid_n": 60,
            "kde_bandwidth": 0.03,
            "likelihood_family": "poisson",
            "session_rate_offsets": true,
            "solver": {
                "irls_n_steps": 50
            },
            "ridge_selection": {
                "mode": "count_normalized_fixed",
                "ridge_frac": 0.02
            },
            "amplitude_calibration": {
                "enabled": true,
                "mode": "train_tuned_inner_loso",
                "alpha_grid": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5],
                "alpha_min": 0.0
            },
            "record_bump_basis_gain": true
        },
        "category_surface": {
            "ridge_selection": {
                "mode": "count_normalized_fixed",
                "ridge_frac": 0.02
            },
            "amplitude_calibration": {
                "enabled": true,
                "mode": "train_tuned_inner_loso",
                "alpha_grid": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5]
            },
            "category_column": "qlvm_category",
            "min_train_events": 20,
            "min_test_events": 5,
            "mean_count_floor": 0.001
        },
        "targets": ["manifold_position"],
        "discrimination_null": {
            "method": "within_session_position_permutation",
            "n_permutations": 1000,
            "sided": "one_sided_upper",
            "pass_rule": "gain_gt_zero_and_exact_p",
            "record_left_tail_anti_alignment": true,
            "n_permutations_category_grain": 5000
        },
        "record_overdispersion_index": true
    }

.. list-table:: ``vocal_decoding`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`require_clean_prevocal_bool <nm-vocal-decoding-require-clean-prevocal-bool>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`usv_manifold_column_names <nm-vocal-decoding-usv-manifold-column-names>`
     - ``["qlvm1", "qlvm2"]``
     - two summary column names
   * - :ref:`max_quiet_tiles_per_session <nm-vocal-decoding-max-quiet-tiles-per-session>`
     - ``5000``
     - tiles per session
   * - :ref:`vocal_occurrence.ridge_frac <nm-vocal-decoding-vocal-occurrence>`
     - ``0.02``
     - fraction of the training count
   * - :ref:`vocal_occurrence.sigma_floor <nm-vocal-decoding-vocal-occurrence>`
     - ``0.05``
     - standard-deviation floor
   * - :ref:`vocal_occurrence.solver.irls_n_steps <nm-vocal-decoding-vocal-occurrence>`
     - ``30``
     - IRLS iterations
   * - :ref:`vocal_occurrence.solver.calibration_steps <nm-vocal-decoding-vocal-occurrence>`
     - ``40``
     - calibration-refit iterations
   * - :ref:`geodesic_metrics.compute <nm-vocal-decoding-geodesic-metrics>`
     - ``true``
     - declared, not read
   * - :ref:`geodesic_metrics.grid_n_per_dim <nm-vocal-decoding-geodesic-metrics>`
     - ``40``
     - declared, not read
   * - :ref:`geodesic_metrics.graph_k <nm-vocal-decoding-geodesic-metrics>`
     - ``8``
     - declared, not read
   * - :ref:`geodesic_metrics.density_exponent <nm-vocal-decoding-geodesic-metrics>`
     - ``1.0``
     - declared, not read
   * - :ref:`geodesic_metrics.decoder_weights_npz_path <nm-vocal-decoding-geodesic-metrics>`
     - path (see JSON above)
     - declared, not read
   * - :ref:`tuning_surface.tuning_basis <nm-vocal-decoding-tuning-surface>`
     - ``'fourier'``
     - ``'fourier'``, ``'bumps'``
   * - :ref:`tuning_surface.fourier_k <nm-vocal-decoding-tuning-surface>`
     - ``2``
     - harmonics per axis
   * - :ref:`tuning_surface.bump_grid_n <nm-vocal-decoding-tuning-surface>`
     - ``10``
     - bump nodes per axis
   * - :ref:`tuning_surface.bump_sigma <nm-vocal-decoding-tuning-surface>`
     - ``0.1``
     - bump width
   * - :ref:`tuning_surface.min_basis_occupancy <nm-vocal-decoding-tuning-surface>`
     - ``5.0``
     - kernel-weighted events
   * - :ref:`tuning_surface.decode_grid_min_occupancy <nm-vocal-decoding-tuning-surface>`
     - ``1.0``
     - KDE training occupancy
   * - :ref:`tuning_surface.grid_n <nm-vocal-decoding-tuning-surface>`
     - ``60``
     - decoding cells per axis
   * - :ref:`tuning_surface.kde_bandwidth <nm-vocal-decoding-tuning-surface>`
     - ``0.03``
     - KDE bandwidth
   * - :ref:`tuning_surface.likelihood_family <nm-vocal-decoding-tuning-surface>`
     - ``'poisson'``
     - ``'poisson'`` (only value)
   * - :ref:`tuning_surface.session_rate_offsets <nm-vocal-decoding-tuning-surface>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`tuning_surface.solver.irls_n_steps <nm-vocal-decoding-tuning-surface>`
     - ``50``
     - IRLS iterations
   * - :ref:`tuning_surface.ridge_selection.mode <nm-vocal-decoding-tuning-surface>`
     - ``'count_normalized_fixed'``
     - ``'count_normalized_fixed'`` (only value)
   * - :ref:`tuning_surface.ridge_selection.ridge_frac <nm-vocal-decoding-tuning-surface>`
     - ``0.02``
     - fraction of the training spike count
   * - :ref:`tuning_surface.amplitude_calibration.enabled <nm-vocal-decoding-tuning-surface>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`tuning_surface.amplitude_calibration.mode <nm-vocal-decoding-tuning-surface>`
     - ``'train_tuned_inner_loso'``
     - ``'train_tuned_inner_loso'`` (only value)
   * - :ref:`tuning_surface.amplitude_calibration.alpha_grid <nm-vocal-decoding-tuning-surface>`
     - ``[0.1, 0.25, 0.5, 0.75, 1.0, 1.5]``
     - list of amplitudes
   * - :ref:`tuning_surface.amplitude_calibration.alpha_min <nm-vocal-decoding-tuning-surface>`
     - ``0.0``
     - smallest allowed grid value
   * - :ref:`tuning_surface.record_bump_basis_gain <nm-vocal-decoding-tuning-surface>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`category_surface.ridge_selection.mode <nm-vocal-decoding-category-surface>`
     - ``'count_normalized_fixed'``
     - as ``tuning_surface``
   * - :ref:`category_surface.ridge_selection.ridge_frac <nm-vocal-decoding-category-surface>`
     - ``0.02``
     - as ``tuning_surface``
   * - :ref:`category_surface.amplitude_calibration.enabled <nm-vocal-decoding-category-surface>`
     - ``true``
     - as ``tuning_surface``
   * - :ref:`category_surface.amplitude_calibration.mode <nm-vocal-decoding-category-surface>`
     - ``'train_tuned_inner_loso'``
     - as ``tuning_surface``
   * - :ref:`category_surface.amplitude_calibration.alpha_grid <nm-vocal-decoding-category-surface>`
     - ``[0.1, 0.25, 0.5, 0.75, 1.0, 1.5]``
     - as ``tuning_surface``
   * - :ref:`category_surface.category_column <nm-vocal-decoding-category-surface>`
     - ``'qlvm_category'``
     - declared, not read
   * - :ref:`category_surface.min_train_events <nm-vocal-decoding-category-surface>`
     - ``20``
     - events per split
   * - :ref:`category_surface.min_test_events <nm-vocal-decoding-category-surface>`
     - ``5``
     - events per split
   * - :ref:`category_surface.mean_count_floor <nm-vocal-decoding-category-surface>`
     - ``0.001``
     - small positive number
   * - :ref:`targets <nm-vocal-decoding-targets>`
     - ``["manifold_position"]``
     - declared, not read
   * - :ref:`discrimination_null.method <nm-vocal-decoding-discrimination-null>`
     - ``'within_session_position_permutation'``
     - declared, not read
   * - :ref:`discrimination_null.n_permutations <nm-vocal-decoding-discrimination-null>`
     - ``1000``
     - number of permutation draws
   * - :ref:`discrimination_null.sided <nm-vocal-decoding-discrimination-null>`
     - ``'one_sided_upper'``
     - declared, not read
   * - :ref:`discrimination_null.pass_rule <nm-vocal-decoding-discrimination-null>`
     - ``'gain_gt_zero_and_exact_p'``
     - declared, not read
   * - :ref:`discrimination_null.record_left_tail_anti_alignment <nm-vocal-decoding-discrimination-null>`
     - ``true``
     - ``true``, ``false``
   * - :ref:`discrimination_null.n_permutations_category_grain <nm-vocal-decoding-discrimination-null>`
     - ``5000``
     - declared, not read
   * - :ref:`record_overdispersion_index <nm-vocal-decoding-record-overdispersion-index>`
     - ``true``
     - ``true``, ``false``

.. _nm-vocal-decoding-require-clean-prevocal-bool:

``require_clean_prevocal_bool``
   Whether to drop an event when another vocalization intrudes on its pre-vocal window.

   :Options:
      * ``true`` — drop such an event
      * ``false`` — keep it
   :Default: ``true``

.. _nm-vocal-decoding-usv-manifold-column-names:

``usv_manifold_column_names``
   The summary columns holding a USV's torus position.

   :Options: a list of the two summary column names of the torus coordinates
   :Default: ``['qlvm1', 'qlvm2']`` (the regular map)

.. _nm-vocal-decoding-max-quiet-tiles-per-session:

``max_quiet_tiles_per_session``
   Cap on the quiet tiles that serve as the vocal-occurrence negatives.

   :Options: a number of tiles per session
   :Default: ``5000`` per session

.. _nm-vocal-decoding-vocal-occurrence:

``vocal_occurrence``
   The fit of the vocal-occurrence axis.

   :Options:
      * ``ridge_frac`` — the ridge, scaled by the training count
      * ``sigma_floor`` — a floor on the standard deviation that z-scores the count column, so a unit silent in quiet cannot produce an exploding column
      * ``solver.irls_n_steps`` / ``solver.calibration_steps`` — the ``solver`` steps: IRLS iterations of each fit and of each calibration refit
   :Default: ``ridge_frac`` ``0.02``, ``sigma_floor`` ``0.05``, ``irls_n_steps`` ``30``, ``calibration_steps`` ``40``

.. _nm-vocal-decoding-geodesic-metrics:

``geodesic_metrics``
   Declared but read by no current neural code.

   :Options: the sub-keys ``compute``, ``grid_n_per_dim``, ``graph_k``, ``density_exponent`` and ``decoder_weights_npz_path``, none of which has an effect
   :Default: as in the JSON above

.. _nm-vocal-decoding-tuning-surface:

``tuning_surface``
   The Poisson tuning surface over the torus.

   :Options:
      * ``tuning_basis`` — ``'fourier'``: a Fourier basis with ``fourier_k`` harmonics per axis; ``'bumps'``: a local basis of ``bump_grid_n`` nodes per axis of width ``bump_sigma``, kept where the kernel-weighted training occupancy reaches ``min_basis_occupancy`` events
      * ``fourier_k`` — Fourier harmonics per axis
      * ``bump_grid_n`` / ``bump_sigma`` / ``min_basis_occupancy`` — the bump basis's nodes per axis, node width and occupancy floor (events)
      * ``grid_n`` — decoding runs on a grid of this many cells per axis, restricted to cells whose training occupancy (a KDE of bandwidth ``kde_bandwidth``) reaches ``decode_grid_min_occupancy``
      * ``likelihood_family`` — ``'poisson'`` (the only implemented value)
      * ``session_rate_offsets`` — ``true`` adds per-session rate offsets
      * ``solver.irls_n_steps`` — IRLS iterations of the surface fit
      * ``ridge_selection`` — ``mode`` ``'count_normalized_fixed'`` (the only implemented value; the ridge is ``ridge_frac`` × the training spike count)
      * ``amplitude_calibration`` — when ``enabled``, a likelihood-sharpening amplitude is tuned over ``alpha_grid`` by inner leave-one-session-out on the training sessions (``mode`` ``'train_tuned_inner_loso'``, the only implemented value; ``alpha_min`` is the smallest grid value allowed), and when disabled the amplitude is 1
      * ``record_bump_basis_gain`` — also record the bump-basis gain beside the Fourier one
   :Default: ``tuning_basis`` ``'fourier'``, ``fourier_k`` ``2``, ``bump_grid_n`` ``10``, ``bump_sigma`` ``0.1``, ``min_basis_occupancy`` ``5.0``, ``grid_n`` ``60``, ``kde_bandwidth`` ``0.03``, ``decode_grid_min_occupancy`` ``1.0``, ``likelihood_family`` ``'poisson'``, ``session_rate_offsets`` ``true``, ``solver.irls_n_steps`` ``50``, ``ridge_selection.ridge_frac`` ``0.02``, ``alpha_min`` ``0.0``, ``record_bump_basis_gain`` ``true``; the rest as in the JSON above

.. _nm-vocal-decoding-category-surface:

``category_surface``
   The same decoder over the QLVM categories.

   :Options:
      * ``ridge_selection`` / ``amplitude_calibration`` — as in ``tuning_surface`` above
      * ``category_column`` — declared but not read
      * ``min_train_events`` / ``min_test_events`` — an outer or inner split with fewer training or test events than these is skipped, too few for a stable per-category rate or a meaningful held-out score
      * ``mean_count_floor`` — the lower bound on a session's mean spike count when the held-out level correction is formed, so a session with no spikes gives a posterior equal to the prior (zero gain) rather than ``0 · log 0`` = NaN; a positive number, small against any real session's mean count
   :Default: ``min_train_events`` ``20``, ``min_test_events`` ``5``, ``mean_count_floor`` ``0.001``; the rest as in the JSON above

.. _nm-vocal-decoding-targets:

``targets``
   Declared but read by no current code.

   :Options: no effect; the position target is always decoded
   :Default: ``["manifold_position"]``

.. _nm-vocal-decoding-discrimination-null:

``discrimination_null``
   The exact within-session permutation null of the identity axis.

   :Options:
      * ``n_permutations`` — the number of permutation draws
      * ``record_left_tail_anti_alignment`` — record the left-tail statistics
      * ``method``, ``sided``, ``pass_rule`` and ``n_permutations_category_grain`` — declared but read by no current code
   :Default: ``n_permutations`` ``1000``, ``record_left_tail_anti_alignment`` ``true``; the rest as in the JSON above

.. _nm-vocal-decoding-record-overdispersion-index:

``record_overdispersion_index``
   Record the count overdispersion index with each decoding.

   :Options:
      * ``true`` — record it
      * ``false`` — do not
   :Default: ``true``
   :Notes:
      * **Definition** — the index is the mean Pearson dispersion ``(count − expected)² / expected`` of the held-out events, where an event's expected count is the calibrated surface rate the decoder scored it with (the centred surface shape times the tuned amplitude, at the held-out session's mean-count level) read at the event's own grid cell, so an untuned Poisson unit gives about ``1``.
      * **Also stored** — the decoding also stores ``nats_per_spike`` (the summed finite per-event gain divided by the spikes of those same events) and ``min_leave_one_fold_out`` (the smallest pooled gain with one fold left out, over the folds whose left-out gain is finite).

**nested_vocal_manifold_position_decoding** — does the neuron add information about a USV's torus position beyond the cohort behaviour control? A REDUCED model (behaviour alone) and a FULL model (behaviour plus the neuron's count) are fitted by the same torus regressor and compared on the held-out macro von Mises log-score.

.. code-block:: json

    "nested_vocal_manifold_position_decoding": {
        "behaviour_control": "rich",
        "reduced_model_features": ["head-head", "allo_yaw-head", "allo_pitch-head", "self.speed"],
        "behaviour_control_model_path": "/mnt/falkner/Bartul/modeling/model_selection_results/behaviour_control_model_ephys_cohort.pkl",
        "vm_score_mode": "macro",
        "region_label_column": "qlvm_category",
        "min_region_events": 20,
        "lambda_smooth": 1.0,
        "l2_reg": 0.01,
        "smoothness_derivative_order": 1,
        "huber_delta": 1.0,
        "learning_rate": 0.001,
        "max_iter": 5000,
        "tol": 0.0001,
        "random_state": 0,
        "behaviour_control_n_lags": 600,
        "sigma_floor": 0.05,
        "prevocal_window_n_bins": 1,
        "rate_transform": "none",
        "rate_basis": "linear",
        "statistic": "posterior",
        "behaviour_prior": {
            "form": "von_mises",
            "concentration": "train_tuned_inner_loso",
            "kappa_grid": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        },
        "discrimination_null": {
            "method": "within_session_pair_permutation",
            "n_permutations": 1000,
            "sided": "one_sided_upper"
        }
    }

.. list-table:: ``nested_vocal_manifold_position_decoding`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`behaviour_control <nm-nested-behaviour-control>`
     - ``'rich'``
     - ``'rich'`` (only value)
   * - :ref:`reduced_model_features <nm-nested-reduced-model-features>`
     - see JSON above
     - the frozen feature list
   * - :ref:`behaviour_control_model_path <nm-nested-behaviour-control-model-path>`
     - path (see JSON above)
     - path to the fitted artifact
   * - :ref:`vm_score_mode <nm-nested-vm-score-mode>`
     - ``'macro'``
     - ``'macro'`` (only value)
   * - :ref:`region_label_column <nm-nested-region-labels>`
     - ``'qlvm_category'``
     - summary column name
   * - :ref:`min_region_events <nm-nested-region-labels>`
     - ``20``
     - events per region
   * - :ref:`lambda_smooth <nm-nested-regressor-penalties>`
     - ``1.0``
     - smoothness strength
   * - :ref:`l2_reg <nm-nested-regressor-penalties>`
     - ``0.01``
     - ridge strength
   * - :ref:`smoothness_derivative_order <nm-nested-regressor-penalties>`
     - ``1``
     - finite-difference order
   * - :ref:`huber_delta <nm-nested-optimiser>`
     - ``1.0``
     - torus units (inert)
   * - :ref:`learning_rate <nm-nested-optimiser>`
     - ``0.001``
     - Adam step size (inert)
   * - :ref:`max_iter <nm-nested-optimiser>`
     - ``5000``
     - step cap (inert)
   * - :ref:`tol <nm-nested-optimiser>`
     - ``0.0001``
     - tolerance (inert)
   * - :ref:`random_state <nm-nested-optimiser>`
     - ``0``
     - seed (inert)
   * - :ref:`behaviour_control_n_lags <nm-nested-behaviour-control-n-lags>`
     - ``600``
     - tracking frames
   * - :ref:`sigma_floor <nm-nested-sigma-floor>`
     - ``0.05``
     - standard-deviation floor
   * - :ref:`prevocal_window_n_bins <nm-nested-neuron-column>`
     - ``1``
     - ``1`` (only value)
   * - :ref:`rate_transform <nm-nested-neuron-column>`
     - ``'none'``
     - ``'none'`` (only value)
   * - :ref:`rate_basis <nm-nested-neuron-column>`
     - ``'linear'``
     - ``'linear'`` (only value)
   * - :ref:`statistic <nm-nested-neuron-column>`
     - ``'posterior'``
     - ``'posterior'`` (only value)
   * - :ref:`behaviour_prior.form <nm-nested-behaviour-prior>`
     - ``'von_mises'``
     - ``'von_mises'``
   * - :ref:`behaviour_prior.concentration <nm-nested-behaviour-prior>`
     - ``'train_tuned_inner_loso'``
     - ``'train_tuned_inner_loso'``, ``'train_residuals'``
   * - :ref:`behaviour_prior.kappa_grid <nm-nested-behaviour-prior>`
     - ``[0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]``
     - list of concentrations
   * - :ref:`discrimination_null.method <nm-nested-discrimination-null>`
     - ``'within_session_pair_permutation'``
     - ``'within_session_pair_permutation'`` (only value)
   * - :ref:`discrimination_null.n_permutations <nm-nested-discrimination-null>`
     - ``1000``
     - number of permutation draws
   * - :ref:`discrimination_null.sided <nm-nested-discrimination-null>`
     - ``'one_sided_upper'``
     - declared, not read

.. _nm-nested-behaviour-control:

``behaviour_control``
   Which behaviour control the REDUCED model uses.

   :Options: ``'rich'`` — the full 4 s behaviour window (the only implemented value)
   :Default: ``'rich'``

.. _nm-nested-reduced-model-features:

``reduced_model_features``
   The frozen behaviour-control feature list.

   :Options: a list of behaviour feature names
   :Default: ``["head-head", "allo_yaw-head", "allo_pitch-head", "self.speed"]`` (the JSON above)
   :Notes: asserted against the selection result file, so a re-run of the selection cannot silently redefine the behaviour the neuron is measured against.

.. _nm-nested-behaviour-control-model-path:

``behaviour_control_model_path``
   The cohort-fitted behaviour-control artifact (``main_behaviour_control_dispatcher --stage fit``).

   :Options: path to the fitted artifact
   :Default: see the JSON above
   :Notes:
      * **Agreement** — the decoding refuses an artifact that disagrees with these settings.
      * **Rows** — the control is fitted on exactly the vocalization rows the per-unit analysis scores: the ``'analysed'`` set of ``vocalization_settings`` (noise and squeak rows removed) restricted to ``vocal_emitter`` (``'all'`` applies no emitter filter), with a vocalization dropped when one from the ``'guard'`` set intrudes on its ``spike_window.pre_offset_seconds`` pre-window.
      * **History end** — every behaviour history, at fit and at application, ends at the frame before onset (``round(start × fps) − 1``).
      * **Old artifacts** — an artifact fitted before 2026-10-18 was trained on the unfiltered summary with histories one frame earlier than the unit side used, and must be refitted. The artifact's counts are stored as ``n_vocalizations`` (top level and per day); an older artifact's ``n_calls`` is read under that name.

.. _nm-nested-vm-score-mode:

``vm_score_mode``
   The held-out von Mises score the REDUCED and FULL models are compared on.

   :Options: ``'macro'`` — the region-balanced von Mises score (the only implemented value)
   :Default: ``'macro'``

.. _nm-nested-region-labels:

``region_label_column`` / ``min_region_events``
   The acoustic regions of the macro score.

   :Options:
      * ``region_label_column`` — the summary column holding the acoustic-region labels of the macro score
      * ``min_region_events`` — the minimum events a region needs to enter it
   :Default: ``qlvm_category`` and ``20``

.. _nm-nested-regressor-penalties:

``lambda_smooth`` / ``l2_reg`` / ``smoothness_derivative_order``
   The torus regressor's penalties.

   :Options:
      * ``lambda_smooth`` — temporal smoothness (behaviour blocks only)
      * ``l2_reg`` — uniform ridge
      * ``smoothness_derivative_order`` — finite-difference order of the smoothness penalty
   :Default: ``1.0``, ``0.01`` and ``1``

.. _nm-nested-optimiser:

``huber_delta`` / ``learning_rate`` / ``max_iter`` / ``tol`` / ``random_state``
   The regressor's optimiser.

   :Options:
      * ``huber_delta`` — Huber transition point, in torus units
      * ``learning_rate`` — initial Adam step size, cosine-decayed to zero
      * ``max_iter`` — step cap
      * ``tol`` — parameter-change convergence tolerance
      * ``random_state`` — initialisation seed
   :Default: ``1.0``, ``0.001``, ``5000``, ``1e-4`` and ``0``
   :Notes: these are the values the regressor used before they became settings; the shipped torus regressor is fitted in closed form, where the optimiser keys are inert.

.. _nm-nested-behaviour-control-n-lags:

``behaviour_control_n_lags``
   History length of the behaviour design, in tracking frames.

   :Options: a number of tracking frames
   :Default: ``600`` (= 4 s at 150 fps)
   :Notes: used by ``main_behaviour_control_dispatcher`` for both stages; the fitted artifact records it, and the per-unit decoding reads it from there.

.. _nm-nested-sigma-floor:

``sigma_floor``
   Floor on the standard deviation z-scoring the neuron's count column.

   :Options: a standard-deviation floor
   :Default: ``0.05``
   :Notes: the same guard as the vocal-occurrence axis.

.. _nm-nested-neuron-column:

``prevocal_window_n_bins`` / ``rate_transform`` / ``rate_basis`` / ``statistic``
   How the neuron enters the FULL model: as ONE z-scored pre-vocal count, and the tested statistic is the posterior form.

   :Options:
      * ``prevocal_window_n_bins`` — ``1`` bin
      * ``rate_transform`` — ``'none'``, no transform
      * ``rate_basis`` — ``'linear'``, a linear column
      * ``statistic`` — ``'posterior'``, the posterior form
   :Default: ``1``, ``'none'``, ``'linear'`` and ``'posterior'``
   :Notes: each accepts only that value.

.. _nm-nested-behaviour-prior:

``behaviour_prior``
   The von Mises behaviour prior of the posterior statistic.

   :Options:
      * ``form`` — ``'von_mises'``
      * ``concentration`` — ``'train_tuned_inner_loso'`` picks it from ``kappa_grid`` by inner leave-one-session-out on the training sessions; ``'train_residuals'`` fits it on the training residuals
      * ``kappa_grid`` — the candidate concentrations
   :Default: ``form`` ``'von_mises'``, ``concentration`` ``'train_tuned_inner_loso'``, ``kappa_grid`` as in the JSON above

.. _nm-nested-discrimination-null:

``discrimination_null``
   The within-session pair permutation null.

   :Options:
      * ``method`` — ``'within_session_pair_permutation'`` (the only implemented value)
      * ``n_permutations`` — the number of permutation draws
      * ``sided`` — declared but not read
   :Default: ``method`` ``'within_session_pair_permutation'``, ``n_permutations`` ``1000``, ``sided`` ``'one_sided_upper'``

**null** — the circular-shift null shared by the kinematic encoding, the transfer, the gating and the vocal-occurrence axis (the spike train is shifted, preserving its rate and burstiness while destroying its alignment with behaviour).

.. code-block:: json

    "null": {
        "n_shuffles": 1000,
        "shuffle_guard_seconds": 20.0,
        "shuffle_seed": 0,
        "screen_n_shuffles": 2000
    }

.. list-table:: ``null`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`n_shuffles <nm-null-n-shuffles>`
     - ``1000``
     - base number of shift draws
   * - :ref:`shuffle_guard_seconds <nm-null-shuffle-guard-seconds>`
     - ``20.0``
     - seconds
   * - :ref:`shuffle_seed <nm-null-shuffle-seed>`
     - ``0``
     - integer seed
   * - :ref:`screen_n_shuffles <nm-null-screen-n-shuffles>`
     - ``2000``
     - number of screen draws

.. _nm-null-n-shuffles:

``n_shuffles``
   Base number of shift draws.

   :Options: a number of draws
   :Default: ``1000``
   :Notes: the empirical p-value floors at ``1 / (n + 1)`` and escalation adds draws where the cohort's Benjamini–Hochberg bar requires them.

.. _nm-null-shuffle-guard-seconds:

``shuffle_guard_seconds``
   The shift lag is drawn from ``[guard, duration - guard]``.

   :Options: a number of seconds
   :Default: ``20.0``
   :Notes: the range is the full wrap minus a band where the shifted train is nearly back in register (the statistic's decorrelation lag was measured at 2.2 s).

.. _nm-null-shuffle-seed:

``shuffle_seed``
   Base seed of every shift draw.

   :Options: an integer seed
   :Default: ``0``
   :Notes: each test derives its own stream from it.

.. _nm-null-screen-n-shuffles:

``screen_n_shuffles``
   Draws of the feature screen.

   :Options: a number of draws, whose floor must lie below the screen's Bonferroni bar
   :Default: ``2000``
   :Notes: the screen's Bonferroni bar ``screen_alpha / n_features`` needs a floor below it.

**significance** — cohort-level multiplicity control.

.. code-block:: json

    "significance": {
        "pvalue_method": "empirical_escalated",
        "fdr_method": "bh",
        "fdr_q": 0.01,
        "escalation_ladder": [],
        "report_bh_floor_requirement": true
    }

.. list-table:: ``significance`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`pvalue_method <nm-significance-declared>`
     - ``'empirical_escalated'``
     - declared, not read
   * - :ref:`fdr_method <nm-significance-declared>`
     - ``'bh'``
     - declared, not read
   * - :ref:`fdr_q <nm-significance-fdr-q>`
     - ``0.01``
     - false-discovery rate
   * - :ref:`escalation_ladder <nm-significance-escalation-ladder>`
     - ``[]``
     - list of draw counts
   * - :ref:`report_bh_floor_requirement <nm-significance-declared>`
     - ``true``
     - declared, not read

.. _nm-significance-fdr-q:

``fdr_q``
   The Benjamini–Hochberg false-discovery rate over the cohort.

   :Options: a false-discovery rate
   :Default: ``0.01``

.. _nm-significance-escalation-ladder:

``escalation_ladder``
   Draw counts to escalate floor-tied p-values through (accumulating, not restarting).

   :Options:
      * ``[]`` — runs the base count only
      * a list of draw counts — the rungs floor-tied p-values escalate through
   :Default: ``[]``
   :Notes: each rung counts FINITE draws only, as only finite draws enter the p-value.

.. _nm-significance-declared:

``pvalue_method`` / ``fdr_method`` / ``report_bh_floor_requirement``
   Declared but read by no current code.

   :Options: no effect; p-values are always empirical with escalation and the correction is always Benjamini–Hochberg
   :Default: ``'empirical_escalated'``, ``'bh'`` and ``true``

**robustness** — planned robustness checks; every key is declared but read by no current code.

.. code-block:: json

    "robustness": {
        "decode_raw_acoustics": true,
        "raw_acoustic_features": ["duration", "mean_freq_hz", "peak_freq_hz", "freq_bandwidth_hz"],
        "time_in_session_nuisance": false,
        "shift_range_check": false
    }

.. list-table:: ``robustness`` at a glance
   :header-rows: 1
   :widths: 44 16 40
   :class: settings-glance

   * - Setting
     - Default
     - Options
   * - :ref:`decode_raw_acoustics <nm-robustness-raw-acoustics>`
     - ``true``
     - declared, not read
   * - :ref:`raw_acoustic_features <nm-robustness-raw-acoustics>`
     - see JSON above
     - declared, not read
   * - :ref:`time_in_session_nuisance <nm-robustness-time-in-session-nuisance>`
     - ``false``
     - declared, not read
   * - :ref:`shift_range_check <nm-robustness-shift-range-check>`
     - ``false``
     - declared, not read

.. _nm-robustness-raw-acoustics:

``decode_raw_acoustics`` / ``raw_acoustic_features``
   Decode the raw acoustic features listed instead of the torus position.

   :Options: planned; declared but read by no current code
   :Default: ``true`` and ``["duration", "mean_freq_hz", "peak_freq_hz", "freq_bandwidth_hz"]``

.. _nm-robustness-time-in-session-nuisance:

``time_in_session_nuisance``
   Add time-in-session as a nuisance covariate.

   :Options: planned; declared but read by no current code
   :Default: ``false``

.. _nm-robustness-shift-range-check:

``shift_range_check``
   Repeat the null with a restricted shift range.

   :Options: planned; declared but read by no current code
   :Default: ``false``

Notebook
--------
The ``modeling_analyses.ipynb`` notebook is the recommended interactive
entry point — it runs the whole workflow above in order from a single
**Parameters** cell. Its detailed walkthrough, knobs, and rendered source
live in :doc:`Notebooks`.
