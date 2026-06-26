# Visualizations subsystem review

_Verified line-by-line sweep of 12 files (~29k LOC): 203 findings. Report-first._

## Summary
- by severity: high 14 · medium 60 · low 129
- by dimension: tests 67 · docs_clarity 55 · correctness 41 · performance 24 · dead_code_naming 16


## `make_neuronal_tuning_figures.py` (34)

### [HIGH] docs_clarity — make_pag_anatomical_gradient_figure docstring says 3 panels but renders 4 and mislabels panels 2/3
`make_neuronal_tuning_figures.py:2373-2390`

Docstring describes 'three panels' with panels 2/3 as raw '2-D KDE density'. Implementation renders four: scatter, occupancy-normalized sig +VMI fraction, occupancy-normalized sig -VMI fraction, and a divergent sig+ minus sig- difference panel (2636-2656). Docstring predates the KDE->fraction switch and the 4th panel.

**Fix:** Enumerate four panels (scatter; +VMI fraction; -VMI fraction; sig+ minus sig- divergent diff) and change 'All three panels' to 'All four panels'.

### [MEDIUM] docs_clarity — Stale comment claims a 'sane fallback for older pkls' the code does not provide
`make_neuronal_tuning_figures.py:5590-5591`

Comment says the plotter reads behavioral_min_occupancy_seconds 'with a sane fallback for older pkls', but 5592-5594 reads it via direct subscript with no fallback (an older pkl would KeyError).

**Fix:** Drop the 'with a sane fallback for older pkls' clause; correct the comment to state the key is required.

### [MEDIUM] docs_clarity — make_peth_timing_distribution_figure docstring omits the 'direction' parameter
`make_neuronal_tuning_figures.py:3245-3268`

Signature (3213) takes direction: str = 'excit', validated at 3275-3278 and threaded into _collect_consistent_peth and the output stem, but the Parameters section never documents it; Description (3224) still says 'excit-PETH' only.

**Fix:** Add a direction (str) Parameters entry ('excit' or 'suppress', validated, default 'excit') and generalize the Description.

### [MEDIUM] docs_clarity — _collect_consistent_property docstring omits 'direction' and hardcodes 'excit' in the modality-key example
`make_neuronal_tuning_figures.py:3424-3442`

Signature (3394) takes direction: str = 'excit' (validated 3479-3482; key f'usv_property_self_{property_name}_{direction}' at 3483). Description (3404) hardcodes the key '..._excit' and the Parameters section never documents direction.

**Fix:** Change the example key to usv_property_self_<property>_<direction> and add a direction (str) Parameters entry.

### [MEDIUM] performance — make_all_category_figures collects the same data twice per segmentation (pickle reloaded 8x)
`make_neuronal_tuning_figures.py:4360-4378`

For each of 4 segmentations it calls both make_category_peak_distribution_figure (4361) and make_category_selectivity_breadth_figure (4370), each independently calling _collect_consistent_category_self (4014, 4160) with identical args, re-reading the triage pickle (3892) and re-parsing unit_catalog.csv (3898-3900). 8 loads for 4 distinct collections.

**Fix:** Collect once per segmentation in make_all_category_figures and pass per_group into render-only helpers, or memoize on (triage_pkl_path, segmentation, k_min, require_majority).

### [MEDIUM] tests — direction-validation ValueErrors (PETH + property collectors/figures) untested
`make_neuronal_tuning_figures.py:3131-3134, 3275-3278, 3479-3482, 3628-3631`

Four entry points raise ValueError for direction not in ('excit','suppress'); the only direction test parametrizes valid 'excit'/'suppress'. All four raise-lines unhit.

**Fix:** Add tests invoking each method with direction='up' under pytest.raises(ValueError).

### [MEDIUM] tests — property_name and segmentation ValueErrors untested
`make_neuronal_tuning_figures.py:3452-3456, 3884-3888`

_collect_consistent_property raises for unknown property_name (3453-3456) and _collect_consistent_category_self raises for unknown segmentation (3885-3888); neither is tested.

**Fix:** Add pytest.raises(ValueError) calls passing property_name='bogus' and segmentation='bogus' using triage_fixture.

### [LOW] correctness — _load_segmentation leaks the NpzFile handle
`make_neuronal_tuning_figures.py:463`

np.load on a .npz returns a lazy NpzFile holding the zip handle open until closed; out copies every array but data is never closed.

**Fix:** Use `with np.load(self._segmentation_path, allow_pickle=True) as data:` and build out inside the with block.

### [LOW] correctness — Constructor uses kwargs.get() with defaults and double-assigns
`make_neuronal_tuning_figures.py:417`

Lines 410-411 set every kwarg via self.__dict__; 417/422 re-assign kslabels/somatic_filter via .get() with defaults, violating the project no-default-.get() convention.

**Fix:** Use explicit membership checks mirroring figures/fig_format style (e.g. frozenset(kwargs['kslabels']) if 'kslabels' in kwargs else frozenset(('good',))).

### [LOW] correctness — make_vmi_distribution_figure: documented shared y-axis never applied (dead y_max_per_panel + wrong 'scatter' wording)
`make_neuronal_tuning_figures.py:2920-2961`

Comment at 2920-2921 says the seven panels share a y-axis (and calls them 'scatter' panels though they are histograms); y_max_per_panel is appended at 2961 but never read (grep: only 2922 init + 2961 append). No set_ylim is applied, so panels auto-scale independently.

**Fix:** Either apply ax.set_ylim(0, max(y_max_per_panel)) in a second pass, or delete y_max_per_panel and reword the comment to 'seven per-region histogram panels auto-scale independently'.

### [LOW] correctness — _draw_bout_raster: anchor_partner_sex and partner-color branch are dead
`make_neuronal_tuning_figures.py:6160`

The skip `if em is not None and em != emitter: continue` (6185-6186) runs after the color computation (6178) but before ax.hlines (6187); only em==emitter (anchor color) and em is None (COLOR_LIGHT) ever reach the draw, so _sex_color(anchor_partner_sex) is computed and discarded for every partner USV.

**Fix:** Move the continue above the color computation, simplify to `color = self._sex_color(anchor_sex) if em == emitter else COLOR_LIGHT`, and delete anchor_partner_sex.

### [LOW] dead_code_naming — Unused p_val from spearmanr unpacking
`make_neuronal_tuning_figures.py:4245`

Line 4245 binds `rho, p_val = spearmanr(x, y)` but only rho is used (4246-4247). The sibling call at 967 already uses `rho, _ = spearmanr(...)`.

**Fix:** Change to `rho, _ = spearmanr(x, y)` to match line 967.

### [LOW] docs_clarity — make_property_tuning_distribution_figure docstring omits the 'direction' parameter
`make_neuronal_tuning_figures.py:3593-3611`

Signature (3561) takes direction: str = 'excit', validated 3628-3631 and used in caption/output stem; Parameters section omits it. Description (3571) still says 'excit-tuned' only.

**Fix:** Add a direction (str) Parameters entry consistent with the other property/PETH methods.

### [LOW] docs_clarity — make_all_property_tuning_distribution_figures docstring omits the 'direction' parameter
`make_neuronal_tuning_figures.py:3783-3798`

Signature (3768) takes direction: str = 'excit' and forwards it at 3812; Parameters section omits it.

**Fix:** Add a direction (str) Parameters entry ('excit' or 'suppress'; forwarded to each per-property figure; default 'excit').

### [LOW] docs_clarity — make_vmi_distribution_figure opening says 'best-session signed VMI' but uses a hybrid median/max/min metric
`make_neuronal_tuning_figures.py:2856-2857`

Opening sentence says 'best-session signed VMI'; the per-unit value (from _collect_vmi_distribution_per_unit) is hybrid (non-sig median, sig+ max, sig- min, sig-both dominant), which the same docstring (2872-2877) and caption describe correctly.

**Fix:** Replace 'best-session signed VMI' with 'one signed VMI value per unit (hybrid median/max/min, see below)'.

### [LOW] docs_clarity — make_three_set_overlap_venn_figure docstring calls the first set 'Behavioral' but rendered label is 'Kinematics'
`make_neuronal_tuning_figures.py:5126`

Description says '(Behavioral / Social / Vocal)' and the set-definition bullet is '**Behavioral**' (5144), but set_labels at 5054 is ('Kinematics','Social Features','Vocal') and the per-region sibling uses 'Kinematics'.

**Fix:** Standardize on rendered names: '(Kinematics / Social Features / Vocal)' and '**Kinematics**' (pose OR movement).

### [LOW] docs_clarity — make_vmi_magnitude_consistency_figure hardcodes 'good + somatic' instead of the configurable filter
`make_neuronal_tuning_figures.py:1652`

Description says 'each good + somatic unit's max |VMI|' although unit scope is configurable (kslabels/somatic_filter) and the caption uses self._unit_filter_label(); sibling collector docstrings (e.g. 738) say 'configured unit filter (default good + somatic)'.

**Fix:** Reword to 'each unit passing the configured unit filter (default good + somatic)'.

### [LOW] docs_clarity — make_pag_anatomical_gradient_figure hardcodes 'good + somatic' instead of the configurable filter
`make_neuronal_tuning_figures.py:2376`

Panel 1 description says 'every good + somatic PAG unit' although unit scope is configurable and the caption uses self._unit_filter_label().

**Fix:** Reword to 'every PAG unit passing the configured unit filter (default good + somatic)'.

### [LOW] docs_clarity — Stale comment: closing an empty PdfPages no longer raises (mpl 3.10.7)
`make_neuronal_tuning_figures.py:694-696`

Comment at 694-696 asserts 'closing one [PdfPages] with zero pages raises'; verified on installed matplotlib 3.10.7 that closing an empty PdfPages does NOT raise and writes no file. A cluster whose only usv_* key has an empty usv_peth sets has_vocal=True but _render_vocal_pages (5881) iterates usv_peth zero times, silently producing no file.

**Fix:** Correct the comment to reflect current matplotlib (empty PdfPages writes no file, does not raise); optionally tighten has_vocal to require a non-empty usv_peth payload.

### [LOW] performance — make_vmi_fr_confound_figure loads the triage pickle twice
`make_neuronal_tuning_figures.py:910-917`

_collect_vmi_best_session (910) already loads the pickle (reading thresholds_used at 783-785); the method re-opens and re-loads the same pickle at 914-915 only to read vmi_alpha/vmi_min_bouts.

**Fix:** Have the collector also return the threshold scalars, or read thresholds_used once and pass them down, eliminating the redundant pickle.load.

### [LOW] performance — make_all_property_tuning_distribution_figures reloads pickle/CSV once per property (8x)
`make_neuronal_tuning_figures.py:3807-3817`

Loops over USV_PROPERTY_ORDER (8) calling make_property_tuning_distribution_figure, each calling _collect_consistent_property (3632) which re-reads the triage pickle (3462) and re-parses unit_catalog.csv (3468-3470). The pickle + catalog are property-independent and amortizable.

**Fix:** If on a hot path, load the triage pickle + build cat_lookup once in the wrapper and thread them through (optional preloaded args in _collect_consistent_property).

### [LOW] performance — Bout raster draws each in-window USV with a separate ax.hlines call
`make_neuronal_tuning_figures.py:6168-6195`

_draw_bout_raster loops over bouts (6168) and inner USVs (6171) issuing one ax.hlines per USV (6187); after the partner-emitter skip (6185) only two color groups remain. The spike side already batches into one ax.vlines (6198).

**Fix:** Accumulate (xmin,xmax,y) for anchor-emitter and unassigned USVs into two lists and issue two batched ax.hlines calls after the loop.

### [LOW] performance — Watershed value_grid built with a Python list comprehension over every pixel
`make_neuronal_tuning_figures.py:6732-6739`

_draw_categorical_watershed maps each pixel via a Python comprehension over label_grid.ravel() (6733-6738) with per-element dict membership + int() cast, run once per categorical cell over the full grid.

**Fix:** Vectorize with a numpy LUT (np.full(max_label+1, nan) assigned at category positions, indexed by label_grid) or np.isin/np.where.

### [LOW] performance — np.vectorize(label_to_dense.get) over label_grid is a disguised Python loop
`make_neuronal_tuning_figures.py:6773`

dense = np.vectorize(label_to_dense.get)(label_grid) at 6773 calls Python dict.get per pixel; unique_labels_in_grid (6771) is already sorted by np.unique.

**Fix:** Replace with dense = np.searchsorted(unique_labels_in_grid, label_grid).astype(np.int32).

### [LOW] performance — region_to_group dict rebuilt from a module constant in every collector
`make_neuronal_tuning_figures.py:787-791`

Ten collectors (789,1115,1584,2009,2769,3126,3474,3904,4713,5257) rebuild the same inversion of module-constant VMI_REGION_GROUPS; the mapping is invariant.

**Fix:** Compute VMI_REGION_TO_GROUP once as a module-level constant next to VMI_REGION_GROUPS and reference it in the collectors.

### [LOW] tests — _parse_behavioral_modality_key None/mismatch branches untested
`make_neuronal_tuning_figures.py:242-291`

Module function with three return None paths (tag mismatch 278, no _excit/_suppress 287, no '.' 289) only exercised indirectly via well-formed keys; hyphenated-feature parsing untested.

**Fix:** Add a direct parametrized test over a valid self key, a hyphenated dyadic key, and three None cases (wrong tag, no direction suffix, no '.').

### [LOW] tests — _decide_strip_xscale linear-fallback branches untested
`make_neuronal_tuning_figures.py:294-338`

Returns 'linear' on finite.size==0 (332) and positive.size==0 (335); existing strip tests only drive the symlog path. All-NaN and all-zero/negative inputs are the edge cases this helper exists for.

**Fix:** Add a direct test: all-NaN -> 'linear'; all-zero -> 'linear'; small-range positive -> 'linear'; wide-range positive -> 'symlog'.

### [LOW] tests — __init__ TypeError on unexpected kwargs untested
`make_neuronal_tuning_figures.py:406-409`

Constructor raises TypeError for kwargs not in expected_kwargs (407-409); the only pytest.raises in the test file (2241) covers somatic_filter ValueError. This construction contract is uncovered.

**Fix:** Add `with pytest.raises(TypeError): NeuronalTuningFigureMaker(..., bogus_kwarg=1)` asserting the message names the offending key.

### [LOW] tests — __init__ empty-kslabels ValueError untested
`make_neuronal_tuning_figures.py:418-421`

Empty kslabels raises ValueError (419-421); no test exercises it.

**Fix:** Add `with pytest.raises(ValueError): _maker(kslabels=())`.

### [LOW] tests — make_property_tuning_distribution_figure all-empty x-axis fallback untested
`make_neuronal_tuning_figures.py:3650-3651`

When no unit is consistent, all_vals is empty and the figure falls back to x_lo,x_hi=0.0,1.0 (3650-3651) with every per-region N=0 panel (3698-3701); the fixture always populates the pos archetype so this path never runs.

**Fix:** Add a test with direction='suppress' or impossibly tight tol/k_min asserting a valid PNG is still written.

### [LOW] tests — make_neuronal_tuning_figures early-return / no-tracking-warning paths untested
`make_neuronal_tuning_figures.py:599-606, 668-672`

Three guards untested: tuning dir absent (600-601), dir present but zero pkls (605-606), and the one-time no-tracking-H5 warning (669-672). Full-pipeline tests always supply complete sessions.

**Fix:** Add tests on (a) a root with no ephys/tuning_curves and (b) an empty tuning_curves dir, capturing message_output to assert the skip messages and that nothing is written.

### [LOW] tests — Per-cluster pickle-load and render-failure except branches untested
`make_neuronal_tuning_figures.py:676-682, 721-725`

Two broad except handlers (failed to load 680-682; failed to render 721-725) log+continue; neither is tested. The render-failure handler is the mechanism that previously masked the empty-PdfPages error.

**Fix:** Add a test writing a non-pickle junk *_tuning_curves_data.pkl alongside a valid one; assert a 'failed to load' message via a recording message_output while the valid cluster still renders.

### [LOW] tests — PAG anatomical-gradient singular-KDE (<3 points) fallback untested
`make_neuronal_tuning_figures.py:2469-2492`

_kde_density returns an all-zero grid for <3 points (2489-2490), driving the frac_vmax<=0 (2543-2544) and diff_abs_max<=0 (2550-2551) fallbacks; the fixture supplies enough significant PAG units that this is never hit.

**Fix:** Add a test with PAG units but <3 significant +VMI units asserting make_pag_anatomical_gradient_figure still writes a valid PNG.

### [LOW] tests — Per-region N=0 panel branch untested for confound / distribution figures
`make_neuronal_tuning_figures.py:937-940, 2930-2933`

The N=0 panel paths (confound 937-940, distribution 2930-2933) are never hit because the fixture populates every canonical region; an empty region is a routine real case.

**Fix:** Add a test where at least one VMI_REGION_ORDER group has zero eligible units; assert make_vmi_fr_confound_figure and make_vmi_distribution_figure still write valid PNGs.


## `usv_summary_statistics.py` (26)

### [HIGH] correctness — pls.concat can raise SchemaError on heterogeneous per-session column dtypes
`usv_summary_statistics.py:409-432, 495`

Each session CSV is read with plain pls.read_csv (line 386), so dtypes are inferred per file. The present branch keeps each acoustic feature's inferred dtype (line 410 pls.col(feature)) while the absent branch null-fills as Float64 (line 411); 'start'/'duration' are also carried raw (line 430). If one session infers e.g. max_amplitude/mean_amplitude (or start/duration) as Int64 and another as Float64/Float64-null, the default-vertical pls.concat at line 495 raises a SchemaError and the whole pipeline crashes on a heterogeneous session set.

**Fix:** Cast carried-through numeric columns to a fixed dtype so the schema is identical across sessions, e.g. pls.col(feature).cast(pls.Float64).alias(feature) in the present branch (mirroring the null branch), cast 'start'/'duration' to Float64, or use pls.concat(..., how='vertical_relaxed').

### [HIGH] performance — Nested (category x stage) loop re-filters the full master DataFrame with three chained filters per cell
`usv_summary_statistics.py:3001-3010`

For every (cat, stage) cell three chained .filter() passes scan the full usv_pls: filter(category==cat).filter(estrous_stage==stage).filter(sex==sex_key). sex_key is fixed for the whole function yet re-filtered per cell, giving C*S*3 full scans over potentially hundreds of thousands of rows.

**Fix:** Filter on sex once before the loop and select needed columns; then either combine the per-cell predicate into one filter, or replace the double loop with a single partition_by(['category','estrous_stage'], as_dict=True)/group_by pass indexed by (cat, stage).

### [MEDIUM] correctness — Territorial-boundary contour levels assume contiguous 0..N-1 category IDs
`usv_summary_statistics.py:2160-2161, 2221, 2293`

griddata interpolates the actual category VALUES (cluster IDs, possibly non-contiguous e.g. {0,2,5,11}). Contour levels np.arange(len(unique_cats)+1)-0.5 (lines 2221, 2293) are tied to the COUNT of categories, not their values, so when IDs are not exactly 0..N-1 the thresholds do not lie on boundaries between adjacent category values in Z; boundary lines render at wrong locations or are omitted.

**Fix:** Map category values to contiguous ordinal codes before interpolation (e.g. pd.factorize), interpolate the codes, then arange levels are correct; or compute levels as midpoints between sorted unique values.

### [MEDIUM] correctness — Male/female fatigue heatmaps can have mismatched category rows in a cross-sex comparison figure
`usv_summary_statistics.py:2375-2386, 1698`

Each panel's pivot_df is indexed only by the categories present in that sex's subset (line 2378) and reindexed only on columns (time bins), not the category index. plot_category_local_fatigue_heatmap (line 1698) has the same pattern. If a category appears for one sex but not the other, the panels get different row sets and the same vertical position maps to different categories in a comparison figure.

**Fix:** Reindex both panels onto a common, complete category index (union across both sexes / full sorted set) with fill_value=0.0 so rows align across panels.

### [MEDIUM] tests — build_master_usv_dataframe behavioral-absent branch and empty background_df schema untested
`usv_summary_statistics.py:465-470, 499-508`

Both build_master tests use behavioral sessions, so the no-behavioral null-fill branch (465-470) and the empty-but-typed background_df (499-508) are never exercised; a USV-only session is a real production case.

**Fix:** Add a test with include_behavioral=False sessions asserting usv_df distance/mf_angle/fm_angle are all null, bg_df.height==0, and bg_df.columns/dtypes match the documented schema.

### [MEDIUM] tests — build_master_usv_dataframe partial-skip survival and per-reason summary untested
`usv_summary_statistics.py:474-482`

Existing tests cover only all-skipped RuntimeError. The per-reason skip Counter, the once-only summary print (475-482), and the partial-skip survival path (some skipped, some loaded) are never exercised.

**Fix:** Add a test with one valid + one missing-file session; assert usv_df contains only the valid rows and (via capsys) the printed summary reports the skip count and reason.

### [MEDIUM] tests — plot_polar_kde_distance_angle 'Insufficient Data for KDE' guard untested
`usv_summary_statistics.py:930-933`

The test passes 300/2000 points, exercising only the full path; the <2-points guard at 930-933 (titles 'Insufficient Data for KDE', early return) is never reached.

**Fix:** Add a test with usv arrays of length 0 or 1; assert ax_raw.get_title()=='Insufficient Data for KDE', stats reflects the sub-2 count, and a Figure is returned.

### [LOW] correctness — Log-scale y-limit can be set non-positive when ratios include 0 or negatives
`usv_summary_statistics.py:1909-1911`

global_min/global_max track ALL finite filtered values (1813-1815), not just positive ones. With use_log_scale=True, a 0 or negative ratio makes global_min<=0 and ax.set_ylim(bottom=global_min*0.8, ...) (1911) sets a non-positive lower bound on a log axis, which matplotlib rejects/renders wrong.

**Fix:** When use_log_scale is True, derive global_min/global_max from strictly-positive values only and guard set_ylim so bottom stays > 0.

### [LOW] correctness — sem() on single-element group yields NaN + RuntimeWarning
`usv_summary_statistics.py:1212`

In plot_distance_by_assignment_kde_anova the descriptive loop computes sem(data) whenever len(data) > 0 (1211-1212); a one-sample group returns NaN and emits a RuntimeWarning. Other call sites guard with n > 1 (lines 1829, 1849).

**Fix:** Compute sem only when len(data) > 1, e.g. sem_val = sem(data) if len(data) > 1 else float('nan').

### [LOW] dead_code_naming — merge_usv_and_behavioral_features is referenced only by its own unit test
`usv_summary_statistics.py:194-257`

merge_usv_and_behavioral_features is exported and unit-tested but has no production caller (no .py/.ipynb reference outside the test); build_master_usv_dataframe does the join inline (451-460). Appears superseded, kept alive only by its test.

**Fix:** Confirm whether it is a supported public API; if not, remove it and its dedicated test, otherwise leave as-is.

### [LOW] dead_code_naming — Docstrings on free functions open with "This method"
`usv_summary_statistics.py:168`

Several module-level (non-class) function docstrings open with 'This method ...' (e.g. line 168 get_session_behavioral_features, line 205 merge_usv_and_behavioral_features). These are functions, not methods -- a name/description mismatch.

**Fix:** Replace leading 'This method' with 'This function' (or a concrete verb) in the affected docstrings.

### [LOW] docs_clarity — Typo "Eestus" in section comment
`usv_summary_statistics.py:2675`

The comment reads '# Figure 2: Eestus stage facets'; 'Eestus' is a misspelling of 'Estrous'.

**Fix:** Reword to '# Figure 2: Estrous stage facets'.

### [LOW] docs_clarity — plot_estrous_ratio_scatter Returns block omits documented stats_dict fields
`usv_summary_statistics.py:1788`

Returns collapses to one line while stats_dict has a per-category schema (n, mean, sem, geom_mean, log_sem, ci_lower, ci_upper at 1818-1823/1875-1880), inconsistent with the file's detailed-docstring convention.

**Fix:** Expand Returns to describe stats_dict as a per-category mapping with those keys.

### [LOW] docs_clarity — plot_estrous_stage_pie_chart Returns block omits stats_dict description
`usv_summary_statistics.py:2036`

Returns is just 'fig, ax, stats_dict'; stats_dict (2052-2056) maps full stage names to percentage-of-sessions floats, not inferable from the bare label.

**Fix:** Document each returned item; note stats_dict maps full estrous-stage names to each stage's percentage of total sessions.

### [LOW] docs_clarity — plot_category_prevalence_and_embedding lacks a return annotation
`usv_summary_statistics.py:2082-2091`

Signature ends '):' at line 2091 with no return type annotation, unlike sibling functions (e.g. line 852). The function returns (fig, axes) at line 2307.

**Fix:** Add '-> tuple[plt.Figure, np.ndarray]' and note in Returns that axes is the (4, 2) Axes array.

### [LOW] tests — plot_polar_kde_distance_angle singular-matrix (LinAlgError) fallback untested
`usv_summary_statistics.py:990-992`

The except np.linalg.LinAlgError block (990-992) sets both titles to 'KDE Computation Failed (Singular Matrix)'; no test feeds collinear/identical >=2-point data to trigger it.

**Fix:** Add a test with identical USV coordinates and a spread background; assert the title contains 'Singular Matrix'.

### [LOW] tests — plot_polar_kde_distance_angle empty valid_mask normalization branch untested
`usv_summary_statistics.py:963-968`

The test uses occupancy_threshold=1e-6, leaving valid_mask non-empty; the else branch at 966-968 (plot_norm=norm_dens; vmax_norm=1.0) is never taken.

**Fix:** Add/parametrize a test with occupancy_threshold absurdly high so dens_all never exceeds it; assert a Figure with the normalized axis titled 'Occupancy-Normalized Likelihood'.

### [LOW] tests — plot_estrous_ratio_scatter log-scale path and n<=1/n==0 stages untested
`usv_summary_statistics.py:1855-1918`

Test uses use_log_scale=False and 6 finite values per stage. The geometric-mean/log-SEM rendering (1855-1865), log-axis ylim (1907-1911), single-value (1868-1873), and empty-stage NaN entry (1817-1824) are uncovered; the geometric-vs-arithmetic switch is the main documented behavior.

**Fix:** Parametrize over use_log_scale in (True, False) with a ratio_dict mixing multi-value, single-value, and empty stages; assert NaN sem for n==1, n==0 entry, finite geom_mean under log scale.

### [LOW] tests — plot_animal_participation_stats empty-input branch untested
`usv_summary_statistics.py:796-808`

Empty-input branch (df.empty -> df['vocal_rate'] empty Series at 798-799; stats fallbacks at 805-808) is uncovered; test only passes a 3-animal dict.

**Fix:** Add a test calling the function with {} asserting total_animals==0, mean_session_count==0, mean_vocal_rate==0, and a two-axis Figure.

### [LOW] tests — plot_duration_histograms_by_sex single-sex (empty subset) branch untested
`usv_summary_statistics.py:1357-1382`

Both sex blocks are guarded by 'if not <sex>_durations.empty' (1357, 1374); the test supplies both sexes so the one-sex-empty case is never exercised.

**Fix:** Add a test passing only 'male' rows; assert 'male_mean' in stats and 'female_mean' not in stats with a two-axis Figure.

### [LOW] tests — plot_distance_by_assignment_kde_anova insufficient-samples and ANOVA-not-significant branches untested
`usv_summary_statistics.py:1161-1167, 1205-1206`

Test feeds 40 well-separated samples per group, always taking ANOVA-significant + Tukey. The not-enough-samples path (1165-1167 False), the p>=0.05 branch (1205-1206), and the len==0 'N/A' line (1215-1216) are uncovered.

**Fix:** Add a test with < min_samples_anova rows per category and another with three identically-distributed groups; assert the respective stats keys present/absent.

### [LOW] tests — plot_unassigned_proportion_vs_distance_jointplot fewer-than-2-points branch untested
`usv_summary_statistics.py:1289-1300`

Pearson computation guarded by 'if len(df_combined) >= 2' (1289); test passes 25 rows so the n<2 path is never reached.

**Fix:** Add a test with a 1-row df_combined asserting 'pearson_r' not in stats while the JointGrid is produced.

### [LOW] tests — plot_hourly_regressions empty-sex and <2-valid-point branches untested
`usv_summary_statistics.py:1469-1500`

Test supplies 60 rows per sex; the empty-df guards (1483, 1496) and the annotate_stats valid<2 early-skip (1470) are never taken.

**Fix:** Add a test with only 'male' rows (and/or a single-point sex); assert 'male_r' in stats, 'female_r' not in stats, two-axis Figure.

### [LOW] tests — plot_local_fatigue_binned_trends empty-df, use_log_scale=True, and single-sex branches untested
`usv_summary_statistics.py:1572-1611`

Test passes both sexes with use_log_scale=False. The log-axis branches (1593-1594, 1610-1611), empty male/female guards (1580, 1597), and binned_df.empty stats fallback (1573-1574) are uncovered; use_log_scale has zero coverage.

**Fix:** Parametrize over use_log_scale in (True, False) and add an empty-binned_df test asserting global_max/min==0.

### [LOW] tests — plot_estrous_usv_rates zero-session-count stage branch untested
`usv_summary_statistics.py:1973-1979`

When session_counts for a stage is 0 the function falls back to 0.0 rates (1978-1979); the test supplies positive counts for every stage.

**Fix:** Add a test with one stage's session_counts==0 (present in category_order); assert that stage's male_rate/female_rate are 0.0 with a Figure returned.

### [LOW] tests — plot_estrous_stage_pie_chart missing-stage filtering and total_sessions==0 branches untested
`usv_summary_statistics.py:2043, 2052-2056`

Filter dropping zero/absent stages (2043) and total_sessions==0 fallback to 0.0 (2054) are uncovered; test passes all four stages with positive counts.

**Fix:** Add a test with session_counts missing stages and an all-zero/empty case; assert only present labels appear and proportions are 0.0 respectively.


## `modeling_plots.py` (23)

### [HIGH] correctness — Confusion matrices built without explicit labels can be misaligned/different-shaped before subtraction
`modeling_plots.py:1985-1992`

In plot_univariate_multinomial_performance both 'actual' and 'null' confusion matrices are computed via confusion_matrix(y_t, y_p, normalize='true') with no labels= argument (lines 1986-1989). sklearn infers the label set per call from the union of values present. The null model typically predicts only the majority class, so null_cm can omit categories and be smaller than actual_cm; diff_cm = actual_cm - null_cm (1992) then either raises a broadcasting ValueError or silently subtracts mismatched categories, corrupting the Information Gain heatmap and its class_names-keyed tick labels.

**Fix:** Pass the canonical class order to both calls: confusion_matrix(y_t, y_p, labels=class_names, normalize='true'), so both are K x K with identical row/column ordering.

### [HIGH] docs_clarity — plot_timescale_audit_per_feature docstring describes the WRONG XC-marker algorithm (says 'longest run', code uses earliest-starting run)
`modeling_plots.py:6524-6539`

Description (6524-6535) says the marker is 'the end of the longest sign-consistent outside-null run' and that runs 'compete for longest'. The marker comes from _signal_outer_run_marker, which picks the EARLIEST-STARTING run (5699-5701) and carries a 'Why first run rather than longest run' rationale. Contradicts both the helper and the inline comment at 6778-6789.

**Fix:** Rewrite to describe the earliest-starting run; change 'longest' to 'earliest-starting', drop 'compete for longest', and update the Reading sentence.

### [HIGH] docs_clarity — plot_significant_filters docstring overstates the auto-inverted lower-is-better metric set
`modeling_plots.py:400-402, 465`

Doc (400-402) claims lower-is-better metrics (ll, nll, rmse, mse, loss) are auto-inverted, but line 465 only special-cases metric == 'll'; 'nll'/'rmse'/'mse'/'loss' fall into the else branch (467) treated as higher-is-better, giving incorrect significance calls. Sibling plot_significant_filters_grid uses the full list.

**Fix:** Either narrow the docstring to 'll' only, or fix line 465 to test membership in the full lower-is-better list; doc and code must agree.

### [MEDIUM] correctness — Empty-condition input crashes np.random.choice / np.nanmin in plot_raw_feature_difference
`modeling_plots.py:925-928`

When a condition has zero epochs, all_target_epochs is np.empty((0, 650)) (line 916). target_subset_size = max(1, int(0 * subset_fraction)) = 1 (925), then np.random.choice(0, size=1, replace=False) on 928 raises ValueError. line 923 np.nanmin on the empty array also warns and returns nan. The empty array is a defensive path but downstream sampling does not guard the zero-epoch case.

**Fix:** After concatenation, if all_target_epochs.shape[0] == 0 or all_other_epochs.shape[0] == 0, print a warning and return before the np.nanmin / sampling block.

### [MEDIUM] correctness — plot_model_selection_results left panel mis-renders (and detected metric label is cosmetic) when primary metric is explained_deviance
`modeling_plots.py:1222-1235, 1430-1510`

Detection block (1226-1235) can set primary_metric='explained_deviance', is_minimization=False, metric_label='Explained Deviance'. is_minimization steers feature selection (1244-1262) but the left-panel trajectory is hardcoded for NLL minimization: chance_nll from baseline_score (1349), cum_nlls from prim_mean (1352), delta=prev_nll-cur_nll (1441), bars grow leftward, x-axis inverted (1508) and labeled 'Negative log-likelihood' (1509). With a higher-is-better deviance metric these are all wrong. Separately metric_label is consumed only by the print at 1237 (axis label is the literal at 1509), so the detected label is cosmetic.

**Fix:** Either drop the explained_deviance detection branch and document NLL-only support, or branch the left-panel rendering on is_minimization (baseline, delta sign, bar direction, x-axis inversion, axis label) as plot_multinomial_selection_trajectory does.

### [MEDIUM] docs_clarity — plot_timescale_audit docstring states wrong x-axis range and tick spacing
`modeling_plots.py:6223-6224`

Doc says range [0, ceil(max_horizon / 2) * 2] with ticks every 2 s. Code (6345-6346) uses x_max_int = max(int(np.ceil(x_max_data)), 1) and range(0, x_max_int + 1), i.e. [0, ceil(max_horizon)] with ticks every 1 s.

**Fix:** Update doc to range [0, ceil(max_horizon)]; ticks every 1 s with only first and last labeled.

### [MEDIUM] docs_clarity — plot_model_selection_results docstring claims three right-panel bars but only two are drawn
`modeling_plots.py:1142-1147`

Doc (1142-1145) says 'three vertical bars ... and final + rejected (drawn grey)'. Code defines bar_group_labels = ['best univariate', 'final model'] (1529) and draws only two bars at x=0 (1541) and x=1 (1549); the rejected step is a grey row in the left panel (1461-1477).

**Fix:** Change to 'two vertical bars ...; a rejected final step, if any, appears as a grey row in the left panel, not as a right-panel bar.'

### [MEDIUM] performance — KDE evaluation grid rebuilt inside per-patch loop in plot_spatial_precision_grid (Euclidean branch)
`modeling_plots.py:4576-4577`

xi_grid, yi_grid = np.mgrid[...] and np.vstack([...flatten()]) (a 2x10000 query) are rebuilt every iteration (4576-4577) from loop-invariant bounds (4540-4541); only kde changes per patch. The torus branch (4717-4721) already hoists grid_pts.

**Fix:** Hoist the mgrid build and grid_pts above the for-loop; inside, only evaluate zi_grid = kde(grid_pts).reshape(xi_grid.shape).

### [MEDIUM] tests — _signal_outer_run_marker has no direct unit test despite rich branching
`modeling_plots.py:5573-5702`

The most logic-dense pure helper (sign-consistent runs, idx_floor filtering, min_run_bins drop, earliest-start tie-break, exceeds_window) is unit-tested only indirectly. Its sibling pure helpers ARE unit-tested. Guards at 5664-5667 and the tie-break at 5699-5701 are unverified.

**Fix:** Add a parametrized TestSignalOuterRunMarker covering positive/negative runs, earliest-start-over-longest, sub-floor drop, short-run drop, exceeds-window, and the three guard inputs.

### [LOW] correctness — Null skill distribution permutation runs before the RNG seed in plot_permutation_test
`modeling_plots.py:4117-4124`

shuffled_null_errors = np.random.permutation(null_free_errors) (4117) and null_skill_dist (4118) are computed before np.random.seed(42) on 4124, so the permutation defining the null is not reproducible across runs even though the bootstraps that follow are seeded.

**Fix:** Move np.random.seed(42) above line 4117 so the permutation defining null_skill_dist is reproducible together with the bootstraps.

### [LOW] correctness — Figures leaked when save_plot is False in plot_significant_filters per-feature loop
`modeling_plots.py:528-536`

One figure per significant feature is created (502); plt.close(fig) only runs inside if save_plot (534); with save_plot False every figure stays open after plt.show() (536), accumulating across the loop and triggering matplotlib's >20-figures warning plus memory growth.

**Fix:** Call plt.close(fig) unconditionally after plt.show().

### [LOW] dead_code_naming — Module-level female_cmap / male_cmap are dead
`modeling_plots.py:106-136`

female_cmap (106) and male_cmap (122) are built via create_colormap at import but never referenced in modeling_plots.py or the repo's .py files; sex-coloring uses male_color/female_color.

**Fix:** Delete 106-120 and 122-136; verify whether the create_colormap import at line 64 becomes dead and remove it too.

### [LOW] dead_code_naming — MISSING_AUDIO_COLOR constant is unused
`modeling_plots.py:91`

MISSING_AUDIO_COLOR = '#FF0000' (91) is defined with a comment but never referenced anywhere in the repo.

**Fix:** Remove line 91; leave a TODO if the audio-missing placeholder is planned.

### [LOW] dead_code_naming — Unused display_name unpacked in plot_multinomial_selection_diagnosis
`modeling_plots.py:3024`

Line 3024 unpacks display_name but it is never used in the function; condition is derived from str(selection_results_path).lower() instead. Siblings at 1194/2324/3340 do use display_name.

**Fix:** Replace the unused binding with _: selection_steps, _, selection_metadata = load_selection_results(...).

### [LOW] dead_code_naming — Redundant local TEXT_COLOR reassignment inside plot_feature_ranking loop
`modeling_plots.py:296`

Line 296 re-defines TEXT_COLOR = '#202020', identical to the module-level TEXT_COLOR at line 90; the shadow adds nothing.

**Fix:** Delete line 296.

### [LOW] docs_clarity — _compute_timescale_horizons Returns docstring says 2-tuple but returns a 3-tuple
`modeling_plots.py:6115-6121`

Returns documents '(acf_horizons, xc_horizons)' and says the XC dict 'additionally records ... in xc_exceeds[feature_name]', but the implementation returns (acf_horizons, xc_horizons, xc_exceeds) at 6190 where xc_exceeds is a separate dict.

**Fix:** Document the 3-tuple (acf_horizons, xc_horizons, xc_exceeds) where each is a {feature_name: ...} dict.

### [LOW] docs_clarity — DeepResultsVisualizer.__init__ docstring omits None-handling for visualization_settings
`modeling_plots.py:3957-3959`

modeling_settings doc (3953-3956) documents the None fallback; visualization_settings doc (3957-3959) does not, although the code performs the same None-triggered JSON load at 3998-4006.

**Fix:** Add: 'If None is provided, it loads from the default visualizations_settings.json configuration file.'

### [LOW] docs_clarity — Stray first-person dev comment in plot_feature_importance
`modeling_plots.py:4318`

Line 4318 reads 'This is where I messed up before by ignoring the means/stds keys' - a self-referential note about a past mistake rather than current logic.

**Fix:** Replace with a comment describing current behavior, e.g. 'Pull per-feature means and stds from their dedicated sub-dictionaries.'

### [LOW] tests — KeyError raise for missing shuffled/null key is untested
`modeling_plots.py:225, 442`

plot_feature_ranking (225) and plot_significant_filters (442) raise KeyError when neither 'shuffled' nor 'null' is present; all fixtures write 'shuffled', so the raise and the 'null'-key branch (222-223/439-440) are never exercised.

**Fix:** Add a fixture with only 'actual' asserting pytest.raises(KeyError) for both; add a positive 'null'-key test.

### [LOW] tests — FileNotFoundError early-return branch in plot_significant_filters_grid is untested
`modeling_plots.py:593-598`

plot_significant_filters_grid catches FileNotFoundError (593-598) and returns None, asymmetric with sibling plotters that let open() raise; no test passes a non-existent path.

**Fix:** Add a test passing a non-existent .pkl path and assert it returns None without raising and writes no SVG.

### [LOW] tests — Female-cohort colour branch (_female_ filename token) is never exercised
`modeling_plots.py:200-205, 429-432, 1200-1205`

All fixtures use '_male_' filenames, so the elif '_female_' branches in plot_feature_ranking, plot_significant_filters, plot_model_selection_results and _order_and_color_predictor_features, plus the no-token default, are untested; a swapped self/other colour in the female path would not be caught.

**Fix:** Parametrize an existing smoke test over a '_female_' and a token-less filename; optionally unit-test _order_and_color_predictor_features with both tokens.

### [LOW] tests — _compute_timescale_horizons not unit-tested for missing-key defaults / short-axis / no-marker fallbacks
`modeling_plots.py:6090-6190`

The helper runs only indirectly; the absent-key 0.5/0.2 defaults (6132-6144), the single-lag sig_min_run_bins=1 branch (6151-6152), and the no-marker feature-exclusion contract (6185-6188) are uncovered because the fixture always supplies keys and forces markers.

**Fix:** Add a unit test omitting the two optional keys, including a feature whose ACF never clears null (assert excluded), and a single-lag signal axis (assert no crash).

### [LOW] tests — NaN-fold fallback for best-univariate in multinomial trajectory is untested
`modeling_plots.py:2426-2430`

plot_multinomial_selection_trajectory has a documented all-NaN-fold fallback to steps_data[0] (2426-2430); the fixture always emits finite metrics, so it is never reached.

**Fix:** Add a fixture with an all-NaN secondary metric but finite primary metric on the anchor step and assert the trajectory SVG still emits.


## `make_anatomy_figures.py` (20)

### [HIGH] tests — make_unit_waveform_figure has no end-to-end test, including both RuntimeError paths and the shank_filter branch
`make_anatomy_figures.py:1022-1573`

No test for the public method; only private helpers are unit-tested. Untested: PTP ranking, n_top_units truncation, probe/shank filters, both schematic_side layouts, IBL position->region colour lookup, schematic ghost rows/scale bar, and BOTH RuntimeError guards (1138-1141, 1157-1160). Fixtures _build_ks_probe/_write_waveform_catalog already exist.

**Fix:** Add test_make_unit_waveform_figure_writes_file (build KS probe + catalog + ibl_RH/channel_locations.json, monkeypatch/pass histology_root, assert file written + figures restored), a schematic_side='left' case, and raise tests for both RuntimeError messages plus a shank_filter keep/drop case.

### [MEDIUM] correctness — Caller-supplied histology_root is silently discarded; schematic always reads _DEFAULT_HISTOLOGY_ROOT
`make_anatomy_figures.py:1099-1110, 1345-1348`

make_unit_waveform_figure converts histology_root to a Path (line 1099) then `del histology_root` (line 1110) with a comment claiming it is unused, but the schematic builds the IBL channel_locations.json path from the module constant _DEFAULT_HISTOLOGY_ROOT (line 1346), not the parameter. The only caller (neuronal_tuning_summary.ipynb) does not pass histology_root, so no active bug today, but it is a latent silent-ignore and the comment is misleading.

**Fix:** Either drop histology_root from the signature, or do not delete it and use it at line 1346: `ibl_path = pathlib.Path(histology_root)/str(mouse_id)/str(rec_date)/ibl_dir/'channel_locations.json'`. Reword the 1107-1110 comment to say the parameter is ignored and the schematic reads _DEFAULT_HISTOLOGY_ROOT.

### [MEDIUM] dead_code_naming — _draw_single_unit_waveforms is dead production code (only a test references it)
`make_anatomy_figures.py:1890-2035`

Only src reference is its own def (plus a docstring mention); the figure calls the _in_brain_space variant (line 1238). Invoked solely by test_make_anatomy_figures.py:678.

**Fix:** Remove the method (1890-2035) and its dedicated test, or wire it to a real entry point if retained intentionally.

### [MEDIUM] dead_code_naming — _load_ibl_brain_coords is unused by any production path
`make_anatomy_figures.py:1697-1748`

Referenced only by its own def, a stale docstring mention at 1780, and test:640. No production code calls it; nothing writes the brain_coords ctx key it produces (the figure reads IBL data inline at 1349-1357).

**Fix:** Remove _load_ibl_brain_coords (and its test), or hook it into the pipeline if brain-space plotting returns.

### [MEDIUM] docs_clarity — _draw_single_unit_waveforms_in_brain_space docstring/name are stale: claims IBL brain (AP,DV) space and a brain_coords ctx key it never uses
`make_anatomy_figures.py:1764-1802`

Description says waveforms sit at IBL brain (AP,DV); ctx Parameters demands brain_coords; Returns calls outputs 'Brain AP and DV centres'. None is true: body reads only ctx['channel_positions'] (probe-local), comment at 1849 says 'Use raw probe-local lateral', returns (plot_lat, plot_axi). brain_coords is never read/set. lateral_offset_um (actually used) is undocumented; waveform_voltage_uv_scale wrongly says 'DV um'.

**Fix:** Rewrite Description to probe-local (lateral, axial) + lateral_offset_um shift; delete the brain_coords line; document lateral_offset_um; change Returns to probe-local lateral/axial; reword voltage scale to 'axial um'; consider renaming away from 'in_brain_space'.

### [MEDIUM] docs_clarity — make_unit_waveform_figure docstring omits ~9 parameters and orders some out of signature order
`make_anatomy_figures.py:1062-1091`

Signature accepts probe_to_hemisphere, histology_root, probe_filter, shank_filter, lateral_jitter_um, inter_shank_wspace, schematic_side, ap_padding_um, dv_padding_um (plus documented ones), but the Parameters block documents only a subset and orders out_dir/mouse_id/session_id ahead of the signature order.

**Fix:** Add Parameters entries for every kwarg (notably probe_filter, shank_filter, schematic_side) and order them to match the signature.

### [MEDIUM] performance — Allen meshes parsed twice per figure; _load_obj_mesh has no memoization
`make_anatomy_figures.py:782-791, 2073, 2135`

_compute_bucket_bboxes (782) parses PAG/MRN/VTA/MB/CENT/SCm/SCs via _load_obj_mesh (2073); the loop at 783-791 then re-parses each via _add_mesh_to_axes (2135). Pure-Python OBJ scan, so every bucket mesh is read+parsed twice per build; video compounds it on the first frame.

**Fix:** functools.lru_cache on _load_obj_mesh keyed by Path, or share parsed (verts, faces) between _compute_bucket_bboxes and _add_mesh_to_axes.

### [MEDIUM] performance — channel_shanks.npy re-loaded from disk many times per waveform figure
`make_anatomy_figures.py:1153, 1176, 1191, 1224, 1298, 1808`

make_unit_waveform_figure and its draw helper re-np.load channel_shanks.npy for the same probe at 1153 (per-unit in shank_filter), 1176 (per-unit), 1191, 1224, 1298, and per draw call 1808. All units on a probe share the array.

**Fix:** Load once per probe into ctx['channel_shanks'] in _gather_probe_context_for_unit and reference it everywhere instead of re-np.load-ing.

### [MEDIUM] tests — make_unit_positions_video and its azimuth update() callback untested
`make_anatomy_figures.py:894-1018`

No coverage for resolve_pdf_path->.with_suffix override (1001-1007), gif vs ffmpeg branch (1008-1011), or the denom=max(1,n_frames-1) seam math (988-989). The PillowWriter gif path runs in-process.

**Fix:** Add test_make_unit_positions_video_gif (monkeypatch _download_allen_mesh to cube OBJ, video_format='gif', small n_frames, assert .gif written) and assert no ZeroDivisionError at n_frames=1.

### [LOW] correctness — hemisphere derivation forces 'L' when probe_filter is None (units may span both probes)
`make_anatomy_figures.py:1343`

Line 1343 `hemisphere = 'R' if str(probe_filter).endswith('0') else 'L'`. probe_filter=None is valid (keeps both probes), but str(None) does not end with '0', so hemisphere is forced to 'L' and a single ibl_LH file colours a mixed-hemisphere schematic. Default probe_filter='imec0' so only the None path bites.

**Fix:** When probe_filter is None, derive the probe/hemisphere from the rendered units (e.g. top_units[0]['probe']), or assert probe_filter is not None before building the single-probe schematic.

### [LOW] correctness — cluster_num regex cl(\d{4}) misparses cluster ids longer than 4 digits
`make_anatomy_figures.py:1611`

_collect_session_clusters uses `str.extract(r'cl(\d{4})').astype(int)`. unit_id is formatted with `:04d` (spike_quality_metrics.py:1134), a MINIMUM of 4 digits, so a cluster id >9999 yields 5 digits and the regex silently captures only the first 4. Safe today, fragile coupling.

**Fix:** Use a delimiter-anchored width-agnostic pattern: `str.extract(r'_cl(\d+)_')`.

### [LOW] correctness — max(totals) raises ValueError when the catalog/pivot is empty
`make_anatomy_figures.py:489, 560`

Both yield panels call `ax.set_ylim(0, max(totals)*1.12)` with totals from pivot.sum(); an empty catalog (after _EXCLUDED_MOUSE_IDS filtering) makes totals empty and builtin max() raises 'max() arg is an empty sequence'.

**Fix:** Guard: `top = totals.max() if totals.size else 1` then set_ylim(0, top*1.12) in both panels.

### [LOW] dead_code_naming — _draw_single_unit_waveforms docstring omits peakch and cluster_num
`make_anatomy_figures.py:1918-1947`

Signature declares cluster_num (1895) and peakch (1897) but the Parameters block omits both, while the twin _in_brain_space documents them. Lower priority because the method is dead code.

**Fix:** Add cluster_num (int) and peakch (int) to the Parameters block, or moot if the method is removed.

### [LOW] dead_code_naming — pos_to_region.get(...) uses .get() implicit-None default, against project convention
`make_anatomy_figures.py:1358-1365`

contact_colours relies on dict.get() returning None for unmapped contacts, then passes None into pool_brain_area (str | None). Project convention forbids .get() with implicit defaults.

**Fix:** Use explicit membership: `region = pos_to_region[key] if key in pos_to_region else None`.

### [LOW] docs_clarity — Stale n_frames example (180/6 s) contradicts default of 240 (8 s)
`make_anatomy_figures.py:931`

n_frames defaults to 240 (line 901); the Parameters note says 'With fps=30 and n_frames=180 the video is 6 s long.'

**Fix:** Update to 'With fps=30 and n_frames=240 the video is 8 s long.'

### [LOW] docs_clarity — build_unit_positions_figure docstring omits rasterize_dense
`make_anatomy_figures.py:712-744`

rasterize_dense (line 676) is used at 770/790/855 but the Parameters block stops at filter_outliers (744). The sibling make_unit_positions_figure documents it.

**Fix:** Add a rasterize_dense Parameters entry (may cross-reference the fuller note).

### [LOW] docs_clarity — n_top_units documented (int) but is int|None with undocumented None-means-all
`make_anatomy_figures.py:1074-1075`

Parameters says 'n_top_units (int)'; signature is int|None=None (1032) and body treats None as render-all (1143-1145).

**Fix:** Change to 'n_top_units (int | None)' and add 'None renders every ranked SU-somatic cluster.'

### [LOW] performance — Two full .iterrows() passes over the same probe_clusters DataFrame
`make_anatomy_figures.py:1652-1659`

cluster_to_bucket (1652-1655) and cluster_to_peakch (1656-1659) are built with two separate iterrows() loops; iterrows boxes each row, doubling the cost when one vectorized pass suffices.

**Fix:** Build both dicts from to_numpy() arrays via dict(zip(...)).

### [LOW] performance — Row-wise df.apply for cell_type derivation instead of vectorized assignment
`make_anatomy_figures.py:337`

_load_catalog uses df.apply(self._classify_cell_type, axis=1), a per-row Python callback building a pd.Series, where np.where on cluster_group/somatic masks is trivially vectorizable.

**Fix:** Vectorize with np.where; keep _classify_cell_type if other callers/tests need it.

### [LOW] tests — _download_allen_mesh download branch untested (only cache-hit covered)
`make_anatomy_figures.py:158-162`

Only the path.exists() early-return is tested; the urlopen/read/write_bytes branch and URL templating are unverified.

**Fix:** Add test_download_allen_mesh_downloads_when_absent monkeypatching _ALLEN_MESH_CACHE and urlopen, asserting the cube bytes round-trip and URL contains '795.obj'.


## `make_behavioral_videos.py` (18)

### [HIGH] correctness — Spike-sound slice assignment can broadcast-error for spikes near window end
`make_behavioral_videos.py:1248`

new_spike_sound_array[sound_start:sound_start + spike_sound.shape[0]] = spike_sound writes the full spike.wav waveform starting at sound_start into an array sized to the whole video duration. For a spike whose sound_start is within spike_sound.shape[0] samples of the array end, the LHS slice is shorter than spike_sound and numpy raises ValueError: could not broadcast input array from shape (N,) into shape (M,), aborting spike-sound generation (and the run).

**Fix:** Clip the write length: end = min(sound_start + spike_sound.shape[0], new_spike_sound_array.shape[0]); new_spike_sound_array[sound_start:end] = spike_sound[:end - sound_start].

### [MEDIUM] correctness — find_region_by_channel None return breaks eventplot colors / tuple unpack
`make_behavioral_videos.py:1950`

find_region_by_channel returns None (line 338) when a cluster channel falls outside every labelled range. At line 1950 (no special units) that None is appended into event_plot_colors, later passed to ax.eventplot(colors=...) which rejects an invalid color. At line 1954 the result is unpacked as special_brain_region, special_color which raises TypeError on None. The pre-filter at 1588 only runs when brain_areas criteria is set and never excludes None-region clusters.

**Fix:** Make find_region_by_channel fall back to the 'other' bucket region/color instead of returning None, or guard both call sites for None before appending/unpacking.

### [MEDIUM] correctness — Speaker-path spectrogram window can go negative and silently wrap the audio slice
`make_behavioral_videos.py:1734`

The in-range guard at line 1704 validates frame_start - half_window_size_frames against 0 in camera frames, before TTL correction. In the speaker branch window_start_signal is then reduced by ttl_start + 20000 (line 1734) with no re-check. If window_start_signal goes negative, speaker_audio_data[window_start_signal:window_end_signal] (line 1736) uses numpy negative-index semantics and yields an empty/wrong segment with no warning, producing a spectrogram over the wrong audio.

**Fix:** After the TTL correction, clamp/validate window_start_signal >= 0 (and window_end_signal <= len); emit a message and return early if the corrected window falls outside speaker audio bounds.

### [MEDIUM] performance — plot_arena_corners_mics re-runs O(n) list .index() lookups for static arena vertices every frame
`make_behavioral_videos.py:1112`

plot_arena_corners_mics is called once per frame (lines 1802, 2059). Inside it arena_node_names.index('North'/'West'/'South'/'East') and index(arena_nc[...]) / index(wall[...]) are evaluated dozens of times per call (lines 1112-1175), each an O(n) scan, while the arena geometry is frame-invariant (always data[0,0,...]).

**Fix:** Resolve node indices once (node_idx = {name: arena_node_names.index(name) ...}) or precompute the corner coordinates a single time before the FuncAnimation loop, and reuse them.

### [MEDIUM] performance — plot_mouse_data recomputes animal_node_names.index(...) in per-mouse/per-node loops every frame
`make_behavioral_videos.py:471`

plot_mouse_data runs once per frame (lines 1833, 2089). It calls animal_node_names.index(...) repeatedly: history loop ~6x per iteration (472-474), node-connection loop 6x per connection (481-483), polygon loop per vertex (491-493); node_connections/node_polygons are re-split via str.split('-') (480, 488) every frame. The mapping never changes.

**Fix:** Build a name->index dict once and pre-split node_connections/node_polygons into index tuples in __init__, then index data with cached integers.

### [LOW] correctness — beh_window_size_sec uses integer //2, mismatching the frame half-window for odd window sizes
`make_behavioral_videos.py:2010`

beh_window_size_sec is passed as beh_features_window_size // 2 (line 2010) and used as the +/- x-axis tick label (line 962), while the plotted-data half-window (line 1664) uses beh_features_window_size / 2. For an odd window size (e.g. 5) the label reads 2 but the data half-window is 2.5 s, so the axis labels misrepresent the actual span.

**Fix:** Pass beh_features_window_size / 2 (floating division) for the label so the displayed tick equals the real half-window.

### [LOW] correctness — Initial-frame USV unassigned-color default differs from animate callback and bypasses palette
`make_behavioral_videos.py:1880`

On the initial static draw, unassigned-emitter USV segments default to a hardcoded '#FFFFFF' (line 1880), while the animate callback defaults to self.visualizations_parameter_dict['unassigned_colors'][0] (line 2139). For a static save unassigned segments render white instead of the configured color, and frame-0 is inconsistent with subsequent frames; the hardcoded white bypasses the configured unassigned-color palette.

**Fix:** Use self.visualizations_parameter_dict['unassigned_colors'][0] in the line-1880 list comprehension, matching line 2139.

### [LOW] correctness — Arena corner markers use named matplotlib colors, not hex strings
`make_behavioral_videos.py:1112`

plot_arena_corners_mics draws the four corner scatters with named colors c='red'/'yellow'/'green'/'blue' (lines 1112-1115), violating the project hex-only color convention the rest of the file follows.

**Fix:** Replace with hex strings (e.g. '#FF0000','#FFFF00','#008000','#0000FF') or palette constants.

### [LOW] dead_code_naming — Dead parameter freq_yticks in plot_spectrogram (never read; body explicitly clears y-ticks)
`make_behavioral_videos.py:555`

plot_spectrogram declares freq_yticks (line 555, documented 594) and both call sites pass spectrogram_yticks (lines 1903, 2156), but the body never reads it and instead clears all y-ticks via set_yticks([]) at line 646. The whole spectrogram_yticks -> freq_yticks chain is dead.

**Fix:** Either wire freq_yticks through (replace set_yticks([]) with set_yticks(freq_yticks) plus labels), or remove the freq_yticks parameter/docstring and the kwargs at 1903/2156 (and the orphaned spectrogram_yticks setting/CLI option).

### [LOW] docs_clarity — beh_window_size_sec docstring says 'Window size in seconds' but value is the HALF window
`make_behavioral_videos.py:836`

plot_behavioral_features documents beh_window_size_sec as 'Window size in seconds.' (lines 836-837), but both call sites pass beh_features_window_size // 2 and the body uses it as the per-side +/- x-axis bound (line 962). The parameter is the half window in seconds, contradicting the docstring and the correctly-named sibling half_window_size_sec.

**Fix:** Reword to 'Half window size in seconds (per-side x-axis bound; equals beh_features_window_size / 2).'

### [LOW] docs_clarity — active_mic_position docstring claims 'defaults to 0' but the parameter has no default
`make_behavioral_videos.py:1074`

plot_arena_corners_mics docstring (lines 1074-1075) states active_mic_position 'defaults to 0', but the signature (line 1012) declares it without a default. The default-of-0 behavior lives in the caller, not this function.

**Fix:** Drop 'defaults to 0' (the parameter is required), or note the caller supplies 0 when no spectrogram channel is selected.

### [LOW] docs_clarity — NVIDIA/h264_nvenc comment and message overstate what the codec branch does
`make_behavioral_videos.py:2251`

The comment 'create a custom writer for NVIDIA GPU acceleration' (line 2251) and message 'Using GPU (h264_nvenc) for video encoding...' (line 2265) hardcode NVENC, but the branch is entered for any non-None animation_codec and uses whatever codec/preset/tune is configured. For a non-NVENC codec these are factually wrong and can mislead encoding-failure debugging.

**Fix:** Generalize to reference the configured codec, e.g. f"Using configured codec ({animation_codec}) for video encoding...".

### [LOW] performance — plot_behavioral_features recomputes columns.index(feature_name) twice per feature every frame
`make_behavioral_videos.py:935`

plot_behavioral_features is called every frame (lines 2003, 2193). For each feature it evaluates beh_feature_data.columns.index(feature_name) twice (lines 935, 940), an O(n_columns) scan recomputed per frame though column order is fixed.

**Fix:** Compute feature_name -> column-index once and reuse it instead of calling columns.index twice per feature per frame.

### [LOW] performance — @njit(parallel=True) on read_ttl_events and filter_spikes_for_raster yields no parallelism
`make_behavioral_videos.py:162`

read_ttl_events (line 162) and filter_spikes_for_raster (line 186) are decorated @njit(parallel=True) but contain only vectorized numpy ops (np.diff/np.where/boolean masking) with no prange, so the parallel pass finds nothing to parallelize, adding compile overhead and a 'no transformation for parallel execution possible' warning with zero speedup.

**Fix:** Drop parallel=True (use plain @njit or @njit(cache=True)) on both functions.

### [LOW] tests — pool_brain_area has no direct test (CENT*/SC* prefix and None->'other' branches uncovered)
`make_behavioral_videos.py:216`

pool_brain_area (lines 216-254) is never imported or called directly in tests/ (grep confirms). Its CENT* and SC* prefix-collapsing and the None/empty->'other' path are never asserted; only indirect calls via find_region_by_channel exercise the exact-match branch.

**Fix:** Add a parametrized test asserting pool_brain_area(None)=='other', ('')=='other', ('PAG')=='PAG' (+MRN/VTA/MB), ('SCdw')=='SC', ('CENT2')=='CENT', ('VISp')=='other'.

### [LOW] tests — _resolve_brain_area_color fallback-to-'other' branch untested
`make_behavioral_videos.py:257`

_resolve_brain_area_color (lines 257-287) is never imported in tests/. The fallback path (bucket absent from scheme -> scheme['other']), documented as the KeyError guard, is never asserted.

**Fix:** Add a test asserting _resolve_brain_area_color('SCdw', {'other':'#B8B8B8'}) == '#B8B8B8' and that a present bucket returns its own color.

### [LOW] tests — Create3DVideo.__init__ unexpected-keyword-argument TypeError is untested
`make_behavioral_videos.py:1296`

__init__ raises TypeError for unexpected kwargs (lines 1296-1299), the same defensive pattern explicitly tested for other classes (test_inference.py:250, test_iui_calculator.py:67, test_analyze.py:2630), but no such test exists for Create3DVideo.

**Fix:** Add with pytest.raises(TypeError, match='unexpected keyword argument'): Create3DVideo(..., bogus_kwarg=1).

### [LOW] tests — beh_features and spectrogram 'too early/too late' early-return guards untested
`make_behavioral_videos.py:1665`

The beh_features guard (lines 1665-1667) and spectrogram guard (lines 1704-1706) emit '...too early or too late...' and return early; grep confirms neither message is asserted in tests/. A regression inverting the bound or dropping the early return would not be caught.

**Fix:** Add integration tests with beh_features_bool/spectrogram_bool True and a video_start_time so frame_start - half_window < 0, asserting the message and that no save occurs.


## `qlvm_torus_traversal_video.py` (18)

### [MEDIUM] docs_clarity — On-screen peak-walk title hardcodes 'Red trail' but the trail color is the user-tunable accent_color
`qlvm_torus_traversal_video.py:777`

The peak-walk title rendered into the video reads 'Red trail = path on torus' (line 777). The trail, head marker and cluster outline are all drawn with cfg['accent_color'] (trail_cmap built from accent_color at lines 442-447; markers at 655/659; contour at 638). The default test config sets accent_color='#00FFFF' (cyan), so even the default render mislabels the trail; any non-red accent makes the caption wrong. A user-visible wording bug in the output video.

**Fix:** Replace the hardcoded 'Red trail' with color-neutral wording, e.g. 'Accent trail = path on torus  -  right grid fills as samples are visited'.

### [MEDIUM] performance — Per-frame O(n^2) trajectory-trail recomputation in update()
`qlvm_torus_traversal_video.py:804-807`

update() rebuilds the trail every frame: traj[:fi+1] % 1.0 then _fading_trail_segments re-stacks/re-diffs/re-masks the whole prefix; total work across n frames is O(n^2). Redundant recompute that grows with frame index, though the per-frame full figure draw is likely the larger cost.

**Fix:** Precompute per phase the full traj%1.0, segment array, recency linspace and wrap keep mask; in update() slice up to fi and call set_segments/set_array, turning the traversal from O(n^2) to O(n).

### [MEDIUM] tests — pool_latents_from_h5 empty/all-NaN early return untested
`qlvm_torus_traversal_video.py:141-142`

The `if not coords_chunks: return np.empty((0,2), dtype=np.float64), index` branch (no qlvm_dim anywhere, or all-NaN) is never exercised; test covers only the partial-NaN happy path. This empty return is the precondition make_video relies on to raise its ValueError.

**Fix:** Add a test with (a) a session with only spectrograms and (b) a session whose qlvm_dim is all-NaN, asserting coords.shape==(0,2), dtype float64, index==[].

### [MEDIUM] tests — torus_forward not tested for multi-row batching / wrap-around invariance
`qlvm_torus_traversal_video.py:86-103`

Test feeds only one (1,2) row; wrap-invariance (the property the NN index depends on) and N>1 batching are unasserted. Verified torus_forward(zeros((3,2))).shape==(3,4) and allclose([[0.1,0.9]] vs [[1.1,-0.1]]).

**Fix:** Assert torus_forward(zeros((3,2))).shape==(3,4) and np.allclose(torus_forward([[0.1,0.9]]), torus_forward([[1.1,-0.1]])).

### [MEDIUM] tests — _shortest_extended (torus shortest-path) has no test
`qlvm_torus_traversal_video.py:186-191`

Geometry primitive for Part 2 shortest-path walks; verified [0.1],[0.9] -> endpoint [-0.1]. Completely untested.

**Fix:** Assert second point ~[-0.1] for a=[0.1],b=[0.9] and b unchanged for a non-wrapping pair.

### [MEDIUM] tests — _curved_path untested, including the degenerate a==b guard
`qlvm_torus_traversal_video.py:171-183`

Builds Part 3 boundary walks; verified the norm<1e-12 guard returns the constant point with no NaN. Untested for both branches.

**Fix:** Test endpoint preservation + perpendicular midpoint displacement, and the degenerate a==b finite-output case.

### [MEDIUM] tests — _fading_trail_segments wrap-discontinuity and <2-point branches untested
`qlvm_torus_traversal_video.py:231-245`

Drops segments jumping > wrap_thresh and short-circuits <2 points; verified single-point returns (0,2,2)/(0,). Core visual helper, zero direct coverage.

**Fix:** Test single-point empty output, a smooth 4-point path's increasing recency, and a >0.5-jump segment being dropped.

### [MEDIUM] tests — .mp4 / FFMpegWriter branch and default output-path derivation untested
`qlvm_torus_traversal_video.py:864-867`

Only .gif outputs are tested and the CLI test mocks the class, so the FFMpegWriter branch (864-867) and the output_path=None default-path derivation (458-462, reads figures.save_directory which _tiny_cfg lacks) are untested.

**Fix:** Add a test with output_path=None and a figures.save_directory tmp dir, mocking FFMpegWriter and ani.save, asserting a qlvm_torus_traversal_*.mp4 path and FFMpegWriter selection.

### [MEDIUM] tests — apply_mask=False, durations-absent, and no-mask-match fallbacks in _get_spec untested
`qlvm_torus_traversal_video.py:514-527`

_get_spec branches not asserted: apply_mask=False, 'durations' not in grp (510-511), and matching.size==0 (519). _write_inputs always provides durations and a full 1:1 mask index, so these never run.

**Fix:** Add a focused test with apply_mask=False, a store lacking durations, and a mask omitting some rows.

### [LOW] correctness — Boundary draw loop missing s < grid_ncols guard (column overflow if boundary_positions_per_walk > 15)
`qlvm_torus_traversal_video.py:824-841`

In the boundary reveal draw loop, s is used directly as the grid COLUMN (slot = row*grid_ncols + s, line 830). The only guard is `if s >= boundary_positions_per_walk: break` (line 825); there is no `s < grid_ncols` guard. The grid has grid_ncols=15 columns, so if boundary_positions_per_walk > 15, s exceeds 14 and slot spills into the next row, corrupting the column=position / row=neighbor layout (the line 831 `slot >= n_grid_slots: continue` only prevents an index error, not the corruption). The un-highlight path at line 822 DOES guard `prev_s < grid_ncols`, confirming 15 is the intended column cap. Config-bounded, but a real silent layout-corruption mode.

**Fix:** Add `if s >= grid_ncols: break` alongside the existing check at line 825, or clamp boundary_positions_per_walk = min(boundary_positions_per_walk, grid_ncols) when reading the config.

### [LOW] dead_code_naming — state['reached_part'] is written but never read (dead field)
`qlvm_torus_traversal_video.py:725, 739`

state['reached_part'] is initialized at line 725 and updated at line 739, but never read; grep confirms it appears only at those two lines. Vestigial with no rendering effect.

**Fix:** Remove the 'reached_part': 0 entry at line 725 and delete the line 739 assignment.

### [LOW] docs_clarity — Stale 'cyan' wording in docstring and comments for user-tunable accent_color artists
`qlvm_torus_traversal_video.py:22-24, 632, 650-651, 657, 794-795, 808-810`

The module docstring (lines 22-24, 31-34), the show_cluster_contour docstring (line 632) and inline comments at 650-651, 657, 794-795, 808-810 all hardcode 'cyan' for artists actually painted with the user-tunable accent_color (settings comment 435-437; contour 638; markers 655/659; trail cmap 442-447). Stale and misleading.

**Fix:** Replace 'cyan' with 'accent-colored' in the docstring (22-24, 31-34) and comments (632, 650-651, 657, 794-795, 808-810).

### [LOW] docs_clarity — make_video Returns section describes a side effect, not the actual return (None)
`qlvm_torus_traversal_video.py:404-405`

make_video is annotated `-> None` and returns nothing, but its Returns section reads 'A video file at output_path.' Per the repo Description/Parameters/Returns convention Returns should state None.

**Fix:** Change Returns to 'None' and move the side-effect description into the Description.

### [LOW] docs_clarity — Ungrammatical 'The arrays ``.npz`` is used' in module docstring
`qlvm_torus_traversal_video.py:18-19`

Subject/verb disagreement: 'The arrays ``.npz`` is used only for the heatmap background...' reads as a missing word.

**Fix:** Reword to 'The ``.npz`` arrays file is used only for the heatmap background, the ws_labels_periodic contours, and the cluster centers.'

### [LOW] docs_clarity — boundary_row_nn magic list [2,1,0,3,4] has no explanatory comment
`qlvm_torus_traversal_video.py:555-556`

boundary_row_nn = [2,1,0,3,4] maps the 5 grid rows to NN ranks, placing the peak (rank 0) in the middle row; boundary_mid_row is the row holding rank 0. Non-obvious and load-bearing for tile selection at 823/833/838, with no comment.

**Fix:** Add a one-line comment explaining the row->NN-rank mapping and that boundary_mid_row is the accent-bordered middle row.

### [LOW] tests — _smooth_jitter endpoint-pinning and degenerate guards untested
`qlvm_torus_traversal_video.py:158-168`

Returns zeros for n<=2 or sigma<=0 and pins perturbation to zero at endpoints; verified zeros for (2,0.1) and sum 0 for (10,0.0). Untested.

**Fix:** Assert zeros for (2,0.1) and (10,0.0); for n=20,sigma=0.05 endpoints ~0 and interior nonzero.

### [LOW] tests — build_phases peak-pair de-collision (divmod / j>=i bump) untested
`qlvm_torus_traversal_video.py:281-288`

divmod(p,K-1) then j+=1 when j>=i skips the i==j diagonal; test only counts phases, never asserts pairs are valid or the min(5,K*(K-1)) cap for small K.

**Fix:** Assert every 'Peak i -> Peak j' has i != j and in range; add a K=2 case asserting 2 peak traversals.

### [LOW] tests — precompute_reveals never tested directly
`qlvm_torus_traversal_video.py:323-348`

Only exercised implicitly in the GIF render; the boundary-vs-peak branch split (335 vs 342) and exact reveal shapes are never asserted.

**Fix:** Unit test with synthetic peak/boundary phases and a stub nn_query, asserting added keys, shapes, and reveal_frames span.


## `make_usv_spectrograms.py` (17)

### [MEDIUM] correctness — plot_embedding docstring claims masks are flipped before applying, but neither this code nor the production feature path flips
`make_usv_spectrograms.py:3139-3142`

The docstring states the function 'flips each mask along the frequency axis before applying it.' No np.flipud occurs in either masked path (894-900, 3797-3808). The production acoustic-feature path (compute_usv_acoustic_features.build_mask_region_masks:144 + compute_acoustic_features:206) ALSO applies the stored segmentations to specs WITHOUT a flip and indexes freq_axis directly against spec rows; those features generate the embeddings the visualization displays. So applying the mask un-flipped is CONSISTENT with production. The candidate's proposed fix (add np.flipud) would corrupt the output; the real defect is only the stale docstring.

**Fix:** Delete or rewrite the flip sentence to state masks are applied directly (no frequency flip), matching build_mask_region_masks/compute_acoustic_features.

### [MEDIUM] correctness — plot_all_channels builds time_vec from end_signal-start_signal while data_slice is numpy-clamped, raising on out-of-range windows
`make_usv_spectrograms.py:758-766`

_resolve_window (358-359) does not clamp end_signal to sample_num. plot_all_channels time_vec uses num=end_signal-start_signal (761) but data_slice (766) is numpy-clamped to sample_num-start_signal. An out-of-range time_window end makes time_vec longer than data_slice, so _render_raw_audio's ax.plot raises 'x and y must have same first dimension'. plot_single_channel (675) is robust; the two modes diverge.

**Fix:** Clamp end_signal=min(round(end_time_sec*sampling_rate), sample_num) (and start_signal=max(0,...)) in _resolve_window, or derive time_vec from data_slice.shape[0] in plot_all_channels.

### [MEDIUM] dead_code_naming — _medoid_xy is dead in production and contradicts the spiral docstring (which claims medoid but code uses arithmetic centroid)
`make_usv_spectrograms.py:2814`

_medoid_xy (2814) is never called in src/ (only its def + tests). Both spiral origins use np.mean: fallback (2967-2968) and main path (3430-3431). The 'spiral' docstring (2893-2899) claims expansion 'from the cluster's medoid (geometric-median ... via Weiszfeld)' and explicitly contrasts it with 'the arithmetic centroid' -- directly contradicted by the np.mean implementation.

**Fix:** Either wire _medoid_xy into the spiral origins (2967-2968 and 3430-3431) to match the docstring, or remove _medoid_xy + its tests and change the 2893-2899 docstring to say 'arithmetic centroid'. (Merges candidate findings #4 and #5.)

### [MEDIUM] docs_clarity — Class docstring lists a nonexistent 'variance-weighted average' rendering mode
`make_usv_spectrograms.py:152`

USVSpectrogramPlotter docstring says it renders 'single-channel, all-channel and variance-weighted average' spectrograms. No variance-weighted-average path exists; the dispatch (1493-1507) is single/all/stitched/sequence.

**Fix:** Replace with the four real modes: single-channel, all-channel, stitched session-timeline, and sequence.

### [MEDIUM] docs_clarity — Module docstring says 'Three rendering modes' but omits plot_sequence (a fourth)
`make_usv_spectrograms.py:5`

Module docstring enumerates plot_single_channel, plot_all_channels, plot_stitched as 'Three rendering modes'; plot_sequence is a fourth dispatched at mode=='sequence' (1500-1501).

**Fix:** Change 'Three' to 'Four' and add a (4) entry for plot_sequence (embedding landscape left, stitched session-timeline spectrogram right).

### [MEDIUM] docs_clarity — _render_raw_audio docstring describes min/max auto-scaling but code uses symmetric +/- peak limits
`make_usv_spectrograms.py:405-407`

Docstring claims y-limits are 'auto-scaled to the actual min/max ... (the floor and ceiling of the window's amplitude range)'. Code (434-440) sets symmetric_limit=abs_peak*1.05 and ylim/yticks to (-symmetric_limit, +symmetric_limit), symmetric about zero.

**Fix:** Reword: 'Y-limits are set symmetrically about zero at +/- 1.05x the window's absolute peak amplitude, with the two y-tick labels at -peak and +peak.'

### [MEDIUM] docs_clarity — build_pooled_embeddings_df Returns schema omits the sex/duration/feature columns the function emits
`make_usv_spectrograms.py:2336-2342`

Returns Schema lists only session_id, row_index, four coords, four labels. The function emits 'sex' (Utf8) and 'duration' (Float64) plus EMBEDDING_FEATURE_COLS (Float64) with null-filled fallbacks (2478-2511); these are in required_cols for cache validation (2354-2359).

**Fix:** Extend the schema list to include sex (Utf8), duration (Float64), and the six acoustic-feature columns (Float64).

### [MEDIUM] docs_clarity — _knn_boundary_grid docstring is missing its Parameters section
`make_usv_spectrograms.py:3017-3040`

_knn_boundary_grid takes 11 parameters but the docstring goes Description -> Returns with no Parameters block, breaking the file's Description/Parameters/Returns convention. density_min_count, density_smoothing_sigma, and the grid extent are only partially explained inline.

**Fix:** Add a Parameters section for all 11 args (x, y, labels, x_lo/x_hi/y_lo/y_hi, n_neighbors, grid_resolution, density_smoothing_sigma, density_min_count).

### [LOW] correctness — Divide-by-zero in _pick_spiral_with_grid grid mapping when the embedding x- or y-extent is degenerate
`make_usv_spectrograms.py:2767-2768`

px/py = (xs_d-gx_lo)/(gx_hi-gx_lo)*(res-1). gx_lo/gx_hi=xx[0]/xx[-1], a linspace over x_lo/x_hi=min/max of all points (3371-3372). A degenerate cohort (collinear/single-point) gives gx_hi-gx_lo==0 -> inf -> undefined .astype(int); inside_box does not protect the division. The caller already guards padding via (x_hi-x_lo or 1.0) at 3373 but not the grid mapping.

**Fix:** den_x=(gx_hi-gx_lo) or 1.0 (and den_y), divide by those; or skip the grid filter when a span is 0 and fall back to the unfiltered spiral at 2785.

### [LOW] docs_clarity — _save_figure suffix doc gives a stale 'avg' example
`make_usv_spectrograms.py:577`

Suffix doc example includes 'avg', a leftover from the nonexistent variance-weighted-average mode. Real suffixes: 'ch{NN}' (715), 'all_channels' (799), 'stitched' (1116), 'sequence_{embedding}' (1459).

**Fix:** Replace 'avg' with a real example, e.g. 'stitched' or 'sequence_qlvm'.

### [LOW] docs_clarity — make_usv_spectrograms docstring describes already-done pipeline wiring as pending
`make_usv_spectrograms.py:1470-1473`

Docstring says it is 'expected to call once a make_usv_spectrograms_bool toggle has been wired through.' The toggle is wired: visualize_data.py:133 checks it and calls the method; GUI sets it (usv_playpen_gui.py:5100).

**Fix:** Reword to present tense: 'invoked by the visualization pipeline (visualize_data.py) when the make_usv_spectrograms_bool toggle is set.'

### [LOW] docs_clarity — scatter_max_points doc hard-codes 'seed=42' though the downsample uses the configurable seed parameter
`make_usv_spectrograms.py:3225-3226`

scatter_max_points doc says 'random sample with seed=42'. The downsample (3360) uses sample(n=..., seed=seed), the configurable seed arg (default 42). The seed param doc (3241-3242) correctly says it seeds both samplings.

**Fix:** Change 'random sample with seed=42' to 'random sample using the seed argument'.

### [LOW] docs_clarity — _render_spectrogram cmap param doc says 'name' but a Colormap object is also accepted/passed
`make_usv_spectrograms.py:488`

cmap documented as 'Matplotlib colormap name.' (str), but supplied by _resolve_cmap() (706, 790) which per its docstring (250-251) may return a str OR a matplotlib.colors.Colormap (cmap_override).

**Fix:** Reword to 'Matplotlib colormap name or Colormap instance (whatever specshow's cmap= accepts).'

### [LOW] docs_clarity — mask_excluded_categories docstring type narrower than the signature (int | None also accepted)
`make_usv_spectrograms.py:3160`

Signature is tuple[int, ...] | int | None (3072); the body normalises None->() and bare int->(int,) (3257-3262). The docstring (3160) documents only '(tuple of int)'.

**Fix:** Document as 'tuple of int | int | None' and note a bare int becomes a one-element tuple, None an empty tuple.

### [LOW] performance — unstretched_specs pre-pass re-reads each pick's durations scalar from HDF5, then the main loop reads it again
`make_usv_spectrograms.py:3746-3793`

When unstretched_specs is True, the pre-pass (3746-3764) reads grp['durations'][spec_idx] per pick; the main loop re-reads grp['durations'][spec_idx] for the same picks at 3793. On a network HDF5 store each scalar read has latency, so every pick incurs two duration reads.

**Fix:** Cache per-session durations arrays in a dict (like mask_index_cache): dur_cache.setdefault(sess, grp['durations'][:]) then index the cached array in both passes.

### [LOW] performance — Histogram computed twice per panel (np.histogram then ax.hist over identical data/bins)
`make_usv_spectrograms.py:1709-1710`

np.histogram(values_disp, bins=bins) (1709) then ax.hist(values_disp, bins=bins, ...) (1710) bin the same array twice. hist_counts/hist_edges are only consumed in the freq_bandwidth_hz branch (1737-1738); for the other four panels the np.histogram output is unused.

**Fix:** Move np.histogram into the freq_bandwidth_hz branch, or draw via ax.stairs(hist_counts, hist_edges, fill=True, ...) instead of ax.hist.

### [LOW] performance — plot_session_usv_timeline filters the full DataFrame six times for per-group counts and draws
`make_usv_spectrograms.py:2189-2199`

group_counts (2189-2192) filters df per group (3 scans), then the draw loop (2198-2199) filters per group again (3 more) -- six full scans where one partition would suffice.

**Fix:** Compute sub-frames once via df.partition_by('sex', as_dict=True), derive group_counts from each .height, and iterate the same dict in the draw loop.


## `usv_interval_summary_statistics.py` (16)

### [HIGH] docs_clarity — Stale Returns docstring for plot_ic_curves: documented stats keys do not exist
`usv_interval_summary_statistics.py:371-372`

Returns documents stats keys {'best_n_comp','best_ic','parsimonious_n_comp','parsimonious_ic','delta_vs_best'} but the code (lines 430-437) builds {'min_ic_per_K', 'selected_n_components'}. None of the five documented keys are produced; a caller relying on the docstring would KeyError.

**Fix:** Rewrite Returns to: Mapping sex -> {'min_ic_per_K': dict mapping int K -> float IC value, 'selected_n_components': int K_selected or None}.

### [MEDIUM] docs_clarity — Stale inline comment in plot_ic_curves contradicts the code and the adjacent comment
`usv_interval_summary_statistics.py:400-401`

Comment says 'outlined (not filled) black square'; the code (412-417) draws a FILLED CIRCLE (marker='o', color=col, edgecolors=edge_color). The next comment block (408-411) correctly says 'a larger filled circle ... no nested square', directly contradicting 400-401.

**Fix:** Delete the 400-401 comment (408-411 already describe the marker correctly) or replace with filled-circle wording.

### [MEDIUM] docs_clarity — Incorrect inset param doc in _draw_qq_into_axes: claims 'no axis labels' but labels are set
`usv_interval_summary_statistics.py:886-887`

Inset param docstring says 'no axis labels', but the inset=True branch (918-919) calls ax.set_xlabel('Observed (s)') and ax.set_ylabel('Model (s)') and the inline comment (916-917) says it retains compact axis labels. The claim is stale.

**Fix:** Change 'no axis labels' to 'compact axis labels (Observed (s) / Model (s)) and no plot title' (the title is omitted in inset mode, line 929).

### [MEDIUM] tests — Student-t (TMixture) dispatch branch in plot_best_fit_with_annotations is never tested
`usv_interval_summary_statistics.py:569-572`

plot_best_fit_with_annotations branches on isinstance(gmm, TMixture) at line 569 calling summarize_best_t_mixture. Grep confirms the test file never imports/constructs a TMixture, so the t branch (and empty-boundaries behaviour) is uncovered through this wrapper.

**Fix:** Add a test passing a TMixture to plot_best_fit_with_annotations, asserting (fig, ax, summary) with 'qq_pearson_r' present and 'logmeans' populated.

### [MEDIUM] tests — _draw_qq_into_axes / plot_qq t-mixture quantile branch is never tested
`usv_interval_summary_statistics.py:898-901`

_draw_qq_into_axes branches at line 898 on isinstance(gmm, TMixture) calling t_mixture_quantile_logspace (899). No TMixture in tests, so this dispatch (the only visualization-layer call site of t_mixture_quantile_logspace) is uncovered.

**Fix:** Add a plot_qq test passing a TMixture so t_mixture_quantile_logspace runs, asserting a finite pearson_r.

### [MEDIUM] tests — selected_K_from_h5 fallback-to-bootstrap_lrt-table path is untested
`usv_interval_summary_statistics.py:1740-1747`

selected_K_from_h5 reads K_selected_* attrs (1731-1737), else falls back to scanning bootstrap_lrt's K_selected_step_up column (1740-1747). The archive fixture always writes the attrs and the existing test only checks the attrs path, so the fallback loop and the 'df is None or df.height == 0 -> return {}' guard are uncovered.

**Fix:** Add a test with attrs missing K_selected_* but bootstrap_lrt populated, asserting derivation from the table; plus a test where bootstrap_lrt is absent so it returns {}.

### [LOW] correctness — source_map lookup uses .get() with empty-string default, masking a configure_path mismatch
`usv_interval_summary_statistics.py:114`

source_list = source_map.get(session_root, "") silently substitutes an empty string when a session root is absent from source_map. Both _read_session_lists and _session_source_map run lines through configure_path, so a present session should always map; a miss indicates a path-resolution mismatch that this default hides, losing provenance for source_list rows. Also violates the project convention against .get() with defaults.

**Fix:** Index directly: source_list = source_map[session_root], letting a genuine mismatch surface as a KeyError.

### [LOW] correctness — null_max extraction uses .get() with np.nan default plus a redundant double .get() call
`usv_interval_summary_statistics.py:1201`

Line 1201 reads null_max = float(res.get("null_max", np.nan)) if res.get("null_max") is not None else (...). It calls res.get("null_max") twice and uses a .get() default, against the convention. load_lrt_sweep_from_h5 (line 1683) always populates null_max, so the fallback branch is effectively dead for current archives; the duplicate lookup is wasteful.

**Fix:** Read once and branch on presence: nm = res.get("null_max"); null_max = float(nm) if nm is not None else (float(lr_null.max()) if lr_null.size else float("nan")).

### [LOW] correctness — mode.get('attrs', {}) uses a .get() default, bypassing direct-key convention
`usv_interval_summary_statistics.py:1729`

attrs = mode.get("attrs", {}) (line 1729) and attrs.get(attr_key) (line 1732) use .get() with defaults. read_usv_interval_h5 constructs every mode dict, so attrs should always be present; the {} default silently turns a malformed archive into 'no selected K' rather than a clear error.

**Fix:** Use attrs = mode["attrs"]; keep the per-key membership check (val = attrs[attr_key] if attr_key in attrs) for the legitimately-optional selected-K attributes.

### [LOW] docs_clarity — Module docstring over-generalizes the (fig, ax, stats_dict) return convention
`usv_interval_summary_statistics.py:8-9`

Module docstring asserts plot functions return (fig, ax, stats_dict). plot_ic_curves returns (f, (ax_left, ax_right), stats) (line 440) and plot_bootstrap_lrt_panel returns (f, axes) with no stats. Per-function docstrings are correct; only the blanket module claim is misleading.

**Fix:** Soften to 'Most single-axis plot functions return (fig, ax, stats_dict); multi-axis panels deviate as documented per function.'

### [LOW] performance — Per-element Python loop over numpy interval arrays with scalar np.log per element
`usv_interval_summary_statistics.py:129-150`

usv_interval['male']/['female'] are numpy float64 arrays. The two loops iterate element-by-element building one dict per interval and calling float(np.log(v)) per scalar, incurring full ufunc dispatch each call. This is the only O(n_intervals) Python-level loop in the compute path and scales with total intervals across sessions x modes x sexes.

**Fix:** Compute log_arr = np.log(usv_interval[sex]) once per sex, then zip the interval and precomputed log arrays when building rows.

### [LOW] performance — df_results['sex'].unique().to_list() recomputed on every loop iteration
`usv_interval_summary_statistics.py:384`

Inside the for sex loop, df_results['sex'].unique().to_list() is recomputed each iteration for a membership test; unique() scans the whole sex column. The present-sex set is loop-invariant.

**Fix:** Hoist: present_sexes = set(df_results['sex'].unique().to_list()); test if sex not in present_sexes.

### [LOW] performance — Repeated full-column polars filter of null_df inside per-row loop (O(rows x null_df))
`usv_interval_summary_statistics.py:1661-1671`

load_lrt_sweep_from_h5 iterates every summary_df row and runs null_df.filter(3-predicate).sort('b') over the entire long-form null table each time. A single partition_by would do it in one pass.

**Fix:** Partition once: null_df.sort('b').partition_by(['sex','K_null','K_alt'], as_dict=True), then look up each group directly.

### [LOW] tests — plot_best_fit_with_annotations qq_inset_bbox=None path (qq_pearson_r stays NaN) is untested
`usv_interval_summary_statistics.py:671-684`

When qq_inset_bbox is None and auto_inset_below_legend is False, no inset is drawn and qq_pearson_r stays float('nan') (line 580, returned 684). No test passes qq_inset_bbox=None, so the documented NaN contract for the no-inset path is unverified.

**Fix:** Add a test calling plot_best_fit_with_annotations(..., qq_inset_bbox=None) and assert np.isnan(summary['qq_pearson_r']).

### [LOW] tests — plot_log_usv_interval_histograms empty-sex branch (NaN median) is untested
`usv_interval_summary_statistics.py:250-251`

Lines 250-251 return float('nan') for median when a sex has no rows; ax.hist guarded on size. The single test always supplies both sexes, so neither the size==0 skip nor the NaN-median fallback is exercised.

**Fix:** Add a test with only 'male' rows; assert stats['n_F']==0 and np.isnan(stats['median_F_sec']) while the male histogram renders.

### [LOW] tests — plot_ic_curves missing-sex skip branch is untested
`usv_interval_summary_statistics.py:383-385`

Lines 383-385 skip a sex absent from df_results['sex'].unique(). Both ic-curve tests supply both sexes, so the continue and resulting stats dict omitting the missing sex are uncovered.

**Fix:** Add a test with only 'male' rows and assert 'female' absent from returned stats while both twin axes still render.


## `auxiliary_plot_functions.py` (15)

### [HIGH] correctness — Diverging colormap raises broadcast error for even cm_length
`auxiliary_plot_functions.py:217-233`

For cm_type=='diverging', the second slice a[cm_length//2:] has length cm_length-(cm_length//2) while the linspace has length cm_length//2+1. These match only for ODD cm_length (e.g. 255, 101). For EVEN cm_length the second assignment errors: empirically verified cm_length=256 -> 'could not broadcast input array from shape (129,) into shape (128,)' and cm_length=100 -> shape (51,) into (50,). The diverging path silently only works for odd lengths.

**Fix:** Compute half = cm_length // 2 and fill a[:half+1] and a[half:] with linspaces whose lengths equal the slice lengths (cm_length-half), or special-case the odd middle element so both assignments and the linspace lengths agree for any cm_length.

### [HIGH] correctness — RGB channels normalized by cm_length instead of 255 (silent wrong colors)
`auxiliary_plot_functions.py:211-232`

All RGB channel values are divided by input_parameter_dict['cm_length'] to map into [0,1]. This is only correct when cm_length==255. qlvm_torus_traversal_video.py:443 actually calls create_colormap with cm_length=256 and cm_end=(255,255,255): 255/256=0.996, so 'white' renders as off-white and accent channels are mis-scaled. ListedColormap silently clips out-of-range values (e.g. cm_length<255), producing wrong colors with no error. The reversed end->start fill ordering for the sequential branch is also uncommented.

**Fix:** Divide channel values by 255.0 (the RGB max) rather than by cm_length in both the sequential (lines 211-216) and diverging (lines 218-233) branches; keep cm_length only as the number of linspace samples / array rows. Add a brief comment on the reversed end->start fill order.

### [MEDIUM] correctness — Integer saturation/change_saturation is silently ignored (only float triggers change)
`auxiliary_plot_functions.py:116-118`

saturation is applied only when isinstance(saturation, float) (line 116). The documented default for change_saturation is 1 (an int, line 164), and qlvm_torus_traversal_video.py:446 passes int 1 while other callers pass 1.0/0.5 as float. Integer values will NOT have saturation applied. The same int/float trap gates the whole luminance/saturation branch at line 182: isinstance(input_parameter_dict['change_saturation'], float). Silent: the requested saturation is ignored with no error.

**Fix:** Test for numeric values inclusive of ints while excluding bools, e.g. isinstance(saturation, (int, float)) and not isinstance(saturation, bool), at line 116 and at line 182 for change_saturation.

### [MEDIUM] dead_code_naming — Module docstring misattributes the code to Crameri perceptually-uniform colormaps
`auxiliary_plot_functions.py:3`

The module docstring states 'Creates perceptually uniform colormaps (per Crameri, F. et al., Nat. Commun. (2020))'. The code does plain linear RGB interpolation between two (or three, diverging) user-supplied RGB anchors via np.linspace, with optional HLS luminance/saturation equalization. It neither imports nor reproduces any Crameri colormaps, and linear RGB interpolation is precisely what Crameri's work argues against. The citation is misleading.

**Fix:** Rewrite the module docstring to describe what the file actually does (linear-interpolated sequential/diverging matplotlib colormaps between user-specified RGB anchors, with optional HLS luminance/saturation equalization, plus animal-color selection helpers) and remove the Crameri citation.

### [MEDIUM] tests — choose_animal_colors has no direct unit test
`auxiliary_plot_functions.py:14-48`

choose_animal_colors (lines 14-48) is a public function not imported or directly tested in tests/visualizations/test_auxiliary_plot_functions.py (its import block at lines 16-19 only pulls in luminance_equalizer and create_colormap). Its male/female branching and the independent n_males/n_females counters when sexes are interleaved are untested.

**Fix:** Add a direct unit test importing choose_animal_colors. Pass exp_info_dict={'mouse_sex': ['male','female','male']} and visualizations_parameter_dict={'male_colors': ['#111111','#222222'], 'female_colors': ['#aaaaaa']}, assert the returned list equals ['#111111','#aaaaaa','#222222'].

### [LOW] correctness — match_by documented default 'max' is not implemented; luminance with match_by=None raises ValueError
`auxiliary_plot_functions.py:55`

The docstring (line 72) states match_by 'defaults to max', but the signature (line 55) is match_by: str | None = None and the dispatch (lines 92-111) has no None branch. When luminance is True/float and match_by is None, all elif branches are skipped and the else raises ValueError. So the documented 'max' default does not exist; harmless today only because every repo caller passes match_luminance_by explicitly.

**Fix:** Either set match_by default to 'max' in the signature so behavior matches the docstring, or correct the docstring to state match_by is required when luminance equalization is requested (and add an explicit None->'max' branch before the else/raise).

### [LOW] correctness — choose_animal_colors treats any non-'male' sex as female and can IndexError
`auxiliary_plot_functions.py:38-46`

The loop classifies every sex not exactly 'male' as female (else branch), so typos/unexpected labels are silently bucketed as female. n_males/n_females index directly into the color lists with no bounds check; more males or females than provided colors raises a bare IndexError. With exp_info_dict defaulting to None, calling with no args raises TypeError on line 38 instead of a descriptive message.

**Fix:** Validate sex against an explicit {'male','female'} set (raise on unknown), guard the color-list indexing with a clear error when colors run out, and raise a descriptive error if exp_info_dict/visualizations_parameter_dict is None.

### [LOW] dead_code_naming — luminance/saturation type hints (bool | None) hide the supported float path
`auxiliary_plot_functions.py:54-56`

luminance (line 54) and saturation (line 56) are annotated bool | None, yet the body explicitly supports float values: isinstance(luminance, float) at line 91, float(luminance) at line 105, isinstance(saturation, float) at line 116. The docstring documents them as 'bool / float'. The annotations understate the accepted types.

**Fix:** Widen the annotations to bool | float | None for both luminance (line 54) and saturation (line 56) so the hints match the documented and implemented behavior.

### [LOW] docs_clarity — match_by docstring omits the 'set' option that the code supports
`auxiliary_plot_functions.py:72`

The docstring for match_by reads "Match luminance by 'max', 'min' or 'mean'; defaults to 'max'." but the code at line 104 also accepts match_by == 'set', and the ValueError at lines 108-109 lists all four valid values. create_colormap's own docstring (line 162) already lists all four correctly, confirming 'set' is intended.

**Fix:** Update the match_by line to: "Match luminance by 'max', 'min', 'mean' or 'set'" to match the implementation and the create_colormap docstring.

### [LOW] docs_clarity — luminance docstring does not explain its dual role as a literal value when match_by='set'
`auxiliary_plot_functions.py:69-70`

The luminance parameter is documented only as 'Equalizes luminance of spectrum ends.' but it has two roles: a flag/gate (True or float triggers equalization at line 91), AND the explicit target luminance consumed directly when match_by == 'set' (line 105). A reader cannot infer that passing a float sets the absolute luminance.

**Fix:** Expand the luminance doc to note that a float value both enables equalization and, when match_by='set', is used directly as the target luminance for both spectrum ends.

### [LOW] docs_clarity — saturation documented as (bool / float) but only float has any effect
`auxiliary_plot_functions.py:73-74`

The saturation parameter is typed '(bool / float)' and defaults to False (line 56). The code only acts on it via isinstance(saturation, float) at line 116; a bool value (including True) is ignored and the original saturation is preserved (lines 120-121). The doc implies a bool toggles saturation change, which it does not.

**Fix:** Reword to clarify that only a float value changes saturation (both ends set to that float), and that any non-float (e.g. the default False) leaves the original saturation unchanged.

### [LOW] docs_clarity — choose_animal_colors Returns documents 'None' that is never returned
`auxiliary_plot_functions.py:16`

The signature return annotation is list | None (line 16) and the Returns section documents only mouse_colors (list). The function always returns a list (line 48) and never returns None, so the | None is misleading and the docstring gives no condition under which None could occur.

**Fix:** Tighten the return annotation to -> list, or if None is genuinely possible (e.g. on bad input) add a branch and document the condition under which None is returned.

### [LOW] tests — create_colormap None-input ValueError path untested
`auxiliary_plot_functions.py:174-176`

Lines 174-176 raise ValueError('create_colormap requires an input_parameter_dict (got None).') when input_parameter_dict is None. No test exercises this guard; test_create_colormap_unknown_type_raises only covers the cm_type ValueError at lines 234-240.

**Fix:** Add a test: with pytest.raises(ValueError, match='requires an input_parameter_dict'): create_colormap(None) (and also create_colormap() since input_parameter_dict defaults to None).

### [LOW] tests — Sequential white-end skip branch of luminance equalisation untested
`auxiliary_plot_functions.py:194-196`

At lines 194-196 the sequential luminance_equalizer call is SKIPPED when cm_end == (255,255,255). The only sequential test overrides cm_end to (10,10,90), forcing the equalize branch to RUN. The white-end skip path (the _cm_params default) is never asserted.

**Fix:** Add a sequential test using the default white cm_end (e.g. create_colormap(_cm_params())) and assert isinstance ListedColormap with cm.N == 64, confirming the skip path produces the expected unequalised ramp.

### [LOW] tests — cm_opacity alpha channel never asserted
`auxiliary_plot_functions.py:241-242`

Lines 241-242 set the alpha channel (column 3) of every colormap entry to cm_opacity. Both create_colormap tests only assert isinstance and cm.N; none inspect alpha. A non-default cm_opacity and its propagation to all entries' alpha is untested.

**Fix:** Add a test with cm_opacity=0.5 and assert np.allclose(cm(np.linspace(0,1,cm.N))[:, 3], 0.5) to verify opacity is applied uniformly.


## `visualize_data.py` (10)

### [HIGH] tests — Per-directory exception handling and failed_directories accumulation untested
`visualize_data.py:141-143`

The except block (lines 141-143) catching the broad exception tuple, emitting traceback.format_exc() via message_output, and appending (one_directory, 'Type: msg') to failed_directories has no test. Existing tests use mocks that always succeed, so failure capture, traceback emission, continuation to next directory, and tuple shape are unverified.

**Fix:** Add a test with make_neuronal_tuning_figures_bool True and two root dirs where make_neuronal_tuning_figures.side_effect = ValueError('boom'); assert no raise, the second dir still processes, and message_output received the traceback text.

### [HIGH] tests — Failure-summary completion-email branch is never exercised
`visualize_data.py:171-196`

Tests reach only the success branch (lines 183-190); the failure branch (lines 171-182) building 'completed with N failure(s)' + failure_summary, sent via send_message (192-196), is uncovered. The honest-failure-reporting behavior has no asserting test.

**Fix:** In the side_effect-failure test, assert the second send_message call has a subject containing 'failure(s)' and a message containing the failed dir name and exception text, and send_message.call_count == 2.

### [MEDIUM] tests — __init__ JSON settings-loading fallback path is never tested
`visualize_data.py:51-57`

All tests pass BOTH input_parameter_dict and root_directories, so the branch at lines 51-57 reading visualizations_settings.json and deriving defaults (self.root_directories = _settings['visualize_data']['root_directories']; self.input_parameter_dict = _settings) is uncovered. A schema rename of 'visualize_data' or 'root_directories' would go uncaught.

**Fix:** Add tests constructing Visualizer() with both args omitted and with each omitted, mocking open/json.load to return a dict with 'visualize_data' -> 'root_directories', then assert the attributes are populated from it.

### [MEDIUM] tests — QLVM cohort-block except path untested
`visualize_data.py:148-155`

test_make_qlvm_torus_traversal_video_logic verifies only the success path. The except at lines 153-155 appending ('qlvm_torus_traversal_video', reason) and emitting a traceback is uncovered.

**Fix:** Add a test with make_qlvm_torus_traversal_video_bool True and make_video.side_effect = RuntimeError('x'); assert no raise, the completion email reports a failure entry labeled 'qlvm_torus_traversal_video', and per-session directories are unaffected.

### [MEDIUM] tests — Embedding-thumbnails cohort-block except path untested
`visualize_data.py:160-166`

test_make_embedding_thumbnails_logic checks only the success call. The except at lines 164-166 appending ('embedding_thumbnails', reason) is uncovered, mirroring the QLVM gap.

**Fix:** Add a test with make_embedding_thumbnails_bool True and render_embedding_thumbnails_for_cohort.side_effect = KeyError('k'); assert no raise and the completion email failure summary contains the 'embedding_thumbnails' label.

### [LOW] correctness — Failure-count denominator excludes cohort-level failures, can print "N of M" with N > M
`visualize_data.py:179`

failed_directories is appended to by the two cohort-level (run-once) blocks (line 155 'qlvm_torus_traversal_video', line 166 'embedding_thumbnails') which are NOT members of self.root_directories. So the completion message at line 179 ('{len(failed_directories)} of {len(self.root_directories)} director(ies) failed') can have numerator > denominator, and 'director(ies)' mislabels the non-directory cohort failures.

**Fix:** Track per-session vs cohort failures separately (or count entries whose first element is in self.root_directories), and phrase the denominator against the true number of attempted units, or drop the 'of M director(ies)' framing and just report failing items by name as failure_summary already does.

### [LOW] correctness — Start-of-run notification e-mail omits no_receivers_notification=False, inconsistent with completion e-mail
`visualize_data.py:102-109`

The start-of-run Messenger call (lines 102-109) does not pass no_receivers_notification, so it defaults to True (send_email.py line 31), printing the 'You chose not to notify anyone via e-mail about PC usage.' line (send_email.py line 161) when no receivers are configured. The completion Messenger call (lines 192-196) explicitly passes no_receivers_notification=False, suppressing that line. Inconsistent console output for the same config within one run.

**Fix:** Pass no_receivers_notification=False on the start-of-run Messenger call too (or drop it from both) so the two notifications behave identically when no receivers are configured.

### [LOW] docs_clarity — --node-lw help text misdescribes its behavior and duplicates --node-connection-lw help
`visualize_data.py:249`

--node-lw (dest node_lw) help is 'Line width for the mouse node connections.', near-identical to --node-connection-lw (line 250). They are distinct settings (visualizations_settings.json lines 141-142) consumed separately: node_lw is the scatter MARKER EDGE width (make_behavioral_videos.py:500 linewidth=node_lw), while node_connection_lw drives animal_line_width for the connecting lines (make_behavioral_videos.py:1839/2095 -> line 484). The node_lw help is wrong and indistinguishable from the connection help.

**Fix:** Reword line 249 help to describe what node_lw actually controls, e.g. help='Line width (edge) for the mouse node markers.' so it is distinct from --node-connection-lw on line 250.

### [LOW] docs_clarity — --spectrogram-yticks help is singular, lacks Hz units, and omits trailing period
`visualize_data.py:264`

The help 'Y-tick position for spectrogram' is the only option help in the block missing a trailing period, reads as singular ('position') despite multiple=True, and omits the Hz unit that adjacent frequency options (lines 263, 267) specify.

**Fix:** Reword to e.g. 'Y-tick position(s) for the spectrogram y-axis (Hz).' to match plurality, units, and punctuation of surrounding option help strings.

### [LOW] tests — CLI list-valued (multiple/nargs) options never pass through modify_settings_json_for_cli
`visualize_data.py:284-292`

parameters_lists (lines 284-285) and provided_params extraction (line 287) for multiple=True / nargs=2 options are never exercised: the only CLI test (test_cli.py test_visualize_3d_data_cli_success) passes a single scalar flag (--animate) and mocks modify_settings_json_for_cli.

**Fix:** Add a CLI test invoking visualize_3D_data_cli with --brain-areas A --brain-areas B and --spectrogram-power-limit 0 100, with a spy on modify_settings_json_for_cli, asserting those keys appear in provided_params and parameters_lists is forwarded.


## `plot_style.py` (4)

### [HIGH] tests — apply_plot_style() (the only public function) has zero test coverage
`plot_style.py:51-94`

apply_plot_style() is in coverage scope (pyproject [tool.coverage] run.source=['usv_playpen'], not in run.omit) yet has no test: grep for 'apply_plot_style' and 'plot_style' across tests/ returns nothing, and tests/visualizations/ has no test_plot_style.py. The function is imported and called at module-import time by multiple production plotting modules, so a regression breaks plotting across the package undetected.

**Fix:** Add tests/visualizations/test_plot_style.py: call apply_plot_style(), assert it returns None and does not raise, then assert the five bundled faces are registered: {str(pathlib.Path(e.fname).resolve()) for e in fm.fontManager.ttflist if e.name == 'Helvetica'} == {str((ps._FONTS_DIR / t).resolve()) for t in ps._HELVETICA_TTFS}. Also assert ps._STYLE_PATH.is_file().

### [MEDIUM] correctness — apply_plot_style() is not idempotent: ttflist leaks 5 Helvetica entries per call and re-parses TTFs every time, contradicting the docstring
`plot_style.py:82-94`

The module docstring (lines 8-10) and function docstring (lines 57-60) assert the call is idempotent because matplotlib's font_manager.addfont is 'no-op-on-repeat'. This is false: FontManager.addfont unconditionally does self.ttflist.append(prop) (verified in matplotlib 3.10.7 source). The prune at lines 88-91 keeps every entry whose resolved path is in bundled_helvetica_paths, so all freshly-appended bundled duplicates survive. Empirically the Helvetica count in ttflist grows 1 -> 5 -> 10 -> 15 across three calls. Since the design (lines 12-26) is to call this at the top of every plotting module and notebook cell, ttflist grows by 5 each invocation in a session, slowing every subsequent findfont lookup, plus each call redundantly re-parses 5 TTF headers (line 85) and re-stats every Helvetica fname via Path(...).resolve() (line 90).

**Fix:** Add a module-level sentinel (e.g. _STYLE_APPLIED = False) and short-circuit at the top of apply_plot_style() when already applied so repeated calls are truly O(1); or dedupe ttflist by resolved path after the addfont loop so each bundled face appears once. Either way, correct the docstring claim that addfont is no-op-on-repeat.

### [MEDIUM] tests — De-shadowing prune of non-bundled 'Helvetica' entries is untested (both keep and drop arms)
`plot_style.py:88-91`

The ttflist comprehension at lines 88-91 is the core correctness behavior described in the docstring (drop any 'Helvetica'-named entry whose resolved path is not bundled). The branch where an external/system font named 'Helvetica' is removed while bundled ones are kept is never exercised, so a logic inversion or an off-by-one in the resolve()/set comparison would pass silently.

**Fix:** In a test, inject a fake non-bundled entry (e.g. append fm.FontEntry(fname='/usr/share/fonts/FakeHelvetica.ttf', name='Helvetica', ...) to fm.fontManager.ttflist) before calling apply_plot_style(), then assert no 'Helvetica' entry with that fname remains while all five bundled paths are still present. Add a second case with a non-'Helvetica' system font and assert it is preserved (exercises the name != 'Helvetica' keep arm).

### [MEDIUM] tests — Idempotency claimed in docstring is never verified (would catch the duplicate-leak bug)
`plot_style.py:82-91`

The module docstring (lines 8-10) and function docstring (lines 57-60) assert the call is idempotent/safe to repeat. No test verifies this, and in fact it is currently false (ttflist grows by 5 per call). A repeat-call test would have caught this regression.

**Fix:** Add a test that calls apply_plot_style() twice and asserts the count of 'Helvetica'-named ttflist entries equals 5 after both the first and second call, confirming no duplication and no over-pruning.


## `figure_io.py` (2)

### [LOW] correctness — Empty-string override_dir silently writes to CWD instead of raising
`figure_io.py:146-156`

The guard on line 146 only checks `if override_dir is None`. An empty string or pathlib.Path("") passes the guard; configure_path("") returns "" (passthrough, confirmed by reading os_utils.configure_path), pathlib.Path("") resolves to "." (CWD), and save_dir.mkdir(parents=True, exist_ok=True) succeeds against CWD so the figure is silently misfiled there. The settings-derived branch (line 148) correctly raises ValueError on empty. Callers build override_dir as pathlib.Path(self.root_directory)/'audio' etc., so an empty root_directory would hit this asymmetry.

**Fix:** Change line 146 to `if not override_dir:` so empty/whitespace overrides fall through to the settings path and the existing ValueError instead of resolving to CWD.

### [LOW] correctness — Docstring/module-doc claim 'svg' fallback but code falls back to 'png'
`figure_io.py:122-124`

Module docstring example (line 12) and resolve_save_path override_format doc (lines 122-124) say the format fallback is "svg", but _DEFAULT_FIG_FORMAT = "png" (line 53) and that is what line 161 uses. The actual visualizations_settings.json figures block also uses "png". Documented svg fallback is stale.

**Fix:** Change documented format fallback from "svg" to "png" in the module docstring example (line 12) and the override_format Parameters entry (lines 122-124).
