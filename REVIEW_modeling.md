# Modeling subsystem review

_Verified line-by-line sweep of 22 files (~24k LOC): 291 findings. Report-first._

## Summary
- by severity: high 6 · medium 94 · low 191
- by dimension: tests 115 · docs_clarity 77 · correctness 49 · performance 36 · dead_code_naming 14


## `jax_neural_network_cnn.py` (30)

### [MEDIUM] correctness — Y_center/Y_scale tanh bounds computed over full Y (incl. test folds) -> mild leakage
`jax_neural_network_cnn.py:1241-1242`

Y_center/Y_scale are computed once from np.max/min over the entire Y before the fold loop, then used in cnn_forward as predictions = Y_center + tanh(logits)*Y_scale (line 694). Each fold's held-out test coordinates therefore co-define the per-fold prediction bounding box. Leakage is small (bounding-box extent only, no labels) but biases the cross-validated generalization estimate.

**Fix:** Compute Y_center/Y_scale inside the fold loop from Y[train_idx] (Y_tr_base) only.

### [MEDIUM] docs_clarity — Module docstring says MaxPool, but the Heavy Path uses average pooling
`jax_neural_network_cnn.py:13-15`

Module docstring point 2 describes the Heavy Path as `Depthwise -> Pointwise -> MaxPool`. The code does average pooling: lines 621-624 reduce_window(..., jax.lax.add, ...) over a stride-2 window of length 2, then line 625 multiplies by 0.5 (labeled `# Average pooling`). MaxPool is concretely incorrect.

**Fix:** Change `MaxPool` to `AvgPool`/`average pooling` in the module docstring.

### [MEDIUM] performance — Cluster-invariant global saliency template recomputed for every cluster in Phase 3
`jax_neural_network_cnn.py:1148-1149`

glob_scalar_fn (1096-1105) never references polygon_centroid, so glob_grads/glob_saliency/global_template (1140,1145,1148) are identical on every call. compute_centroid_saliency is invoked once per cluster in the loop at line 1824/1832, so with K clusters it runs K redundant full-batch backward passes for the global path.

**Fix:** Compute the global saliency template once before the cluster loop and pass it in, OR split the method so the global path runs once.

### [MEDIUM] tests — apply_temporal_warping has zero direct test coverage
`jax_neural_network_cnn.py:135-182`

apply_temporal_warping is never tested directly; the pipeline only sets warp_range and exercises it transitively without assertions. The center-anchored warp math (177), symmetric edge clipping (178), linear interp (180), and the identity invariant (warp_factor==1.0 -> output==input) are all unverified. No TestApplyTemporalWarping class exists.

**Fix:** Add a TestApplyTemporalWarping class: identity no-op, center fixed point, squeeze/stretch shape+edge clamp, and a hand-computed interp case.

### [MEDIUM] tests — Inception (use_inception_kernels=True) branch of init and forward is entirely untested
`jax_neural_network_cnn.py:383-396`

Both test files force use_inception_kernels=False (test_pipeline_cnn.py:174, JSON default False). The Multi-Scale Inception Block-0 init (386-396) and forward (587-599) paths never run, so their param-shape/forward-shape contract is unvalidated.

**Fix:** Add a test with use_inception_kernels=True and a small inception_kernel_sizes, asserting b0_dw_w_*/b0_pw_w/b0_sc_w shapes and cnn_forward output shape.

### [LOW] correctness — Inception depthwise keys can collide with SE-block keys at >=4 inception kernels
`jax_neural_network_cnn.py:386-387,424,427`

Inception Block-0 depthwise weights use k[j] for j in range(len(inc_kernels)); the i=0 SE block uses k[3] and k[4]. Default inception_kernel_sizes=[3,15,31] (j in {0,1,2}) is safe, but any config with >=4 kernels makes b0_dw_w_3 reuse k[3] (==b0_se_w1's key), correlating initializations.

**Fix:** Allocate inception depthwise keys from a disjoint slice (e.g. k[40+j]) or assert len(inc_kernels) <= 3.

### [LOW] correctness — best_params stays None when epochs==0, crashing final evaluate_batched
`jax_neural_network_cnn.py:1583,1589,1642`

best_params/best_state init to None and are only assigned inside the `if epoch % 5 == 0` block. If hp['epochs']==0 the loop never runs and line 1642 evaluate_batched(best_params=None, ...) crashes inside cnn_forward. No validation that epochs>=1 exists (default 301).

**Fix:** After the epoch loop add `if best_params is None: best_params, best_state = dict(params), dict(state)` or assert epochs >= 1.

### [LOW] correctness — steps_per_epoch sizing consumes the shared rng as a side effect
`jax_neural_network_cnn.py:1538`

In the non-KDE path steps_per_epoch is computed by calling get_grid_balanced_indices(..., rng=rng) just to measure len()//batch_size. This consumes rng.choice draws from the single shared rng before any epoch runs, shifting the random stream relative to the use_kde_weights=True path for the same seed. The grid-balanced length is deterministic given occupancy, so the draw is purely wasted and stream-perturbing.

**Fix:** Size steps_per_epoch without consuming the live rng (pass np.random.default_rng(0), or compute per-cell target counts analytically).

### [LOW] correctness — np.digitize on linspace(min,max,grid_size) buckets the max coordinate into an extra out-of-grid cell
`jax_neural_network_cnn.py:234-238`

x_bins/y_bins = linspace(min,max,grid_size); np.digitize (right=False) returns index grid_size for any value equal to the max edge. Max-coordinate samples land in cell grid_size, not grid_size-1, creating a sparse extra row/column beyond the documented grid_size x grid_size mesh.

**Fix:** Use bins=linspace(min,max,grid_size+1) with np.clip of indices to [0, grid_size-1] so the max value shares the last real cell.

### [LOW] dead_code_naming — _output_axes_count takes an hp parameter it never reads
`jax_neural_network_cnn.py:270-279`

_output_axes_count(hp) ignores hp entirely; body is `return 2`. Call sites at 453/455 pass hp only to satisfy the signature, and the test calls it as _output_axes_count({}). The hp arg is genuinely dead and implies a config-derived value that does not exist.

**Fix:** Make it zero-arg `_output_axes_count() -> int` (updating the two call sites and the test), or document that hp is reserved/unused.

### [LOW] dead_code_naming — loss unpacked from value_and_grad but never used
`jax_neural_network_cnn.py:1452`

`(loss, s_new), grads = grad_fn(p, s)` binds loss, but _compute_grads returns only grads, s_new, rng_key; loss is never read. The tuple unpack is required by has_aux=True but the name `loss` is dead.

**Fix:** Rename the unused binding to `_loss` or `_`.

### [LOW] dead_code_naming — period_for_decode computed unconditionally but only used on the torus sin_cos path
`jax_neural_network_cnn.py:1069`

period_for_decode = jnp.asarray(self.manifold_period) is evaluated every call but consumed only inside _decode when torus_sin_cos is True; on euclidean/legacy 'raw' runs it is dead.

**Fix:** Move jnp.asarray(self.manifold_period) inside the `if torus_sin_cos:` branch of _decode.

### [LOW] dead_code_naming — polygon_centroid parameter name is a vestige; it is a cluster centroid, not a polygon
`jax_neural_network_cnn.py:984`

compute_centroid_saliency takes polygon_centroid: Tuple[float,float], but the value passed at line 1835 is tuple(centroid) from derive_cluster_geometry (an empirical cluster center-of-mass). The docstring (995-998) even says it replaced the legacy binary polygon boundary. The polygon_ prefix is misleading.

**Fix:** Rename the parameter to cluster_centroid (and local polygon_centroid_jax), updating the call site and docstring.

### [LOW] docs_clarity — Flatten docstring claim stale given use_hybrid_flatten path
`jax_neural_network_cnn.py:16-17`

Module docstring point 3 says timing is preserved by flattening `instead of globally averaging it`. But cnn_forward supports hp['use_hybrid_flatten'] (lines 657-659) which concatenates a global-average-pooled vector onto the flattened tensor, doing both.

**Fix:** Add a clause noting that with use_hybrid_flatten the flattened features are concatenated with a global-average-pooled summary.

### [LOW] docs_clarity — 'Euclidean Err' log message misleading on torus runs (model-free prior)
`jax_neural_network_cnn.py:1515`

Prints `[Model-free prior] Euclidean Err:` but err is computed via pairwise_distance(..., metric=self.manifold_metric, ...) (1509-1512), i.e. wrap-aware torus distance when manifold_metric=='torus'.

**Fix:** Make the label metric-aware, e.g. f"{self.manifold_metric.capitalize()} Err".

### [LOW] docs_clarity — 'Euclidean Err' log message misleading on torus runs (per-strategy)
`jax_neural_network_cnn.py:1646`

Prints `[{strategy}] Euclidean Err: {best_err}` but best_err derives from signed_diff_jax(..., metric=self.manifold_metric, ...) (1623-1626), a wrap-aware distance on torus runs.

**Fix:** Replace the hard-coded 'Euclidean Err' with a metric-aware label.

### [LOW] docs_clarity — Phase-2 per-feature print labels metric-aware delta as 'Delta E'
`jax_neural_network_cnn.py:1724`

Prints `Delta E: {mu}` where delta_e (line 1706) = err_perm - base_err and err_perm uses wrap-aware signed_diff_jax (1701-1704). The 'E' (Euclidean) shorthand is misleading on torus runs.

**Fix:** Use a metric-neutral label such as 'Delta Err' or 'Delta dist'.

### [LOW] docs_clarity — Class docstring describes permutation importance as 'Delta Euclidean Error' though it is metric-aware
`jax_neural_network_cnn.py:763-764`

NeuralContinuousCNNRunner class docstring point 4 says importance is quantified via 'Delta Euclidean Error', but Phase-2 (1701-1704) uses signed_diff_jax with manifold_metric (wrap-aware on torus).

**Fix:** Reword to 'Delta manifold-distance error (Euclidean or wrap-aware torus, per manifold_metric)'.

### [LOW] docs_clarity — 'Dynamic Interception' comment for kinematic masking is confusing jargon
`jax_neural_network_cnn.py:1606`

Line 1606 labels the step `# 2. Kinematic Masking (Dynamic Interception)`. 'Dynamic Interception' appears nowhere in apply_kinematic_masking's docstring (which describes 1D Cutout / channel-blinding) and adds confusion.

**Fix:** Drop 'Dynamic Interception' or use the function's framing, e.g. '(1D Cutout: randomly blind feature channels)'.

### [LOW] performance — get_grid_balanced_indices fully materialized just to measure steps_per_epoch
`jax_neural_network_cnn.py:1538`

steps_per_epoch = len(get_grid_balanced_indices(...))//batch_size runs the full digitize + Python dict grouping + per-cell draws, discarding all but the length; the same pass repeats each epoch at line 1596, across all (fold x strategy).

**Fix:** Compute steps_per_epoch from the deterministic per-cell target sizes without materializing the index arrays, or share a cell-occupancy helper.

### [LOW] performance — KDE weight normalization recomputed every epoch despite being epoch-invariant
`jax_neural_network_cnn.py:1591-1592`

p_weights = w_tr / np.sum(w_tr) (line 1592) is recomputed every epoch though w_tr is fixed per fold/strategy.

**Fix:** Hoist p_weights to just before the `for epoch` loop.

### [LOW] performance — Per-epoch Python dict loop over all training rows in get_grid_balanced_indices
`jax_neural_network_cnn.py:241-244`

Cell grouping is a pure-Python loop over every training row (line 241) into a dict-of-lists, called once per epoch per (fold x strategy), O(N) Python iterations over numpy data.

**Fix:** Vectorize with np.unique(return_inverse=True) + np.argsort/np.split on a flat cell id.

### [LOW] performance — apply_temporal_warping uses a per-feature np.interp loop though the warp grid is shared across features
`jax_neural_network_cnn.py:176-180`

t_query depends only on warp_factors[i] (line 177), identical across features, yet np.interp is called per feature (179-180) once per training batch. Since input_t is the integer grid, the interp can be a vectorized gather over all features.

**Fix:** Per sample compute t_query once, floor to lo/frac, then gather x_seq[i][:,lo]*(1-frac)+x_seq[i][:,lo+1]*frac across all features.

### [LOW] performance — apply_kinematic_masking double Python loop over batch x features each training batch
`jax_neural_network_cnn.py:122-131`

Masking iterates `for i: for f:` with a per-pair scalar rng.integers (line 127) every training batch when masking is on; mask decisions are already vectorized (line 120).

**Fix:** Draw all start indices in one rng.integers((batch_size,n_feats)) call and iterate only over np.argwhere(mask_decisions).

### [LOW] tests — use_hybrid_flatten=True branch of init and forward is untested
`jax_neural_network_cnn.py:436-437`

use_hybrid_flatten forced False (test_pipeline_cnn.py:175). The flattened_size+=channels[-1] (436-437) and out_gap concatenate (657-659) never run; dense1_w padding depends on flattened_size.

**Fix:** Add a forward-pass variant with use_hybrid_flatten=True asserting init succeeds, dense1_w sizing, and correct finite output.

### [LOW] tests — _batch_norm_1d inference branch (is_training=False) is untested
`jax_neural_network_cnn.py:513-515`

TestBatchNorm1d calls only is_training=True (lines 231, 244). The else branch (513-515) using frozen mean/var and returning them unchanged is the path used at every eval/inference call yet is unverified.

**Fix:** Add test_inference_uses_frozen_stats: non-trivial mean/var, is_training=False, assert hand-computed output and that returned new_mean/new_var equal the inputs.

### [LOW] tests — HashableDict.__hash__ nested list/dict handling is untested
`jax_neural_network_cnn.py:68-76`

HashableDict is imported and used as a hashing wrapper but its custom __hash__ (68-76) for nested lists/dicts has no direct assertion; a regression surfaces only as an opaque JIT 'unhashable type' error.

**Fix:** Add tests: equal dicts with nested list/dict hash equal, hash differs on value change, hash() does not raise on the real cnn_continuous block.

### [LOW] tests — _use_sin_cos_torus_output missing-key (KeyError) path untested
`jax_neural_network_cnn.py:313-321`

TestOutputHeadGates covers the truth table (167-178) and the ValueError on invalid encoding (180-185), but not the strict-lookup KeyError when 'cnn_torus_output_encoding' is absent on a torus run (line 315).

**Fix:** Add test_missing_encoding_key_raises_keyerror with pytest.raises(KeyError): _use_sin_cos_torus_output({'manifold_metric':'torus'}).

### [LOW] tests — gelu activation and hard_sigmoid SE-gate branches of cnn_forward are untested
`jax_neural_network_cnn.py:573-574`

act=relu vs gelu on hp['act_func'] (573) and se_gate=sigmoid vs hard_sigmoid on hp['se_activation'] (574). All tests use the shipped defaults relu/sigmoid, so the alternate branches never run; a typo in either selector would pass CI.

**Fix:** Add a parametrized forward-pass test over act_func and se_activation asserting correct shape and finite outputs.

### [LOW] tests — cnn_forward dropout masking branch is exercised but never asserted to drop
`jax_neural_network_cnn.py:670-673`

The dropout block (670-673) runs only when is_training and dropout_rate>0 and rng_key is not None. The pipeline forces dropout_rate=0.0; the architecture gradient test uses 0.3 with a fixed key but never asserts dropout zeros activations or that two keys differ, so the inverted-scaling logic (line 673) and rng dependence are unvalidated.

**Fix:** Add test_dropout_is_active_in_training: two distinct rng_keys yield different outputs; rng_key=None equals the no-dropout output.


## `load_input_files.py` (24)

### [MEDIUM] correctness — Unguarded None file paths produce opaque crashes in load_behavioral_feature_data
`load_input_files.py:54-59`

features_csv_file_path (54) and track_file_path (55) come from next(..., None); a missing file makes them None, then h5py.File(name=None) (56) and pls.read_csv(source=None) (59) raise a confusing low-level TypeError/OSError. Sibling loaders (318-320, 628-630) guard csv_path is None with a warning + continue; this loader has no skip path.

**Fix:** After lines 54-55 add explicit None checks mirroring the other loaders, e.g. if track_file_path is None or features_csv_file_path is None: print warning and continue.

### [MEDIUM] docs_clarity — Stale docstring: gmm_params 'falls back to hard-coded defaults' is false
`load_input_files.py:266-269`

Docstring (268) claims a None fallback, but lines 308-309 index gmm_params['male']/['female'] unconditionally and no hard-coded defaults exist (grep confirms only the three doc lines mention 'hard-coded'). None raises TypeError.

**Fix:** Reword to state gmm_params is required (dict with 'male'/'female' -> 'means'/'sds') and that None raises TypeError; or implement the fallback.

### [MEDIUM] docs_clarity — Stale docstring: gmm_params fallback claim is false in find_variable_length_bouts
`load_input_files.py:856-859`

Docstring (858) claims a None fallback, but lines 888-889 dereference gmm_params unconditionally and no defaults exist.

**Fix:** Reword the gmm_params entry to state it is required and None raises TypeError (or implement the fallback).

### [MEDIUM] docs_clarity — Stale Process Outline reference to hard-coded GMM defaults
`load_input_files.py:826-828`

Process Outline step 2 (826-828) says params come 'from gmm_params or hard-coded defaults'; no defaults exist, params dereferenced at 888-889.

**Fix:** Remove 'or hard-coded defaults' so the outline matches the code.

### [MEDIUM] docs_clarity — Missing docstring entries for category_column and noise_column in find_variable_length_bouts
`load_input_files.py:799-885`

Signature declares category_column (812) and noise_column (813), both used (976-986, 954-955), but Parameters ends at noise_vocal_categories (872-875) and documents neither.

**Fix:** Add Parameters entries for category_column and noise_column mirroring find_bout_epochs 278-298.

### [MEDIUM] docs_clarity — Missing docstring entry for noise_column in find_usv_categories
`load_input_files.py:542-554`

noise_column (554) is used for global noise filtering (673-674) but never documented in Parameters (568-602), though category_column and noise_vocal_categories are.

**Fix:** Add a noise_column Parameters entry explaining it is the supercategory column for noise removal, kept separate from category_column.

### [MEDIUM] docs_clarity — Returns docstring omits continuous_supercategory and continuous_category keys
`load_input_files.py:605-617`

find_usv_categories writes 'continuous_supercategory' (763-765) and 'continuous_category' (766-769); both are consumed downstream (modeling_usv_manifold_position.py:809-814), but the Returns key list (608-616) omits them.

**Fix:** Add both keys to the Returns list, noting they are per-USV label arrays aligned 1:1 with continuous_onsets and present only when the prefixed columns exist.

### [MEDIUM] performance — O(M x N) Python loop for 'bout'-mode negative events; replaceable with vectorized searchsorted
`load_input_files.py:489-502`

Per-onset O(N) boolean reduction over all_usv_starts (493-496) with two array allocations each iteration. all_usv_starts is sorted (380), so two np.searchsorted calls compute all counts at once.

**Fix:** lo = searchsorted(all_usv_starts, all_clean_onsets); hi = searchsorted(all_usv_starts, all_clean_onsets + usv_bout_time); usv_events_negative = all_clean_onsets[(hi-lo)==0].

### [MEDIUM] tests — Middle clean-zone branch of _get_clean_tiled_epochs untested
`load_input_files.py:138-141`

The middle-gap loop (139-141) and no-overlap merge branch (112-115) never run because no test produces two distinct merged forbidden intervals.

**Fix:** Add a test with two widely-separated USVs (starts [10.0,50.0], stops [10.5,50.5], fh=1.0, dur=100.0) asserting tiles fall inside the clean gap (11.5<=t<=50.0) and every window avoids both zones.

### [MEDIUM] tests — find_usv_categories manifold-columns-absent branch untested
`load_input_files.py:743-748`

Only the all-columns-present path of the manifold guard (744) is tested; the missing-column path (continuous_onsets/targets stay None) is not.

**Fix:** Add a test with manifold_column_names=['vae_umap1','vae_umap2'] against a CSV with only 'vae_umap1' and assert continuous_onsets/continuous_targets are None.

### [MEDIUM] tests — find_usv_categories partial super/category label columns untested
`load_input_files.py:759-769`

continuous_supercategory/continuous_category are written only when the derived columns exist; the existing test supplies both, leaving the absent-label-columns branch untested. These keys are consumed downstream so the absence contract matters.

**Fix:** Add a test with manifold columns present but no vae_supercategory/vae_category, asserting those keys are absent while continuous_targets is populated.

### [LOW] correctness — Warning text says 'Complexity = 0' but masks fallback yields per-syllable count (mask=1)
`load_input_files.py:909`

Warning (909) says 'Complexity = 0', but fallback masks = np.ones(len(starts)) (993) makes total_mask_complexity = syllable count (1020) and mean_mask_complexity = 1.0 (1021), never 0.

**Fix:** Either set masks = np.zeros(len(starts)) so complexity is truly 0, or change the warning text to state complexity defaults to the syllable count (mask=1 per USV).

### [LOW] dead_code_naming — Top-level 'usv_count'/'usv_rate' mouse-dict keys are written but never read
`load_input_files.py:401-402`

find_bout_epochs assigns top-level 'usv_count'/'usv_rate' (401-402). All consumers read 'usv_rate'/'usv_event' only from the nested 'continuous_vocal_signals' sub-dict; grep finds no top-level read. The sibling functions deliberately omit these keys. Local usv_frame_events/usv_frame_rate are still needed (410, 412, 525).

**Fix:** Delete the two assignment lines while keeping the local usv_frame_events / usv_frame_rate variables.

### [LOW] docs_clarity — Typo in comment: 'Get data is target-vs-other structure'
`load_input_files.py:682`

Comment at 682 says 'is' where 'in' is meant.

**Fix:** Reword to 'Get data in target-vs-other structure' or 'Build target-vs-other event structure'.

### [LOW] docs_clarity — proportion_smoothing_sd unit described as 'bins' here but 'frames' elsewhere
`load_input_files.py:247-249`

find_bout_epochs (248) says '(in bins)'; find_usv_categories (594) and find_variable_length_bouts (864-865) say 'in frames'. Traces are frame-indexed, so 'frames' is correct.

**Fix:** Change '(in bins)' to '(in frames)' on line 248.

### [LOW] docs_clarity — find_bout_epochs Returns docstring understates per-mouse keys written
`load_input_files.py:302-305`

Returns (304) lists only positive_events/negative_events; each entry also stores 'start','stop','continuous_vocal_signals' (and the dead 'usv_count'/'usv_rate').

**Fix:** Enumerate the per-mouse keys actually exported ('start','stop','continuous_vocal_signals','positive_events','negative_events'); drop usv_count/usv_rate from code and docs together.

### [LOW] performance — Duplicate unique() computation in find_usv_categories
`load_input_files.py:691, 733`

unique_cats (691) and unique_cats_raw (733) compute the identical value on the same frame/column with no intervening mutation.

**Fix:** Reuse unique_cats in the category_streams loop instead of recomputing unique_cats_raw.

### [LOW] performance — Repeated per-category mouse_usvs.filter() passes instead of one partition_by
`load_input_files.py:691-740`

Same per-category .filter() runs in three loops (693-700, 720-730, 733-740), scanning the column ~3*K times.

**Fix:** Compute parts = mouse_usvs.partition_by(category_column, as_dict=True) once and index it in all three loops.

### [LOW] performance — Smoothed vocal trace rebuilds the binary occupancy already computed for the binary trace
`load_input_files.py:393-399`

usv_frame_events (393) and usv_frame_rate (397) call _generate_vocal_trace with identical arrays; the smoothed call re-runs the full fill loop (189-203) before convolving.

**Fix:** Split out a _build_binary_trace helper (or accept a precomputed binary trace) and convolve it for the smoothed variant.

### [LOW] tests — find_variable_length_bouts multi-bout IBI split untested
`load_input_files.py:994-999`

The IBI split producing multiple bouts (995-999) is never asserted; all tests form a single bout or one syllable.

**Fix:** Add a test with two clusters separated by a gap >= IBI threshold, min_vocalizations=2, asserting bout_onsets has two entries.

### [LOW] tests — find_bout_epochs negative-event rejection (future USV present) untested
`load_input_files.py:499-502`

The usvs_in_future==0 rejection branch (489-501) is never exercised with a USV inside a tiled onset's future window.

**Fix:** Add a bout-mode test with an uncategorized USV inside the future window of an otherwise-clean tiled onset, asserting that onset is dropped from negative_events.

### [LOW] tests — find_usv_categories noise filter with noise_column absent untested
`load_input_files.py:673-674`

The present-column path is tested; the noise_vocal_categories-given-but-column-missing no-op path is not.

**Fix:** Add a test with noise_vocal_categories=[0] and a CSV lacking noise_column, asserting all USVs survive.

### [LOW] tests — load_behavioral_feature_data missing-H5 and multi-session paths untested
`load_input_files.py:54-56`

Single happy-path test; missing-H5 failure and multi-session accumulation loop untested.

**Fix:** Add a multi-session test asserting both session keys in all three dicts, and (after a None guard) a missing-H5 test.

### [LOW] tests — _generate_vocal_trace smooth_sd==0 guard and NaN preservation untested
`load_input_files.py:205-208`

The smooth_sd>0 guard (205) is only tested for smooth_sd>0; smooth_sd==0 (returns raw binary) and preserve_nan are not exercised distinctly.

**Fix:** Add a test with smooth_sd=0.0 asserting the result equals the raw binary trace.


## `modeling_collinearity_audit.py` (21)

### [HIGH] docs_clarity — Five parameters of audit_predictor_timescales are undocumented in the Parameters section
`modeling_collinearity_audit.py:769-806`

Parameters block ends at input_metadata (806) but the signature (677-681) declares shuffle_range_seconds, event_intervals_per_session (required, raises at 1027-1033), bout_onset_times_per_session (required, raises at 1019-1026), signal_floor_seconds, signal_min_run_seconds — none documented. The two required dicts are the function's most important inputs.

**Fix:** Add Parameters entries for all five, noting the two dicts are required and what each supplies.

### [MEDIUM] correctness — ACF circular-shift null uses non-nan-aware reductions; a short session NaN-poisons the whole feature null
`modeling_collinearity_audit.py:995-997`

Line 984 gates only on acf_long_stack[f_i,s_i,0] being finite. A session shorter than acf_extended_max_lag (=max_lag_frames+shuffle_max_frames) returns acf[0]=1 (finite) but a NaN tail from _per_session_acf (lines 569-573), so it passes the gate. Lines 988-989 then copy NaN-tailed windows into pool, and lines 995-997 reduce with plain np.mean/np.percentile (not nan-aware), collapsing the entire null at any poisoned lag to NaN. RuntimeWarning is suppressed at 994 so it is silent. The display-ACF path (955-957) and the signal-correlation null (1210,1227-1236) both correctly use nan-aware reductions; the ACF null is the lone inconsistent path.

**Fix:** Use np.nanmean/np.nanpercentile at lines 995-997, or drop pool rows containing any NaN before reducing.

### [MEDIUM] correctness — Signal-correlation null slice silently wraps (negative index) when max_lag_frames > shuffle_min_frames
`modeling_collinearity_audit.py:1202`

Line 1202 reads xcorr_full[S_int - max_lag_frames : ...] with S_int >= shuffle_min_frames. The start is only guaranteed >= 0 when shuffle_min_frames >= max_lag_frames. Upstream validation in modeling_utils.py (1576-1580) only enforces 0 < shuffle_min < shuffle_max and never ties shuffle_min to max_lag. Defaults (max_lag=10s, shuffle_min=20s) are safe, but a config of e.g. max_lag=30s, shuffle_min=20s makes the start index negative; numpy then produces a wrapped slice silently, corrupting the null with no error.

**Fix:** Add a guard requiring shuffle_min_frames >= max_lag_frames (here or in the upstream validator) raising a clear ValueError; alternatively skip/clamp shifts where S_int - max_lag_frames < 0.

### [MEDIUM] performance — Per-event window means computed with a Python loop instead of a cumulative-sum sliding window
`modeling_collinearity_audit.py:148-156`

The per-event inner loop (149-154) is O(n_events*history_frames) Python work per feature; since starts=ends-history_frames (126) every window has equal width and reduces to one cumulative-sum pass. Dominant cost of _build_event_summary_matrix on dense-USV cohorts.

**Fix:** Clean the column once, compute csum=np.concatenate([[0.0],np.cumsum(col_values,dtype=np.float64)]), then window_means=((csum[ends]-csum[starts])/history_frames).astype(np.float32).

### [MEDIUM] performance — Per-shuffle signal null window gathered with a Python loop over shifts
`modeling_collinearity_audit.py:1199-1204`

The for j,S in enumerate(shifts) loop (1199-1204) runs n_shuffles (default 1000) per (feature,session); equal-width windows allow a single fancy-index gather. RNG draw is unchanged (single integers call at 1180).

**Fix:** offsets=np.arange(-max_lag_frames,max_lag_frames+1); idx=shifts[:,None]+offsets[None,:]; null_per_sess_shuffle[s_i,:,:]=(xcorr_full[idx]/denom).astype(np.float32). Guard the negative-index case.

### [MEDIUM] tests — _per_session_acf short-series NaN-pad branch is untested
`modeling_collinearity_audit.py:569-573`

The short-series all-NaN-pad branch (569-573) is never exercised; existing TestPerSessionAcf tests use long traces or hit the constant early-return. This branch feeds the ACF-null NaN-poisoning correctness issue.

**Fix:** Add test_short_series_is_nan_padded calling _per_session_acf on a length-4 non-constant trace with max_lag_frames=20; assert shape (21,), acf[0]~1.0, tail all-NaN, leading finite.

### [MEDIUM] tests — All-features-constant timescale headline else-branch is documented-but-untested (and may crash)
`modeling_collinearity_audit.py:1360-1391`

A test comment explicitly notes the all-session-constant case crashes the headline nanargmax and is not exercised. The else branch (1387-1391) producing peak_feat='-'/direction='-' is unpinned.

**Fix:** Add a test making every feature constant across all sessions; assert the audit completes with the else-branch headline, or convert the noted crash into an xfail/guard test.

### [LOW] correctness — Event-to-frame mapping inconsistent between the two audits (round vs floor)
`modeling_collinearity_audit.py:125`

_build_event_summary_matrix uses np.round(ev_times*fps) (line 125); _binary_event_trace uses np.floor(event_times*fps) (line 658). Each audit is internally consistent, but the same onset can land one frame apart between audits, making cross-audit comparisons off by up to one frame. Silent, not a crash.

**Fix:** Pick one convention and use it in both _build_event_summary_matrix (125) and _binary_event_trace (658), documenting the choice.

### [LOW] correctness — n_bouts counts all onset timestamps, including out-of-bounds ones that produced no Y impulse
`modeling_collinearity_audit.py:1282-1284`

n_total_bouts is incremented by the raw size of bout_onset_times_per_session[sess_id] (1282-1284), but _binary_event_trace filters onsets to (idx>=0)&(idx<n_frames) (line 659). Out-of-bounds onsets contribute no impulse, so reported n_bouts/n_events can exceed the impulses that actually drove the result.

**Fix:** Count only in-bounds onsets (mirror the 0<=idx<n_frames filter) when accumulating n_total_bouts.

### [LOW] dead_code_naming — Payload key 'rho_signal_per_session_mean' is written but never read by any consumer
`modeling_collinearity_audit.py:1324`

The cohort mean is stored under both 'rho_signal' (1323) and 'rho_signal_per_session_mean' (1324), and again in the empty branch (861-862). Plotting consumers (visualizations/modeling_plots.py:5523,6129,6592) read only payload['rho_signal']; nothing reads 'rho_signal_per_session_mean'.

**Fix:** Drop the redundant key, or add a one-line note it is a self-documenting alias for external readers.

### [LOW] docs_clarity — signal_floor_seconds / signal_min_run_seconds are stored but never used here, with no note saying so
`modeling_collinearity_audit.py:680-681, 870-871, 1336-1337`

Declared (680-681), stored into payload (870-871, 1336-1337), but never used in any computation in this function (they gate a marker in modeling_plots.py downstream). No note marks them stored-but-unused-here, unlike configured_filter_history (778-780).

**Fix:** Add a short note that they are recorded for downstream plot/provenance use only and do not gate computation here.

### [LOW] docs_clarity — Docstring attributes wrapper-side kwargs to this function
`modeling_collinearity_audit.py:708-715`

Lines 708-715 list bout_onset_event_key, precomputed_bout_onset_times, bout_onset_times_per_session as 'The kwarg names', but the first two are wrapper kwargs in run_predictor_audits (modeling_utils.py), not parameters of this function; only bout_onset_times_per_session is local.

**Fix:** Reword to distinguish the wrapper kwargs from this function's local kwarg.

### [LOW] performance — NaN check/replacement repeated per event window instead of once per column
`modeling_collinearity_audit.py:152-153`

np.isnan(chunk).any()/np.nan_to_num run per window (152-153) though a column's NaN status is window-independent; cleaning once removes the redundant scan and enables the cumsum vectorization.

**Fix:** Apply col_values=np.nan_to_num(col_values,nan=0.0) once after line 148 and drop the per-window branch.

### [LOW] performance — ACF null pool filled with a nested Python loop over shifts
`modeling_collinearity_audit.py:988-997`

The for S in shifts loop (988-990) can be a single fancy-index gather; same width per window. Cheaper than the signal null but the same vectorization applies.

**Fix:** offsets=np.arange(max_lag_frames+1); idx=shifts[:,None]+offsets[None,:]; pool[pool_idx:pool_idx+n_shuffles,:]=acf_long_stack[f_i,s_i][idx]; advance pool_idx.

### [LOW] tests — _integrated_autocorr_time acf.size < 2 branch is untested
`modeling_collinearity_audit.py:600-601`

The 'or acf.size < 2' clause (line 600) is never hit; existing tests trip the finiteness guard or use long ACFs.

**Fix:** Add a test asserting nan for np.array([1.0]) and np.array([]).

### [LOW] tests — _integrated_autocorr_time no-sign-change (sum-to-end) branch is untested
`modeling_collinearity_audit.py:603-605`

The cutoff=acf.size fallback (line 604) when the ACF never crosses <=0 is untested; existing tests all cross zero.

**Fix:** Add a test with a strictly-positive monotone-decreasing ACF and assert tau == 1.0 + 2.0*sum(tail).

### [LOW] tests — Positive-lag 'feature leads bout' headline direction branch is untested
`modeling_collinearity_audit.py:1381-1386`

Of the three direction branches, bout-leads (<0) and simultaneous (==0) have dedicated tests; feature-leads (>0, lines 1383-1384) does not.

**Fix:** Mirror the negative-lag test with the feature shifted EARLIER so the cross-correlation peaks at positive k; assert direction == 'feature leads bout'.

### [LOW] tests — Dropped-features '(+ N more)' summary branch untested in _build_event_summary_matrix
`modeling_collinearity_audit.py:175-180`

The n_drop>6 '(+ N more)' truncation (177-178) is never exercised; the existing test drops one feature.

**Fix:** Add a session pair missing >6 distinct generic columns and assert kept list + '(+ N more)' via capsys.

### [LOW] tests — Zero-variance '(+ N more)' summary branch untested in audit_predictor_collinearity
`modeling_collinearity_audit.py:433-436`

The n_constant>6 '(+ N more)' suffix (434) is never hit; existing tests drop 1 or 4 constant columns.

**Fix:** Add a test with >6 constant feature columns; assert survivors and optionally capsys the '(+ N more)' text.

### [LOW] tests — IBI gap computation with unsorted/multi-USV/single-USV intervals only weakly covered
`modeling_collinearity_audit.py:1268-1284`

The argsort (1273), gaps.size>0 guard (1280), non-finite filter (1279), and single-USV (no gap) path are exercised only with pre-sorted evenly-spaced intervals.

**Fix:** Add a timescale test with unsorted starts/stops for one session and a single-USV session for another; assert ibi_empirical_pcts match sorted-order gaps and the single-USV session contributes to n_usvs but no gap.

### [LOW] tests — _vif_from_design LinAlgError fallback to inf is untested
`modeling_collinearity_audit.py:262-263`

The except np.linalg.LinAlgError -> inf branch (262-263) is unreached by existing tests (constant short-circuits, exact collinearity is capped).

**Fix:** Document as effectively-unreachable with a comment, or monkeypatch np.linalg.lstsq to raise and assert the VIF entry is inf.


## `model_selection.py` (20)

### [MEDIUM] correctness — Multinomial model-free baseline AUC unguarded against all-NaN/single-fold folds (NaN mean/SE poisons 1SE rule)
`model_selection.py:3207-3209`

After the Step-0 baseline loop, valid_auc = [m for m in ...['auc'] if not np.isnan(m)] can be empty when every fold's roc_auc_score raised. np.mean([]) is nan, np.std([], ddof=1) is nan, and len==0/1 makes SE nan/inf. best_current_score / best_current_se become NaN, so every subsequent 1SE accept test (best_cand_score - best_current_score) > best_cand_se evaluates False and the anchor plus every forward feature is silently rejected, terminating with an empty model. The analogous manifold baseline (4256-4263) guards with `if valid_scores.size else float('-inf')` and `... if valid_scores.size > 1 else 0.0`; this selector is inconsistent.

**Fix:** Guard as the manifold variant does: best_current_score = float(np.mean(valid_auc)) if valid_auc else float('-inf'); best_current_se = float(np.std(valid_auc, ddof=1) / np.sqrt(len(valid_auc))) if len(valid_auc) > 1 else 0.0.

### [MEDIUM] docs_clarity — Stale copy-paste log message: multinomial anchor reject says 'spatial baseline'
`model_selection.py:3384`

In multinomial_vocal_category_model_selection the anchor-rejection log prints '*** ANCHOR REJECTED: Failed to beat spatial baseline. ***'. There is no spatial concept in the multinomial selector; its baseline is the model-free marginal class-prior AUC. The wording is copied verbatim from continuous_vocal_manifold_model_selection (4483) where 'spatial baseline' is correct, and misleads anyone reading multinomial run logs.

**Fix:** Reword to match the multinomial baseline, e.g. '*** ANCHOR REJECTED: Failed to beat the model-free marginal-prior baseline. Continuing from Empty Model. ***'.

### [MEDIUM] performance — Basis projection of already-selected features recomputed per candidate (bout_parameter, sklearn)
`model_selection.py:2481-2482`

In the sklearn forward loop, X_tr_stacked = np.hstack([np.dot(x, basis_matrix) for x in trial_tr]) (and the test analogue) re-projects EVERY feature in trial_tr through basis_matrix on every candidate and fold. trial_tr = fold_base_X_tr[fold_idx] + [new feat], so the projection of the K already-selected features is identical across candidates within a step yet redone each time. RidgeCV is closed-form/cheap, so these np.dot projections are a large share of per-candidate cost: O(n_candidates * n_folds * K) where O(n_folds * K) would suffice. fold_base_X_tr already materializes raw slices once per step (2444-2447).

**Fix:** When model_type=='sklearn', precompute per step/fold fold_base_proj_tr[fold_idx] = np.hstack([np.dot(x, basis_matrix) for x in fold_base_X_tr[fold_idx]]) (and _te) alongside 2444-2447, then inside the candidate loop only project the new feature and hstack onto the cached base (handling the step-0 empty-base case).

### [MEDIUM] performance — vocal_category session-fold path re-pools and re-projects all trial features per fold per candidate (subsumes the 1726-1727 projection redundancy)
`model_selection.py:1688-1689`

In the 'session' branch, _pool_category_features(all_feature_data, trial_feats, ...) is called for both train (1688) and test (1689) on every fold of every candidate. trial_feats = current_model_features + [feat], so the K already-selected features are re-pooled (re-concatenated across sessions) per candidate. A pooled_category_cache already exists (1405-1421) and is used by the 'mixed' branch but ignored here. The selected features are then also re-projected through basis_matrix at 1726-1727, so pooling+projection of the K selected features is repeated O(n_candidates) times.

**Fix:** Reuse pooled_category_cache in the 'session' branch by precomputing per fold the row indices belonging to the train/test session subset for each feature's cached pooled arrays, then slice instead of calling _pool_category_features per candidate; at minimum pool/project only the single new feat per candidate and cache the selected-feature pooled+projected base once per step per fold (train and test).

### [MEDIUM] performance — bout_onset session-fold path re-pools anchor + every trial feature per fold per candidate, ignoring pooled_feature_cache
`model_selection.py:856-861`

pooled_feature_cache is built once (594-627) holding each feature's full pooled (pos, neg) design matrix, but is consumed only by the 'mixed' branch. The 'session' branch instead calls pool_session_arrays for the anchor (849, 856) AND every feature in trial_features (860-861) on every fold of every candidate. pool_session_arrays re-concatenates the same per-session arrays each call, so the inner body performs O(n_steps * n_features * n_folds * len(trial_features)) redundant pooling passes -- exactly what the cache removed for 'mixed'.

**Fix:** Precompute once per fold (outside the candidate loop) a session->row index mapping over each feature's cached pooled array and index pooled_feature_cache[feat] by fold rather than calling pool_session_arrays per candidate. At minimum, lift the anchor pooling (849/856) out of the per-candidate loop since the anchor train/test split is fixed for a given fold.

### [MEDIUM] tests — Resume 'Selection already converged -> Stopping loop' branch untested (bout-parameter + multinomial + manifold)
`model_selection.py:2301-2303`

The convergence-stop branch at 2295-2303 (valid checkpoint whose best candidate FAILS the (mean - se) > best_current_score promotion test, printing '[RESUME] Selection already converged. Stopping loop.' and setting step_counter = last_step) is never exercised. Existing resume tests in test_model_selection_tail.py never reference that string nor assert the stop-without-append behavior. The identical branch exists for multinomial (3112) and manifold (4150). A regression inverting the comparison would silently re-run a finished selection.

**Fix:** Add a bout-parameter resume test dropping a step pickle whose candidates_summary holds one finite candidate with mean_explained_deviance only marginally above and within one se_explained_deviance of the stored baseline_score, so (mean - se) <= baseline; assert the run stops at last_step with current_model_features unchanged. Parametrize across multinomial (mean_auc/se_auc) and manifold (mean_r2/se_r2).

### [MEDIUM] tests — get_unrolled_X_for_multivariate per-feature sample-count mismatch ValueError untested
`model_selection.py:205-206`

test_get_unrolled_X_validation_raises (test_model_selection_tail.py 1225-1237) covers only the empty-list guard (189-190) and the Frame-mismatch guard (193-194). The third guard at 205-206 -- 'Sample count mismatch at feature index {i}' -- has no test. This guard protects the column-alignment invariant of the unrolled design matrix; a regression dropping it would produce a silently misaligned X.

**Fix:** Extend test_get_unrolled_X_validation_raises with two arrays whose row counts disagree, e.g. [np.zeros((5, HISTORY_FRAMES)), np.zeros((4, HISTORY_FRAMES))], and assert pytest.raises(ValueError, match='Sample count mismatch at feature index 1').

### [LOW] correctness — Standard-error computations with ddof=1 emit NaN for a single valid fold, poisoning the 1SE accept rule
`model_selection.py:792, 943, 1639, 1798, 2420, 2555, 3362, 3559`

These SE computations np.std(valid, ddof=1) / np.sqrt(len(valid)) are guarded only by a non-empty check (if valid:), not by len > 1. With exactly one finite fold, np.std([x], ddof=1) divides by N-ddof=0 -> nan, which flows into se_ll/se_auc/se_explained_deviance and the accept test; any `> NaN` is False so a candidate that succeeded in only one fold is always rejected. The manifold selector (4261, 4458, 4585) uniformly guards with `... if valid_scores.size > 1 else 0.0`. Line 3209 is folded into the separate all-NaN baseline finding.

**Fix:** Apply the manifold guard uniformly: se = np.std(valid, ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0 at each site.

### [LOW] correctness — pyGAM `converged` diagnostic derived from last deviance diff is inconsistent with the LogisticRegression path in the same module
`model_selection.py:771, 923, 1606, 1761, 2370, 2508`

The pyGAM per-fold flag is bool(gam_diffs and gam_diffs[-1] < gam_kwargs['tol']) -- reports converged whenever the LAST recorded diff is below tol, a possible false positive for a fold that exhausted max_iter. The LogisticRegression branch in the same function uses the iteration-count test fold_converged = bool(fold_n_iter < lr_params['max_iter']) (1590, 1745). The docstrings (442-444, 2013-2015) advertise converged=False as the audit signal for max_iter terminations, so the diff-based variant defeats the stated purpose. Mislabels a diagnostic field only.

**Fix:** Derive convergence from the iteration count to match the LR path, e.g. converged = bool(len(gam_diffs) > 0 and len(gam_diffs) < gam_kwargs['max_iter']).

### [LOW] correctness — Quantile binning collapse to a single bin silently disables stratification with no warning
`model_selection.py:2185-2186`

n_bins = max(2, min(10, len(y_global) // n_splits)); bins = np.unique(np.percentile(...)). For right-skewed targets with many tied values np.unique can collapse to fewer than 2 distinct edges, in which case y_binned is set to all zeros (2186) and StratifiedShuffleSplit/StratifiedGroupKFold silently degrade to unstratified splitting. Downstream _run_metadata/console still report split_strategy as if stratification were honored, so the degradation is invisible.

**Fix:** Emit a warning when len(bins) < 2 (single-bin fallback) so the loss of stratification is visible in the run log, mirroring other explicit warnings in this module.

### [LOW] dead_code_naming — Docstring + inline comment reference a nonexistent recompute_filter_shapes utility; the 'two code paths' justification is false
`model_selection.py:236-240`

The docstring of compute_filter_shapes_per_fold_bout_onset (236-240) and the inline comment (994-996) claim a recompute_filter_shapes legacy-pickle recovery utility also calls this helper, justifying its extraction. A repo-wide grep finds recompute_filter_shapes ONLY in these two strings -- no definition exists. The function has exactly one caller (1012), so the 'guarantees both code paths exercise the same fitting' rationale is untrue.

**Fix:** Reword the docstring (236-240) and comment (994-996) to drop the recompute_filter_shapes reference and state the helper has a single caller (the bout-onset final-refit block), kept factored out for readability.

### [LOW] docs_clarity — Non-obvious chance_ll = np.log(2) lacks an explanatory comment
`model_selection.py:459, 1238`

chance_ll = np.log(2) is the binary-classifier chance log-loss (balanced 50/50 coin) used as the initial best_current_score baseline in bout_onset_model_selection (459) and vocal_category_model_selection (1238), but neither site explains why log(2) is the chance floor.

**Fix:** Add a short inline comment at both sites, e.g. chance_ll = np.log(2)  # mean log-loss of a balanced 50/50 coin: the chance floor.

### [LOW] docs_clarity — Parameters section uses `name (type)` instead of the file's `name : type` NumPy style
`model_selection.py:265-303`

compute_filter_shapes_per_fold_bout_onset writes its Parameters entries as `cv_folds (list of dict)`, `current_model_features (list of str)`, etc., whereas every other docstring in this file uses NumPy `name : type` (e.g. 71, 121, 163, 1081). This is the only docstring that diverges, rendering inconsistently under NumPy-doc tooling.

**Fix:** Convert the parameter headers to `name : type` form, e.g. `cv_folds : list of dict`, to match the rest of the file.

### [LOW] docs_clarity — Multinomial step-dict doc: baseline_score description omits its Step-0 meaning
`model_selection.py:2727`

The data-persistence docstring states 'baseline_score' (float): The AUC of the current_features model. At Step 0 current_features is the empty list and the persisted baseline_score is the model-free marginal-prior AUC (null_model_free), not the AUC of any feature model, which the description does not cover.

**Fix:** Clarify, e.g. 'baseline_score' (float): macro OvR AUC of the current_features model; at Step 0 this is the model-free marginal-prior baseline AUC.

### [LOW] performance — Already-selected feature blocks re-hstacked per candidate (multinomial)
`model_selection.py:3432-3433`

X_tr_stacked = np.hstack([binned_data[f][tr_idx_model] for f in trial_feats]) (and the test analogue) rebuilds the wide design matrix for all trial_feats on every candidate and fold. The columns for current_model_features are identical across candidates within a step (tr_idx_model is seeded by random_seed + fold_idx + 1, deterministic per fold per 2913-2917), so the fancy-index + hstack of those K blocks is repeated (n_candidates - 1) extra times per fold. Secondary to the JAX fit cost but pure redundant array materialization in the hottest loop.

**Fix:** Per step and fold, cache base_tr = np.hstack([binned_data[f][tr_idx_model] for f in current_model_features]) (and base_te) once, then build X_tr_stacked = base_tr if not base else np.hstack([base_tr, binned_data[feat][tr_idx_model]]) inside the candidate loop.

### [LOW] performance — Already-selected feature blocks re-hstacked per candidate (manifold)
`model_selection.py:4515-4516`

X_tr_stacked = np.hstack([binned_data[f][tr_idx] for f in trial_feats]) and X_te_stacked rebuild the full stacked design for every candidate and fold. tr_idx/te_idx are fixed per fold and the selected-feature columns are constant across candidates within a step, so the index+hstack of the K selected-feature blocks is recomputed per candidate. Dominant cost is the JAX fit / inner-CV tuner, so secondary, but the redundant large-array hstacks add memory traffic in the innermost loop.

**Fix:** Cache the stacked selected-feature base (np.hstack over current_model_features) once per fold at the start of each step, then concatenate only the new feat block inside the candidate loop instead of re-stacking the whole trial_feats list.

### [LOW] tests — _harvest_upstream_metadata Level-2-preference / Level-1-fallback / legacy-None branches untested
`model_selection.py:54-105`

_harvest_upstream_metadata (used only by bout_onset_model_selection) has several distinct branches never directly unit-tested: Level-2 _input_metadata preferred (86-87), Level-2 absent -> Level-1 fallback (91-92), both present -> Level-1 dropped (94), neither present -> legacy (None, ...), and _run_metadata present/absent (97-98). The bout-onset pipeline tests exercise the happy path implicitly but never assert the returned tuple or the legacy-None fallback; test_modeling_metadata.py tests inject_metadata/extract_metadata_blocks, not this wrapper.

**Fix:** Add a direct unit test importing _harvest_upstream_metadata: univariate_data {'_input_metadata': A, '_run_metadata': R, 'feat': ...} and input_data {'_input_metadata': B, 'feat': ...}; assert it returns (A, R), reserved keys popped, 'feat' remains. Add a Level-1-only case asserting (B, None) and a legacy case asserting (None, None).

### [LOW] tests — _make_step_wrapper None-block short-circuit untested
`model_selection.py:135-141`

_make_step_wrapper returns a closure that always injects _run_metadata but only adds _input_metadata / _univariate_metadata when the respective upstream block is not None (137-140). The legacy-artifact path (input_md and/or univariate_md None) is never directly asserted; pipeline tests always supply both blocks. A regression unconditionally injecting None would write a None-valued reserved key that downstream consolidation asserts against.

**Fix:** Add a direct unit test: wrap = _make_step_wrapper(None, None, {'k': 'v'}); out = wrap({'feat': 1}); assert '_run_metadata' in out and '_input_metadata' not in out and '_univariate_metadata' not in out, and the input is not mutated. Add a non-None case asserting all three keys present.

### [LOW] tests — _balance_multivariate_arrays insufficient-data sentinel return untested
`model_selection.py:1147-1148`

_balance_multivariate_arrays returns the sentinel ([], [], None, None) when either class is empty (1147-1148). The category selector relies on this sentinel via 'if not X_tr_t' guards (e.g. 1711). The forced-failure category test drives the fit-exception path, not this empty-class sentinel; there is no direct unit test. A regression returning a wrong-shaped tuple would break the 'if not X_tr_t' continue checks.

**Fix:** Add a direct unit test: X_targ_list=[np.zeros((0, HISTORY_FRAMES))], X_other_list=[np.zeros((5, HISTORY_FRAMES))] -> assert exact ([], [], None, None). Add a balanced case (5 vs 8 rows) asserting equal-length arrays of length 5 and y ones/zeros of length 5.

### [LOW] tests — _pool_category_features missing-session and empty-feature fallback untested
`model_selection.py:1101-1108`

_pool_category_features skips sessions absent from a feature's dict (the 'if sess in all_feature_data[feat]' guard at 1103) and emits an empty (0, history_frames) array when a feature gathered no sessions (1107-1108). Neither branch is directly tested; the category pipeline tests always populate every (feature, session) cell.

**Fix:** Add a direct unit test where 'featB' is missing 'session_1'; assert featB's returned arrays have fewer rows than featA. Add a case requesting a feature whose session_list contains only ids absent from its dict and assert shape (0, history_frames).


## `modeling_vocal_categories_binomial.py` (16)

### [HIGH] correctness — Uncaught TypeError when target mouse has no post-filter USVs (target_events/other_events is None)
`modeling_vocal_categories_binomial.py:438-442, 471`

In find_usv_categories (load_input_files.py) each mouse entry is initialized with 'target_events': None / 'other_events': None (lines 659-660). If a mouse has zero USVs after the noise/history filters, the loop hits 'if mouse_usvs.height == 0: continue' (lines 679-680) BEFORE target/other events are assigned, leaving them None. At lines 439-440 here, target_times/other_times are read for the target mouse t_name; the surrounding try/except only catches KeyError, but the key IS present with value None. The None flows into _collect_category_windows at line 471, where np.round(None * sampling_rate) raises TypeError. That exception is NOT caught (only KeyError is at line 441), so a single session where the target mouse never vocalized crashes the entire epoch-extraction loop and aborts extract_and_save_category_input_data with no output pickle.

**Fix:** After reading target_times/other_times, guard against None ('if target_times is None or other_times is None: continue', or coerce to np.empty(0)), or broaden the except at line 441 to 'except (KeyError, TypeError): continue'.

### [MEDIUM] correctness — roc_auc_score raises (and silently drops the whole fold) when a test fold is single-class
`modeling_vocal_categories_binomial.py:875`

Test folds preserve the natural class prior and are not guaranteed to contain both classes. In 'session' strategy a held-out session can contain only target or only other epochs; in 'mixed' a tiny minority class can land entirely in train. When y_te has one unique label, roc_auc_score(y_te, y_prob) at line 875 raises ValueError. Because AUC is the first metric in the block and the block sits under the broad try/except at line 905, the exception aborts ALL remaining per-fold metrics for that split (brier, mcc, score, recall, f1, confusion_matrix, n_iter, converged, fit_time) even though they are well-defined on a single-class fold. The fold is left entirely NaN with only a 'Fit error' line printed.

**Fix:** Guard AUC specifically (compute only when len(np.unique(y_te)) > 1, else leave AUC NaN) so the remaining single-class-valid metrics for that fold are still recorded.

### [MEDIUM] docs_clarity — deviance_explained metric missing from the per-split metrics docstring
`modeling_vocal_categories_binomial.py:696-708`

The _run_modeling_category docstring 'Metrics saved per split' enumerates auc, score, recall, f1, ll, brier, ece, mcc, confusion_matrix, and optimizer diagnostics, but omits deviance_explained, which is in the metrics list (line 740) and computed/stored per split (line 881). A reader relying on the docstring would not know this saved field exists.

**Fix:** Add a bullet for deviance_explained after the ll bullet describing it as McFadden-style 1 - LL/ln(2) relative to the balanced-trained 0.5-intercept null; the proper-scoring effect size used for the univariate plots.

### [MEDIUM] performance — Cross-validation splits regenerated twice (once per strategy) in _run_modeling_category
`modeling_vocal_categories_binomial.py:765-773`

The 'for strat in strategies:' loop (strategies = ['actual', 'null']) calls create_category_splits(feature_data, strategy='actual') on line 770 once per strategy, i.e. twice per feature. Both passes operate on the IDENTICAL real Target-vs-Other splits; the only difference is the null pass permutes the training labels per split (line 780). Regenerating the splitter for the null pass redundantly reruns pool_session_arrays, balance_two_class_arrays, concat_two_class_with_labels, shuffle_train_test_arrays per split (and the full StratifiedShuffleSplit for 'mixed'). All duplicated work scales with epoch count and runs for every feature.

**Fix:** Materialize splits once before the strategy loop: splits = list(self.create_category_splits(feature_data, strategy='actual')), then iterate over the cached splits, applying the label permutation to a copy of y_tr only for the null pass. Halves the pooling/balancing/shuffling cost without changing results.

### [MEDIUM] tests — _collect_category_windows interior-NaN zeroing and end-of-recording bounds clip are untested
`modeling_vocal_categories_binomial.py:103, 108-112`

The slicer's core documented behavior is untested. (1) The within-window NaN-to-0.0 replacement (line 111) is never exercised: the only window test (test lines 1268-1282) feeds times=np.zeros(5) that trip the empty guard at lines 104-105 and return before the slicing loop; extraction smoke tests assert finite on NaN-free synthetic traces. (2) The valid_mask end clip (ends <= max_frame_idx at line 103) and the mixed valid/invalid case (n_valid_events < n_events) are never tested; only the starts>=0 all-invalid side is.

**Fix:** Add a direct unit test with column_data containing an interior NaN inside an otherwise-valid window (col[100]=nan; window spanning index 100) asserting 0.0 at that position and preserved neighbours; and a test mixing in-bounds, off-the-start, and off-the-end events asserting the returned row count equals only the valid events and surviving rows match expected slices.

### [MEDIUM] tests — sklearn actual-branch coefs_projected / optimal_C results never asserted
`modeling_vocal_categories_binomial.py:826-827, 759-760`

The sklearn engine writes results['actual']['coefs_projected'][split_idx] (line 826) and results['actual']['optimal_C'][split_idx] (line 827), allocated at lines 759-760. No test asserts these keys exist or are populated: test_dispatch_category_sklearn_writes_per_feature_pickles checks ll/auc/score/brier/ece/mcc/f1/recall and filter_shapes (test lines 751-756) but not coefs_projected or optimal_C, so a regression dropping or mis-shaping the back-projection would pass silently.

**Fix:** Assert payload[feat_key]['actual']['coefs_projected'].shape == (split_num, n_bases) and ['optimal_C'].shape == (split_num,), and that at least one fitted fold leaves a finite value in each.

### [MEDIUM] tests — deviance_explained metric never asserted
`modeling_vocal_categories_binomial.py:881`

deviance_explained (computed at line 881 as 1 - LL/ln(2)) is in the metrics list at line 740 and is the documented proper-scoring effect size used for univariate significance. No test references it: the metric loops in the dispatcher / pygam engine tests (test lines 751, 840) enumerate ll/auc/score/brier/ece/mcc/f1/recall but omit deviance_explained, so its presence, shape, and finiteness are unverified.

**Fix:** Add 'deviance_explained' to the metric tuples asserted in test_dispatch_category_sklearn_writes_per_feature_pickles and test_run_modeling_category_pygam_engine, checking shape == (split_num,) and that finite folds yield a finite value.

### [MEDIUM] tests — Per-split confusion_matrix and optimizer diagnostics (n_iter/converged/fit_time) untested
`modeling_vocal_categories_binomial.py:892-897, 818-823, 852-854`

results[key]['confusion_matrix'] (line 892, shape (n_splits,2,2)), n_iter (line 895), converged (line 896), fit_time (line 897) — documented silent-failure detectors (lines 709-711, 746-753) — are not asserted anywhere. This also covers the sklearn n_iter_ diagnostic except branch (lines 818-823) and the pygam logs_-derived n_iter/converged path (lines 852-854), none of which are exercised. A regression that stopped recording converged (so non-converged folds at max_iter went undetected) or mis-shaped the confusion matrix would pass silently.

**Fix:** Extend a dispatcher/engine test to assert confusion_matrix.shape == (split_num, 2, 2) (integer 2x2 summing to test-fold N on finite folds) and that 'converged'/'n_iter'/'fit_time' arrays exist with shape (split_num,) and carry finite values on fitted folds; ideally add a pygam case with empty logs_['diffs'].

### [LOW] correctness — Project-wide USV grand totals computed from a single anchor feature, not all features
`modeling_vocal_categories_binomial.py:480-483, 498-500`

all_sessions (line 480), total_target/total_rest (lines 482-483) are derived only from final_data[first_feat], where first_feat = final_features[0] (alphabetically first key). These are printed as 'Total target USVs', 'Total other USVs', and 'Grand total USVs (N)' (lines 498-500). Vocal-syntax features (e.g. other.usv_cat_*) exist only in sessions where the partner vocalized, so the alphabetically-first feature is not guaranteed present in every session; when it is not, the printed grand total under-counts. Display-only (does not affect saved arrays or modeling).

**Fix:** Derive the project-wide totals from a feature known present in every session (a core kinematic feature) or take the union of per-session counts across features, rather than from the first sorted key.

### [LOW] correctness — ECE uses unclipped probabilities while log-loss uses clipped ones
`modeling_vocal_categories_binomial.py:873`

y_prob_clipped (line 872) is clipped to [1e-15, 1-1e-15] and used for log_loss (line 876), but y_proba_2d (line 873) is built from the raw unclipped y_prob and passed to expected_calibration_error (line 888). For the pygam path y_prob is a mean of per-frame probabilities and is not clipped anywhere; an exact 0.0/1.0 mean is possible, making 1.0 - y_prob exactly 1.0/0.0 in the column-stack. Does not crash ECE but is a minor inconsistency in which probability vector feeds calibration vs. log-loss.

**Fix:** Build y_proba_2d from y_prob_clipped (np.column_stack([1.0 - y_prob_clipped, y_prob_clipped])) so both the proper-scoring metric and the calibration metric see the same bounded probabilities.

### [LOW] dead_code_naming — Pre-loop pred_idx/targ_idx initialization is dead and relies on loop-trailing values downstream
`modeling_vocal_categories_binomial.py:266-267`

pred_idx/targ_idx are initialized at lines 266-267, then unconditionally overwritten inside the session loop by resolve_mouse_roles (lines 276-283). After the loop they hold the LAST session's values and are used at lines 314-315, 380-381, 436-437. targ_idx = abs(pred_idx - 1) at line 267 is immediately overwritten before any use (pure dead code). Currently safe only because model_predictor_mouse_index is session-invariant; if role resolution ever became session-dependent the trailing values would be silently wrong for the metadata/extraction phase.

**Fix:** Compute pred_idx/targ_idx once explicitly before the loop (they are constant) and drop the per-iteration reassignment, or assert they are constant across sessions; remove the redundant targ_idx = abs(pred_idx - 1) at line 267.

### [LOW] dead_code_naming — Redundant alias key = strat
`modeling_vocal_categories_binomial.py:771`

Inside _run_modeling_category, line 771 assigns key = strat and from that point key and strat are always identical (strat drives the checks at 776, 825, 863; key only indexes results[...]). The two names never diverge, so key is a pure alias of the loop variable strat, which reads as if they could differ.

**Fix:** Drop key = strat (line 771) and index results[strat][...] directly throughout the loop body, or keep a single name. No behavior change.

### [LOW] docs_clarity — Comment says macro-F1 but the code computes binary F1
`modeling_vocal_categories_binomial.py:738`

The comment justifying dropping precision (line 738) states macro-F1 already summarizes the precision/recall trade-off. The F1 actually computed at line 885 uses average='binary', and the docstring at line 701 correctly describes f1 as binary F1. The word macro-F1 is stale and contradicts both the code and the docstring.

**Fix:** Change macro-F1 to binary F1 in the comment so it matches average='binary' at line 885 and the docstring at line 701.

### [LOW] docs_clarity — Misleading warning text on the ECE (calibration) except handler
`modeling_vocal_categories_binomial.py:890`

The except block guarding ECE prints '[warn] fold diagnostic metric could not be recorded' (line 890), the exact message copied from the optimizer-diagnostic block at lines 821-823 where it refers to n_iter/converged. Here it guards expected_calibration_error, a calibration metric, not an optimizer diagnostic, so logs will mislead readers into thinking a convergence diagnostic failed.

**Fix:** Reword to identify the metric, e.g. print(f"[warn] ECE (calibration) metric could not be recorded: {e}").

### [LOW] docs_clarity — No comment explaining the per-frame label expansion / prediction averaging in the pyGAM path
`modeling_vocal_categories_binomial.py:836, 860`

In the pyGAM engine, y_tr_gam = np.repeat(y_tr, self.history_frames) expands each window-level label to one per frame (line 836) so the GAM fits long-form per-frame rows from unroll_history_matrix, and the per-frame probabilities are averaged back to one per window at line 860 (np.mean(y_prob_frame.reshape(X_te.shape), axis=1)). This frame-level fit / window-level aggregation is non-obvious and load-bearing yet has no explanatory comment.

**Fix:** Add a brief comment near line 836 (and/or 860) noting the pyGAM engine fits per-frame rows by repeating each window label across its history_frames samples, then averages per-frame predicted probabilities back to one per-window probability for metric computation.

### [LOW] performance — Python loop with per-row .copy() in _collect_category_windows can be vectorized
`modeling_vocal_categories_binomial.py:109-112`

The window-gathering loop iterates over every valid event in Python, slicing column_data[s:e].copy() and replacing NaNs row-by-row (lines 109-112). Since all windows have fixed length history_frames and valid_ends = valid_starts + history_frames (contiguous fixed-width slices), the gather can be a single vectorized fancy-index. This runs per feature x per session x (target + other events), so per-event overhead and per-row copies accumulate.

**Fix:** Build the index matrix once and gather in one shot: idx = valid_starts[:, None] + np.arange(history_frames)[None, :]; out = column_data[idx]; then np.nan_to_num(out, copy=False) (or out[np.isnan(out)] = 0.0). Replaces the Python loop and N per-row .copy() calls with one vectorized gather plus one NaN replacement, preserving the same shape and dtype.


## `modeling_vocal_onsets.py` (15)

### [MEDIUM] docs_clarity — Contradictory recall docstring citing wrong F1 variant
`modeling_vocal_onsets.py:765-766`

The recall field doc says 'kept because macro-F1 already summarizes the precision/recall trade-off' — a justification to drop recall, not keep it — and references macro-F1 although f1 is computed with average='binary' (lines 867/929). The pygam doc at 1028 phrases the same field correctly.

**Fix:** Reword to e.g. 'recall : positive-class recall (sensitivity); retained alongside binary F1 because it isolates the false-negative rate F1 blends with precision.'

### [MEDIUM] docs_clarity — sklearn Returns docstring omits deviance_explained field
`modeling_vocal_onsets.py:758-782`

_make_branch includes 'deviance_explained' (813), populated at 889 as the primary effect size for significance ranking (comment 884-888), yet the Returns bullet list (758-782) never documents it. The field downstream model selection screens on is undocumented in the public docstring.

**Fix:** Add a deviance_explained bullet describing the McFadden-style 1 - LL/ln(2) effect size used as the significance basis.

### [MEDIUM] docs_clarity — pygam Returns docstring omits deviance_explained field
`modeling_vocal_onsets.py:1024-1036`

pygam branch dict includes 'deviance_explained' (1082), populated at 1164 and 1241, but the Returns bullet list (1024-1036) omits it.

**Fix:** Add a deviance_explained bullet to the pygam Returns list matching the sklearn one.

### [MEDIUM] performance — Full-data pooling at 636 is wasted work in the 'session' split strategy
`modeling_vocal_onsets.py:635-638`

pool_session_arrays over ALL sessions (636) plus n_pos_total/n_neg_total (637-638) are consumed only by the 'mixed' branch (647, 642-643/649-650). The 'session' branch (671-707) re-pools per train/test subset at 688-689 and never references them, so for split_strategy=='session' line 636 runs a full np.concatenate of the feature's entire positive+negative data per feature per engine call and discards it.

**Fix:** Move the pool_session_arrays call and n_pos_total/n_neg_total computation inside the 'if split_strategy == "mixed":' block.

### [LOW] correctness — Epoch loop reuses module-level mouse indices instead of re-resolving per session
`modeling_vocal_onsets.py:440-441`

In the epoch-extraction loop, predictor_mouse_name/target_mouse_name at 440-441 are indexed with predictor_mouse_idx/target_mouse_idx, which carry the last value left by the per-session resolve_mouse_roles loop (228-235); the audit (424-425) and harmonize (283-284) also use that post-loop value. Safe today only because resolve_mouse_roles returns the same fixed indices every session (driven by model_predictor_mouse_index); would silently mislabel self/other if role assignment ever became session-dependent.

**Fix:** Re-resolve via resolve_mouse_roles(...) per session inside the 438 loop and use its returned indices, or add an explicit assertion/comment documenting that these indices are session-invariant by contract.

### [LOW] correctness — Substring membership test for vocal_signal_columns_added metadata is semantically wrong
`modeling_vocal_onsets.py:398`

c.split('.',1)[-1] in (voc_settings['usv_predictor_type'] or '') is a substring test, not equality. With the default usv_predictor_type='categories_rate' the real suffixes (usv_event/usv_rate/usv_cat_N from build_vocal_signal_columns) are not substrings of it, so this first clause contributes nothing and the OR clause at 399 does all the work; the intent is clearly equality and the substring form is misleading/fragile across configs. Provenance-metadata only, cannot corrupt modeled arrays.

**Fix:** Compare suffixes against an explicit expected-vocal-suffix set with equality rather than substring membership against the predictor-type string.

### [LOW] correctness — history_frames can silently become 0 with no fail-fast guard
`modeling_vocal_onsets.py:116`

history_frames = int(np.floor(camera_rate * filter_history_sec)) becomes 0 if filter_history is sub-frame, yielding (0, 0) window arrays and degenerate empty downstream results rather than a clear error. Default filter_history is 4s so this requires misconfiguration, but there is no guard.

**Fix:** After computing self.history_frames in __init__, raise a clear ValueError if it is < 1.

### [LOW] dead_code_naming — Unused loop control variable session_idx (ruff B007)
`modeling_vocal_onsets.py:438`

enumerate index session_idx at line 438 is never referenced in the loop body (439-497); the enumerate wrapper is vestigial. Confirmed by ruff B007.

**Fix:** Drop the enumerate wrapper: for beh_session_id in tqdm(processed_beh_feature_data_dict.keys(), desc='Extracting Epochs'):

### [LOW] docs_clarity — Typo 'Tho code' in pygam docstring
`modeling_vocal_onsets.py:987`

'Tho code covers the following steps:' should read 'The code'.

**Fix:** Change 'Tho' to 'The'.

### [LOW] docs_clarity — Stray asterisk in comment 'Balance the *training set ONLY'
`modeling_vocal_onsets.py:691`

Unmatched asterisk makes the comment read as broken markup; the parallel comment at 658 is clean.

**Fix:** Reword to '# Balance the TRAINING set ONLY (test fold keeps the natural class prior).'

### [LOW] performance — Partial-dependence grids rebuilt every split despite being loop-invariant
`modeling_vocal_onsets.py:1143-1144`

grid_X_0/grid_X_1 (1143-1144) and grid_X_0_null/grid_X_1_null (1218-1219) depend only on history_frames and time_indices (set at 1042/1094), are identical across all splits and both branches, yet rebuilt each split. Arrays are tiny; free readability-neutral hoist.

**Fix:** Build the two grids once just after time_indices (line 1094) and reuse them in both the actual and null branches.

### [LOW] tests — Untested: onset_target_category warning branch in non-individual mode
`modeling_vocal_onsets.py:325-328`

The branch where onset_target_category is set but model_target_vocal_type != 'individual' (warning printed 326-328, bare analysis_tag at 348, no onset_target_category injected into analysis_specific) is never exercised; only the active-category 'individual' path (test at test_pipeline_onsets_extra.py:689) is covered.

**Fix:** Add a test with model_target_vocal_type='bout' and onset_target_category=1 asserting the filename lacks 'cat_', analysis_specific has no onset_target_category, and optionally the warning is printed (capsys).

### [LOW] tests — Untested: intra-session alignment-FAILED branch
`modeling_vocal_onsets.py:509-553`

alignment_passed=False / mismatched_sessions append (521-524) and the 'FAILED (False)' / 'Dimensional mismatch' prints (549-552) are never reached; grep finds no test asserting those strings. All tests drive the passing path.

**Fix:** Refactor the alignment check into a helper returning (alignment_passed, mismatched_sessions) and unit-test it with a ragged dict, or monkeypatch window collection to force a mismatch and assert the alert via capsys.

### [LOW] tests — Untested: out-of-range gmm_component_index -> NaN ibi_thresholds in onset pipeline
`modeling_vocal_onsets.py:360-368`

The gmm_idx >= len(params['means']) -> float('nan') branch (368) is untested for VocalOnsetModelingPipeline; only the in-range float(_calculate_ibi_threshold(...)) branch (362-366) runs. The multinomial analog is explicitly covered (test_pipeline_multinomial.py:1441 test_extraction_out_of_range_gmm_index_writes_nan_ibi).

**Fix:** Mirror the multinomial test: gmm_component_index larger than len(gmm_params['male']['means']), then assert np.isnan on ibi_thresholds male/female in the saved pickle.

### [LOW] tests — Untested: basis_matrix row-count assertion in sklearn runner
`modeling_vocal_onsets.py:793-797`

The assert basis_matrix.shape[0] == self.history_frames failure path (793-797) is never triggered; TestSklearnRunner always passes np.eye(HISTORY_FRAMES) (test_pipeline_onsets_extra.py:377,402).

**Fix:** Add a test passing basis_matrix=np.eye(HISTORY_FRAMES + 1) and assert pytest.raises(AssertionError, match='diverged').


## `manifold_metric.py` (15)

### [MEDIUM] docs_clarity — Comment states wrong torus_embed component ordering (sines-first vs actual cosines-first)
`manifold_metric.py:627`

Line 627 calls the sines-first layout (s_1,s_2,c_1,c_2) 'the torus_embed ordering', but torus_embed (line 722) is concat([cos, sin]) = (c_1,c_2,s_1,s_2), cosines-first. The stated layout is the swap of the actual one; the label is a factual error.

**Fix:** State once that torus_embed is cosines-first (c_1,c_2,s_1,s_2) and the CNN interleaves per-axis (s_1,c_1,s_2,c_2); fix line 627 so it no longer calls the sines-first block 'the torus_embed ordering'.

### [MEDIUM] performance — Three full (n,n) temporary arrays materialized in distance_correlation reductions
`manifold_metric.py:247-249`

(a_c*b_c).mean(), (a_c*a_c).mean(), (b_c*b_c).mean() each allocate a ~50 MB (n=2500) product before reducing; runs n_rep=3 times per bundle. einsum fusion removes the temporaries with identical numerics.

**Fix:** dcov2 = float(np.einsum('ij,ij->', a_c, b_c))/a_c.size; dvar_a = float(np.einsum('ij,ij->', a_c, a_c))/a_c.size; dvar_b = float(np.einsum('ij,ij->', b_c, b_c))/b_c.size.

### [MEDIUM] tests — Weighted (non-uniform) branch of total_dispersion is never tested for correctness
`manifold_metric.py:570-577`

Only the ValueError guard (574-575) is hit by test_zero_weight_sum_raises; the actual weighted summation and the documented uniform==None invariant (the *len(Y) rescale) are unasserted. Confirmed against the test file.

**Fix:** Add a test asserting the weighted value matches an independent computation and that weights=ones reproduces the weights=None result.

### [MEDIUM] tests — Torus weighted circular_mean branch is untested
`manifold_metric.py:518-523`

Torus + non-uniform weights (sin/cos weighted by w[:,None], lines 519-520) is never exercised; weighted test is euclidean-only, torus tests are uniform-only. Confirmed against the test file.

**Fix:** Add a seam-straddling weighted test asserting the torus centroid lands near the weighted cluster.

### [MEDIUM] tests — euclidean_mae_weighted output value is never verified with real weights
`manifold_metric.py:441-442`

Every TestManifoldPredictionMetrics call omits weights (defaults uniform); the non-uniform weighting and the +1e-12 denom are unasserted. Confirmed.

**Fix:** Add a test with explicit non-uniform weights asserting the value against an independent computation and that it differs from euclidean_mae.

### [MEDIUM] tests — dcor_prediction_truth mismatched-length ValueError is untested
`manifold_metric.py:321-326`

The length-mismatch guard has no covering test; all dcor tests pass paired equal-length arrays. Confirmed.

**Fix:** Add with pytest.raises(ValueError) on mismatched-length inputs.

### [LOW] correctness — _spear except-ValueError branch is dead; constant/degenerate input returns nan from spearmanr, not a raise
`manifold_metric.py:420-424`

Confirmed empirically: spearmanr([1,1,1],[1,2,3])[0] returns nan, not a ValueError. The try/except ValueError at 420-423 never fires; the np.isfinite(value) guard on line 424 handles the real nan path. The except branch is vestigial.

**Fix:** Drop the try/except ValueError and rely on np.isfinite(value), or comment that spearmanr signals degeneracy via nan return.

### [LOW] docs_clarity — Wrap range documented as half-open (-period/2, period/2] but is actually closed [-period/2, period/2]
`manifold_metric.py:85`

Confirmed: np.round round-half-to-even makes exact half-period differences land on either boundary depending on parity (+0.5 stays +0.5, +1.5 maps to -0.5). The half-open claim at lines 85, 103, 159, 192 overstates the guarantee; the true range is closed.

**Fix:** Document the closed range [-period/2, period/2] at lines 85, 103, 159, 192 (or change the fold to deterministically yield half-open).

### [LOW] docs_clarity — total_dispersion return docstring misstates the uniform-weight reduction as 'times 1/N'
`manifold_metric.py:560-563`

Both branches return the plain sum: weights=None returns np.sum(diff**2) (line 571); explicit uniform returns np.sum(w*diff**2)*len(Y) = plain sum (line 577). There is no residual 1/N; the *len(Y) cancels it. The docstring contradicts its own '# scale to match unweighted convention' comment.

**Fix:** Reword to: reduces to the plain unweighted sum of squared distances (the internal 1/N is cancelled by the * len(Y) factor).

### [LOW] performance — ss_res recomputes dx**2 + dy**2 already held in euclidean_dist
`manifold_metric.py:433`

Line 400 computes euclidean_dist = sqrt(dx**2+dy**2); line 433 recomputes the sum as np.sum(dx**2+dy**2). euclidean_dist**2 recovers it exactly.

**Fix:** ss_res = float(np.sum(euclidean_dist ** 2)).

### [LOW] performance — Grand mean computed as a separate third full-array pass in _double_center
`manifold_metric.py:244`

_double_center does M.mean(axis=0), M.mean(axis=1), and M.mean(); the grand mean equals the mean of either marginal (verified exact/near-exact), so the third O(n^2) pass is avoidable. Called twice per dcor call.

**Fix:** r=M.mean(axis=1,keepdims=True); c=M.mean(axis=0,keepdims=True); g=r.mean(); return M-r-c+g.

### [LOW] tests — r2_spatial denom<=0 fallback (constant truth -> 0.0) is untested
`manifold_metric.py:433-435`

Line 435 returns 0.0 when total_dispersion(Y_true)<=0; all fixtures use non-degenerate truth. Confirmed.

**Fix:** Add a constant-Y_true test asserting r2_spatial == 0.0.

### [LOW] tests — dcor_prediction_truth n_rep<1 -> nan branch is untested
`manifold_metric.py:330-335`

Empty vals (n_rep<1) returns float('nan') at line 335; no test passes n_rep=0. Confirmed.

**Fix:** Add a test asserting np.isnan for n_rep=0.

### [LOW] tests — _spear ValueError->nan and constant-input nan paths in manifold_prediction_metrics are untested
`manifold_metric.py:419-424`

_pearson zero-denominator (415-416) and _spear non-finite (424) nan paths are never driven by a constant-axis Y; test_density_draw uses non-degenerate data. Confirmed.

**Fix:** Add a constant-axis test asserting pearson_x and spearman_x are nan.

### [LOW] tests — mahalanobis_mae quad-clip path (np.maximum(quad,0.0)) is untested
`manifold_metric.py:403-407`

test_mahalanobis_requires_cov uses np.eye(2), which never yields negative quad, so the clip at line 407 is never exercised. Confirmed.

**Fix:** Add a near-singular train_cov_inv test asserting mahalanobis_mae finite and >= 0.


## `consolidate_model_selection_results.py` (13)

### [MEDIUM] correctness — _univariate_metadata silently dropped if the first metadata-bearing step file lacks it
`consolidate_model_selection_results.py:408-413, 429-436, 468-469`

canonical_univariate_md is bound only inside `if canonical_input_md is None` (lines 410-413), together with cur_univ. If that first metadata-bearing file omits `_univariate_metadata`, cur_univ is None (line 408), so canonical_univariate_md becomes None and is never re-populated: the cross-step branch at 429 is guarded by `canonical_univariate_md is not None`, so later files that DO carry the block are never adopted. Result: the consolidated artifact omits `_univariate_metadata` (line 468 False) even though some step files carried it, silently losing Level-2 provenance; the behavior is order-dependent.

**Fix:** Adopt the univariate block lazily: when `cur_univ is not None and canonical_univariate_md is None`, set canonical_univariate_md = cur_univ (independent of the canonical_input_md bind). Keep the equality assert guarded by both being non-None.

### [MEDIUM] docs_clarity — Module docstring describes prefix inference as 'longest common prefix' but code requires a single identical full prefix
`consolidate_model_selection_results.py:55-58`

Lines 55-58 say the prefix is inferred as the 'longest common model_selection_*_step_ prefix shared across the directory's *.pkl files'. _infer_prefix (101-145) does no longest-common computation: it captures the full group(1) prefix per file into a set and raises if the set has more than one element. 'longest common' wrongly implies partial-stem matching that does not exist.

**Fix:** Reword to: 'infers it as the single model_selection_<descriptor>_step_ prefix shared by every *.pkl file; if the directory carries more than one distinct such prefix it aborts and asks for an explicit --prefix.'

### [LOW] correctness — Mismatch diff ignores provenance keys honored by the equality check, producing noisy messages
`consolidate_model_selection_results.py:417, 424, 432`

On a mismatch the message is built from `_diff_metadata(canonical, cur)` (417/424/432), but `_diff_metadata` (272-293) takes no ignore_keys, whereas metadata_blocks_equal was called with ignore_keys=ignore_provenance_keys (applied at top level only, per modeling_metadata.py:1177/1184). So the raised diff can list git_commit/git_dirty/package_version differences that were intentionally ignored, adding confusing noise to the message.

**Fix:** Add an ignore_keys parameter to _diff_metadata (applied at the top level, mirroring metadata_blocks_equal) and pass ignore_keys=ignore_provenance_keys at the three call sites so the reported diff reflects only the keys that drove the failure.

### [LOW] docs_clarity — --prefix CLI help repeats the inaccurate 'longest common' prefix wording
`consolidate_model_selection_results.py:513-515`

argparse help for --prefix (513-515) says 'defaults to the longest common model_selection_*_step_ shared by every input file'. Same inaccuracy as the module docstring: _infer_prefix requires the full per-file captured prefix to be identical across all files, not a longest-common substring.

**Fix:** Change to 'defaults to the single model_selection_*_step_ prefix shared by every input file; aborts if the directory mixes prefixes.'

### [LOW] docs_clarity — _parse_step_idx docstring omits the repo-standard Parameters/Returns sections
`consolidate_model_selection_results.py:88-98`

_infer_prefix (102-127) and _build_default_output_filename use the full Description/Parameters/Returns(/Raises) template, but _parse_step_idx documents behavior in prose only, with no Parameters block for filename/prefix and no Returns block, despite the load-bearing -1 sentinel.

**Fix:** Add Parameters (filename: str; prefix: str) and Returns (int: the step index, or -1 when the filename does not parse against prefix) sections to match the file's other helpers.

### [LOW] docs_clarity — _file_mtime_iso docstring omits Parameters/Returns sections used by sibling helpers
`consolidate_model_selection_results.py:148-156`

The docstring describes behavior in prose but does not use the Parameters/Returns structure applied to _infer_prefix/_build_default_output_filename. The path argument (str or pathlib.Path) and the exact return format ('Z'-suffixed ISO-8601 UTC string, microseconds stripped) are noted only informally.

**Fix:** Restructure into Parameters (path: str or pathlib.Path) and Returns (str: ISO-8601 UTC mtime with a trailing 'Z', microseconds stripped) to match the other helpers.

### [LOW] tests — _parse_step_idx non-digit body and non-.pkl branches untested
`consolidate_model_selection_results.py:95-98`

test_parse_step_idx_matches_prefix (test 239-245) covers only a numeric match (==3) and a name not starting with the prefix (==-1). The non-digit-body branch (body.isdigit() False -> -1, e.g. a final-fit '..._step_final.pkl') and the non-.pkl branch (not endswith('.pkl') -> -1, e.g. a '*.tmp') are never exercised, though both are real selection-directory artifacts the consolidator must skip.

**Fix:** Add asserts: cms._parse_step_idx('model_selection_test_step_final.pkl', 'model_selection_test_step_') == -1 and cms._parse_step_idx('model_selection_test_step_3.tmp', 'model_selection_test_step_') == -1.

### [LOW] tests — Mixed legacy + metadata directory (legacy_run_seen with canonical_input_md present) untested
`consolidate_model_selection_results.py:477-480`

test_allow_legacy_writes_legacy_filename (test 454-472) uses an all-legacy directory, so canonical_input_md is None and the legacy_selection_<ts>.pkl branch (475-476) fires. The complementary branch at 477-480 -- legacy_run_seen True AND canonical_input_md is not None (a directory mixing one metadata-less step with metadata-bearing ones under allow_legacy=True) -- is never hit; it must build the default model_selection_final_ name and hoist the canonical blocks.

**Fix:** Add a test writing step_0 with full md_in/md_run and step_1 metadata-less, call cms.consolidate(..., allow_legacy=True), and assert Path(out).name.startswith('model_selection_final_') (not 'legacy_selection_') and cons['_input_metadata'] == md_in.

### [LOW] tests — move_to_steps_subdir OSError warn-and-continue branch untested
`consolidate_model_selection_results.py:499-500`

test_move_to_steps_subdir_relocates (test 501-519) only exercises the happy path where every shutil.move succeeds. The except OSError branch (499-500) that prints a WARNING and continues (so moved < total) is never covered, so a regression turning warn-and-continue into a hard failure would pass CI.

**Fix:** Add a test that monkeypatches shutil.move (in the cms module) to raise OSError, runs consolidate(..., move_to_steps_subdir=True), and asserts it returns out_path without raising and that captured output contains 'could not move' and 'Moved 0/'.

### [LOW] tests — _build_default_output_filename skip-graft branch (pre-augmented tag) untested
`consolidate_model_selection_results.py:242-255`

The graft is covered for a bare 'category_3' tag (test 300-313) and a bare 'multinomial' tag (test 314-325). The skip branch at line 245 -- cat_col not in analysis_tag evaluating False because the tag already contains the column (e.g. 'multinomial_vae_supercategory' with usv_category_column_name='vae_supercategory') -- is untested, despite being the documented 'new pipelines hand it in pre-augmented' path.

**Fix:** Add a test with analysis_tag='multinomial_vae_supercategory' and analysis_specific usv_category_column_name='vae_supercategory'; assert the tag is emitted verbatim (no doubled '_vae_supercategory').

### [LOW] tests — _build_default_output_filename with exactly one of input/run metadata absent untested
`consolidate_model_selection_results.py:257-264`

Tests cover both blocks present (281-325) and both None (294-298). The asymmetric branches are uncovered: input present but run None (split_strategy -> 'unknown', 257-261) and input None but run present (sex/cohort/analysis_tag -> 'unknown', graft block skipped). Each is a distinct conditional path.

**Fix:** Add asserts: cms._build_default_output_filename(md_in, None) ends with '_unknown.pkl' and cms._build_default_output_filename(None, {'split_strategy':'mixed'}) == 'model_selection_final_unknown_unknown_unknown_mixed.pkl'.

### [LOW] tests — Asymmetric _univariate_metadata presence (one file has block, other lacks it) untested
`consolidate_model_selection_results.py:429-436`

test_univariate_metadata_mismatch_raises (test 415-429) requires both step files to carry _univariate_metadata. The documented tolerance (comment 392-394) where one file has the block and another lacks it -- the silent no-op path via the guard at 429 -- is never asserted, so a regression that started raising on asymmetric univariate blocks would go uncaught.

**Fix:** Add a test where step_0 carries _univariate_metadata and step_1 omits it (matching _input/_run); assert consolidate succeeds and cons['_univariate_metadata'] equals step_0's block.

### [LOW] tests — _file_mtime_iso ISO-8601/'Z' formatting never directly asserted
`consolidate_model_selection_results.py:148-155`

_file_mtime_iso is only exercised indirectly via consolidate round-trips; no test asserts its output shape. The 'Z' substitution and microsecond stripping (line 155) are load-bearing for the _consolidation_metadata timestamps but unverified, so a formatting regression (losing 'Z' or leaking '+00:00') would pass.

**Fix:** Add a unit test that touches a tmp file and asserts cms._file_mtime_iso(file) matches r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'.


## `modeling_vocal_bout_parameters.py` (12)

### [MEDIUM] correctness — Stratified splitter can raise uncaught errors on degenerate quantile bins / fold counts
`modeling_vocal_bout_parameters.py:530-572`

The try/except at 531-536 wraps only the binning. The splitter .split() calls (StratifiedShuffleSplit at 547, StratifiedGroupKFold at 567) are unguarded. Skewed bout targets frequently produce duplicate percentile edges, collapsing y_binned into tiny strata; StratifiedShuffleSplit then raises 'least populated class has only N members' and StratifiedGroupKFold raises 'n_splits cannot be greater than members in each class'. The mixed branch has no try/except and the session branch only catches StopIteration, so the error aborts the univariate run for that feature.

**Fix:** After computing y_binned, clamp the effective splits/folds to the minimum class count (min_class = np.min(np.bincount(y_binned)); n_folds = max(2, min(n_folds, min_class)) and cap StratifiedShuffleSplit n_splits), or fall back to non-stratified KFold/GroupShuffleSplit when stratification is infeasible.

### [MEDIUM] correctness — Intra-session alignment failure is reported but the misaligned artifact is still saved
`modeling_vocal_bout_parameters.py:454-478`

When alignment_passed is False (412-420) the code prints an ALERT at 454-456 but then unconditionally saves the pickle at 475-478, writing a known-misaligned modeling artifact to disk that looks like a normal output.

**Fix:** Raise (or refuse to write the pickle and return early) when alignment_passed is False, rather than persisting a known-misaligned artifact.

### [MEDIUM] docs_clarity — Docstring names stale settings keys (vocal_output_type / vocal_output_partner_only)
`modeling_vocal_bout_parameters.py:115-116`

The docstring (115-116) describes settings keys vocal_output_type and vocal_output_partner_only, but the code reads usv_predictor_type and usv_predictor_partner_only (lines 152-153). A reader cross-referencing the settings JSON will look up non-existent keys.

**Fix:** Rename docstring references vocal_output_type -> usv_predictor_type and vocal_output_partner_only -> usv_predictor_partner_only.

### [MEDIUM] tests — Intra-session alignment FAILURE path is never tested
`modeling_vocal_bout_parameters.py:402-457`

The alignment-False branch (412-420 setting mismatched_features, ALERT at 454-456) is never exercised; the extraction test asserts only the PASSED case. A regression breaking per-feature bout alignment would go uncaught.

**Fix:** Refactor 402-457 into a testable helper and add a unit test with one feature deliberately short or reordered, asserting alignment_passed is False and mismatched_features contains the offending feature.

### [MEDIUM] tests — partner_only=False self-density guardrail ([PROTECTED]) never exercised in extraction
`modeling_vocal_bout_parameters.py:442, 153, 195-198`

The headline scientific guardrail (partner_only=False allows self usv_cat_* but excludes self usv_rate/usv_event; lines 153, [PROTECTED] tag 442) is never tested end-to-end; all extraction tests use the default partner_only. The session-removal deletion loop at 195-198 is also not directly asserted.

**Fix:** Add an extraction test with usv_predictor_partner_only=False and a full-syntax usv_predictor_type; assert keys include self.usv_cat_* but no self.usv_rate/self.usv_event, while other.usv_rate/other.usv_event are present.

### [LOW] dead_code_naming — Redundant exception classes in except tuple (vestigial)
`modeling_vocal_bout_parameters.py:1050`

except (ValueError, ZeroDivisionError, Exception) lists ValueError and ZeroDivisionError alongside their base Exception, so the first two are unreachable as distinct catch targets. The sibling failure branches at 753, 831, 990 all use bare except Exception, making this an inconsistent vestige.

**Fix:** Replace with except Exception as e: to match the other three failure branches.

### [LOW] docs_clarity — Sklearn evaluation docstring under-describes the metric bundle actually computed
`modeling_vocal_bout_parameters.py:878-886`

The evaluation section (878-886) mentions only Spearman and MSLE, but the method computes and stores explained_deviance (D2), residual_deviance, pearson_r, mae, rmse (lines 973-979), as fully documented in the pyGAM branch.

**Fix:** Enumerate the same metrics (D2, residual_deviance, pearson_r, mae, rmse) or cross-reference the pyGAM 'Metrics Calculated' block.

### [LOW] performance — Per-bout window validity and len(col_data) recomputed for every column
`modeling_vocal_bout_parameters.py:368-393`

The nested loop recomputes start = frame_idx - self.history_frames and the validity test start >= 0 and frame_idx <= len(col_data) inside the per-column loop, although onset_frames and history_frames are identical for every column and len(col_data) is constant per column. The parent (modeling_vocal_onsets.py 478-491) computes a per-column valid_mask vectorized once.

**Fix:** Compute starts/valid_mask/valid_starts/valid_ends and kept targets once per session before the for col loop; inside the loop only gather windows.

### [LOW] performance — Repeated np.unique over the same anchor 'groups' array
`modeling_vocal_bout_parameters.py:424-425, 436-437, 468-471`

np.unique(...) on the anchor groups is recomputed at 424, per-feature at 437, and at 468; the backfill at 468-471 then runs np.sum(grp == sess_id) per unique session, making it O(n_sessions * n). The anchor array equals ref_groups already held at 408.

**Fix:** For the backfill use uniq, counts = np.unique(grp, return_counts=True) and build n_events_per_session from zip(uniq, counts); reuse ref_groups where possible.

### [LOW] tests — Bout-window boundary guard (start >= 0) drop path untested
`modeling_vocal_bout_parameters.py:386-393`

Only bouts whose history window fits are appended (388); bouts with onset within the first history_frames frames are silently skipped. No test confirms the early bout is dropped consistently across all features. An off-by-one regression would change n_bouts undetected.

**Fix:** Build a session with one bout onset < history_frames plus one valid bout; assert the early bout is excluded from every feature's X/y/groups and counts match across features.

### [LOW] tests — IBI-threshold NaN fallback (gmm_idx out of range) untested
`modeling_vocal_bout_parameters.py:286-294`

When gmm_idx >= len(params['means']) the code stores float('nan') (293-294); every test uses an in-range index so the NaN branch never runs.

**Fix:** Add a test setting gmm_component_index beyond the GMM means and assert md['ibi_thresholds']['male']/['female'] are NaN.

### [LOW] tests — create_data_splits StopIteration/continue branch untested
`modeling_vocal_bout_parameters.py:566-570`

If StratifiedGroupKFold.split yields nothing for a seed, next() raises StopIteration and continue skips it (569-570), silently reducing the number of yielded splits. Session-strategy tests always get a valid fold.

**Fix:** Monkeypatch StratifiedGroupKFold so .split returns an empty iterator with split_strategy='session' and assert fewer splits than split_num without raising.


## `modeling_usv_manifold_position.py` (12)

### [MEDIUM] correctness — Empty test/train session list crashes splitter with opaque np.concatenate error
`modeling_usv_manifold_position.py:256-284`

n_test_sessions = int(len(unique_sessions) * test_prop) truncates toward zero. With few sessions and a small test_prop, n_test_sessions becomes 0 so te_sess = shuffled[:0] is empty and line 283 np.concatenate raises 'need at least one array to concatenate'. The symmetric case (n_test_sessions == len(unique_sessions)) leaves tr_sess empty at 284. No precondition validates session count against test_prop; small cohorts get an opaque error from inside the rejection loop.

**Fix:** After computing n_test_sessions, raise a clear ValueError if not (0 < n_test_sessions < len(unique_sessions)) before entering the rejection loop.

### [MEDIUM] docs_clarity — Stale error-message wording: 'bivariate Gaussian / CNN regression'
`modeling_usv_manifold_position.py:754-756`

The 2-D target guard raises mentioning 'bivariate Gaussian / CNN regression', but the module docstring (8-18) states the bivariate-Gaussian density head was removed and replaced with plain 2-D regression. 'CNN regression' is also wrong for this JAX linear-regression module. The message contradicts the file header.

**Fix:** Reword to describe the current model, e.g. '...currently assumes a 2-D (x, y) manifold target. Received {len(manifold_cols)} columns: {manifold_cols}.'

### [MEDIUM] docs_clarity — get_stratified_spatial_splits_stable docstring omits the metric and period parameters
`modeling_usv_manifold_position.py:176-217`

The signature (166-167) accepts metric and period; metric='torus' makes KMeans run on the 4-D torus embedding. The Parameters block documents groups through widen_every and stops, never mentioning metric or period.

**Fix:** Add Parameters entries for metric (euclidean flat KMeans vs torus 4-D sin/cos embedding) and period (per-axis wrap period, ignored when euclidean).

### [MEDIUM] docs_clarity — _tune_manifold_regularization docstring omits smoothness_derivative_order, use_lax_loop, metric, period
`modeling_usv_manifold_position.py:428-483`

The signature accepts smoothness_derivative_order (376), use_lax_loop (383), metric (385), period (386), all forwarded to inner fits and the inner splitter. The Parameters block documents through regressor_cls and jumps to Returns, omitting all four. metric/period select torus vs euclidean inner-CV geometry.

**Fix:** Add Parameters entries for smoothness_derivative_order, use_lax_loop, metric and period.

### [MEDIUM] performance — Per-sample Python slicing loop in epoch extraction should be a vectorized strided gather
`modeling_usv_manifold_position.py:1053-1089`

For every column of every session the inner for-i loop (1068-1073) copies one fixed-width window at a time into X_arr and runs np.isnan(chunk).any() with a conditional nan_to_num per row. All windows have identical width history_frames and the valid-onset filter (826-834) guarantees in-bounds, so the whole per-column gather is a single strided index. Cost is O(n_columns * num_samples * history_frames) in the interpreter.

**Fix:** idx = slice_starts[:, None] + np.arange(self.history_frames)[None, :]; X_arr = np.nan_to_num(col_values[idx].astype(np.float32), nan=0.0). Drops the Python loop and per-row isnan branch with identical output; slice_ends becomes unused.

### [MEDIUM] tests — Supercategory/category label carry-through in the loader is never exercised
`modeling_usv_manifold_position.py:808-848, 1075-1088`

extract_and_save_continuous_data reads continuous_supercategory/continuous_category, builds valid_super/valid_cat, attaches packet['supercategory']/['category'] (844-847), and the slicing side writes session_entry labels (1085-1088). The _synth USV CSV writes only vae_category/manifold columns and no supercategory source, so find_usv_categories never surfaces those keys; the loader tests exercise only the absent branch. CNN tests build artifacts directly, bypassing the loader. A regression dropping/misaligning the label arrays would pass.

**Fix:** Add a with-labels variant to the _synth USV CSV builder writing the supercategory/category source columns, then assert the loader output's per-session entries carry supercategory/category arrays of length == X.shape[0], 1:1 aligned with onsets, for every feature key.

### [MEDIUM] tests — compute_inverse_density_weights clip_percentile capping is never asserted
`modeling_usv_manifold_position.py:147-148`

TestInverseDensityWeights.test_euclidean_and_torus_unit_mean checks only finiteness, positivity, unit mean with defaults. The clipping logic (np.percentile then np.clip a_max) is never verified, despite outlier-capping being the reason the clip exists.

**Fix:** Add a test with a dense core plus far-flung outliers and a low clip_percentile, asserting no post-normalisation weight exceeds the clipped value's image and that lowering clip_percentile strictly reduces w.max().

### [LOW] correctness — bin_size larger than the history window silently collapses X to zero time bins
`modeling_usv_manifold_position.py:1306-1320`

When bin_size > 1, new_T = T // bin_size; if bin_size > T then new_T == 0 and X_sess reshapes to (N, 0), so n_time_bins becomes 0. The empty design matrix propagates downstream and fails far from the source with no clear diagnostic.

**Fix:** After computing new_T, raise ValueError if new_T < 1 reporting bin_size and T.

### [LOW] dead_code_naming — Vestigial dead branch: n_feats>0 ternary and if sample_feat guard are unreachable-false
`modeling_usv_manifold_position.py:1097-1102`

Line 1091 returns early when final_data is empty, so afterward feature_names is always non-empty, n_feats > 0 is always True, the 'else None' arm at 1097 is unreachable, and 'if sample_feat:' at 1099 is always True. The summary block is effectively unconditional dead defensiveness.

**Fix:** Replace with feature_names = sorted(final_data.keys()); sample_feat = feature_names[0]; drop the ternary and the if sample_feat wrapper and de-indent the summary block.

### [LOW] docs_clarity — Comment claims r2_spatial 'is the selection score' but on torus the selection score is dcor_xy
`modeling_usv_manifold_position.py:1531-1532`

The comment says r2_spatial is first because it is the selection score, but the class docstring (1163-1170) states the score is geometry-dependent (r2_spatial on Euclidean/VAE/UMAP, dcor_xy on torus). The comment is euclidean-centric.

**Fix:** Qualify: 'r2_spatial is first because it is the selection score on Euclidean/VAE/UMAP manifolds (on the torus the selection score is dcor_xy).'

### [LOW] performance — Redundant fancy-index + np.unique on proxy_labels[te_idx] inside rejection-sampling loop
`modeling_usv_manifold_position.py:286-292`

Inside the rejection loop (up to max_total_attempts=50000), proxy_labels[te_idx] is gathered and np.unique'd at 287, then the identical proxy_labels[te_idx] is gathered and np.unique(return_counts=True) recomputed at 290. Both the gather and the sort repeat per accepted-coverage candidate.

**Fix:** Compute te_labels = proxy_labels[te_idx]; te_unique, te_counts = np.unique(te_labels, return_counts=True) once; use len(te_unique) at 289 and te_counts at 290.

### [LOW] tests — 'No valid continuous targets extracted' ValueError is untested
`modeling_usv_manifold_position.py:851-852`

When all_valid_Y_list is empty (every onset fails the history_frames <= f_idx <= max_frame_idx gate at 828), the method raises ValueError. Existing extraction tests use sessions with plenty of post-warm-up events, leaving this misconfigured-filter_history failure mode untested.

**Fix:** Add a test where all USVs fall inside the warm-up window asserting pytest.raises(ValueError, match='No valid continuous targets').


## `jax_bivariate_regression.py` (11)

### [MEDIUM] docs_clarity — _loss_fn docstring omits metric and period parameters
`jax_bivariate_regression.py:463-503`

_loss_fn takes metric: str (line 449) and period: float (line 450), but the Parameters block stops at smoothness_derivative_order (lines 491-495) and never documents either. These two args switch the residual between flat-space and wrap-aware torus differences, so the omission is a real gap given the project's DETAILED-docstring convention.

**Fix:** Append metric ('euclidean' or 'torus'; selects flat vs wrap-aware residual) and period (per-axis wrap period, torus path only) entries to the Parameters block.

### [MEDIUM] tests — Torus dcor_xy path in evaluate_metrics is run but never value-asserted
`jax_bivariate_regression.py:957-963`

The torus branch (lines 957-961) computes dcor_prediction_truth(Y_pred, Y_true, ...) — the documented TORUS selection score. TestTorusMetric.test_torus_fit_predict_evaluate (test_jax_bivariate_regression.py:288-312) asserts r2_spatial, mahalanobis_mae and euclidean_mae but never inspects dcor_xy, so the estimator's own torus dcor_xy integration (nan, swapped Y_pred/Y_true, wrong period/random_state) would not fail any test.

**Fix:** In test_torus_fit_predict_evaluate add assert np.isfinite(metrics['dcor_xy']) and assert metrics['dcor_xy'] >= 0.0; optionally assert dcor_xy is nan on a euclidean fit to pin both sides of the line 957 vs 963 branch.

### [MEDIUM] tests — r2_spatial zero-dispersion fallback (denom <= 0 -> 0.0) is never exercised
`jax_bivariate_regression.py:972-973`

Line 973 r2_spatial = float(1.0 - (ss_res/denom)) if denom > 0 else 0.0 has an untested else branch. All bivariate tests use non-degenerate targets so total_dispersion is always > 0. A constant Y_true (zero dispersion) takes this branch; a regression removing the guard would divide by zero silently.

**Fix:** Add a test fitting on / calling evaluate_metrics with a constant Y_true (e.g. y = np.tile([0.3, 0.7], (n, 1))) and assert metrics['r2_spatial'] == 0.0.

### [LOW] correctness — sample_weight length not validated against n_samples in fit
`jax_bivariate_regression.py:579-582`

check_array(sample_weight, ensure_2d=False) at line 582 never checks length against n_samples. A broadcast-compatible but wrong-length weight (e.g. length 1) silently mis-weights via weights * huber_per_sample in the loss rather than raising.

**Fix:** After the check_array call, add: if sample_weight.shape[0] != n_samples: raise ValueError(...).

### [LOW] correctness — evaluate_metrics weights neither length-checked nor array-validated
`jax_bivariate_regression.py:877-896`

weights is used directly in euclidean_mae_weighted = sum(weights * euclidean_dist)/(sum(weights)+1e-12) (lines 894-896) with no check_array and no length check against Y_true.shape[0]. A wrong-length or 2-D weights array broadcasts silently, corrupting only the weighted MAE.

**Fix:** Validate with check_array(weights, ensure_2d=False) and assert weights.shape[0] == Y_true.shape[0] before use, mirroring fit().

### [LOW] correctness — Zero/near-zero-sum sample weights collapse the loss silently
`jax_bivariate_regression.py:588`

sample_weight / (np.mean(sample_weight) + 1e-12) at line 588: an all-zero weight vector normalises to ~0, weighted_loss ~0, and the model trains only on the L2 + smoothness regularisers with no data signal. The 1e-12 guard prevents a divide-by-zero but masks the degenerate input. circular_mean/total_dispersion in manifold_metric.py (lines 504, 575) already raise on non-positive weight sums.

**Fix:** Guard with if np.mean(sample_weight) <= 0: raise ValueError(...) rather than silently adding 1e-12, matching the circular_mean/total_dispersion convention.

### [LOW] docs_clarity — Module loss formula hardcodes second-time-derivative but order is configurable
`jax_bivariate_regression.py:33`

Line 33 writes the smoothness term as 'sum over second-time-derivatives of W**2' and lines 42-44 say 'the discrete second derivative ... just as in the old Gaussian model', but smoothness_derivative_order (default 2) also supports order 1 (line 538 jnp.diff(..., n=smoothness_derivative_order, ...), documented at lines 270-292).

**Fix:** Reword line 33 / lines 42-44 to reference the smoothness_derivative_order-th (1st- or 2nd-order) discrete time derivative, consistent with the ||D_order W||^2 phrasing at lines 501-502.

### [LOW] docs_clarity — Confusing 'variance gradients stay stable' wording in _initialize_params docstring
`jax_bivariate_regression.py:411`

Line 411 reads 'Xavier/Glorot normal scaling so variance gradients stay stable'. 'variance gradients' is garbled; Xavier scaling stabilises the variance of gradients/activations, not the gradient of a variance.

**Fix:** Reword to 'so the variance of the gradients stays stable in the early Adam steps' or 'so activation/gradient magnitudes stay well-scaled'.

### [LOW] docs_clarity — _use_lax_loop comment describes rejected per-instance caching, contradicting the module-scope shape-keyed implementation
`jax_bivariate_regression.py:388-403`

The __init__ comment (lines 392-397) claims the jitted function 'closes over X_j, Y_j ... cache is keyed per-instance rather than per-shape' and 'pays the full compile cost per instance.' The actual fused path calls module-scope _bivariate_train_loop_jit (lines 650-665), documented as shape-keyed and reused across instances (lines 174-184, 641-649). The two explanations contradict each other.

**Fix:** Update the __init__ comment to reflect the module-scope implementation: a large one-time per-shape compile cost (so the lax path stays off by default for the tuner path), without claiming per-instance cache keying.

### [LOW] tests — huber_delta=np.inf (documented pure-squared-loss recovery) is untested
`jax_bivariate_regression.py:519`

The docstrings (lines 40, 297) state huber_delta=np.inf recovers pure squared-error regression; in the loss (line 519) quad = jnp.minimum(norms, huber_delta) with inf makes quad==norms, lin==0, collapsing huber_per_sample to 0.5*norms**2. No test passes huber_delta=np.inf, so this contract and the inf-handling of jnp.minimum are unverified.

**Fix:** Add a test fitting clean linear data with huber_delta=np.inf, asserting raw MAE < 0.05 and finite coef_ (no nan from inf arithmetic).

### [LOW] tests — euclidean_mae_weighted value/torus correctness asserted only for finiteness
`jax_bivariate_regression.py:894-896`

test_weighted_metric_uses_supplied_weights (test_jax_bivariate_regression.py:209-217) only asserts np.isfinite(metrics['euclidean_mae_weighted']); it never compares against a hand-computed sum(w*dist)/sum(w) or checks it differs from the unweighted euclidean_mae. The weighting math at lines 894-896 (incl. the +1e-12 denominator) could be wrong and still pass, and is never exercised on the torus path.

**Fix:** Compare against a manual np.sum(weights*euclidean_dist)/np.sum(weights) reference on the snapped predictions, and add a torus variant asserting the weighted MAE is finite and >= 0 with non-uniform weights.


## `modeling_cross_session_normalization.py` (11)

### [HIGH] correctness — Pooling `break` excludes co-suffixed columns (self/other) from global stats while z-score applies them to all
`modeling_cross_session_normalization.py:155-159`

harmonize_session_columns (modeling_utils.py:567-581) fills BOTH `{t_name}.{suffix}` (self) and `{p_name}.{suffix}` (other) for every non-dyadic ego feature, so a single session DataFrame holds two columns sharing the same suffix (e.g. mouseA.speed and mouseB.speed). The pooled-stats loop appends only the FIRST match per DataFrame then `break`s (line 159), but the z-score loop (186-194) has no break and normalizes EVERY matching column with that single-column statistic. Empirically verified: with mouseA.speed=[1,2,3] and mouseB.speed=[100,200,300] in one session, pooled mean/std come from mouseA only (mean=2, std=1) and mouseB is z-scored to [98,198,298] instead of being centered/scaled. Dyadic columns are renamed to a single standalone column upstream so they are unaffected; this hits the common self/other ego features.

**Fix:** Remove the `break` on line 159 so all columns whose suffix matches `one_feature` contribute to pooled_series_list, matching the no-break z-score loop and the documented pooled-statistics contract.

### [MEDIUM] tests — `usv_*` skip branch is entirely untested
`modeling_cross_session_normalization.py:139-146`

The `elif 'usv_' not in base_feature:` guard makes a usv_ feature bypass theoretical-bounds clipping and avoid a feature_bounds KeyError. Verified self.usv_rate=[1,2,1000] keeps 1000 (not nulled) and z-scores it, with empty feature_bounds raising no error. No test covers this branch.

**Fix:** Add a test pooling a usv_-suffixed feature with a far-out-of-bounds value and empty feature_bounds; assert the value is NOT nulled, is z-scored, and no KeyError is raised.

### [MEDIUM] tests — Zero-std / constant-feature divide-by-zero guard untested
`modeling_cross_session_normalization.py:180-183`

Lines 180-183 force std=1.0 (and mean=0.0 when also None) for constant/all-null features. Verified a constant self.speed=[5,5,5] returns [0,0,0] not NaN/inf. Dedicated guard with no coverage.

**Fix:** Add a test with a feature constant across sessions; assert all-0.0 finite output. Optionally an all-null case to exercise global_mean is None.

### [MEDIUM] tests — NaN-does-not-poison-global-stats guard (fill_nan(None)) untested for abs / smooth-abs branches
`modeling_cross_session_normalization.py:175`

fill_nan(None) at line 175 is the central robustness claim: NaN preserved by abs()/smooth-abs branches must not poison pooled mean/std. No test injects a NaN into an abs/smooth-abs column and checks another session stays finite; deleting line 175 would fail no test.

**Fix:** Add a two-session test where one session's abs (and smooth-abs) feature contains a NaN; assert the other session's z-scores are all finite, exercising the fill_nan(None) coercion.

### [LOW] docs_clarity — "in place" comment is inaccurate for polars with_columns
`modeling_cross_session_normalization.py:116`

Comment says "Clean the data *in place* in the DataFrames" but line 149 reassigns data_dict[...] = ....with_columns(...); with_columns returns a new DataFrame. Only the dict entry is mutated. Docstring already states dict-level mutation correctly, so the comment is contradictory.

**Fix:** Reword to reflect dict-level reassignment, e.g. "# Clean the data, reassigning each session's DataFrame in the dict (with_columns returns a new frame)".

### [LOW] docs_clarity — z-score comment claims "operates in-place" which is not true for with_columns
`modeling_cross_session_normalization.py:191`

Comment "This operates in-place and preserves height" is inaccurate: with_columns (line 197) yields a new DataFrame reassigned into the dict; the original frame is not mutated. "preserves height" is correct.

**Fix:** Reword to e.g. "# Apply z-score as a Polars expression; with_columns preserves row count (height) and yields a new frame reassigned below".

### [LOW] performance — Per-feature re-scan of every session's columns (O(F*S*C) string-splits)
`modeling_cross_session_normalization.py:152-197`

The feature loop re-scans every session's full column list twice per feature (155-159 and 186-194), each doing column.split('.')[-1] on every column, plus a third scan in the cleaning loop. The per-session feature->column mapping is fixed and could be built once.

**Fix:** Build a per-session {base_feature: [columns]} mapping once with one split per column, then replace inner column scans with dict lookups; reuse it in the cleaning loop.

### [LOW] tests — Documented 'smooth-abs wins over abs_features' precedence untested
`modeling_cross_session_normalization.py:124-136, 137`

Docstring 96-98 contracts that a suffix in both smooth_abs_features and abs_features is treated as smooth-abs (branch ordering 124 vs 137). No test passes the same suffix in both.

**Fix:** Add a test passing one suffix in both args with 0.0 input; assert sqrt(x^2+eps^2) applied (0->eps), proving the dict key wins.

### [LOW] tests — Prefix-stripping/other.-prefix pooling and non-listed-column pass-through untested
`modeling_cross_session_normalization.py:119-121`

Line 120 strips prefix so self.speed and other.speed both match; line 121 leaves non-listed columns untouched. All tests use only self.-prefixed, in-list columns. No test covers an other.-prefixed column pooled with self., nor a column not in feature_lst passed through unchanged.

**Fix:** Add a test with both self.speed and other.speed plus an unrelated column not in feature_lst; assert the unrelated column is unchanged and both prefixed columns share pooled stats.

### [LOW] tests — Feature present in feature_lst but absent from all DataFrames (empty pooled list) untested
`modeling_cross_session_normalization.py:161-162`

Lines 161-162 guard against pls.concat([]) (which raises) for a listed feature absent from every session. No test exercises this.

**Fix:** Add a test with feature_lst including an absent suffix; assert the call completes and present features are still z-scored.

### [LOW] tests — Non-float dtype fallback of fill_nan guard untested
`modeling_cross_session_normalization.py:175`

Line 175 only calls fill_nan when dtype.is_float(), else uses the raw series. All tests use float inputs, so the integer-typed fallback is never exercised.

**Fix:** Add a test with an integer-typed feature column; assert it z-scores without error, exercising the non-float fallback.


## `consolidate_univariate_results.py` (11)

### [MEDIUM] correctness — Mismatch error names `pkl_files[0]` as canonical file, but canonical metadata may come from a later file
`consolidate_univariate_results.py:294,301`

`canonical_input_md` / `canonical_run_md` are captured from the FIRST file that actually carries metadata (lines 281-283, guarded by `if canonical_input_md is None`). In `--allow_legacy` mode the leading sorted files can be legacy (no metadata), so the canonical block is taken from e.g. `pkl_files[2]`. Both mismatch errors (lines 294, 301) hard-code `pkl_files[0].name` as the 'A' side, so the reported filename points at a file that contributed no metadata, sending the user to debug the wrong file.

**Fix:** Track the path of the file that set the canonical metadata (e.g. `canonical_md_path = fp` alongside lines 281-283) and reference that variable in both error messages instead of `pkl_files[0].name`.

### [MEDIUM] tests — `ignore_provenance_keys` is never verified to actually ignore a divergent key
`consolidate_univariate_results.py:290-303`

Tests prove a mismatch on a SUBSTANTIVE key raises, but no test proves the positive contract that the default `ignore_provenance_keys` causes two files differing ONLY on e.g. `git_commit` to merge SUCCESSFULLY. The entire reason the feature exists (lines 173-175, 285-291) is uncovered; a regression that stopped honoring `ignore_keys` would pass all current tests.

**Fix:** Add a test where two per-feature pickles share identical md_in/md_run except md_run differs only on 'git_commit'; assert `consolidate` succeeds and merges both. Optionally assert `ignore_provenance_keys=()` makes the same pair raise.

### [LOW] correctness — Mismatch diff message ignores `ignore_provenance_keys`, reporting spurious provenance differences
`consolidate_univariate_results.py:292,299`

Equality is tested with `metadata_blocks_equal(..., ignore_keys=ignore_provenance_keys)` (lines 290-291, 297-298), which correctly skips `git_commit` / `git_dirty` / `package_version`. But when it returns False, the diff is built with `_diff_metadata(canonical, cur)` (lines 292, 299), which takes no ignore-list and recurses over ALL keys. The printed error can list the intentionally-ignored provenance keys alongside the real offending key, misleading the reader into thinking provenance caused the abort. The abort itself is correct (driven by a non-ignored key).

**Fix:** Give `_diff_metadata` an `ignore_keys: tuple = ()` parameter, subtract it from the top-level key sets (mirroring `metadata_blocks_equal`), and pass `ignore_keys=ignore_provenance_keys` at lines 292 and 299; update the docstring accordingly.

### [LOW] correctness — `_parse_feature_idx` returns wrong index when the analysis tag contains an all-digit underscore token
`consolidate_univariate_results.py:114-117`

The parser scans `parts[1:]` and returns the first `isdigit()` token. The schema is `univariate_<analysis_tag>_<idx:04d>_<safe_feat>_<ts>`, and `<analysis_tag>` precedes `<idx>`. If the tag itself contains an all-digit token (e.g. `cohort_2024` -> tokens `['univariate','cohort','2024',idx,...]`), `2024` is returned instead of the real index. This only affects the sort order of `pkl_files` (lines 234-235), not which data is merged, but a wrong sort changes which file becomes `pkl_files[0]`, compounding the misleading-filename issue.

**Fix:** Anchor the parse to the documented position (the zero-padded 4-digit index immediately after the analysis tag), e.g. a regex tied to the `_NNNN_` index token, rather than 'first digit token wins'.

### [LOW] dead_code_naming — Unused loop variable `idx` in feature-merge loop
`consolidate_univariate_results.py:249`

`for idx, fp in enumerate(pkl_files):` (line 249) binds `idx`, which is never referenced anywhere in the loop body (lines 250-313). Ordering is implied by append order and error messages use `pkl_files[0].name` / `fp.name`, never `idx`. The `enumerate` wrapper is vestigial.

**Fix:** Drop the `enumerate(...)` and bind `fp` directly: `for fp in pkl_files:`.

### [LOW] docs_clarity — Docstrings state legacy filename is `legacy_<ts>.pkl` but code emits `legacy_univariate_<ts>.pkl`
`consolidate_univariate_results.py:70,195-196,204-206`

The module docstring (line 70) and the `consolidate` docstring (lines 195-196 and 204-206) all describe a `legacy_<ts>.pkl` filename, but line 335 produces `f"legacy_univariate_{ts}.pkl"`. The documented prefix `legacy_` does not match the actual `legacy_univariate_`. The test at line 197 (`startswith('legacy_univariate_')`) confirms `legacy_univariate_` is the intended name, so the docstrings are stale.

**Fix:** Update the three docstring references to read `legacy_univariate_<ts>.pkl` to match line 335.

### [LOW] docs_clarity — `_diff_metadata` docstring omits the missing-key output format
`consolidate_univariate_results.py:146-151`

The docstring (lines 147-150) describes the return value as only `'<path>: <a_val> != <b_val>'` strings, but the function also emits `'<path><key>: missing in B'` / `'missing in A'` entries (lines 156, 158) for keys present in only one block. That output form is undocumented (and is exercised by the test at lines 89-90).

**Fix:** Document the `missing in A` / `missing in B` output form in the docstring.

### [LOW] tests — Zero-feature-key branch of `len(feat_keys) != 1` is untested
`consolidate_univariate_results.py:256-262`

Line 258 raises when a per-feature pickle does not contain exactly one feature key. `test_multiple_feature_keys_raises` covers the >1 case only. The zero-feature-key scenario (a pickle containing only `_input_metadata`/`_run_metadata`, so `feat_dict` is empty after `extract_metadata_blocks`) is never exercised.

**Fix:** Add a test writing `{'_input_metadata': md_in, '_run_metadata': md_run}` (no feature key) and assert `cuv.consolidate(str(tmp_path))` raises `ValueError` matching 'Expected exactly one feature key'.

### [LOW] tests — `delete_individuals_after` OSError warning branch is untested
`consolidate_univariate_results.py:348-356`

`test_delete_individuals_after_removes_merged` (lines 205-218) covers only the successful unlink path. The `except OSError` branch (lines 354-355) that prints a WARNING and yields a `deleted < total` count is never exercised.

**Fix:** Add a test that monkeypatches `Path.unlink` to raise OSError for one merged path, then assert via capsys that '[consolidate] WARNING: could not delete' is printed and the artifact still exists.

### [LOW] tests — CLI `--ignore_provenance_keys` comma-split/strip parsing is untested
`consolidate_univariate_results.py:375-381`

`TestConsolidatorCLI` always uses the default `ignore_provenance_keys`. The parsing at line 381 (`tuple(k.strip() for k in cli_args.ignore_provenance_keys.split(',') if k.strip())`), including whitespace stripping and empty-token filtering, is never exercised with a custom value.

**Fix:** Add a CLI test passing `--ignore_provenance_keys ' git_commit , , git_dirty '` over two pickles differing only on git_commit; assert exit is None and 'OK:' is printed.

### [LOW] tests — `_file_mtime_iso` ISO/Z formatting has no direct test
`consolidate_univariate_results.py:120-128`

`_file_mtime_iso` is called indirectly during round-trips, but no test asserts its return-value contract: an ISO-8601 UTC string with `+00:00` rewritten to trailing `Z` and microseconds zeroed (line 127). A regression in the replacement or truncation would pass silently.

**Fix:** Add a unit test in `TestUnivariateHelpers` that touches a file, calls `cuv._file_mtime_iso(str(f))`, and asserts it endswith('Z'), lacks '+00:00', and matches `YYYY-MM-DDTHH:MM:SSZ` (no fractional seconds).


## `modeling_vocal_categories_multinomial.py` (11)

### [MEDIUM] docs_clarity — Comment mislabels the 'null' strategy as 'X-shuffled' when it shuffles labels (y)
`modeling_vocal_categories_multinomial.py:1520`

The comment reads 'within-session X-shuffled null'. The null strategy shuffles vocal category labels y within each session (lines 1406-1411), not X. The method docstring states this correctly at line 1235. Misleads readers into thinking the permutation acts on the design matrix.

**Fix:** Change 'within-session X-shuffled null' to 'within-session label-shuffled null' (or 'y-shuffled') to match the actual permutation target and the method docstring.

### [LOW] correctness — Temporal binning silently collapses the window to zero predictors when bin_size > n_frames
`modeling_vocal_categories_multinomial.py:1199-1201`

new_T = T // bin_size; if user-configurable bin_resizing_factor exceeds history length T, new_T becomes 0, producing a degenerate (N,0) design matrix and n_time_bins=0 that silently flows into the JAX estimator instead of failing loudly. Shipped defaults (history 600, bin 1) cannot trigger it, but bin_resizing_factor is user-set and unvalidated.

**Fix:** Add a check that new_T >= 1 (e.g. if new_T == 0: raise ValueError(f"bin_size={bin_size} exceeds history length {T} for feature {feat}")) before the reshape.

### [LOW] correctness — Onset frame uses np.round while the rest of the codebase floors event onsets
`modeling_vocal_categories_multinomial.py:743`

frame_indices = np.round(start_times * fps).astype(int) rounds to nearest, whereas canonical onset->frame conversions in load_input_files.py use np.floor (confirmed lines 191 and 517). This defines the multinomial history window boundary up to one frame differently from the binary/continuous pipelines sharing the same source onsets, with no comment justifying the divergence.

**Fix:** If matching the other pipelines is intended, use np.floor(start_times * fps).astype(int); otherwise add a one-line comment justifying the deliberate np.round choice.

### [LOW] dead_code_naming — Dead instance attribute self.feature_boundaries in MultinomialModelRunner.__init__
`modeling_vocal_categories_multinomial.py:1115-1116`

MultinomialModelRunner.__init__ copies feature_boundaries onto self.feature_boundaries, but grep shows the only runner occurrence is this assignment. The pipeline (not the runner) reads boundaries at line 813 via getattr on the pipeline instance set at 647-648; the runner never reads its copy. Vestigial copy-paste from the manifold runner.

**Fix:** Remove the if hasattr(...) self.feature_boundaries = ... block from MultinomialModelRunner.__init__ since the runner never reads it.

### [LOW] docs_clarity — Stale settings key in docstring: 'save_dir' instead of io.save_directory
`modeling_vocal_categories_multinomial.py:681`

The Returns section says "Saves a pickle file to self.modeling_settings['save_dir']". No such key is read; the save location resolves from self.modeling_settings['io']['save_directory'] (confirmed lines 924 and 1012).

**Fix:** Update the docstring to reference self.modeling_settings['io']['save_directory'].

### [LOW] docs_clarity — Stale/incorrect settings-key examples in extract_and_save docstring
`modeling_vocal_categories_multinomial.py:676-677`

Line 676 lists 'category_column_name' under 'vocal_features' but the read key at line 700 is 'usv_category_column_name'. Line 677 lists a 'features' block with 'filter_history', but no 'features' block is read; filter_history comes from 'model_params' (line 703).

**Fix:** Correct line 676 to 'usv_category_column_name' under 'vocal_features', and correct line 677 to reference the 'model_params' block (filter_history) rather than a nonexistent 'features' block.

### [LOW] tests — Single-session 'session'-strategy guard (ValueError) is untested
`modeling_vocal_categories_multinomial.py:178-183`

The 'session' branch raises ValueError when len(unique_sessions) < 2 (lines 178-183). The splitter tests never feed a single-session cohort to the session strategy (disjointness test uses 6 sessions; no-valid-fold test uses 3). This early guard and its message are uncovered.

**Fix:** Add a test passing groups=np.zeros(N, int) (one unique session) with split_strategy='session', y covering all n_categories, asserting pytest.raises(ValueError, match='at least 2 sessions').

### [LOW] tests — n_test_sessions floor (max(1, ...)) for small test_prop is untested
`modeling_vocal_categories_multinomial.py:189`

Line 189 floors the test-session count to >=1 to prevent an empty test set when int(len*test_prop) rounds to 0. No test drives the session strategy with a small enough test_prop to trigger the floor; a regression removing max(1, ...) would pass CI.

**Fix:** Add a session-strategy test with 2 sessions (each carrying all classes) and test_prop=0.2, asserting it returns n_splits folds each with a non-empty class-complete test set.

### [LOW] tests — Tolerance-widening accept path (widen_step>0) is never exercised to acceptance
`modeling_vocal_categories_multinomial.py:227-233`

Lines 227-233 widen current_tolerance every widen_every failures. The only session-stress test sets widen_step=0.0 (confirmed line 1271), so the tolerance never increases and no fold is accepted after a widening; the accept-after-widening branch and recording of the widened float in fold_tolerances (225) are untested.

**Fix:** Add a session-strategy test with initial tolerance=0.0 and widen_step>0 / small widen_every on data where moderate widening admits folds; assert len(cv_folds)==n_splits and at least one fold_tolerances entry > 0.0.

### [LOW] tests — balance_predictions_bool=True prediction path is never exercised
`modeling_vocal_categories_multinomial.py:1581-1582`

predict_proba/predict are called with balanced=hp['balance_predictions_bool'] (1581-1582). Every test config sets balance_predictions_bool=False (line 113), so the balanced-prediction path through the runner is never run end-to-end.

**Fix:** Add a runner test (or parametrize the tri-strategy test) with balance_predictions_bool=True asserting the run completes and produces finite metrics.

### [LOW] tests — NaN-chunk -> nan_to_num substitution in epoch slicing is not asserted
`modeling_vocal_categories_multinomial.py:966-970`

extract_and_save replaces NaN in sliced windows with 0.0 via np.nan_to_num. Synthetic builders produce clean data so the if np.isnan(chunk).any() branch is never taken; a regression removing the scrub would leak NaN rows undetected.

**Fix:** Add an extraction test injecting a NaN in a frame inside an event's history window, asserting resulting X is finite everywhere.


## `main_univariate_dispatcher.py` (11)

### [MEDIUM] correctness — Negative feature_idx silently wraps to a valid feature instead of erroring
`main_univariate_dispatcher.py:220`

The bounds check at line 220 only guards the upper bound: `if args.feature_idx >= len(all_features)`. argparse uses type=int (line 363) with no lower-bound validation, so a negative --feature_idx passes the check, then `all_features[args.feature_idx]` (line 224) wraps via Python negative indexing to a feature from the end of the list. The job completes and writes a result pickle for the WRONG feature, with the filename embedding the negative index (`{args.feature_idx:04d}`).

**Fix:** Tighten the guard to `if args.feature_idx < 0 or args.feature_idx >= len(all_features):` at line 220, and/or validate feature_idx >= 0 at argparse time.

### [MEDIUM] docs_clarity — dispatch_univariate_job docstring lists only 3 of the 5 supported analysis types
`main_univariate_dispatcher.py:164-167`

The docstring (line 166) says it selects '(Onset, Multinomial JAX, or Continuous JAX)', but the function routes five types: 'onset', 'category', 'params', 'multinomial', 'continuous' (routing lines 244-303; argparse choices line 357). 'category' and 'params' are omitted, contradicting the module header (lines 10-15) that enumerates all five.

**Fix:** Reword to enumerate all five routed types (Onset, Category, Params, Multinomial JAX, Continuous JAX) to match the module-level list and argparse choices.

### [MEDIUM] performance — Input pickle is fully deserialized twice on every CPU-based task (onset/category/params)
`main_univariate_dispatcher.py:215-246`

For CPU tasks the input .pkl is pickle.load-ed in full twice. First at line 215 (`loaded = pickle.load(f)`) to build the key list and harvest _input_metadata; that dict is discarded at line 218 (`del loaded`). Then line 246 (`load_pickle_modeling_data(args.input_data)`) calls pickle.load on the same file again (verified full load at load_input_files.py:1051) just to take data_dict[feature_name] (line 247). The two-phase design benefits only the JAX branches (which pass pkl_path and never call load_pickle_modeling_data here). For CPU tasks it doubles the most expensive operation per SLURM array task with no memory benefit.

**Fix:** In the CPU branch, capture `feat_data = loaded[feature_name]` during the first load before `del loaded`, and pass it through instead of calling load_pickle_modeling_data again at line 246. Keep key/metadata-only handling for the JAX branches.

### [MEDIUM] tests — Untested: input_metadata present but missing 'analysis_tag' falls back to analysis_type for filename
`main_univariate_dispatcher.py:331-333`

Lines 329-332 choose the filename tag: (a) input_metadata not None AND 'analysis_tag' in it -> use the tag; (b) otherwise -> use args.analysis_type. Tests cover full-metadata-with-tag (test_onset_sklearn_route_writes_artifact / test_continuous_route_unwraps_tuple) and input_metadata is None (test_legacy_input_without_metadata_warns). The middle case -- input_metadata non-None but no 'analysis_tag' key -- is never tested and must NOT raise KeyError but fall through to analysis_type.

**Fix:** Add a test writing an input pickle whose _input_metadata is non-empty but omits 'analysis_tag'; run the onset+sklearn route and assert exactly one 'univariate_onset_0000_*.pkl' is written (fell back to analysis_type, no KeyError).

### [LOW] correctness — B-spline knot construction assumes history_frames is large enough for n_splines
`main_univariate_dispatcher.py:111`

At line 111 `knots = np.linspace(0, max_k, p['n_splines'] - p['degree'] + 1).astype(int)` (max_k = max(0, w - degree), line 110). With shipped config (n_splines=32, degree=3) this requests 30 integer knot positions over [0, w-3]. When w < 32, `.astype(int)` truncation yields duplicate / non-strictly-increasing knots, which are passed as x-data to si.splrep (modeling_bases_functions.py:217) that requires strictly increasing x and raises ValueError. With shipped defaults w is large (camera_rate * filter_history_sec), so this is a latent config-dependent crash, not a default-config bug.

**Fix:** Before building knots, validate w is large enough for n_splines/degree, or de-duplicate (np.unique) the integer knots and assert strict monotonicity, raising a clear error if the history window is too short.

### [LOW] docs_clarity — Returns docstring for get_basis_matrix_standardized omits the second None-return case
`main_univariate_dispatcher.py:83-86`

The Returns section says None is returned 'if the current model_engine is pygam'. The early return at lines 90-91 fires for ANY model_engine != 'sklearn' (not just pygam), and basis_matrix additionally stays None (line 95) if basis_type matches none of raised_cosine/bspline/laplacian_pyramid/identity (no else branch). The single-cause framing is incomplete and slightly incorrect.

**Fix:** Reword to: 'Returns None when model_engine is not sklearn (e.g. pygam), or when model_basis_function does not match a recognized basis type.'

### [LOW] docs_clarity — Module-header 'Memory Guarding' implies key-only load is JAX-specific, but it runs for every analysis type
`main_univariate_dispatcher.py:19-22`

The header says the two-phase load applies 'For JAX/GPU tasks'. In the implementation the key-only load + `del loaded` + `gc.collect()` (lines 215-232) runs unconditionally for all analysis types, including the CPU paths (onset/category/params) which re-load full data via load_pickle_modeling_data at line 246. The 'For JAX/GPU tasks' qualifier misdescribes when the guard runs.

**Fix:** Reword to clarify the key-only indexing pass runs for all tasks, and that CPU paths subsequently re-load full data while JAX paths defer loading to the runner via pkl_path, which is what actually preserves GPU headroom.

### [LOW] tests — Untested: continuous route non-tuple return path (res = raw_res)
`main_univariate_dispatcher.py:302`

Line 302 `res = raw_res[1] if (isinstance(raw_res, tuple) and len(raw_res) == 2) else raw_res` has two arms. test_continuous_route_unwraps_tuple (line 459) only covers the tuple arm (return ('ignored', {'r2': 0.9})). The else arm -- run_univariate_training returns a bare dict, which is the runner's actual annotated return type (-> dict) -- is never tested.

**Fix:** Add a test patching ContinuousModelRunner so run_univariate_training.return_value is a plain dict e.g. {'r2': 0.5}, dispatch 'continuous', and assert the written payload['self.speed'] == {'r2': 0.5}.

### [LOW] tests — Untested: unrecognized basis_type returns None despite sklearn engine
`main_univariate_dispatcher.py:97-132`

In get_basis_matrix_standardized, when model_engine == 'sklearn' but model_basis_function is none of {raised_cosine, bspline, laplacian_pyramid, identity}, no branch assigns basis_matrix, so it stays None (line 95), the plot guard at line 142 short-circuits, and the function returns None. test_pygam_engine_returns_none only covers None-via-pygam.

**Fix:** Add a test with model_engine='sklearn' and model_basis_function='nonexistent_basis'; assert get_basis_matrix_standardized returns None and no basis_verification.png is written.

### [LOW] tests — Untested: FileExistsError race path in atomic plot lock
`main_univariate_dispatcher.py:143-156`

The atomic lock uses lock_file.touch(exist_ok=False) inside try/except FileExistsError (lines 145, 155-156) to swallow the race where two jobs pass the 'not lock_file.exists()' check and both touch. test_existing_lock_skips_plot pre-creates the lock, so the guard at line 143 is False and the try-body/except is never entered. The race branch has no coverage.

**Fix:** Add a test patching Path.touch to raise FileExistsError while lock_file.exists() returns False so the guard is entered; assert the basis matrix is returned and no PNG is written.

### [LOW] tests — Untested: bspline and laplacian_pyramid plot_bool=False suppression
`main_univariate_dispatcher.py:108-118`

test_plot_bool_toggles_parametric_basis_plot (line 201) only parametrizes plot_bool on/off for 'raised_cosine'. The same plot_bool gate at line 142 applies to bspline and laplacian_pyramid, both reading their own p['plot_bool'] (settings lines 218, 224), but neither is tested with plot_bool=False, so a regression in those two blocks' key lookup would go uncaught. Marginal -- shares the same code path as raised_cosine.

**Fix:** Extend test_plot_bool_toggles_parametric_basis_plot to parametrize over ['raised_cosine','bspline','laplacian_pyramid'] asserting plot_bool=False writes no PNG and plot_bool=True writes one.


## `acoustic_manifold_geometry.py` (10)

### [HIGH] correctness — Lattice replication inflates KDE bandwidth ~period, defeating wrap-aware centring for straddling clusters
`acoustic_manifold_geometry.py:144-176`

On torus the KDE is fit on a 9x lattice-replicated cloud (pts_kde), and gaussian_kde infers covariance from that replicated cloud whose per-axis std is dominated by the period spacing (~0.8-0.95 for period 1), not the intra-cluster spread. Reproduced directly: a tight cluster (true std 0.03) centred on the 0.0/1.0 seam yields an effective per-axis bandwidth ~0.20-0.23 (sqrt of kde.covariance diagonal) under Scott's rule, and with kde_bandwidth=0.8 -- the exact value the docstring (lines 106-108) recommends to reproduce the QLVM pipeline -- the returned centre is the EMPTY geometric centre [0.513, 0.498] instead of the true seam at x=0/1. This is precisely the antipode/cell-centre failure the replication and module docstring (lines 16-18, 74-79) claim to prevent.

**Fix:** Derive the KDE bandwidth from the ORIGINAL (un-replicated) per-label covariance and override it on the replicated KDE (compute cov from pts, set kde.covariance/inv_cov, or set_bandwidth scaled to the original-cloud variance) instead of letting gaussian_kde infer covariance from the replicated cloud. At minimum, fix the docstring's 0.8 recommendation, which silently returns a wrong centre on the wrap seam.

### [MEDIUM] docs_clarity — Comment states replicated cloud is (3*N, 2) but it is (9*N, 2)
`acoustic_manifold_geometry.py:146-147`

The torus branch builds the full 3x3 lattice (3 dx x 3 dy = 9 shifts, lines 149-152) and concatenates pts+s for all 9 shifts (line 153), producing a (9*N, 2) cloud. The inline comment on line 147 says 'The KDE consumes the (3*N, 2) replicated point cloud', contradicting both the code and its own next sentence ('sums contributions from all nine shifts') and the module docstring's '3x3 lattice-replicated copy' / 'nine shifts'.

**Fix:** Change '(3*N, 2)' to '(9*N, 2)' on line 147.

### [MEDIUM] tests — max_points_per_label sub-sampling branch is never exercised
`acoustic_manifold_geometry.py:141-142`

The branch 'if n_total > max_points_per_label: pts = pts[rng.choice(...)]' (lines 141-142) is never hit: all fixtures use at most 500 points/cluster and the default cap is 30000, so the sub-sample path, the rng.choice draw, and its interaction with rng_seed determinism (line 128) are uncovered.

**Fix:** Add a test building a tight Gaussian blob with e.g. 200 points and call with max_points_per_label=50; assert the centre is within tolerance and that two calls with the same rng_seed return identical centres.

### [MEDIUM] tests — kde_bandwidth argument forwarding is untested
`acoustic_manifold_geometry.py:157`

kde_bandwidth (forwarded to gaussian_kde bw_method on line 157) is documented as the way to reproduce the QLVM watershed seeding (pass 0.8) but no test passes a non-None value -- every test relies on the default Scott path. A regression dropping/renaming/mis-passing the argument would not be caught. Verified a euclidean blob with kde_bandwidth=0.8 recovers the centre within tolerance (err 0.005), so a euclidean smoke test is safe.

**Fix:** Add a test passing kde_bandwidth=0.8 on a euclidean blob asserting the centre is recovered within tolerance, plus a tiny and a large scalar bandwidth confirming both float values are accepted and produce a finite (2,) centre.

### [MEDIUM] tests — max_radius clamp in uniform mode is never tested
`acoustic_manifold_geometry.py:276-277`

test_max_radius_clamps exercises only adaptive mode (mode='adaptive'). The uniform-mode clamp on lines 276-277 ('if max_radius is not None: uniform_radius = min(uniform_radius, max_radius)') has zero coverage. A bug applying the cap only in adaptive mode, or before vs after the global-min reduction, would pass.

**Fix:** Add a test calling derive_cluster_geometry(..., mode='uniform', max_radius=R) where the unclamped uniform radius exceeds R, asserting every cluster radius == R while nearest_neighbour_distance is the raw per-centre value.

### [LOW] correctness — On torus the data-range grid is built AFTER subsampling, so a missed straddle can narrow the grid
`acoustic_manifold_geometry.py:141-164`

When n_total > max_points_per_label, pts is replaced by a uniform subsample (line 142) and the grid bounds (lines 162-164) are computed from the subsample min/max. For a wrap-straddling cluster the subsample could (low probability) drop all points on one side of the seam, collapsing pts.min/pts.max to a narrow non-straddling range and yielding a grid that no longer covers the true seam mode, making the returned centre depend on the rng_seed draw on straddling clusters.

**Fix:** Compute the grid bounds (and torus clamp) from the full per-label finite points Y[mask] before subsampling, and only subsample the points fed into gaussian_kde.

### [LOW] performance — O(n^2) nearest-neighbour distance via a nested Python loop instead of vectorized broadcasting
`acoustic_manifold_geometry.py:264-272`

The per-centre nearest-neighbour distance uses a double Python for-loop over all (i, j) pairs, calling signed_diff once per pair and computing each unordered pair twice. The sibling manifold_metric._geodesic_distance_matrix already provides the exact vectorized form (signed_diff broadcast to (n,n) then sqrt sum). Verified the vectorized replacement reproduces d_nn = [1,1,3] on the test centres. n is the number of cluster centres so memory is never a concern.

**Fix:** from .manifold_metric import _geodesic_distance_matrix; D = _geodesic_distance_matrix(C, metric=metric, period=period); np.fill_diagonal(D, np.inf); d_nn = D.min(axis=1).

### [LOW] performance — Redundant per-axis min/max reductions over the per-label point cloud
`acoustic_manifold_geometry.py:162-164`

pts.min(axis=0) and pts.max(axis=0) are each computed twice: on line 162 (for margin) and again on lines 163-164 (for xmin/ymin and xmax/ymax). Four full-array reductions over a cloud of up to max_points_per_label rows where two would suffice, multiplied by the number of labels.

**Fix:** Compute pmin = pts.min(axis=0) and pmax = pts.max(axis=0) once, then margin = 0.05*(pmax-pmin).max(); xmin,ymin = pmin-margin; xmax,ymax = pmax+margin.

### [LOW] tests — usv_in_circle has no metric/period validation test
`acoustic_manifold_geometry.py:332`

usv_in_circle's _validate_metric_period call (line 332) is never exercised with an invalid metric or non-positive period; the other two public functions have explicit validation tests but usv_in_circle does not, leaving its line-332 error path uncovered for this function specifically.

**Fix:** Add a test asserting usv_in_circle(Y, centroid, radius, metric='spherical') raises ValueError matching 'manifold_metric must be one of', and metric='torus', period=0.0 raises ValueError matching 'manifold_period must be a positive'.

### [LOW] tests — Torus grid-clamping branch (centre forced inside canonical cell) is not asserted on
`acoustic_manifold_geometry.py:166-170`

Lines 166-170 clamp the KDE evaluation grid to [0, period]^2 so the returned centre is guaranteed inside the canonical cell. test_torus_wrap_boundary_recovers_origin checks only wrap-aware distance (d_axis = min(c, period-c)), which would still pass even if the returned centre fell slightly outside [0, period). No test asserts 0 <= c_est <= period.

**Fix:** In test_torus_wrap_boundary_recovers_origin (or a new test) additionally assert (c_est >= 0.0).all() and (c_est <= period).all() to lock in the canonical-cell guarantee.


## `modeling_utils.py` (10)

### [MEDIUM] docs_clarity — Module docstring map omits 9 functions (the entire metrics + audit-orchestration layer)
`modeling_utils.py:13-65`

The lettered map enumerates A, C-N (prepare_modeling_sessions through bounded_test_proportion) but the file also defines brier_score_multi, expected_calibration_error, safe_matthews_corrcoef, safe_confusion_matrix, align_probs_to_canonical, pearson_r_safe, root_mean_squared_error, mean_absolute_error_1d, and run_predictor_audits (934-1702), none in the map. The opening summary (5-6) still claims the module only consolidates the extract_and_save_* prologue.

**Fix:** Add a second titled group (e.g. 'Metrics & audit orchestration') covering the 9 missing functions and broaden the opening sentence.

### [MEDIUM] docs_clarity — safe_matthews_corrcoef docstring mis-describes its NaN fallback and omits NaN from Returns
`modeling_utils.py:1069-1090`

The docstring attributes the single-class 0.0 to 'this wrapper preserves that behavior', but sklearn returns 0.0 for single-class (and even empty) inputs without raising (verified). The wrapper's actual added behavior — catch ValueError, return NaN (1089-1090) — is undocumented, and Returns (1083-1084) lists only 'The Matthews correlation coefficient', omitting NaN.

**Fix:** Reword to state sklearn already returns 0.0 without raising for the degenerate case, and that this wrapper additionally returns NaN for inputs sklearn rejects; add NaN to Returns.

### [MEDIUM] tests — prepare_modeling_sessions generic-exception RuntimeError wrapping branch is untested
`modeling_utils.py:138-139`

The except Exception as e: raise RuntimeError(...) branch (138-139) has no test; only success, FileNotFoundError, and ValueError paths are covered. A regression to this branch's re-raise type would not be caught.

**Fix:** Add a test pointing session_list_file at a directory (f = tmp_path/'d'; f.mkdir()) so open() raises IsADirectoryError, and assert pytest.raises(RuntimeError).

### [MEDIUM] tests — expected_calibration_error empty-input (n_total==0) NaN branch is untested
`modeling_utils.py:1038-1039`

The guard if n_total == 0: return float('nan') (1038-1039) is never exercised; tests cover calibrated, overconfident, and non-2D cases only. An empty fold (class never observed in a held-out session) is a real scenario.

**Fix:** Add assert np.isnan(expected_calibration_error(np.array([]), np.array([]), np.empty((0, 2)))).

### [LOW] correctness — align_probs_to_canonical does not verify model_classes subset membership before searchsorted assignment
`modeling_utils.py:1182-1183`

The function enforces the ascending-order invariant loudly (1171-1175) but never checks that every model_classes value is actually present in canonical_classes. np.searchsorted returns only an insertion index, so a model class value between two canonical values silently writes probabilities to the wrong canonical column, and a value above all canonical values yields index==len and an out-of-bounds assignment at line 1183. The docstring asserts subset-ness but it is unchecked, contradicting the loud-failure philosophy that motivated the ascending-order assertion.

**Fix:** After target_cols = np.searchsorted(...), guard the overflow (np.any(target_cols >= len(canonical_arr))) and verify membership (np.all(canonical_arr[target_cols] == np.asarray(model_classes))), raising a clear ValueError otherwise.

### [LOW] correctness — Negative gmm_component_index passes the < len guard and indexes the wrong GMM component
`modeling_utils.py:1560`

if gmm_idx < len(params['means']) (1560) admits negative indices (e.g. -1), which then index params['means'][gmm_idx] from the end of the array, silently selecting the wrong component and producing a misleading IBI threshold headline. A non-negative out-of-range index correctly falls to the NaN branch; a negative one does not.

**Fix:** Tighten to if 0 <= gmm_idx < len(params['means']) so a negative or out-of-range index falls to the NaN branch.

### [LOW] docs_clarity — expected_calibration_error Returns omits the empty-input NaN case
`modeling_utils.py:1023-1024`

Code returns float('nan') when n_total == 0 (1038-1039), but Returns states only 'The top-label expected calibration error on [0, 1]'. NaN is outside [0,1] and is a real documented behavior.

**Fix:** Add a sentence to Returns noting NaN is returned when y_true is empty (n_total == 0).

### [LOW] tests — bounded_test_proportion min_test_sessions parameter never exercised with a non-default value
`modeling_utils.py:894`

TestBoundedTestProportion only uses the default min_test_sessions=1; the multi-session floor (min_test_sessions / n_sessions on 931) is untested.

**Fix:** Add bounded_test_proportion(0.1, n_sessions=10, min_test_sessions=2) == pytest.approx(0.2) and (0.5, 10, 2) == 0.5.

### [LOW] tests — build_vocal_signal_columns custom usv_self_exclude argument is untested
`modeling_utils.py:326`

Only the default usv_self_exclude is tested; a caller-supplied override (empty tuple, or a different key) used at line 393 is never exercised.

**Fix:** Add a test with usv_self_exclude=() asserting mTarget.usv_rate/usv_event appear, and usv_self_exclude=('usv_cat_0',) asserting the category signal drops while rate/event survive.

### [LOW] tests — select_kinematic_columns derivative addition on dyadic_pose and dyadic_engagement buckets is untested
`modeling_utils.py:306`

_maybe_add_derivatives is called in all three buckets (291, 306, 316) but tests cover derivatives only for the egocentric bucket; a regression dropping the call from either dyadic branch would pass.

**Fix:** Add a dyadic_pose test (mA-mB.nose-nose plus _1st_der companion, include_1st_derivatives=True) asserting the derivative column is kept; analogously for dyadic_engagement.


## `modeling_metadata.py` (10)

### [MEDIUM] docs_clarity — Module docstring lists only 3 reserved keys; RESERVED_METADATA_KEYS has 4
`modeling_metadata.py:35-36`

Lines 34-36 enumerate the reserved-key vocabulary as (_input_metadata, _run_metadata, _univariate_metadata), but RESERVED_METADATA_KEYS at lines 80-81 contains four entries, additionally _consolidation_metadata. The module docstring undercounts the reserved set the rest of the module relies on.

**Fix:** Add `_consolidation_metadata` to the parenthetical list at lines 35-36 so the module docstring matches RESERVED_METADATA_KEYS.

### [MEDIUM] docs_clarity — build_run_metadata JAX field group omits the two balance_* flags it emits for multinomial
`modeling_metadata.py:565-566`

The JAX-hyperparameters bullet says the multinomial-only addition is just `focal_loss_gamma`, but lines 652-655 add three multinomial-only keys: focal_loss_gamma, balance_predictions_bool, and balance_train_bool. The inline comment at line 624 even references 'the `balance_*` flags', so the docstring is internally inconsistent.

**Fix:** Update lines 565-566 to read 'Plus `focal_loss_gamma`, `balance_predictions_bool`, and `balance_train_bool` for multinomial only.'

### [MEDIUM] docs_clarity — build_run_metadata 'Engine' field group omits 6 top-level split-config fields actually built
`modeling_metadata.py:560`

The 'Engine' field group lists only (analysis_type, model_engine, basis_function), but lines 614-619 unconditionally add random_seed_outer, spatial_cluster_num, test_proportion, session_split_max_attempts, session_split_widen_step, and session_split_widen_every. These six fields appear in every run-metadata block but are documented in no field group.

**Fix:** Add the six top-level fields to the Field groups section (extend 'Outer-loop layout' or add a 'Split configuration' group covering random_seed_outer / spatial_cluster_num / test_proportion / session_split_*).

### [MEDIUM] docs_clarity — build_run_metadata Field groups never document the pygam/sklearn CPU-path blocks it emits
`modeling_metadata.py:558-575`

The Field groups section documents only JAX-path blocks. For analysis_type in ('onset','category','params') the code at lines 670-695 emits a `pygam_hyperparameters` block (pygam engine) or a `sklearn_hyperparameters` block (sklearn engine, including a logistic_regression or ridge_regression sub-block). For all three CPU analysis types these are the only hyperparameter blocks present, yet they are entirely undocumented in the Field groups list.

**Fix:** Add a CPU-path field group documenting `pygam_hyperparameters` and `sklearn_hyperparameters` (and their nested keys) alongside the JAX field groups.

### [MEDIUM] tests — Legacy *_step_*.pkl directory branch in load_selection_results is untested
`modeling_metadata.py:1114-1125`

When given a directory with no selection_*.pkl but containing legacy *_step_*.pkl files, lines 1114-1125 build a distinct FileNotFoundError message via the legacy_present=True branch pointing users to consolidate_model_selection_results. The test suite only covers the empty-directory case (test_empty_directory_raises_file_not_found, legacy_present=False). The legacy-present message is a documented contract (docstring lines 1097-1099) but no test asserts it.

**Fix:** Add a test that writes a `foo_step_0.pkl` into tmp_path with no selection_*.pkl, calls load_selection_results(str(tmp_path)), asserts pytest.raises(FileNotFoundError), and asserts the message contains 'consolidate_model_selection_results'.

### [LOW] docs_clarity — inject_metadata **metadata_blocks parameter doc has a broken/confusing sentence
`modeling_metadata.py:936-939`

The parameter note reads 'Keyword arguments whose names must be drawn from RESERVED_METADATA_KEYS (passed without the leading underscore is *not* permitted — ...)'. The parenthetical 'passed without the leading underscore is not permitted' is grammatically broken and obscures the intent that callers must pass the full underscore-prefixed reserved name as the kwarg.

**Fix:** Reword to e.g. 'Keyword names must be exact members of RESERVED_METADATA_KEYS, i.e. include the leading underscore (e.g. `_input_metadata=...`). A name without the underscore is rejected because the embedded key must match the reserved name verbatim.'

### [LOW] docs_clarity — derive_camera_fps_field empty-input behavior is undocumented
`modeling_metadata.py:1034-1035`

The function returns an empty dict `{}` when camera_fr_dict is empty (lines 1034-1035), but the Returns section only states 'Single float when all sessions match; original dict otherwise', leaving the empty-map edge case undocumented.

**Fix:** Add a sentence to the Returns/Description noting that an empty input map returns an empty dict `{}`.

### [LOW] tests — build_run_metadata CPU-path fall-through (neither pygam nor sklearn) is untested
`modeling_metadata.py:670-695`

For analysis_type in ('onset','category','params'), lines 671-695 emit a pygam block only when model_engine=='pygam' and a sklearn block only when model_engine=='sklearn'. If model_engine is any other string, neither block is emitted and the metadata silently omits any hyperparameter sub-block. Tests only ever pass 'sklearn' or 'pygam' (TestBuildRunMetadata), so this no-op fall-through is uncovered.

**Fix:** Add a test calling _build with _full_modeling_settings(model_engine='jax') and analysis_type='onset', asserting common fields are present but both 'pygam_hyperparameters' and 'sklearn_hyperparameters' are absent.

### [LOW] tests — metadata_blocks_equal dict-vs-scalar leaf comparison is untested
`modeling_metadata.py:1183-1188`

metadata_blocks_equal recurses only when BOTH values are dicts (line 1183); when one key holds a dict and the matching value in the other block is a scalar/list, control falls to the else branch (lines 1186-1187) comparing with `va != vb`. Tests cover nested-dict-equal, nested-dict-mismatch, key-set mismatch, and ignore_keys but never the type-asymmetric case, a realistic schema-drift scenario consolidators rely on to reject merges.

**Fix:** Add a test asserting metadata_blocks_equal({'cfg': {'x': 1}}, {'cfg': 5}) is False, pinning the dict-vs-scalar leaf path.

### [LOW] tests — derive_experimental_condition case-insensitivity and mute/intact precedence are untested
`modeling_metadata.py:264-330`

The docstring (line 274) and implementation (line 312 `.name.lower()`) promise case-insensitive matching, but every TestDeriveExperimentalCondition case uses an all-lowercase filename, so the .lower() normalization is never exercised. Additionally the precedence rule (mute_* checks at lines 318-321 run before the intact_partners check at line 325) is untested: a filename containing both substrings would return a mute label, and no test pins this ordering.

**Fix:** Add (1) a test with an uppercased filename (e.g. 'MUTE_FEMALE_SESSIONS.TXT') asserting 'male_mute_partner'; and (2) a test with a filename containing both 'intact_partners' and 'mute_female' asserting the mute branch wins.


## `modeling_bases_functions.py` (9)

### [LOW] correctness — laplacian_pyramid raises opaque 'need at least one array to stack' for degenerate widths
`modeling_bases_functions.py:55`

When every level yields len(cens) <= 1, B stays empty and np.stack(B).T raises ValueError: need at least one array to stack. I reproduced this for width=1 at any levels/step. The candidate's claimed repro config (width=4, levels=6) does NOT crash (returns shape (4,8)), so that detail was wrong, but the underlying unguarded-crash bug is genuine for degenerate widths.

**Fix:** Guard before stacking: `if not B: raise ValueError(f'laplacian_pyramid produced no basis vectors for width={width}, levels={levels}, step={step}')`, giving an informative error instead of the numpy internal one.

### [LOW] docs_clarity — Typo 'Padds' in the w parameter docstring
`modeling_bases_functions.py:128`

The docstring for parameter w reads 'Padds or discards as needed.' 'Padds' is a misspelling of 'Pads'. Confirmed at line 128.

**Fix:** Change 'Padds or discards as needed.' to 'Pads or discards as needed.'

### [LOW] docs_clarity — splrep outputs rebind documented inputs positions/degree with different semantics
`modeling_bases_functions.py:217-220`

si.splrep returns (knot_vector, coefficients, k). Verified: input positions=[0,5,10,15,20] (5 elements) is rebound to a 9-element knot vector, and degree (input order) is rebound to splrep's returned k. Shadowing two documented input parameters with semantically different values is non-obvious and uncommented.

**Fix:** Unpack into distinct names, e.g. `knots, coe_ffs, k = si.splrep(...)`, and use those below; or add a comment noting splrep returns the full knot vector and spline order that overwrite the inputs.

### [LOW] performance — Unnecessary np.tile allocation in _normalizecols; broadcasting avoids the copy
`modeling_bases_functions.py:104`

A / np.tile(norms, (rows,1)) materializes a full [rows x cols] copy purely to match A's shape. Verified that broadcasting the (cols,) norm vector gives identical results after nan_to_num without the intermediate copy. Runs at the end of every raised_cosine() call.

**Fix:** Replace with `B = A / np.sqrt(np.sum(A ** 2, axis=0))`, which broadcasts the (cols,) vector across rows; keep the subsequent np.nan_to_num(B) guard.

### [LOW] tests — laplacian_pyramid normalize=False branch untested
`modeling_bases_functions.py:52-54`

Both pyramid tests use normalize=True. The normalize=False branch (raw Gaussian, no unit-norm step) is never exercised and yields non-unit-norm columns (distinct observable behavior).

**Fix:** Add a test laplacian_pyramid(width=64, levels=4, step=1.0, fwhm=1.0, normalize=False) asserting columns are NOT unit norm.

### [LOW] tests — laplacian_pyramid step!=1.0 (half-levels) path untested
`modeling_bases_functions.py:26-27`

No test uses a fractional step despite step=0.5 being documented. Confirmed step=0.5 yields 392 columns vs 232 for step=1.0 (width=64, levels=4), so the path is observably distinct and uncovered. (Dropped the candidate's unverified empty-cens false-branch sub-claim, whose repro config does not actually produce empty cens.)

**Fix:** Add a test with step=0.5 asserting strictly more basis columns than step=1.0 for the same width/levels.

### [LOW] tests — raised_cosine nbasis<nb (trim) and nbasis=None branches untested
`modeling_bases_functions.py:166-172`

Existing shape test uses nbasis=8 with nb=7, hitting only the zero-pad branch. Verified default nb=7 and nbasis=2 trims to 2 columns. Trim branch and nbasis=None pass-through are uncovered.

**Fix:** Add tests: nbasis=2 asserting .shape[1]==2 (trim); nbasis=None asserting column count equals natural nb (=7).

### [LOW] tests — raised_cosine w<rows (discard) and w=None branches untested
`modeling_bases_functions.py:177-183`

Shape test uses w=30 (zero-pad branch only). Verified w=3 trims rows to 3. The discard branch (kbasis[-w:, :]) and w=None pass-through are uncovered.

**Fix:** Add tests: w=3 asserting .shape[0]==3 (discard); w=None asserting the natural row count.

### [LOW] tests — bsplines periodic=True branch and non-default degree untested
`modeling_bases_functions.py:205-220`

Both bsplines tests use periodic=False, degree=3. The periodic=True path (per=periodic in si.splrep) and other degrees are uncovered, and periodic is a documented public option.

**Fix:** Add a periodic=True test asserting a finite [width, len(positions)] matrix, and a degree=1 test.


## `jax_multinomial_logistic_regression.py` (8)

### [MEDIUM] correctness — Smoothness penalty silently zeroed when n_time_bins <= smoothness_derivative_order
`jax_multinomial_logistic_regression.py:68`

In _loss_fn (lines 480-489) and its module-level mirror _multinomial_loss_static (lines 67-70) the smoothness term is dW = jnp.diff(W_reshaped, n=smoothness_derivative_order, axis=1) over an axis of length n_time. Confirmed empirically: with order=2 and n_time<3 (or order=1 and n_time<2) jnp.diff returns shape (n_feats, 0, n_classes), so jnp.sum(dW**2) is exactly 0 and smooth_loss vanishes with no error/warning. Reachable: model_selection.py computes new_T = T // bin_size (lines 2862, 3933) passed as n_time_bins, collapsing to 1 or 2 for large bin_resizing_factor. The fit then reports a non-zero lambda_smooth while the penalty has no effect. __init__ only validates the derivative order, never n_time_bins against it.

**Fix:** In fit, after validating shapes (~line 528), add: if self.n_time_bins <= self.smoothness_derivative_order: raise ValueError(...) explaining jnp.diff yields an empty array and the penalty is silently zero.

### [MEDIUM] performance — Default-path nested step() JITs per estimator instead of reusing a shape-keyed cache
`jax_multinomial_logistic_regression.py:578`

On the default path (_use_lax_loop=False) step is a nested closure defined inside fit() on every call (line 578), closing over optimizer and self.lambda_smooth/l2_reg/focal_gamma/smoothness_derivative_order. JAX keys its jit cache on the Python function object, so each newly constructed estimator triggers a fresh XLA compile of the grad/update graph on its first step even when input shapes match. This is the cost the module-level _multinomial_loss_static / _multinomial_train_loop_jit were written to avoid (comments lines 45-52, 97-107), but the default path does not route through them. The class docstring notes the inner-CV tuner builds ~175 estimators per outer fold (lines 313-314).

**Fix:** Hoist step to module scope or delegate to a module-level jitted function taking optimizer-free traced regularisation scalars plus static n_feats/n_time/smoothness_derivative_order, mirroring _multinomial_train_loop_jit, so the default path reuses one compiled graph across same-shape estimators.

### [MEDIUM] tests — Temporal-smoothness penalty never exercised at unit level with non-zero lambda_smooth
`jax_multinomial_logistic_regression.py:480`

Every fit-level test in tests/modeling/test_jax_multinomial_logistic_regression.py uses _fit_kwargs (line 60) which hard-codes lambda_smooth=0.0, so the smoothness machinery in _loss_fn (lines 480-489) and _multinomial_loss_static multiplies to zero and is never numerically validated by the dedicated suite. Order-1-vs-order-2 semantics and per-class scaling are the estimator's reason for existing yet are only exercised indirectly in test_pipeline_multinomial.py, so a penalty-math regression could pass the unit suite while only failing the slower pipeline test.

**Fix:** Add a unit test calling _loss_fn directly with hand-built W: order=1 yields zero smoothness for a flat filter and >0 for a ramp; order=2 yields ~zero for a linear ramp and >0 for a curved filter. Set lam_l2=0, focal_gamma=0 to isolate the term.

### [LOW] docs_clarity — predict_proba X-shape docstring wrong (says n_features, must be n_features * n_time_bins)
`jax_multinomial_logistic_regression.py:675`

predict_proba's X parameter is documented as shape (n_samples, n_features) on line 675, but the design matrix has one column per feature-by-time-bin. fit (line 500) and the class docstring (line 197) state (n_samples, n_features * n_time_bins), and predict_proba multiplies X by self.coef_.T whose second dim is n_features * n_time_bins (lines 697/699). The annotation is materially misleading.

**Fix:** Change the shape annotation on line 675 to (n_samples, n_features * n_time_bins).

### [LOW] docs_clarity — predict X-shape docstring wrong (says n_features, must be n_features * n_time_bins)
`jax_multinomial_logistic_regression.py:714`

predict's X parameter is documented as shape (n_samples, n_features) on line 714. predict delegates to predict_proba which requires n_features * n_time_bins columns. Inconsistent with fit (line 500) and the class docstring (line 197).

**Fix:** Change the shape annotation on line 714 to (n_samples, n_features * n_time_bins).

### [LOW] docs_clarity — Verbose 'Converged at iteration {i}' log disagrees with stored n_iter_ (i+1)
`jax_multinomial_logistic_regression.py:613`

On convergence completed_iter = i + 1 (line 606) and n_iter_ = completed_iter, documented as 1-indexed (line 262). The verbose message on line 613 prints 'Converged at iteration {i}' using the 0-indexed loop counter, so the printed number is one less than n_iter_ reports.

**Fix:** Print the 1-indexed value: print(f"Converged at iteration {i + 1} with combined-update norm {diff:.2e}").

### [LOW] docs_clarity — Nested step() helper has no docstring despite non-obvious JIT-static logic
`jax_multinomial_logistic_regression.py:578`

The nested step closure (line 578) takes seven positional params with static_argnums=(4, 5) but has no docstring explaining w_batch is the unit-mean per-class weight vector reused as focal-alpha and smoothness scaler, nor why n_feats/n_time are static. Every other callable in the module carries a detailed docstring and the project convention requires detailed docstrings, so this is a doc gap.

**Fix:** Add a short docstring to step clarifying that w_batch is the unit-mean class-weight vector and that n_feats/n_time are static JIT args needed to reshape W for the smoothness penalty.

### [LOW] tests — NotFittedError path in predict_proba/predict is untested
`jax_multinomial_logistic_regression.py:693`

predict_proba calls check_is_fitted(self, ['coef_', 'intercept_', 'log_priors_']) on line 693 and predict routes through it (line 728). No test calls predict/predict_proba before fit, so the guard is never verified; a refactor dropping one required attribute name or setting an attribute late in fit would not be caught.

**Fix:** Add a test constructing the estimator without fit and asserting pytest.raises(NotFittedError) for both model.predict(X) and model.predict_proba(X).


## `manifold_torus_regression.py` (7)

### [HIGH] docs_clarity — fit docstring falsely claims it builds a snap kd-tree and leaves snap behaviour unchanged
`manifold_torus_regression.py:178-183`

The fit docstring (lines 178-183) claims the training statistics include a 'kd-tree on the 4-D torus embedding for snapping' computed 'exactly as in SmoothBivariateRegression so the inherited evaluate_metrics and predict snap behaviour are unchanged.' The code at lines 256-274 deliberately builds NO snap kd-tree and sets self._train_kdtree = None, disabling snapping. The docstring misrepresents a load-bearing design decision.

**Fix:** Reword lines 178-183 to state NO snap kd-tree is built (self._train_kdtree = None), that only the circular-mean centroid and wrap-aware inverse covariance are computed as in the parent, and that snap is intentionally disabled (see inline rationale at lines 256-271).

### [MEDIUM] docs_clarity — Class docstring claims _train_kdtree is inherited unchanged, but fit overrides it to None
`manifold_torus_regression.py:63-68`

The class docstring (lines 63-68) lists '_train_kdtree' among attributes 'inherited unchanged, so the persisted metric bundle is identical in schema to the coordinate model.' But fit explicitly sets self._train_kdtree = None (line 274), differing from the parent which builds a cKDTree. Contradicts the inline comment (256-271) and predict docstring (293-295).

**Fix:** Remove _train_kdtree from the 'inherited unchanged' list or qualify it: note it is deliberately set to None (snapping disabled) while train_mean_ / train_cov_inv_ are computed as in the coordinate model.

### [MEDIUM] tests — Untested ValueError: X column-count mismatch vs n_features*n_time_bins
`manifold_torus_regression.py:213-218`

fit() raises ValueError when n_inputs != n_features*n_time_bins (lines 213-218). No test in tests/modeling/test_manifold_torus_regression.py exercises this branch; all tests pass correctly-shaped X, so the guard and its message are never asserted.

**Fix:** Add a test with n_features=2, n_time_bins=5 (expected 10 columns) but X with 7 columns, then pytest.raises(ValueError, match=r"n_features\(2\) \* n_time_bins\(5\) = 10 columns").

### [MEDIUM] tests — Untested ValueError: target y not exactly 2 columns
`manifold_torus_regression.py:219-222`

fit() raises ValueError when y.shape[1] != 2 (lines 219-222). No test passes a y with 1 or 3 columns; check_X_y(multi_output=True) does not reject a 3-col y, so this is the only line catching that misuse, and it is uncovered.

**Fix:** Add a test with a valid X for n_features=1,n_time_bins=4 and y of shape (n,3), then pytest.raises(ValueError, match='exactly 2 columns').

### [LOW] tests — Untested period validation path (_validate_metric_period) in fit
`manifold_torus_regression.py:207`

fit() calls _validate_metric_period(self.metric, self.period) at line 207, which raises ValueError on non-positive/non-finite period (manifold_metric.py:72-75). test_metric_guard_rejects_non_torus only covers the metric!='torus' guard (203-206); the period branch is never hit with metric='torus' and an invalid period.

**Fix:** Add tests constructing _torus_kwargs(1,1) with period=0.0 and period=-1.0, asserting pytest.raises(ValueError, match='positive finite') on .fit.

### [LOW] tests — encode/decode and fit never tested with a non-unit period
`manifold_torus_regression.py:99-130`

_encode (line 99: 2*pi*Y/period), _decode (lines 128-129: ang/(2*pi)*period % period), and the fit/predict chain scale by self.period, but every test fixes period=1.0. A bug dropping the *self.period factor in _decode would not be caught; test_encode_decode_round_trip only validates period=1.0.

**Fix:** Parametrize test_encode_decode_round_trip over period in (1.0, 2*np.pi, 6.28), generating Y in [0,period), asserting _decode(_encode(Y)) == Y; optionally fit with a non-unit period and assert predictions stay in [0, period).

### [LOW] tests — n_features==1 smoothness branch never validated bit-for-bit against the reference solve
`manifold_torus_regression.py:156-157`

_smoothness_penalty branches: n_features==1 returns block directly (line 157), n_features>1 returns block_diag (line 158). test_closed_form_matches_normal_equations only runs n_features=2 (block_diag branch). The n_features==1 path is only hit by shape/contract tests that never compare coef_/intercept_ to _manual_closed_form, so a single-feature penalty defect would pass undetected.

**Fix:** Add a case to test_closed_form_matches_normal_equations with n_features=1, n_time_bins>=4, comparing coef_/intercept_ to _manual_closed_form (which already handles n_features==1 at line 115).


## `main_model_selection_dispatcher.py` (4)

### [HIGH] correctness — Broad except swallows failures and returns exit code 0, defeating SLURM FAIL detection
`main_model_selection_dispatcher.py:176-178`

dispatch_model_selection wraps every selector call in try/except Exception, prints the traceback, then falls through to return normally, so the process exits 0 even when model selection crashes. model_selection_behavior.sh relies on `set -e` (line 47) and `#SBATCH --mail-type=FAIL` (line 10) to detect/notify on failures; because the dispatcher returns 0, set -e never trips and SLURM never marks the job FAILED, making a crashed run indistinguishable from a successful one. The module docstring (lines 23-24) claims the traceback is captured for debugging cluster failures, but capturing without re-raising/non-zero exit turns a hard failure into a silent one. Verified: the except block ends at traceback.print_exc() with no re-raise or sys.exit.

**Fix:** After printing the traceback in the except block, re-raise (`raise`) or call `sys.exit(1)` so the process exits non-zero, preserving the verbose traceback while letting set -e and SLURM FAIL mail trigger.

### [MEDIUM] correctness — Dispatcher always overrides the continuous framework's designed p_val default (0.05 -> 0.01)
`main_model_selection_dispatcher.py:159-168, 202`

--pval defaults to 0.01 (line 202) and dispatch_model_selection always passes p_val=args.pval explicitly to every selector. continuous_vocal_manifold_model_selection has its own designed default p_val=0.05 (model_selection.py:3641, confirmed). Running 'continuous' without --pval silently applies 0.01 instead of the framework's intended 0.05, tightening candidate screening for the continuous/Gaussian manifold framework. The SLURM script also hardcodes PVAL=0.01, so 0.05 is never reached in practice. Silent behavioral divergence from the selector's documented default.

**Fix:** Set argparse --pval default to None and only pass p_val=args.pval when not None (letting each selector use its own default), or branch the continuous case to default to 0.05 when --pval is unset; update model_selection_behavior.sh if 0.05 is intended.

### [MEDIUM] tests — output_directory kwarg forwarding is never asserted in any selector route test
`main_model_selection_dispatcher.py:118,129,142,153,165`

dispatch_model_selection forwards args.output_dir to every selector as output_directory= (lines 118, 129, 142, 153, 165). Verified in tests/modeling/test_dispatchers.py: test_onset_route only asserts use_top_rank_as_anchor, p_val, univariate_results_path; no test asserts output_directory == args.output_dir or input_data_path == args.input_path. A regression mis-wiring output_dir would pass the whole suite.

**Fix:** In test_onset_route, after kwargs = spy.call_args.kwargs, add assert kwargs['output_directory'] == args.output_dir and assert kwargs['input_data_path'] == args.input_path.

### [LOW] correctness — Unreachable else branch for unknown analysis_type is dead code that also exits 0 on its impossible path
`main_model_selection_dispatcher.py:170-172`

The else branch printing 'FATAL: Unknown analysis type' and returning is unreachable from the CLI: argparse restricts --analysis_type to choices=['onset','category','params','multinomial','continuous'] (lines 185-187). A test does call dispatch_model_selection directly with 'bogus' to cover it, so it is reachable via direct programmatic calls. Even when reached it returns (exit 0) on a 'FATAL' condition, repeating the silent-success pattern.

**Fix:** If keeping defensive handling for direct programmatic calls, raise ValueError instead of printing and returning so the failure is not silent; otherwise document it as deliberately defensive.
