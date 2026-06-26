# Analyses subsystem review

_Verified line-by-line sweep of 11 files (~12.7k LOC): 174 findings. Report-first._

## Summary
- by severity: high 7 · medium 46 · low 121
- by dimension: tests 66 · docs_clarity 45 · performance 28 · correctness 23 · dead_code_naming 12


## `compute_behavioral_features.py` (27)

### [MEDIUM] correctness — get_back_angles optimizer can crash on sentinel return (TypeError on list index)
`compute_behavioral_features.py:1046`

distance_to_x_axis calls get_rotation([0.0, argument_ang[0], argument_ang[1]]), so the optimizer's candidate roll lands in get_rotation's arg[1]. get_rotation (line 998) returns the list sentinel ([-1],[-1]) whenever abs(arg[1])>pi/2. Nelder-Mead is unconstrained (only the x0 start is bounded to (-pi/4,pi/4), line 1063-1071) and can probe roll beyond pi/2 during simplex expansion, after which rot_check is the Python list [-1] and line 1049 (rot_check[:, 0]) raises 'TypeError: list indices must be integers or slices, not tuple', aborting per-session feature computation. The pi/2 guard was written for the 3-angle case but the 2-parameter objective feeds an unconstrained roll into it.

**Fix:** In distance_to_x_axis, handle the sentinel before indexing (e.g. `if isinstance(rot_check, list): return np.inf`), so out-of-range candidates yield a large finite objective instead of a sentinel that is then indexed as an array.

### [MEDIUM] docs_clarity — Stale 'defaults to None' for required parameter diff_bins
`compute_behavioral_features.py:167-168`

The docstring states `diff_bins : int  Number of bins for the central difference derivative; defaults to None.` but the signature (line 136) declares `diff_bins: int` with no default — it is a required positional argument, and the sibling capture_fr param even documents 'Required.' The 'defaults to None' clause is false and stale.

**Fix:** Drop the false default, e.g. `Number of bins for the central difference derivative; required.`

### [MEDIUM] docs_clarity — Stale 'defaults to None' on required calculate_speed parameters
`compute_behavioral_features.py:563-566`

calculate_speed documents `capture_framerate (int / float)  Recording camera framerate; defaults to None (fps).` and `smoothing_time_window (int / float)  Time window to perform smoothing over; defaults to None (s).` Neither has a default in the signature (both are required positionals), so 'defaults to None' is incorrect.

**Fix:** Drop 'defaults to None', e.g. `Recording camera framerate, in fps; required.` and `Time window to perform smoothing over, in seconds; required.`

### [MEDIUM] docs_clarity — Incorrect description of back yaw in get_back_angles docstring
`compute_behavioral_features.py:957-958`

The docstring summarizes outputs as `pitch: angle between the back and the horizontal plane` and `yaw: angle between the back and the z-axis`. Pitch is correct (line 1084 computes arctan2(z, sqrt(x^2+y^2))). Yaw is wrong: it is computed as -arctan2(y, x) (lines 1094-1095), the azimuthal angle in the XY plane (rotation about z), not the 'angle between the back and the z-axis' (which would be the complement of pitch).

**Fix:** Reword to e.g. `yaw: azimuthal angle of the back in the horizontal (XY) plane, i.e. rotation about the z-axis (signed, from -arctan2(y, x))`.

### [MEDIUM] performance — generate_feature_distributions uses Python bin loops instead of np.histogram/np.histogram2d
`compute_behavioral_features.py:106-129`

Both branches loop over bin edges in Python and call np.sum over the full feature array on every bin. The 1D branch does num_bins full-array passes; the 2D spatial branch does bins_in_one_dir^2 (14x14=196) full-array passes, each allocating two boolean masks over all n_frames. np.histogram/np.histogram2d computes the identical occupancy in one optimized pass.

**Fix:** Replace the 1D loop with `counts, bin_edges = np.histogram(feature_arr, bins=np.linspace(min_val, max_val, num_bins + 1)); occ_array = counts / camera_fr` and the 2D loop with np.histogram2d. Note np.histogram uses half-open [lo,hi) bins vs the current (lo,hi]; verify the boundary convention is acceptable (see the related occupancy-undercount finding).

### [MEDIUM] tests — get_euler_ang gimbal-lock branch (problematic_indices) is untested
`compute_behavioral_features.py:904, 931-946`

The only test (test_get_euler_ang_identity_returns_zero_angles, test_analyze.py:544) feeds the identity matrix and exercises only the good_indices path (lines 911-929). The gimbal-lock branch (temp < 1e-4, line 904), which sets roll=0 and recovers pitch/yaw from indices 2,3,4 (lines 931-946), plus the np.setdiff1d split, has zero coverage; a sign error or index typo there would pass CI silently.

**Fix:** Add a test constructing a rotation matrix with row-2 giving pitch +90 deg (temp<1e-4); assert roll==0.0, pitch==+90, and yaw matches atan2(R[1,0],R[1,1]). Add a mirror -90 deg case.

### [MEDIUM] tests — calculate_sei default v_max (np.nanpercentile) branch untested
`compute_behavioral_features.py:330-331`

Both SEI tests (test_analyze.py:2886 and 2918) pass explicit v_max=0.1, so the `if v_max is None: v_max = np.nanpercentile(speed_arr, 99)` default branch (lines 330-331) never executes. This is the production path (save_behavioral_features_to_file calls calculate_sei without v_max), so the actual default normalization and NaN handling of speed_arr are unverified.

**Fix:** Add a test calling calculate_sei with v_max=None and a multi-frame speed_arr containing NaNs; assert the result is finite and equals the explicit-v_max=np.nanpercentile(speed_arr,99) result.

### [MEDIUM] tests — get_head_root below-tolerance (degenerate-norm) NaN branch untested
`compute_behavioral_features.py:687-697`

test_get_head_root_propagates_nan_in_inputs (test_analyze.py:537) only covers NaN-in-input propagation. The degenerate-geometry branch — h_x_len < spatial_resolution_tolerance (Nose coincident with Head) or h_z_len < tolerance (forward axis parallel to inter-aural axis) set to NaN — is never exercised with finite inputs.

**Fix:** Add a test: (a) Nose==Head asserts NaN matrix; (b) Ear_L-Ear_R collinear with Nose-Head (all points on x-axis) asserts NaN matrix from the h_z branch.

### [MEDIUM] tests — get_egocentric_direction yaw sign and target-behind cases untested
`compute_behavioral_features.py:515-528`

Existing tests cover only target-in-front (yaw==0, test 480), target-above (pitch==90, test 489), and below-tolerance NaN (test 497). The yaw computation atan2(v_local.y, v_local.x) (line 515) is never tested for sign-bearing yaw (left/right) or target-behind (yaw==+/-180); a sign flip or argument swap would pass all current tests.

**Fix:** Add a test: head_root=I, head_pivot=0; target=[0,1,0] assert yaw==+90; target=[0,-1,0] assert yaw==-90; target=[-1,0,0] assert abs(yaw)==180. Optionally target=[0,0,-1] -> pitch==-90.

### [LOW] correctness — Occupancy histogram silently drops values exactly equal to min_val
`compute_behavioral_features.py:125-129`

Both the 1D (line 127) and 2D (lines 111-117) bin assignments use the half-open rule (arr > bin_edges[i-1]) & (arr <= bin_edges[i]). The lowest bin requires arr > bin_edges[0] == min_val, so any frame whose value is exactly min_val is excluded from every bin. For features with lower bound 0 (speed, neck_elevation, tail_curvature, distance/SEI features) this drops genuinely-zero frames from the occupancy total. Affects only the distribution PDFs/occupancy arrays, not the saved CSV feature values.

**Fix:** Special-case the first bin to use >= bin_edges[0] (or use np.histogram, which closes the last bin), and apply the same fix to the 2D branch.

### [LOW] correctness — Speed smoothing kernel becomes all-NaN when floor(smoothing_time_window*fps)==0
`compute_behavioral_features.py:574-576`

Gaussian1DKernel(stddev=int(np.floor(smoothing_time_window * capture_framerate))) yields stddev=0 whenever the product < 1. With the hardcoded 0.015 s window (line 1882) this happens for capture rates <= 66 fps; astropy then builds a degenerate kernel and convolve returns an all-NaN speed trace (and its derivatives, acceleration, SEI speed gate) with only a RuntimeWarning. Latent because the playpen runs at 150 fps.

**Fix:** Clamp the stddev to a minimum of 1: stddev=max(1, int(np.floor(smoothing_time_window * capture_framerate))), or raise if the product rounds to 0.

### [LOW] correctness — calculate_tail_curvature divides by segment length with no zero/NaN guard
`compute_behavioral_features.py:404-411`

tangent_vectors = segment_vectors / segment_lengths[..., np.newaxis] (line 404) and curvature = np.diff(tangent_vectors)/segment_lengths[:, :-1, np.newaxis] (line 410) divide by raw inter-node segment lengths with no spatial_resolution_tolerance guard, unlike get_head_root/get_back_root/get_egocentric_direction which floor tiny norms to NaN. If two consecutive tracked tail nodes coincide (segment length 0), the result is 0/0 or x/0, propagating NaN/inf into the per-frame mean curvature with no warning.

**Fix:** Set segment_lengths below spatial_resolution_tolerance to np.nan before the two divisions, matching the tolerance pattern used elsewhere, so degenerate frames yield explicit NaN.

### [LOW] correctness — Occupancy y-tick math produces a negative/unordered upper tick when occupancy max < 10
`compute_behavioral_features.py:1694-1696`

ax.set_yticks(ticks=[0, int(np.ceil(temp_ymax)) - 10], labels=['0', f'{int(np.ceil(temp_ymax)) - 10}']). When peak occupancy (temp_ymax, seconds) is below 10, int(np.ceil(temp_ymax)) - 10 is negative, putting the upper tick below 0 and out of order with the '0' tick; at exactly 10 both ticks collapse to [0,0]. Cosmetic mislabeling on low-occupancy panels; the hardcoded -10 offset has no axis-aware justification.

**Fix:** Use the actual axis max for the upper tick (ticks=[0, temp_ymax] with a formatted label) or clamp upper = max(1, int(np.ceil(temp_ymax))) instead of subtracting 10.

### [LOW] dead_code_naming — Unused local back_rotator from get_rotation unpacking
`compute_behavioral_features.py:1077`

In get_back_angles, `rotated_back_directions, back_rotator = get_rotation([0.0, temp_angles_back[0], temp_angles_back[1]])` (line 1077) binds back_rotator, which is never referenced afterward (only this line and the docstring at 991 mention it). Ruff F841 does not flag tuple-unpack targets, so it slips through. Per the project's proactive dead-code convention this is a vestigial binding.

**Fix:** Replace with `rotated_back_directions, _ = get_rotation([0.0, temp_angles_back[0], temp_angles_back[1]])`.

### [LOW] dead_code_naming — Unused local rot_m in distance_to_x_axis
`compute_behavioral_features.py:1046`

Inside the Nelder-Mead objective, `rot_check, rot_m = get_rotation([0.0, argument_ang[0], argument_ang[1]])` (line 1046) binds rot_m, but only rot_check is used in the subsequent check_vec computation; rot_m is never referenced again. Like back_rotator, ruff F841 ignores it as a tuple-unpack target.

**Fix:** Replace with `rot_check, _ = get_rotation([0.0, argument_ang[0], argument_ang[1]])`.

### [LOW] docs_clarity — Unclosed parenthesis in tail-curvature comment
`compute_behavioral_features.py:2081`

The inline comment `# # tail curvature (arbitrary units` opens a parenthesis never closed, leaving the unit annotation truncated.

**Fix:** Close it: `# # tail curvature (arbitrary units)`.

### [LOW] docs_clarity — Module-header [B] labels mislabel the symmetric distance features
`compute_behavioral_features.py:13`

The module docstring's [B] list begins `(0) Nose distance ... (3) TTI distance`, but the emitted columns are the symmetric pair distances nose-nose, TTI-TTI, nose-TTI, TTI-nose (columns built at lines 2450/2460/2470/2480). 'Nose distance' / 'TTI distance' obscures that these are nose-to-nose and TTI-to-TTI inter-animal distances.

**Fix:** Rename header entries to match emitted columns, e.g. `(0) Nose-Nose distance ... (3) TTI-TTI distance`, mirroring the DataFrame column names.

### [LOW] docs_clarity — Egocentric social-angle column-layout comment is asymmetrically worded
`compute_behavioral_features.py:2404-2405`

The comment reads `columns 0..3 = yaw to {m2.Nose, m1.Nose seen from m2, m2.TTI, m1.TTI seen from m2}`. Columns 1 and 3 carry the observer qualifier ('seen from m2'), but the parallel entries 'm2.Nose'/'m2.TTI' omit it, so a reader cannot tell columns 0 and 2 are 'm2.Nose seen from m1' / 'm2.TTI seen from m1' (code at lines 2418/2429 uses head_root=root1, head_pivot=head1).

**Fix:** Make all four entries symmetric, e.g. `columns 0..3 = yaw of {m2.Nose seen from m1, m1.Nose seen from m2, m2.TTI seen from m1, m1.TTI seen from m2}; columns 4..7 = matching pitch.`

### [LOW] docs_clarity — Pitch range stated inconsistently between docstring body and Returns
`compute_behavioral_features.py:503-505`

In get_egocentric_direction the docstring body states pitch is `in (-90, 90] deg` (line 448) while the Returns section states `[-90, 90] degrees` (lines 503-505). Pitch derives from arctan2(z, sqrt(x^2+y^2)) whose range is the closed [-90, 90], so the body's `(-90, 90]` is the inaccurate one.

**Fix:** Make both read `[-90, 90]`, correcting line 448.

### [LOW] performance — Repeated mouse_nodes.index() linear scans recomputed per mouse
`compute_behavioral_features.py:1870-2129`

mouse_nodes is a Python list, so each .index(name) is an O(n_nodes) scan; there are 53 such calls. Several are mouse-independent yet recomputed every mouse_num iteration: exclude_tail_points and exclude_tail_mask (lines 1870-1877), head_input_arr indices (1916-1944), back_root indices, and tail_points indices (2087-2120).

**Fix:** Build `node_idx = {name: i for i, name in enumerate(mouse_nodes)}` once and read node_idx['Nose'] etc., and hoist the mouse-independent index lookups and exclude_tail_mask above the `for mouse_num` loop.

### [LOW] performance — Per-Series with_columns rebuilds the DataFrame many times per mouse
`compute_behavioral_features.py:2146-2341`

Each behavioral_features_df = behavioral_features_df.with_columns(pls.Series(...)) adds one column and reassigns the frame; there are 80 such calls. Polars with_columns materializes a new DataFrame each call, so adding columns one at a time is O(columns^2) in frame-construction overhead and creates many intermediate frames.

**Fix:** Accumulate the per-mouse/per-pair series in a list and issue one `behavioral_features_df.with_columns(series_list)` per mouse/pair, or build a name->ndarray dict and construct the frame once.

### [LOW] performance — Spatial occupancy re-selects spaceX/spaceY from the DataFrame and double-copies
`compute_behavioral_features.py:2688-2700`

The spatial branch calls behavioral_features_df.select('<mouse>.spaceX').to_numpy() and the spaceY equivalent, each wrapped again in np.array() (an extra copy of an already-contiguous ndarray). Runs once per mouse, so not hot, but the head_position arrays already hold spaceX/Y in memory and the polars select round-trip plus redundant np.array() copy are avoidable.

**Fix:** Stack directly from head_position (np.stack((head_position[mouse_num,:,0], head_position[mouse_num,:,1]), axis=1)); at minimum drop the np.array() wrapper since to_numpy() already returns an ndarray.

### [LOW] performance — Unnecessary defensive .copy() of reshaped rotation matrix in get_euler_ang
`compute_behavioral_features.py:898`

rot_matrix_reshaped = np.reshape(rot_matrix, shape=(rot_matrix.shape[0], 9)).copy() forces a full copy of the (n_frames, 9) array. The subsequent code only reads columns (arctan2, multiplications) and never mutates rot_matrix_reshaped in place, so the copy is a redundant allocation per mouse.

**Fix:** Drop the trailing .copy(); np.reshape already returns a new array object and no in-place write occurs in this function.

### [LOW] tests — calculate_sei zero-body-length NaN guard untested
`compute_behavioral_features.py:310-311`

Line 311 sets body_length[body_length == 0] = np.nan to guard an observer whose nose and TTI coincide. No test drives obs_nose == obs_tti, so NaN propagation through d_norm (line 315) into sei is unverified; removing the guard (div-by-zero) would not be caught.

**Fix:** Add a single-frame test with obs_nose == obs_tti (idx_nose and idx_tti at the same coordinate); assert np.isnan(sei[0]).

### [LOW] tests — get_back_root root_inv branch has no NaN/degenerate-input test
`compute_behavioral_features.py:803-829`

test_get_back_root_returns_nan_when_neck_and_tti_coincident (test_analyze.py:2994) covers only the default branch's coincident-points NaN. The root_inv branch (lines 803-829) has its own independent root_len < tolerance NaN guard (line 809); its coincident-Neck/TTI degenerate case is untested, although root_inv is the production branch for back pitch/yaw.

**Fix:** Add a test feeding neck_pos==tti_pos with root_method='root_inv' and assert an all-NaN matrix (under filterwarnings('ignore::RuntimeWarning')).

### [LOW] tests — plot_feature_distributions None-guard (no-op path) untested directly
`compute_behavioral_features.py:1492-1520`

plot_feature_distributions is exercised only via the end-to-end save test (test 2744). Its guard `if feature_dict is not None and mouse_id_list is not None:` (line 1521) and the kwarg type-coercion (lines 1492-1512 coerce non-dict/non-list kwargs to None) are never tested directly; a feature_dict=None call should produce no PDF and not crash.

**Fix:** Add a test instantiating FeatureZoo and calling plot_feature_distributions with feature_dict omitted; assert no PDF is created and no exception is raised.

### [LOW] tests — calculate_derivatives is_angle second-derivative no-wrap and multi-column behavior untested
`compute_behavioral_features.py:193-204`

test_calculate_derivatives_angle_wraps_at_pi (test_analyze.py:440) verifies first-derivative wrapping but discards second_der via `_` and uses a single column. The documented contract that the SECOND derivative is intentionally left unwrapped (lines 199-204) and the per-column independence of the global first_der>180/<-180 wrap (lines 194-195) are never asserted.

**Fix:** Add a 2-column angle test where one column crosses +/-180 and the other does not; assert the wrapped column's first_der stays bounded, the other is unaffected, and second_der at a wrap frame is not re-wrapped.


## `compute_neuronal_tuning_curves.py` (26)

### [HIGH] tests — _save_partial_to_cluster_pkl triage_stats one-level merge is untested
`compute_neuronal_tuning_curves.py:1263-1273`

The module's cross-path invariant (running behavioral and vocal in either order never clobbers the other) is enforced solely by the special-cased one-level merge of triage_stats at lines 1263-1271. Grep confirms no test references _save_partial_to_cluster_pkl or exercises a second write into an existing pkl. A regression here silently drops one modality's triage stats.

**Fix:** Add a test: call _save_partial_to_cluster_pkl twice on the same cluster (first {'triage_stats': {'vmi': {...}}, ...}, then {'triage_stats': {'behavioral': {...}}, ...}); reload and assert both triage_stats['vmi'] and ['behavioral'] survive and non-triage keys coexist; also assert a non-dict triage_stats takes the wholesale-overwrite branch.

### [HIGH] tests — Partner emitter side of vocal compute never exercised
`compute_neuronal_tuning_curves.py:1788-1797`

The synthetic fixture builds a single-emitter session (track_names=[b'm1'], include_partner_vocalization_tuning_bool=False), so the partner branch (1792-1797) and the two-side sort-by-count self/partner selection (1783-1785) are entirely uncovered, as is the partner role through _compute_one_cluster_vocal's accumulation/VMI loops.

**Fix:** Add a two-emitter fixture variant (e.g. m1 ~80 USVs, f1 ~40) with include_partner_vocalization_tuning_bool=True and n_usv_min_partner below the smaller count; assert both 'self' and 'partner' keys with correct emitter strings, and that outputs are keyed by both emitters with role/sex set.

### [MEDIUM] correctness — Non-integer categorical columns with nulls crash _build_vocal_side_precompute
`compute_neuronal_tuning_curves.py:1840-1846`

For each CATEGORICAL_FEATURE the null filter `cat_values[cat_values >= 0]` is applied ONLY when dtype.kind is in 'iu'. polars .to_numpy() on an integer column that contains nulls yields a float array (NaN) or object array (None). In those branches `non_null` keeps the sentinels, so `unique_cats = np.array(sorted(set(non_null.tolist())))` (object/None -> TypeError) and `cat_to_idx = {int(c): ...}` (int(nan) -> ValueError, int(None) -> TypeError) at line 1846 crash. This construction is undefended (unlike per-element `_safe_idx`), and the call site at line 1486 has no try/except, so a single null in a non-int-typed category column aborts the whole vocal side for the session. Conditional on the category columns actually containing nulls after the `vae_supercategory != 0` filter; plausible with real USV summary CSVs.

**Fix:** Drop null sentinels regardless of dtype before building unique_cats, e.g. `non_null = np.array([c for c in cat_values.tolist() if c is not None and not (isinstance(c, float) and np.isnan(c))])`, then keep the `>= 0` integer filter so `sorted(set(...))` and `int(c)` only see integer-coercible labels.

### [MEDIUM] dead_code_naming — Dead attribute: NeuronalTuning._segmentation_path is set but never read
`compute_neuronal_tuning_curves.py:1215-1219`

self._segmentation_path is assigned in __init__ (1215-1219) but never read anywhere in this file. Repo grep confirms the only readers are the unrelated visualization class in make_neuronal_tuning_figures.py and its test; the compute class never loads the npz. The categorical vocal compute derives categories from the USV summary CSV columns (CATEGORICAL_FEATURES), not from the segmentation file.

**Fix:** Remove the self._segmentation_path assignment (lines 1215-1219) from NeuronalTuning.__init__.

### [MEDIUM] dead_code_naming — Misleading __init__ docstring: segmentation path is not used by categorical vocal compute
`compute_neuronal_tuning_curves.py:1189-1191`

The __init__ docstring claims it pins the segmentation file 'used by the categorical vocal compute.' The categorical vocal compute never loads usv_latent_embedding_segmentation.npz; it reads vae_/qlvm_ category columns straight from *_usv_summary.csv. The npz is consumed only by the separate visualization class.

**Fix:** Remove the clause about pinning the segmentation path from the __init__ docstring (and the attribute per the related finding), or correct it to say the categorical compute reads category labels from the USV summary CSV columns.

### [MEDIUM] docs_clarity — Class docstring lists `shuffle_chunk_size`, a key never read by this class
`compute_neuronal_tuning_curves.py:1177`

The class docstring's vocal-path key list ends with `shuffle_chunk_size` (line 1177), but grep shows it is never read in this file (only present at line 1177 and in analyses_settings.json). It is consumed elsewhere in the pipeline, not by NeuronalTuning, so listing it here as a key this class reads is incorrect and misleading.

**Fix:** Remove `shuffle_chunk_size` from the vocal-path key list in the class docstring (line 1177).

### [MEDIUM] docs_clarity — Class docstring omits settings keys the code actually reads
`compute_neuronal_tuning_curves.py:1167-1177`

The behavioral-path key list omits smoothing_sd (read at 1560), shuffle_seed (1582), behavioral_min_occupancy_seconds (1697-1698), and circular_features (1959); the vocal-path list omits smoothing_sd (2548) and shuffle_seed (2564). The documented key inventory is incomplete on both paths.

**Fix:** Add smoothing_sd, shuffle_seed, behavioral_min_occupancy_seconds, circular_features to the behavioral key list, and smoothing_sd, shuffle_seed to the vocal key list in the class docstring.

### [MEDIUM] docs_clarity — _compute_one_cluster_vocal Returns docstring omits the `triage_stats` top-level key
`compute_neuronal_tuning_curves.py:2530-2535`

The Returns section (2533-2534) lists usv_peth, usv_property_tuning, usv_category_tuning, usv_category_peth, usv_metadata, but the method also builds partial["triage_stats"] (initialized at 2717 with the vmi block, extended by _attach_vocal_triage_stats at 2890). triage_stats is a primary output and is special-cased in the pkl merge, yet undocumented here.

**Fix:** Add `triage_stats` (carrying vmi plus per-modality vocal triage blocks) to the Returns key list.

### [MEDIUM] performance — Behavioral feature columns re-converted polars->numpy on every offset and cluster
`compute_neuronal_tuning_curves.py:1590`

Inside _compute_one_cluster_behavioral, `np.array(behavioral_data[column])` (line 1590) runs inside the `for one_offset` loop (1565) and `for column` loop (1586). The numpy materialization is independent of one_offset and of the cluster, yet each column is re-extracted O(offsets) times per cluster and again for every cluster (the same behavioral_data is reused). Spatial columns at 1657-1658 have the same redundant re-extraction.

**Fix:** Materialize each needed column once per session (e.g. a {column: np.asarray(behavioral_data[column])} dict built in _load_behavioral_inputs, which already returns the reused bundle) and index that dict at 1590/1657-1658.

### [MEDIUM] performance — Occupancy binning recomputed per temporal offset though offset-invariant
`compute_neuronal_tuning_curves.py:188-192`

generate_ratemaps recomputes occ_idx = searchsorted(bin_edges, feature_arr) (188) and occ_counts = bincount(...) (191-192) on every call. These depend only on feature_arr/bin_edges/num_bins, none of which change across the temporal_offsets loop (only the spike side shifts). For O offsets x F columns x clusters, the full searchsorted+bincount over n_frames is repeated redundantly. The 2D spatial occupancy (165-175) has the same property.

**Fix:** Split occupancy out of the spike-count path: compute occ_idx/occ_counts (and occ_2d) once per (column, num_bins) per session and pass cached occupancy into a spike-only routine, or memoize on (id(feature_arr), num_bins, min_val, max_val). bin_edges/bin_centers are likewise cacheable.

### [MEDIUM] performance — Per-category membership mask rebuilt on every shuffle iteration
`compute_neuronal_tuning_curves.py:2695-2703`

In _compute_one_cluster_vocal, inside the k-loop (2604, n_shuffles+1 iterations) -> per role -> per cat_feat (2673) -> per category (2695), line 2698 computes member = cat_idx == i_c and 2702 reduces per_anchor_bin_count[member].sum(axis=0). cat_idx (anchor_cat_idx_dense) is k-invariant precompute, yet the boolean masks are rebuilt every shuffle x 4 features x n_cats. With hundreds/thousands of shuffles this is large repeated mask construction (also valid_cat_idx at 2687).

**Fix:** Precompute per-category boolean membership masks (and the list of categories passing count>=n_min_category & member.any()) once in _build_vocal_side_precompute under anchor_categorical[cat_feat]; in the k-loop keep only the data-dependent reduction. Equivalently use a one-hot grouped reduction over all categories at once.

### [MEDIUM] tests — _build_vocal_side_precompute None-return paths untested
`compute_neuronal_tuning_curves.py:1788-1791`

Lines 1788-1791 return None when the most-vocal side has fewer USVs than n_usv_min_self; lines 1780-1781 return None when sides_to_run is empty. The fixture always exceeds n_usv_min_self (5), so neither None branch is exercised.

**Fix:** Add tests: (a) a bundle with fewer USVs than n_usv_min_self asserting None; (b) a bundle whose emitters never match male/female (sides_to_run empty) asserting None.

### [MEDIUM] tests — _load_vocal_inputs early-None paths (missing column, all-noise) untested
`compute_neuronal_tuning_curves.py:1367-1371`

_load_vocal_inputs returns None when the CSV lacks a 'vae_supercategory' column (1367-1368) and when every row has vae_supercategory==0 so the filtered table is empty (1369-1371). The existing test_load_vocal_inputs_returns_none_when_no_inputs uses an empty tmp_path (no CSV at all) and reaches neither branch.

**Fix:** Two tests: (a) CSV present but without 'vae_supercategory' column (plus the tracking H5) -> assert None; (b) CSV with all vae_supercategory==0 -> assert None.

### [MEDIUM] tests — Per-cluster vocal failure try/except is untested
`compute_neuronal_tuning_curves.py:1496-1507`

The vocal per-cluster compute is wrapped in try/except (1496-1506): a raising cluster is logged ('vocal: cluster ... failed') and skipped via continue, so one bad cluster does not abort the session. This resilience path has no test.

**Fix:** Monkeypatch _compute_one_cluster_vocal to raise on the first of two clusters and succeed on the second; run calculate_neuronal_tuning_curves and assert the failure is logged-and-skipped while the second cluster's pkl is still written.

### [MEDIUM] tests — FileNotFoundError hard-error path (no spike .npy) untested
`compute_neuronal_tuning_curves.py:1452-1457`

calculate_neuronal_tuning_curves raises FileNotFoundError when no spike .npy files exist under <root>/ephys/**/cluster_data/ (lines 1452-1457) - the docstring's one hard error. No test asserts this raise.

**Fix:** Add a test that builds the ephys dir tree with no .npy under cluster_data and asserts pytest.raises(FileNotFoundError, match='No spike .npy files').

### [MEDIUM] tests — _attach_vocal_triage_stats usv_category_peth best-cat selection untested
`compute_neuronal_tuning_curves.py:2236-2295`

The usv_category_peth block (2236-2295) has nontrivial cross-category argmax (line 2273 `np.isfinite(pz) and pz > best_abs_z`) and an all-NaN fallback (best_cat stays -1, best_abs_z -> NaN at 2282-2284). The smoke test only asserts key presence in triage_stats; it never checks best_cat correctness, per_category contents, or the all-NaN fallback.

**Fix:** Add a focused test with a synthetic usv_category_peth payload where one category has a clear excitation peak and another is flat; assert best_cat is the peaked category, best_abs_z finite, per_category has both ids; add an all-NaN case asserting best_cat==-1 and best_abs_z NaN.

### [MEDIUM] tests — _anchor_bin_validity_grid require_clean_prior effect not asserted
`compute_neuronal_tuning_curves.py:459-475`

test_anchor_bin_validity_grid_post_and_prior (line 3209) calls the grid with require_clean_prior=True and require_clean_post=True but asserts only grid.shape and grid.dtype (3217-3218). The prior-cleanliness per-bin invalidation (459-473) and the post clause (455-457) run but their boolean effect on specific cells is never checked.

**Fix:** Construct starts/stops where a prior other-USV stop falls inside a specific pre-anchor bin window and assert the exact True/False cells of the returned grid (and a separate require_clean_post case), not just shape/dtype.

### [LOW] correctness — Within-USV spike count uses [start, stop) while VMI uses [start, stop]
`compute_neuronal_tuning_curves.py:2639-2641`

In _compute_one_cluster_vocal both within-USV bounds use side="left" (lines 2639-2640) -> half-open [start, stop). In _compute_vmi_for_emitter the within-USV count uses side="left" for start and side="right" for stop (lines 2451-2452) -> closed [start, stop]. The two within-USV firing-rate estimates use different boundary conventions for the same USVs. Exact float-boundary hits are rare, so practical impact is negligible, but the asymmetry is a genuine inconsistency.

**Fix:** Pick one convention in both places; to match VMI, change line 2640 to side="right".

### [LOW] docs_clarity — Confusing 'the VMI block above' reference in _attach_vocal_triage_stats docstring
`compute_neuronal_tuning_curves.py:2071-2076`

The docstring says triage_stats["vmi"] is 'added by the VMI block above.' There is no VMI block above this method; triage_stats["vmi"] is populated in the caller _compute_one_cluster_vocal (lines 2734-2753). 'above' misleadingly reads as earlier-in-this-method.

**Fix:** Reword to 'added by `_compute_one_cluster_vocal` before this method is invoked.'

### [LOW] docs_clarity — Garbled comment 'Plain D NaN policy' (duplicated at 2778)
`compute_neuronal_tuning_curves.py:1622`

Line 1622 reads 'Plain D NaN policy: ...' which is not meaningful English (likely intended 'Plain 1D NaN policy'). The same garbled token recurs at line 2778 ('plain D NaN policy (see _compute_one_cluster_behavioral)'), so the cross-reference points at incoherent wording.

**Fix:** Reword both occurrences to a clear phrase (e.g. 'Smoothing NaN policy:' or 'Plain 1D NaN policy:') at lines 1622 and 2778.

### [LOW] docs_clarity — Cryptic accumulator names q3w_rates / q3p_rates lack explanatory comment
`compute_neuronal_tuning_curves.py:2582-2594`

q3w_rates (2582) and q3p_rates (2589) feed usv_category_tuning and usv_category_peth respectively, but the q3w/q3p abbreviations are undocumented, unlike the self-describing peth_rates/prop_rates. A reader cannot map the name to its output.

**Fix:** Add an inline comment at 2582/2589 stating q3w_rates -> usv_category_tuning (within-USV per-category rate) and q3p_rates -> usv_category_peth (per-category time-resolved PETH), or rename to cat_within_rates / cat_peth_rates.

### [LOW] performance — _latest_other_stop_before_anchor uses O(n_anchors x n_usvs) loop with per-iteration concatenate
`compute_neuronal_tuning_curves.py:334-347`

The function loops over every anchor (334), slices stops[:end], and when the anchor is in range builds a new array via np.concatenate to drop one element before .max() (343-345). O(n_usvs) per anchor plus an allocation per anchor. stops[:end].max() differs from the desired value only when the anchor's own stop is the max, so the concatenate is avoidable. This is session-level precompute (once per session), so impact is bounded.

**Fix:** Use np.maximum.accumulate over start-sorted stops for an O(n_usvs) running max of stops starting before each anchor, then correct only the case where the anchor's own stop equals that running max.

### [LOW] performance — require_clean_prior validity uses a nested anchor x bin Python loop
`compute_neuronal_tuning_curves.py:459-473`

In _anchor_bin_validity_grid the require_clean_prior branch runs an outer anchor loop and inner per-bin loop (471) testing (cand_stops > bin_lo_abs[k,i]).any() for each bin. Since bin_lo_abs[k,:] is monotonic, the per-bin .any() re-scans candidate stops repeatedly. Session-level precompute, so low priority, but avoidable.

**Fix:** Vectorize the inner loop: compute max_cand_stop = cand_stops.max() once per anchor, then validity[k,:] &= ~(max_cand_stop > bin_lo_abs[k,:]) over all bins. Any overlapping prior USV's stop exceeds the bin's lower edge, so only the max candidate stop is needed.

### [LOW] performance — VMI per-bout USV rates use Python list and np.where(bout_idx==b) rescan
`compute_neuronal_tuning_curves.py:2435-2455`

_compute_vmi_for_emitter loops for b in range(n_bouts) (2435), calls usv_indices = np.where(bout_idx == b)[0] (2443) rescanning the full bout_idx per bout (O(n_bouts x n_usvs)), then accumulates a Python list of per-USV scalar searchsorted rates (2451-2453). VMI runs once per cluster (observed only, no shuffle loop), so impact is moderate, but the per-bout rescan is gratuitous.

**Fix:** Compute all within-USV spike counts vectorized via searchsorted on em_starts/em_stops, divide by em_durations, mask ud>0, then aggregate to per-bout means with a single grouped reduction keyed on bout_idx (e.g. np.add.at) instead of np.where + Python list per bout.

### [LOW] tests — NeuronalTuning.__init__ unexpected-kwarg TypeError not directly tested
`compute_neuronal_tuning_curves.py:1206-1211`

NeuronalTuning.__init__ overrides kwarg validation (1206-1211) building its own message via type(self).__name__ and its own expected set {'root_directory','tuning_parameters_dict','message_output'}. The existing TypeError test (line 2614) covers FeatureZoo, not the subclass override.

**Fix:** Add a test instantiating NeuronalTuning with a bogus kwarg inside pytest.raises(TypeError, match='unexpected keyword argument'), asserting the message names the bad key and lists the three expected keys.

### [LOW] tests — usv_category_tuning n_usv_min_category gating (sparse category -> NaN) not asserted
`compute_neuronal_tuning_curves.py:2668-2691`

Per-category within-USV rate is computed only where count_per_cat >= n_min_category and occ_per_cat > 0 (the `enough` mask, line 2690); sparse categories stay NaN. The same gate guards usv_category_peth (line 2696 continue). The fixture has many USVs so the gate is always passed, and no test asserts the sparse-category NaN behavior.

**Fix:** Build a session where one vae_category has fewer than n_usv_min_category instances; assert its usv_category_tuning rate is NaN (well-populated category finite) and its usv_category_peth row is all-NaN.


## `generate_audio_files.py` (20)

### [HIGH] performance — Quadratic O(N^2) array growth via repeated np.concatenate in naturalistic playback loop
`generate_audio_files.py:314,345,370,374,385`

replay_wav_arr starts empty (line 314) and is rebuilt every loop iteration with np.concatenate (lines 345, 370, 374); each call allocates a fresh array and copies the entire accumulated buffer. Over an ~18-min file at the naturalistic rate the buffer reaches tens of millions of int16 samples recopied across thousands of appends, giving O(N^2) memory traffic that dominates the ~2-min runtime the docstring flags. Accumulating chunks in a Python list and a single np.concatenate after the loop makes it O(N).

**Fix:** Append each chunk (silence/snippet/silence) to a Python list, then replay_wav_arr = np.concatenate(chunks) once after the while loop (before the line-385 slice); initialize chunks=[] instead of the line-314 empty-array seed.

### [HIGH] performance — Quadratic O(N^2) array growth via repeated np.concatenate in create_usv_playback_wav loop
`generate_audio_files.py:450,460`

replay_wav_arr (line 450) is grown by np.concatenate((replay_wav_arr, random_wav_file_data, arr_start_with_ipi)) once per USV (line 460) for total_usv_number (e.g. 10k) iterations; each call recopies the whole buffer, giving O(N^2) total copying. This explains the docstring's '~18 minutes for 10k USVs'. Collecting per-USV chunks in a list and concatenating once is O(N).

**Fix:** Build a list seeded with arr_start_with_ipi, append random_wav_file_data and arr_start_with_ipi each iteration, then np.concatenate(chunks) once after the loop.

### [MEDIUM] correctness — WAV truncated to target length but spacing/usvids .txt metadata are not, causing audio/metadata divergence
`generate_audio_files.py:384-385`

The inner sequence loop (lines 353-374) appends USVs and IUIs without re-checking the time budget (the break at line 336 only guards the ISI at the top of the outer loop), so replay_wav_arr routinely overshoots total_acceptable_playback_time within the final sequence. Line 385 hard-slices the WAV to target_samples, but the two .txt files (replay_txt_file sample counts, usv_id_txt_file labels written at lines 349/359/360/371/372) are never truncated. The trailing .txt entries therefore describe audio samples that line 385 sliced away, so any downstream alignment walking spacing.txt cumulatively desynchronizes from the WAV past the truncation point.

**Fix:** Stop appending to the .txt files once the cumulative sample offset would exceed target_samples (mirror the slice), or post-process both .txt files after the loop to drop entries whose cumulative offset exceeds target_samples, so the metadata exactly describes the written WAV.

### [MEDIUM] correctness — Concatenating snippet data of unknown dtype with int16 zeros can silently upcast the output WAV dtype
`generate_audio_files.py:370`

random_wav_file_data is whatever dtype scipy.io.wavfile.read returns for the snippet (int16/int32/float32/uint8). It is concatenated with explicitly int16 silence (lines 342/345/370/374); np.concatenate upcasts to the common dtype, so a single int32/float32 snippet promotes replay_wav_arr, and wavfile.write (line 390) then writes that promoted dtype, changing bit depth/encoding versus the int16 silence and the int16 seed at line 314. A float32 snippet (expected in [-1,1]) mixed with int16 zeros also yields nonsensical relative amplitudes.

**Fix:** After each wavfile.read, validate random_wav_file_data.dtype == np.int16 and raise a clear error (or cast with correct scaling) before concatenation so silence and snippets share dtype and amplitude scale.

### [MEDIUM] performance — Redundant disk re-reads of the same snippet WAVs on every draw (no caching)
`generate_audio_files.py:355,356,458,459`

Both generators draw a random file from the fixed wav_files_list (lines 355/458) and call wavfile.read (lines 356/459) every iteration. Since snippets are drawn with replacement from a small fixed pool, the same files are re-read hundreds-to-thousands of times across a 10k-USV / 18-min build. Reading each unique snippet once into a dict keyed by path eliminates the repeated I/O.

**Fix:** Preload before the loop: wav_cache = {p: wavfile.read(p)[1] for p in wav_files_list}, then use random_wav_file_data = wav_cache[random_wav_file] inside the loop.

### [LOW] correctness — Empty/missing snippet directory yields an opaque IndexError from py_rng.choice
`generate_audio_files.py:313`

wav_files_list = sorted(playback_snippets_dir.glob('*.wav')) (line 313, and line 445 in the sibling method) is not checked for emptiness. A missing or empty directory makes the list empty and py_rng.choice(wav_files_list) at line 355 (and line 458) raises 'IndexError: Cannot choose from an empty sequence' after all the model/file setup, with no mention of the directory, making the misconfiguration hard to diagnose.

**Fix:** After globbing, if not wav_files_list: emit a clear message naming playback_snippets_dir via self.message_output and return (or raise FileNotFoundError) before entering the generation loop.

### [LOW] dead_code_naming — Vestigial first-iteration if/else in ISI write (both arms equivalent)
`generate_audio_files.py:341-346`

The `if total_playback_time_created == 0:` arm sets replay_wav_arr = np.zeros(isi_samples, int16) (line 342) and the else arm does np.concatenate((replay_wav_arr, np.zeros(isi_samples, int16))) (line 345); both write 'ISI \n'. Since replay_wav_arr is seeded as an empty int16 array (line 314), np.concatenate((<empty int16>, <int16 zeros>)) equals the np.zeros arm exactly, so the else branch alone produces identical behavior on every iteration including the first. The special case is vestigial.

**Fix:** Collapse lines 341-346 to a single replay_wav_arr = np.concatenate((replay_wav_arr, np.zeros(isi_samples, dtype=np.int16))) followed by one usv_id_txt_file.write('ISI \n').

### [LOW] dead_code_naming — Unused loop variable usv_num in create_usv_playback_wav
`generate_audio_files.py:457`

for usv_num in tqdm(range(total_usv_number)): binds usv_num but grep confirms it is never referenced in the loop body (it appears only on line 457). The iteration count, not the index, is what matters.

**Fix:** Rename to _: for _ in tqdm(range(total_usv_number)):.

### [LOW] docs_clarity — Stray leading backtick on docstring line for step (3)
`generate_audio_files.py:490`

Line 490 begins with a literal backtick before the whitespace: "`       (3) stationary noise reduction is applied to the signal". Steps (1), (2), (4) have no such character; this is a copy/paste artifact corrupting the numbered list in the frequency_shift_audio_segment docstring.

**Fix:** Remove the leading backtick so the line reads '        (3) stationary noise reduction is applied to the signal', matching the other steps' indentation.

### [LOW] docs_clarity — Non-obvious kHz unit convention for wav_sampling_rate has no explanatory comment
`generate_audio_files.py:339`

Sample counts are computed as seconds * wav_sampling_rate * 1e3 (lines 339, 357, 367, 384) and the output rate as int(wav_sampling_rate * 1e3) (lines 391, 468), which only makes sense because the sampling rate is stored in kHz (250). With no comment, * 1e3 on a sampling rate reads like a possible bug.

**Fix:** Add a brief comment at the first use (near line 339) noting that wav_sampling_rate is in kHz, so * 1e3 converts to samples-per-second (Hz).

### [LOW] docs_clarity — playback_seed comment describes only py_rng, omitting the numpy rng it also seeds
`generate_audio_files.py:301-304`

The comment at lines 301-304 ends 'A local random.Random is used instead of the global `random` module so the draw is reproducible without mutating global state.' But playback_seed seeds two generators: the numpy rng (line 306, driving ISI/IUI interval draws via _draw_bounded_seconds at lines 332/366 and sequence length via rng.normal at line 352) and py_rng (line 307, only USV-file selection). The comment omits the numpy generator, which produces the substantive interval/length structure.

**Fix:** Reword to cover both generators, e.g. note that a numpy Generator (interval and sequence-length draws) and a local random.Random (USV file selection) are both seeded from playback_seed, the latter avoiding mutation of global `random` state.

### [LOW] docs_clarity — Empty Parameters section in create_naturalistic_usv_playback_wav docstring
`generate_audio_files.py:254-256`

Lines 254-256 have a 'Parameters\n----------' header followed immediately by a blank line and the Returns section, leaving Parameters empty. The sibling create_usv_playback_wav (lines 407-410) documents 'None / Inputs are read from self.create_playback_settings_dict ...'. The empty section is inconsistent with the repo's Description/Parameters/Returns convention.

**Fix:** Mirror create_usv_playback_wav: under Parameters add a 'None' entry noting inputs come from self.create_playback_settings_dict (naturalistic block) and self.exp_id.

### [LOW] tests — _split_iui_isi K<2 ValueError path is untested
`generate_audio_files.py:60-65`

_split_iui_isi raises ValueError with a K-specific message when n_components < 2 (lines 60-65). test_audio_gen.py only injects a 3-component model, so this guard branch and message have zero coverage.

**Fix:** Add a unit test calling _split_iui_isi with a 1-component TMixture and pytest.raises(ValueError, match=r'need >= 2 Student-t'), asserting the message includes 'K=1'.

### [LOW] tests — _split_iui_isi split/renormalization logic never asserted directly
`generate_audio_files.py:67-85`

_split_iui_isi pools all-but-slowest into the IUI model with renormalized weights (line 72) and the slowest (index K-1) into a single-component ISI model (lines 79-84). Tests exercise this only indirectly and never assert renormalization (weights sum to 1), that ISI is the slowest component, or that IUI has K-1 components. A regression dropping the wrong component or skipping renormalization would pass.

**Fix:** Add a unit test on a known 3-component ascending TMixture asserting np.isclose(iui_model.weights_.sum(), 1.0), iui_model.n_components == 2, and isi_model.means_.ravel()[0] equals the original largest mean.

### [LOW] tests — AudioGenerator.__init__ unexpected-kwarg TypeError is untested
`generate_audio_files.py:199-202`

The constructor raises TypeError naming the offending key(s) when given a kwarg outside expected_kwargs (lines 198-202). test_audiogen_init_stores_arbitrary_kwargs (test line 38) only checks the happy path; the rejection branch is uncovered, unlike the equivalent guards tested in sibling modules.

**Fix:** Add a test asserting pytest.raises(TypeError, match=r'unexpected keyword argument') for AudioGenerator(exp_id='x', bogus_kwarg=1, message_output=lambda *a, **k: None).

### [LOW] tests — Cluster-path (non-mounted) branch never exercised
`generate_audio_files.py:272-274`

Both methods branch on os.path.ismount between a local mount path and find_cluster_path() (lines 269-274 and 423-428). Every test patches os.path.ismount to return True (test lines 291, 374), so the find_cluster_path() branches are never run; a regression in cluster-path construction would go undetected.

**Fix:** Add a variant patching os.path.ismount to False and find_cluster_path to return str(tmp_path), asserting outputs land under the cluster-rooted directory.

### [LOW] tests — Female sex-inference and per-sex clip_pct selection untested
`generate_audio_files.py:286`

Sex is inferred from the prefix (line 286) and clip_pct read per-sex from naturalistic_interval_clip_pct[sex] (line 288). _run_naturalistic_playback defaults prefix='male' and all tests pass 'male', so the female branch, female clip_pct lookup, and female snippet-dir resolution are never exercised.

**Fix:** Add a test calling the naturalistic helper with prefix='female' asserting it reads naturalistic_interval_clip_pct['female'], resolves the female snippet dir, and writes a WAV.

### [LOW] tests — Output truncation to target_samples is never asserted
`generate_audio_files.py:384-385`

After the loop, replay_wav_arr is sliced to target_samples (lines 384-385) before being written. Tests only assert write_mock.call_count == 1 (test line 456) and never inspect the data= argument length. A regression removing the truncation (file overshoots requested duration) would pass.

**Fix:** Capture write_mock.call_args.kwargs['data'] and assert its shape[0] <= int(total_time * 250 * 1e3).

### [LOW] tests — _mixture_log_bounds determinism and ordering untested
`generate_audio_files.py:118-122`

_mixture_log_bounds uses a fixed-seed RNG (line 118) so bounds are deterministic and returns (lo_log, hi_log) with lo < hi (lines 120-121). No test asserts two calls return identical bounds or that lo < hi, leaving the docstring's reproducibility invariant unverified.

**Fix:** Add a test calling _mixture_log_bounds twice on the same TMixture asserting equal tuples and lo < hi.

### [LOW] tests — _draw_bounded_seconds bound enforcement untested
`generate_audio_files.py:157-160`

_draw_bounded_seconds reject-resamples in log space until lo_log <= draw <= hi_log then returns np.exp(draw) (lines 157-160). No test verifies returned values lie within [exp(lo_log), exp(hi_log)]; this clipping is the function's entire purpose for heavy-tailed low-nu components.

**Fix:** Add a test calling _draw_bounded_seconds many times with tight bounds on a known model, asserting every return is within [exp(lo_log), exp(hi_log)].


## `unit_triage_aggregator.py` (18)

### [HIGH] correctness — Duplicate session basename in a condition list double-counts per_session evidence
`unit_triage_aggregator.py:1112-1117, 1199-1211`

Section 3 builds condition_sessions[cond] (lines 1112-1117) with no de-duplication: every non-blank line becomes pathlib.Path(stripped).name and is appended. If a session basename repeats in a condition .txt (literal repeat, or two distinct full paths whose basenames collide), section 4 processes that session's pkls twice. sessions_tested is guarded against repeats (line 1199 `if sess not in cond_block["sessions_tested"]`), but the parallel mod_block["per_session"].append(entry) at line 1211 has NO matching guard. So n_tested = len(ps) (line 1232) is inflated, n_significant double-counted, consistency skewed, and _aggregate_modality_stats medians biased. Output is internally inconsistent (sessions_tested deduped, per_session not). Verified: no dedup of per_session exists anywhere in the post-processing loops (1228-1255).

**Fix:** De-duplicate basenames when building condition_sessions (preserve order, drop already-seen names) — this also avoids redundant pkl I/O — or guard the per_session append the same way sessions_tested is guarded.

### [MEDIUM] docs_clarity — _to_jsonable docstring describes json.dump serialization that never happens in this module
`unit_triage_aggregator.py:43-75`

The docstring (lines 48, 56) says it coerces types so json.dump survives and that unhandled types hit json.dump's default error. The module never calls json.dump/json.dumps (only json.load at 777); output is pickled. The helper is unreferenced in src/ entirely. The docstring frames the function around a non-existent code path, misleading a reader about why it exists.

**Fix:** Remove _to_jsonable if dead, or reword the docstring to drop the json.dump framing and state its actual role (a numpy/non-finite -> native coercion helper retained for callers/tests).

### [MEDIUM] docs_clarity — Comment claims consumers filter admin keys via _FLAG_ADMIN_KEYS, but nothing does
`unit_triage_aggregator.py:481-487`

The comment (lines 482-483) states 'Consumers that only want the metric values filter these out via _FLAG_ADMIN_KEYS.' grep shows no consumer uses it. The one site that strips an admin key (line 1208) filters only the literal "tested". The comment documents a mechanism no code uses, misleading a maintainer into thinking the constant is load-bearing.

**Fix:** Either wire _FLAG_ADMIN_KEYS into the filtering site (replace the hardcoded `if k == "tested"` skip with membership in _FLAG_ADMIN_KEYS) or reword the comment to say it is merely provided for downstream consumers, not already used.

### [LOW] correctness — VMI of exactly 0.0 is classified as 'suppress' rather than no-modulation
`unit_triage_aggregator.py:303, 580`

_flag_vmi line 303 (`direction = "excit" if vmi > 0 else "suppress"`) and flag_one_cluster line 580 (`inferred_dir = "excit" if vmi > 0 else "suppress"`) both route a finite vmi == 0.0 into the suppress branch. At line 578 a finite 0.0 passes the `not math.isfinite` guard, so flag_one_cluster always emits key vmi_<role>_suppress for a zero VMI regardless of significance. In _flag_vmi it only matters if vmi==0 also clears the p gate (unlikely for a true zero effect). Marginal but a genuine semantic edge.

**Fix:** Decide vmi==0 handling explicitly: skip it (return None,None in _flag_vmi; continue in flag_one_cluster) or document that 0 maps to suppress by convention.

### [LOW] correctness — int(payload["n_bouts"]) not guarded against NaN, unlike the _safe_float-coerced fields
`unit_triage_aggregator.py:139`

In _extract_vmi_metrics, n_bouts is coerced with a bare int(payload["n_bouts"]) (line 139) while every float field uses _safe_float. If a pkl stores n_bouts as float NaN, int(float('nan')) raises ValueError, aborting the whole aggregation on one bad block rather than skipping it. n_valid_pairs (line 140) is at least None-guarded.

**Fix:** Coerce defensively, e.g. _nb = _safe_float(payload["n_bouts"]); "n_bouts": int(_nb) if _nb is not None else 0.

### [LOW] correctness — Cross-session aggregates (max/median) span non-significant sessions, undocumented
`unit_triage_aggregator.py:858-908`

_aggregate_modality_stats._vals (lines 858-871) pulls metric values across the entire per_session list with no filter on the per-entry significant flag. So max_abs_peak_z, median_peak_z, max_abs_vmi, min_pvalue, max_info_rate_bps etc. summarize ALL tested sessions, not only significant ones. May be intentional (full-distribution reporting), but the docstring (lines 826-855) does not state it, which is easy to misread given the adjacent n_significant/consistency machinery.

**Fix:** If full-distribution is intended, add one line to the docstring stating aggregates span all tested sessions regardless of significance; otherwise filter per_session by e["significant"].

### [LOW] dead_code_naming — _to_jsonable is dead production code (module pickles output, never JSON-dumps)
`unit_triage_aggregator.py:43-75`

The module's only output is pickle.dump; the only json.* call is json.load at line 777 (reading analyses_settings.json). grep confirms no json.dump/json.dumps anywhere in the module, and the only importers of _to_jsonable are tests (tests/analyses/test_analyze.py). No __all__/__init__ re-export; only aggregate_units_across_conditions is imported externally. The numpy/non-finite coercion _to_jsonable performs is never exercised in production — vestigial from an earlier JSON-output design replaced by pickle.

**Fix:** Remove _to_jsonable (and its test) if the JSON-output path is gone, or wire a JSON sidecar into the write step if intended.

### [LOW] dead_code_naming — _FLAG_ADMIN_KEYS frozenset is defined but read by nobody
`unit_triage_aggregator.py:481-487`

_FLAG_ADMIN_KEYS (lines 484-487) is referenced only by its own defining lines and the comment above it. grep across src/ and tests/ finds zero readers. The downstream filtering its comment claims is instead done inline at lines 1207-1210, which skips only the literal key "tested", not via this set. Dead constant; also note the set omits "direction" but that is irrelevant since nothing consumes it.

**Fix:** Delete the constant and its comment (481-487), or actually use it at the per_session entry-build (1207-1210) by skipping keys in _FLAG_ADMIN_KEYS.

### [LOW] docs_clarity — Prose docstring says enriches with brain_area, but output field is anatomy_region
`unit_triage_aggregator.py:930-932`

The function prose (lines 931-932) says each cluster is enriched with mouse_id, rec_date, and brain_area (the catalog column is indeed brain_area, line 1082). But the emitted record (line 1193) and the Output schema (line 963) name the field anatomy_region. A reader matching prose to schema sees two names for the same datum with no note of the brain_area->anatomy_region rename.

**Fix:** Note the rename in the prose, e.g. '...and brain_area (surfaced in the output as anatomy_region)'.

### [LOW] docs_clarity — Output-schema consistency field documented without defining what it measures
`unit_triage_aggregator.py:970`

The Output schema lists "consistency": float (line 970) but never defines it. Its meaning (n_significant / n_tested, 0.0 when n_tested == 0) is only discoverable from lines 1236-1238. Given the maintainer's preference for detailed docstrings, the headline output field deserves a one-line gloss.

**Fix:** Add a short gloss, e.g. 'consistency = n_significant / n_tested (fraction of tested sessions flagged significant; 0.0 when n_tested == 0)'.

### [LOW] performance — VMI/categorical/spatial metrics extracted twice per emitter in flag_one_cluster
`unit_triage_aggregator.py:576-589`

Line 576 calls _extract_vmi_metrics(payload), then line 581 calls _flag_vmi which re-extracts at line 289. Same pattern: categorical (651 + _flag_categorical 368) and spatial (732 + _flag_spatial 396). Each extraction reruns _safe_float coercions; scaled by clusters x emitters x features across many pkls this is redundant, though each call is cheap.

**Fix:** Derive significant inline from the already-extracted metrics dict (re-implementing the simple gate) rather than re-invoking _flag_*. NOTE: this duplicates the gate logic into flag_one_cluster, risking future divergence from the standalone _flag_* helpers; only worth doing if the redundant extraction is measured to matter.

### [LOW] performance — catalog.groupby('rec_date')['mouse_id'] computed twice
`unit_triage_aggregator.py:1088-1102`

Line 1088 builds date_mouse_counts = catalog.groupby('rec_date')['mouse_id'].nunique(); line 1101 builds date_to_mouse = catalog.groupby('rec_date')['mouse_id'].first().to_dict(). Same column grouped on the same key twice.

**Fix:** grp = catalog.groupby('rec_date')['mouse_id'], then date_mouse_counts = grp.nunique() and date_to_mouse = grp.first().to_dict().

### [LOW] performance — units dict traversed three times in separate nested loops
`unit_triage_aggregator.py:1228-1255`

Post-processing walks units.values() three times: 1228-1238 (n_tested/n_significant/consistency), 1240-1243 (sort sessions_tested), 1247-1255 (sort per_session + aggregate). The first two iterate the same conditions structure with no inter-dependency; only the per_session sort (1250) must precede the aggregate call (1253), and those already share loop 3. All three could collapse into one traversal.

**Fix:** Fuse into a single for unit / for cond_block / for mod_block block: sort sessions_tested, then per mod_block sort per_session, compute n_tested/n_significant/consistency, set aggregate. Keep the per_session sort before the aggregate to preserve determinism.

### [LOW] tests — _aggregate_modality_stats _vals partial non-finite/non-numeric filtering is untested
`unit_triage_aggregator.py:858-871`

_vals skips None (862-863), non-coercible values (866-867), and non-finite floats (869). test_aggregate_modality_stats_dispatch_by_prefix (test file lines 520-567) feeds only clean finite values; test_aggregate_modality_stats_empty_per_session_yields_none (1080-1095) feeds empty lists. No test mixes finite values with NaN/None/strings to verify the partial-drop path (e.g. a NaN peak_z dropped from the median while a finite one survives).

**Fix:** Add a case like _aggregate_modality_stats('usv_peth_self_excit', [{'peak_z': 4.0}, {'peak_z': float('nan')}, {'peak_z': None}, {'peak_z': 'x'}]) asserting max_abs_peak_z == 4.0 and median_peak_z == 4.0.

### [LOW] tests — _parse_unit_id malformed numeric-prefix int() path is untested
`unit_triage_aggregator.py:819-821`

test_parse_unit_id_raises_on_malformed_input (test file lines 505-517) only covers the token-count ValueError (source lines 814-817). The int() conversions at source lines 819-821 (removeprefix then int) raise ValueError on a non-numeric body (e.g. 'imecX_cl0007_ch207_good' or 'imec0_clYY_ch207_good'); that distinct raise path is not asserted.

**Fix:** Add to the malformed test: pytest.raises(ValueError) for _parse_unit_id('imecX_cl0007_ch207_good') and _parse_unit_id('imec0_clYY_ch207_good').

### [LOW] tests — Aggregator path where an on-disk pkl has no triage_stats is untested end-to-end
`unit_triage_aggregator.py:1172-1175`

Source lines 1172-1175 set records={} for a pkl whose triage_stats is missing/non-dict while still creating the unit slot. The test helper _write_cluster_pkl supports triage_stats=None (test lines 573-611), but no aggregator test ever calls it with None (all six call sites pass a real triage dict), so the documented invariant 'unit recorded with zero modalities' is unverified end-to-end.

**Fix:** Add an aggregator test writing a catalog-listed pkl with triage_stats=None and assert the unit_uid appears in out['units'] with conditions[cond]['modalities'] == {} while sessions_tested still lists the session.

### [LOW] tests — _extract_categorical_metrics / _extract_spatial_metrics None->-1 sentinel coercions untested
`unit_triage_aggregator.py:186-214, 217-246`

_extract_categorical_metrics maps best_cat=None -> -1 (line 211); _extract_spatial_metrics maps peak_row/peak_col=None -> -1 (lines 244-245). The payload helpers _categorical_payload (best_cat=2) and _spatial_payload (peak_row=10, peak_col=15) always set integers, so the None branches are never hit. flag_one_cluster's own best_cat None->-1 (line 671) is likewise untested.

**Fix:** Add direct extractor tests with best_cat=None / peak_row=None / peak_col=None asserting the -1 sentinel, or extend the payload helpers to allow None.

### [LOW] tests — flag_one_cluster omitting a VMI block on non-finite vmi is untested
`unit_triage_aggregator.py:578-579`

When metrics['vmi'] is None or not math.isfinite (line 578) the VMI block is skipped (continue, 579), emitting no key — unlike every other modality. No test feeds NaN/inf vmi to confirm the key is absent. Confirmed: searching the test suite, no flag_one_cluster test passes a non-finite vmi; the vmi tests use finite values (0.7, -0.4).

**Fix:** Add a test with triage_stats={'vmi': {'self': payload with vmi=float('nan'), p=1e-4, n_bouts=50}} asserting 'vmi_self_excit'/'vmi_self_suppress' absent from records.


## `usv_interval_archive.py` (18)

### [MEDIUM] correctness — _decode_attr silently coerces numeric-looking string attributes to numbers (git_sha corruption)
`usv_interval_archive.py:401-408`

Lines 401-408 blanket-attempt json.loads on every string attribute. A git short SHA written at compute_inter_usv_interval_distributions.py:858 can be all digits (e.g. "1234567"), and would be read back as the int 1234567 instead of the original string. The writer _attr_value (lines 301-307) only JSON-encodes list/tuple/dict and stores plain strings verbatim, so there is no marker to distinguish an intentionally JSON-encoded structured value from a coincidentally-parseable scalar string. Verified: writer at line 302 only json.dumps containers; reader at 405-406 json.loads any string. Any all-digit / 'true' / numeric-looking string attr is silently retyped on read.

**Fix:** Restrict decoding to strings that begin with '[' or '{' (the only containers the writer ever JSON-encodes), leaving scalar/SHA strings untouched; or tag structured attrs at write time and only decode tagged ones.

### [MEDIUM] performance — Structured-array fill round-trips numeric columns through Python lists (to_list) instead of to_numpy
`usv_interval_archive.py:111-113`

Lines 111-113 materialize every column via df[col].to_list() then assign into the numpy structured field, boxing/unboxing each element. The intervals table is one-row-per-interval across many sessions (hundreds of thousands of rows). Benchmarked n=500000: to_list+assign ~28.9 ms vs to_numpy+assign ~0.4 ms. Only variable-length-string columns need the list path.

**Fix:** Carry a string-flag on each np_field; for non-string fields do arr[col] = df[col].to_numpy(); keep to_list() only for the str_dtype columns. Note: nullable-int columns (finding at 110-113) must still be handled (to_numpy on a null int column yields object/NaN considerations) -- combine the two fixes.

### [MEDIUM] performance — Read path materializes every numeric column via tolist() before rebuilding the polars frame
`usv_interval_archive.py:157-169`

The else branch at 166-167 calls arr[col].tolist() for every non-object column, boxing into a Python list that polars re-parses. Benchmarked n=500000: tolist+build ~14.8 ms vs np.ascontiguousarray+build ~4.7 ms. String columns (kind 'O', 161-165) must keep the decode loop; numeric/bool fields can be passed as numpy arrays. Structured-field views are non-contiguous so wrap in np.ascontiguousarray.

**Fix:** In the else branch replace cols[col] = col_data.tolist() with cols[col] = np.ascontiguousarray(col_data); leave the object/string branch unchanged.

### [MEDIUM] tests — write_ivi_h5 multi-mode write and populated optional tables are untested
`usv_interval_archive.py:258-274`

_example_per_mode_payload (test_analyze.py:2084) builds only one mode ('s2s') with gmm_fits/bootstrap_lrt/bootstrap_lrt_null all None. The per-mode loop (258-274) is never tested with two modes, and the non-None write branch (271-274) for the optional tables is never written or read back. read_usv_interval_h5 mode iteration (357) is exercised for a single group only.

**Fix:** Extend the payload to include both 's2s' and 'e2s' with non-None gmm_fits/bootstrap_lrt/bootstrap_lrt_null, then assert read_usv_interval_h5 returns both modes with those tables reconstructed (height/columns) rather than None.

### [MEDIUM] tests — reconstruct_best_model best-rep selection among multiple reps is not actually verified
`usv_interval_archive.py:493`

The sort/head(1) lowest-IC selection at line 493 is never positively asserted. test_reconstruct_best_model_falls_back_when_cv_all_nan (2200) builds two reps but its own comment (2208-2209) admits it only checks model is not None, not that the lower-IC rep's parameters win. No test gives reps with distinct means and confirms the low-IC rep is the one reconstructed.

**Fix:** Build a 2-rep DF where the lower-IC rep has distinct logmean values vs the higher-IC rep, then assert model.means_ matches the low-IC rep's means.

### [LOW] correctness — _polars_to_h5 raises TypeError if an integer/bool column contains nulls
`usv_interval_archive.py:110-113`

For Int*/UInt* columns the structured field dtype is np.int64/np.uint64 (lines 98-101); assigning a Python list containing None (a polars null) at line 113 raises TypeError. Verified empirically: int+null -> 'TypeError: int() argument ... not NoneType'; float+null survives (None -> NaN). Current callers only put non-null ints in n_dropped_* / n_comp / rep, so this is latent, but this is a generic serializer and any nullable-int table would crash at write time.

**Fix:** For integer/boolean columns with nulls (df[col].null_count() > 0), cast to float64 to allow NaN encoding, or store a separate null mask, instead of assigning None into an integer numpy field.

### [LOW] correctness — ic_col NaN fallback only triggers for default 'cv_neg_loglik'; a caller-supplied all-NaN ic_col picks an arbitrary rep
`usv_interval_archive.py:486-493`

The NaN->bic fallback is guarded by `if chosen_ic == "cv_neg_loglik"` (line 488). A non-default ic_col that is entirely NaN, or a case where both cv and bic are NaN, falls through to sort(..., nulls_last=True).head(1) at line 493, which returns the first original-order row rather than a best rep, with no warning. The summary-stats wrapper (usv_interval_summary_statistics.py:1882) does forward a caller-supplied ic_col, so the non-default path is reachable in principle, though all current callers default it.

**Fix:** After computing chosen_ic, re-check the chosen column is not all-NaN regardless of which ic_col was requested; if so, fall back (bic, then aic) and/or emit a warning.

### [LOW] dead_code_naming — Writer name write_ivi_h5 inconsistent with reader read_usv_interval_h5 and module naming
`usv_interval_archive.py:207`

The public writer is write_ivi_h5 (line 207) while the paired reader is read_usv_interval_h5 (line 313), the module is usv_interval_archive.py, and the docstring filename pattern is usv_interval_analysis_<...>.h5. The abbreviation 'ivi' is undocumented and appears nowhere else meaningfully. The mismatched read_usv_interval_h5 / write_ivi_h5 pair is confusing. Call sites verified: compute_inter_usv_interval_distributions.py:48,881; usv_interval_summary_statistics.py:42,1564; plus tests at test_analyze.py, test_visualize_stats.py, test_iui_calculator.py.

**Fix:** Rename write_ivi_h5 -> write_usv_interval_h5 to mirror the reader, updating the two src call sites, the reader docstring reference (line 317), and the importing/patching tests. If kept for back-compat, document 'ivi' = inter-USV interval in the docstring.

### [LOW] docs_clarity — Module docstring says root attr `created_at`, writer docstring + real writer use `created_at_iso`
`usv_interval_archive.py:11-13`

Module docstring line 12 lists `created_at`, but the writer Parameters (line 234) and the actual writer (compute_inter_usv_interval_distributions.py:857) use `created_at_iso`. The module docstring key is stale and misleads anyone reading the attribute back by name.

**Fix:** Change line 12 of the module docstring to `created_at_iso` to match the writer and the real key.

### [LOW] docs_clarity — gmm_order docstring asserts ascending-log-mean ordering the function never enforces
`usv_interval_archive.py:442-449`

Lines 446-449 state components are 'returned in ascending log-mean order' and the model is 'pre-sorted'. The code (lines 496-525) copies weight_k/logmean_k/logsd_k/nu_k in column order k=1..K and returns np.arange(K) unconditionally with no sort or check; the property holds only if the writer stored components sorted. The docstring states an unverified upstream precondition as a guarantee the function establishes.

**Fix:** Reword to make the precondition explicit: 'Assumes the archived per-component columns were already stored in ascending log-mean order by the compute path; this function preserves that order and returns gmm_order = arange(K).'

### [LOW] docs_clarity — ic_col docstring describes NaN->bic fallback as general, but it is special-cased to the default
`usv_interval_archive.py:464-466`

Lines 463-466 say ic_col 'falls back to bic when CV values are NaN', but the code (line 488) only applies the fallback when chosen_ic == 'cv_neg_loglik'. A non-default ic_col gets no fallback. The docstring implies a general property.

**Fix:** Clarify: 'Defaults to cv_neg_loglik; only when left at that default and all CV values in the (sex,K) slice are NaN does selection fall back to bic. A non-default ic_col is used as-is with no fallback.'

### [LOW] tests — Fallback stringify branch in _polars_to_h5 is untested
`usv_interval_archive.py:106-108`

test_polars_h5_roundtrip_mixed_dtypes (test_analyze.py:1985) covers Int64/UInt32/Float64/Utf8/Boolean only. The else branch at lines 106-108 (unmapped dtype -> str_dtype) is never hit; no test confirms a Datetime/Categorical/List column stringifies without raising at line 112-113.

**Fix:** Add a round-trip test with an unmapped-dtype column (e.g. pls.Categorical or pls.Datetime), asserting it serializes and reads back as a string column.

### [LOW] tests — Float32 dtype mapping path in _polars_to_h5 is untested
`usv_interval_archive.py:102`

Line 102 maps Float32 (and Float64) to np.float64; the schema_json/dtype_lookup restoration (line 148 'Float32': pls.Float32) brings it back as Float32. The round-trip test uses only Float64. Verified empirically that a Float32 column round-trips with schema preserved, but nothing asserts it.

**Fix:** Add a pls.Series(dtype=pls.Float32) column to the round-trip test and assert rt.schema for it is pls.Float32.

### [LOW] tests — Narrow signed-int (Int8/Int16/Int32) schema restoration round-trip not asserted
`usv_interval_archive.py:98-99`

Lines 98-99 map narrow signed ints to np.int64; restoration to the original narrow dtype goes through dtype_lookup (line 146). Only Int64 is tested. Verified empirically that Int16 round-trips back to pls.Int16, but no test asserts it.

**Fix:** Add a pls.Series(dtype=pls.Int16) column to the round-trip test and assert rt.schema for it is pls.Int16.

### [LOW] tests — _attr_value arbitrary-object fallback (json.dumps default=str) is untested
`usv_interval_archive.py:307`

test_attr_value_* (1949, 1957) cover None/bool/int/float/str/list/dict/tuple. The final fallback at line 307 (json.dumps(v, default=str) for other objects, e.g. pathlib.Path) is never exercised, despite source_lists provenance being path-like (compute path stringifies them at line 859, but a caller could pass Path objects).

**Fix:** Add a test that _attr_value(Path('/x/y')) returns a JSON string and round-trips through _decode_attr.

### [LOW] tests — reconstruct_best_model explicit non-default ic_col path is untested
`usv_interval_archive.py:486-491`

All reconstruct_best_model tests (2164-2210) use default ic_col='cv_neg_loglik'. The branch where a caller passes ic_col='bic' (skipping the CV-NaN fallback entirely, chosen_ic stays 'bic') is never exercised, even though the summary-stats wrapper forwards ic_col.

**Fix:** Add a test calling reconstruct_best_model(df, sex='male', K=2, ic_col='bic') with reps differing in bic, asserting the lowest-bic rep is selected and cv values do not influence selection.

### [LOW] tests — reconstruct_best_model t-mixture nu values not asserted
`usv_interval_archive.py:519-525`

test_reconstruct_best_model_t_mixture (2177) only asserts isinstance TMixture and order==arange(2). The nu_k extraction (519-521) and propagation into TMixture (522-524) are unverified; a wrong nu column or mis-shape would pass. TMixture stores nus_ (mixture_model_utils.py:1166), so it is assertable.

**Fix:** Assert the reconstructed TMixture's nus_ equals the row's nu_k (5.0 each), and means_/covariances_ match logmean_k / logsd_k**2.

### [LOW] tests — _decode_attr bytes-wrapped JSON payload not tested
`usv_interval_archive.py:399-408`

test_decode_attr_recovers_python_types (1967) covers b'null'->None and str-JSON decoding, but never a bytes value carrying a JSON container (e.g. b'[1,2,3]'). h5py can return attribute strings as bytes, so the bytes->str->json.loads path (399 then 405-406) is only partially covered.

**Fix:** Add: _decode_attr(b'[1, 2, 3]') == [1, 2, 3] and _decode_attr(b'not json') == 'not json'.


## `mixture_model_utils.py` (17)

### [MEDIUM] correctness — rng.choice(p=weights) can raise on float drift in mixture weights
`mixture_model_utils.py:1784`

_sample_from_mixture passes weights = np.asarray(model.weights_).flatten() directly to rng.choice(K, size=N, p=weights). numpy validates abs(sum(p)-1) <= sqrt(eps) (~1.49e-8) and raises ValueError otherwise. For TMixture, weights come from w_new = n_k / N (line 1416) and are never renormalized after the final M-step before being stored in best_model (line 1427); their sum equals sum(z)/N which is 1 only up to float accumulation over N. This sampler drives every bootstrap replicate in bootstrap_lrt (line 1958), so one drifting replicate aborts the entire LRT. Renormalizing before the draw makes it robust.

**Fix:** Before sampling: weights = np.asarray(model.weights_).flatten(); weights = weights / weights.sum(), then pass to rng.choice.

### [MEDIUM] tests — Zero-mode path is untested (gmm_modes empty return -> report_gmm_stats -> summarize_best_gmm)
`mixture_model_utils.py:690-691, 747-748, 804`

Confirmed the three coupled empty-mode branches exist: gmm_modes returns (unique, np.empty((0,))) at 690-691; report_gmm_stats early-returns means, sds, np.empty((0,)), np.empty((0,)) at 747-748 guarded by modes_arr.shape[0] == 0; summarize_best_gmm has np.exp(modes_log) if modes_log.size else modes_log at 804. The test files only fit well-separated mixtures yielding >=1 mode (test_mixture_model_utils.py 118-168, test_analyze.py 1455-1472); no test drives the zero-mode return or the downstream empty-array plumbing. A mis-shaped empty array would go uncaught.

**Fix:** Add a test that monkeypatches gmm.score_samples to return a monotone array over the grid (no strict interior peak), then assert gmm_modes returns (np.empty((0,1)), np.empty((0,))), report_gmm_stats returns 2-shaped means/sds plus two empty arrays, and summarize_best_gmm returns modes_sec as an empty array (false branch of line 804).

### [LOW] correctness — best_ll is one M-step stale relative to the parameters stored in best_model
`mixture_model_utils.py:1397-1427`

In fit_log_t_mixture, ll (line 1403) is computed from the parameters at the top of the iteration (the previous M-step output). The M-step then overwrites mu/sigma2/nu/w at line 1419, and best_model is built from those updated parameters at line 1427 while best_ll is set to the pre-update ll (line 1426). So best_ll does not correspond to best_model's actual parameters; it lags one M-step. This is the score used to pick the winning restart across n_init, so on early break (1421-1422) or close restarts it can mis-rank and report a likelihood the returned model does not attain. Confirmed by reading the loop.

**Fix:** After the EM loop (or after the final M-step) recompute the log-likelihood of the final (mu, sigma2, nu, w) and use that value both for the best_ll comparison and for the stored best_ll, e.g. compute log_norm from the final params and final_ll = float(np.sum(log_norm)).

### [LOW] dead_code_naming — Module docstring omits the bootstrap-LRT family and summarize_best_t_mixture
`mixture_model_utils.py:6-21`

Confirmed: the module docstring (lines 6-15) lists summarize_best_gmm (line 10) and report_gmm_stats but omits summarize_best_t_mixture from the Student-t bullet (12-15), and never mentions the LRT block actually present in the file: bootstrap_lrt (1848), select_n_components_step_up_lrt (1981), _sample_from_mixture (1753), _lr_statistic (1813). bootstrap_lrt and select_n_components_step_up_lrt are first-class public API imported by compute_inter_usv_interval_distributions.py (confirmed lines 36/44). The maintainer prefers fixing doc omissions. (Note: the finder's claim that summarize_best_gmm is omitted is wrong; it is present — only summarize_best_t_mixture is missing.)

**Fix:** Add a third bullet to the module docstring covering bootstrap_lrt and select_n_components_step_up_lrt, and add summarize_best_t_mixture to the Student-t surface list so the enumeration matches the actual public functions.

### [LOW] docs_clarity — Subsampling rationale comment is garbled (missing noun) — merged duplicate
`mixture_model_utils.py:1746-1749`

Confirms the comment near line 1748 has a missing noun ('this keeps the tractable') and an inconsistent N_subsample vs n_subsample capitalization; the sentence reads as if a clause was dropped during editing. This consolidates the two candidate findings that both target the same lines 1746-1749 comment. Note: line 1748 falls inside a docstring/procedure region; the exact phrase should be confirmed in context, but the broken clause is a real readability defect.

**Fix:** Rewrite to a complete thought matching the parameter name, e.g. 'For large datasets we subsample to n_subsample so the observed and bootstrap LR statistics are on the same N scale; this keeps the computation tractable while preserving validity, since the LRT is asymptotic and converges for any sufficiently large n_subsample.'

### [LOW] docs_clarity — gmm_cv_neg_loglik Returns omits the n_samples < n_folds early-return case
`mixture_model_utils.py:914-921`

Confirmed: the Returns note (lines 919-920) documents np.inf only for 'any fold has fewer training samples than n_components', but the function also returns float('inf') at line 925-926 when n_samples < n_folds (KFold cannot be constructed). That guard is undocumented.

**Fix:** Add to the Returns note: 'Also returns np.inf when there are fewer samples than n_folds (KFold cannot be constructed).'

### [LOW] docs_clarity — t_mixture_cv_neg_loglik Returns omits the N < n_folds early-return case
`mixture_model_utils.py:1513-1518`

Confirmed: Returns (lines 1516-1518) documents np.inf only for the per-fold 'fewer training samples than n_components' case, but the function also returns float('inf') at line 1523-1524 when N < n_folds. Undocumented.

**Fix:** Add to the Returns note: 'Also returns np.inf when there are fewer samples than n_folds.'

### [LOW] docs_clarity — _t_update_nu docstring does not name the 50.0 fallback value
`mixture_model_utils.py:1054-1063`

The Description (lines 1058-1063) says it 'falls back to a near-Gaussian default' and references nu 'above ~50', and Returns says '[2.001, 200]'. The actual fallback on brentq ValueError is return 50.0 (line 1110). Naming the value 50.0 explicitly in the Description ties the prose to the code. Minor doc precision improvement (maintainer prefers fixing omissions, not trimming).

**Fix:** In the Description, state the fallback explicitly: '...falls back to a near-Gaussian default of nu = 50.0 when the equation does not change sign in the bracket...'.

### [LOW] performance — brentq inner lambda re-extracts GMM params on every CDF evaluation
`mixture_model_utils.py:353-359`

Confirmed: gmm_quantile_logspace (called per-quantile, n_q default 200 from qqplot_gmm) passes lambda v: gmm_cdf_logspace(v, gmm) - qi to brentq. gmm_cdf_logspace (313-320) re-runs gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_.flatten() and reallocates a zeros buffer on every evaluation. mu/sd are already extracted at 347-348 but w is not, and the closure does not reuse them. Hoisting a closed-form CDF over pre-extracted (mu, sd, w) avoids thousands of redundant flatten/sqrt/alloc calls. Real but the workload is tiny (one figure, ~200 quantiles), so severity lowered to low.

**Fix:** Extract w once alongside mu/sd, define a local cdf closure summing ww*norm.cdf((v-m)/s) over the pre-extracted arrays, and call brentq on lambda v, qi=qi: cdf(v) - qi.

### [LOW] performance — t-mixture brentq inner lambda re-extracts params and reloops components per evaluation
`mixture_model_utils.py:1658-1665`

Confirmed: t_mixture_quantile_logspace passes lambda v: t_mixture_cdf_logspace(v, model) - qi to brentq per quantile. t_mixture_cdf_logspace (1615-1622) re-reads model.means_[k,0], np.sqrt(model.covariances_[k,0,0]), model.nus_[k] inside a Python per-component loop and reallocates out on every call; mu/sd extracted at 1652-1653 are not reused. Same micro-optimization as the Gaussian analog; same low-hotness caveat (Q-Q figure only).

**Fix:** Pre-extract mu, sd, nu, weights once, define a closure computing the weighted sum of t_dist.cdf((v-mu)/sd, df=nu) vectorized over components, and pass it to brentq.

### [LOW] performance — Loop-invariant inv_var recomputed every Newton iteration in gmm_modes 1D branch
`mixture_model_utils.py:625`

Confirmed line 625: inv_var = 1.0 / Sig[:, 0, 0] is constant across all peaks and all Newton iterations but is recomputed inside the inner for _ in range(max_iter) loop (up to n_peaks*max_iter times). The per-component log-weight/log-scale constants in the responsibility comprehension (615-620) are likewise loop-invariant. mu/sd are already hoisted at 598-599; inv_var can be hoisted too.

**Fix:** Compute inv_var = 1.0 / Sig[:, 0, 0] and per-component constant log(w[k]) - 0.5*log(Sig[k,0,0]) once before the for i in peaks_idx loop and reuse inside the Newton iteration; this also enables a vectorized responsibility computation.

### [LOW] tests — Midpoint-fallback root selection in gmm_boundaries_logspace is untested
`mixture_model_utils.py:281-284`

Confirmed line 284: x = candidates[0] if candidates else (r1 if abs(r1-mid) < abs(r2-mid) else r2). Existing boundary tests cover the linear branch, negative-discriminant NaN, in-bracket finite root (test_analyze.py 2488-2499), and a=0/b=0 NaN; none constructs a positive-discriminant quadratic whose two roots both fall outside [mu1, mu2], so the else clause of the candidates ternary is never taken.

**Fix:** Add a test with two adjacent unequal-variance components whose two quadratic roots both lie outside [mu1, mu2], then assert the returned boundary equals the root nearest 0.5*(mu1+mu2) rather than NaN, exercising line 284's else clause.

### [LOW] tests — tau != 0.5 boundary-direction behavior (the documented sign-fix) is untested
`mixture_model_utils.py:211-237`

Confirmed: the docstring (211-222) documents that increasing tau moves the boundary toward the left component and notes an earlier version had the wrong sign; the tau-dependent term + 2.0*np.log((1.0-tau)/tau) is line 267. Every boundary test uses tau=0.5 (test_mixture_model_utils.py 84/112; test_analyze.py 713, 2482, 2497). A re-introduced sign flip on line 267 would pass the whole suite.

**Fix:** Add a test with two well-separated equal-variance components asserting directionality: boundary(tau=0.7) < boundary(tau=0.5) < boundary(tau=0.3), locking in the documented sign convention.

### [LOW] tests — fit_log_t_mixture RuntimeError guard (best_model is None) is untested
`mixture_model_utils.py:1429-1431`

Confirmed lines 1429-1431 raise RuntimeError when best_model is None. No test passes n_init=0 (grep found none). The branch is uncovered. Note the suggested n_init=0 trigger would work since the for-loop at 1386 would not execute, leaving best_model None.

**Fix:** Add a test calling fit_log_t_mixture(intervals, n_components=2, n_init=0) and assert it raises RuntimeError matching 'did not produce a valid fit'; or add an upstream guard rejecting n_init<1 and test that.

### [LOW] tests — Weight-floor clamp in fit_log_t_mixture init is untested
`mixture_model_utils.py:1392-1394`

Confirmed lines 1393-1394: w = np.where(w < 1e-3, 1e-3, w); w = w / w.sum(). No test forces a degenerate KMeans partition where some np.mean(labels==k) < 1e-3, so the clamp is never validated. The where() always executes but its effect (a clamped weight) is never asserted on a fixture that triggers it.

**Fix:** Add a test fitting fit_log_t_mixture on data with K requested but only ~K-1 separable clusters (e.g. two tight blobs, n_components=3, n_init=1) and assert all returned model.weights_ are >= ~1e-3 and sum to 1.

### [LOW] tests — gmm_modes 1D non-finite-Hessian gradient fallback operand is not independently exercised
`mixture_model_utils.py:632-634`

Confirmed line 632: if not np.isfinite(hess) or hess >= 0.0. The existing fallback test (test_mixture_model_utils.py 144-168) targets the hess >= 0.0 sub-condition via a flat-topped mixture; the not np.isfinite(hess) operand is not deliberately driven. Branch-coverage-wise the OR's first operand is never independently true. Low severity since the fallback itself is covered and a non-finite Hessian is an unusual regime.

**Fix:** Extend the fallback test (or add one) that forces a non-finite Hessian at a polished candidate (e.g. a component with an extremely small Sig entry making inv_var overflow) and assert gmm_modes still returns finite modes.

### [LOW] tests — _lr_statistic only tested for alt-better (positive) case; zero/reshape cases untested
`mixture_model_utils.py:1813-1845`

Confirmed: test_lr_statistic_alt_better_returns_positive (test_analyze.py 1612-1621) passes log_x.ravel() (1D) and asserts a positive LR. The function computes -2*(ll_null-ll_alt) and reshapes input to 2D at line 1842; the ~0 case (identical null/alt) and the already-(N,1) input path are not separately tested. Minor robustness gap.

**Fix:** Add a test passing the same fitted model as both model_null and model_alt and assert _lr_statistic is approximately 0.0; add a second assertion passing log_x as a (N,1) array to confirm line-1842 reshape handles 1D and 2D identically.


## `compute_inter_usv_interval_distributions.py` (14)

### [HIGH] tests — fit_gmm_sweep has no direct test; t-mixture path and skip branches fully uncovered
`compute_inter_usv_interval_distributions.py:288-508`

fit_gmm_sweep is the largest branch-dense public function in the module yet is never tested directly. In test_iui_calculator.py it is monkeypatched away (line 229). It is exercised only indirectly via run_bic_sweep in tests/visualizations/test_visualize_stats.py::test_run_bic_sweep_feeds_plot_ic_curves, which passes only model_class='gauss' with 200 samples/sex (so len>=cv_n_folds and len>=2 always hold). Uncovered: (a) ValueError at 367-371 for bad model_class — NOTE: this IS covered indirectly by test_save_iui_invalid_model_class_raises (the driver re-checks at 627) but NOT at the fit_gmm_sweep boundary itself; (b) the entire t-mixture path (394-402, 444-455, 503-506); (c) the cv-skip branch 'if len(iui) < cv_n_folds: continue' (381-382); (d) 'if len(iui) < 2: continue' (428-429).

**Fix:** Add direct unit tests for fit_gmm_sweep: (1) ValueError match 'model_class must be' for model_class='bogus'; (2) gauss run with one key len < cv_n_folds asserting that key's cv_neg_loglik is NaN while params still fit; (3) a key with len < 2 producing zero rows for that key; (4) a model_class='t' run on a small bimodal sample asserting nu_k finite and boundary_*_k NaN.

### [MEDIUM] tests — compute_session_usv_intervals stop-from-duration branch (no 'stop' column) untested
`compute_inter_usv_interval_distributions.py:233-237`

The three compute_session_usv_intervals tests (test_analyze.py 2295-2364) all build fake_usv frames containing a 'stop' column, so only the 'if "stop" in usv_info.columns' branch (231-232) runs. The else branch (233-237) deriving stop = start + duration when no 'stop' column exists is never executed. This is a plausible production path, though note das_inference.py always writes a 'stop' column, so the no-stop case may not occur for in-house CSVs; still a defensible coverage gap for externally produced frames.

**Fix:** Add a test where load_and_filter_usv_data returns a frame WITHOUT 'stop' (only start, duration, emitter), interval_type='e2s', asserting intervals use start+duration (e.g. start=[0.0,1.0], duration=[0.2,0.2], emitter=['M','M'] -> 1.0-0.2=0.8).

### [MEDIUM] tests — Cross-sex (M-F-M / unassigned) emitter gating untested
`compute_inter_usv_interval_distributions.py:259-268`

The pointer-gating at 260-268 is the core correctness claim (docstring 149-155: a male->female->male triplet must NOT record a male-male interval skipping the female). test_compute_session_usv_intervals_basic_pairs uses a cleanly grouped M,M,F,F sequence (confirmed test_analyze.py:2306) where the pointer advance is incidental. No test covers interleaved M-F-M ordering or an emitter matching neither ID (the line 228 'unassigned' otherwise-branch and the 'usv0_sex in (male,female)' guard at line 260).

**Fix:** Add a test with emitter sequence ['M','F','M'] (s2s) asserting out['male'] and out['female'] are empty (no male-male interval skips the female). Add a second test with an emitter matching neither id (e.g. ['M','UNK','M']) asserting the unknown call routes to 'unassigned' and contributes no interval.

### [LOW] dead_code_naming — fit_gmm_sweep / 'gmm_fits' names hide the Student-t path; signature default diverges from config default
`compute_inter_usv_interval_distributions.py:288-301`

fit_gmm_sweep (line 288) and the archive group 'gmm_fits' (lines 740/760) are named for the Gaussian case but fully support model_class='t' (lines 394-402, 444-455, 503-506). The shipped config default is model_class='t' (analyses_settings.json:38), so under default config the function name and HDF5 group describe a model the run did not fit. The signature default model_class='gauss' (line 300) also diverges from the config default 't'. Class docstring already says 'GMM / t-mixture sweep' (line 516). This is a naming/clarity issue; renaming touches call sites in visualizations + tests.

**Fix:** Document the model-class-agnostic behavior at the function/group level (cheapest), or rename to fit_mixture_sweep / mixture_fits and update call sites in visualizations/usv_interval_summary_statistics.py and tests. Optionally align the signature default with the config default 't'.

### [LOW] docs_clarity — Docstrings/comment claim non-positive intervals are 'only possible in e2s'; s2s ties also yield non-positive
`compute_inter_usv_interval_distributions.py:16-19, 155, 273`

Module docstring (16-19), function docstring (154-155), and the per-mode-drop docs claim non-positive intervals occur only in e2s. In s2s, two consecutive same-sex USVs with identical 'start' produce interval == 0, which fails the strict 'interval > 0' test at line 262 and increments n_dropped. So s2s can also drop intervals. This is a minor doc inaccuracy; given start-sorting upstream, identical-start same-emitter pairs are rare but possible. Real but very low impact.

**Fix:** Soften wording to note s2s can also produce a zero (non-positive) interval when two same-animal calls share an identical start timestamp, so the drop count is meaningful (if rarely nonzero) in both modes.

### [LOW] docs_clarity — Comment claims modes are 'top-N by density' but code keeps first-N by ascending location
`compute_inter_usv_interval_distributions.py:484`

Comment at line 484 reads 'mixture modes (top-N by density, sorted ascending in seconds)'. report_gmm_stats (mixture_model_utils.py:750-752) re-sorts modes by ascending LOCATION via np.argsort(modes), discarding any density ordering; the loop at 487-489 keeps the first min(max_modes_reported, n_modes), i.e. lowest-location modes, with no density-based selection. For the t-path (449-453) the entries are per-component means, not mixture modes. The comment misstates the selection criterion (location, not density). Merged with the duplicate docs_clarity finding for the same line.

**Fix:** Reword to e.g. '# mixture modes (Gaussian) or per-component peaks (t), kept in ascending-location order; first max_modes_reported recorded.' Drop the 'top-N by density' claim.

### [LOW] docs_clarity — Module docstring omits the Student-t mixture path the file fully supports
`compute_inter_usv_interval_distributions.py:4-6`

Module docstring (lines 4-6) says the analysis sweeps 'a 1D Gaussian Mixture Model'. The file is model-class agnostic: fit_gmm_sweep accepts model_class in ('gauss','t') and dispatches to fit_log_t_mixture / report_t_mixture_stats / t_mixture_icl / t_mixture_cv_neg_loglik (lines 300, 367, 394-402, 444-455). Config default is even 't'. The class docstring at line 516 correctly says 'GMM / t-mixture sweep', so the module header is stale relative to the rest of the file.

**Fix:** Extend the module docstring to state the sweep covers either a Gaussian or a Student-t mixture (selected via model_class), matching line 516.

### [LOW] docs_clarity — Returns docstring calls t-path mode_sec_k/density_k 'mixture modes' though they are per-component means
`compute_inter_usv_interval_distributions.py:355-364`

fit_gmm_sweep Returns section (lines 361-362) describes mode_sec_k/density_k as 'the top max_modes_reported mixture modes'. Accurate only for gauss. For model_class='t', lines 449-453 set modes_log = logmeans.copy() and densities = mode_dens, so the columns hold per-component log-means and the mixture density at each component peak, not GMM-style mixture modes, and there is no top-N-by-density selection. Minor doc precision issue.

**Fix:** Add a clause that for the Student-t path mode_sec_k/density_k carry per-component peak locations/densities rather than distinct mixture modes.

### [LOW] performance — Per-session USV CSV + tracking H5 re-read once per interval_type (doubled I/O)
`compute_inter_usv_interval_distributions.py:670-705`

The outer loop 'for interval_type in interval_types:' (670) iterates 's2s' and 'e2s', and for each session calls compute_session_usv_intervals (676), which re-runs extract_session_metadata (recursive glob + h5py open) AND load_and_filter_usv_data (recursive glob + full read_csv + filter). The only per-mode difference is the cheap usv0_tag/usv1_tag selection (204) and the O(n) pointer loop (255-269). So each session's H5 + CSV (+ two recursive globs) are parsed TWICE per run. The comment at 605-606 claims the per-session pass is 'shared', but it is duplicated. Real inefficiency; magnitude is exactly 2x for the 2-element interval_types tuple.

**Fix:** Split compute_session_usv_intervals into a per-session load step (metadata + start/stop/sex arrays) cached once, and a cheap per-mode _intervals_from_arrays(start_arr, stop_arr, sex_arr, interval_type). Run both modes over the cached arrays, removing one glob+H5-read+glob+CSV-read+filter per session.

### [LOW] performance — Session-list files opened and parsed twice; configure_path runs twice per line
`compute_inter_usv_interval_distributions.py:645-646`

_read_session_lists (line 645 / def 51) and _session_source_map (line 646 / def 98) each independently open every file in session_lists, iterate every line, and call configure_path on each line (88 and 129). They produce complementary outputs (ordered de-duped list vs session->source-stem map) from the same line scan, so each list file is read twice and each path runs through configure_path twice.

**Fix:** Merge into a single traversal returning both the ordered sessions list and the source_map dict (one configure_path call per line, one open per file), called once at 645-646.

### [LOW] performance — tidy_rows built with per-element Python loop and scalar np.log instead of vectorized
`compute_inter_usv_interval_distributions.py:694-705`

Lines 694-705 iterate 'for v in arr:' over each session's male/female interval arrays, appending a dict per interval and calling float(np.log(v)) per scalar. The arrays are numpy float arrays; scalar np.log pays ufunc dispatch per element. For cohorts with many thousands of intervals this is a hot O(total_intervals) Python loop feeding pls.from_dicts at 723.

**Fix:** Hoist log_arr = np.log(arr) out of the inner loop (vectorized), or build per-session polars frames directly from numpy interval_s/log_interval arrays with broadcast string columns and pls.concat them.

### [LOW] tests — Raw-vs-stripped emitter ID matching (null-byte/whitespace padding) untested
`compute_inter_usv_interval_distributions.py:188-191, 223-228`

Lines 190-191 strip null bytes/whitespace from male_id/female_id, and sex_expr (223-228) matches BOTH the raw padded id (224-225) AND the stripped id (226-227). The inline comment (219-222) records that matching only the raw form previously produced a silent M=0/F=0 regression. No test passes a metadata id with trailing null/whitespace against a clean CSV emitter, so the regression-guarding stripped-match arms (226-227) are uncovered.

**Fix:** Add a test where extract_session_metadata returns male_id='M\x00 ' / female_id='F\x00' but the emitter column holds clean 'M'/'F', asserting intervals are assigned to the correct sex (not all 'unassigned'), exercising the stripped-id arms at 226-227.

### [LOW] tests — load_and_filter_usv_data FileNotFoundError early-return branch untested
`compute_inter_usv_interval_distributions.py:200-201`

compute_session_usv_intervals returns {} on FileNotFoundError from load_and_filter_usv_data (200-201), distinct from the metadata except path at 185-186. test_compute_session_usv_intervals_missing_session_returns_empty (test_analyze.py:2279) only patches extract_session_metadata to raise; the USV-file-missing branch is not covered.

**Fix:** Add a test where extract_session_metadata succeeds but load_and_filter_usv_data raises FileNotFoundError, asserting the function returns {}.

### [LOW] tests — metadata IndexError early-return arm untested (only FileNotFoundError covered)
`compute_inter_usv_interval_distributions.py:185-186`

Line 185 catches (FileNotFoundError, IndexError) from extract_session_metadata, returning {}. extract_session_metadata raises IndexError when a session has <2 tracks (_usv_io.py:55-56). test_compute_session_usv_intervals_missing_session_returns_empty raises only FileNotFoundError, so the IndexError arm is unverified.

**Fix:** Parametrize the existing missing-session test to also raise IndexError, asserting {} is returned.


## `decode_experiment_label.py` (10)

### [LOW] docs_clarity — Docstring legend misspells estrous stage 'metestrus' as 'matestrus'
`decode_experiment_label.py:44`

Line 44 reads `m - matestrus`. The canonical spelling used by the project GUI (src/usv_playpen/usv_playpen_gui.py:3055: ['N/A','proestrus','estrus','metestrus','diestrus']) is 'metestrus'. The docstring is inconsistent with the rest of the codebase and could mislead a reader about the stage denoted by 'm'.

**Fix:** Change line 44 from `m - matestrus` to `m - metestrus`.

### [LOW] docs_clarity — Decoded value string misspells 'metestrus' as 'matestrus'
`decode_experiment_label.py:82`

decoding_dict maps `"m": "matestrus"` (line 82), so output_dict['mouse_estrus'] returns the misspelled stage while the GUI (usv_playpen_gui.py:3055) uses 'metestrus'. This is load-bearing output. Note: the downstream notebook usv_summary_statistics_plots.ipynb:833 also carries 'matestrus', so a fix must update that notebook too (and the test parametrize at test_analyze.py:363).

**Fix:** Change line 82 to `"m": "metestrus",` and update the notebook (ipynb:833) and test expectation (test_analyze.py:363) to match.

### [LOW] docs_clarity — Docstring parameter type omits the None case
`decode_experiment_label.py:50`

Line 50 documents `experiment_code (str)` while the signature is `experiment_code: str | None = None` and None is an accepted input that yields a None return. The type label is incomplete.

**Fix:** Change line 50 to `experiment_code (str | None)`.

### [LOW] docs_clarity — Returns docstring does not document the None return path
`decode_experiment_label.py:53-58`

The Returns section only describes `output_dict (dict)`, but the function returns None for None/non-str input (line 114), matching the `dict | None` annotation. Documenting the None path makes the contract explicit.

**Fix:** Add a sentence to the Returns section noting None is returned when experiment_code is None or not a string.

### [LOW] tests — experiment_type letters A, H, V, U, O, P, X, Y have no decode test
`decode_experiment_label.py:87`

experiment_type regex covers 14 codes (lines 62-73) but tests only exercise E, C, B, Q, L, D (test_analyze.py:325-357). Codes A (ablation), H (chemogenetics), O (optogenetics), P (playback), V (devocalization), U (urine/bedding), X (females), Y (males) are decoded but never asserted, so a typo in those dict values would go undetected.

**Fix:** Add a parametrized test mapping each letter to its expected decoded string and assert it appears in extract_information(f'{letter}1')['experiment_type'].

### [LOW] tests — Exact experiment_type list (findall/append accumulation) never asserted
`decode_experiment_label.py:104-105`

Tests only assert membership ('ephys' in / 'courtship' in) for 'E2CFM' (test_analyze.py:325-326); no test asserts the exact list value/length/ordering produced by the re.findall+append loop at lines 104-105.

**Fix:** Add assert extract_information('EC1')['experiment_type'] == ['ephys','courtship'] to lock in accumulation and ordering.

### [LOW] tests — mouse_housing 'group' (G) decode never asserted
`decode_experiment_label.py:90`

mouse_housing regex r'[SG]' (line 90) maps G->group (line 79), but only the 'single' (S) case is asserted (test_analyze.py:338). The 'BG12' test (line 344) checks only mouse_number, not mouse_housing == ['group']. The G->group decode is untested.

**Fix:** Extend the 'BG12' test (test_analyze.py:344) with assert out['mouse_housing'] == ['group'].

### [LOW] tests — mouse_sex male-only case and order preservation untested
`decode_experiment_label.py:89`

mouse_sex regex r'[MF]' (line 89). Tests assert ['female','male'] for 'E2CFM' (test_analyze.py:328) and ['female'] solo (line 337), but no test covers a male-only code (-> ['male']) nor reversed input order ('MF' -> ['male','female']) to confirm findall preserves source order.

**Fix:** Add assert extract_information('E1MS')['mouse_sex'] == ['male'] and a reversed-order case.

### [LOW] tests — Empty-string input branch untested
`decode_experiment_label.py:60`

An empty string '' is a str and not None, so it passes the guard at line 60 and returns the all-default output_dict rather than None. The None (line 314) and non-string (lines 318-319) paths are tested, but the falsy-but-valid '' case is not.

**Fix:** Add a test asserting extract_information('') returns the all-default dict (not None).

### [LOW] tests — Multi-digit-run boundary and leading-zero number cases untested
`decode_experiment_label.py:107-110`

The mouse_number branch (lines 107-110) uses re.search (first digit run only) then int(). Tests cover '12' (line 344) and no-digit->0 (line 349), but not a code where digit runs are separated by letters (e.g. 'E1B2' -> first run '1') nor leading zero (e.g. 'E01' -> 1). These pin the first-match-only and int() coercion behavior.

**Fix:** Add assert extract_information('E1B2')['mouse_number'] == 1 and assert extract_information('E01')['mouse_number'] == 1.


## `analyze_data.py` (10)

### [MEDIUM] docs_clarity — analyze_data docstring omits inter-USV interval analysis and lists wrong order
`analyze_data.py:64-69`

The docstring (lines 65-68) enumerates four analyses: behavioral features, neuronal tuning, playback WAV, frequency shift. The method also runs a fifth analysis -- inter-USV-interval distributions (line 117, InterUSVIntervalCalculator) -- which is entirely absent. The stated order is also stale: actual execution is inter-USV intervals (117), playback WAV (122-130), then per-directory behavioral features (139), neuronal tuning (145), frequency shift (151).

**Fix:** Add a bullet for inter-USV-interval distribution computation and reorder the list to match actual execution order (inter-USV intervals, playback WAV generation, then per-directory behavioral features / neuronal tuning / frequency shift).

### [MEDIUM] tests — generate_usv_interval_distributions_cli has no test
`analyze_data.py:359-384`

generate_usv_interval_distributions_cli (the inter-USV interval / GMM-sweep CLI entry point, ~20 options including list-valued session_lists/noise_categories routed via parameters_lists at line 375) is never imported or invoked in any test. tests/foundation/test_cli.py imports only generate_beh_features_cli, generate_usv_playback_cli, and generate_rm_files_cli (lines 9-13). No coverage exists for the provided_params collection, the parameters_lists=['session_lists','noise_categories'] wiring, or the dispatch to InterUSVIntervalCalculator.save_inter_usv_interval_distributions_to_file (lines 383-384).

**Fix:** Add a test in tests/foundation/test_cli.py mirroring test_generate_rm_files_cli_success: import generate_usv_interval_distributions_cli, mock InterUSVIntervalCalculator and modify_settings_json_for_cli (returning a dict), invoke with multi-value options, assert exit_code == 0 and that save_inter_usv_interval_distributions_to_file was called once.

### [MEDIUM] tests — generate_naturalistic_usv_playback_cli is untested
`analyze_data.py:226-254`

generate_naturalistic_usv_playback_cli (lines 226-254) is never imported or invoked in tests/foundation/test_cli.py, unlike its sibling generate_usv_playback_cli (tested at test_cli.py:53). It uniquely exercises the create_naturalistic_usv_playback_wav settings key (line 253) and backend method (line 254). Separately, line 247 uses fully-qualified click.core.ParameterSource.COMMANDLINE whereas the other three CLIs use the bare imported ParameterSource (lines 217, 293, 326); this divergent line is entirely uncovered.

**Fix:** Add test_generate_naturalistic_usv_playback_cli_success in tests/foundation/test_cli.py: import the command, mock AudioGenerator and modify_settings_json_for_cli, runner.invoke(generate_naturalistic_usv_playback_cli, ['--exp-id', 'TestExp']), assert exit_code == 0 and create_naturalistic_usv_playback_wav called once. Optionally normalize line 247 to the bare ParameterSource for consistency.

### [MEDIUM] tests — compute_inter_usv_interval_distributions routing branch is never exercised
`analyze_data.py:117-119`

InterUSVIntervalCalculator is patched in mock_dependencies (test_analyze.py:150) but no test sets analyses_booleans['compute_inter_usv_interval_distributions_bool']=True. The branch at lines 117-119 (constructs InterUSVIntervalCalculator and calls save_inter_usv_interval_distributions_to_file) is therefore uncovered, while every other analysis flag has a dedicated routing test (behavioral 207, tuning 229, both playback variants 247/268, freq-shift 290).

**Fix:** Add test_compute_inter_usv_interval_logic mirroring test_compute_tuning_curves_logic: set mock_settings['analyses_booleans']['compute_inter_usv_interval_distributions_bool']=True, construct Analyst with root_directories=[], call analyze_data(), assert mock_dependencies['InterUSVIntervalCalculator'] was called once and its return_value.save_inter_usv_interval_distributions_to_file was called once.

### [LOW] correctness — generate-rm multi-value CLI options stored as tuples, not lists (inconsistent with sibling commands)
`analyze_data.py:258, 263, 293-297`

generate_rm_files_cli calls modify_settings_json_for_cli WITHOUT a parameters_lists argument (lines 295-297), so parameters_lists defaults to [] (cli_utils.py:164-165). For any provided param not in parameters_lists, cli_utils.py:169 writes ctx.params[param_name] verbatim. --temporal-offsets (multiple=True, line 258) and --peth-window-seconds (nargs=2, line 263) therefore arrive as Python tuples and are stored as tuples, whereas the JSON defaults are lists (analyses_settings.json:90 'temporal_offsets': [0]; line 97 'peth_window_seconds': [-2, 0]). Sibling CLIs generate_beh_features_cli (line 324) and generate_usv_interval_distributions_cli (line 375) explicitly pass parameters_lists for their multi-value options so they are coerced to lists (cli_utils.py:170-171). The asymmetry is real and contradicts the in-file pattern; it does not currently crash only because downstream code is tuple-tolerant.

**Fix:** Add parameters_lists=['temporal_offsets', 'peth_window_seconds'] to the modify_settings_json_for_cli call in generate_rm_files_cli (lines 295-297), mirroring the sibling CLIs, so multi-value overrides are stored as lists consistent with the JSON defaults.

### [LOW] correctness — Analysis exceptions outside the hard-coded except tuple skip the completion email entirely
`analyze_data.py:159-167`

The per-directory loop catches only an explicit tuple (line 159: OSError, RuntimeError, TypeError, IndexError, IOError, EOFError, TimeoutError, NameError, KeyError, ValueError, AttributeError). An analysis error outside this set (e.g. MemoryError, or a library exception subclass not derived from one of these) propagates out of _analyze_data_impl before the completion-email block (lines 167-192), so the honest failure notification the comment at lines 163-166 was added to provide is never sent for that exception class. KeyboardInterrupt/SystemExit are correctly excluded. Severity is low because the listed types already cover essentially all realistic analysis failures.

**Fix:** Either broaden the handler to `except Exception as exc:` (KeyboardInterrupt/SystemExit derive from BaseException and still propagate), or wrap the loop+email block so the completion email is sent from a finally clause regardless of which exception escaped.

### [LOW] docs_clarity — generate_naturalistic_usv_playback_cli docstring is a verbatim copy of the non-naturalistic CLI docstring
`analyze_data.py:234-237`

Line 237 reads 'A command-line tool to generate USV playback WAV files.' -- identical to generate_usv_playback_cli (line 207) -- and never mentions 'naturalistic', even though this command calls a different settings key (create_naturalistic_usv_playback_wav, line 253) and a different backend method (AudioGenerator.create_naturalistic_usv_playback_wav, line 254).

**Fix:** Reword line 237 to 'A command-line tool to generate naturalistic USV playback WAV files.' (or otherwise describe how it differs from the non-naturalistic playback command).

### [LOW] docs_clarity — generate_rm docstring labels two distinct vocal analyses both as 'Q3'
`analyze_data.py:278-280`

The Description (lines 278-280) lists 'Q1 pre-USV PETH, Q2 within-USV continuous-property tuning, Q3 within-USV categorical, Q3 per-category PETH'. The Q3 label is applied to two different items, which reads as a typo (second should likely be Q4) or an undocumented sub-grouping, making the count/numbering of distinct vocal analyses ambiguous.

**Fix:** Clarify the numbering: if the two are sub-parts of one question write 'Q3a within-USV categorical and Q3b per-category PETH'; if separate, renumber the second to Q4.

### [LOW] tests — Analyst.__init__ default settings-loading branch is never tested
`analyze_data.py:52-58`

Every test constructs Analyst with explicit input_parameter_dict AND root_directories, so the branch at lines 52-54 (opening analyses_settings.json when either arg is None) and the fallbacks at lines 56-57 are never executed. NOTE: the finding's accompanying 'latent NameError' claim is FALSE -- the guard at line 52 is `input_parameter_dict is None OR root_directories is None`, so _settings is bound whenever EITHER arg is None, which is exactly when lines 56-57 reference it; the partial-override case does not raise NameError. The only real issue is the missing coverage of the default-loading branch.

**Fix:** Add a test constructing Analyst() with no args (with the settings JSON present or modify_settings_json_for_cli-style fixture) and assert self.input_parameter_dict and self.root_directories are populated from analyses_settings.json. Do not file the NameError concern as a bug.

### [LOW] tests — RuntimeWarning suppression scoping in analyze_data is not asserted
`analyze_data.py:87-89`

analyze_data (lines 87-89) exists to scope warnings.simplefilter('ignore', RuntimeWarning) around _analyze_data_impl via catch_warnings, per the rationale comment (lines 78-86) about not silencing warnings process-wide. No test verifies that a RuntimeWarning raised inside the run is suppressed nor that suppression is reverted afterwards (a RuntimeWarning after analyze_data returns should still surface). This is the load-bearing behavior the refactor introduced.

**Fix:** Add a test that monkeypatches _analyze_data_impl to emit warnings.warn('x', RuntimeWarning), uses recwarn to assert no RuntimeWarning escapes analyze_data(), then emits a RuntimeWarning after the call and asserts it IS recorded, proving catch_warnings restored the global filter state.


## `neuronal_coactivity_engine.py` (9)

### [HIGH] performance — apply_circular_shift re-sorts every neuron from scratch each shuffle iteration
`neuronal_coactivity_engine.py:417-421`

apply_circular_shift runs np.sort((spikes + shift_s) % total_duration_s) for every neuron on every call. It is called n_shuffles times in perform_circular_shuffle (default 1000) and n_shuffles*n_sessions times in perform_chained_circular_shuffle, so this O(N_spikes log N_spikes) sort per neuron per iteration dominates the null-distribution runtime the file otherwise carefully optimizes (cf. the binary-search comment at lines 55-60). Input arrays are already sorted (Kilosort output and this function's own prior output). A constant additive circular shift of a sorted array splits it into exactly two already-sorted runs at the wrap point, so the result is an O(N) split-and-concatenate via searchsorted on T - shift_s instead of a full sort.

**Fix:** Replace per-neuron np.sort with a rotation: split = np.searchsorted(spikes, total_duration_s - shift_s, side='left'); shifted = np.concatenate([(spikes[split:] + shift_s) - total_duration_s, spikes[:split] + shift_s]) (valid for shift_s in [0, total_duration_s), which the default [20,60] shift window satisfies). Preserves the docstring's sorted-output guarantee.

### [MEDIUM] docs_clarity — perform_chained_circular_shuffle docstring omits min_shift_s and max_shift_s parameters
`neuronal_coactivity_engine.py:530-545`

The signature (lines 516-517) takes min_shift_s=20.0 and max_shift_s=60.0, but the Parameters section documents only session_onsets, session_neural_data, session_durations, window_s, n_shuffles, and seed. The two shift-bound parameters that define the circular-offset range are entirely missing, violating the file's Description/Parameters/Returns convention. n_shuffles (lines 539-540) also lacks the ', optional'/default annotation the sibling perform_circular_shuffle uses.

**Fix:** Add min_shift_s and max_shift_s entries (mirroring perform_circular_shuffle lines 456-459, defaults 20.0/60.0) and align the n_shuffles entry with the ', optional'/default style.

### [MEDIUM] docs_clarity — compute_sliding_coactivity Returns omits the 'time_bins' key
`neuronal_coactivity_engine.py:703-706`

The function returns a dict with keys 'time_bins', 'r_sc', and 'similarity' (lines 723-727), but the Returns section says only "Arrays of 'r_sc' and 'similarity' for each time bin." The 'time_bins' array (per-step offsets, line 710/724) is returned but undocumented.

**Fix:** Document all three returned keys, e.g. "Dictionary with 'time_bins' (per-step onset offsets in seconds), 'r_sc', and 'similarity', each a length-n_steps array."

### [MEDIUM] docs_clarity — perform_label_permutation_test Returns omits null_mean and null_std keys
`neuronal_coactivity_engine.py:624-628`

The Returns docstring lists the per-metric mapping as {observed_delta, null, p_a_gt_b, p_two_tailed, z_score}, but the code (lines 666-674) also stores 'null_mean' and 'null_std' in each per-metric result dict. Two of the seven returned keys are undocumented.

**Fix:** Add null_mean and null_std to the documented key list, e.g. metric -> {observed_delta, null, null_mean, null_std, p_a_gt_b, p_two_tailed, z_score}.

### [MEDIUM] performance — cosine_similarity materializes a full T x T matrix only to average its off-diagonal
`neuronal_coactivity_engine.py:288-290`

cosine_similarity(count_matrix.T) (line 288) builds the full (N_trials x N_trials) similarity matrix and only the upper-triangle mean is consumed (lines 289-290). compute_coactivity_metrics runs inside every bootstrap/shuffle/permutation iteration, so this allocates an O(T^2) matrix per iteration when only the scalar mean off-diagonal cosine is needed. With L2-normalized columns U, the mean off-diagonal equals (U.sum(axis=1) @ U.sum(axis=1) - num_trials) / (num_trials*(num_trials-1)), an O(N*T) reduction.

**Fix:** Normalize columns once (norms = np.linalg.norm(count_matrix, axis=0); guard zero norms), then mean_offdiag = (U.sum(axis=1) @ U.sum(axis=1) - num_trials) / (num_trials*(num_trials-1)). Add a brief comment since the algebra is non-obvious. Note: a zero-norm column currently yields NaN in cosine_similarity which np.nanmean drops, so the guard must reproduce that (set those columns to 0 so they contribute 0, not NaN) to stay numerically equivalent.

### [LOW] correctness — compute_coactivity_metrics returns real NaNs (not the early-exit sentinel) when num_trials < 2
`neuronal_coactivity_engine.py:275`

The function only early-exits on num_neurons < 2 (line 274). When num_neurons >= 2 but num_trials < 2, np.triu_indices(num_trials, k=1) is empty and np.nanmean of an empty slice emits a 'Mean of empty slice' RuntimeWarning returning NaN, while cosine_similarity and np.corrcoef on a single column degenerate. perform_label_permutation_test then computes `null >= nan` -> all False, yielding p = 1/(n+1) with no signal the comparison was undefined.

**Fix:** Mirror the num_neurons guard with a num_trials < 2 early return of the NaN sentinel dict so callers see an explicit, intentional NaN, and optionally propagate a degenerate-resample flag.

### [LOW] dead_code_naming — Unused loop variable `duration` in the dummy metric-key pass
`neuronal_coactivity_engine.py:554`

In perform_chained_circular_shuffle the first loop `for onsets, neural, duration in zip(...)` (line 554) only derives metric keys; its body (line 556) never references `duration`. Durations are only needed in the real shuffle loop (line 567, used at 570). The binding is dead and F841 does not flag tuple-unpacking targets.

**Fix:** Rename the unused target to `_duration`: `for onsets, neural, _duration in zip(session_onsets, session_neural_data, session_durations):`.

### [LOW] docs_clarity — load_animal_sessions documents message_output default as 'print' but signature default is None
`neuronal_coactivity_engine.py:935-936`

The Parameters entry says `message_output : Callable, optional` "Diagnostic sink; defaults to the built-in print." The signature default is None (line 891); the print fallback is applied at runtime via `log = message_output or print` (line 946). The same phrasing appears in compute_group_acoustics (lines 1041-1042). The wording reads as if the default value were print itself rather than None.

**Fix:** Reword to "defaults to None, in which case the built-in print is used" in both load_animal_sessions (935-936) and compute_group_acoustics (1041-1042). Also note the annotation is `Any`, not `Callable`, in the signature.

### [LOW] performance — Dummy metric-key probe in perform_circular_shuffle does redundant full-matrix work
`neuronal_coactivity_engine.py:479-480`

To discover the three metric keys, perform_circular_shuffle extracts the snippet matrix over ALL onsets (line 479) and runs the full compute_coactivity_metrics (line 480), discarding the result. Only the dict keys ('r_sc','similarity','pop_corr') are needed. The chained variant already avoids this by slicing onsets[:2] (line 556); this variant pays a full corrcoef/cosine pass at setup.

**Fix:** Probe keys cheaply mirroring the chained variant: extract_snippet_matrix(onsets[:2], neural_data, window_s) before compute_coactivity_metrics, or hoist the known key tuple to a module-level constant reused by all callers.


## `_usv_io.py` (5)

### [LOW] correctness — Null in noise column is silently dropped rather than retained
`_usv_io.py:110-112`

Confirmed empirically (polars 1.34.0): for a null value, pls.col(noise_col_id).is_in(noise_categories) evaluates to null, and ~null stays null, which filter() drops. Per the docstring, noise is defined as rows whose noise-col value IS in noise_categories; a null value is not in that list, so the vocalization should be retained, but it is instead discarded. The asymmetry is a genuine, subtle data-loss path. Triggerability is uncertain in practice because the cluster/noise column is normally written as non-null integer IDs by the upstream pipeline, hence low severity.

**Fix:** Make null handling explicit, e.g. usv_info.filter(pls.col(noise_col_id).is_in(noise_categories).not_().fill_null(True)) to retain rows whose noise value is null, or document/validate that the noise column is guaranteed non-null.

### [LOW] docs_clarity — Returns docstring claims a column subset the function never enforces
`_usv_io.py:96-97`

The Returns section states the DataFrame 'Contains USV starts, durations, emitters, and a newly calculated frame_index'. In fact the function returns every column present in the source *_usv_summary.csv (noise rows removed) plus the appended frame_index; it never selects a starts/durations/emitters subset. The wording implies a curated column set that is not produced, which is a factual inaccuracy (not mere verbosity).

**Fix:** Reword to: 'All columns from the USV summary CSV with noise rows removed, plus a newly calculated frame_index column.'

### [LOW] docs_clarity — Description lists 'experimental codes' (plural) but returns a single experiment_code
`_usv_io.py:27`

Line 27-28 says the function extracts 'mouse track names, recording frame rate, and experimental codes'. The actual return is a single 'experiment_code', and the raw track names are not returned as such — they are mapped to 'male_id'/'female_id'. The plural and the 'track names' framing are mildly inconsistent with the documented return keys.

**Fix:** Reword line 27-28 to match the returned keys, e.g. '...including animal identity strings (male_id, female_id), recording frame rate, and the experimental code.'

### [LOW] tests — Empty noise_categories (no-filter) branch is untested
`_usv_io.py:110-112`

Confirmed: the only test passing noise_categories=[] (test_load_and_filter_usv_data_missing_csv_raises, line 266-272) raises FileNotFoundError at line 103-105 before reaching the filter. The empty-list path where is_in([]) yields all-False so ~ retains every row is never verified on a valid session. A regression inverting the filter or mishandling the empty list would not be caught.

**Fix:** Add a test using a valid synthetic session with n_noise>0, call load_and_filter_usv_data(..., noise_categories=[]), and assert df.height equals the full (noise-inclusive) row count.

### [LOW] tests — frame_index column dtype (UInt32) is not asserted
`_usv_io.py:114-115`

Confirmed: test_load_and_filter_usv_data_drops_noise_and_adds_frame_index (line 249-263) checks only df['frame_index'][0] value and column presence, not dtype. The explicit .cast(pls.UInt32) is a contract; dropping it (leaving Float64 after floor()) would still pass the numeric-equality assertion, so the dtype contract is unverified.

**Fix:** Extend the existing happy-path test with assert df.schema['frame_index'] == pls.UInt32 (or df['frame_index'].dtype == pls.UInt32).
