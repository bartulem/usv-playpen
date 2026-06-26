# Processing subsystem review

_Verified line-by-line sweep of 20 files (~10.5k LOC): 280 findings. Report-first._

## Summary
- by severity: high 8 · medium 66 · low 206
- by dimension: tests 119 · docs_clarity 61 · correctness 55 · performance 24 · dead_code_naming 21


## `synchronize_files.py` (30)

### [HIGH] correctness — `or` instead of `and` causes None-arithmetic crash when tracking start/end not found
`synchronize_files.py:344-346`

`find_lsb_changes` returns `(None, None, ...)` (line 444) when `total_frame_number + largest_break_end_hop > ttl_break_end_samples.shape[0]`. The guard at line 344 is `if (tracking_start, tracking_end) != (None, None) or largest_break_duration_sec < 2:`. When start/end are None AND the largest break is < 2 s, the left operand is False but the right is True, so the branch is entered and line 346 computes `None - None`, raising TypeError. The else branch (lines 398-400, 'Tracking end exceeds e-phys recording boundary, so not found') is clearly the intended destination for the None case. The operator should be `and`.

**Fix:** Change line 344 to `if (tracking_start, tracking_end) != (None, None) and largest_break_duration_sec < 2:` so the body is not entered when tracking_start/tracking_end are None.

### [HIGH] tests — find_ipi_intervals static method: all four LSB-alignment branches untested (always mocked)
`synchronize_files.py:446-494`

find_ipi_intervals (lines 446-494) is a @staticmethod @njit function with four branches selected by `ipi_start_samples[0] < ipi_end_samples[0]` (line 479) and array-size equality (lines 480, 487). Grep confirms it is mocked out in every test that touches it (lines 758, 1199, 1297) and never invoked for real, so none of the duration arithmetic (+1, *1000/audio_sr scaling, [:-1]/[1:] slicing) is exercised. This is the core audio-IPI extraction routine and is entirely unverified.

**Fix:** Add direct tests calling Synchronizer.find_ipi_intervals on small synthetic LSB sound arrays covering all four combinations, asserting ipi_durations_ms and audio_ipi_start_samples match hand-computed values at a known audio_sr_rate.

### [MEDIUM] correctness — Cross-device IPI start-sample difference assumes equal-length arrays
`synchronize_files.py:922-924`

Line 922 sets `audio_devices_start_sample_differences = audio_ipi_start_samples` for the first device; line 924 computes `audio_devices_start_sample_differences - audio_ipi_start_samples` for the second. If the two devices detected a different number of IPI start samples (a dropped/extra pulse — exactly the desync this module exists to detect), the elementwise subtraction raises a broadcasting ValueError, crashing the whole method before the per-file results are returned. No length check or alignment precedes the subtraction.

**Fix:** Before subtracting, guard for equal length (truncate both to `min(len(...))` with a logged warning, or emit a message and skip the cross-device comparison when counts differ) so unequal pulse counts do not crash.

### [MEDIUM] docs_clarity — find_audio_sync_trains docstring omits consequential side effects (npy write, irreversible rmtree)
`synchronize_files.py:823-837`

The docstring documents only the returned ipi_discrepancy_dict, but the method also writes nidq_ipi_data.npy (lines 907-908) and, when the audio/video sync passes tolerance, irreversibly deletes the audio/original directory via shutil.rmtree (lines 1011-1014). Deleting source data is a consequential, undocumented behavior.

**Fix:** Extend the Description/Returns to note that NIDQ IPI data is cached to disk and that the original (uncropped) audio directory is removed when the audio/video sync passes tolerance.

### [MEDIUM] performance — Brightest-frame search re-decodes the same frames once per LED with random seeks
`synchronize_files.py:538-561`

In gather_px_information the per-LED loop (line 538) runs the brightest-frame search (lines 551-561) independently for each of the 3 LED positions. Each iteration calls `cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)` then `cap.read()`, forcing decoder keyframe re-seeks on an H.264 stream essentially every frame, and the same first ~1.5*fps frames are decoded and BGR2GRAY-converted three separate times. The three LED ROIs could be scanned in a single sequential pass: read each frame once, convert once, update all three per-LED peak trackers.

**Fix:** Read the first max_frame_num frames sequentially once (cap.set(POS_FRAMES,0) then a single read loop), convert each to grayscale once, and inside that loop update peak_intensity/peak_intensity_frame_loc for all three LED ROIs; do the per-LED centroid seek afterward.

### [MEDIUM] performance — CoolTerm log re-read and re-parsed once per sync camera
`synchronize_files.py:774-782`

In find_video_sync_trains the block reading the CoolTerm log and building arduino_ipi_durations (lines 774-782) sits inside the per-camera loop (line 752). The file content does not depend on camera_dir, so it is opened, readlines()'d, parsed line-by-line and converted to an array again for every sync camera. It should be read and parsed once before the camera loop.

**Fix:** Hoist the CoolTerm read/parse out of the camera loop (compute arduino_ipi_durations once before iterating camera_dir) and reference the cached array inside the loop.

### [MEDIUM] performance — O(n*m) Python loop with per-iteration list() conversion to map audio IPI starts to video frames
`synchronize_files.py:989-1004`

The loop over _audio_starts (lines 990-1004) computes, per audio IPI start, the last video frame start preceding it via `list(temp_arr).index(max(temp_arr[negative_mask]))` (line 1004) after recomputing temp_arr and negative_mask over the full video_fr_starts_in_samples array each iteration. This is O(n_audio * n_video) with a per-iteration Python-list materialization. Since video_fr_starts_in_samples is monotonically increasing (cumulative frame starts), the result is `np.searchsorted(video_fr_starts_in_samples, _audio_starts, side='left') - 1`, with idx < 0 as the NaN case.

**Fix:** Replace the loop with `idx = np.searchsorted(video_fr_starts_in_samples, _audio_starts, side='left') - 1`, mark `idx < 0` as NaN (emitting the existing per-event message), and use idx for the rest.

### [MEDIUM] tests — Module-level find_events has no direct unit test
`synchronize_files.py:37-66`

find_events (lines 37-66) is a public, pure, easily-testable function implementing the rising/falling debounce logic. Grep confirms it is never referenced directly in the test suite; it is only exercised transitively through find_video_sync_trains' integration test, which asserts only ipi_starts.size > 0 and cannot pin the `+1` offset on neg_events (line 64) or the `stable` precondition (line 59).

**Fix:** Add a direct test constructing a small diffs array with a known rising edge preceded by a stable sample and a falling edge, asserting find_events returns the exact pos_events/neg_events (verifying the +1 offset and the stability gate), plus an all-below-threshold case returning two empty arrays.

### [MEDIUM] tests — filter_events_by_duration: empty-input early return and glitch-removal logic untested directly
`synchronize_files.py:96-137`

filter_events_by_duration (lines 96-137) has an empty-input early return (lines 121-122) and the core glitch-removal logic (is_short & is_flip, union1d at lines 127-132). Grep confirms no direct test; only the happy-path integration test runs it. The min_duration=35 glitch filtering and flip detection are load-bearing for correct sync and have no dedicated assertions.

**Fix:** Add direct tests: (1) empty inputs return two empty arrays; (2) an opposite-type pair separated by < min_duration (a glitch) is removed while a >= min_duration pair is retained; (3) a short same-type pair (not a flip) is preserved.

### [MEDIUM] tests — validate_sequence: empty, <2-event, and non-alternating branches untested directly
`synchronize_files.py:139-181`

validate_sequence (lines 139-181) has three distinct paths: empty-input early return (lines 160-161), `len(all_events) < 2` passthrough (lines 165-166), and the non-alternating removal branch (lines 169-174). Grep confirms no direct test; it is the final correctness gate before IPI duration computation, so a wrong-index removal would corrupt sync without any test catching it.

**Fix:** Add direct tests: empty -> two empty arrays; single event passes through; two consecutive same-type events drop the second; an already-alternating sequence is returned unchanged.

### [MEDIUM] tests — attempt_sequence_match: failure path and pos/neg size-mismatch branches not tested directly
`synchronize_files.py:621-719`

attempt_sequence_match (lines 621-719) contains the threshold-sweep loop, the pos/neg size-reconciliation branches (lines 679-699), and the final `return None, None, False` failure path (line 719). The integration test drives only a clean equal-count match, so the unequal-count branches, the `neg.size > pos.size` trim (lines 679-680), and the no-match failure return are not asserted; the find_video_sync_trains median->max fallback (lines 800-810) is only tested on the success path.

**Fix:** Add a direct test calling attempt_sequence_match with (1) a signal that does not match arduino_ipi_durations within tolerance, asserting (None, None, False); (2) a signal producing unequal pos/neg counts that still matches, asserting correct ipi_start_frames. Optionally add a find_video_sync_trains case where median fails but max succeeds.

### [MEDIUM] tests — crop_wav_files_to_video undersized-cropped padding branch (m_longer) never driven
`synchronize_files.py:1266-1274`

The m_longer LSB-overwrite has three sub-branches (lines 1262-1274): equal, oversized `>` (1264-1265), and the undersized padding `else` (1266-1274). The only test, test_crop_wav_files_to_video_m_longer_oversized_cropped, uses cropped_size=40 (> the slave window), hitting only the `>` branch (confirmed line 1575). The padding `else` arithmetic, np.full value-fill, and LSB-padding concatenation are untested.

**Fix:** Add test_crop_wav_files_to_video_m_longer_undersized_cropped calling _crop_both_m_longer with _make_read_side_effect(cropped_size=10) so the padding `else` at lines 1266-1274 runs; assert the written data length equals the slave duration.

### [MEDIUM] tests — crop_wav_files_to_video undersized-cropped padding branch (s_longer) never driven
`synchronize_files.py:1296-1304`

Symmetric to the m_longer gap: the s_longer LSB-overwrite padding `else` (lines 1296-1304) is only exercised via test_crop_wav_files_to_video_s_longer_oversized_cropped with cropped_size=40 (> master window, confirmed line 1653), which takes the `>` branch (1294-1295). The undersized padding path is never executed.

**Fix:** Add test_crop_wav_files_to_video_s_longer_undersized_cropped calling _crop_both_s_longer with _make_read_side_effect(cropped_size=10) so the padding `else` at lines 1296-1304 runs.

### [MEDIUM] tests — validate_ephys_video_sync above-threshold and tracking-exceeds-boundary branches untested
`synchronize_files.py:393-400`

validate_ephys_video_sync has two failure branches with no test: (1) the above-tolerance branch (lines 393-396) that Counts sync values and emits the 'above threshold' message without writing changepoints; (2) the else (lines 398-400) taken when find_lsb_changes returns (None,None) with a large break. The existing tests cover only the zero-divergence happy path (line 375) and the missing-JSON path. A regression that wrote changepoints despite an out-of-tolerance divergence would not be caught.

**Fix:** Add two tests: (1) make the duration difference exceed npx_ms_divergence_tolerance, assert no changepoints_info_*.json is written; (2) construct a sync channel where find_lsb_changes returns (None,None) with a large break and assert the recording is skipped (which would also expose the line 344 logic bug).

### [LOW] correctness — Off-by-one in boundary guard allows out-of-bounds index into ttl_break_end_samples
`synchronize_files.py:441-442`

The guard `if (total_frame_number + largest_break_end_hop) <= ttl_break_end_samples.shape[0]:` (line 441) uses `<=`, but inside line 442 indexes `ttl_break_end_samples[largest_break_end_hop + total_frame_number]`. When `total_frame_number + largest_break_end_hop == shape[0]`, the guard passes but the access index equals shape[0], which is out of bounds (valid indices 0..shape[0]-1) and raises IndexError instead of taking the None return path. The same pattern recurs at line 881 for the NIDQ branch.

**Fix:** Change the comparison to strict `<` so the highest accessed index `largest_break_end_hop + total_frame_number` stays within bounds; apply the same fix to the NIDQ guard at line 881.

### [LOW] correctness — np.argmax / np.max on empty diff array when fewer than 2 TTL edges are detected
`synchronize_files.py:434-439`

Lines 434/437/439 compute `np.argmax(ttl_break_end_samples[1:] - ttl_break_end_samples[:-1]) + 1` and `np.max(...)`. If `ttl_break_end_samples` has 0 or 1 elements (corrupt/empty sync channel), the differenced array is empty and both raise `ValueError: attempt to get argmax/max of an empty sequence`, producing an opaque crash rather than the intended 'not found' return path.

**Fix:** Add an early check: if `ttl_break_end_samples.shape[0] < 2`, return `(None, None, 0, ttl_break_end_samples, 0)` before calling argmax/max.

### [LOW] correctness — find_ipi_intervals indexes element [0] of possibly empty edge arrays inside njit
`synchronize_files.py:479-492`

After computing `ipi_start_samples`/`ipi_end_samples` (lines 475-476), line 479 unconditionally reads `ipi_start_samples[0]` and `ipi_end_samples[0]`. If the LSB channel has no transitions of one polarity (silent/constant channel), one array is empty and the indexing raises an out-of-bounds error inside the @njit function (a hard-to-diagnose numba error). No length check precedes the [0] access.

**Fix:** Guard for `ipi_start_samples.size == 0 or ipi_end_samples.size == 0` and return empty arrays (or raise a clear error) before the `ipi_start_samples[0] < ipi_end_samples[0]` comparison.

### [LOW] correctness — cv2.VideoCapture handle leaked if an exception occurs before release()
`synchronize_files.py:527-618`

`cap = cv2.VideoCapture(video_of_interest)` is opened at line 527 and only released at line 618. Any exception in between (e.g. a missing LED key at line 538/595-599, a cvtColor failure, or the memmap creation at 592-593 failing) propagates without releasing the capture, leaking the underlying file/device handle. The function is not wrapped in try/finally.

**Fix:** Wrap the body after the `cv2.VideoCapture` call in `try: ... finally: cap.release()` so the handle is always released, including on error paths.

### [LOW] correctness — Ragged video sync sequences crash np.array stack / equality check
`synchronize_files.py:861-926`

Line 861 builds `video_sync_sequence_array = np.array(list(video_sync_sequence_dict.values()))`. If two cameras matched IPI subarrays of different lengths, modern NumPy raises a ValueError at line 861 (ragged sequence without dtype=object); if it produced an object array, the line 926 `(video_sync_sequence_array == video_sync_sequence_array[0]).all()` could raise an ambiguous-truth-value error. Either way, mismatched-length sequences across cameras are not routed to the existing 'do not match' branch (line 1026) but cause an unhandled crash. (The existing test at line 1158 only covers equal-length, differing-value sequences.)

**Fix:** Verify all matched sequences share the same length before stacking (compare lengths explicitly) and route mismatched lengths to the 'sequences do not match' message rather than constructing a ragged array.

### [LOW] dead_code_naming — Unused unpacked return values ttl_break_end_samples and largest_break_end_hop in validate_ephys_video_sync
`synchronize_files.py:341`

find_lsb_changes returns a 5-tuple; at line 341 validate_ephys_video_sync unpacks ttl_break_end_samples and largest_break_end_hop, but neither is referenced anywhere in the rest of the method (verified by grep: only definitions appear, no later use in lines 342-401). They are dead bindings at this call site (they ARE used at the separate crop_wav_files_to_video call site, lines 1097-1104).

**Fix:** Replace the two trailing names with throwaway underscores: `(tracking_start, tracking_end, largest_break_duration, _, _) = self.find_lsb_changes(...)`.

### [LOW] dead_code_naming — Unused loop index npx_idx in validate_ephys_video_sync
`synchronize_files.py:306`

The loop `for npx_idx, npx_recording in enumerate(sorted(...))` binds npx_idx, but grep shows it is never used anywhere in the file. The enumerate() is therefore unnecessary.

**Fix:** Drop the enumerate and iterate directly: `for npx_recording in sorted(pathlib.Path(self.root_directory).rglob(...)):`.

### [LOW] docs_clarity — Broken grammar in gather_px_information docstring
`synchronize_files.py:504-505`

Line 504 reads 'This method takes find sync LEDs in video frames, and gathers information about their intensity changes over time.' The phrase 'takes find' is grammatically broken (leftover from the 'This method takes ...' template used by sibling methods).

**Fix:** Reword to e.g. 'This method finds the sync LEDs in video frames and gathers information about their intensity changes over time.'

### [LOW] docs_clarity — find_lsb_changes Returns mislabels largest_break_end_hop as a sample position
`synchronize_files.py:425-428`

The Returns block (line 428) describes largest_break_end_hop as 'sample position of the largest break'. In the body it is an INDEX (hop number) into ttl_break_end_samples obtained via `np.argmax(...) + 1` (line 434/437), not a sample position; the actual sample position would be `ttl_break_end_samples[largest_break_end_hop]`. The description is incorrect and could mislead callers who unpack this value.

**Fix:** Reword to clarify it is the index ('hop') into ttl_break_end_samples marking the end of the largest break (the recording-start hop), not a sample position.

### [LOW] docs_clarity — Misleading inline comment in find_ipi_intervals
`synchronize_files.py:474`

The comment at line 474 'get switches from ON to OFF and vice versa (both look at the 0 value positions)' is inaccurate. The two following lines compute `np.diff(lsb_array) < 0` (falling edge -> ipi_start_samples) and `np.diff(lsb_array) > 0` (rising edge -> ipi_end_samples); they detect opposite transitions, not 'both ... the 0 value positions'.

**Fix:** Reword to describe the two transition directions explicitly, e.g. '# falling edges (1->0) mark IPI starts; rising edges (0->1) mark IPI ends'.

### [LOW] docs_clarity — find_video_sync_trains Returns mislabels frames and uses vague 'as per user definition'
`synchronize_files.py:741-742`

The Returns block (lines 741-742) calls the array 'the OFF-event start frames (as per user definition)'. The body derives these as the frame after the positive (ON) edge (`temp_ipi_start_frames = pos_significant_events + 1`, lines 685/716), so 'OFF-event start frames' is inaccurate, and 'as per user definition' adds nothing.

**Fix:** Reword to state precisely what the frames are, e.g. 'the start frames of each detected IPI (the frame following each ON edge)', and drop 'as per user definition'.

### [LOW] docs_clarity — Magic expression for brightest-frame search window lacks comment
`synchronize_files.py:533`

`max_frame_num = int(round(sync_camera_fps + (sync_camera_fps / 2)))` (line 533) computes the number of frames to scan (~1.5 s of footage) when searching for the brightest LED frame, with no comment explaining the 1.5x factor.

**Fix:** Add a brief comment, e.g. '# scan the first ~1.5 s of frames (1.5x fps) to locate the brightest LED frame'.

### [LOW] docs_clarity — Class-level docstring is a stray comment with a stale 'dictionary below' reference
`synchronize_files.py:186-190`

The Synchronizer class docstring (lines 186-190) only describes the LED pixel coordinate dictionary ('In the dictionary below, you can find px values ...'). It does not describe the class's purpose, and the 'dictionary below' it references is now built in the separate _build_led_px_dict static method (lines 192-228) rather than literally appearing below the docstring, so the reference is stale.

**Fix:** Replace with a proper class-level summary of the Synchronizer's role and move the LED-px note to a comment on _build_led_px_dict where the dictionary now lives.

### [LOW] performance — Repeated filesystem glob for changepoints_info JSON computed three times
`synchronize_files.py:358-391`

Within validate_ephys_video_sync, `sorted(pathlib.Path(root_ephys).glob('changepoints_info_*.json'))` is evaluated three times: the existence check (line 358), the file open (line 359), and the success message (line 391). Each re-walks and re-sorts the directory.

**Fix:** Compute `existing_changepoint_files = sorted(pathlib.Path(root_ephys).glob('changepoints_info_*.json'))` once, then reuse it at lines 358, 359, and 391.

### [LOW] tests — _combine_and_sort_events helper has no test for sorting/typing
`synchronize_files.py:68-94`

_combine_and_sort_events (lines 68-94) stamps type 1 on pos events and -1 on neg events and sorts by frame index. It is private but underpins both filter_events_by_duration and validate_sequence, whose downstream indexing assumes the (N,2) [frame, type] layout.

**Fix:** Add a direct test passing interleaved pos/neg frame indices and asserting the returned array is sorted by column 0 with column 1 values +1 for pos and -1 for neg.

### [LOW] tests — gather_px_information end-of-stream warning and grayscale branches untested
`synchronize_files.py:606-608`

gather_px_information has an early-break branch (lines 606-608) for ret=False before total_frame_number frames, and the grayscale (frame.ndim != 3) path at lines 614-616. test_gather_px_information_writes_led_memmap (line 466) writes 15 color frames and reads 10, so neither the short-read warning nor the single-channel memmap path is exercised.

**Fix:** Add a test writing fewer frames than total_frame_number (request more) asserting no error and zero rows past the decodable count; optionally a grayscale source to cover lines 614-616.


## `modify_files.py` (24)

### [HIGH] tests — split_clusters_to_sessions is entirely untested
`modify_files.py:84-216`

Operator.split_clusters_to_sessions has no test in tests/processing/test_modify_files.py (which only covers concatenate_audio_files, concatenate_binary_files, rectify_video_fps). Untested branches include the phy_curation_bool False path (177-179), the tracking_start_end-NaN fallback (161-164), the good/mua save path with min_spike_num gate (201-205), the noise/unsorted accumulators (207-211), and the frame clamp at line 196.

**Fix:** Add a test building a synthetic ephys mirror with a kilosort dir (spike_clusters.npy, spike_times.npy, cluster_info.tsv), a changepoints_info_*.json, and a video/*_camera_frame_count_dict.json; mock smart_wait. Assert: a good/mua cluster above min_spike_num writes a (2, n_spikes) .npy; a below-threshold cluster writes nothing; noise/unsorted increment counts; absent cluster_info.tsv emits the 'Phy2 curation has not been done' message and saves nothing; a NaN tracking_start_end falls back to session_start_end.

### [MEDIUM] correctness — Frame-index clamp only catches `== frame_least`, not `> frame_least`
`modify_files.py:196`

At line 195-196, `session_spikes_fps = np.round(session_spikes_sec * esr_dict[session_key])` can yield values strictly greater than `frame_least_dict[session_key]` because the spike-window second value (derived from se_dict bounds and the calibrated headstage SR) and the empirical camera SR `esr_dict` are independent quantities; rounding the last spikes can land at or above frame_least. The guard `session_spikes_fps[session_spikes_fps == frame_least_dict[session_key]] = frame_least_dict[session_key]-1` only remaps the exact-equality case, leaving any value > frame_least unclamped, so a downstream consumer using row 1 of session_spikes as a frame index into an array of length frame_least could index out of bounds.

**Fix:** Clamp with `>=`: `session_spikes_fps[session_spikes_fps >= frame_least_dict[session_key]] = frame_least_dict[session_key] - 1`.

### [MEDIUM] correctness — `delete_old_file` guard tests a different file than it unlinks for calibration directories
`modify_files.py:886-887`

Line 887 unlinks `current_working_dir / target_file`. For calibration sub-directories, `current_working_dir = sub_directory` (line 877) and `target_file = f"000000.{vid_ext}"` (line 878). But the guard on line 886 checks `(current_working_dir / f"{conv_target}_{cam_serial}.{vid_ext}").is_file()`, i.e. it tests for `<conv_target>_<serial>.<ext>` inside the calibration sub-directory, which never exists there. So the guard is always False for calibration dirs and the intended `000000.<ext>` cleanup is silently skipped. For non-calibration dirs `current_working_dir = video_dir`, guard file == unlink target, so it works only there.

**Fix:** Make the existence check use the file actually deleted: `if (current_working_dir / target_file).is_file(): (current_working_dir / target_file).unlink()`.

### [MEDIUM] performance — Full re-scan of spike_clusters once per cluster (O(n_clusters x n_spikes))
`modify_files.py:187`

Inside the per-cluster loop, line 187 runs `np.where(spike_clusters == cluster_info[idx, 'cluster_id'])` which scans the entire spike_clusters array (routinely tens of millions of spikes) once per good/mua cluster. With hundreds of clusters this is O(n_clusters x n_spikes), the dominant runtime cost of split_clusters_to_sessions on real Neuropixels datasets.

**Fix:** Group spikes by cluster id in a single pass before the loop (e.g. argsort + searchsorted boundaries, or a dict cluster_id -> spike sample times) so each cluster's events come from a slice instead of a full re-scan, reducing total cost to roughly O(n_spikes log n_spikes) once.

### [MEDIUM] tests — multichannel_to_channel_audio is entirely untested
`modify_files.py:403-495`

multichannel_to_channel_audio has no test. The FileNotFoundError at lines 462-466 guards a documented naming bug yet is unasserted; the happy path (channel separation, name_origin = master stem[2:] at 467, master/slave concatenation, shutil.rmtree of audio/temp at 495) is unverified.

**Fix:** Add an error-path test (no 'm_*.wav' present -> pytest.raises(FileNotFoundError)) and a happy-path test (place 'm_<datetime>.wav', mock Popen to create expected temp/output files, mock wait_for_subprocesses, assert name_origin from stem[2:] and that audio/temp is removed).

### [MEDIUM] tests — hpss_audio is entirely untested
`modify_files.py:497-559`

hpss_audio has no test. The STFT/HPSS/iSTFT pipeline reads params from input_parameter_dict['hpss_audio'] and the int16 clipping at lines 549-551 is correctness-sensitive (dropping .astype('int16') or mis-clipping silently corrupts saved audio). The empty-input case (no .wav files) is also untested.

**Fix:** Add a test writing a short synthetic mono int16 WAV into audio/cropped_to_video, mock smart_wait, run hpss_audio, and assert audio/hpss/<stem>_hpss.wav exists with dtype int16, the same sampling rate, and length == input length. Add an empty-directory case asserting no output and no crash.

### [MEDIUM] tests — filter_audio_files is entirely untested
`modify_files.py:561-620`

filter_audio_files has no test. Untested: the freq ordering at lines 587-588 feeding the sox 'sinc {freq_hp}-{freq_lp}' argument at line 604 (note freq_hp before freq_lp), the per-directory loop (594), and the len(all_audio_files) > 0 gate (602).

**Fix:** Add a test setting filter_dirs to one dir and filter_audio_format='wav', writing tiny WAVs, mocking smart_wait and patching subprocess.Popen (capturing args) and wait_for_subprocesses; assert Popen args contain 'sinc' then exactly f'{freq_hp}-{freq_lp}', that audio/<dir>_filtered is created, and outputs are '<stem>_filtered.wav'. Add an empty-input case asserting no Popen call.

### [MEDIUM] tests — concatenate_video_files is entirely untested
`modify_files.py:670-736`

concatenate_video_files has no test. Untested: the calibration/serial filter (690-691, 727-728), the len(all_video_files) > 1 gate (699), the concat-list .txt writing (702-704), and the post-concat cleanup loop (726-736) that unlinks the list file and shutil.move's the output into video/.

**Fix:** Add a test creating video/<date>.<serial> dirs with 2+ matching files plus a calibration dir, setting the relevant settings, mocking smart_wait and wait_for_subprocesses, and patching Popen to create the expected output. Assert the concat list has one 'file ...' line per input, ffmpeg runs only for the >1-file dir (not calibration), the list .txt is unlinked, and the output lands in video/. Add a single-video case asserting no ffmpeg call and no move.

### [LOW] correctness — Parallel `sorted(glob(...bin*))`/`sorted(glob(...meta*))` zip can pair mismatched files
`modify_files.py:274-275`

Line 274-275 zips two independently-globbed/sorted lists with trailing-`*` patterns (`*{npx_file_type}.bin*` / `*{npx_file_type}.meta*`). The trailing `*` matches more than the canonical `.bin`/`.meta` (e.g. `.bin.tmp`, sidecars), and any count asymmetry or differing lexical order pairs a `.bin` with the wrong `.meta`, attributing channel count/headstage SN to the wrong binary; zip also silently truncates to the shorter list.

**Fix:** Iterate the `.bin` files and derive each meta path deterministically (replace `.bin` with `.meta`, assert it exists) instead of zipping two globs; tighten the glob to exact suffixes.

### [LOW] correctness — Per-file metadata variables can be stale or undefined if a meta key is missing
`modify_files.py:279-302`

`total_num_channels`, `headstage_sn`, `imec_probe_sn`, and `spike_glx_sr` are assigned only inside the conditional branches of the meta-parse loop (lines 282-292) and read after the loop (lines 300-302, 306-309). If a `.meta` file is missing one of these keys, the value is either undefined on the first file (NameError) or silently carried over from the previously parsed file, producing a silently-wrong `file_duration_samples`.

**Fix:** Initialize these to None before the per-line loop and validate they were set after parsing each meta, raising a clear error if any required key was absent.

### [LOW] correctness — `np.memmap` opened per binary file is never released
`modify_files.py:304`

Line 304 creates `one_recording = np.memmap(filename=one_file, mode='r', dtype='int16', order='C')` for every binary file across every root directory, but only `one_recording.shape[0]` is used (lines 306-319). The mmap/file handle is held until GC, accumulating open mmaps when concatenating many sessions/probes.

**Fix:** Read the size cheaply via `one_file.stat().st_size // np.dtype('int16').itemsize`, or `del one_recording` after extracting shape[0].

### [LOW] correctness — `name_origin` via `split('_')[1]` breaks if the name token contains underscores
`modify_files.py:650`

Line 650 `name_origin = list(data_dict.keys())[0].split('_')[1]` takes the second underscore-separated token of the first concatenated filename. This is the same positional-split fragility the maintainer already fixed in `multichannel_to_channel_audio` (see the explanatory comment at lines 451-467). If `name_origin` itself contains an underscore, only its first sub-token is captured and the memmap filename is silently wrong.

**Fix:** Derive `name_origin` by stripping the known device prefix and the trailing `_chNN.<ext>` suffix (mirroring the stem-based approach in multichannel_to_channel_audio) rather than positional `split('_')[1]`.

### [LOW] dead_code_naming — Unused enumerate index `ord_idx` in concatenate_binary_files (save-dir loop)
`modify_files.py:254`

Line 254 `for ord_idx, one_root_dir in enumerate(self.root_directory):` enumerates an index never referenced in the loop body. Confirmed by ruff B007.

**Fix:** Use `for one_root_dir in self.root_directory:`.

### [LOW] dead_code_naming — Unused enumerate index `ord_idx` in concatenate_binary_files (per-probe loop)
`modify_files.py:271`

Line 271 `for ord_idx, one_root_dir in enumerate(self.root_directory):` enumerates an index never used in the loop body. Confirmed by ruff B007.

**Fix:** Use `for one_root_dir in self.root_directory:`.

### [LOW] dead_code_naming — Unused enumerate index `sd_idx` in rectify_video_fps (encode loop)
`modify_files.py:789`

Line 789 `for sd_idx, sub_directory in enumerate(sorted(video_dir.iterdir())):` enumerates an index never referenced. Confirmed by ruff B007.

**Fix:** Use `for sub_directory in sorted(video_dir.iterdir()):`.

### [LOW] dead_code_naming — Unused enumerate index `sd_idx` in rectify_video_fps (move loop)
`modify_files.py:858`

Line 858 `for sd_idx, sub_directory in enumerate(sorted(video_dir.iterdir())):` enumerates an index never referenced. Confirmed by ruff B007.

**Fix:** Use `for sub_directory in sorted(video_dir.iterdir()):`.

### [LOW] dead_code_naming — Unpacked `D_percussive` is never used in hpss_audio
`modify_files.py:535`

Line 535 `D_harmonic, D_percussive = librosa.decompose.hpss(...)` unpacks both components, but only `D_harmonic` is used (line 542). Tuple-unpacking targets are not flagged by F841, but D_percussive is genuinely unused.

**Fix:** Replace with `D_harmonic, _ = librosa.decompose.hpss(...)`.

### [LOW] docs_clarity — Stray semicolon in split_clusters_to_sessions Returns docstring
`modify_files.py:104`

Line 104 reads `Arrays that contain spike times: seconds (row 0) and frames (row 1).;` with a dangling `;` after the period.

**Fix:** Remove the trailing semicolon (or restructure so the sentence flows into 'saved as .npy files...').

### [LOW] docs_clarity — concatenate_audio_files docstring states transposed memmap shape
`modify_files.py:634`

The Returns block (line 634) says 'shape: n_channels X n_samples', but the produced memmap (lines 656-659) is shaped (dim_1, dim_2) where dim_1 = wav_data.shape[0] (samples, line 651) and dim_2 = number of files/channels (line 652), and the fill loop writes audio_mm_arr[:, file_idx] = wav_data (line 662). The actual layout is (n_samples X n_channels), the transpose of the documented shape.

**Fix:** Correct the documented shape to '(n_samples X n_channels)'.

### [LOW] docs_clarity — multichannel_to_channel_audio docstring typo 'where' should be 'were'
`modify_files.py:408`

Line 408-409 reads '...concatenates single channel files via Sox, since multichannel files where split due to a size limitation.' 'where' should be 'were'.

**Fix:** Fix 'where' -> 'were'. (Per maintainer preference, do not trim verbose docs; only correct the typo and optionally clarify ordering.)

### [LOW] performance — Redundant np.sort on indices already returned in ascending order
`modify_files.py:187`

`np.where(condition)[0]` on a 1-D boolean mask returns indices in strictly increasing order, so wrapping it in `np.sort(...)` re-sorts an already-sorted array each cluster iteration. `np.take` on line 188 does not require sorted indices either.

**Fix:** Drop the sort: `cluster_indices = np.where(spike_clusters == cluster_info[idx, 'cluster_id'])[0]` (or eliminate entirely if the grouping refactor is applied).

### [LOW] performance — list(data_dict.keys())[0] rebuilt three times
`modify_files.py:650-653`

Lines 650, 651, and 653 each call `list(data_dict.keys())[0]`, materializing the full keys list three times to fetch the same first key.

**Fix:** Compute once: `first_key = next(iter(data_dict.keys()))` and reuse on lines 650/651/653.

### [LOW] tests — rectify_video_fps dropped-frame / None-fallback / all-NaN-SR branches untested
`modify_files.py:896-913`

The single rectify_video_fps test exercises only the clean no-dropped-frames path. Uncovered: the dropped-frame WARNING (819-821) where the camera is not counted; the total_frame_number-is-None fallback (896-901) writing the int(1e9)/1e9 sentinels; and the all-NaN empirical_camera_sr branch (910-913) writing median as float('nan').

**Fix:** Add a test with a dropped-frame camera asserting the WARNING and sentinel values in the written JSON, and a test where encode_camera_serial_num never matches a sub-directory so empirical_camera_sr stays all-NaN, asserting median_empirical_camera_sr is written as NaN with the WARNING.

### [LOW] tests — rectify_video_fps conduct_concat=True path and metadata=None path untested
`modify_files.py:763-772`

The lone rectify_video_fps test always calls conduct_concat=False, so the no-concat copy block (763-772) is taken; the default conduct_concat=True path (skipping that block) is never tested. The metadata-is-None branch (line 816 guard, no session_duration write) is also uncovered.

**Fix:** Add a conduct_concat=True test that pre-populates video/ with the conv_target_<serial>.<ext> files and asserts the per-camera copy did not run, plus a test patching load_session_metadata to return (None, path) asserting save_session_metadata is never called while the JSON is still written.


## `anipose_operations.py` (23)

### [MEDIUM] correctness — Session root resolution silently defaults to cwd on no match, last-wins on multiple match
`anipose_operations.py:342-347`

In ConvertTo3D.__init__, session_root_joint_date_dir defaults to pathlib.Path() ('.') and session_root_name to ''. The loop over (root/'video').iterdir() filtering `is_dir() and '_' not in name` neither sorts nor breaks. If NO subdir matches, both stay at the empty/cwd defaults so downstream paths become e.g. './<name>_points3d.h5' and writes/reads silently land in cwd instead of failing. If MORE THAN ONE matches, the LAST one returned by the unsorted iterdir() wins (filesystem-dependent, non-deterministic). The rest of this file was deliberately moved to first_match_or_raise for loud-failure + determinism; this loop was not. The existing test test_convert_to_3d_init_handles_session_with_no_matching_dir even codifies the silent './' default.

**Fix:** Collect matches into a sorted list; raise a clear error on zero matches (fail loudly instead of using cwd) and break on the first sorted match (or raise on multiple), mirroring first_match_or_raise used elsewhere in this file.

### [MEDIUM] dead_code_naming — Subprocess wait label "SLEAP→.slp conversion" is backwards
`anipose_operations.py:401`

wait_for_subprocesses at line 398-406 uses label="SLEAP→.slp conversion", but the subprocess (lines 376-387) runs sleap-convert which reads an existing .slp file and writes a .analysis.h5 file (`--format analysis -o {stem}.analysis.h5`). Nothing is converted *to* .slp; the .slp is the input. The method name sleap_file_conversion and its docstring at line 355 correctly say 'SLP to H5 conversion'. This label surfaces in user-facing progress/timeout/error messages via message_output, so the arrow is actively misleading.

**Fix:** Change label to describe the real direction/target, e.g. label=".slp→.h5 conversion" or "SLEAP .slp→.analysis.h5 conversion".

### [MEDIUM] dead_code_naming — _settings['root_directory'] fallback can only ever KeyError (effectively dead)
`anipose_operations.py:338-339`

_settings is loaded only inside `if input_parameter_dict is None or root_directory is None:` (lines 331-336). Line 339's else-branch `_settings['root_directory']` is reachable when root_directory is None, but the anipose_operations block in _parameter_settings/processing_settings.json contains only the key 'ConvertTo3D' (verified: no 'root_directory' key), so `_settings['root_directory']` would raise KeyError rather than returning a usable default. The fallback is therefore dead-as-working-code. In practice all callers pass a non-None root_directory, masking it.

**Fix:** Either add a 'root_directory' key to the anipose_operations block in processing_settings.json so the default works, or drop the `else _settings['root_directory']` fallback and require root_directory to be supplied (matching every real caller).

### [MEDIUM] tests — rotate_x / rotate_y / rotate_z have no direct unit tests
`anipose_operations.py:218-302`

The three public rotation helpers are central to the translate_rotate_metric geometry but none is tested directly. The full-pipeline test only asserts output H5 shape/node count, not numerical correctness, so a sign error in any rotation matrix (sin/-sin placement at lines 240-242, 268-272, 297-299) would pass silently. These are pure deterministic functions, ideal for exact numeric assertions.

**Fix:** Add unit tests rotating a known vector (e.g. [1,0,0]) by theta=pi/2 about each axis, asserting np.allclose against the analytic result, plus theta=0 returns input unchanged.

### [MEDIUM] tests — extract_skeleton_nodes is untested in isolation
`anipose_operations.py:128-178`

extract_skeleton_nodes is public and exercised only indirectly via translate_rotate_metric. Its non-trivial logic is untested directly: the sorting_key_list mapping py/id 1/2 -> index-1 else index-2 (lines 164-171), the first-vs-subsequent links branch (157-162), and the skeleton_arena_bool=True branch (173-176) that prefixes nodes at idx>=4 with 'ch_' while leaving the first four bare. A regression there would surface only as a downstream KeyError, not a clear failure.

**Fix:** Add a unit test writing a small synthetic skeleton JSON (links + nodes with py/id values) to tmp_path, asserting sorted node order; add a skeleton_arena_bool=True case asserting indices 0-3 stay bare and >=4 gain 'ch_'.

### [MEDIUM] tests — find_mouse_names modern flat cage/subject branch untested
`anipose_operations.py:63-73`

Both find_mouse_names tests cover only the legacy suffixed-key branch (and one incidental metadata path). The modern flat-key branch (lines 63-73), reading user_meta_data['cage'] and ['subject'] as comma lists and appending 'cage_subject' or bare 'cage' when subject is empty (lines 69-72), is never exercised — and it is the current primary code path.

**Fix:** Add a test mocking new_for_filename to return a store whose user_metadata has cage='c1,c2', subject='m1,' and assert find_mouse_names returns ['c1_m1','c2'] (covering the joined and empty-subject bare-cage cases).

### [LOW] correctness — zip(strict=False) over cage/subject can silently drop an animal on ragged input
`anipose_operations.py:63-73`

In the modern flat cage/subject branch, zip is called with strict=False over cage.split(',') and subject.split(','). On a length mismatch (e.g. cage='c1,c2' but subject='m1') strict=False silently truncates to the shorter list, dropping a genuine track name — contradicting the deliberately careful m1/m2 fallthrough below (100-119) that goes out of its way never to drop a present mouse. A dropped name changes the shape of the saved track_names dataset relative to tracked animals.

**Fix:** Use strict=True so a length mismatch raises loudly, or handle the ragged case explicitly the way the m1/m2 block does.

### [LOW] correctness — Legacy metadata branch reads a non-deterministic video subdirectory
`anipose_operations.py:52-120`

When metadata is None, the loop iterates (root/'video').iterdir() (unsorted) and breaks on the first subdir passing the '.'/'_'/not-calibration filter (lines 53-59). Because iterdir() order is filesystem-dependent, which imgstore metadata.yaml is read is non-deterministic; if camera subdirs carried inconsistent user_metadata the resolved track_names could differ across runs. The rest of the file was changed to sort (first_match_or_raise) for determinism; this loop was not.

**Fix:** Sort the iterdir() result (e.g. sorted(..., key=lambda p: p.name)) before selecting the first qualifying subdirectory.

### [LOW] docs_clarity — Broken wording in find_mouse_names root_directory param doc
`anipose_operations.py:41`

The Parameters entry at line 41 reads 'The directory where of the session.' — 'where of' is ungrammatical; it means the session's root directory.

**Fix:** Reword to 'The root directory of the session.'

### [LOW] docs_clarity — Missing space after comma in find_mouse_names docstring
`anipose_operations.py:36`

Line 36 reads 'NB: from v0.8.12 onwards,it uses the usv_playpen native metadata files.' — no space after the comma ('onwards,it').

**Fix:** Insert a space: 'NB: from v0.8.12 onwards, it uses ...'.

### [LOW] docs_clarity — Subject-verb agreement error in find_mouse_names docstring
`anipose_operations.py:33-34`

Lines 33-34 read 'the function\n    check for either the old or the new version of the metadata.yaml file.' — 'the function check' should be 'the function checks'.

**Fix:** Reword to 'the function checks for either the old or the new version of the metadata.yaml file.'

### [LOW] docs_clarity — Hard-coded -45 deg Z rotation lacks an explanatory comment
`anipose_operations.py:719`

After the geometry-derived X/Y/Z rotations, line 719 applies z_theta_extra = -math.pi/4 to both arena (line 720) and mouse data (line 861). Unlike the earlier rotations whose angles come from corner geometry, this is a fixed -45 deg turn with no comment explaining why (it brings the diagonally-oriented North/East/South/West corners onto the X/Y axes so the corner-correction block at 722-810 can apply per-corner signed offsets). The repo convention favors an explanatory comment for non-obvious constants like this.

**Fix:** Add a brief comment above line 719 explaining the fixed -pi/4 rotation aligns the diagonal corner layout to the X/Y axes for the subsequent corner-edge correction.

### [LOW] docs_clarity — Corner-correction +/-0.025 offsets are undocumented magic numbers
`anipose_operations.py:722`

The section comment at line 722 motivates the block, but the repeated +0.025 / -0.025 offsets per corner (lines 761, 766, 775, 780, 789, 794, 803, 808) are unexplained constants with no statement that they are a metric (meters) rail offset shifting each corner from the inner edge to the outer rail edge. A reader cannot tell what 0.025 represents or its units.

**Fix:** Add a comment or named constant noting 0.025 is the rail offset in meters that moves each corner from inner edge to outer rail edge.

### [LOW] docs_clarity — translate_rotate_metric session_idx kwarg is undocumented
`anipose_operations.py:605-606`

The signature is `def translate_rotate_metric(self, **kwargs)` and the body reads kwargs['session_idx'] (lines 619-623), the only accepted keyword; it indexes experimental_codes and is bounds-checked at 880-884. The Parameters section of the docstring (lines 604-606) is empty, so the accepted keyword and its effect/default are undocumented.

**Fix:** Document the keyword, e.g. 'session_idx (int): index into experimental_codes selecting this session's experiment code; defaults to 0.'

### [LOW] performance — Redundant array .copy() before each rotate_* call
`anipose_operations.py:692-720`

Lines 692, 698, 711, 718 (arena) and 851, 854, 857, 860 (mouse) do `*_temp = data.copy()` then pass the copy into rotate_z/y/x. The rotate_* helpers (218-302) only call np.matmul, which never mutates its input and always allocates a fresh output, so each .copy() is a throwaway allocation. On the mouse branch the copied array is full-session-length (N_FRAMES x N_ANIMALS x N_NODES x 3), making these real extra allocations on the larger code path.

**Fix:** Pass the array directly: arena_data = rotate_z(arena_data, z_theta), etc., and drop the *_temp intermediates (and the four mouse_data copies at 851-861).

### [LOW] performance — Repeated arena_nodes.index() rescans recompute indices already in node_list_indices
`anipose_operations.py:723-810`

The four corner indices are computed once into node_list_indices = [North, East, South, West] at lines 654-659, but from line 723 they are recomputed via arena_nodes.index(...) repeatedly: 8 calls in the arena_center_out_distance block (731-734 and 745-748, the same four nodes listed twice for nanmin and nanmax over the identical slice) plus 3 in each of the four corner-assignment blocks (756-810) — ~20 redundant O(n) list scans. The nanmin/nanmax pair also builds the identical fancy-index slice twice. This is the single-frame arena path so absolute cost is tiny, but it is pure redundant recomputation.

**Fix:** Reuse cached indices (north_i,east_i,south_i,west_i = node_list_indices); slice the corner block once (corner_xy = arena_data[0,0,[north_i,west_i,south_i,east_i],:2]) and reuse it for both nanmin and nanmax; replace each arena_nodes.index('East') etc. in the assignment blocks with the cached int.

### [LOW] tests — translate_rotate_metric metadata=None and missing-'Session' paths untested
`anipose_operations.py:892-897`

In the animal branch, two cases are uncovered: (1) load_session_metadata returning metadata=None (line 892) skips the metadata update and forces find_mouse_names down the legacy imgstore path; (2) metadata present but missing 'Session' must raise the explicit KeyError at lines 894-897. Both animal-branch tests supply metadata containing 'Session', so neither is exercised.

**Fix:** Add an animal-branch test with load_session_metadata mocked to return (None, path) asserting the H5 is still written; add another returning metadata without 'Session' asserting pytest.raises(KeyError).

### [LOW] tests — delete_original_h5=False retention not asserted
`anipose_operations.py:924-927`

test_translate_rotate_metric_animal_branch uses delete=True and asserts the original H5 is removed. The complementary delete_original_h5=False branch (unlink at line 927 skipped) is set up in the session_idx-out-of-range test but that test raises before reaching the write/delete, so no test verifies the original <session>_points3d.h5 is RETAINED when delete_original_h5=False.

**Fix:** Add an animal-branch test with delete=False asserting the transformed H5 is written AND the original _points3d.h5 still exists.

### [LOW] tests — __init__ JSON-fallback path untested
`anipose_operations.py:331-339`

Every ConvertTo3D test passes both root_directory and input_parameter_dict, so the JSON-loading branch at lines 331-336 (triggered when either is None) is never executed, and the consumption fallbacks at 338-339 are never taken. Note this also masks the dead _settings['root_directory'] fallback flagged separately.

**Fix:** Add a test constructing ConvertTo3D with input_parameter_dict=None and a valid root_directory containing video/<dir>, asserting self.input_parameter_dict equals the JSON ConvertTo3D block.

### [LOW] tests — redefine_cage_reference_nodes untested in isolation
`anipose_operations.py:181-215`

redefine_cage_reference_nodes is public and only exercised implicitly via translate_rotate_metric. Its contract (selecting [0,0,node_list_indices[i],:] for four indices and vstacking into a (4,3) array, always from frame 0 / animal 0) is never directly asserted, yet several downstream rotation-angle computations depend on the corner ordering it returns.

**Fix:** Add a unit test passing a known (1,1,N,3) array and node_list_indices, asserting the result equals np.vstack of the four selected rows and has shape (4,3).

### [LOW] tests — find_mouse_names legacy m2-with-populated-cage path untested
`anipose_operations.py:100-119`

The legacy test covers only the empty-cage_m2 fallback (line 117). The primary path where both mouse_ID_m2 and cage_ID_m2 are populated -> appends 'cage_subject' (lines 112-115), and the elif at 118 where only cage_ID_m2 is present (mouse blank) -> bare cage, are untested. A regression collapsing these branches would not be caught.

**Fix:** Add legacy tests: (a) cage_ID_m2='c2', mouse_ID_m2='m2' -> ['c1_m1','c2_m2']; (b) cage_ID_m2='c2', mouse_ID_m2='' -> ['c1_m1','c2'].

### [LOW] tests — find_mouse_names metadata-provided 'Subjects'-absent path untested
`anipose_operations.py:122-124`

The else branch (metadata not None) at lines 122-124 is only touched incidentally inside an animal-branch test with a single subject. There is no direct find_mouse_names test for the metadata path, and the `'Subjects' not in metadata -> []` fallback (line 123) is never asserted.

**Fix:** Add direct tests: find_mouse_names(metadata={'Subjects':[{'subject_id':'a'},{'subject_id':'b'}]}) -> ['a','b']; and find_mouse_names(metadata={}) -> [].

### [LOW] tests — translate_rotate_metric session_idx non-int fallback untested
`anipose_operations.py:619-623`

session_idx defaults to 0 when the kwarg is absent or not an int (the isinstance(...,int) guard at lines 619-623). Tests pass only int values (0, 5) or nothing; the non-int coercion is never exercised, so removing the isinstance check would go unnoticed.

**Fix:** Add an animal-branch test calling translate_rotate_metric(session_idx='not_an_int') and assert it behaves as session_idx=0 (writes experimental_code equal to exp_codes[0]).


## `preprocessing_plot.py` (20)

### [HIGH] correctness — plt.subplots collapses to a 1D array when there is exactly one device, breaking ax[row, col] indexing
`preprocessing_plot.py:315-319`

`fig, ax = plt.subplots(nrows=2, ncols=len(plot_statistics_dict.keys()), figsize=(12.8, 9.6))`. When `len(plot_statistics_dict)` is 1, matplotlib returns `ax` as a 1D array of shape (2,) rather than (2, 1). Every subsequent `ax[0, device_num]` / `ax[1, device_num]` access then raises `IndexError: too many indices for array`. Confirmed empirically: subplots(nrows=2, ncols=1) yields shape (2,) and ax[0,0] raises IndexError. The method only works because all tests/callers supply >=2 devices.

**Fix:** Add `squeeze=False` to the `plt.subplots(...)` call so `ax` is always a 2D (2, ncols) array, making `ax[0, device_num]` / `ax[1, device_num]` valid even for a single device.

### [MEDIUM] correctness — SEM divides nanstd by total array size (including NaNs) instead of the non-NaN count
`preprocessing_plot.py:106-108`

`error_sem = np.nanstd(...) / np.sqrt(ipi_discrepancy_ms.size)`. `nanstd` excludes NaNs from the spread, but `.size` counts ALL elements including NaNs, so when any NaNs are present the denominator N is larger than the N used for the std, understating the SEM and narrowing the 99% CIs (lines 110-115). Confirmed: for a 3-element array with one NaN, .size=3 but non-NaN count=2.

**Fix:** Use the NaN-aware count for the denominator, e.g. `n = np.count_nonzero(~np.isnan(ipi_discrepancy_dict[device_id]['ipi_discrepancy_ms']))` and divide `np.nanstd(...)` by `np.sqrt(n)`; guard against `n == 0`.

### [MEDIUM] dead_code_naming — Misleading getattr: viz_settings reads 'visualizations_parameter_dict', an attribute never set, so it is always None
`preprocessing_plot.py:834`

Line 834 passes `viz_settings=getattr(self, 'visualizations_parameter_dict', None)`. `__init__` never sets `self.visualizations_parameter_dict` (it stores the incoming dict as `self.input_parameter_dict`), and SummaryPlotter is only constructed inline in preprocess_data.py (lines 238, 571) and discarded, so the attribute is never set externally either. The getattr therefore always returns None.

**Fix:** Either set `self.visualizations_parameter_dict` in `__init__` so the lookup resolves to a real value, or replace the getattr with an explicit `viz_settings=None` to remove the misleading attribute lookup.

### [MEDIUM] docs_clarity — Incorrect unit label 'kHz' for audio sampling rate (value is in Hz)
`preprocessing_plot.py:598`

`audio_sampling_rate = example_audio_file.getframerate()` (line 145) returns frames/second (Hz). Line 598 renders this raw value as `{audio_sampling_rate} kHz`, so for a 250000 Hz recording the figure text reads '250000 kHz' — off by a factor of 1000.

**Fix:** Either label the raw value as 'Hz', or divide by 1000 when formatting, e.g. `{audio_sampling_rate / 1000:g} kHz`.

### [MEDIUM] tests — Default-config __init__ branch is untested and would KeyError on root_directory
`preprocessing_plot.py:46-54`

All three tests construct SummaryPlotter with explicit input_parameter_dict and root_directory, so the default-load branch (lines 46-53) is never exercised. Confirmed the JSON block processing_settings.json['preprocessing_plot'] contains only a 'SummaryPlotter' key and NO 'root_directory' key, so `SummaryPlotter(root_directory=None)` would raise KeyError at line 52 (`_settings['root_directory']`). This latent defect is invisible to the suite.

**Fix:** Add a test constructing SummaryPlotter() / SummaryPlotter(root_directory=None) and assert the resulting attributes; this will surface the missing 'root_directory' key (KeyError) and pin down intended default behavior.

### [LOW] correctness — All-NaN or empty discrepancy array makes most_extreme_value raise on int(np.round(nan))
`preprocessing_plot.py:291-297`

`most_extreme_value = int(np.round(np.nanmax(np.abs(ipi_discrepancy_ms))))`. If a device's `ipi_discrepancy_ms` is entirely NaN, `np.nanmax` returns NaN and `int(np.round(nan))` raises `ValueError: cannot convert float NaN to integer`; an empty array makes `np.nanmax` raise ValueError. The median/mean/SEM computed earlier (lines 94-108) would likewise be NaN. There is no guard for the empty/all-NaN device case.

**Fix:** Before computing `most_extreme_value`, skip or special-case devices whose `ipi_discrepancy_ms` is empty or entirely NaN (fall back to a default extreme value and omit the median/mean axvlines).

### [LOW] correctness — Phidget channels are NaN-filtered by mutating the caller-supplied dictionary in place
`preprocessing_plot.py:738-740`

In the `s_` branch the code reassigns `phidget_data_dictionary['humidity'|'lux'|'temperature']` to its non-NaN subset (lines 738-740, 768-770, 798-800), mutating the dict passed in by the caller. This is an unexpected side effect on an input argument. (The empty-after-filter `.min()`/`.max()` is already swallowed by the surrounding try/except, and the global stats on lines 118-134 are computed BEFORE this block and never recomputed, so the candidate's 'order-dependence corruption' claim does not hold.)

**Fix:** Filter NaNs into local variables instead of mutating the caller's dict, e.g. `hum_clean = phidget_data_dictionary['humidity'][~np.isnan(phidget_data_dictionary['humidity'])]`, and use the locals for the inset plots.

### [LOW] dead_code_naming — Dead variable: cam_durations list is built but never read
`preprocessing_plot.py:159, 194`

`cam_durations` is initialized at line 159 and appended to at line 194 but never read anywhere in src/ or tests/ (grep confirms only the init and append). `video_duration` itself is still used at line 196 for cam_esr, so only the accumulation is dead.

**Fix:** Remove line 159 (`cam_durations = []`) and line 194 (`cam_durations.append(video_duration)`); keep the `video_duration` local at line 193.

### [LOW] dead_code_naming — self.input_parameter_dict is set but never read
`preprocessing_plot.py:53`

`self.input_parameter_dict` is assigned at line 53 but never read anywhere in this file (grep shows only the assignment). Callers in preprocess_data.py do pass `input_parameter_dict`, so the constructor parameter currently has no functional effect on the produced figure.

**Fix:** If genuinely unused, drop `self.input_parameter_dict` and the parameter (updating the two call sites), or wire it through to viz settings if that was the intent (see the visualizations_parameter_dict finding).

### [LOW] docs_clarity — Stale Returns docstring: method returns None, not a figure
`preprocessing_plot.py:77-78`

The `preprocessing_summary` Returns section documents `preprocessing_plot (fig)`, but the method has no return statement (it saves the figure on line 831 and calls `plt.close()` on line 838) and is annotated `-> None` (line 58). The documented return value contradicts both the code and the type hint.

**Fix:** Change the Returns section to document `None`, and describe the on-disk figure (written to `<root>/sync/`) in the Description section.

### [LOW] docs_clarity — Typo in docstring: 'variables measure with the phidget device'
`preprocessing_plot.py:64`

The docstring reads 'variables measure with the phidget device (humidity, illumination, temperature)' (line 64); 'measure' should be 'measured'.

**Fix:** Reword to 'variables measured with the phidget device (humidity, illumination, temperature)'.

### [LOW] docs_clarity — __init__ docstring missing the message_output parameter
`preprocessing_plot.py:36-39`

The `__init__` signature (line 27) accepts `input_parameter_dict`, `root_directory`, and `message_output`, but the Parameters section documents only the first two; `message_output (Callable | None)` (logging callback defaulting to print, line 54) is undocumented. The repo convention is full Parameters docstrings.

**Fix:** Add a Parameters entry for `message_output (Callable | None)` describing the logging/message callback, defaulting to None (falls back to built-in print).

### [LOW] docs_clarity — ipi_discrepancy_dict docstring omits the consumed NIDQ keys
`preprocessing_plot.py:70-71`

The Parameters entry for `ipi_discrepancy_dict` (lines 70-71) describes only IPI discrepancies and video start frames, but the method also reads `nidq_ipi_discrepancy_ms` and `nidq_ipi_start_samples` (lines 407, 413-414) for the optional NIDQ inset.

**Fix:** Extend the description to list the keys accessed: `ipi_discrepancy_ms`, `video_ipi_start_frames`, and the optional `nidq_ipi_discrepancy_ms` / `nidq_ipi_start_samples`.

### [LOW] tests — message_output default-to-print branch untested
`preprocessing_plot.py:54`

All tests pass `message_output=lambda *_a, **_k: None`, so the `... if message_output is not None else print` fallback (line 54) never takes the print branch.

**Fix:** Add a test constructing SummaryPlotter(..., message_output=None) and assert plotter.message_output is print.

### [LOW] tests — _pad_two padding branch (fewer than 2 subjects) is untested
`preprocessing_plot.py:210-232`

`_pad_two` exists to guard sessions with 0 or 1 subject, but the metadata test always supplies exactly 2 subjects, so the padding-with-'-' path (line 231-232) is never exercised.

**Fix:** Add a test with metadata['Subjects'] of length 1 (and/or 0) and assert the figure still renders without IndexError.

### [LOW] tests — Empty experimenter-in-metadata branch (keeps default) untested
`preprocessing_plot.py:246-247`

When `metadata['Session']['experimenter']` is falsy, the `if experimenter_from_meta:` guard (line 246) keeps the 'Ø' default. The metadata test always supplies 'bm', so the falsy branch is never taken.

**Fix:** Add a test variant with `metadata['Session']['experimenter'] == ''` and assert the figure still renders (experimenter stays 'Ø').

### [LOW] tests — Legacy single-subject Motif user_metadata parsing branches untested
`preprocessing_plot.py:255-286`

In the metadata-absent branch, the `*_2` fields are only assigned when `len(entry) > 1` (lines 255-256, 260-261, 265-266, 270-271, 275-276, 280-281, 285-286). The no-metadata test always supplies two-element comma lists ('M,F', etc.), so the single-animal path (where animal_2/cage_2/etc. stay 'Ø') is never covered.

**Fix:** Add a no-metadata test using single-value fields (e.g. subject='M' with no comma) and assert the figure renders with the *_2 fields remaining 'Ø'.

### [LOW] tests — Video subdir filter (calibration / non-Motif dirs) skip branch untested
`preprocessing_plot.py:183-188`

The iterdir loop only processes subdirs that are directories, contain '.', contain '_', and do NOT contain 'calibration' (lines 183-188). Every test creates exactly one matching dir, so the skip-false branches (a 'calibration' dir, a stray file, a dir without '.'/'_') are never exercised.

**Fix:** Add sibling entries under video/ (e.g. a 'cam.02_calibration' dir, a plain file, a dir named 'misc') and assert only the real camera dir contributes.

### [LOW] tests — Missing camera_frame_count JSON error path untested
`preprocessing_plot.py:82-86`

`first_match_or_raise` for 'video/*_camera_frame_count_dict.json' (lines 82-86) raises when absent. There is a dedicated test for the missing-WAV FileNotFoundError but none for a session missing the camera frame-count JSON.

**Fix:** Add a test that builds a session without the *_camera_frame_count_dict.json and asserts preprocessing_summary raises (matching the 'camera frame count JSON' label).

### [LOW] tests — Empty-after-NaN-filter ValueError branch in phidget insets untested
`preprocessing_plot.py:746-755`

Each s_* inset wraps `set_xticks(...)` (whose args call `.min()/.mean()/.max()`) in try/except ValueError (lines 746-755, 776-785, 806-815) to tolerate channels that become empty after NaN filtering. No test feeds an all-NaN (or empty) phidget channel, so the except-pass path is uncovered.

**Fix:** Add a test where a phidget channel is entirely NaN and assert preprocessing_summary still completes and writes the figure.


## `assign_vocalizations.py` (16)

### [MEDIUM] correctness — ZeroDivisionError when assignments array is empty (zero vocalizations)
`assign_vocalizations.py:389-393`

total = len(assignments) (line 385); lines 389, 390, 393 compute round(assigned*100/total, 2), round(unassigned*100/total, 2) and round(count*100/total, 2). If the model_predictions.npz assignments array is empty, total == 0 and these raise ZeroDivisionError, crashing run_vocalocator_ssl after the expensive subprocesses already ran.

**Fix:** Guard the percentage reporting with a total == 0 short-circuit (early-return with a 'no vocalizations to assign' message), or use a safe denominator like max(total, 1) for the display arithmetic only.

### [MEDIUM] correctness — Log messages say 'percentage' but report raw counts
`assign_vocalizations.py:257-261`

All five message_output calls at lines 257-261 read 'Vocalization percentage attributed to ...' but interpolate raw boolean-mask sums (none_in_set.sum(), one_in_set.sum(), mouse_one_vocalizations.sum(), mouse_two_vocalizations.sum(), two_in_set.sum()), which are integer counts, not percentages. This is a silent reporting inaccuracy that misleads the operator reading the log, and contrasts with run_vocalocator_ssl (lines 389-393) which correctly prints percentages.

**Fix:** Either reword the messages to 'Vocalization count attributed to ...', or convert to actual percentages (e.g. 100 * none_in_set.sum() / len(pts_in_set)) to match the wording, mirroring run_vocalocator_ssl.

### [MEDIUM] tests — KeyError branch when npz lacks a '*assignments' array is untested
`assign_vocalizations.py:373-378`

In run_vocalocator_ssl, if model_predictions.npz contains no key ending in 'assignments', assignment_array_candidates is empty and lines 373-378 raise KeyError listing archive keys. The only full-path test (test_run_vocalocator_ssl_full_path_writes_emitter_column, test line 827) always saves assignments=np.array([0,1,-1]) (test line 863), so this guard never fires. A regression removing the guard (reverting to IndexError on assignment_array_candidates[0]) would not be caught.

**Fix:** Add a test mirroring the full-path test but saving model_predictions.npz with a non-matching key (e.g. np.savez(sl_dir/'model_predictions.npz', foo=np.array([0,1,-1]))) and assert pytest.raises(KeyError, match=r"No '\*assignments' array found").

### [MEDIUM] tests — run_vocalocator assignment branches and emitter/metadata writeback never exercised
`assign_vocalizations.py:243-261`

test_vocalocator_run_vocalocator_subprocess_invocation (test line 262) mocks are_points_in_conf_set to a single return_value np.array([True, False]) (test line 315). Since the source stacks this for mouse_idx in range(2) (line 243), the same array is reused for both mice, yielding pts_in_set sums of {1,0}, so none_in_set, two_in_set, and mouse_two_vocalizations (lines 245-255) are all empty. The test asserts only that assessment_assn.npy exists and patches load_session_metadata to (None, None) (test line 305), leaving the emitter-column writeback (lines 275-283) and metadata-update branch (lines 290-297) entirely uncovered for run_vocalocator.

**Fix:** Use side_effect=[arr_mouse0, arr_mouse1] so are_points_in_conf_set returns distinct per-mouse membership producing all four cases. Provide a non-None metadata dict and assert: assessment_assn.npy contents equal the expected vector, the usv_summary.csv 'emitter' column is rewritten per track_names, metadata['Session']['session_usv_assigned'] is True, and per-subject num_assigned_vocalizations equals (assignments==mouse_idx).sum().

### [LOW] correctness — USV onset frame index can exceed track length, raising IndexError
`assign_vocalizations.py:147-148`

onsets_in_video_frames = (onsets_in_seconds * video_frame_rate).astype(int) is used directly to fancy-index tracks[onsets_in_video_frames] at line 148. If any onset second times the median empirical camera sampling rate yields a frame index >= tracks.shape[0] (realistic for a vocalization near the end of a recording, or when the empirical rate is slightly higher than the rate implied by the track count), this raises IndexError and aborts dataset preparation. There is no clamping or bounds check.

**Fix:** Clamp the indices before indexing, e.g. onsets_in_video_frames = np.clip((onsets_in_seconds * video_frame_rate).astype(int), 0, tracks.shape[0] - 1), or validate and raise with a clear message on out-of-range onsets.

### [LOW] correctness — Audio slice truncated at end-of-recording while length_idx is computed from nominal lengths
`assign_vocalizations.py:150-153`

audio is built by slicing the memmap handle[onset:offset, :] (line 150), and numpy silently truncates when offset exceeds audio_file_sample_num. But audio_lengths = usv_offsets_in_samples - usv_onsets_in_samples (line 152) and length_idx = np.cumsum(...) (line 153) are computed from the nominal onset/offset difference, not the actual slice length. If any usv_offsets_in_samples exceeds audio_file_sample_num, the written length_idx no longer matches the concatenated audio for that and subsequent calls, misaligning downstream per-call segment boundaries.

**Fix:** Derive audio_lengths from the actual slices (audio_lengths = np.array([seg.shape[0] for seg in audio])) or clip usv_offsets_in_samples to audio_file_sample_num before computing both the slices and the lengths.

### [LOW] correctness — Hardcoded range(2) assumes exactly two mice while pipeline is otherwise animal-count-general
`assign_vocalizations.py:243`

pts_in_set is stacked over range(2) at line 243, hardcoding two mice, even though prepare_for_vocalocator derives num_animals = tracks.shape[1] generically (line 162) and run_vocalocator_ssl iterates an arbitrary track_names list. For a session with a different animal count, true_locs[:, mouse_idx, ...] for range(2) silently considers only the first two animals, or with one animal indexes true_locs[:,1] out of bounds. The assumption is implicit and unchecked.

**Fix:** Derive the mouse count from true_locs.shape[1] instead of literal range(2), or add an explicit assert/guard that true_locs.shape[1] == 2 with a message documenting the limitation.

### [LOW] correctness — run_vocalocator leaves unassigned/ambiguous USVs with stale 'emitter' values
`assign_vocalizations.py:275-283`

The polars chain at lines 276-281 uses .otherwise(pls.col('emitter')), so vocalizations with assignment -1 (no mouse) or 2 (ambiguous) retain whatever was previously in the 'emitter' column of the CSV. By contrast run_vocalocator_ssl (line 396) starts emitter_expression from pls.lit(value=None, dtype=pls.String), explicitly nulling all non-assigned rows. On reprocessing or with prior emitter labels, run_vocalocator can silently leave incorrect/stale labels on -1 and 2 calls.

**Fix:** Mirror the SSL approach: build the emitter expression starting from pls.lit(value=None, dtype=pls.String) so unassigned (-1) and ambiguous (2) calls are explicitly cleared instead of inheriting prior column values.

### [LOW] dead_code_naming — Dead variable 'pdfs' captured from get_conf_sets_6d but never used
`assign_vocalizations.py:241`

conf_sets, _, pdfs = get_conf_sets_6d(raw_output, arena_dims, 1.0, True) binds pdfs at line 241 but pdfs is never referenced again (only occurrence is this assignment). Only conf_sets is consumed (line 243). The positional True (return_pdf=True) is passed solely to obtain this dead value; note get_conf_sets_6d always materializes the pdfs array internally (utils line ~529) but only returns it when return_pdf is True.

**Fix:** Drop the unused binding: call get_conf_sets_6d with return_pdf defaulting to False and unpack conf_sets, _ = get_conf_sets_6d(raw_output, arena_dims, 1.0). (The internal pdfs list is still built inside the util, so this only removes the dead call-site binding and array return, not the internal work.)

### [LOW] docs_clarity — Stale comment claims it raises an error when calibration file is missing
`assign_vocalizations.py:333`

The comment at line 333 'Locate the calibration file or raise an error if not found' does not match the code path: next(...) raises StopIteration, but that is caught at line 357 and the function merely logs 'No calibration NPZ file found...' and returns at line 360; no error propagates to the caller. The comment overstates the failure handling.

**Fix:** Reword to reflect actual behavior, e.g. '# Locate the calibration file; a missing file raises StopIteration, handled below by logging and returning early.'

### [LOW] docs_clarity — Unexplained *1000 unit conversion lacks a comment
`assign_vocalizations.py:148`

track_locations_at_usv_onsets = tracks[onsets_in_video_frames] * 1000 at line 148 silently converts 3D track coordinates from meters to millimeters to match arena_dims_units = 'mm' set at line 157. The factor 1000 has no explanatory comment, unlike inline comments elsewhere in the file.

**Fix:** Add a brief comment such as '# convert track coordinates from meters to mm to match arena_dims units'.

### [LOW] performance — Per-mouse .with_columns loop forces N full DataFrame passes instead of one chained expression
`assign_vocalizations.py:275-281`

The non-SSL path builds the 'emitter' assignment by calling usv_summary_df.with_columns(...) once per mouse inside the loop (lines 276-281), each triggering a full evaluation pass and re-reading the prior iteration's emitter column. The functionally-equivalent SSL path (lines 396-407) accumulates a single chained pls.when(...).then(...).otherwise(...) and applies .with_columns once, which is the more efficient and idiomatic polars pattern.

**Fix:** Mirror the SSL implementation: initialize emitter_expression (e.g. pls.col('emitter') to preserve existing values, or pls.lit(None) to clear), chain emitter_expression = pls.when(...).then(pls.lit(mouse)).otherwise(emitter_expression) in the loop, then call usv_summary_df.with_columns(emitter_expression.alias('emitter')) once after the loop.

### [LOW] performance — Redundant np.array() copy of memmap slice
`assign_vocalizations.py:150`

At line 150, np.array(handle[onset:offset, :]) forces a copy of each memmap slice, and to_float() immediately calls .astype(np.float32) (per its definition), producing a second owned array. The explicit np.array(...) wrapper creates an extra intermediate copy of every USV audio segment that is discarded one step later.

**Fix:** Drop the np.array(...) wrapper and pass the slice directly: to_float(handle[onset:offset, :]). The .astype in to_float already realizes an owned float32 array, preserving correctness.

### [LOW] tests — subprocess.CalledProcessError catch in run_vocalocator_ssl is untested
`assign_vocalizations.py:357-360`

The except at line 357 catches both StopIteration (missing cal file) and subprocess.CalledProcessError, logging 'No calibration NPZ file found...or the subprocess failed.' and returning early. test_run_vocalocator_ssl_no_calibration_file_returns_early (test line 784) exercises only the StopIteration path. The CalledProcessError path (cal file present but subprocess.run(check=True) fails) is never tested, so a regression catching only StopIteration would slip through.

**Fix:** Add a test placing a *cal*.npz in model_dir, patching subprocess.run to raise subprocess.CalledProcessError(1, 'cmd'), asserting early return (no model_predictions.npz read, no CSV write) and that the failure message was emitted.

### [LOW] tests — Subject-matching loop (str(subject_id) vs track_name) no-match / type-coercion paths untested
`assign_vocalizations.py:286-297`

Both run_vocalocator (lines 292-296) and run_vocalocator_ssl (lines 418-422) iterate metadata['Subjects'] and only set num_assigned_vocalizations when str(subject['subject_id']) == track_name, breaking on first match. The ssl full-path test (test line 869) uses subjects whose ids exactly equal the string track_names, so the str() coercion path (integer subject_id) and the no-matching-subject path are never tested.

**Fix:** Extend the ssl full-path test (or add one) with Subjects containing an integer subject_id matching a string track_name plus an extra non-matching subject; assert the integer-id subject gets num_assigned_vocalizations set and the non-matching subject is left untouched.

### [LOW] tests — Missing 'median_empirical_camera_sr' key in camera frame-count JSON is untested
`assign_vocalizations.py:114-115`

prepare_for_vocalocator opens the camera frame-count JSON and indexes ['median_empirical_camera_sr'] (line 115) with a direct key access (no .get default, per project convention), so a present-but-malformed JSON raises KeyError. The happy-path layout (test line 649) always writes the key, so the KeyError path is uncovered.

**Fix:** Add a test building the prepare layout but writing camera_frame_count_dict.json without 'median_empirical_camera_sr' and asserting prepare_for_vocalocator raises KeyError.


## `build_qlvm_training_set.py` (15)

### [HIGH] correctness — _apply_simple_resize slices the signal window in native coordinates after zooming, discarding real signal on time-upsampling
`build_qlvm_training_set.py:168-174`

_apply_simple_resize zooms spec to target_shape (line 168-169), scaling the time axis by zoom_factors[1] = target_shape[1]/spec.shape[1]. After the zoom, the native signal window (duration columns) occupies duration*zoom_factors[1] columns. But signal_length is computed in NATIVE coordinates as min(duration, target_shape[1]) (line 170) and used to slice spec_interp[:, :signal_length] (line 171) from the ZOOMED array, then center-padded (172-174). Empirically confirmed: with spec.shape[1]=40, duration=40, target_shape=(128,128), the zoomed signal fills 128 columns but signal_length=min(40,128)=40, so only the first 40 of 128 columns are kept -- 88 columns of real signal are DISCARDED. This is exactly the configuration the build tests use (_write_session_h5 n_t=40, durations=40, _CFG target_shape [128,128], time_stretch=False), so the default non-warp path corrupts/truncates data; the tests only assert output shape, not content, so it passes. When duration<spec.shape[1] but T_native<T_target the signal is also mis-centered (e.g. dur=20,T=100,target=64 places the ~13 zoomed columns centered as if 20 wide). The time-stretch path is unaffected (it interpolates in native coords).

**Fix:** Compute the signal length in zoomed coordinates before slicing: signal_length = min(int(round(duration * zoom_factors[1])), target_shape[1]); slice spec_interp[:, :signal_length] and recompute left_pad/right_pad from that. Add a content assertion (e.g. total signal energy preserved within rounding) to a build test so this path is checked beyond shape.

### [MEDIUM] docs_clarity — compute_selected_indices docstring omits the per-session equal-quota subsampling mechanism
`build_qlvm_training_set.py:65-72`

The docstring says samples are subsampled so the total 'approaches dataset_size_constraint', but the algorithm derives a single samples_per_session = target_total // len(sessions) (line 99) and caps EACH session to that quota (lines 104-106). This per-session equal-quota behavior (balanced across sessions, not proportional to session size; sessions with fewer valid USVs keep all of theirs so the total can fall short) is materially non-obvious and undocumented.

**Fix:** Extend the Description: 'Subsampling applies an equal per-session quota (target_total // n_sessions); sessions with fewer valid spectrograms than the quota keep all of theirs, so the realized total may be below the target.' Optionally add a one-line comment at line 99.

### [MEDIUM] tests — Zero-duration placeholder exclusion in compute_selected_indices is untested
`build_qlvm_training_set.py:91, 103`

The (durations > 0) half of the mask (lines 91, 103) deliberately drops duration==0 placeholder rows, documented at lines 66-67. The only filter test (durations [10,60,20,200,5]) has no zero, so the >0 predicate is never exercised; a regression to durations<length_threshold alone would still pass. Verified compute_selected_indices({'s': np.array([0,10,0,20])}, 50.0, None, 0)['s'] == [1, 3].

**Fix:** Add a test asserting zero-duration rows are dropped: compute_selected_indices({'s': np.array([0,10,0,20])}, 50.0, None, 0)['s'].tolist() == [1, 3].

### [MEDIUM] tests — Proportional dataset_size_constraint branch (0 < c <= 1) is untested
`build_qlvm_training_set.py:95-98`

Two size-constraint branches exist: absolute count when >1 (line 96) and proportional int(total_filtered*constraint) when in (0,1] (line 98). Only the absolute branch is tested (constraint=8). Verified: {'a': arange(1,11)} with constraint=0.5 yields 5 rows.

**Fix:** Add a fractional-constraint test: compute_selected_indices({'a': np.arange(1,11)}, 50.0, 0.5, 1)['a'] has length 5.

### [MEDIUM] tests — stretch_specs duration clamping edge cases (duration<=0 and duration>T) are untested
`build_qlvm_training_set.py:211`

Line 211 clamps duration to [1, spec.shape[1]]. test_stretch_specs_outputs_target_shape uses durations [90,45,10], all within [1,90], so neither the lower clamp (duration<=0, which would otherwise make time_orig=np.arange(0) and break the interpolator at lines 134/140) nor the upper clamp is exercised.

**Fix:** Add a test with durations=[0, 200] against a (.,.,90) spec for both time_stretch True/False, asserting out.shape==(2,*target_shape) and np.isfinite(out).all().

### [MEDIUM] tests — build early-return 'No spectrograms survived filtering' path is untested
`build_qlvm_training_set.py:426-428`

When every session is filtered out, specs_list is empty and build emits 'No spectrograms survived filtering; nothing written.' and returns at line 428 before np.concatenate([]) at line 430. No test drives this; a regression that crashed on empty concatenate would go uncaught.

**Fix:** Add a test where all durations >= length_threshold (or length_threshold below n_t), assert build() returns without creating any npz, optionally capture the 'nothing written' message.

### [LOW] docs_clarity — Inline phase-label comments skip Phase 4 (the combine step is unlabeled)
`build_qlvm_training_set.py:445`

Phase labels run 'Phase 1' (380), 'Phase 2' (390), 'Phase 3' (396), then jump to 'Phases 5-6' (445). The concatenate/combine step at lines 430-443 (building all_specs/all_durations/all_spec_ids/all_masks + masking summary) is the logical fourth phase but is unlabeled, while the build() docstring (line 329) lists 'combine' as a distinct step. The numbering gap misleads a reader mapping comments to the docstring step list.

**Fix:** Add '# Phase 4: combine the per-session selected arrays into single cross-session arrays (+ zero-mask placeholders under "none") and report mask coverage.' above line 430, or renumber line 445 to 'Phases 4-5:' so the numbering is contiguous and matches the build() docstring.

### [LOW] docs_clarity — build() Returns block describes output files instead of the None return value
`build_qlvm_training_set.py:335-337`

build() is annotated -> None (line 324) and returns nothing, but its Returns section (line 337) reads 'train_data.npz + val_data.npz (or full_data.npz) + metadata.npz', conflating written files with the return value. The sibling CLI docstring (line 525) correctly states 'None'. Inconsistent convention.

**Fix:** Change the Returns body to 'None' and fold the file-output description into the Description text (e.g. '...and writes train_data.npz + val_data.npz (or full_data.npz) plus metadata.npz to output_directory.'), mirroring the CLI docstring.

### [LOW] performance — build_session_masks does O(M x N_selected) repeated full scans of spectrogram_index
`build_qlvm_training_set.py:272-276`

The loop calls np.flatnonzero(spectrogram_index == int(summary_row)) per selected row, each a full O(M) scan, giving O(M x N_selected). For sessions with many kept USVs and instances this scales quadratically. A single grouping pass (argsort+searchsorted or a dict built once) reduces lookup to roughly O(M log M + N_selected).

**Fix:** Build a spectrogram_index -> mask-row mapping once before the loop (np.argsort + np.searchsorted on the sorted index, or a single-pass defaultdict(list)) and look up per row instead of rescanning the full array.

### [LOW] performance — Under masking_type='none' a full (N,F,T) zero mask array is allocated and shuffled through train_test_split unused
`build_qlvm_training_set.py:437, 452-455`

When masking_type != 'sam', all_masks = np.zeros_like(all_specs) (line 437) allocates a second full-size (N,F,T) array, which train_test_split then fancy-index-copies into train/val partitions (452-455), yet the per-split 'none' branch at line 472 rebuilds masks via np.zeros_like(resized) and ignores split_masks. So a full-size zero array is materialized and copied purely to keep the split signature uniform, roughly doubling peak spectrogram memory.

**Fix:** Avoid threading the full-size zero mask through train_test_split when masking_type != 'sam' (split a lightweight placeholder or reconstruct zero masks per split from the split spec shape, as line 472 already does).

### [LOW] tests — Empty durations_by_key short-circuit in compute_selected_indices is untested
`build_qlvm_training_set.py:94`

Line 94 guards 'if dataset_size_constraint is not None and durations_by_key:' so an empty dict avoids the division-by-zero at line 99 (target_total // len(durations_by_key)). Verified compute_selected_indices({}, 50.0, 8, 0) == {}. No test covers this guard.

**Fix:** Add assert compute_selected_indices({}, 50.0, 8, 0) == {} to pin the empty-input guard against a ZeroDivisionError regression.

### [LOW] tests — build per-session 'idx.size == 0 -> continue' branch is untested
`build_qlvm_training_set.py:411-412`

A session whose selected index array is empty is skipped via 'if idx.size == 0: continue' (lines 411-412) while other sessions still contribute. Every multi-session test has all sessions surviving, so the skip-one-keep-rest branch is uncovered.

**Fix:** Add a two-session build where session A is fully filtered (all durations >= threshold) and B survives; assert output contains only B's spec_ids.

### [LOW] tests — metadata.npz contents (None->nan constraint, dynamic per-split count keys) never asserted
`build_qlvm_training_set.py:489, 496`

Tests only assert metadata.npz.is_file() (line 112) and never load its contents. The None->np.nan substitution (line 489) and the dynamic n_train/n_val/n_full keys from name.split('_')[0] (line 496) are logic-bearing and untested; a wrong key prefix or un-converted None would not be caught.

**Fix:** Load metadata.npz in a build test and assert np.isnan(meta['dataset_size_constraint']) when constraint is None, meta['target_shape'].tolist()==[128,128], and presence of n_train+n_val (or n_full) matching written counts.

### [LOW] tests — build_session_masks not unit-tested directly (multi-instance union and no-group return)
`build_qlvm_training_set.py:220-277`

build_session_masks is only exercised indirectly. Its np.any union over multiple segmentation rows mapped by spectrogram_index (272-276) and the no-mask-group all-ones/zero-count early return (266-267) are not pinned by a direct unit test. test_build_sam_masking_without_mask_group_falls_back covers the no-group path end-to-end, but the multi-row union OR is only checked via masks_len==2, not the boolean OR of distinct regions.

**Fix:** Add a direct unit test on a tmp h5 with two overlapping/disjoint segmentations both at spectrogram_index 0, asserting masks[0] == seg0|seg1 and masks_len.tolist()==[2,...]; plus a no-mask-group case.

### [LOW] tests — CLI comma-splitting with empty/whitespace entries is untested
`build_qlvm_training_set.py:536`

Line 536 parses root_directories via [p.strip() for p in root_directories.split(',') if p.strip()], trimming whitespace and dropping empty tokens. The CLI test passes a clean two-path string, so the strip/empty-filter logic is uncovered.

**Fix:** Add a CLI test with --root-directories '/a/sess1, ,/b/sess2,' asserting the builder receives ['/a/sess1', '/b/sess2'].


## `export_yolo_dataset.py` (15)

### [HIGH] correctness — Double-quoted data.yaml path corrupts Windows paths (backslash escapes)
`export_yolo_dataset.py:286`

Line 286 emits `path: "{output_dir}"` and the comment on line 284 explicitly justifies the double-quote for Windows drive letters. But in a YAML double-quoted scalar, backslashes are escape sequences. I confirmed with PyYAML: `path: "C:\new\spectro\test"` raises a ScannerError (`found unknown escape character 's'`), and sequences like `\n`/`\t` would silently become newline/tab. Since output_dir is a pathlib.Path that uses native separators, on Windows this produces a corrupt or unparseable `path:` and Ultralytics fails to resolve the dataset. The single-quoted form `path: 'C:\new\spectro\test'` parses correctly (verified). The stated Windows justification is exactly the case the code breaks.

**Fix:** Use a single-quoted YAML scalar: `f"path: '{output_dir}'\n"` (single-quoted scalars do not process backslash escapes; only `''` is special). Alternatively emit `output_dir.as_posix()` so separators are forward slashes on every platform.

### [LOW] correctness — No bounds check on validation_split: >1 or <0 yields garbled split counts
`export_yolo_dataset.py:248-250`

validation_split is read straight from config (line 201) with no validation. If validation_split > 1, `n_val = round(n_total * validation_split)` exceeds n_total, the slice `rng.permutation(n_total)[:n_val]` returns all positions (everything goes to val), and the summary on line 296 reports `n_total - n_val` as a negative train count. A negative split is similarly unguarded. There is no upstream bounds check. That said, the config default is 0.2 and this is a developer-set settings value (not user-facing free input), so misuse is unlikely in practice; the existing tests at split=0.0 and 1.0 pass. Downgrading from the candidate's medium: it is a real robustness gap but low likelihood and low blast radius.

**Fix:** Validate early alongside the existing label_source check (around line 206), e.g. `if not (0.0 <= validation_split <= 1.0): raise ValueError(...)`.

### [LOW] correctness — Two-pass re-open re-reads durations and re-runs np.flatnonzero (latent index coupling + redundant I/O)
`export_yolo_dataset.py:237-242`

val_positions is computed from the phase-1 catalog (lines 237-242) and consumed in phase 2 (lines 255-263) by re-opening the same files, re-iterating the same groups, and re-reading durations + re-running np.flatnonzero. The iteration is deterministic for a fixed file so this is correct in practice, and the durations arrays are small so the I/O cost is modest. The two-pass structure is required because the seeded split must run over the whole catalog before any image is written. The only redundancy is the duplicated durations read / flatnonzero, plus a latent coupling if an H5 were mutated between passes. I merged the candidate's correctness finding (line 237-242) and performance finding (line 237-242,255-261) since they describe the same code and the same fix.

**Fix:** In phase 1, store the valid-row list per (h5_path, session_id) alongside the catalog and reuse it in phase 2 instead of re-reading durations and re-running np.flatnonzero. Optional given small arrays; do not complicate the clear two-phase flow unless the catalog is large.

### [LOW] docs_clarity — Docstring 'never accidentally empty' overstates the guarantee when n_val rounds to 0
`export_yolo_dataset.py:180-181`

The export docstring (lines 180-181) states the seeded permutation means 'the val set is never accidentally empty.' But `n_val = round(n_total * validation_split)` (line 248) is 0 whenever `n_total * validation_split < 0.5` (small datasets or small split fraction), leaving images/val empty. The word 'accidentally' partly covers this (the point is removing per-image coin-flip variance, which is true), but the absolute phrasing 'never empty' is misleading since a deterministic round-to-zero still empties val. The test at line 193 only covers split=0.34 on n=3 (rounds to 1).

**Fix:** Either enforce a floor (`n_val = max(1, round(...))` when n_total > 0 and validation_split > 0) or soften the docstring to note the val set is empty only when the rounded fraction is zero.

### [LOW] docs_clarity — spec_to_yolo_image docstring omits the max(1, ...) 1-bin floor on window width
`export_yolo_dataset.py:60-75`

The Description (line 60: window `[0, min(duration, T))`), the duration parameter (line 68: width is `min(duration, T)`), and the Returns (line 75: `W == min(duration, T)`) all state the width without the floor. The code at line 83 is `eff_duration = max(1, min(int(duration), n_time))`, guaranteeing at least one time bin. The documented formula is wrong when min(duration, T) <= 0. export() only passes duration > 0 rows, but spec_to_yolo_image is a standalone documented helper. I merged the candidate's three separate findings (Returns, Description, duration param) into one since they are the same omission across the same docstring.

**Fix:** State the width as `max(1, min(duration, T))` in the Description, the duration parameter, and the Returns, noting the floor guarantees a non-empty image.

### [LOW] docs_clarity — export() Returns section documents a side effect for a None-returning method
`export_yolo_dataset.py:189-191`

The signature is `def export(self) -> None:` yet the Returns section (line 191) reads 'A YOLO dataset directory (images/, labels/, data.yaml).' Every other docstring in the file documents the actual return value and states None explicitly when nothing is returned (e.g. __init__ at 128, _cc_labels at 162-163, the CLI at 320-322). This Returns text describes a filesystem artifact rather than the return value, inconsistent with the file convention.

**Fix:** State 'None' as the return value (the directory creation is already described in the Description), or 'None (writes a YOLO dataset directory ... as a side effect).'

### [LOW] performance — Per-row float32 cast is immediately upcast to float64 in render_spec_image
`export_yolo_dataset.py:264`

Line 264 does `spec = specs[row].astype(np.float32)`, producing a float32 copy, then spec_to_yolo_image -> render_spec_image (boxes.py:277) does `arr = np.asarray(spec, dtype=np.float64)`, upcasting to float64 (a second copy). float32 -> float64 is lossless so the rendered image is bit-identical whether the row is read as float32 or its native dtype; the explicit float32 cast is a redundant intermediate copy per spectrogram. The CC detector also receives spec but boxes.py likewise casts internally. Minor; matters only with many/large specs.

**Fix:** Read the row without the explicit float32 cast (`spec = specs[row]`), letting render_spec_image cast to float64 once. Verify the CC detector tolerates the native dtype (it casts internally, so it should).

### [LOW] tests — spec_to_yolo_image duration-clamping branches (min clamp, max(1,...) floor) untested
`export_yolo_dataset.py:82-84`

`eff_duration = max(1, min(int(duration), n_time))` has three behaviors but test_spec_to_yolo_image_shape (tests file line 57) only covers duration=40 on a 128-wide spec (the normal duration < T case). The min() clamp (duration > T should give W=T) and the max(1, ...) floor (duration <= 0 should give W=1, not 0-width which would break Image.save) are uncovered. The floor branch is only reachable via the direct unit test since export() passes duration > 0.

**Fix:** Add parametrized cases to test_spec_to_yolo_image_shape: duration=200 on a 128-wide spec asserts width == 128; duration=0 asserts width == 1 and shape == (128, 1, 3).

### [LOW] tests — Multiple session groups within one H5 file untested
`export_yolo_dataset.py:239-242`

Phase 1 (line 239) and Phase 2 (line 257) iterate `for session_id in h5_file["spectrogram"]`, and the docstring (line 234-235) states an H5 may hold more than one session group. _write_session_h5 (tests line 51) writes exactly one group (_SESSION_ID). The multi-group iteration and spec_id namespacing (session_id_row) across groups are never exercised.

**Fix:** Extend _write_session_h5 (or add a helper) to create two groups in one H5 with valid durations, then assert images/labels are written under both session prefixes with no spec_id collision.

### [LOW] tests — Multiple root_directories (multi-file catalog) untested
`export_yolo_dataset.py:223-230`

spectrogram_h5_paths is built over self.root_directories (lines 223-230) and both phases iterate every file, with stable file-list ordering feeding the split. Every test passes root_directories=[str(root)] (single root). The multi-file enumeration, ordering, combined-dataset split, and first_match_or_raise per root are uncovered.

**Fix:** Write two roots via _write_session_h5 with distinct session ids, pass root_directories=[str(root_a), str(root_b)], and assert the combined image count and that spec_ids from both appear.

### [LOW] tests — label_source='manual' with a spectrogram lacking a label file (background) untested
`export_yolo_dataset.py:272-276`

The docstring (lines 22-23) states manual-source spectrograms with no {spec_id}.txt get an empty label file (treated as background). At lines 272-276, manual + has_manual False yields raw_lines=[] and an empty label file. test_export_manual_copies_labels (tests line 91) only covers the manual-file-exists case; test_export_merge_uses_manual_then_cc covers the merge cc-fallback branch, which is different. The manual-source empty-label branch is uncovered.

**Fix:** Add a manual test with durations=[40, 60] and a manual file only for row 0; assert labels/train/<session>_1.txt exists and is empty, and the cc detector is not invoked.

### [LOW] tests — Blank-line stripping of manual label files untested
`export_yolo_dataset.py:275-276`

Lines 275-276 read manual text via splitlines() and drop blank lines with `[line for line in raw_lines if line.strip()]` (comment: Ultralytics rejects empty label lines). test_export_manual_copies_labels uses a single clean line, so the stripping that distinguishes this from a verbatim copy is never exercised.

**Fix:** Write a manual file with two real lines separated by a blank line and a trailing newline; assert the output label file's splitlines() equals exactly the two non-blank lines.

### [LOW] tests — data.yaml contents (nc, names, quoted path) never asserted
`export_yolo_dataset.py:285-292`

Tests assert (out_dir / 'data.yaml').is_file() (tests line 79) but never read it. The YAML at lines 285-292 includes nc: 1, names '0: usv', train/val paths, and the (currently double-quoted) path line. A regression in any of these would pass all current tests. This is especially relevant given the path-quoting bug above: a test asserting the path uses single quotes would lock in the fix.

**Fix:** In test_export_cc_writes_dataset, read data.yaml and assert 'nc: 1', 'train: images/train', '0: usv' are present, and assert the path line uses the corrected single-quote form.

### [LOW] tests — Export summary box-count / val-train log message untested
`export_yolo_dataset.py:294-297`

n_boxes (line 282) and the summary message (lines 294-297) reporting 'N spectrogram images (M boxes, ...; V val / T train)' are never asserted; message_output is always a no-op lambda in tests. A miscount or swapped val/train count would go undetected.

**Fix:** Capture message_output into a list in one cc test and assert the summary string contains the expected val/train counts and a 'boxes' substring.

### [LOW] tests — CLI error path and ParameterSource filtering not covered end-to-end
`export_yolo_dataset.py:303-340`

test_export_yolo_dataset_cli_routes_and_splits_paths (test_qlvm_pipeline_clis.py:108) mocks YOLODatasetExporter and modify_settings_json_for_cli, so it only checks comma-splitting (line 333) and one call. The provided_params COMMANDLINE filtering (line 325) and the manual/merge ValueError surfacing through the CLI (real export with label_source='manual' and no manual dir -> non-zero exit) are untested. I trimmed the candidate's provided_params assertion suggestion since asserting on an internal list is brittle; the error-path test is the actionable part.

**Fix:** Add a CLI test passing --label-source manual without --manual-labels-directory against a synthesized session (not mocking the exporter) and assert result.exit_code != 0 with 'manual_labels_directory' in result.output.


## `das_inference.py` (15)

### [MEDIUM] docs_clarity — ValueError message references parameter names that do not exist
`das_inference.py:385-389`

The message says "check `freq_lower_bound` vs `nfft` / sampling rate", but neither key exists. The real settings keys are `low_freq_cutoff` (line 334, processing_settings.json:207) and `len_win_signal` (line 331, processing_settings.json:206). A user hitting this error would search for names that are not in the codebase.

**Fix:** Reword to reference the real keys, e.g. "check `low_freq_cutoff` vs `len_win_signal` / sampling rate".

### [MEDIUM] docs_clarity — summarize_das_findings Returns docstring omits leading usv_id column and misspells EMITTER
`das_inference.py:193-196`

The Returns block lists (N_USV, START, STOP, DURATION, PEAK_AMP_CH, MEAN_AMP_CH, CHs_COUNT, CHS_DETECTED, EMMITER). The written CSVs (lines 496-508, 553-565, 579-591) all begin with a `usv_id` column and end with `emitter`; the docstring omits usv_id and doubles the M in EMMITER.

**Fix:** Update to include the leading USV_ID column and fix the spelling: (N_USV, USV_ID, START, STOP, DURATION, PEAK_AMP_CH, MEAN_AMP_CH, CHS_COUNT, CHS_DETECTED, EMITTER).

### [MEDIUM] performance — Memmap window read up to three times per USV (peak argmax, mean argmax, STFT)
`das_inference.py:353-374`

Inside the per-USV loop the same sample window is sliced from the np.memmap three times: line 354 (np.argmax for peak_amp_ch), line 358 (np.abs(...).mean(axis=0) for mean_amp_ch), and lines 374/403 (STFT input). Each fresh index of a memmap triggers a separate read of the same byte range from disk. With n_usv potentially in the thousands this multiplies disk I/O. Materializing the window once and reusing it for all three eliminates the redundant reads with identical results.

**Fix:** Read once: window = np.asarray(audio_file_data[start_usv:stop_usv, :]); then peak_amp_ch = np.unravel_index(np.argmax(window), window.shape)[1]; mean_amp_ch = np.argmax(np.abs(window).mean(axis=0)); and feed the STFT from window[:, usv_detected_chs].astype('float32').T (and window[:, usv_detected_chs[0]] in the single-channel branch) instead of re-slicing the memmap.

### [MEDIUM] tests — lower_bin overflow ValueError (line 384-390) is never tested
`das_inference.py:384-390`

The defensive ValueError raised when lower_bin >= spectrogram_data_selected_ch.shape[1] has no covering test in tests/processing/test_inference.py; the happy-path test uses sampling_rate that keeps lower_bin small, so the raise is dead from the suite's perspective.

**Fix:** Add a test that mutates summarize_das_findings['low_freq_cutoff'] to a value larger than sampling_rate/2 so lower_bin exceeds the STFT freq-axis length, then assert pytest.raises(ValueError) matching 'exceeds STFT freq-axis length' (it propagates because the except only catches IndexError/FileNotFoundError).

### [MEDIUM] tests — condition_0 drop path (peak/mean amp channel not in detected channels) untested deterministically
`das_inference.py:364-368, 438-453`

Phase 4 marks a USV for dropping when its peak or mean amplitude channel is not among its detected channels (condition_0_list, lines 365-368; drop at 449). The fixture injects random signal across channels, so condition_0 is nondeterministic and no test forces it True and asserts the drop. This is the primary noise-rejection rule.

**Fix:** Add a test that writes audio whose loudest samples sit on a channel NOT in chs_detected (zero all detected channels, inject peak into a non-detected one), seed the RNG, then assert that USV is dropped (df.height) and the drop count message reflects it.

### [LOW] correctness — np.nanpercentile over an all-NaN descriptor array returns NaN, silently disabling that filter
`das_inference.py:416-427`

mean_signal_correlations is filled only on the multi-channel branch (line 398) and signal_variance only on the single-channel branch (line 411). If every merged USV (with n_usv>1) is detected on a single channel, mean_signal_correlations stays entirely NaN; conversely signal_variance can be entirely NaN if every USV is multi-channel. np.nanpercentile over an all-NaN array emits a RuntimeWarning and returns NaN. The test suite even acknowledges this by silencing the warning at tests/processing/test_inference.py:414. With NaN, condition_1/condition_2 (lines 440-448) are guarded by `not np.isnan(...)` so they correctly no-op, but the cutoff line drawn in the histogram (lines 470/481) and the displayed cutoff message (lines 429/432) are NaN, and the configured floor/ceiling (noise_corr_cutoff_min / noise_var_cutoff_max) does not deterministically take effect because max(nan, min)/min(nan, max) is order-dependent and typically returns nan.

**Fix:** Guard each percentile call: only call np.nanpercentile when np.any(~np.isnan(arr)); otherwise fall back to the configured noise_corr_cutoff_min / noise_var_cutoff_max so the cutoff is deterministic and no all-NaN-slice RuntimeWarning is raised.

### [LOW] dead_code_naming — Parameter/attribute 'exp_settings_dict' is stored but never read and never passed by any caller of this class
`das_inference.py:50, 84`

__init__ accepts exp_settings_dict (line 50) and stores self.exp_settings_dict (line 84); grep shows it is referenced only at lines 50, 64, 84 within das_inference.py and is never read. All four call sites (preprocess_data.py:297, 303, 923, 960) construct FindMouseVocalizations without exp_settings_dict, and the tests never pass it. It is a dead parameter and dead attribute carrying no behavior.

**Fix:** Remove the exp_settings_dict parameter, its docstring entry (lines 64-65), and the self.exp_settings_dict assignment (line 84), unless deliberately retained as a forward-compatible hook (in which case add a comment noting it is currently unused).

### [LOW] docs_clarity — Skip message describes a stricter filename pattern than the regex enforces
`das_inference.py:248-251`

The message says expected pattern '<device>_..._<chXX>_annotations.csv', implying chXX immediately precedes '_annotations'. The regex (line 42) is ^([ms])_.*_(ch\d{2})_.*annotations\.csv$, which tolerates arbitrary content between chXX and 'annotations' (per the header example ch01_cropped_to_video_hpss_filtered_annotations.csv at lines 34-37).

**Fix:** Reword to '<device>_..._<chXX>_...annotations.csv' to match the regex and the header comment.

### [LOW] docs_clarity — Non-obvious negative-index mmap filename parsing lacks an explanatory comment
`das_inference.py:318-323`

The unpacking parses the mmap filename via underscore splits with negative indices: [-1][:-5]->data_type (strips '.mmap'), [-2]->channel_num, [-3]->sample_num, [-4]->audio_sampling_rate. The ordering is non-intuitive and the [:-5] slice is unexplained; the identical block is duplicated at lines 523-528 with no comment.

**Fix:** Add a brief comment documenting the expected trailing filename layout (e.g. '..._<sampling_rate>_<sample_num>_<channel_num>_<dtype>.mmap') and what [:-5] strips, at both line 318 and line 523.

### [LOW] performance — Duplicate memmap window read in the lone-USV branch
`das_inference.py:538-543`

The n_usv==1 branch indexes audio_file_data[start_usv:stop_usv, :] twice: line 539 (argmax for peak_amp_ch) and line 543 (np.abs(...).mean for mean_amp_ch), reading the same memmap window from disk twice.

**Fix:** Materialize once: window = np.asarray(audio_file_data[start_usv:stop_usv, :]); compute both peak_amp_ch and mean_amp_ch from window.

### [LOW] tests — single-channel variance branch inside Phase-4 loop never reached with real data
`das_inference.py:399-414`

The else branch (lines 399-414) computing signal_variance only runs when a merged USV has exactly one detected channel within the n_usv>1 loop. All multi-USV tests use channels=('ch01','ch02') so every merged USV is multi-channel; the single-USV test uses the n_usv==1 branch (line 510), not this loop. So signal_variance and its condition_2 drop (445-448) are untested.

**Fix:** Add a test with n_usv>1 where merged USVs are each detected on a single channel (two non-overlapping USVs each only on ch01), so the variance branch executes; assert a low-variance USV is dropped via condition_2.

### [LOW] tests — noise cutoff clamping (noise_corr_cutoff_min / noise_var_cutoff_max) untested
`das_inference.py:416-427`

noise_corr_cutoff=max(6th pctile, noise_corr_cutoff_min) and noise_var_cutoff=min(94th pctile, noise_var_cutoff_max) clamp data-driven cutoffs to configured bounds; no test verifies the floor/ceiling wins, and the cutoff messages (lines 428-433) are unchecked. A swapped max/min would pass the current suite.

**Fix:** Add a test setting noise_corr_cutoff_min above any achievable correlation (e.g. 2.0) and assert the logged correlation cutoff equals that floor; set noise_var_cutoff_max very low and assert the variance cutoff is clamped. Capture via message_output=msgs.append.

### [LOW] tests — metadata session_usv_count update untested for the multi-USV and no-filter paths
`das_inference.py:593-600`

Lines 598-600 set metadata['Session']['session_usv_count']=len(merged) and save when metadata is not None. Only the single-USV test (test_summarize_das_findings_single_usv_writes_csv_and_counts_it, line 525) passes non-None metadata; all multi-USV/no-filter tests mock load_session_metadata to return (None, None), so the count update is not asserted for the main >1-USV path.

**Fix:** Patch load_session_metadata to return ({'Session': {}}, meta_path) in a multi-USV test and assert metadata['Session']['session_usv_count']==len(merged) and save_session_metadata.assert_called_once(); same for the no-filter path.

### [LOW] tests — IndexError arm of the broad except is untested
`das_inference.py:602-605`

The except catches (IndexError, FileNotFoundError) (line 602). The FileNotFoundError arm is covered by test_summarize_das_findings_missing_mmap_reports_real_cause (line 564), which now triggers first_match_or_raise -> FileNotFoundError, not IndexError. No test triggers the IndexError arm (e.g. an mmap filename lacking enough underscore tokens for int(...split('_')[-2/-3/-4]) at lines 318-323).

**Fix:** Add a test with valid multi-USV annotations plus an mmap whose name lacks the expected trailing _<sr>_<n>_<ch>_<dtype>.mmap structure so the int(split('_')[-N]) parsing raises, and assert the skip message names the real exception type and the method does not crash. (Note: malformed-token parsing raises ValueError, which the except does NOT catch — consider whether IndexError is even reachable, or widen the except.)

### [LOW] tests — das_command_line_inference move filter not tested for the non-matching (lookalike) case
`das_inference.py:175-177`

Lines 175-177 move only files whose name endswith f'.{save_format}'. The existing test verifies a matching annotations file IS moved, but no test verifies a sibling NOT ending in save_format (a leftover .wav, or a substring-lookalike like 'foo_csv.bak') is left in place. The comment explicitly notes endswith (not substring) to avoid moving lookalikes; that distinction is unguarded.

**Fix:** Extend the inference test to drop a 'leftover.wav' and a substring-lookalike (e.g. 'foo_csv.bak') into hpss_dir and assert they remain there and were not moved into das_annotations.


## `assign_vocalizations_utils.py` (15)

### [HIGH] correctness — LinAlgError fallback produces an unnormalized PDF, yielding empty confidence sets
`assign_vocalizations_utils.py:268-272`

In eval_pdf_with_angle the main path normalizes probs to sum to 1 (line 263). The singular-covariance fallback instead does probs[closest_point_idx, :] = 1, so the returned array sums to len(points_angular) (45 in get_conf_sets_6d via points_angular at line 482). Downstream get_confidence_set(total_pdf, 0.95) builds its set via cumsum < 0.95; verified empirically that a probs array summing to 45 yields the first cumsum value = 1.0, so the strict < 0.95 mask is all-False and an EMPTY confidence set is returned for any vocalization whose covariance is singular. NOTE: the existing test test_eval_pdf_with_angle_handles_singular_covariance (test_process.py:1914) asserts out.sum()==4.0, so the unnormalized fallback is the currently tested/encoded behavior; the bug is the downstream empty-set consequence, not just the raw sum. Fixing requires updating that test too.

**Fix:** Normalize the fallback before returning, e.g. set probs[closest_point_idx, :] = 1.0 / len(points_angular) (or add probs /= probs.sum()) so the fallback PDF is a proper distribution; update test_process.py:1914 which currently asserts the unnormalized sum.

### [MEDIUM] tests — get_conf_sets_6d is never executed directly, only mocked
`assign_vocalizations_utils.py:450-535`

get_conf_sets_6d is the central orchestrator wiring convert_from_arb, compute_covs_6d, make_xy_grid, estimate_angle_pdf, eval_pdf_with_angle, and get_confidence_set via joblib.Parallel. In test_inference.py:310 it is replaced by mocker.patch(...); a grep for non-mocked calls in tests/ returns nothing. Consequently the inner routine worker, the joblib fan-out, the result-stacking, and both the return_pdf=True and default return_pdf=False branches are untested. Regressions in routine wiring or returned-tuple arity would go uncaught.

**Fix:** Add a direct test calling get_conf_sets_6d on a tiny deterministic input (e.g. raw_output of shape (2, 27), arena_dims_mm = [400,400]), asserting return_pdf=False gives a 2-tuple of (2,100,100) arrays and return_pdf=True gives a 3-tuple with pdfs of shape (2,100,100,45).

### [LOW] correctness — Confidence-set construction drops the threshold-crossing cell (slight undercoverage)
`assign_vocalizations_utils.py:382`

get_confidence_set uses sorted_indices[cumsum < confidence_level], including only cells whose cumulative mass is strictly below the level, which excludes the cell that brings cumulative mass up to/past the level. The resulting highest-density set covers slightly LESS than confidence_level (verified: a uniform 1000-cell PDF at level 0.95 yields 949 cells / 0.949 mass instead of >=0.95). Standard HPD construction includes the crossing cell. This biases every confidence set marginally smaller than nominal, affecting are_points_in_conf_set decisions, but only by ~one cell out of thousands.

**Fix:** Include the crossing cell, e.g. k = np.searchsorted(cumsum, confidence_level) + 1; idx = sorted_indices[:k], or shift the comparison so the first cell reaching the level is retained.

### [LOW] correctness — Angle lookup bins do not match the PDF's angular grid (half-bin misalignment)
`assign_vocalizations_utils.py:569`

The joint PDF's angular axis (get_conf_sets_6d line 482) is indexed by bin CENTERS of np.linspace(-pi, pi, 46) (centers approx -3.0718..3.0718). But are_points_in_conf_set line 569 digitizes head_to_nose_yaw against angle_bins = np.linspace(-pi, pi, 45, endpoint=False) whose edges start at -pi (-3.1416). Both produce 45 entries (no OOB), but lookup index k does NOT correspond to the same angular slice k of the PDF; verified the grids are offset by up to half a bin (e.g. angle -3.0 maps to lookup idx 1 / PDF center -2.93, though it is closest to center idx 0 / -3.07). Drifts assignment near wrap-around.

**Fix:** Derive the lookup binning from the same edges used to build the PDF (np.linspace(-pi, pi, 46)) and digitize against those, so the lookup index aligns with the PDF center index.

### [LOW] correctness — to_float assumes integer input dtype; np.iinfo raises on float arrays
`assign_vocalizations_utils.py:151`

to_float computes np.iinfo(input_array.dtype).max, which raises ValueError ('Invalid integer data type') on a floating-point input (verified). The sole caller (assign_vocalizations.py:150) passes raw integer audio so it works, but the function is a general helper whose docstring states only 'Converts the input array to float16' and does not state the integer-only precondition.

**Fix:** Document the integer-input precondition in the docstring, or guard with if np.issubdtype(input_array.dtype, np.integer) and pass float inputs through.

### [LOW] correctness — write_to_h5 raises on an empty audio list
`assign_vocalizations_utils.py:199`

f.create_dataset(name='audio', data=np.concatenate(audio, axis=0)) calls np.concatenate on the audio list. If audio is empty (a session with zero detected USV segments), np.concatenate raises ValueError ('need at least one array to concatenate') and no file is written (verified). The caller (assign_vocalizations.py:150) builds audio from usv_segments, which can plausibly be empty.

**Fix:** Guard the concatenation, e.g. audio_data = np.concatenate(audio, axis=0) if len(audio) else np.empty((0, n_channels), dtype=np.float16), so an empty USV set still writes a valid zero-length dataset.

### [LOW] dead_code_naming — Dead parameters center_rad / concentration and dead von Mises branch in eval_pdf_with_angle
`assign_vocalizations_utils.py:214-215`

The histogram is None von Mises branch (lines 251-254) that consumes center_rad/concentration via sp_vonmises.pdf is never exercised: the sole production caller routine() (line 510) passes histogram=est_angle_pdf, and both unit tests (test_process.py:1890, :1906) pass histogram=. A repo-wide grep confirms no other caller and no test sets histogram=None. So the if histogram is None branch is dead and center_rad/concentration are dead parameters at their None defaults.

**Fix:** Remove center_rad/concentration and the histogram is None branch (making histogram required), or add a test that calls eval_pdf_with_angle with histogram=None and explicit center_rad/concentration to exercise the branch.

### [LOW] docs_clarity — to_float docstring omits the normalization step (the core behavior)
`assign_vocalizations_utils.py:134-149`

The Description says only 'Converts the input array to float16' but the implementation first normalizes by the integer dtype max: (input_array.astype(np.float32) / np.iinfo(input_array.dtype).max).astype(np.float16). The division rescales an integer PCM array into [-1,1] (or [0,1]), the meaningful operation, and is not mentioned. The integer-input precondition is also undocumented.

**Fix:** Reword the Description to state it normalizes an integer-typed array to [-1,1] by dividing by the dtype's max representable value, then casts to float16, intended for raw integer PCM audio, with an integer-dtype precondition.

### [LOW] docs_clarity — eval_pdf_with_angle Description is stale: it is a joint spatial-by-angular PDF, not a plain MVN
`assign_vocalizations_utils.py:218-245`

The Description states 'Evaluate the multivariate normal PDF at points' but the function computes a joint distribution: a 2D spatial Gaussian (einsum log-prob with precision matrix) multiplied by an angular distribution (von Mises when histogram is None, else normalized histogram), combined in log space, max-subtracted, exponentiated, and normalized to sum to 1 over a (spatial x angular) grid. It returns a discrete normalized PMF, not a continuous MVN value. The angular component, normalization, branch, and LinAlgError fallback are all undocumented.

**Fix:** Rewrite the Description to describe the joint spatial-angular PMF (product of 2D Gaussian over space and angular distribution over bins, log-space combination with max-subtraction, normalized to sum to 1) and document the LinAlgError degenerate-covariance fallback.

### [LOW] docs_clarity — write_to_h5 audio parameter type in docstring is malformed and disagrees with the signature
`assign_vocalizations_utils.py:174`

The docstring lists audio (list[tuple[np.ndarray]) with unbalanced brackets and a missing second tuple element, while the signature is audio: list[tuple[np.ndarray, Any]]. Note the actual caller (assign_vocalizations.py:150) passes a list[np.ndarray], not a list of tuples, so both the docstring and the signature annotation are arguably inaccurate. The concatenation along axis 0 into a single 'audio' dataset is also not described.

**Fix:** Fix the bracket typo, reconcile the annotation with the actual caller's list[np.ndarray], and describe the axis-0 concatenation behavior.

### [LOW] docs_clarity — are_points_in_conf_set: undocumented binning scheme and node indices
`assign_vocalizations_utils.py:564-572`

The function bins with x_bins/y_bins of 100 points and angle_bins of 45 (endpoint=False), then np.digitize(...) - 1. The reader is not told these must match the grid/histogram built in get_conf_sets_6d, that -1 converts digitize's 1-based result to 0-based, that x/y are safe only because nose_points are clipped (line 566) while angles are not, nor that points indices 0 and 1 correspond to nose vs head in the head_to_nose computation (line 564-565).

**Fix:** Add comments explaining the binning scheme, the -1 conversion, the clipping-dependent index safety, and which point indices map to which body nodes.

### [LOW] docs_clarity — compute_covs_6d: undocumented Cholesky-factor construction and arena scaling
`assign_vocalizations_utils.py:275-304`

The Description only says 'Computes the covariance matrix from the raw output of the model.' The body packs the lower triangle of a Cholesky factor L from raw_outputs[:, 6:], applies softplus to the diagonal for positivity, scales by 0.5 * arena_dims.max(), and forms covs = L @ L^T. This parameterization and the expected raw_outputs layout (6 means in [:6], 21 lower-triangular entries in [6:27], num_outputs >= 27) are undocumented.

**Fix:** Document the construction (lower-triangular Cholesky factor with softplus-positive diagonal, scaled by half the larger arena dimension, covs = L L^T) and the expected raw_outputs layout/size.

### [LOW] performance — Large pdfs array is always materialized even when return_pdf is False
`assign_vocalizations_utils.py:532-535`

Line 530 unconditionally executes pdfs = np.array([result[2] for result in results]), stacking every per-vocalization full PDF (each ~100x100x45 float64 ~ 3.4 MB) into one array. When return_pdf=False (default), this stacked array is built and immediately discarded at line 535, wasting time and a potentially multi-GB allocation for large vocalization counts.

**Fix:** Move pdfs = np.array([result[2] for result in results]) inside the if return_pdf: block so it is only built when returned.

### [LOW] tests — eval_pdf_with_angle von Mises branch (histogram=None) is untested
`assign_vocalizations_utils.py:251-254`

eval_pdf_with_angle has two angular paths: histogram is None -> sp_vonmises.pdf(loc=center_rad, kappa=concentration) (lines 252-254), and the histogram path (line 256). Both tests (test_process.py:1890, :1906) pass histogram=, and grep finds no center_rad/concentration/vonmises usage in tests. A break in the von Mises call signature would be undetected.

**Fix:** Add a test calling eval_pdf_with_angle with histogram=None, center_rad=0.0, concentration=2.0 and a non-singular cov_2d, asserting shape (*render_dims, len(points_angular)) and sum ~1.0.

### [LOW] tests — to_float only tested for int16; unsigned/other-width dtypes uncovered
`assign_vocalizations_utils.py:151-153`

to_float divides by np.iinfo(input_array.dtype).max, which maps unsigned types (uint8/uint16) [0,max]->[0,1] with no negative range vs signed int16's [-1,1]. The only test (test_process.py:1823) covers int16. Since audio is int16 this is low impact, but the unsigned path is unverified.

**Fix:** Add a parametrized np.uint8 case (e.g. [0,255]) asserting output maps to [0.0, ~1.0] and dtype float16.


## `generate_spectrograms.py` (14)

### [LOW] correctness — Stored duration can exceed the spectrogram array width when a USV is longer than num_time_bins
`generate_spectrograms.py:137-139`

original_time_bins is captured from the native pre-fix STFT frame count (spec_db.shape[1]) before librosa.util.fix_length truncates the array to num_time_bins (128). So when a USV exceeds 128 frames the 'durations' dataset advertises a value larger than the stored array's actual time-bin width. This is semantically inaccurate, but it is NOT a runtime hazard: every consumer of the durations dataset already clamps to the stored width before slicing -- usv_embedding_explorer.py:794 (dur = max(1, min(dur, spec.shape[1]))), make_usv_spectrograms.py:891 (valid_cols = max(1, min(int(sess_durations[spec_idx]), n_time_in_spec))), and the model-feeding stretch_specs in build_qlvm_training_set.py:211 (duration = int(min(max(int(durations[idx]), 1), spec_work.shape[1]))). The candidate's claimed harm (over-read past the signal / index out of bounds) cannot occur in any referenced reader.

**Fix:** Optionally clamp the reported duration to the stored width at the source (e.g. return min(original_time_bins, num_time_bins)) for semantic correctness, so 'durations' never advertises more frames than exist. Low priority since all downstream readers already clamp defensively; this is a clarity/correctness-of-stored-metadata nicety, not a bug fix.

### [LOW] correctness — round() for segment sample bounds diverges from das_inference's floor/ceil convention
`generate_spectrograms.py:282-283`

Sample bounds use s0=round(t0*sr) and s1=round(t1*sr) (lines 282-283), whereas das_inference.py uses int(np.floor(start*sr)) and int(np.ceil(stop*sr)) (lines 351-352, 536-537). round() can drop up to one sample at each boundary versus floor/ceil, so the analyzed window is not exactly the sample span DAS used. The module docstring's 'parses exactly as das_inference' refers specifically to the mmap-FILENAME parsing (line 14-16), not to sample-bound derivation, so this does not violate a documented contract. With offset=0.0 the effect is sub-sample.

**Fix:** For consistency with das_inference, use s0=max(0, int(np.floor(t0*audio_sampling_rate))) and s1=min(sample_num, int(np.ceil(t1*audio_sampling_rate))).

### [LOW] correctness — Empty frequency band (freq_mask selecting 0 rows) would crash np.interp with an opaque error
`generate_spectrograms.py:124-135`

If min_freq/max_freq are misconfigured so freq_mask selects zero FFT bins (min_freq>max_freq, or the band entirely above Nyquist), power_spec becomes shape (0, T); spec_db.shape[0]==0 != num_freq_bins enters the resample branch with freq_orig=np.linspace(0,1,0)=[] and np.interp(freq_interp, [], []) raises an opaque error deep in the per-channel loop, with no upfront validation. There is no min_freq<max_freq check anywhere upstream (grep-confirmed). With current settings (30k-120k, Nyquist 125k) the band is valid, so this is a robustness gap for misconfiguration only.

**Fix:** Validate up front (before the STFT loop) that min_freq < max_freq and that freq_mask.any() is True, raising a clear parameter error.

### [LOW] dead_code_naming — Stale parameter `noverlap` documented but never read; inconsistent with hop_length
`generate_spectrograms.py:76`

The compute_usv_spectrogram docstring (line 76) lists `noverlap` as a consumed spec_params key, and the settings block (processing_settings.json:222) plus the test fixture (test_generate_spectrograms.py:27) define noverlap=1792. But the function never reads spec_params['noverlap'] -- STFT framing is driven solely by nperseg and hop_length (lines 93-97, 110-122). It is dead/stale, and internally inconsistent: with nperseg=2048 and hop_length=512 the effective overlap is 1536, not the documented/configured 1792, so a maintainer editing noverlap to retune overlap would see no effect.

**Fix:** Remove `noverlap` from the docstring param list (line 76); since hop_length is the live knob, also remove noverlap from the settings block and test fixture, OR if a fixed overlap is the intended contract, derive hop_length from noverlap (hop_length = nperseg - noverlap) so the configured value takes effect. Removing is simpler.

### [LOW] dead_code_naming — Misleading module docstring: "JAX-native" port that contains no JAX
`generate_spectrograms.py:12-13`

The module docstring calls this the "in-house, JAX-native (torch-free) port" (lines 12-13), but there is no jax/jnp usage anywhere in the file (grep-confirmed); the math is plain librosa + numpy, as the docstring itself states one line later (line 14). 'JAX-native' mischaracterizes the implementation -- the accurate intent is 'torch-free'. A maintainer searching for a JAX implementation would be misled.

**Fix:** Reword lines 12-13 to drop 'JAX-native', e.g. 'the in-house, torch-free port of generate_spectrograms.py + spec_func.get_spec_librosa; the spectrogram math is plain librosa/numpy'.

### [LOW] performance — Invariant FFT-frequency axis and band mask recomputed per channel per USV
`generate_spectrograms.py:124-125`

freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=nperseg) and freq_mask = (freqs >= min_freq) & (freqs <= max_freq) (lines 124-125) are inside the per-channel loop (for ch_idx in range(n_channels)), and compute_usv_spectrogram is called once per USV. All four inputs (sampling_rate, nperseg, min_freq, max_freq) are constant for the entire session run, so this reallocates the same array and recomputes the same boolean mask n_channels times per USV across thousands of USVs. Hoisting above the loop is a pure win with no readability cost. NOTE: the candidate rated this medium; the per-call cost is tiny relative to the librosa.stft inside the same loop, so the real-world speedup is marginal -- low severity.

**Fix:** Compute freqs and freq_mask once immediately before the `for ch_idx in range(n_channels):` loop (after nperseg/min_freq/max_freq are read on lines 93-95) and reuse freq_mask inside the loop.

### [LOW] performance — Constant interpolation grids rebuilt on every frequency-resample
`generate_spectrograms.py:131-132`

freq_interp = np.linspace(0, 1, num_freq_bins) (line 131) is fully constant across all channels and USVs; freq_orig = np.linspace(0, 1, spec_db.shape[0]) (line 132) depends only on the band-masked bin count, also constant for the whole run. Both are rebuilt inside the per-channel resample branch (n_channels times per USV). They can be hoisted above the channel loop (freq_orig from int(freq_mask.sum())).

**Fix:** Precompute freq_interp once before the channel loop, and freq_orig once from int(freq_mask.sum()) before the loop; reuse both in the resample branch.

### [LOW] tests — Zero-variance uniform-weight fallback in compute_usv_spectrogram is untested
`generate_spectrograms.py:148-149`

When weights.sum() == 0 (all channels zero-variance but >= nperseg long) the code falls back to weights = np.ones_like(weights) (lines 148-149). No test exercises this: existing tests use random noise (test_compute_usv_spectrogram_shape_and_range) or too-short zeros that return None first (test_compute_usv_spectrogram_too_short_returns_none). The fallback line is never executed.

**Fix:** Add a test passing a constant-valued segment of shape (>=nperseg, n_channels), e.g. np.full((4096, 4), 5.0), so every channel passes the length guard but has zero variance; assert compute_usv_spectrogram returns a non-None (128, 128) array (proving np.average did not divide by zero weights).

### [LOW] tests — normalize=False branch of compute_usv_spectrogram is untested
`generate_spectrograms.py:152-156`

The normalize parameter's False path (lines 152-156 skipped, returning the raw variance-weighted dB spectrogram, which can be negative and unbounded) is never called: every test uses normalize=True (explicit or default). The un-normalized return is unvalidated.

**Fix:** Add a test calling compute_usv_spectrogram(segment, sr, _SPEC_PARAMS, normalize=False) on a noise segment and assert shape (128, 128) and that the result is NOT clamped to [0,1] (e.g. spec.min() < 0.0, since power_to_db ref=max yields <= 0 dB), confirming the normalization branch was bypassed.

### [LOW] tests — hop_length-is-None default (nperseg // 4) branch is untested
`generate_spectrograms.py:97`

Line 97 derives hop_length = nperseg // 4 when spec_params['hop_length'] is None. The fixture _SPEC_PARAMS always sets hop_length=512, so the else-default branch is never executed.

**Fix:** Add a test copying _SPEC_PARAMS with hop_length=None and assert compute_usv_spectrogram on a valid noise segment returns a non-None (128, 128) spectrogram.

### [LOW] tests — Empty-summary early return ('No USVs ... no spectrograms written') is untested
`generate_spectrograms.py:301-305`

The `if not all_specs:` early return (lines 301-305) is dead in the suite: test_generate_session_spectrograms_invalid_usv_is_placeholder calls _build_session(n_usv=0) but then overwrites the CSV with a 1-row DataFrame (lines 121-123), so usv_summary_df.height == 1 and all_specs always has one row. A genuinely empty (0-row) usv_summary.csv is never tested, so this branch and its message are uncovered.

**Fix:** Add a test writing a header-only usv_summary.csv (0 data rows), run generate_session_spectrograms, and assert (a) no <session>_spectrograms.h5 is created and (b) the 'No USVs in summary' message is emitted (capture via a list-appending message_output callback).

### [LOW] tests — Non-zero offset and sample-index clamping in generate_session_spectrograms are untested
`generate_spectrograms.py:279-283`

The fixture sets offset=0.0, so the padding (lines 280-281: t0=start-offset, t1=stop+offset) and clamps (lines 282-283: s0=max(0,...), s1=min(sample_num,...)) are never exercised with a value that triggers clamping. A USV near t=0 with positive offset (t0<0 -> s0 clamped) and one near the recording end (t1>length -> s1 clamped) are the boundary cases these guards exist for; neither is covered.

**Fix:** Add a test with offset>0 (e.g. 0.2 s) and a usv_summary containing a USV at start=0.0 (forces s0 clamp) and one whose stop+offset exceeds the 1 s recording (forces s1 clamp); assert the run completes and both rows produce valid (duration>0) spectrograms.

### [LOW] tests — first_match_or_raise error paths (missing CSV / missing mmap) are untested
`generate_spectrograms.py:235-247`

generate_session_spectrograms calls first_match_or_raise for *_usv_summary.csv (lines 235-240) and *.mmap (lines 243-247). Neither raise path is covered -- every test builds a complete session. A session missing either input should raise a clear error, currently unverified at this module's level.

**Fix:** Add a test building a session, deleting the *.mmap (or the *_usv_summary.csv), and asserting generate_session_spectrograms raises (pytest.raises) the expected error from first_match_or_raise.

### [LOW] tests — Mmap filename dtype/channel/sample/sr parsing is only exercised for int16
`generate_spectrograms.py:249-254`

Lines 249-254 parse data_type/channel_num/sample_num/audio_sampling_rate by splitting the .mmap filename on '_' and slicing [:-5] off the dtype token. The fixture always uses int16. The [:-5] slice strips exactly '.mmap' (5 chars) so it is correct for any dtype, but no test locks this in for a different dtype/channel-count encoding.

**Fix:** Add a parametrized test building the mmap with a float32 dtype and asserting spectrograms are produced and data_type/shape are read back correctly, locking the filename parsing across dtypes.


## `prepare_cluster_job.py` (12)

### [MEDIUM] correctness — iterdir() on a missing 'video' directory raises FileNotFoundError, aborting the whole batch and truncating job_list.txt
`prepare_cluster_job.py:84`

Line 84 `(pathlib.Path(root_dir) / 'video').iterdir()` has no existence guard. Since the output file is opened in 'w' mode at line 79 before the loop, a root_directory lacking a 'video' subdir raises FileNotFoundError mid-loop, leaving a truncated/partial job_list.txt and skipping all remaining roots. Realistic batch-run failure (typo'd root, partially-populated session).

**Fix:** Before iterating, guard: `video_root = pathlib.Path(root_dir) / 'video'; if not video_root.is_dir(): self.message_output(...); continue`.

### [MEDIUM] dead_code_naming — Dead/broken fallback: _settings['root_directory'] references a non-existent settings key
`prepare_cluster_job.py:44`

Confirmed: the prepare_cluster_job block in processing_settings.json has keys [camera_names, inference_root_dir, centroid_model_path, centered_instance_model_path] and NO 'root_directory'. The only caller (preprocess_data.py:185) always passes root_directory. So the `else _settings['root_directory']` branch on line 44 is unreachable in practice and would raise KeyError if ever reached, instead of providing a default.

**Fix:** Remove the `else _settings['root_directory']` fallback and make root_directory required, or add a 'root_directory' key to the prepare_cluster_job block. Removal preferred since every caller supplies it.

### [MEDIUM] tests — Default-settings load path (no args) is entirely untested
`prepare_cluster_job.py:39-41`

All four tests (test_process.py:1746-1808) pass both input_parameter_dict and root_directory. The line-39 branch that opens and json.loads processing_settings.json's prepare_cluster_job block is never executed; a broken relative path (line 40) or renamed key (line 41) would pass the suite.

**Fix:** Add a test constructing PrepareClusterJob() with no args (or one arg None) and assert input_parameter_dict / root_directory populate from the packaged JSON (e.g. 'camera_names' key present).

### [MEDIUM] tests — Non-.mp4 items inside a camera directory are never exercised
`prepare_cluster_job.py:89`

The session_tree fixture (1717-1732) writes only .mp4 files into camera dirs. The line-89 false branch (skipping a sibling .slp/.h5/.json) is uncovered. A regression widening the match (e.g. emitting a job line for the generated .slp result) would not be caught — notably relevant since line 91 derives a .slp path in the same directory.

**Fix:** Add a non-mp4 file (e.g. clip_a.slp, notes.txt) alongside the mp4s in the configured camera dir and assert job_list still has exactly 2 lines with no .slp/.txt input path.

### [MEDIUM] tests — Missing 'video' subdirectory under a root_directory has no error-path test
`prepare_cluster_job.py:84`

No test feeds a root_directory lacking a 'video' subdir, so the line-84 FileNotFoundError failure mode (see the correctness finding) is unspecified and uncovered.

**Fix:** Add a test with a root lacking a 'video' subdir asserting the intended behavior (currently raises FileNotFoundError; if the code is fixed to skip+message, assert an empty/partial job_list).

### [LOW] docs_clarity — __init__ Parameters listed out of signature order
`prepare_cluster_job.py:27-32`

Signature order is input_parameter_dict, root_directory, message_output (lines 16-18); docstring lists root_directory before input_parameter_dict (lines 27-30). Minor cross-reference mismatch.

**Fix:** Reorder docstring Parameters to match signature: input_parameter_dict, root_directory, message_output.

### [LOW] docs_clarity — Non-obvious settings-fallback logic has no explanatory comment
`prepare_cluster_job.py:39-44`

The load-on-either-None guard (line 39) paired with per-attribute ternaries (43-44) is non-obvious; a brief comment would clarify why _settings is loaded when only one arg is None and is intentionally undefined when both are supplied.

**Fix:** Add a one-line comment above line 39 explaining the independent-fallback rationale.

### [LOW] docs_clarity — Two-stage SLEAP model mapping (centroid/centered-instance -> first/second) lacks a comment
`prepare_cluster_job.py:71-75`

Lines 71-75 map centroid_model_path to 'first' and centered_instance_model_path to 'second', with an empty-string branch switching between single-model and two-model job lines (mirrored at 92-95). The SLEAP top-down two-stage convention encoded here is undocumented.

**Fix:** Add a comment near line 71 noting 'first' = centroid, 'second' = centered-instance for SLEAP top-down inference, and that an empty centered_instance_model_path emits a single-model job line.

### [LOW] docs_clarity — Module docstring omits the cluster-path conversion role
`prepare_cluster_job.py:1-4`

The module docstring (lines 1-4) only mentions creating the video list; the central behavior of converting local lab-share paths to cluster mount paths (to_cluster_path/configure_path) so job_list.txt is cluster-runnable is not mentioned.

**Fix:** Extend the module docstring to mention local-to-cluster path conversion via to_cluster_path / configure_path producing a cluster-ready job_list.txt.

### [LOW] performance — configure_path(inference_root_dir) computed twice
`prepare_cluster_job.py:77,79`

configure_path(self.input_parameter_dict['inference_root_dir']) is evaluated on line 77 (mkdir) and again on line 79 (open path), and re-wrapped in pathlib.Path twice. configure_path is non-trivial (calls _host_experimenter, _host_lab_shares, loops over the share table). Hoisting into one local removes the duplicate resolution and improves readability with no behavior change.

**Fix:** `inference_root = pathlib.Path(configure_path(self.input_parameter_dict['inference_root_dir']))` once, then use it for mkdir (line 77) and open (line 79).

### [LOW] tests — Empty root_directory list produces an empty job_list with no test
`prepare_cluster_job.py:80`

With an empty root_directory list, the line-80 loop never runs and job_list.txt is created empty. The empty-input edge (file still created at 79, zero-length) is untested.

**Fix:** Add a test with root_directory=[] asserting job_list.txt exists and read_text() == ''.

### [LOW] tests — isnumeric() exclusion of the non-numeric processed dir only asserted implicitly
`prepare_cluster_job.py:85`

The fixture creates video/non_numeric/ignored.mp4 (1728-1730); tests assert a line count of 2, excluding it implicitly, but no test directly asserts 'non_numeric'/'ignored.mp4' is absent from job_list.txt.

**Fix:** Add explicit assertion `assert 'non_numeric' not in job_list and 'ignored.mp4' not in job_list` to the one-line-per-mp4 test.


## `compute_usv_acoustic_features.py` (11)

### [MEDIUM] correctness — Mask-by-multiplication assumes non-negative spectrograms; breaks for dB-scaled (normalize=False) input
`compute_usv_acoustic_features.py:206-240`

compute_acoustic_features masks regions by elementwise multiplication (line 206: masked_specs = specs * region), zeroing every out-of-region/pad pixel. This is only correct when in-region spectrogram values are >= 0. generate_spectrograms writes librosa.power_to_db(power_spec, ref=np.max), i.e. dB values in roughly [-80, 0], and only shifts them to [0,1] when the user-configurable 'normalize' setting is True. Under normalize=False the in-region dB values are negative, so the 0.0 fill dominates: max_amplitude (line 240) returns ~0 from a padded pixel, peak_freq (lines 222-223) lands on a zeroed pixel, sum_region/freq_power can go negative corrupting mean_freq, bandwidth and freq_power_norm, and spectral_entropy takes log of a negative 'prob' -> NaN. There is no guard that specs are non-negative. NOTE: the candidate's claim that 'the module header here even warns spectrograms may be unnormalized' is FALSE -- the header (lines 1-28) contains no such warning. Also the default is normalize=True (function default and processing_settings.json:228), so the standard pipeline path is correct; this is a latent fragility only under the non-default normalize=False, which is why severity is medium not high.

**Fix:** Either validate input is non-negative (raise/warn if specs.min() < 0, or document that normalize=True is required and check it), or compute reductions over the boolean region as a selector rather than a multiplier: max via np.where(region.astype(bool), specs, -np.inf).max(...), argmax over the same, and accumulate sums only over true pixels. This makes the features correct for both normalized and dB spectrograms.

### [MEDIUM] docs_clarity — Stale docstring: merge is joined on _usv_row, not a non-existent spec_id
`compute_usv_acoustic_features.py:290`

merge_features_into_summary's docstring (line 290) states features are 'joined on the per-USV index encoded in each spec_id'. There is no spec_id anywhere in the code; the join key is _usv_row, the CSV row position: usv_indices = np.flatnonzero(durations > 0) (line 329), stored as _usv_row in features_df (line 366), matched against usv_df.with_row_index(name='_usv_row') (line 377) in the left join (line 378). The docstring describes a spec_id-parsing scheme that does not exist. (The test file header lines 7-8 repeat the same stale phrasing.)

**Fix:** Reword line 290 to describe the actual mechanism, e.g. '...written into the matching summary rows (joined on _usv_row, the positional index of each USV row in the summary CSV, 1:1 with the spectrogram rows). USVs absent from the H5 get null features.'

### [MEDIUM] tests — freq_bandwidth_hz value and clamp branch never asserted
`compute_usv_acoustic_features.py:225-229`

freq_bandwidth_hz comes from cumulative-energy crossings clamped via np.minimum to n_freq-1 (lines 227-229). test_compute_acoustic_features_peak_frequency only checks finiteness (line 49); no test pins a bandwidth value or exercises the clamp branch where cumsum never reaches high_energy_frac so the raw index would equal n_freq. An off-by-one or wrong-axis indexing in the band math would pass silently.

**Fix:** Add a test with a synthetic spec whose energy is concentrated in a known frequency band and assert freq_bandwidth_hz == freq_axis[high_bin]-freq_axis[low_bin] for the expected bins; add a case where freq_power rises monotonically so cumsum never crosses high_energy_frac, asserting high_bin clamps to n_freq-1.

### [LOW] correctness — All-False mask union yields silent zero-region features (no time-window fallback)
`compute_usv_acoustic_features.py:143-148`

build_mask_region_masks falls back to the time-window only when a USV has zero mask rows (mask_rows.size == 0, line 143). If a USV has >=1 mask row but their boolean union np.any(segmentations[mask_rows], axis=0) (line 144) is entirely False (a degenerate/empty segmentation), region_masks[i] stays all-zero and fallback_count is not incremented. Downstream that USV gets n_region=0 and the _EPS-floored degenerate features (mean_freq=0, peak_freq=freq_axis[0], max_amplitude=0, bandwidth=0, entropy=log(n_freq)) emitted as if valid. Likelihood depends on whether generate_masks can emit an all-False segmentation row.

**Fix:** After line 144 check region_masks[i].any(); if all-False, take the same time-window fallback branch (lines 146-147) and increment fallback_count, so a degenerate mask is treated like a missing one.

### [LOW] docs_clarity — Non-obvious flat-argmax-to-frequency-row arithmetic lacks an inline comment
`compute_usv_acoustic_features.py:222-223`

peak_freq = freq_axis[flat_argmax // n_time] (line 223) where flat_argmax is the argmax over the flattened (F, T) spectrogram. The line-221 comment explains the intent ('row of the loudest in-window pixel') but not that integer-dividing a flat (F*T) index by n_time recovers the frequency-row index. Aligns with the maintainer's documented preference for verbose explanatory docs.

**Fix:** Add an inline comment on line 223, e.g. '# `// n_time` maps the flat (F*T) argmax index back to its frequency-row index.'

### [LOW] docs_clarity — Cumulative-energy crossing via boolean-sum lacks an inline comment
`compute_usv_acoustic_features.py:227-228`

low_bin/high_bin = np.minimum((cumsum < frac).sum(axis=1), n_freq - 1) (lines 227-228) use the count of bins below the threshold as the index of the first crossing bin. The line-225 comment explains the feature ('span between cumulative-energy crossings') but not that the boolean .sum() locates the crossing index. Consistent with the verbose-docs preference.

**Fix:** Add an inline comment, e.g. '# count of bins below the threshold == index of the first bin whose cumulative energy crosses it (clamped to the last bin).'

### [LOW] performance — Redundant full (N,F,T) reduction for n_region; derivable from region_counts
`compute_usv_acoustic_features.py:208`

n_region = region_full.sum(axis=(1, 2)) (line 208) traverses the entire (N,F,T) broadcast array. Line 213 already computes region_counts = region_full.sum(axis=2) ([N,F]), and n_region is exactly region_counts.sum(axis=1) (O(N*F) instead of O(N*F*T)). Reordering line 213 above 208 lets both n_region and freq_power reuse region_counts.

**Fix:** Move region_counts = region_full.sum(axis=2) above line 208, replace line 208 with n_region = region_counts.sum(axis=1), and reuse region_counts on line 214.

### [LOW] tests — spectral_entropy and mean_freq_hz values never validated
`compute_usv_acoustic_features.py:232-242`

spectral_entropy (lines 232-233) and mean_freq_hz (line 219) are only checked for finiteness; no test pins their numeric values. A uniform freq-power profile has known maximal entropy (~log(F)) and centroid mean_freq; a single-row profile has near-zero entropy. A sign error or wrong reduction axis would go undetected.

**Fix:** Add a test feeding a uniform-power spec asserting spectral_entropy ~= log(n_freq) and mean_freq_hz ~= freq_axis mean; add a single-row-power spec asserting entropy ~= 0 and mean_freq == that row's frequency.

### [LOW] tests — Degenerate empty-region (_EPS guard) path untested
`compute_usv_acoustic_features.py:199-242`

compute_acoustic_features relies on the _EPS floor (lines 214, 215, 219, 232, 239) to stay finite when a region is entirely empty (n_region=0, sum_region=0, region_counts=0). No test passes an all-zero region_masks row, so the divide-by-(0+_EPS) and log(_EPS) branches preventing NaN/inf are never exercised; a regression removing an _EPS would not be caught.

**Fix:** Add a test calling compute_acoustic_features with region_masks containing one all-zero (F,T) mask and assert every FEATURE_COLUMNS value is finite, with mean_amplitude==0, max_amplitude==0, bandwidth==0.

### [LOW] tests — Integration mask path with fallback_count>0 untested
`compute_usv_acoustic_features.py:337-350`

merge_features_into_summary's mask branch (lines 337-350) emits the fallback message and calls build_mask_region_masks. test_merge_features_uses_mask_group_when_present uses a single fully-masked USV (fallback_count=0); the unit test test_build_mask_region_masks_unions_and_falls_back covers fallback only at the unit level. No end-to-end merge test exercises a mask group where a valid USV has no segmentation row (fallback>0).

**Fix:** Extend a merge test with 2 valid USVs where the mask group references only one; assert both rows get finite features and the unmasked USV's peak_freq matches the brightest pixel in its time-window fallback.

### [LOW] tests — CLI --low-energy-frac/--high-energy-frac override path not exercised
`compute_usv_acoustic_features.py:409-415`

compute_usv_acoustic_features_cli computes provided_params from ParameterSource.COMMANDLINE (line 409) and threads them through modify_settings_json_for_cli. test_compute_usv_acoustic_features_cli_routes invokes with only --root-directory and mocks modify_settings_json_for_cli, so the branch where the energy-frac options are supplied and must appear in provided_params is uncovered.

**Fix:** Add a CLI test invoking with --low-energy-frac/--high-energy-frac and assert modify_settings_json_for_cli was called with provided_params containing both keys.


## `preprocess_data.py` (11)

### [MEDIUM] docs_clarity — prepare_data_for_analyses docstring step list is stale and incomplete
`preprocess_data.py:118-138`

The numbered step list (1-20) omits six sub-steps the method body actually runs: prepare data for vocal assignment (line 308), assign vocalizations to mice (line 314), generate USV spectrograms (line 325), generate USV masks (line 331), compute USV acoustic features (line 337), and infer QLVM latents (line 343). None appear in the docstring. The ordering also no longer reflects execution: audio-video sync and the summary plot (lines 229-241) run before HPSS (250), filtering (256), mmap stacking (262), and the Anipose steps; and the ephys steps (18-20) execute in the separate all-directories branch (lines 163-194), not the per-directory loop.

**Fix:** Regenerate the step list to include the six missing sub-steps (vocalocator prep/assign, USV spectrograms, USV masks, acoustic features, QLVM latents) and reorder to reflect the two execution branches (all-at-once ephys/cluster/SLEAP vs. per-directory).

### [MEDIUM] tests — _stamp_processing_version mutation and guard branches are never directly tested
`preprocess_data.py:37-73`

Confirmed no test references _stamp_processing_version. It is only exercised indirectly with a bare tmp_path lacking any *_metadata.yaml, so only the early-return (lines 61-62) is covered. Uncovered: the happy-path mutation/round-trip (lines 70-73, including the '.dev' suffix-strip in metadata.version('usv-playpen').split('.dev')[0] and the atomic_output_path + SmartDumper write), the non-dict YAML guard (66-67), and the missing-key guard (68-69). A regression in the version-stamping key or dev-suffix split would pass the suite.

**Fix:** Add a focused unit test calling _stamp_processing_version directly: (1) write {session}_metadata.yaml with Session.usv_playpen_processing_version='v0.0.0', call, assert the key equals f"v{metadata.version('usv-playpen').split('.dev')[0]}" and other Session keys/order preserved; (2) write a top-level-list YAML and assert unchanged; (3) write metadata with no 'Session' (and one with Session but no version key) and assert unchanged. Optionally monkeypatch metadata.version to a '...dev...' string to assert the suffix strip.

### [LOW] correctness — anipose_trm else-branch only prints a message and is not recorded as a skipped/failed step
`preprocess_data.py:286-293`

When triangulate_arena_points_bool is False and len(experimental_codes) != len(root_directories), the requested coordinate transformation is silently skipped with only self.message_output(...), and nothing is appended to failed_preprocessing. The completion e-mail therefore reports unqualified success even though a requested step did not run. This is consistent with the maintainer's explicit failure-honesty machinery (the comment block at lines 355-357), so recording the skip is in keeping with intent. This is a borderline design observation (the skip is arguably a config-validation warning, not a processing error), hence low severity.

**Fix:** Append a descriptive entry to failed_preprocessing in the else branch, e.g. (one_directory, 'anipose_trm skipped: experimental_codes count does not match root directories'), so the completion summary reflects the skip.

### [LOW] dead_code_naming — Vestigial @click.pass_context/**kwargs/provided_params in sleap_file_conversion_cli (no tunable options)
`preprocess_data.py:730-750`

sleap_file_conversion_cli declares only --root-directory yet still uses @click.pass_context, **kwargs, and builds provided_params (line 744) which is always [] because there are no extra options. The modify_settings_json_for_cli call then just reloads unchanged processing_settings.json. prepare_vcl_assign_cli (line 1035) shows the simpler pattern with no ctx/kwargs. Confirmed the command is a live entry point (pyproject.toml:125 sleap-to-h5); only the internal param-collection plumbing is dead.

**Fix:** Drop @click.pass_context/**kwargs/provided_params and load processing_settings.json directly as prepare_vcl_assign_cli does, or add a one-line comment noting the no-op if kept for cross-CLI consistency.

### [LOW] docs_clarity — Double space and 'IR lights pulses' typo in step (16) description
`preprocess_data.py:134`

Line 134 reads 'checks audio-video sync using  Arduino-controlled IR lights pulses' with a double space after 'using' and the awkward 'IR lights pulses'.

**Fix:** Reword to 'checks audio-video sync using Arduino-controlled IR light pulses' (single space, 'light').

### [LOW] docs_clarity — __init__ docstring lists parameters out of signature order
`preprocess_data.py:89-98`

Signature (lines 78-82) is (exp_settings_dict, input_parameter_dict, root_directories, message_output) but the Parameters section documents exp_settings_dict, root_directories, input_parameter_dict, message_output. input_parameter_dict is described tersely as 'Analyses parameters' even though for this preprocessing class it carries processing booleans and module settings. Note: all four params ARE documented (no omission), contrary to the title's 'omits exp_settings_dict' wording, which is incorrect.

**Fix:** Reorder the Parameters entries to match the signature and clarify that input_parameter_dict holds processing booleans and module-specific settings.

### [LOW] docs_clarity — Comment 'configure video properties via ffmpeg' does not scope the concat/fps blocks
`preprocess_data.py:205`

The comment at line 205 sits above the concatenation block (206-209) but the separate fps-rectification block (211-214) has no comment of its own; the generic phrasing covers neither cleanly.

**Fix:** Add '# # # concatenate video files' above line 206 and '# # # rectify video fps (re-encode)' above line 211, or reword the existing comment to scope the concatenation block.

### [LOW] tests — All-at-once (ephys/cluster) exception-recording branch is untested
`preprocess_data.py:192-194`

test_prepare_data_records_step_failure (test_process.py:2951) covers only the per-directory except (lines 351-353) via hpss_audio.side_effect. The all-at-once try/except (lines 164-194) handling concatenate_binary_files / split_clusters_to_sessions / video_list_to_txt has its except at lines 192-194 untested; the distinct failure label 'ephys/cluster preprocessing (all root directories)' and its completion-email path are never validated. test_prepare_data_all_at_once_split_and_sleap_cluster (2881) covers only the happy path.

**Fix:** Add a test mirroring test_prepare_data_records_step_failure for the all-at-once branch: set conduct_ephys_file_chaining=True, mock_dependencies['Operator'].return_value.concatenate_binary_files.side_effect=RuntimeError('boom'), run prepare_data_for_analyses(), and assert the completion Messenger.send_message subject contains 'failure'.

### [LOW] tests — multichannel-to-single-channel non-empty 'audio/original' skip guard is not covered
`preprocess_data.py:217-221`

test_prepare_data_all_per_directory_steps_dispatch (test_process.py:2840) creates an EMPTY audio/original so the len(...iterdir())==0 guard fires and multichannel_to_channel_audio is called. The opposite case (audio/original already populated -> conversion SKIPPED) is never tested, so a regression inverting/dropping the guard (re-running the expensive split over existing output) would not be caught.

**Fix:** Add a test enabling conduct_audio_multichannel_to_single_ch with a non-empty audio/original (touch a dummy .wav), run prepare_data_for_analyses(), and assert mock_dependencies['Operator'].return_value.multichannel_to_channel_audio.assert_not_called().

### [LOW] tests — anipose_trm triangulate_arena_points_bool=True arm is untested
`preprocess_data.py:287-288`

The dispatch condition at lines 287-288 is an OR. test_prepare_data_all_per_directory_steps_dispatch (2855) satisfies only the experimental_codes-match arm (['E1M'] for one dir). test_prepare_data_trm_experimental_code_mismatch_logs (2914) sets both arena_points_bool=False and experimental_codes=[]. No test exercises the triangulate_arena_points_bool=True arm with mismatched codes, so dropping that arm would still pass.

**Fix:** Add a test enabling anipose_trm with triangulate_arena_points_bool=True and intentionally mismatched experimental_codes (e.g. [] with one root dir), then assert mock_dependencies['ConvertTo3D'].return_value.translate_rotate_metric was called once.

### [LOW] tests — concatenate_binary_files_cli / split_clusters_to_sessions_cli empty-valid_dirs branch is untested
`preprocess_data.py:984-990`

Both multi-root ephys CLIs filter valid_dirs by pathlib.Path(path).is_dir() and only stamp/dispatch when len(valid_dirs)>0 (lines 984-990 and 1023-1030). test_preprocess_cli_commands_dispatch (test_process.py:3046-3052) only exercises the happy path with a real tmp_path. The false branch (all paths non-existent -> valid_dirs empty -> neither _stamp_processing_version nor Operator runs) is never tested.

**Fix:** Add a test invoking concatenate_binary_files_cli (and split_clusters_to_sessions_cli) with --root-directories pointing at a non-existent path; assert exit_code==0 and mock_dependencies['Operator'].assert_not_called().


## `qlvm_latents.py` (10)

### [MEDIUM] docs_clarity — infer_and_merge docstring claims join on per-USV index in spec_id, but code joins on CSV row index
`qlvm_latents.py:193-194`

The docstring says merged '(joined on the per-USV index in each ``spec_id``)'. The code never references spec_id; spec_id is a training-set artifact built only in build_qlvm_training_set.py and the per-session *_spectrograms.h5 read here has no such field. The merge derives positional indices via usv_indices = np.flatnonzero(durations > 0) (line 232) and joins on _usv_row from with_row_index (lines 259-260).

**Fix:** Reword to describe the actual mechanism: joined on the positional USV row index (spectrogram rows 1:1 with usv_summary.csv rows); drop the spec_id reference.

### [MEDIUM] tests — build_lattice 'fibonacci' branch is never exercised
`qlvm_latents.py:99-100`

test_build_lattice_korobov_and_roberts only covers 'korobov' and 'roberts'. The 'fibonacci' branch (qlvm_latents.py:99-100, gen_fib_basis(cfg['fib_m'])) is never reached via build_lattice; the end-to-end tests set fib_m but always use lattice_type='korobov'.

**Fix:** Add to the lattice test: fib = ql.build_lattice({'lattice_type': 'fibonacci', 'fib_m': 8}); assert fib.shape == (21, 2)  # fib(8)=21.

### [LOW] correctness — NpzFile handle leak in load_decoder_params
`qlvm_latents.py:68`

`raw = np.load(configure_path(weights_npz_path))` opens a zip-backed NpzFile holding an OS file handle. The function builds `params` by copying each array out via `jnp.asarray(raw[key])` but never calls `raw.close()` nor uses a `with` block, so the handle leaks until GC. Since all arrays are materialized into `params`, closing right after the loop is safe.

**Fix:** Wrap in `with np.load(configure_path(weights_npz_path)) as raw:` and build params inside the block (return after).

### [LOW] correctness — Two leaked NpzFile handles for the reference watershed grids
`qlvm_latents.py:215-218`

`fine_ref` and `coarse_ref` from `np.load(...)` (lines 215-216) are NpzFile objects holding OS file handles. Only `ws_labels_periodic` is extracted (lines 217-218, fully materialized on access) and neither NpzFile is closed, leaking two handles per `infer_and_merge` call.

**Fix:** Use context managers: `with np.load(configure_path(cfg['reference_arrays_fine_npz_path'])) as fine_ref: fine_grid = fine_ref['ws_labels_periodic']` (likewise coarse).

### [LOW] docs_clarity — labels_for_coords docstring cites a nonexistent inference_latents.py
`qlvm_latents.py:114-115`

The docstring attributes the lookup convention to ``inference_latents.py``'s convention `label = grid[int(y*res), int(x*res)]`. No file named inference_latents.py exists in the repo (grep finds the name only in this docstring and a historical mention in qlvm_torus_traversal_video.py). The convention itself is correctly documented inline.

**Fix:** Drop or correct the parenthetical file attribution; the inline convention description suffices.

### [LOW] docs_clarity — Docstring says USVs 'absent from the H5' get nulls, but rows are present and filtered by duration>0
`qlvm_latents.py:230-231`

The comment at 230-231 ('embed only the real (duration > 0) USVs') is accurate, but the docstring at 194 says 'USVs absent from the H5 get nulls'. The rows are NOT absent from the H5 — they are present but skipped by the duration > 0 filter (line 232), so the docstring is imprecise about why those rows receive nulls.

**Fix:** Tighten the docstring at line 194 to 'USVs with non-positive duration are skipped and get nulls' instead of 'absent from the H5'.

### [LOW] tests — build_lattice ValueError on unknown lattice_type is untested
`qlvm_latents.py:101-102`

The error path at qlvm_latents.py:101-102 (raise ValueError 'unknown lattice_type ... expected korobov|roberts|fibonacci') has no test. It is the only explicit input-validation guard in the module and is uncovered.

**Fix:** Add: with pytest.raises(ValueError, match='unknown lattice_type'): ql.build_lattice({'lattice_type': 'spiral', 'latent_dim': 2, 'n_points': 16}).

### [LOW] tests — load_decoder_params no-prefix branch is untested
`qlvm_latents.py:71`

test_load_decoder_params_strips_prefix only feeds 'decoder.'-prefixed keys, exercising only the strip side of the conditional at qlvm_latents.py:71. The else branch (key without 'decoder.' prefix kept verbatim) is untested.

**Fix:** Add a test with un-prefixed keys: np.savez(p, **{'0.weight': np.ones((4,4)), '0.bias': np.zeros(4)}); assert set(load_decoder_params(str(p))) == {'0.weight', '0.bias'}.

### [LOW] tests — infer_and_merge zero-real-USVs edge case (all durations==0) is untested
`qlvm_latents.py:232-234`

All end-to-end tests use durations [128, 0, 128]. The boundary where np.flatnonzero(durations > 0) is empty (line 232) — a session with no real USVs, yielding an empty embed_data batch and all-null qlvm_* columns via the left join — is never tested.

**Fix:** Add a test with durations np.array([0,0,0]); assert output df.height==3, QLVM_COLUMNS present, every qlvm_dim1 is None.

### [LOW] tests — labels_for_coords clipping at coords==1.0 / out-of-range is untested
`qlvm_latents.py:141-142`

labels_for_coords clips px/py into [0, res-1] (lines 141-142) so a coord at/just above 1.0 (float seam edge) does not index out of bounds. test_labels_for_coords_lookup_convention only uses interior coords in [0, 0.9], so the clip upper-bound guard is never triggered. embed_data returns torus coords in [0,1) but float edge cases at the seam make this guard load-bearing.

**Fix:** Extend the lookup test with a boundary coord [1.0, 1.0] and assert it clips to grid[res-1, res-1] for both grids (no IndexError, correct clamp).


## `generate_masks.py` (10)

### [MEDIUM] tests — generate_masks_cli command is entirely untested
`generate_masks.py:372-409`

The Click command generate_masks_cli (lines 372-409) has zero coverage: grep for 'generate_masks_cli'/'generate-usv-masks' across tests/ returns nothing. The CLI plumbing is unverified -- that COMMANDLINE-sourced options become provided_params (line 397), that modify_settings_json_for_cli is wired with settings_dict='processing_settings' (lines 399-403), and that overrides reach MaskGenerator.generate_session_masks. CliRunner is an established pattern in this repo.

**Fix:** Add a CliRunner test invoking generate_masks_cli with --root-directory plus overrides (e.g. --detector cc, --yolo-conf 0.1), monkeypatching modify_settings_json_for_cli or MaskGenerator.generate_session_masks to capture the call; assert the overrides forward and that a nonexistent --root-directory yields exit_code != 0.

### [LOW] correctness — Unconditional configure_path(yolo_weights) crashes the documented cc fallback when weights are null
`generate_masks.py:208`

Line 208 runs `yolo_weights = configure_path(cfg['yolo_weights'])` unconditionally, before the detector branch. configure_path (os_utils.py:306) immediately does `if "{experimenter}" in pa`, which raises TypeError if pa is None. The yolo_weights file-existence validation at line 239 is correctly guarded by `detector == 'yolo'`, but configure_path runs first regardless. A user selecting detector='cc' (the documented no-weights fallback, module docstring line 17, CLI Choice line 374) and setting yolo_weights to JSON null gets an opaque TypeError instead of a clean run. The shipped default has yolo_weights as a real templated string, so this only bites a user who nulls it; hence low.

**Fix:** Only translate yolo_weights when needed, e.g. `yolo_weights = configure_path(cfg['yolo_weights']) if (detector == 'yolo' and cfg['yolo_weights']) else cfg['yolo_weights']`. Note the cc test already passes yolo_weights='' (empty string), which configure_path tolerates; only JSON null/None triggers the crash.

### [LOW] correctness — Reported valid-USV count uses duration>0 but the kernel segments only duration>=duration_min
`generate_masks.py:261-263`

Line 261 computes `valid_count = int(np.count_nonzero(durations > 0))` and line 263 prints 'Segmenting {valid_count} valid USVs (of {num_specs})'. The kernel process_session_batch_boxprompt processes rows with `d >= duration_min` (boxprompt_utils.py:376; default duration_min=10 per processing_settings.json:241). Any USV with 0 < duration < 10 is counted in this start message but silently skipped by the kernel, so the up-front count overstates what is actually segmented. Messaging-only mismatch; the final 'Wrote ... across N USVs' message (lines 363-366) reports the true count.

**Fix:** Match the kernel threshold: `valid_count = int(np.count_nonzero(durations >= cfg['duration_min']))`.

### [LOW] docs_clarity — Positional None (detector_cfg) in process_session_batch_boxprompt call has no explanatory comment
`generate_masks.py:313-318`

The call passes a bare positional `None` (line 318) as the kernel's 5th positional argument, which is `detector_cfg` (boxprompt_utils.py:335). The comment block at lines 310-312 explains the specs/durations arguments but says nothing about this None, and a reader skimming the call cannot tell what the 4th positional after `predictor` means. On the cc path the kernel uses detector_cfg's default; on the yolo path detect_fn is used instead.

**Fix:** Pass by keyword (`detector_cfg=None`) or add a one-line inline comment naming it.

### [LOW] docs_clarity — generate_session_masks docstring omits the cc detector branch
`generate_masks.py:165-182`

The method docstring (lines 165-167) describes the flow as 'runs the YOLO box detector + SAM2 box-prompt segmenter', stating YOLO unconditionally. But the code branches on detector (lines 299-308): for 'cc' no detect_fn is built and the kernel falls back to its connected-component detector. The module docstring (line 17) documents cc, so the method docstring is inconsistent with both the module docstring and the actual two-branch behavior.

**Fix:** Add a sentence noting the detector is selectable: 'yolo' builds a learned detect_fn; 'cc' builds no detect_fn and the kernel uses its connected-component detector.

### [LOW] performance — Double full-array allocation in flatten_session_masks (per-mask frame + np.stack copy)
`generate_masks.py:96-115`

For each mask the loop allocates a fresh (F, T) zero frame (line 102), writes into it, and appends to seg_rows; after the loop np.stack (line 109) allocates the full (M, F, T) output and copies each frame in a second time. Every mask is materialized twice and M intermediate frames plus the Python list are held alive until the stack, doubling peak memory and copy traffic for the flatten step. M is cheaply computable up front so the output can be preallocated and filled in place.

**Fix:** Compute M = sum(len(processed_masks[r]) for r in processed_masks); allocate segmentations = np.zeros((M, num_freq_bins, num_time_bins), dtype=bool) and spectrogram_index = np.empty((M,), dtype=np.int64); in the sorted-keys loop write segmentations[i, :, :valid_cols] = seg[:, :valid_cols] and spectrogram_index[i] = spec_row, incrementing i. Keep the (0, F, T) empty case.

### [LOW] tests — flatten_session_masks column-clamping branch (mask wider than num_time_bins) untested
`generate_masks.py:103-104`

Line 103 `valid_cols = min(seg.shape[1], num_time_bins)` and line 104 `frame[:, :valid_cols] = seg[:, :valid_cols]` clamp masks wider than the stored width. All canned segmentations use _seg widths 25/40/60 with _N_TIME=128, so seg.shape[1] > num_time_bins (the truncating side of min and the slice on the right of line 104) is never exercised. A bug swapping the clamp would not be caught.

**Fix:** Add a case where a mask exceeds num_time_bins, e.g. flatten_session_masks({0: [{'segmentation': _seg(200)}]}, _N_FREQ, 128) and assert segmentations.shape == (1, 128, 128) and segmentations[0, 10, :].all() with no broadcast error.

### [LOW] tests — Empty-session write branch in generate_session_masks (uncompressed datasets) untested end-to-end
`generate_masks.py:359-361`

When no masks are produced, lines 359-361 take the else branch writing uncompressed empty datasets with total_masks=0 (line 351), and the final message (lines 363-366) calls len(np.unique(spectrogram_index)) on a zero-length array. test_flatten_session_masks_empty covers only the standalone flatten function; no MaskGenerator run feeds an all-empty processed_masks dict, so this orchestrator branch and the np.unique-on-empty edge are never executed.

**Fix:** Add a test with durations [0, 0] and canned={0: [], 1: []}, install fake kernels, run generate_session_masks, then assert the mask group exists, total_masks attr == 0, segmentations.shape == (0, F, T), spectrogram_index.shape == (0,), and that it completes without error.

### [LOW] tests — first_match_or_raise missing-H5 failure path untested
`generate_masks.py:250-254`

generate_session_masks reads the spectrogram H5 via first_match_or_raise (lines 250-254), which raises when no *_spectrograms.h5 exists. All tests pre-create the H5 via _make_session_h5, so the missing-H5 branch at this call site is never hit.

**Fix:** Add a test creating only root/audio/spectrograms (no .h5) with valid model paths and fake kernels installed, then assert generate_session_masks raises with the 'per-session spectrogram H5' label.

### [LOW] tests — os.chdir round-trip and detect_fn=None (cc) reaching the kernel not asserted
`generate_masks.py:281-308`

Two orchestration details run but are never asserted. (1) Lines 287-297 chdir into sam2_model_dir, build the predictor, and restore cwd in a finally; build_predictor is faked, so the chdir/finally restore executes but is never checked -- a regression that failed to restore cwd would pass silently. (2) test_generate_session_masks_cc_detector_skips_yolo_weights confirms the cc run writes a group but never asserts detect_fn is None reaches process_session_batch_boxprompt (line 330), i.e. that detector!='yolo' truly skips get_detector.

**Fix:** In the cc test, capture kwargs in the fake process_session_batch_boxprompt and assert detect_fn is None. Add an assertion that Path.cwd() is unchanged after generate_session_masks (including a variant where build_predictor raises) to pin the finally-restore.


## `extract_phidget_data.py` (10)

### [MEDIUM] tests — None-default __init__ branch untested; line 49 references a 'root_directory' key absent from settings.json
`extract_phidget_data.py:43-54`

Every test builds Gatherer via _gatherer() (test line 32-33) passing BOTH args, so the None-default branch (lines 43-47, 49, 52-53) is never exercised. Line 49 falls back to _settings["root_directory"], but processing_settings.json['extract_phidget_data'] contains only a 'Gatherer' key (verified: no 'root_directory'). Thus Gatherer(root_directory=None, input_parameter_dict=...) raises an opaque KeyError at line 49. Production callers (preprocess_data.py lines 231, 561) always pass root_directory, so this is a latent dead-branch defect, not a live production bug.

**Fix:** Either add the missing 'root_directory' default to the settings JSON (if a default is intended) or remove the dead fallback at line 49; add a test pinning whichever behavior is chosen, plus a test for Gatherer(input_parameter_dict=None, root_directory=str(tmp_path)) covering lines 44-47/53.

### [LOW] correctness — KeyError if any phidget record lacks 'sensor_time'
`extract_phidget_data.py:118-121`

sorted(..., key=itemgetter("sensor_time")) at lines 119-121 raises an unhandled KeyError if any loaded record lacks the 'sensor_time' key, unlike the downstream sensor keys hum_h/lux/hum_t at lines 131-136 which are guarded with 'in'. This is inconsistent with the file's otherwise deliberate error-path hardening (lines 87-88, 98-101, 109-112).

**Fix:** Either document that 'sensor_time' is guaranteed on every record, or validate/skip records missing the key before sorting and raise a descriptive ValueError naming the offending file.

### [LOW] docs_clarity — Cryptic sensor-key-to-output mapping (hum_t -> temperature) lacks a comment
`extract_phidget_data.py:130-136`

The raw record keys map non-intuitively: 'hum_h' -> humidity, 'hum_t' -> temperature (NOT humidity despite the 'hum_' prefix), 'lux' -> lux. The 'hum_t' = temperature mapping is genuinely misleading by name. The loop at lines 130-136 has no comment explaining this; the only documentation of the mapping lives in the test helper (_realistic_record, test lines 65-73), not the source.

**Fix:** Add a brief comment above the loop noting that 'hum_h' is humidity (%), 'hum_t' is temperature (C) from the same combined humidity/temperature sensor, and 'lux' is illumination.

### [LOW] docs_clarity — Class docstring 'aligned per-sample arrays' overstates cross-sensor alignment
`extract_phidget_data.py:19-20`

The class docstring (lines 19-20) says logs are returned 'as aligned per-sample arrays'. The implementation (lines 124-136) builds three NaN-filled arrays indexed by sort order of 'sensor_time' and fills each index only from whichever keys are present in that record (independent 'if' guards). Values are NaN where a sensor key is absent; there is no guaranteed synchronized lux/temperature/humidity triple per index. 'Aligned per-sample' could mislead a reader into assuming every index has all three sensor values.

**Fix:** Reword to clarify the arrays are index-aligned by ascending sensor_time with NaN where a sensor value was absent in that record.

### [LOW] docs_clarity — Sort comment is vague ('particular dictionary key') and omits rationale
`extract_phidget_data.py:118`

The comment at line 118 'sort phidget_data by particular dictionary key' names no key and gives no reason. Since this sort by 'sensor_time' establishes the chronological index alignment of the returned arrays (and multi-file loads at lines 113-116 are concatenated in filename order, not time order), the rationale is load-bearing.

**Fix:** Reword to: '# Sort records by their acquisition timestamp (sensor_time) so exported arrays are chronological; multi-file loads are concatenated in filename order, not time order.'

### [LOW] docs_clarity — __init__ Parameters list order mismatches signature and omits required dict nesting
`extract_phidget_data.py:33-36`

The Parameters block (lines 33-36) lists root_directory before input_parameter_dict, but the signature (line 24) is input_parameter_dict, root_directory. More substantively, input_parameter_dict is described only as 'Processing parameters' while the code at lines 51 and 89 requires a specific nesting: ['extract_phidget_data']['Gatherer']['prepare_data_for_analyses']['extra_data_camera']. Given the project's detailed-docstring convention, the expected nesting is worth documenting.

**Fix:** Reorder the Parameters entries to match the signature and note that input_parameter_dict must contain ['extract_phidget_data']['Gatherer'] (with a 'prepare_data_for_analyses' sub-dict providing 'extra_data_camera').

### [LOW] tests — Asymmetric one-None __init__ combination untested
`extract_phidget_data.py:49-54`

The two ternaries at lines 49 and 50-54 resolve independently, so a caller can provide one arg and leave the other None. No test covers root_directory provided + input_parameter_dict=None (which loads _settings['Gatherer'] at line 53, works) nor the inverse (which hits the KeyError covered by the prior finding). The per-argument fallthrough is otherwise unverified.

**Fix:** Add a test Gatherer(input_parameter_dict=None, root_directory=str(tmp_path)) asserting input_parameter_dict resolves to the JSON 'Gatherer' block while root_directory uses the provided value.

### [LOW] tests — Substring camera match and sorted-first selection of multiple matching dirs untested
`extract_phidget_data.py:90-97`

Line 94 selects via substring (extra_data_camera in one_dir.name) and line 93 iterates sorted(...), so the lexicographically-first matching dir wins deterministically. Tests only ever create one camera dir, so neither the multi-match sorted-first determinism nor a true substring (non-equal) match is verified.

**Fix:** Add a test with two matching dirs (e.g. 20250101_120000.21372315 and 20250101_130000.21372315) holding distinct records and assert the records from the lexicographically-first dir are returned.

### [LOW] tests — Stray-only camera dir (no *extra_data* json) raising 'No phidget' is untested
`extract_phidget_data.py:108-112`

test_stray_non_extra_data_json_is_ignored (test lines 91-98) covers a stray .json alongside a real phidget file. test_empty_json_set_raises (test lines 107-110) uses make_json=False (zero files). Neither covers the case where the camera dir contains only a stray non-'*extra_data*' .json: the glob at line 108 returns empty and lines 109-112 must raise 'No phidget'. That glob-filters-to-empty path is unverified.

**Fix:** Add a test creating a camera dir containing only frame_metadata.json (no *extra_data*.json) and assert FileNotFoundError matching 'No phidget'.

### [LOW] tests — Empty-but-valid record list ([]) yielding zero-length arrays is untested
`extract_phidget_data.py:124-136`

A phidget file containing valid [] makes phidget_data_sorted empty (line 119), so lines 124-128 build three length-0 arrays and the loop at 130 never runs, returning empty arrays rather than erroring. This file-present-but-no-records boundary differs from test_empty_json_set_raises (no files) and is unpinned.

**Fix:** Add a test with _make_session(tmp_path, [[]]) and assert the result has keys {humidity, lux, temperature} each of length 0.


## `train_qlvm.py` (9)

### [MEDIUM] docs_clarity — Module docstring says the torus shift is per-epoch, but it is per-batch (contradicts the train() docstring)
`train_qlvm.py:25-26`

Lines 25-26 state 'Each epoch applies a fresh random torus shift to the whole lattice (the QMC integration trick).' This is stale: train_epoch (loop.py:15-23) calls model(base_sequence, random=True, mod=True) once per batch inside the for-batch loop, so a fresh shift is applied every BATCH. The train() docstring at line 233 correctly says 'a fresh random torus shift every batch', so the two docstrings in this file directly contradict each other.

**Fix:** Change line 25-26 to 'Each batch applies a fresh random torus shift to the whole lattice (the QMC integration trick).' to match the per-batch shift in train_epoch and the wording at line 233.

### [LOW] correctness — np.random.seed is vestigial for the training RNG path; DataLoader workers are unseeded when num_workers>0
`train_qlvm.py:297-298`

torch.manual_seed(seed) (line 297) plus np.random.seed(seed) (line 298) are the only seeding. Verified against qmc_base.py: QMCLVM.__init__ defaults shift_function=torch.rand and forward() calls self.shift_function(...), so the per-batch torus shift draws from torch's global RNG. The train_loader uses shuffle=True with no explicit generator (lines 309-311), so shuffling also uses torch's global RNG. Nothing in the training path consumes numpy's global RNG, so np.random.seed is a no-op for reproducibility. Separately, when cfg['num_workers']>0 (line 263, forwarded at 310/318), each worker process gets its own un-seeded RNG with no worker_init_fn/generator, so runs are not reproducible with multiple workers. Silent gap, not a crash.

**Fix:** Drop the vestigial np.random.seed(seed) (or document it has no effect on the training RNG), and to fix worker reproducibility pass an explicit torch.Generator().manual_seed(seed) to the train DataLoader plus a worker_init_fn seeding per worker.

### [LOW] correctness — np.mean(batch_losses) yields a silent nan if a split produces zero batches
`train_qlvm.py:342-347`

train_epoch (loop.py:14-33) initializes epoch_losses=[] and only appends inside the for-batch loop, so it returns [] when the loader yields no batches (an empty/near-empty split). np.mean([]) returns nan with a RuntimeWarning rather than failing, so the per-epoch line at 342-343/347 prints 'train evidence loss nan' and training proceeds on a model that saw no data. n_train is computed at line 308 but never validated to be > 0 before constructing the loader.

**Fix:** After loading, guard n_train > 0 (ideally at least one batch) right after line 312 and raise a clear ValueError if the training split is empty, turning the silent nan into an actionable error.

### [LOW] correctness — Single korobov_a is reused for both train_n_points and test_n_points, producing a mismatched test lattice
`train_qlvm.py:324-325`

Lines 324-325 forward the same cfg['korobov_a'] to build_lattice for both train_n_points and test_n_points. gen_korobov_basis's docstring (sampling.py:62-68) states the generating integer a is specific to the point count (1021->76, 2039->1487, 4093->1516). The shipped train_qlvm config (processing_settings.json:273-275) uses korobov_a=76 with train_n_points=1021 (matched) but test_n_points=2039 (recommended a=1487). The test/validation Korobov lattice is therefore poorly equidistributed, biasing the reported validation evidence. There is no API hook to specify the correct a for the test lattice.

**Fix:** Add a test_korobov_a setting (or derive the recommended a from the point count) so the test lattice uses an a matched to test_n_points; at minimum document that korobov_a must be valid for BOTH train_n_points and test_n_points.

### [LOW] dead_code_naming — CLI lattice-type token 'fib' is inconsistent with the inference path's required 'fibonacci'
`train_qlvm.py:375`

The CLI declares --lattice-type as click.Choice(['korobov','roberts','fib']) (line 375). This file's build_lattice (lines 129-139) treats any non-korobov/non-roberts value as the Fibonacci fallback, so 'fib' trains fine. The JAX inference build_lattice in qlvm_latents.py:99 accepts only the literal 'fibonacci' and raises ValueError otherwise. Note the train_qlvm and infer_qlvm_latents config blocks have SEPARATE lattice_type keys (processing_settings.json:272 vs 302), so this is not a literal single-value round-trip; the real defect is the token inconsistency (train side wants 'fib', inference side wants 'fibonacci'), forcing the user to use two different spellings for the same lattice family across the pipeline.

**Fix:** Change the CLI choice token from 'fib' to 'fibonacci' (click.Choice(['korobov','roberts','fibonacci'])) and dispatch this file's build_lattice on lattice_type == 'fibonacci' explicitly (matching qlvm_latents) so a typo'd value fails loudly instead of silently selecting Fibonacci.

### [LOW] tests — _load_split masks_len-absent fallback branch is untested
`train_qlvm.py:216`

Line 216's ternary `masks_len = data['masks_len'].astype(...) if 'masks_len' in data else np.zeros(specs.shape[0], dtype=np.int64)` has an else-branch that is never exercised: the test helper _write_training_npz (test_train_qlvm.py:56-63) always writes a masks_len array, so an npz lacking masks_len only hits this path in production.

**Fix:** Add a test that writes an .npz with spectrograms but no masks_len key, call QLVMTrainer(...)._load_split(path) directly, and assert it returns a TensorDataset of the right length with an all-zeros label tensor of shape (n_samples,).

### [LOW] tests — build_lattice Fibonacci success path (gen_fib_basis) is never invoked by tests
`train_qlvm.py:139`

test_build_lattice_shapes (test_train_qlvm.py:77-82) covers only 'korobov' and 'roberts'; test_build_lattice_fib_requires_2d (122-126) covers only the ValueError when latent_dim != 2. The actual Fibonacci return at line 139 (return gen_fib_basis(m=fib_m), reached when lattice_type is neither korobov nor roberts AND latent_dim == 2) is never executed, so the fib success path and its (n_points, 2) shape are unverified.

**Fix:** Add an assertion (e.g. to test_build_lattice_shapes) calling build_lattice('fib', latent_dim=2, korobov_a=3, n_points=17, fib_m=5) and asserting the returned tensor's shape[1] == 2.

### [LOW] tests — val_freq > 1 epoch-gating and final-epoch fallback are untested
`train_qlvm.py:339-348`

_TINY_CFG (test_train_qlvm.py:30-45) always sets val_freq=1 and n_epochs=2, so (epoch+1)%val_freq==0 is True every epoch in every training test. The branch where the per-epoch message is skipped because (epoch+1)%val_freq != 0 but still emitted on the final epoch via 'or epoch == n_epochs - 1' (lines 339 and 345) is never reached, for both the val-loader and no-val-loader branches.

**Fix:** Add a test with val_freq=2, n_epochs=3 (with a val split) capturing messages via a list-appending message_output, asserting log lines appear only for the gated/final epochs; mirror it without val_data.npz to cover the line-345 elif under val_freq>1.

### [LOW] tests — Checkpoint contents and decoder-weight value fidelity are asserted only by file existence
`train_qlvm.py:352-362`

test_train_writes_checkpoint_and_bridge_weights (test_train_qlvm.py:85-119) asserts the checkpoint .tar exists (line 104) but never torch.loads it to confirm the model/optimizer/loss-trajectory round-trips (save(...) at line 352 with all_losses, accumulated at line 337). The weights .npz keys are checked (110-111) but the values are not compared against the live model.decoder.state_dict() (line 357), so a silent corruption in the detach/cpu/numpy conversion at 359-362 would still pass.

**Fix:** torch.load the checkpoint and assert it carries a non-empty loss trajectory plus model/optimizer state; separately assert a couple of exported .npz arrays allclose the live decoder.state_dict() values to lock the numeric train->JAX bridge.


## `load_audio_files.py` (9)

### [HIGH] tests — struct.error sox-repair branch is entirely untested
`load_audio_files.py:133-167`

The malformed-header recovery path has zero coverage: lines 133-148 (catch struct.error, spawn static_sox), 149-155 (RuntimeError on sox failure / missing corrected file, with the 'Original file left untouched' guarantee), and 156-167 (unlink original, rename corrected, re-read). Grep of test_process.py finds no DataLoader test exercising struct/sox/repair. This is the most safety-critical logic in the file (it unlinks the user's original recording) yet is untested.

**Fix:** Add tests via mocker.patch on usv_playpen.processing.load_audio_files.subprocess.run plus a wavfile.read raising struct.error first call: (1) success path (original unlinked, *_correct.wav renamed, dict populated from second read); (2) sox returncode != 0 raises RuntimeError AND original still exists; (3) sox returns 0 but no correct_file raises RuntimeError and original preserved.

### [MEDIUM] correctness — dtype computation crashes (IndexError) on empty WAV data
`load_audio_files.py:179-181`

type(wave_data_dict[one_file.name]["wav_data"].ravel()[0]).__name__ indexes element [0] of the flattened array. If a loaded WAV contains zero samples (header-valid but empty/truncated file, or a sox-repaired file that ends up empty), .ravel()[0] raises IndexError and aborts the entire load loop with no recovery.

**Fix:** Derive the name from wav_data.dtype.name instead of materializing a scalar via .ravel()[0]; this is equivalent for the numpy dtypes scipy/librosa return and avoids the IndexError on empty arrays.

### [MEDIUM] docs_clarity — Docstring describes wrong dictionary key structure
`load_audio_files.py:85-88`

The load_wavefile_data Returns docstring says the "starting key in the dictionary is \"session_id\"", but the code keys the returned dict by one_file.name (the WAV filename) at line 113, not by a session_id. Stale and misleading about how to index the result.

**Fix:** Reword to: the top-level key is the WAV file's name (one_file.name), with "sampling_rate", "wav_data" and "dtype" as sub-keys.

### [MEDIUM] tests — dtype lookup only exercised for int16/float32; KeyError path untested
`load_audio_files.py:179-181`

Tests assert dtype only for int16 (scipy, line 1309) and float32 (librosa, line 1352). Other reachable scipy dtypes (int32, uint8, float64) and the KeyError that occurs when an array's scalar type name is absent from known_dtypes are untested.

**Fix:** Parametrize over dtype in {int32, uint8, float64} via _write_wav(dtype=...) asserting out[...]['dtype']; add a test monkeypatching wavfile.read to return an unmapped dtype and assert KeyError (documenting current behavior).

### [LOW] correctness — '.wav' substring match is too loose, case-sensitive, and matches directories
`load_audio_files.py:112`

The selection guard uses `".wav" in one_file.name`, a substring test. It matches non-WAV names containing '.wav' (e.g. 'recording.wav.bak'), silently skips uppercase '.WAV', and since iterdir() also yields directories, a directory whose name contains '.wav' would be passed to wavfile.read/librosa.load.

**Fix:** Use `one_file.is_file() and one_file.suffix.lower() == ".wav"` instead of the substring membership test.

### [LOW] correctness — Interrupted sox repair can leave a stray '_correct.wav' that is loaded as a real recording
`load_audio_files.py:139-157`

The repair branch writes correct_file = '<stem>_correct.wav' into one_file.parent (the scanned directory). If a run is interrupted between sox completing (157) and not yet renaming, a leftover '<stem>_correct.wav' is picked up on a later run by the loose match at line 112 and loaded as a genuine recording, injecting a duplicate/partial channel into wave_data_dict.

**Fix:** Exclude '_correct.wav' sidecars from the selection filter, or write the repaired file to a temp path outside the scanned directory, so an interrupted repair cannot be mistaken for a real recording.

### [LOW] dead_code_naming — Unreachable keys in known_dtypes lookup table
`load_audio_files.py:35-62`

known_dtypes is consumed only at 179-181 via type(...ravel()[0]).__name__. Verified at runtime that numpy scalar .__name__ yields bare names ('int16','float32','uint8', ...), never an 'np.'-prefixed form and never 'int'/'float'/'str'/'dict'. So the 12 'np.*'-prefixed entries and the four builtin entries ('int','float','str','dict', which also map to type objects rather than dtype strings) can never be matched and are vestigial.

**Fix:** Remove the unreachable 'np.*'-prefixed and 'int'/'float'/'str'/'dict' entries, keeping only the bare numpy scalar names .__name__ can produce (int8/int16/.../float64).

### [LOW] performance — Loop-invariant conditional_arg and library re-read from nested dict on every file
`load_audio_files.py:95-121`

Inside the per-file inner loop, conditional_arg (lines 95-110, looked up and len()'d twice) and library (lines 118-121) are re-dereferenced from self.input_parameter_dict on every iteration even though both are constant for the whole call. Pure redundant nested-dict lookups in the loop (cost dwarfed by WAV I/O, hence low).

**Fix:** Before the `for one_dir` loop bind `conditional_arg = self.input_parameter_dict["load_wavefile_data"]["conditional_arg"]` and `library = self.input_parameter_dict["load_wavefile_data"]["library"]`; inside the loop use `additional_condition = (not conditional_arg) or all(cond in one_file.name for cond in conditional_arg)` and compare against `library`.

### [LOW] tests — Multi-directory wave_data_loc loop is not covered
`load_audio_files.py:92`

All DataLoader tests (test_process.py 1294-1375) pass a single-element list to wave_data_loc, so the outer `for one_dir in ... wave_data_loc` loop is never run with more than one directory; cross-directory accumulation into one wave_data_dict is untested.

**Fix:** Add a test writing WAVs into two tmp subdirectories, pass both in wave_data_loc, and assert the returned dict contains files from both.


## `qlvm_model.py` (6)

### [MEDIUM] tests — gen_fib_basis: only shape asserted, lattice values and fib(m-1) generator untested
`qlvm_model.py:73-91`

test_fib_and_roberts_shapes (test file lines 69-70) asserts only fib.shape == (21, 2) for gen_fib_basis(8). The lattice values and the generator z = [1, fib(m-1)] are never checked, so an off-by-one in the _fibonacci index or in _fibonacci itself would still pass. Verified by execution: gen_fib_basis(8)[1] == [1.0, 13.0]/21 == [0.04761905, 0.61904762] (fib(7)=13).

**Fix:** Add a known-answer assertion, e.g. `assert np.allclose(fib[1], np.array([1.0, 13.0]) / 21.0)`, pinning down both _fibonacci and the m-1 generator index.

### [MEDIUM] tests — roberts_sequence: values and golden-ratio root untested; only trivial row 0 checked
`qlvm_model.py:94-118`

test_fib_and_roberts_shapes asserts only rob.shape == (50, 2) and rob[0] == [0.0, 0.0]; row 0 is trivially zero for any basis since np.arange starts at 0, so it proves nothing about the Newton-solved root (lines 114-117) or the basis formula 1 - 1/x**(1+arange) (line 117). A bug there would pass. Verified: the rows are an exact integer multiple of rob[1] (rob[i] == i*rob[1]).

**Fix:** Add a non-trivial known-answer check. NOTE: the candidate's proposed value is WRONG — actual `roberts_sequence(50,2)[1] % 1` is [0.24512233, 0.43015972] (the code uses 1 - 1/x**..., not 1/x**...), not [0.7548777, 0.5698403]. Use `assert np.allclose(rob[1], [0.24512233, 0.43015972], atol=1e-6)` and/or `assert np.allclose(rob, np.arange(50)[:, None] * rob[1][None, :])`.

### [MEDIUM] tests — embed_data has no known-answer test; full chain only shape/range-checked
`qlvm_model.py:356-386`

test_decoder_forward_and_embed_end_to_end (test lines 126-143) calls embed_data only for shape/range. The recovery known-answer (test lines 100-105) re-implements the embed tail INLINE (posterior @ torus_basis_forward(lattice) then reverse) rather than calling embed_data, so embed_data's own composition (decode_lattice_atlas -> posterior_over_lattice -> weighted average -> reverse) is never asserted to produce a correct coordinate. A wiring bug inside embed_data would not be caught.

**Fix:** Add a known-answer test driving embed_data end-to-end: data = decode_lattice_atlas(lattice, params)[k][None] for chosen k, then `assert np.allclose(embed_data(lattice, data, params)[0], np.asarray(lattice[k] % 1), atol=1e-3)`.

### [LOW] docs_clarity — posterior_over_lattice: evidence subtraction is a softmax no-op, not flagged
`qlvm_model.py:352-353`

`evidence` (line 352) is a per-row constant: logsumexp(lls, axis=1, keepdims=True) minus the scalar log(K). Subtracting a per-row constant from softmax logits is shift-invariant, so `jax.nn.softmax(lls - evidence, axis=1)` (line 353) is numerically identical to `softmax(lls)`. The docstring (lines 335-337) lists the evidence term alongside the likelihood and softmax as if it shapes the posterior, and there is no inline note that the subtraction is retained only for line-by-line parity with QMCLVM.posterior_probability. A reader simplifying or checking parity could be misled. Verified: softmax is shift-invariant, confirmed against the code.

**Fix:** Add an inline comment at line 352-353, e.g. `# evidence is a per-row constant; subtracting it leaves the softmax unchanged. Kept only for exact parity with QMCLVM.posterior_probability.`

### [LOW] performance — Inference path not jax.jit-compiled unlike sibling JAX modules
`qlvm_model.py:356-386`

embed_data and its callees (decoder_forward's four conv_transpose2d blocks, binary_lp's two (B,K,128,128) einsums, posterior_over_lattice) run eagerly. Sibling JAX modules wrap compute in jax.jit (verified: jax_multinomial_logistic_regression.py:76/371, jax_bivariate_regression.py:149/437). Eager dispatch forgoes XLA fusion of the conv stack and the two einsums. Shapes are static per session, so a top-level jit over embed_data (params as a pytree) would compile cleanly.

**Fix:** Wrap embed_data (or at least decoder_forward + binary_lp) in jax.jit for XLA fusion, consistent with the other modeling modules.

### [LOW] tests — torus_basis_forward column ordering/values not asserted standalone (only via roundtrip)
`qlvm_model.py:124-142`

torus_basis_forward is exercised only through reverse(forward(z)) (test line 79) and the inline embed tail. A compensating convention error shared by forward and reverse (e.g. swapped cos/sin halves) would survive the roundtrip, since reverse's atan2(data[:, d:], data[:, :d]) assumes the [cos..., sin...] ordering forward produces. The (N,d)->(N,2d) doubling and the [cos, sin] layout are never pinned independently. Verified: forward([[0.0, 0.25]]) == [[1, 0, 0, 1]] (atol 1e-6).

**Fix:** Add a direct assertion: for z=[[0.0, 0.25]], assert forward(z) shape (1,4) and np.allclose(forward(z), [[1, 0, 0, 1]], atol=1e-6), pinning the column ordering independently of reverse.


## `train_masks.py` (5)

### [MEDIUM] tests — train() never asserts the hyperparameters from cfg are forwarded to model.train()
`train_masks.py:129-138`

model.train() at lines 129-138 is passed epochs=n_epochs, imgsz=imgsz, batch=batch_size, device=device, data=str(data_yaml), exist_ok=True, and YOLO(base_weights) at line 128. The fake _FakeYOLO.train (tests/processing/test_train_masks.py:43) only reads kwargs['project'] and kwargs['name'] and ignores the rest, and no test asserts the forwarded values. A regression swapping epochs/batch, dropping exist_ok=True, passing the wrong imgsz, or wrong data path would pass all current tests; base_weights reaching YOLO() (line 128) is likewise unasserted.

**Fix:** Capture the kwargs the fake YOLO.train receives (and the weights arg to _FakeYOLO.__init__) and assert data == str(dataset_dir/'data.yaml'), epochs == _CFG['train_masks']['n_epochs'], imgsz == 128, batch == batch_size, device == device, name == run_name, exist_ok is True, project == str(output_dir), and YOLO was constructed with base_weights.

### [LOW] correctness — pathlib.Path(None) raises opaque TypeError when dataset/output directory is unset
`train_masks.py:113,121`

With a valid input_parameter_dict (so cfg = self.input_parameter_dict['train_masks'] at line 105 succeeds) but dataset_directory/output_directory left at their None defaults (lines 43-44), train() calls pathlib.Path(self.dataset_directory) at line 113 with None, raising an opaque TypeError rather than the 'clear error' the module/method docstrings promise. Only reachable via direct programmatic construction (the CLI marks both required=True), so impact is low. Note: with the default empty input_parameter_dict the first failure is instead KeyError on 'train_masks' at line 105, not the TypeError.

**Fix:** Before line 113, validate self.dataset_directory and self.output_directory are not None and raise a descriptive ValueError, so direct programmatic use fails with a clear message consistent with the documented contract.

### [LOW] tests — CLI provided_params / ParameterSource.COMMANDLINE filtering is untested
`train_masks.py:182`

Line 182 (provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]) is the only non-trivial CLI logic. The sole CLI test (tests/processing/test_qlvm_pipeline_clis.py:126) passes only the two required dirs and never --base-weights/--n-epochs/--batch-size, so the branch distinguishing COMMANDLINE-supplied options from defaults is never exercised; modify_settings_json_for_cli is mocked, so nothing asserts the provided_params it receives.

**Fix:** Add a CLI test invoking train_masks_cli with --base-weights/--n-epochs/--batch-size (plus required dirs), patch modify_settings_json_for_cli, and assert its provided_params kwarg equals {'base_weights','n_epochs','batch_size'}; a run omitting them yields an empty list.

### [LOW] tests — Default fallbacks in __init__ (message_output->print, input_parameter_dict->{}) are untested
`train_masks.py:74-75`

Lines 74-75 default input_parameter_dict to {} and message_output to print when None. Every test passes both explicitly, so the else-branches never run. The {} default is also the path that makes cfg = self.input_parameter_dict['train_masks'] (line 105) raise KeyError when no settings are supplied.

**Fix:** Add a unit test constructing MaskDetectorTrainer() with no args asserting trainer.input_parameter_dict == {} and trainer.message_output is print; optionally assert train() (with smart_wait patched) raises KeyError on 'train_masks'.

### [LOW] tests — CLI routing test does not assert output_directory kwarg is forwarded
`train_masks.py:139`

test_train_masks_cli_routes (tests/processing/test_qlvm_pipeline_clis.py:139) asserts call_args.kwargs['dataset_directory'] but never asserts 'output_directory' (set from --output-directory at train_masks.py:191-192). A wiring regression dropping or mis-mapping output_directory would not be caught.

**Fix:** Add assert mock_cls.call_args.kwargs['output_directory'] == str(tmp_path / 'out') to test_train_masks_cli_routes.
