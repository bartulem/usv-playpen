# Root-level + GUI review

_Verified line-by-line sweep of the 6 root files incl. usv_playpen_gui.py (~9k LOC): 66 findings. Report-first._

## Summary
- by severity: high 0 · medium 23 · low 43
- by dimension: tests 39 · docs_clarity 14 · correctness 8 · dead_code_naming 4 · performance 1


## `usv_playpen_gui.py` (27)

### [MEDIUM] correctness — replace_name_in_path mis-pairs destinations with experimenter names and treats name as regex
`usv_playpen_gui.py:1068-1075`

The zip pairs each path in recording_files_destinations with [name for name in experimenter_list for loc in recording_files_destinations if name in loc], which is experimenter-outer/destination-inner and not positionally aligned to recording_files_destinations. Multi-destination calls would substitute the wrong name; if the matched name is absent from the path re.search runs against, .span() raises AttributeError. All current call sites (5023-5031, 5442-5452) pass single-element lists, so this is latent, not live. name_in_path is also passed to re.search as a regex pattern; harmless for the current alphabetic experimenter_list but fragile.

**Fix:** Pair each destination with the name actually occurring in it: matched = next((n for n in experimenter_list if n in path), None); use path.find(matched) and slice. Avoids the cross-product zip, the None.span() crash, and regex treatment.

### [MEDIUM] correctness — Post-recording metadata update silently discarded by _save_metadata_to_yaml guard
`usv_playpen_gui.py:6766-6770`

_start_recording sets self.metadata_settings = updated_metadata and calls _save_metadata_to_yaml(), which returns at line 2400 unless the central widget is VideoSettings. record_four sets the central widget to ConductRecording (3182) and the Record button connects to _start_recording (6462), so the guard is False and the recording-updated metadata is never written to _metadata.yaml.

**Fix:** Dump the returned metadata to _metadata.yaml directly rather than routing through the VideoSettings-gated _save_metadata_to_yaml.

### [MEDIUM] dead_code_naming — Dead class: AudioSettings is defined but never instantiated
`usv_playpen_gui.py:1156-1172`

AudioSettings (1156) is never instantiated; record_two uses VideoSettings (2648). grep across the repo finds AudioSettings only at its definition. Vestigial from when audio/video settings were separate windows.

**Fix:** Delete the AudioSettings class (1156-1172).

### [MEDIUM] performance — Subject form edits write full JSON repo AND full metadata YAML to disk on every keystroke
`usv_playpen_gui.py:1599-1668`

_on_subject_form_changed is wired to textChanged of every subject-form QLineEdit (3070-3071) and ends (1667-1668) by calling _update_subject_in_repository (deepcopy at 1773 + json.dump of subject_repository at 1565-1566) and _save_metadata_to_yaml (yaml.dump at 2429-2437 + _update_metadata_preview which yaml.dumps again at 2320-2326). Each keystroke = 1 JSON write + 1 YAML write + 2 YAML serializations on the GUI thread. Confirmed the form lives in VideoSettings so the 2400 guard does not short-circuit it.

**Fix:** Debounce with a single-shot QTimer (300-500ms) firing a single _persist_subject_edits; keep in-memory update immediate.

### [MEDIUM] tests — replace_name_in_path matching branch has no test coverage
`usv_playpen_gui.py:1045-1079`

grep finds no test referencing replace_name_in_path; the matching branch (1068-1075) computing the re.search span and splice is unexercised. A regression in the span math or zip pairing would be silent.

**Fix:** Add unit tests for the match, multi-destination, and no-match passthrough cases.

### [MEDIUM] tests — EphysDialog.save_and_accept probe_sn/probe_reused coercion untested
`usv_playpen_gui.py:645-651`

The add-mode test only asserts the intervention is a dict; never sets probe_sn. Coercion at 645-651 (probe_reused->bool, digit tokens->int, non-numeric->str, empty->[]) is uncovered and feeds ephys/NWB export.

**Fix:** Extend the EphysDialog test to assert probe_sn list coercion and probe_reused bool.

### [MEDIUM] tests — _on_subject_form_changed rename/collision logic untested
`usv_playpen_gui.py:1570-1668`

No test touches the rename/collision handler with its guards (1599-1606), blank-id short-circuit (1633), collision refusal against repository and session (1648-1653), and clean-rename preset-drop + active_subject_id bump (1657-1663). These guard against metadata/preset corruption.

**Fix:** Add tests for non-id edit, unique rename, colliding rename, and blank id.

### [MEDIUM] tests — _on_subject_selected_from_completer dedupe and deepcopy-append branches untested
`usv_playpen_gui.py:1716-1738`

No test references it; the already-in-session info-box branch and the deepcopy-then-append branch (load-bearing to prevent cross-store corruption) are both uncovered.

**Fix:** Add tests asserting no duplicate append and a distinct appended object.

### [MEDIUM] tests — _remove_subject index bounds-checking untested
`usv_playpen_gui.py:2211-2229`

The 'subjects and 0 <= index < len' guard (2226) no-ops on out-of-range/negative/empty and otherwise pops and re-saves; none of these branches tested.

**Fix:** Add in-range, out-of-range, and negative-index tests.

### [MEDIUM] tests — _on_equipment_checkbox_toggled add/remove branches untested
`usv_playpen_gui.py:2231-2275`

Untested: dot->underscore key transform, lazy Equipment init, checked-branch copy + sync_equipment_dynamic_fields, unchecked-branch delete. Drives equipment metadata written to _metadata.yaml.

**Fix:** Add checked/unchecked/None-Equipment tests with stubs.

### [MEDIUM] tests — _add_subject new/update/weight branches untested
`usv_playpen_gui.py:2167-2209`

Untested: empty-id early return, weight literal-eval-with-fallback, update-existing vs append-new. Determines duplicate-row creation.

**Fix:** Add empty-id, new-id, existing-id, and non-numeric-weight tests.

### [LOW] correctness — Font family lookup can raise IndexError if no bundled font registers
`usv_playpen_gui.py:1352-1354`

If every addApplicationFont returns -1, font_file_loc stays None and applicationFontFamilies(None)[0] raises IndexError at GUI startup. Edge case (corrupt/missing bundled fonts in a packaged install).

**Fix:** Guard: if font_file_loc is None fall back to a system family or raise a clear error naming the missing fonts.

### [LOW] correctness — Intervention edit button uses 'Unknown' subject_id fallback, breaking lookup
`usv_playpen_gui.py:2113`

subject.get('subject_id', 'Unknown') (2113) is bound into _open_edit_intervention_dialog (2124); the handler looks up s['subject_id'] == subject_id (2149), so 'Unknown' never matches an id-less subject (and 2149 would KeyError on a subject lacking the key). The remove-button path keys off the list index (2097), so this is an inconsistency. Subjects always have a subject_id in practice (form validation requires it).

**Fix:** Key the edit dialog off the subject's list index like the remove button, or skip building edit buttons for subjects without a valid subject_id.

### [LOW] dead_code_naming — Misleading method name _combo_box_prior_audio_device_camera_input has nothing to do with cameras
`usv_playpen_gui.py:6158`

The handler (6158) serves the Trgbox-USGH device(s) combo (m/s/both, variable_id='device_receiving_input', connected at 3720). Neither the name nor the docstring 'Audio device camera input combo box.' relates to cameras.

**Fix:** Rename to _combo_box_device_receiving_input and fix the docstring; update the single connect at 3720.

### [LOW] docs_clarity — replace_name_in_path docstring parameter descriptions are swapped/wrong
`usv_playpen_gui.py:1055-1061`

experimenter_list is documented as 'Path to be modified.' and recording_files_destinations as 'New name to be used.' Both are backwards: experimenter_list is the names searched for, recording_files_destinations is the paths rewritten, exp_id is the inserted name.

**Fix:** Reword each parameter to match actual roles (names list / paths list / inserted id).

### [LOW] docs_clarity — _nudge_button_icon_up docstring states wrong default for shift_px
`usv_playpen_gui.py:124`

shift_px docstring says '(default 2)' but signature is shift_px: int = 1 (104) and the Description says 1px.

**Fix:** Change '(default 2)' to '(default 1)'.

### [LOW] docs_clarity — initialize_main_window return annotation/docstring do not match the returned tuple
`usv_playpen_gui.py:7145-7149`

Returns (usv_playpen_app, usv_playpen_window) at 7252 and main() unpacks two values (7269), but the annotation is -> QMainWindow and Returns says 'The initialized GUI windows.' The no_splash param also uses inline style, not the repo block convention.

**Fix:** Annotate -> tuple[QApplication, QMainWindow], document the (app, window) pair, and reformat no_splash to block style.

### [LOW] docs_clarity — highlightBlock docstring has an empty Parameters section
`usv_playpen_gui.py:220-232`

highlightBlock(self, text: str) documents Returns but leaves Parameters empty despite the real text argument.

**Fix:** Add text (str) - the line of text Qt passes for this block.

### [LOW] docs_clarity — Typo 'target_line_ediT' in _open_directory_dialog Parameters
`usv_playpen_gui.py:6982`

Parameters documents 'target_line_ediT (QLineEdit)' with a stray capital T; actual parameter is target_line_edit (sibling _open_file_dialog is correct).

**Fix:** Fix to 'target_line_edit'.

### [LOW] docs_clarity — User-facing dialog title typo 'Avisoft sase directory'
`usv_playpen_gui.py:2497`

The Browse directory-dialog title reads 'Select Avisoft sase directory' ('sase' instead of 'base').

**Fix:** Change to 'Select Avisoft base directory'.

### [LOW] docs_clarity — _remove_subject Parameters block omits index_to_remove
`usv_playpen_gui.py:2211-2223`

_remove_subject(self, index_to_remove: int) leaves Parameters empty though the arg is the index to pop.

**Fix:** Add index_to_remove (int) description.

### [LOW] tests — _save_metadata_to_yaml write path untested (only early-return covered)
`usv_playpen_gui.py:2386-2439`

Existing test covers only the non-VideoSettings early return; the institution_edit guard, Session assignments, keywords split (2425), YAML dump, and preview are uncovered.

**Fix:** Add a write-path test asserting keywords split and a second-guard early-return test.

### [LOW] tests — YamlHighlighter.highlightBlock key-detection untested
`usv_playpen_gui.py:220-237`

No test references YamlHighlighter; the key regex (217) coloring is untestable-regression-prone but headlessly testable via QTextDocument.

**Fix:** Attach YamlHighlighter to a QTextDocument and assert key spans are formatted and value lines are not.

### [LOW] tests — _update_nested_dict_value normalization and traversal untested
`usv_playpen_gui.py:6933-6969`

No test references it; str(Path(text)) normalization (6957-6958) and keys_path traversal are uncovered. A test would also catch the empty-string -> '.' behavior.

**Fix:** Add traversal, slash-path, and empty-string tests.

### [LOW] tests — _validate_subject_form enable/disable toggle untested
`usv_playpen_gui.py:2277-2303`

Both branches (missing-attrs early return, required-fields toggle of add_subject_btn) are uncovered; gates whether a subject can be added.

**Fix:** Add fill/blank/missing-attr tests.

### [LOW] tests — _load_subject_repository corruption fallbacks untested
`usv_playpen_gui.py:1523-1547`

Resilience branches (missing file -> [], non-list -> [], JSONDecodeError -> []) are uncovered; they exist to survive a corrupted subject_presets.json.

**Fix:** Point subject_repo_path at tmp files (absent, non-list, malformed, valid) and assert results.

### [LOW] tests — _start_visualizations / _start_analyses forwarders untested
`usv_playpen_gui.py:6688-6718`

test_start_handlers_invoke_backends covers processing/calibration/recording but not the visualize_data and analyze_data forwarders; a wrong-backend swap would be invisible.

**Fix:** Add MagicMock assertions for both forwarders.


## `yaml_utils.py` (14)

### [LOW] correctness — sensor_exposure_time and sensor_gain can be written with mismatched completeness
`yaml_utils.py:393-396`

exposures and gains are accumulated independently in the per-camera loop (lines 389-392), then written behind independent guards at 393 and 395. A cam_block that is a dict but lacks only 'gain' (or only 'exposure_time') leaves one list complete (len==sorted_cams) and the other short, so the complete array is written while the incomplete one is silently skipped, desyncing the two per-sensor arrays on disk.

**Fix:** Gate both writes on a single completeness check: write both or neither when exposures and gains and len(exposures)==len(sorted_cams)==len(gains).

### [LOW] correctness — Multiple *_metadata.yaml files silently resolved to the first sorted match
`yaml_utils.py:132-136`

load_session_metadata does sorted(path.glob('*_metadata.yaml'))[0] with no warning when more than one match exists. Metadata is treated as precious elsewhere, so silently picking one of several is an unsafe ambiguity that save_session_metadata can later overwrite.

**Fix:** When len(metadata_path_list) > 1, emit a warning via logger naming the chosen file and the count of matches.

### [LOW] docs_clarity — Docstring omits the bool() coercion applied to device_sync
`yaml_utils.py:265`

Docstring line 265 documents device_sync <- exp_settings_dict['audio']['usgh_devices_sync'] verbatim, but line 351 writes bool(audio_section['usgh_devices_sync']). It is the only audio field transformed rather than passed through.

**Fix:** Update the mapping line to note the coercion: device_sync <- bool(exp_settings_dict['audio']['usgh_devices_sync']).

### [LOW] docs_clarity — Docstring for sensor_sn omits the digit-string -> int conversion
`yaml_utils.py:279-280`

Docstring says sensor_sn <- sorted ascending list of expected_cameras, but line 376 converts all-digit serials to int (sn_values) before writing, which changes how the values serialize in YAML.

**Fix:** Note in the docstring that all-digit serials are emitted as ints.

### [LOW] tests — device_br-absent front-insertion fallback untested
`yaml_utils.py:453-457`

When sync_LEDs exists but lacks device_br, the 'if not inserted' branch (453-457) prepends device_port at the front. No test exercises this: both insert tests include a device_br anchor and the two no-op tests bail earlier.

**Fix:** Add a test where sync_LEDs lacks device_br and assert device_port lands at the front with the given value.

### [LOW] tests — Equipment-missing and non-dict sync_LEDs guards untested
`yaml_utils.py:432-437`

Line 432-434 (Equipment key absent, e.g. md={'Session':{}}) is never taken: the no-op test uses md={'Equipment':{}} which passes that check and hits 435-437. The sync_LEDs-present-but-not-a-dict guard (435-437) is also untested.

**Fix:** Add no-op tests for set_sync_LEDs_device_port({'Session':{}}, 'COM9') and set_sync_LEDs_device_port({'Equipment':{'sync_LEDs':'notadict'}}, 'COM9').

### [LOW] tests — save_session_metadata YAMLError logging path untested
`yaml_utils.py:486-487`

The except yaml.YAMLError branch (486-487) logging 'Error saving metadata file' is never exercised; all save tests pass serializable data.

**Fix:** Monkeypatch yaml.dump to raise yaml.YAMLError, call save_session_metadata with a logger sink, and assert the error message is logged.

### [LOW] tests — Codec passthrough for unmapped codec untested
`yaml_utils.py:367`

Line 367 .get(codec_short, codec_short) passes unmapped/already-long codecs through, but only the 'hq' translation is tested.

**Fix:** Assert an unmapped value (e.g. 'nvenc-ll-yuv420') passes through to output_file_codec unchanged.

### [LOW] tests — Lexicographic serial-sort fallback (non-digit serials) untested
`yaml_utils.py:375`

Line 375 uses lexicographic sort and line 376 keeps non-digit serials as strings only when not all serials are digit strings; both serial tests use all-digit serials, so this branch never runs.

**Fix:** Add a test with alpha serials (e.g. ['camB','camA']) asserting sensor_sn==['camA','camB'] kept as strings with exposure/gain aligned.

### [LOW] tests — Exposure/gain abort path on non-dict cam_block untested
`yaml_utils.py:385-388`

Lines 385-388 reset exposures/gains and break when a cam_block is missing or not a dict, after which the length guards skip both writes. This abort-and-skip path is uncovered.

**Fix:** Add a test where a serial in expected_cameras has no cameras_config entry; assert sensor_count/sensor_sn update but sensor_exposure_time/sensor_gain retain their original values.

### [LOW] tests — expected_cameras present but cameras_config empty/absent untested
`yaml_utils.py:380`

Line 380 guards the per-camera loop with 'if cameras_config'; the case where expected_cameras is set but cameras_config is empty/absent (count/sn written, exposure/gain untouched) is uncovered.

**Fix:** Add a test with expected_cameras set and no cameras_config; assert sensor_count/sensor_sn written and exposure/gain unchanged.

### [LOW] tests — Non-string arduino_sync_port skip branch untested
`yaml_utils.py:343-345`

Line 344 writes device_port only when arduino_sync_port is a str; the negative branch (None/absent/non-string) is untested.

**Fix:** Add a test with arduino_sync_port=None asserting device_port keeps its fixture value.

### [LOW] tests — Absence of usgh_devices_sync / fabtast skip branches untested
`yaml_utils.py:350-354`

Lines 350/353 gate audio writes on key presence; the case where audio_Avisoft block exists but the toml audio section lacks these keys (writes skipped while block present) is uncovered.

**Fix:** Add a test on _equip_fixture() with exp audio section lacking usgh_devices_sync and fabtast; assert device_sync/device_sr keep fixture values.

### [LOW] tests — Multi-file sorted-first-match selection untested
`yaml_utils.py:132-136`

test_load_session_metadata_loads_first_match writes only one file, so the 'sorted picks alphabetically-first among several' behavior is never verified.

**Fix:** Write 'a_metadata.yaml' and 'b_metadata.yaml' with distinct contents and assert the 'a_metadata.yaml' data/path is returned.


## `os_utils.py` (11)

### [MEDIUM] docs_clarity — Module docstring is stale and undersells the file's scope
`os_utils.py:1-5`

The header describes the file as only 'Configure path to the OS in use and small subprocess/glob helpers', but it now also implements lab-share token expansion (expand_lab_share, _host_lab_shares, recording_destinations), {experimenter} templating (_host_experimenter), atomic file publishing (atomic_output_path), data-root resolution (resolve_data_root), and embedding-landscape path resolution (resolve_embedding_arrays_path / resolve_consolidated_h5_path / resolve_pooled_embeddings_cache).

**Fix:** Expand the module docstring to enumerate the responsibility groups: lab CUP-share resolution, OS + cluster path translation, {experimenter} templating, atomic output publishing, subprocess/glob helpers, and embedding-landscape path resolution.

### [MEDIUM] docs_clarity — wait_for_subprocesses Returns docstring claims None for terminated procs, but code re-polls real exit codes
`os_utils.py:554-556,573-576`

The max_seconds doc (554-556) and Returns doc (573-576) state the still-running slots 'are left as None'. The code re-polls every handle at line 616 ('status = [p.poll() for p in subps_list]') after terminate()/kill(); a killed process has exited by then, so poll() returns its actual (typically negative, e.g. -15/-9) return code, not None. The docstring is misleading.

**Fix:** Reword to say that on timeout-without-raise the slots hold the terminated subprocesses' actual return codes (typically a negative signal code such as -15 or -9), and only remain None if a handle could not be polled to completion.

### [MEDIUM] tests — SIGKILL fallback path in wait_for_subprocesses is never exercised
`os_utils.py:605-610`

The kill() branch (605-610) fires only if a subprocess is still running after terminate() plus the 3 s grace. FakePopen.terminate() (test line 472-474) sets _terminated=True so its next poll() returns the terminate code, meaning kill() is never reached. FakePopen defines self.killed (line 462) but no test asserts it; the OSError suppression around kill() (609-610) is also untested.

**Fix:** Add a FakePopen variant whose terminate() does NOT make poll() return (ignores SIGTERM), drive a timeout with raise_on_timeout=False, and assert proc.killed is True; add a case where kill() raises OSError to cover 609-610.

### [MEDIUM] tests — newest_match_or_raise missing-root branch untested
`os_utils.py:727-730`

newest_match_or_raise has its own root.exists() guard raising FileNotFoundError 'does not exist' (727-730), distinct from first_match_or_raise's. Only test_first_match_raises_on_missing_root (test line 383) covers first_match's guard; no test calls newest_match_or_raise with a non-existent root.

**Fix:** Add test_newest_match_raises_on_missing_root that calls os_utils.newest_match_or_raise(tmp_path / 'absent', '*.bin') and asserts pytest.raises(FileNotFoundError, match='does not exist').

### [MEDIUM] tests — newest_match_or_raise default st_ctime key closure never executed
`os_utils.py:731-733`

The only successful test call (test line 407) passes key=lambda p: p.name; the empty-match test (413) raises before reaching max(). The default-key path (if key is None: def key(p): return p.stat().st_ctime, 731-733) is therefore never run, leaving the documented default ordering untested.

**Fix:** Add a test creating two matching files with distinct mtimes/ctimes (via os.utime) and calling newest_match_or_raise(tmp_path, '*.bin') WITHOUT a key, asserting the newest is returned.

### [MEDIUM] tests — _host_experimenter missing-'experimenter'-key KeyError untested
`os_utils.py:229-234`

_host_experimenter raises KeyError when the host config parses but lacks an 'experimenter' entry (229-234). Tests cover the read (test line 55) and the missing-file RuntimeError (62) but not a parseable config without the key. _host_lab_shares has analogous lab_shares/file_server KeyError tests (143-167) but the experimenter equivalent is absent.

**Fix:** Add a test writing a TOML with no 'experimenter' key, point _HOST_CONFIG_PATH at it, clear _HOST_EXPERIMENTER_CACHE, and assert pytest.raises(KeyError, match='experimenter').

### [LOW] correctness — Killed subprocesses are never reaped (wait()) and status not refreshed on the raising path
`os_utils.py:605-617`

After SIGTERM/SIGKILL the kill loop (605-610) only calls poll(), never wait(); a child killed by signal can be left a zombie until reaped, and poll() reaps only if the child has already exited. On the raise_on_timeout=True path the function raises at 611 before the final status re-poll at 616. Minor resource hygiene; the docstring documents the None slots on the non-raising path so behavior matches contract.

**Fix:** After the kill loop add a short reaping pass, e.g. for i in still_running_idx: with contextlib.suppress(OSError, subprocess.TimeoutExpired): subps_list[i].wait(timeout=1), before raising/returning.

### [LOW] tests — TomlDecodeError branch of host-config readers untested
`os_utils.py:160,222`

Both _host_lab_shares (160) and _host_experimenter (222) catch (OSError, toml.TomlDecodeError) and wrap as RuntimeError 'Cannot read the host config'. Existing tests trigger only the OSError half via an absent file (test lines 62-67, 134-140). A malformed-but-present TOML (the TomlDecodeError half) is never exercised.

**Fix:** Add a test writing invalid TOML, point _HOST_CONFIG_PATH at it, clear the cache, and assert pytest.raises(RuntimeError, match='Cannot read the host config') for both readers.

### [LOW] tests — Cache-hit fast-path of _host_lab_shares/_host_experimenter untested
`os_utils.py:155-156,217-218`

Both resolvers short-circuit on a populated cache (155-156 and 217-218). No test pre-populates the cache and asserts the cached value is returned without re-reading the config (all tests clear the cache first).

**Fix:** Add a test that sets _HOST_SHARES_CACHE / _HOST_EXPERIMENTER_CACHE to a sentinel, points _HOST_CONFIG_PATH at a non-existent path, and asserts the resolver returns the sentinel without raising.

### [LOW] tests — recording_destinations edge cases (empty selection, multiple labs, absent name) untested
`os_utils.py:110-118`

test_recording_destinations_derives_selected_only (test line 182) covers exactly one selected lab. Uncovered: empty selected_labs (two empty lists), multiple labs at once (both OS-form lists in lab_shares order), and a selected name absent from lab_shares (silently skipped). The set/order/skip logic at 110-118 is only partially exercised.

**Fix:** Parametrize: (a) selected_labs=[] -> ([], []); (b) selected_labs=['falkner','murthy'] -> both destinations in table order with correct Windows backslash forms; (c) selected_labs=['nonexistent'] -> ([], []).

### [LOW] tests — atomic_output_path BaseException cleanup path not explicitly covered
`os_utils.py:519-522`

The except clause catches BaseException (519) specifically so KeyboardInterrupt/SystemExit also clean up the temp sibling and re-raise. test_atomic_output_path_preserves_original_on_error (test line 350) only raises RuntimeError (an Exception). The BaseException-only behaviour (the reason it isn't a plain except Exception) is untested.

**Fix:** Add a test that raises KeyboardInterrupt inside the with-block, assert it propagates (pytest.raises(KeyboardInterrupt)), the pre-existing final file is untouched, and no .data.txt.tmp* sibling remains.


## `send_email.py` (8)

### [MEDIUM] dead_code_naming — Dead attribute exp_settings_dict — stored but never read
`send_email.py:59`

Constructor accepts exp_settings_dict (line 29) and assigns self.exp_settings_dict (line 59), but no Messenger instance ever reads it. Verified: grep for .exp_settings_dict matches only controller/GUI classes, never a Messenger instance. All 8 call sites pass exp_settings_dict=self.exp_settings_dict (preprocess_data.py:171,396; behavioral_experiments.py:1032,1542; visualize_data.py:102,193; analyze_data.py:110,191) with no effect on send_message().

**Fix:** Remove the exp_settings_dict parameter (line 29), its docstring entry (lines 43-44), and the self.exp_settings_dict assignment (line 59), then drop the exp_settings_dict=self.exp_settings_dict kwarg from the 8 call sites.

### [MEDIUM] tests — KeyError path for missing [email] section is never tested
`send_email.py:96-100`

get_email_params raises a custom KeyError when the INI exists but lacks the [email] section (lines 96-100). No test exercises this branch directly or via send_message; tests/foundation/test_send_email.py covers only FileNotFoundError.

**Fix:** Add a test writing an INI with a non-email section and asserting pytest.raises(KeyError) on get_email_params(); also assert send_message(...) is False and logs contain 'KeyError'.

### [MEDIUM] tests — no_receivers_notification log branch and False (silent) variant untested
`send_email.py:161-164`

Empty receivers + no_receivers_notification=True emits the notification string (lines 162-164); the existing test asserts only the None return, never the string. The no_receivers_notification=False path (elif on line 161 skipped, no log, returns None at line 166) is entirely uncovered.

**Fix:** Extend the no-receivers test to capture logs and assert the notification string; add a silent-path test: Messenger(receivers=[], no_receivers_notification=False, ...) asserting send_message is None and logs == [].

### [LOW] docs_clarity — get_email_params summary line is stale/incomplete (omits host and port)
`send_email.py:67`

Line 67 says the method 'gets the lab e-mail address and password', but it returns four values: email_host, email_port, email_address, email_password (return tuple line 101, Returns types line 75). The summary is inaccurate, not merely terse.

**Fix:** Reword line 67 to 'This method gets the lab e-mail host, port, address, and password needed to send a message.'

### [LOW] docs_clarity — Returns description omits host and port
`send_email.py:75-76`

Returns type line 75 lists all four values; the prose on line 76 reads only 'Lab e-mail address and password.', omitting host and port that are also returned.

**Fix:** Change line 76 to 'Lab e-mail host, port, address, and password.'

### [LOW] tests — '<unresolved host>' fallback in error log is not asserted
`send_email.py:155`

When get_email_params raises before email_host is assigned, email_host stays None and line 155 yields host_info='<unresolved host>'. test_send_message_missing_creds_returns_false_and_logs hits this path but asserts only 'FileNotFoundError', never the fallback string. The resolved-host branch is asserted (smtp.test:465).

**Fix:** In that test, additionally assert any('<unresolved host>' in m for m in logs).

### [LOW] tests — Multi-receiver 'To' header join is untested
`send_email.py:142`

Line 142 builds the To header via ', '.join(self.receivers). All sending tests use a single receiver ['a@b.org'], so the comma-space join of multiple addresses is never verified.

**Fix:** Add a test with Messenger(receivers=['a@b.org','c@d.org'], ...) and FakeSMTP asserting FakeSMTP.last_msg['To'] == 'a@b.org, c@d.org'.

### [LOW] tests — __init__ default normalization (receivers=None, message_output=None) untested
`send_email.py:58-60`

Lines 58-60 normalize defaults: receivers=None -> [], message_output=None -> print. No test constructs Messenger() with these omitted to confirm receivers defaults to [] (so send_message returns None) and message_output falls back to print.

**Fix:** Add test_init_defaults: msgr = Messenger(); assert msgr.receivers == [] and msgr.message_output is print; assert msgr.send_message(subject='s', message='m') is None.


## `cli_utils.py` (3)

### [MEDIUM] tests — parameters_lists list-coercion branches of modify_settings_json_for_cli are untested
`cli_utils.py:167-173`

modify_settings_json_for_cli has three coercion branches: line 169 (param not in parameters_lists, value as-is), line 171 (in parameters_lists AND tuple -> list(...)), line 173 (in parameters_lists AND not tuple -> [value]). In tests/foundation/test_cli_utils.py the function is only called without parameters_lists (lines 137-139, 151-153, defaulting to None), and test_cli.py mocks it. Branches 171 and 173 never execute under test, so a regression in tuple->list coercion or scalar->[value] wrapping would go undetected.

**Fix:** Add a test loading a real settings_dict with an unambiguous key, calling modify_settings_json_for_cli with parameters_lists=[key]: once with ctx.params[key] a tuple (assert written value == list(tuple)) and once with a scalar (assert written value == [scalar]).

### [LOW] dead_code_naming — Unreachable except ValueError branch in StringTuple.convert
`cli_utils.py:67-70`

The try block (lines 50-65) contains only value.split(',') (line 52), a str.strip list comprehension, len(), self.fail(...), and tuple(parts). None of these raise ValueError for a str input: str.split never raises ValueError, and self.fail raises click.BadParameter (a subclass of UsageError/ClickException, not ValueError). The except ValueError handler at lines 67-70 and its 'could not be parsed as a comma-separated pair' message are therefore dead/unreachable code.

**Fix:** Remove the dead try/except ValueError (lines 50, 67-70). The arity validation (len(parts) != 2) at lines 55-62 already covers the only real failure case.

### [LOW] docs_clarity — override_toml_values docstring never documents value and omits comma-to-list behavior
`cli_utils.py:320-332`

The overrides parameter bullet list (lines 322-323) documents only key.path and dangles; value is never described, nor the non-obvious rule that a comma in value yields a LIST (lines 341-343) while a comma-free value yields a single scalar, with per-item _convert_value type coercion (bool/int/float/quote-stripped str). A reader cannot predict from the docstring that 'a.b=1,2,3' yields a list of ints.

**Fix:** Extend the overrides docstring to document value: it is passed through _convert_value for type coercion, and a comma in value produces a list of converted items whereas a comma-free value produces a single converted scalar.


## `time_utils.py` (3)

### [MEDIUM] tests — Untested branch: instance() returns a non-QApplication QCoreApplication
`time_utils.py:31`

is_gui_context() (line 31) uses isinstance(QCoreApplication.instance(), QApplication). The test file covers only instance() is None (returns False, test line 14) and a real QApplication via the qapp fixture (returns True, test line 18). The discriminating case that justifies isinstance over a plain `instance() is not None` truthiness check - a live QCoreApplication that is NOT a QApplication (headless/console Qt loop) returning False - is untested. A regression loosening the check to truthiness would not be caught.

**Fix:** Add a test monkeypatching time_utils.QCoreApplication.instance to return a non-QApplication object and asserting is_gui_context() is False, e.g. monkeypatch.setattr(time_utils.QCoreApplication, 'instance', lambda: object()); assert time_utils.is_gui_context() is False.

### [MEDIUM] tests — Untested int() truncation of fractional seconds in the qWait branch
`time_utils.py:61`

smart_wait()'s GUI branch (line 61) computes QTest.qWait(int(seconds * 1000)). The only GUI-path test (test line 30-34) passes integer seconds=2 yielding exactly 2000 ms, so the int() truncation of a fractional millisecond result is never exercised. seconds=0.0015 -> int(1.5) == 1 and seconds=0.0009 -> int(0.9) == 0; this truncation-toward-zero is load-bearing for a precise-waiting utility and untested. The non-GUI branch passes the raw float (test line 26-27), so the two branches differ in rounding, which is exactly what should be pinned.

**Fix:** Add a GUI-branch test with a fractional input: monkeypatch time_utils.QTest.qWait to record its argument, call time_utils.smart_wait(app_context_bool=True, seconds=0.0015), and assert recorded == [1] (and/or seconds=0.0009 -> 0).

### [LOW] correctness — GUI branch int() truncates sub-millisecond waits to zero, diverging from time.sleep branch
`time_utils.py:61`

Line 61 does QTest.qWait(int(seconds * 1000)). int() truncates toward zero, so seconds < 0.001 (e.g. 0.0005) becomes int(0.5) == 0 (no wait at all), while the time.sleep branch (line 63) would actually wait. The branches also round fractional inputs inconsistently. Qt's qWait is inherently millisecond-resolution so sub-ms precision is impossible regardless, but the int() truncation (vs round) makes the discrepancy worse and can silently yield a zero wait, which contradicts the docstring's 'may be fractional for more precise timing'. In practice this utility is called with second-scale waits, so impact is minor.

**Fix:** Use round() instead of int(): QTest.qWait(round(seconds * 1000)). Optionally clamp to a minimum of 1 ms when seconds > 0 and document that GUI waits are millisecond-resolution.
