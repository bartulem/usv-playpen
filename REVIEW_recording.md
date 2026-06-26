# Recording subsystem review (`behavioral_experiments.py`)

_Verified line-by-line sweep: 21 findings. Report-first._

## Summary
- by severity: high 1 · medium 7 · low 13
- by dimension: tests 12 · correctness 5 · docs_clarity 2 · performance 2

### [HIGH] tests — Audio-recording branch of conduct_behavioral_recording (incl. verify_avisoft abort path) is untested
`behavioral_experiments.py:1094-1141`

Both full-orchestration tests avoid this block: test_conduct_behavioral_recording_orchestrates_calls (test line 901) sets conduct_audio_recording=False (test line 877), and test_..._raises_on_bad_credentials (test line 961) sets it True but raises FileNotFoundError at get_cup_mount_params before reaching line 1094. Consequently the modify_audio_file gate, CPU affinity/priority assembly (1107-1128), Avisoft Start-Process construction, and critically the verify_avisoft_is_recording() failure path (1137-1141) that kills CoolTerm/Avisoft and raises RuntimeError('...did not produce audio bytes...') are never exercised. A regression in the abort logic would not be caught.

**Fix:** Add a test with conduct_audio_recording=True, monkeypatching verify_avisoft_is_recording to return False and subprocess.Popen, asserting conduct_behavioral_recording raises RuntimeError matching 'did not produce audio bytes' AND that two Stop-Process Popen calls were issued before the raise; add a sibling with verify returning True asserting the recording proceeds.

### [MEDIUM] correctness — Mount-failure error message reports the wrong PC IP
`behavioral_experiments.py:774`

The loop at line 769 is `for ip_address_pc in [ip_address, second_ip_address]:` and probes each PC via `check_remote_mount(hostname=ip_address_pc, ...)` (line 772). But the RuntimeError at line 774 interpolates `{ip_address}` (the master PC) instead of the loop variable `{ip_address_pc}`. When the mount check fails on the SECOND tracking PC, the error blames the master PC, misdirecting debugging. `{lin_directory}` is the same for both PCs, so the message gives no way to tell which machine failed.

**Fix:** Change the f-string to use the loop variable: `raise RuntimeError(f"Mount point {lin_directory} on {ip_address_pc} is not valid; fix the mount and try again.")`.

### [MEDIUM] correctness — Backup-drive robocopy /MIR subprocesses are fire-and-forget (never awaited or rc-checked)
`behavioral_experiments.py:1530-1538`

The backup-mirror robocopy at lines 1532-1538 is launched with `subprocess.Popen(... '/MIR' ...)` but the handle is never stored, waited on, or rc-checked. Line 1540 immediately logs 'Transferring audio/video files finished' and the method returns at 1551. This contrasts with the primary audio move (lines 1437-1445) which uses `wait_for_subprocesses` with a 15-min budget and per-file outcome reporting. Via the CLI path, after conduct_behavioral_recording() returns the process can exit, orphaning the robocopy children mid-transfer and leaving a partial/empty backup mirror with no error surfaced. A robocopy rc>=8 (genuine failure) is never detected.

**Fix:** Collect the backup Popen handles and pass them to wait_for_subprocesses (as done for the audio move at 1437) with a robocopy-aware rc check, or at minimum store the handles and `.wait()` + check rc before logging completion and returning.

### [MEDIUM] docs_clarity — check_camera_vitals docstring claims camera_fr 'defaults to None' but parameter has no default
`behavioral_experiments.py:758-759`

Lines 758-759 state "camera_fr (int / float) Camera sampling rate; defaults to None." but the signature at line 750 is `def check_camera_vitals(self, camera_fr: int | float) -> None:` with no default. The 'defaults to None' clause is stale and falsely implies the argument is optional.

**Fix:** Remove the incorrect '; defaults to None' clause so it reads "camera_fr (int / float) Camera sampling rate."

### [MEDIUM] tests — Audio file-move pipeline (rename, robocopy, outcome summary, orphan scan) has zero coverage
`behavioral_experiments.py:1305-1523`

Gated on conduct_audio_recording (line 1305), which no test sets True past the credential check, so the entire PLAN/EXECUTE robocopy block is untested: the total==0 prefix-rename loop (1313-1335), the multichannel total!=0 branch with relevant_file_count math (1336-1358), wait_for_subprocesses, the rc>=8/rc is None failure classification (1482-1496), the 'done: n_ok/n_total copied' summary (1501-1503), the orphan scan (1512-1523), and the OSError local-rename 'continue' (1331-1333, 1354-1356) / FileNotFoundError skip (1323-1325) branches.

**Fix:** Add a conduct_audio_recording=True test with audio.general.total=0 seeding a .wav under <avisoft_base>/ch1, mocking subprocess.Popen and wait_for_subprocesses to return a failing rc (e.g. [8]) with a tempfile of fake Win32 output, asserting the FAILED and orphan-scan messages; add a sibling returning rc=[1] asserting 'done: 1/1 copied, 0 failed'; add a third forcing Path.rename to raise OSError asserting the 'Local rename failed' skip.

### [MEDIUM] tests — Dropout-count JSON emission branch in conduct_behavioral_recording is untested
`behavioral_experiments.py:1257-1277`

The block gated on conduct_audio_recording and usghflags != 1574 (line 1257) that builds audio_triggerbox_sync_info_dict, loops ch1/ch13->m/s calling count_last_recording_dropouts, emits the '[***Important!***]'/None/zero messages (1268-1274), and writes audio/audio_triggerbox_sync_info.json (1276-1277) is never reached because all orchestration tests skip the audio path. count_last_recording_dropouts is unit-tested in isolation but its integration here (None message path and the >0 'Important' path) is not.

**Fix:** In a conduct_audio_recording=True test set usghflags != 1574, seed ch1/ch1.log and ch13/ch13.log with and without 'dropout' tokens, and assert audio_triggerbox_sync_info.json is written under <win_dest>/audio with expected num_dropouts per device plus the '[***Important!***]' message when dropouts > 0.

### [MEDIUM] tests — verify_avisoft_is_recording multichannel branch (total != 0) untested
`behavioral_experiments.py:275-276`

All four verify_avisoft tests (test lines 571-631) set audio.general = {'total': 0}, exercising only the used_mics channel-dir resolution at lines 273-274. The else branch at lines 275-276, which hard-codes channel_dirs to ch1 and ch13 (mic indices [0, 12]) when total != 0, is never executed, so a regression in multichannel channel-directory selection would be missed.

**Fix:** Add a verify_avisoft test with avisoft_dir_factory([0, 12]) and audio.general={'total': 1}, dropping a fresh .wav into both ch1 and ch13, asserting it returns True only after BOTH channel dirs show growth (and False if only one does).

### [MEDIUM] tests — check_camera_vitals multi-camera connect/disconnect and MotifMulticamFrameRate path untested
`behavioral_experiments.py:781-811`

test_check_camera_vitals_happy_path_single_camera (test lines 662-691) covers only one expected camera. The branch at lines 781-792 (1 < len(expected_cameras) < 5) that diffs api 'cameras' against expected_cameras and issues multicam/connect_camera/disconnect_camera calls, and the multi-camera frame-rate path at lines 809-811 (api.call('cameras/configure', MotifMulticamFrameRate=...)), are never exercised.

**Fix:** Add a check_camera_vitals test with expected_cameras=['SN1','SN2'] where the fake api 'cameras' returns ['SN2','SN3'], asserting api.call invoked with 'multicam/connect_camera/SN1' and 'multicam/disconnect_camera/SN3', and 'cameras/configure' with MotifMulticamFrameRate=camera_fr.

### [LOW] correctness — Always-true guard `if run_avisoft_command or (affinity_arg ...)` around Avisoft priority/affinity block
`behavioral_experiments.py:1119`

`run_avisoft_command` is assigned a non-empty f-string at line 1116 and never reassigned to a falsy value before line 1119, so `if run_avisoft_command or (affinity_arg and affinity_arg.strip()):` is unconditionally True and the `or` operand is dead. The body's `Start-Sleep`/`Get-Process` appendix is therefore always added regardless of whether priority/affinity were requested. The inner `if cpu_priority:` (1122) and `if affinity_arg...:` (1125) do the real gating so behavior is correct, but the outer guard is misleading vestigial logic that suggests a non-functional opt-out.

**Fix:** Drop the outer `if` (it always runs) or make it meaningful: `if cpu_priority or (affinity_arg and affinity_arg.strip()):` so the Get-Process/Start-Sleep wrapper is only appended when priority/affinity is actually requested.

### [LOW] correctness — usghflags==1574 branch compares against 1574 but writes 1572, inflating `changes` every run
`behavioral_experiments.py:936-939`

For device_num != 0 with usghflags == 1574, line 937 guards on `not math.isclose(devices['usghflags'], float(config[...]))` (compares on-disk value to 1574) but line 938 writes `str(devices['usghflags'] - 2)` = '1572'. On the next run the on-disk value is 1572, so `isclose(1574, 1572)` is still False, the branch rewrites '1572' again and increments `changes` (line 939) on every recording even though content already converged. The config content is stable but `changes` is permanently inflated and the file is rewritten each run.

**Fix:** Compute the target once and both compare and write it: `target = devices['usghflags'] - 2; if not math.isclose(target, float(config[...])): config[...] = str(target)`.

### [LOW] correctness — None dropout count overwrites initialized 0 and is JSON-serialized as null
`behavioral_experiments.py:1266`

`audio_triggerbox_sync_info_dict[log_device]['num_dropouts']` is initialized to 0 (lines 1259-1260) then unconditionally overwritten with `dropout_count` at line 1266. `count_last_recording_dropouts` returns None when the chX.log file is missing (line 69; distinct from 0 = 'no recording found', line 77). In the missing-file case the dict stores None, which json.dump writes as null at line 1277. The user is warned (line 1269) but the persisted artifact loses the int contract: any downstream reader doing arithmetic/comparison on num_dropouts breaks on null, and 'file missing' vs '0 dropouts' is conflated at the JSON layer.

**Fix:** Either coerce when persisting (`= dropout_count if dropout_count is not None else 0`) or deliberately keep None but document the null contract for downstream readers of audio_triggerbox_sync_info.json.

### [LOW] docs_clarity — Magic usghflags value 1574 (gate + the -2 write) is unexplained anywhere in the file
`behavioral_experiments.py:1257`

The dropout-counting block is gated on `usghflags != 1574` (line 1257), and the same value drives the value-minus-2 write at lines 936-939. The literal 1574 carries device-specific USGH meaning that is never explained, so a reader cannot tell what mode 1574 denotes, why it disables dropout counting, or why the non-primary device must be written as value-2.

**Fix:** Add a brief comment at line 1257 (and a parallel note near 936-939) explaining what usghflags==1574 denotes for the USGH device and why that mode skips dropout counting and requires the -2 written value.

### [LOW] performance — Camera connect/disconnect set differences recomputed twice each
`behavioral_experiments.py:785-792`

In check_camera_vitals, `set(expected_cameras) - set(temp_camera_serial_num)` is built in the len()>0 guard (line 785) and rebuilt identically in the loop (line 786); the disconnect direction repeats this (lines 790-791). Four set constructions plus two differences where two of each suffice. Cost is trivial (handfuls of serial strings) so this is a readability/recomputation cleanup, not a hot path.

**Fix:** Compute each difference once into a local, e.g. `cameras_to_connect = set(expected_cameras) - set(temp_camera_serial_num)`, iterate it directly (a non-empty set is truthy, so the explicit len()>0 guard can be dropped).

### [LOW] performance — newest_wav stats the chosen .wav twice per poll
`behavioral_experiments.py:313-314`

In verify_avisoft_is_recording's newest_wav helper, `max(wavs, key=lambda p: p.stat().st_ctime)` (line 313) already stat()s every candidate, then line 314 calls newest.stat() again to read st_size, duplicating one stat for the winner. File and poll counts are tiny so wall-clock impact is negligible, but the duplicate stat is trivially avoidable.

**Fix:** Cache the stats once, e.g. `stats = {p: p.stat() for p in wavs}; newest = max(stats, key=lambda p: stats[p].st_ctime); return newest, stats[newest].st_size`.

### [LOW] tests — check_camera_vitals browser-monitoring branches untested
`behavioral_experiments.py:814-819`

The monitor_recording block at lines 814-819 (monitor_specific_camera True -> webbrowser.open(meta['camera_info']['stream']['preview']['url']) and the else monitor_url branch) is never hit: the single-camera happy-path test sets monitor_recording=False (test line 666). webbrowser.open is never patched/asserted.

**Fix:** Add a check_camera_vitals test with monitor_recording=True and monitor_specific_camera=True, patch behavioral_experiments.webbrowser.open, make the fake api return a camera meta dict with the nested preview url, and assert webbrowser.open is called with it; add a sibling with monitor_specific_camera=False asserting monitor_url.

### [LOW] tests — disable_ethernet True branch of conduct_behavioral_recording untested
`behavioral_experiments.py:1156-1185`

full_recording_settings sets disable_ethernet=False (test line 878), so the Disable-NetAdapter (1158-1164) and Enable-NetAdapter (1177-1185) Popen blocks, their 'Ethernet DISCONNECTED'/'Ethernet RECONNECTED' messages, and the extra 20s smart_wait (1185) are never exercised.

**Fix:** Add an orchestration variant with disable_ethernet=True asserting subprocess.Popen was called with a 'Disable-NetAdapter' argument and later 'Enable-NetAdapter', and the disconnect/reconnect messages were emitted.

### [LOW] tests — Backup-destination /MIR robocopy branch untested
`behavioral_experiments.py:1530-1538`

The block at lines 1530-1538 that spawns a robocopy /MIR per backup destination when len(total_dir_name_windows) > 1 is never triggered: the orchestration test's get_custom_dir_names stub (test lines 943-945) returns a single-element windows list. Multi-lab/backup mirroring is unverified.

**Fix:** Add an orchestration test where get_custom_dir_names returns two windows destinations and assert subprocess.Popen was called with args ['robocopy', <primary>, <backup>, '/MIR'] once per extra destination.

### [LOW] tests — CoolTerm .stc Port-rewrite logic not asserted (including the insert(0) fallback)
`behavioral_experiments.py:1063-1072`

conduct_behavioral_recording rewrites the 'Port = ...' line of coolterm_config.stc (lines 1064-1068) or inserts one at the top when none exists (lines 1069-1070). The orchestration test seeds a .stc with 'Port = COM1' (test lines 841-843) but never reads the file back to assert the port was rewritten to arduino_sync_port (COM7). The rewrote_port=False insert-at-0 path is completely uncovered.

**Fix:** After the orchestration run, read coolterm_config.stc and assert it contains 'Port = COM7' with Baudrate preserved; add a case seeding a .stc with NO Port line and assert 'Port = COM7' was inserted as the first line.

### [LOW] tests — Metadata write and version-stamping in conduct_behavioral_recording not verified on disk
`behavioral_experiments.py:1218-1234`

test_conduct_behavioral_recording_orchestrates_calls asserts only session_id is stamped into the returned dict (test line 958). It does not assert the _metadata.yaml file is written to <win_dest> (lines 1224-1234), that Environment.playpen_version is set (line 1220), or the conditional usv_playpen_recording_version stamping (lines 1221-1222) gated on the key already being present.

**Fix:** In the orchestration test include 'usv_playpen_recording_version' in metadata_settings['Session'], then assert <win_dest>/sess_metadata.yaml exists with Environment.playpen_version and Session.usv_playpen_recording_version starting 'v'; add a sibling without the key asserting it is not added.

### [LOW] tests — Sync-file FileNotFoundError skip branch untested
`behavioral_experiments.py:1249-1250`

When newest_match_or_raise finds no CoolTerm .txt, lines 1249-1250 emit 'Sync file move skipped: ...' instead of raising. The orchestration test seeds an empty coolterm/Data dir but does not assert this message, nor does any test assert the successful shutil.move at lines 1247-1248.

**Fix:** Add an orchestration test where coolterm/Data has no .txt and assert a 'Sync file move skipped' message is emitted (no raise); add a sibling seeding a .txt and asserting it is moved into <win_dest>/sync/.

### [LOW] tests — modify_audio_file total==1 channelname branch and 5.09 MaxFileSize/triggertype timer branches untested
`behavioral_experiments.py:905-911`

modify_audio_file's general-key loop sets FileNameMode.channelname per audio.general.total: 0->'0' (907-908), 1->'1' (909-911); the modify_audio_file tests use total=0 fixtures and never assert channelname, so the total==1 branch, the 5.09-minute MaxFileSize branch (891-893), and the triggertype==41 timer branch (913-917) are uncovered.

**Fix:** Add a modify_audio_file test overriding audio.general.total=1 asserting FileNameMode.channelname='1' and MaxFileSize.minutes='5.09'; add another with mics_config.triggertype=41 asserting Configuration.timer == (video_session_duration+.36)*60.
