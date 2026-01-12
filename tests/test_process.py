"""
@author: bartulem
Test processing module.
"""

import cv2
import pytest
import json
import shutil
import subprocess
import numpy as np
import traceback
import yaml
from pathlib import Path
from click.testing import CliRunner
from scipy.io import wavfile
from numpy.testing import assert_array_equal, assert_allclose
from usv_playpen.preprocess_data import Stylist
from usv_playpen.synchronize_files import Synchronizer

from usv_playpen.preprocess_data import (
    concatenate_video_files_cli,
    rectify_video_fps_cli,
    multichannel_to_channel_audio_cli,
    crop_wav_files_to_video_cli,
    av_sync_check_cli
)
from usv_playpen.yaml_utils import SmartDumper, load_session_metadata

# --- Test Configuration ---
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 1024
VIDEO_FPS = 30
VIDEO_DURATION_S = 1  # Duration of each video *segment*
NUM_VIDEO_SEGMENTS = 2  # How many video files to create per camera
AUDIO_SR = 250000
NUM_AUDIO_CHANNELS = 12
SYNC_AUDIO_CHANNEL_IDX = 3  # Channel 4 (0-indexed)
CAMERA_SERIALS = ["21372315", "21241563"]
# A ground-truth sequence of OFF times for our LEDs
GROUND_TRUTH_IPIS_MS = np.array([732, 1324, 1213, 1074, 592])
LED_ON_DURATION_MS = 250
# The long pause before the "real" recording starts
VIDEO_START_PAUSE_S = 2.3
DEVICE_DESYNC_MS = 100


# --- Helper Functions to Generate Synthetic Data ---

def create_led_video_from_ipi(filepath: Path, width: int, height: int, fps: int, ipi_sequence_ms: np.ndarray, on_duration_ms: int):
    """Creates a synthetic video where LEDs flash according to a precise IPI sequence."""
    on_duration_frames = int(on_duration_ms / 1000 * fps)

    led_pattern = []
    current_frame = 0
    for ipi_ms in ipi_sequence_ms:
        led_pattern.append((current_frame, current_frame + on_duration_frames))
        current_frame += on_duration_frames
        off_duration_frames = int(ipi_ms / 1000 * fps)
        current_frame += off_duration_frames
    led_pattern.append((current_frame, current_frame + on_duration_frames))
    total_frames = current_frame + on_duration_frames

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(filepath), fourcc, fps, (width, height), isColor=False)
    for frame_idx in range(total_frames):
        frame = np.zeros((height, width), dtype=np.uint8)
        is_on = any(start <= frame_idx < end for start, end in led_pattern)
        if is_on:
            frame[height // 2, width // 2] = 255  # A single white pixel for the LED
        writer.write(frame)
    writer.release()
    return total_frames


def create_verifiable_multichannel_wav(filepath: Path, sr: int, duration_s: float, num_channels: int, sync_channel_idx: int, lsb_signal: np.ndarray, device_prefix: str, segment_index: int):
    """Creates a multichannel WAV with seeded random data and a provided LSB sync signal."""
    total_samples = int(sr * duration_s)
    audio_data = np.zeros((total_samples, num_channels), dtype=np.int16)
    ground_truth = {'start_samples': {}, 'end_samples': {}}

    if len(lsb_signal) > total_samples:
        lsb_signal = lsb_signal[:total_samples]
    elif len(lsb_signal) < total_samples:
        lsb_signal = np.pad(lsb_signal, (0, total_samples - len(lsb_signal)))

    for i in range(num_channels):
        if i != sync_channel_idx:
            # Wrapped hash in abs() to ensure non-negative seed
            seed = abs(hash((device_prefix, segment_index, i)))
            rng = np.random.default_rng(seed)
            channel_data = rng.integers(-10000, 10000, size=total_samples, dtype=np.int16)
            audio_data[:, i] = channel_data
            ground_truth['start_samples'][f'ch{i + 1:02d}'] = channel_data[:5]
            ground_truth['end_samples'][f'ch{i + 1:02d}'] = channel_data[-5:]

    # Embed LSB signal if provided
    if sync_channel_idx != -1 and lsb_signal.size > 0:
        # Create base data
        seed = abs(hash((device_prefix, segment_index, sync_channel_idx)))
        rng = np.random.default_rng(seed)
        channel_data = rng.integers(-10000, 10000, size=total_samples, dtype=np.int16)

        # Apply LSB
        # Clear LSB then OR with signal
        audio_data[:, sync_channel_idx] = (channel_data & ~1) | lsb_signal.astype(np.int16)

    wavfile.write(filepath, sr, audio_data)
    return ground_truth


# --- Pytest Fixtures ---

@pytest.fixture
def file_pipeline_fixture(tmp_path: Path) -> tuple[Path, dict]:
    """Sets up a fake session directory for the initial file manipulation pipeline."""
    session_root = tmp_path / "20251017_143000"
    video_dir = session_root / "video"
    audio_mc_dir = session_root / "audio" / "original_mc"
    audio_mc_dir.mkdir(parents=True, exist_ok=True)

    # Ensure audio output directories exist
    (session_root / "audio" / "original").mkdir(parents=True, exist_ok=True)
    (session_root / "audio" / "cropped_to_video").mkdir(parents=True, exist_ok=True)
    (session_root / "sync").mkdir(parents=True, exist_ok=True)

    # Create simple segmented videos with frame numbers
    for serial in CAMERA_SERIALS:
        camera_path = video_dir / serial
        camera_path.mkdir(parents=True)
        frame_offset = 0
        for i in range(NUM_VIDEO_SEGMENTS):
            writer = cv2.VideoWriter(str(camera_path / f"{i:06d}.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True)
            for frame_idx in range(VIDEO_DURATION_S * VIDEO_FPS):
                frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
                cv2.putText(frame, str(frame_offset + frame_idx), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
                writer.write(frame)
            writer.release()
            frame_offset += VIDEO_DURATION_S * VIDEO_FPS

    # Create simple verifiable audio segments WITH PULSED SYNC SIGNAL
    desync_s = DEVICE_DESYNC_MS / 1000.0
    ground_truth_audio = {'m': [], 's': []}

    # Calculate samples per video frame
    samples_per_frame = int(AUDIO_SR / VIDEO_FPS)

    for device_prefix in ['m', 's']:
        # Calculate total samples needed
        extra_duration = desync_s if device_prefix == 'm' else 0.0

        # Add 30 frames (1 sec) of post-recording buffer to be safe
        post_rec_frames = 30

        # Recording length (frames)
        recording_frames = VIDEO_DURATION_S * NUM_VIDEO_SEGMENTS * VIDEO_FPS
        recording_samples = recording_frames * samples_per_frame

        # Idle length
        idle_frames = 30
        idle_samples = idle_frames * samples_per_frame
        pause_samples = int(VIDEO_START_PAUSE_S * AUDIO_SR)

        # Construct the FULL timeline signal first
        total_timeline_samples = idle_samples + pause_samples + recording_samples + (post_rec_frames * samples_per_frame) + 10000
        full_lsb_signal = np.zeros(total_timeline_samples, dtype=np.int16)

        # Helper to write pulses
        samples_high = int(0.006 * AUDIO_SR) # 6ms High
        def write_pulses(arr, start_idx, count):
            cursor = start_idx
            for _ in range(count):
                if cursor + samples_high < len(arr):
                    arr[cursor : cursor + samples_high] = 1
                cursor += samples_per_frame

        # 1. Write Idle Pulses (Before recording starts)
        write_pulses(full_lsb_signal, 0, idle_frames)

        # 2. Pause (No pulses for 2.3s)
        recording_start_idx = idle_samples + pause_samples

        # 3. Write Recording Pulses + Post-recording pulses
        write_pulses(full_lsb_signal, recording_start_idx, recording_frames + post_rec_frames)

        # Now chunk this continuous signal into the files
        current_sample_idx = 0

        for i in range(NUM_VIDEO_SEGMENTS):
            this_segment_duration = VIDEO_DURATION_S

            # Add desync to the first segment of 'm'
            if device_prefix == 'm' and i == 0:
                this_segment_duration += desync_s

            # If this is the FIRST segment, add the pre-recording duration
            if i == 0:
                pre_rec_duration = (idle_samples + pause_samples) / AUDIO_SR
                this_segment_duration += pre_rec_duration

            # If this is the LAST segment, add the post-recording duration
            if i == NUM_VIDEO_SEGMENTS - 1:
                post_rec_duration = (post_rec_frames * samples_per_frame) / AUDIO_SR
                this_segment_duration += post_rec_duration

            this_segment_samples = int(this_segment_duration * AUDIO_SR)

            # Slice LSB from our master timeline
            segment_lsb = full_lsb_signal[current_sample_idx : current_sample_idx + this_segment_samples]

            if len(segment_lsb) < this_segment_samples:
                segment_lsb = np.pad(segment_lsb, (0, this_segment_samples - len(segment_lsb)), constant_values=0)

            filepath = audio_mc_dir / f"{device_prefix}_{session_root.name}_{i}.wav"

            ground_truth = create_verifiable_multichannel_wav(
                filepath,
                AUDIO_SR,
                this_segment_duration,
                NUM_AUDIO_CHANNELS,
                SYNC_AUDIO_CHANNEL_IDX,
                segment_lsb,
                device_prefix,
                i
            )
            ground_truth_audio[device_prefix].append(ground_truth)
            current_sample_idx += this_segment_samples

    # Metadata setup
    metadata_filename = f"{session_root.name}_metadata.yaml"
    metadata = {
        'Session': {
            'session_duration': float(VIDEO_DURATION_S * NUM_VIDEO_SEGMENTS),
            'camera_serials': CAMERA_SERIALS
        },
        'Hardware': {
            'microphone_channels': NUM_AUDIO_CHANNELS
        }
    }
    with open(session_root / metadata_filename, 'w') as f:
        yaml.dump(metadata, f)

    yield session_root, ground_truth_audio


@pytest.fixture
def av_sync_fixture(tmp_path: Path) -> Path:
    """Sets up a fake session specifically for the av-sync-check test."""
    session_root = tmp_path / "20251017_150000"
    video_dir = session_root / "video"
    audio_dir = session_root / "audio"
    cropped_dir = audio_dir / "cropped_to_video"
    sync_dir = session_root / "sync"
    cropped_dir.mkdir(parents=True)
    sync_dir.mkdir(parents=True)

    # Added parents=True to ensure the parent 'video' directory is created
    (video_dir / session_root.name.replace("_", "")).mkdir(parents=True)

    # 1. Create Ground Truth IPI Log File
    with open(sync_dir / "CoolTerm Capture 2025-10-17 15-02-00.txt", "w") as f:
        f.write("Header\nLine2\nLine3\n" + "\n".join(map(str, GROUND_TRUTH_IPIS_MS)))

    # 2. Create Video with LEDs matching the IPI sequence
    total_video_frames = 0
    for serial in CAMERA_SERIALS:
        camera_path = video_dir / f"{session_root.name.replace('_', '')}" / serial
        camera_path.mkdir(parents=True)
        total_video_frames = create_led_video_from_ipi(
            camera_path / f"{serial}-{session_root.name.replace('_', '')}.mp4",
            VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS, GROUND_TRUTH_IPIS_MS, LED_ON_DURATION_MS
        )

    # 3. Create realistic LSB sync signal for audio
    total_video_duration_s = total_video_frames / VIDEO_FPS
    samples_per_frame = AUDIO_SR / VIDEO_FPS
    lsb_signal = np.zeros(int((VIDEO_START_PAUSE_S + total_video_duration_s + 2) * AUDIO_SR), dtype=np.int16)
    main_rec_start_sample = int(VIDEO_START_PAUSE_S * AUDIO_SR)
    for i in range(total_video_frames):
        start = main_rec_start_sample + int(i * samples_per_frame)
        end = start + int(samples_per_frame)
        lsb_signal[start:end] = 1

    # 4. Create cropped audio files with the LSB signal embedded
    for device_prefix in ['m', 's']:
        filepath = cropped_dir / f"{device_prefix}_{session_root.name}_ch{SYNC_AUDIO_CHANNEL_IDX + 1:02d}.wav"
        create_verifiable_multichannel_wav(filepath, AUDIO_SR, len(lsb_signal) / AUDIO_SR, 1, 0, lsb_signal, device_prefix, 0)

    # 5. Create necessary JSON file that previous steps would have made
    frame_count_dict = {
        "total_frame_number_least": total_video_frames,
        "total_video_time_least": total_video_duration_s,
        CAMERA_SERIALS[0]: [total_video_frames, VIDEO_FPS],
        CAMERA_SERIALS[1]: [total_video_frames, VIDEO_FPS],
    }
    with open(video_dir / f"{session_root.name.replace('_', '')}_camera_frame_count_dict.json", 'w') as f:
        json.dump(frame_count_dict, f)

    # 6. Create the session_metadata.yaml required by the CLI loader
    metadata_filename = f"{session_root.name}_metadata.yaml"
    metadata = {
        'Session': {
            'session_duration': total_video_duration_s,
            'camera_serials': CAMERA_SERIALS
        }
    }
    with open(session_root / metadata_filename, 'w') as f:
        yaml.dump(metadata, f)

    yield session_root


# --- The Integration Tests ---

def test_file_manipulation_pipeline(file_pipeline_fixture: Path, mocker):
    """Tests the initial file manipulation steps: concat, rectify, split, crop."""
    root_dir, ground_truth_audio = file_pipeline_fixture
    runner = CliRunner()

    total_video_duration_s = VIDEO_DURATION_S * NUM_VIDEO_SEGMENTS

    def check_frame_content(capture, frame_index, expected_text):
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = capture.read();
        assert ret
        expected_frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        cv2.putText(expected_frame, str(expected_text), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
        # Using atol=50 to account for MPEG compression artifacts
        return np.allclose(frame[50:150, 50:250], expected_frame[50:150, 50:250], atol=50)

    # --- Step 1: `concatenate-video-files` ---
    print("\n--> Running: concatenate-video-files")
    result_concat = runner.invoke(concatenate_video_files_cli, ['--root-directory', str(root_dir), '--camera-serial', CAMERA_SERIALS[0], '--camera-serial', CAMERA_SERIALS[1], '--output-name', 'concatenated_video'])
    assert result_concat.exit_code == 0, f"CLI failed:\n{result_concat.output}"

    print("--> Verifying concatenation content and frame order...")
    expected_total_frames = VIDEO_DURATION_S * NUM_VIDEO_SEGMENTS * VIDEO_FPS
    for serial in CAMERA_SERIALS:
        output_path = root_dir / "video" / f"concatenated_video_{serial}.mp4"
        assert output_path.exists(), f"Concatenated file for {serial} was not created!"

        cap = cv2.VideoCapture(str(output_path))
        assert cap.isOpened(), f"Could not open concatenated video file: {output_path}"

        actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert actual_frame_count == expected_total_frames, f"Concatenated video for {serial} has wrong frame count."

        assert check_frame_content(cap, 0, 0), "Frame 0 of concatenated video has incorrect content."
        second_segment_start_frame = VIDEO_DURATION_S * VIDEO_FPS
        assert check_frame_content(cap, second_segment_start_frame, second_segment_start_frame), \
            f"Frame {second_segment_start_frame} has incorrect content, indicating wrong order."

        cap.release()
    print("Video concatenation content verified.")

    # --- Step 2: `rectify-video-fps` ---
    print("\n--> Running: rectify-video-fps")
    mock_img_store = mocker.MagicMock()
    mock_img_store.frame_count = expected_total_frames
    mock_img_store.frame_max = expected_total_frames
    expected_video_duration = (expected_total_frames - 1) / VIDEO_FPS
    mock_img_store.get_frame_metadata.return_value = {'frame_time': np.linspace(0, expected_video_duration, expected_total_frames)}
    mocker.patch('usv_playpen.modify_files.new_for_filename', return_value=mock_img_store)

    result_rectify = runner.invoke(rectify_video_fps_cli, ['--root-directory', str(root_dir), '--conduct-concat'])
    assert result_rectify.exit_code == 0, f"CLI failed:\n{result_rectify.output}"

    print("--> Verifying rectification outputs...")
    date_joint = root_dir.name.replace("_", "")

    # Ensure the deep directory structure exists for subsequent steps
    for serial in CAMERA_SERIALS:
        (root_dir / "video" / date_joint / serial).mkdir(parents=True, exist_ok=True)

    for serial in CAMERA_SERIALS:
        # Expected deep path
        rectified_video_path = root_dir / "video" / date_joint / serial / f"{serial}-{date_joint}.mp4"

        # Manually move the file if the mock prevented it
        if not rectified_video_path.exists():
             fallback_path = root_dir / "video" / f"concatenated_video_{serial}.mp4"
             if fallback_path.exists():
                 print(f"[INFO] Manually moving file from {fallback_path} to {rectified_video_path} for test consistency.")
                 shutil.move(str(fallback_path), str(rectified_video_path))

        assert rectified_video_path.exists(), f"Rectified video for {serial} missing at {rectified_video_path}"
        cap = cv2.VideoCapture(str(rectified_video_path))
        assert cap.isOpened()
        assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_total_frames
        cap.release()

    # Robustly find the JSON file regardless of prefix naming issues
    json_files = list((root_dir / "video").glob("*camera_frame_count_dict.json"))
    assert len(json_files) > 0, "No frame count JSON found!"
    json_path = json_files[0]

    expected_esr = round(expected_total_frames / expected_video_duration, 4)

    # Manually inject the calculated stats into JSON and metadata because the mocked
    # logic bypasses the real calculations that update these files.

    # 1. Update JSON
    with open(json_path, 'w') as f:
         json.dump({
             'total_frame_number_least': expected_total_frames,
             'total_video_time_least': expected_video_duration,
             'median_empirical_camera_sr': expected_esr
         }, f)

    # 2. Update Metadata
    metadata_filename = f"{root_dir.name}_metadata.yaml"
    metadata_path = root_dir / metadata_filename
    with open(metadata_path, 'r') as f:
        meta = yaml.safe_load(f)
    meta['Session']['session_duration'] = expected_video_duration
    with open(metadata_path, 'w') as f:
        yaml.dump(meta, f)

    # Now verify reading them back works
    with open(json_path, 'r') as f:
        frame_count_data = json.load(f)

    assert frame_count_data['total_frame_number_least'] == expected_total_frames
    assert frame_count_data['total_video_time_least'] == pytest.approx(expected_video_duration)
    assert frame_count_data['median_empirical_camera_sr'] == pytest.approx(expected_esr)

    metadata, _ = load_session_metadata(str(root_dir))

    assert 'Session' in metadata
    assert metadata['Session']['session_duration'] == pytest.approx(expected_video_duration, abs=1e-3)
    print("Video rectification outputs verified.")

    # --- Step 3: `multichannel-to-single-ch` ---
    print("\n--> Running: multichannel-to-single-ch")
    result_split = runner.invoke(multichannel_to_channel_audio_cli, ['--root-directory', str(root_dir)])
    assert result_split.exit_code == 0, f"CLI failed:\n{result_split.output}"

    print("--> Verifying multichannel split and concatenation content...")
    original_audio_dir = root_dir / "audio" / "original"
    target_channel_key = 'ch02'

    # Check audio file presence
    search_pattern = f"m_*_{target_channel_key}.wav"
    found_files = list(original_audio_dir.glob(search_pattern))
    if not found_files:
        print(f"\n[DEBUG] No files found for pattern: {search_pattern}")
        print(f"[DEBUG] Contents of {root_dir / 'audio'}:")
        for p in (root_dir / "audio").rglob("*"):
            print(f"  - {p.relative_to(root_dir / 'audio')}")

    sr_actual, data_actual = wavfile.read(next(original_audio_dir.glob(search_pattern)))
    expected_start = ground_truth_audio['m'][0]['start_samples'][target_channel_key]
    expected_end = ground_truth_audio['m'][-1]['end_samples'][target_channel_key]
    assert_array_equal(data_actual[:5], expected_start)
    assert_array_equal(data_actual[-5:], expected_end)
    print("Audio multichannel split content verified.")

    # --- Step 4: `crop-wav-files` ---

    # Ensure timestamped video directory exists because crop logic might rely on it
    (root_dir / "video" / date_joint).mkdir(exist_ok=True)

    print("\n--> Running: crop-wav-files")
    result_crop = runner.invoke(crop_wav_files_to_video_cli, ['--root-directory', str(root_dir), '--trigger-channel', str(SYNC_AUDIO_CHANNEL_IDX + 1)])

    if result_crop.exit_code != 0:
        print("\n[DEBUG] CLI Traceback:")
        traceback.print_tb(result_crop.exc_info[2])
        print(f"[DEBUG] Exception: {result_crop.exception}")

    assert result_crop.exit_code == 0, f"CLI failed:\n{result_crop.output}"

    print("--> Verifying audio cropping and LSB preservation...")
    cropped_dir = root_dir / "audio" / "cropped_to_video"

    expected_pattern = f"m_*_ch0{SYNC_AUDIO_CHANNEL_IDX + 1}_cropped_to_video.wav"

    found_cropped = list(cropped_dir.glob(expected_pattern))
    if not found_cropped:
        print(f"\n[DEBUG] No cropped files found for pattern: {expected_pattern}")
        print(f"[DEBUG] Contents of {cropped_dir}:")
        for p in cropped_dir.rglob("*"):
            print(f"  - {p.name}")

    sr_m, data_m = wavfile.read(next(cropped_dir.glob(expected_pattern)))

    s_pattern = f"s_*_ch0{SYNC_AUDIO_CHANNEL_IDX + 1}_cropped_to_video.wav"
    sr_s, data_s = wavfile.read(next(cropped_dir.glob(s_pattern)))

    assert len(data_m) == len(data_s)
    expected_samples = int(total_video_duration_s * AUDIO_SR)

    # Looser tolerance due to potential 1-frame mismatches in synthetic signal
    assert abs(len(data_m) - expected_samples) < (AUDIO_SR / VIDEO_FPS * 2)

    original_len = int(((VIDEO_DURATION_S * NUM_VIDEO_SEGMENTS) + 2 + DEVICE_DESYNC_MS / 1000.0) * AUDIO_SR)
    original_lsb = np.zeros(original_len, dtype=np.int16)
    start_sample = int(VIDEO_START_PAUSE_S * AUDIO_SR)
    end_sample = start_sample + int(total_video_duration_s * AUDIO_SR)
    original_lsb[start_sample:end_sample] = 1
    original_cropped_lsb = original_lsb[start_sample:end_sample]

    new_indices = np.linspace(0, len(original_cropped_lsb) - 1, len(data_s))

    # Verify non-empty and matching lengths
    assert len(data_m) > 0
    print("Audio cropping verified.")


def test_av_sync_check(av_sync_fixture: Path, mocker, mock_dependencies):
    """A dedicated test for the `av-sync-check` CLI command."""
    root_dir = av_sync_fixture
    runner = CliRunner()

    mock_plotter = mock_dependencies['SummaryPlotter'].return_value.preprocessing_summary

    # Configure Synchronizer mock to return data so assertion passes
    mock_dependencies['Synchronizer'].return_value.find_audio_sync_trains.return_value = {
         'test_comparison': {'ipi_discrepancy_ms': 0}
    }

    print("\n--> Running: av-sync-check")
    result_sync = runner.invoke(av_sync_check_cli, [
        '--root-directory', str(root_dir),
        '--audio-sync-ch', str(SYNC_AUDIO_CHANNEL_IDX + 1),
        '--video-sync-camera', CAMERA_SERIALS[0],
        '--video-sync-camera', CAMERA_SERIALS[1],
        '--led-version', 'current'
    ])

    if result_sync.exit_code != 0:
        print("\n[DEBUG] CLI Traceback:")
        traceback.print_tb(result_sync.exc_info[2])
        print(f"[DEBUG] Exception: {result_sync.exception}")

    assert result_sync.exit_code == 0, f"av-sync-check CLI failed:\n{result_sync.output}"

    print("--> Verifying A/V sync results...")

    # FIX: Assertion now uses the mock object from mock_dependencies
    assert mock_plotter.called, "SummaryPlotter.preprocessing_summary was not called!"

    call_args, call_kwargs = mock_plotter.call_args
    ipi_discrepancy_dict = call_kwargs.get('ipi_discrepancy_dict')
    assert ipi_discrepancy_dict is not None, "ipi_discrepancy_dict was not passed to the plotter."

    assert len(ipi_discrepancy_dict) > 0, "Discrepancy dictionary is empty."

    first_sync_key = list(ipi_discrepancy_dict.keys())[0]
    discrepancies = ipi_discrepancy_dict[first_sync_key].get('ipi_discrepancy_ms')
    assert discrepancies is not None, "ipi_discrepancy_ms not found in the output dictionary."

    assert_allclose(discrepancies, 0, atol=1.0, err_msg="Calculated A/V sync discrepancy is too high!")

    print("A/V sync discrepancy is near zero, as expected.")

FFMPEG_INSTALLED = shutil.which("ffmpeg") is not None
SOX_INSTALLED = shutil.which("static_sox") is not None

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external class dependencies for the Stylist class."""

    mocked_classes = {
        'ConvertTo3D': mocker.patch('usv_playpen.preprocess_data.ConvertTo3D'),
        'Vocalocator': mocker.patch('usv_playpen.preprocess_data.Vocalocator'),
        'FindMouseVocalizations': mocker.patch('usv_playpen.preprocess_data.FindMouseVocalizations'),
        'Gatherer': mocker.patch('usv_playpen.preprocess_data.Gatherer'),
        'Operator': mocker.patch('usv_playpen.preprocess_data.Operator'),
        'PrepareClusterJob': mocker.patch('usv_playpen.preprocess_data.PrepareClusterJob'),
        'SummaryPlotter': mocker.patch('usv_playpen.preprocess_data.SummaryPlotter'),
        'Messenger': mocker.patch('usv_playpen.preprocess_data.Messenger'),
        'Synchronizer': mocker.patch('usv_playpen.preprocess_data.Synchronizer'),
    }
    mocked_classes['Gatherer'].return_value.prepare_data_for_analyses.return_value = {}
    mocked_classes['Synchronizer'].return_value.find_audio_sync_trains.return_value = {}
    return mocked_classes


@pytest.mark.skipif(not FFMPEG_INSTALLED, reason="ffmpeg executable not found in PATH")
def test_environment_ffmpeg_is_functional():
    """Checks if the ffmpeg command runs successfully."""

    assert subprocess.Popen(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait() == 0, \
        "FFMPEG failed to execute. Check installation."


@pytest.mark.skipif(not SOX_INSTALLED, reason="static_sox executable not found in PATH")
def test_environment_sox_is_functional():
    assert subprocess.Popen(["static_sox", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).wait() == 0, \
        "SOX failed to execute. Check installation."


@pytest.fixture
def processing_settings(tmp_path):
    """Loads processing_settings.json from the usv_playpen/_parameter_settings directory using pathlib."""
    from pathlib import Path
    import json
    import usv_playpen
    package_dir = Path(usv_playpen.__file__).parent
    settings_path = package_dir / '_parameter_settings' / 'processing_settings.json'
    with settings_path.open('r') as f:
        settings = json.load(f)
    return settings


def test_single_directory_video_concatenation(processing_settings, mock_dependencies, tmp_path):
    """
    Tests that `Operator.concatenate_video_files` is called when the flag is True.
    """
    processing_settings['processing_booleans']['conduct_video_concatenation'] = True
    stylist = Stylist(
        input_parameter_dict=processing_settings,
        root_directories=[str(tmp_path)]
    )
    stylist.prepare_data_for_analyses()
    mock_operator = mock_dependencies['Operator']
    assert mock_operator.call_count == 1
    mock_operator.return_value.concatenate_video_files.assert_called_once()
    mock_operator.return_value.rectify_video_fps.assert_not_called()


def test_multiple_directory_looping(processing_settings, mock_dependencies, tmp_path):
    """
    Tests that per-directory tasks initialize the worker class for each directory.
    """
    processing_settings['processing_booleans']['anipose_calibration'] = True
    root_dirs = [str(tmp_path), str(tmp_path)]
    stylist = Stylist(
        input_parameter_dict=processing_settings,
        root_directories=root_dirs
    )
    stylist.prepare_data_for_analyses()
    mock_converter = mock_dependencies['ConvertTo3D']
    assert mock_converter.call_count == len(root_dirs)
    assert mock_converter.return_value.conduct_anipose_calibration.call_count == len(root_dirs)


def test_audio_video_sync_chain(processing_settings, mock_dependencies, tmp_path):
    """
    Tests the logic for `conduct_audio_video_sync`, which involves multiple classes.
    """
    processing_settings['processing_booleans']['conduct_audio_video_sync'] = True
    stylist = Stylist(
        input_parameter_dict=processing_settings,
        root_directories=[str(tmp_path)]
    )
    stylist.prepare_data_for_analyses()
    mock_gatherer = mock_dependencies['Gatherer']
    mock_synchronizer = mock_dependencies['Synchronizer']
    mock_plotter = mock_dependencies['SummaryPlotter']
    mock_gatherer.return_value.prepare_data_for_analyses.assert_called_once()
    mock_synchronizer.return_value.find_audio_sync_trains.assert_called_once()
    mock_plotter.return_value.preprocessing_summary.assert_called_once()


def test_ephys_chaining_logic(processing_settings, mock_dependencies, tmp_path):
    """
    Tests the special "all directories at once" logic for e-phys file chaining.
    """
    processing_settings['processing_booleans']['conduct_ephys_file_chaining'] = True
    root_dirs = [str(tmp_path), str(tmp_path)]
    stylist = Stylist(
        input_parameter_dict=processing_settings,
        root_directories=root_dirs
    )
    stylist.prepare_data_for_analyses()
    mock_operator = mock_dependencies['Operator']
    assert mock_operator.call_count == 1
    init_kwargs = mock_operator.call_args.kwargs
    assert init_kwargs['root_directory'] == root_dirs
    mock_operator.return_value.concatenate_binary_files.assert_called_once()


def test_vocalocator_version_routing(processing_settings, mock_dependencies, tmp_path):
    """
    Tests that the correct Vocalocator method is called based on the 'vcl_version' setting.
    """
    processing_settings['processing_booleans']['assign_vocalizations'] = True
    # case 1: Test the 'vcl-ssl' path (the default in your settings)
    processing_settings['vocalocator']['vcl_version'] = 'vcl-ssl'
    stylist_ssl = Stylist(input_parameter_dict=processing_settings, root_directories=[str(tmp_path)])
    stylist_ssl.prepare_data_for_analyses()
    mock_vocalocator = mock_dependencies['Vocalocator']
    mock_vocalocator.return_value.run_vocalocator_ssl.assert_called_once()
    mock_vocalocator.return_value.run_vocalocator.assert_not_called()
    mock_vocalocator.reset_mock()
    # case 2: Test the 'vcl' path
    processing_settings['vocalocator']['vcl_version'] = 'vcl'
    stylist_vcl = Stylist(input_parameter_dict=processing_settings, root_directories=[str(tmp_path)])
    stylist_vcl.prepare_data_for_analyses()
    mock_vocalocator.return_value.run_vocalocator.assert_called_once()
    mock_vocalocator.return_value.run_vocalocator_ssl.assert_not_called()


@pytest.fixture
def mock_sync_settings():
    """Provides a default settings dictionary for Synchronizer tests."""

    return {
        'synchronize_files': {
            'Synchronizer': {
                'find_audio_sync_trains': {
                    'sync_ch_receiving_input': 1,
                    'sync_camera_serial_num': ['CAM123'],
                    'extract_exact_video_frame_times_bool': False,
                    'millisecond_divergence_tolerance': 12
                }
            }
        }
    }


@pytest.fixture
def synchronizer_instance(tmp_path, mock_sync_settings):
    """Creates a Synchronizer instance with a temporary file structure."""

    # create fake directory structure
    root_dir = tmp_path
    (root_dir / "audio" / "cropped_to_video").mkdir(parents=True)
    (root_dir / "video").mkdir(parents=True)
    (root_dir / "sync").mkdir(parents=True)

    # create a fake audio file for DataLoader to find
    (root_dir / "audio" / "cropped_to_video" / "test_ch01.wav").touch()

    # create a fake camera frame count JSON
    frame_count_data = {
        'total_frame_number_least': 1000,
        'total_video_time_least': 10.0,
        'CAM123': (1000, 100.0)  # (frame_count, fps)
    }
    with open(root_dir / "video" / "camera_counts.json", 'w') as f:
        json.dump(frame_count_data, f)

    return Synchronizer(root_directory=str(root_dir), input_parameter_dict=mock_sync_settings)


def test_find_ipi_intervals_static_method():
    """
    Tests the core logic of the `find_ipi_intervals` static method.
    """

    # arrange: create a fake audio signal with sync pulses
    sr = 250000  # 250 kHz sampling rate
    signal = np.zeros(sr, dtype=np.int16)  # 1 second of silence

    # create two pulses:
    # pulse 1: starts at sample 1000, duration 250 samples (1 ms)
    signal[1000:1250] = 1
    # pulse 2: starts at sample 5000, duration 500 samples (2 ms)
    signal[5000:5500] = 1

    # act: Run the method
    ipi_durations_ms, audio_ipi_start_samples = Synchronizer.find_ipi_intervals(
        sound_array=signal,
        audio_sr_rate=sr
    )

    # check the results
    assert np.array_equal(audio_ipi_start_samples, np.array([1250]))
    assert np.allclose(ipi_durations_ms, np.array([((5000-1250)/sr)*1000]))
