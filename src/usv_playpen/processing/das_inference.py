"""
@author: bartulem
Run USV inference on WAV files and create annotations.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
from collections.abc import Callable
from datetime import datetime

import librosa
import matplotlib.pyplot as plt
import numpy as np
import polars as pls
import soundfile as sf
from tqdm import tqdm

from ..os_utils import (
    atomic_output_path,
    configure_path,
    first_match_or_raise,
    wait_for_subprocesses,
)
from ..time_utils import is_gui_context, smart_wait
from ..visualizations.figure_io import save_figure
from ..visualizations.plot_style import apply_plot_style
from ..yaml_utils import (
    load_session_metadata,
    read_excluded_audio_channels,
    save_session_metadata,
)

apply_plot_style()


# DAS annotation filenames are produced by `das predict` from the input WAV's
# basename, so they reach us as `<device>_<...>_<chXX>_<...>annotations.csv`,
# e.g. `m_260421185826_ch01_cropped_to_video_hpss_filtered_annotations.csv`,
# where <device> is 'm' (master) or 's' (slave) and <chXX> is the two-digit
# channel index. Both the timestamp segment before <chXX> and the
# pipeline-suffix segment after it ('cropped_to_video_hpss_filtered') contain
# underscores, so the regex anchors only on the device prefix and the channel
# token and tolerates any intervening/trailing content (the trailing `.*` is
# what lets the channel sit anywhere before `annotations.csv`, not just
# immediately before it).
_DAS_ANNOTATION_FILE_RE = re.compile(r"^([ms])_.*_(ch\d{2})_.*annotations\.csv$")


def _remerge_from_consensus(merged: list, min_duration_s: float, span_factor: float,
                            max_dissenting_channels: int, min_agreeing_channels: int,
                            max_depth: int, min_gap_s: float = 0.0, _depth: int = 0) -> tuple:
    """
    Description
    -----------
    Re-merge intervals whose extent is dictated by one channel contradicting the
    rest, and return the corrected interval list.

    The greedy union takes the outermost edge across channels, which is what
    preserves the faint onsets and offsets only the nearest microphones register.
    The same rule means a single channel that fuses a run of calls into one long
    detection drags every other channel's boundaries out with it. Observed in
    session 20250919_145712, where 23 of 24 channels resolved seven separate
    calls between 0.9 s and 1.9 s while one channel emitted a single 897 ms
    detection: the merged USV became 901 ms, and re-inference did not change it
    because the fused detection is in the raw annotation.

    An interval is only reconsidered when all of the following hold, so that
    dense-but-genuine runs of overlapping calls are left untouched:

    1. it is longer than ``min_duration_s`` -- ordinary calls are never examined;
    2. at least one and at most ``max_dissenting_channels`` contributing channels
       have a segment longer than ``span_factor`` times the median
       longest-segment across the contributing channels;
    3. at least ``min_agreeing_channels`` channels remain after setting those
       aside;
    4. re-merging the remaining channels actually yields more than one interval.

    Condition 4 matters: where the long channel agrees with everyone else, the
    consensus re-merge reproduces one interval and nothing is changed. A channel
    set aside for the boundary is still credited in ``chs_detected`` for every
    sub-interval its original segment covers, because it did hear the call --
    it merely failed to separate it -- and dropping it would understate
    ``chs_count`` for the downstream noise rejection.

    This is deliberately far narrower than the coverage-watershed merge reverted
    in c9882bb, which trimmed every USV to a fraction of its peak channel count
    and clipped real edges throughout. This touches only intervals where one
    channel contradicts a clear majority: 3 of 1043 on the session above.

    Parameters
    ----------
    merged : list
        Merged interval dicts, each carrying ``start``, ``stop``, ``chs_detected``
        (a set) and ``segments`` (the contributing ``(start, stop, channel)`` tuples).
    min_duration_s : float
        Only intervals longer than this are examined.
    span_factor : float
        A channel dominates when its longest segment exceeds this multiple of the
        median longest-segment across contributing channels.
    max_dissenting_channels : int
        Fire only when at most this many channels dominate.
    min_agreeing_channels : int
        Fire only when at least this many channels remain once dissenters are set aside.

    Returns
    -------
    (corrected, n_split) : tuple[list, int]
        The corrected interval list and the number of intervals that were split.
    """

    corrected = []
    n_split = 0
    for interval in merged:
        segments = interval['segments']
        if interval['stop'] - interval['start'] <= min_duration_s:
            corrected.append(interval)
            continue

        longest_per_channel = {}
        for seg_start, seg_stop, seg_channel in segments:
            span = seg_stop - seg_start
            if span > longest_per_channel.get(seg_channel, 0.0):
                longest_per_channel[seg_channel] = span
        median_longest = statistics.median(longest_per_channel.values())
        dissenting = {channel for channel, span in longest_per_channel.items()
                    if span > span_factor * median_longest}
        retained = [seg for seg in segments if seg[2] not in dissenting]

        if (not dissenting or len(dissenting) > max_dissenting_channels
                or len({seg[2] for seg in retained}) < min_agreeing_channels):
            corrected.append(interval)
            continue

        sub_intervals = _greedy_merge_segments(sorted(retained, key=lambda seg: seg[0]))

        # A cut needs a gap the segmenter could actually have produced. DAS is run
        # with --segment-fillgap 0.015, so it closes any gap under 15 ms WITHIN a
        # channel: measured over 221,522 within-channel gaps in six sessions, 99.87%
        # are >= 15 ms and the 1st percentile is 21 ms. A finer gap in the merged
        # output therefore cannot come from the segmentation at all -- it is
        # manufactured by taking the union across channels and cutting it again, where
        # different channels' detections happen to stop at slightly different instants.
        # Left alone it puts boundaries inside calls: 20250928_172408 at 565.074 s was
        # cut into 18 ms and 133 ms pieces divided by 0.1 ms, mid-call, and
        # 20251004_162927 at 303.498 s lost a call to gaps of 2.4 and 11.1 ms.
        # Matching the floor to the segmenter's own resolution is principled rather
        # than tuned, and it stays discriminating: at 20250928_175135 303.130 s a
        # 17.1 ms gap survives while a 4.9 ms one closes.
        if min_gap_s > 0 and len(sub_intervals) > 1:
            fused = [sub_intervals[0]]
            for piece in sub_intervals[1:]:
                if piece['start'] - fused[-1]['stop'] < min_gap_s:
                    fused[-1]['stop'] = max(fused[-1]['stop'], piece['stop'])
                    fused[-1]['chs_detected'] |= piece['chs_detected']
                    fused[-1]['segments'].extend(piece['segments'])
                else:
                    fused.append(piece)
            sub_intervals = fused

        if len(sub_intervals) < 2:
            corrected.append(interval)
            continue

        n_split += 1
        # Only the INTERNAL cuts come from the consensus. The outer edges stay the
        # union's, because setting a channel aside for the split also removes its
        # contribution to the interval's own start and stop -- and that contribution
        # is the faint onset or offset only the nearest microphone registered, which
        # the union exists to keep. Measured before this was fixed: 20250919_145712
        # at 100.059 s had its start pulled 11.2 ms later, and 20240311_143803 at
        # 1102.237 s lost 27.1 ms off its end.
        sub_intervals[0]['start'] = interval['start']
        sub_intervals[-1]['stop'] = interval['stop']
        for sub in sub_intervals:
            # credit a set-aside channel wherever its own segment covers this sub-interval
            for seg_start, seg_stop, seg_channel in segments:
                if seg_channel in dissenting and seg_start < sub['stop'] and sub['start'] < seg_stop:
                    sub['chs_detected'].add(seg_channel)
        # Recurse. A single pass tests dissent against the median longest-segment of
        # the WHOLE interval, so inside a long one a channel fusing just two adjacent
        # calls is invisible: in 20240311_143803 at 300.816 s the interval spans 779 ms
        # with a 96 ms median, and the channels bridging two 30-40 ms calls sit at
        # 102-107 ms -- only 1.1x, far under the threshold. Re-testing each sub-interval
        # against its own, much shorter, median exposes them. Depth is capped because
        # each level can only shorten intervals, but a pathological channel set could
        # otherwise recurse further than is meaningful.
        deeper, deeper_splits = _remerge_from_consensus(
            sub_intervals, min_duration_s, span_factor, max_dissenting_channels,
            min_agreeing_channels, max_depth, min_gap_s=min_gap_s, _depth=_depth + 1
        ) if _depth < max_depth else (sub_intervals, 0)
        n_split += deeper_splits
        corrected.extend(deeper)

    return corrected, n_split


def _greedy_merge_segments(sorted_segments: list) -> list:
    """
    Description
    -----------
    Merge time-sorted ``(start, stop, channel)`` segments into overlapping-interval
    dicts by greedy union.

    Two intervals overlap (open-ended) when ``a_start < b_stop and b_start < a_stop``;
    because the input is sorted by start time, only the running stop needs comparing.
    Each interval retains the segments that formed it so a later pass can reason about
    which channels set its boundaries.

    Parameters
    ----------
    sorted_segments : list
        ``(start_seconds, stop_seconds, channel)`` tuples, sorted by start time.

    Returns
    -------
    merged (list)
        Interval dicts with ``start``, ``stop``, ``chs_detected``, ``segments`` and
        the amplitude placeholders the summary writer expects.
    """

    merged = []
    for seg_start, seg_stop, seg_channel in sorted_segments:
        if merged and seg_start < merged[-1]['stop']:
            merged[-1]['stop'] = max(merged[-1]['stop'], seg_stop)
            merged[-1]['chs_detected'].add(seg_channel)
            merged[-1]['segments'].append((seg_start, seg_stop, seg_channel))
        else:
            merged.append({
                'start': seg_start,
                'stop': seg_stop,
                'chs_detected': {seg_channel},
                'segments': [(seg_start, seg_stop, seg_channel)],
                'peak_amp_ch': 0.0,
                'mean_amp_ch': 0.0,
            })
    return merged


def _write_usv_summary_csv(merged: list, out_path: pathlib.Path) -> None:
    """Write the per-session USV summary CSV from a list of merged interval dicts.

    Single source of truth for the summary schema, shared by all three
    ``summarize_das_findings`` branches (noise-filtered with >1 USV, the single-USV
    case, and filtering-disabled) so the column set / formatting can never drift
    between them.

    Parameters
    ----------
    merged : list
        List of merged USV interval dicts, each carrying ``start``/``stop``/
        ``peak_amp_ch``/``mean_amp_ch``/``chs_count``/``chs_detected`` keys.
    out_path : pathlib.Path
        Destination path for the ``*_usv_summary.csv`` file.

    Returns
    -------
    (None)
    """
    pls.DataFrame({
        "usv_id": [f"{_num:04d}" for _num in range(len(merged))],
        "start": [u['start'] for u in merged],
        "stop": [u['stop'] for u in merged],
        "duration": [u['stop'] - u['start'] for u in merged],
        "peak_amp_ch": [float(u['peak_amp_ch']) for u in merged],
        "mean_amp_ch": [float(u['mean_amp_ch']) for u in merged],
        "chs_count": [float(u['chs_count']) for u in merged],
        "chs_detected": [str(u['chs_detected']) for u in merged],
        "emitter": [None] * len(merged),
    }).write_csv(file=out_path)


def _ladder_gap_leading_stops(
    start_seconds: np.ndarray,
    stop_seconds: np.ndarray,
    sampling_rate: int,
    stride_samples: int,
    tolerance_samples: int,
    max_rung: int,
) -> np.ndarray:
    """
    Description
    -----------
    Finds the seam-ladder gaps in one channel's raw DAS annotations and returns
    the stop time of the call leading each such gap.

    The legacy (non-overlapping) DAS window tiling judged each stitched window
    span without acoustic context from its neighbours, so a real inter-call
    pause whose faint flanking call edges straddled window seams was emitted
    with both endpoints ON the seam grid: the measured gap width is exactly
    ``k * stride_samples + 1`` samples (the +1 is the segment-extraction
    boundary convention). Real pauses cannot repeat to the sample, so a gap
    within ``tolerance_samples`` of that fingerprint for k = 1..``max_rung``
    identifies a seam-snapped pair with essentially no false positives at the
    raw-annotation level.

    Parameters
    ----------
    start_seconds (np.ndarray)
        Segment start times (s) of one channel's raw DAS annotations, sorted
        ascending, noise rows removed.
    stop_seconds (np.ndarray)
        Matching segment stop times (s).
    sampling_rate (int)
        Audio sampling rate (Hz) the annotation times refer to.
    stride_samples (int)
        Window-stitching stride (in samples) of the legacy DAS model that
        produced the annotations (settings key
        ``seam_repair_legacy_stride_samples``).
    tolerance_samples (int)
        Maximum deviation (in samples) from the exact ladder fingerprint for a
        gap to count as seam-snapped.
    max_rung (int)
        Highest ladder multiple k to test.

    Returns
    -------
    leading_stops (np.ndarray)
        Stop times (s) of the segments immediately preceding each seam-ladder
        gap; empty when the channel has fewer than two segments or no ladder
        gaps.
    """
    if len(start_seconds) < 2:
        return np.empty(0, dtype=float)
    gap_samples = np.round((start_seconds[1:] - stop_seconds[:-1]) * sampling_rate).astype(np.int64)
    rung = np.round(gap_samples / stride_samples).astype(np.int64)
    on_ladder = (
        (rung >= 1)
        & (rung <= max_rung)
        & (np.abs(gap_samples - (rung * stride_samples + 1)) <= tolerance_samples)
    )
    return stop_seconds[:-1][on_ladder]


def _repaired_facing_edges(
    annotation_starts: np.ndarray,
    annotation_stops: np.ndarray,
    snippet_offset: float,
    stop_first: float,
    start_second: float,
    mid_first: float,
    mid_second: float,
    max_boundary_shift: float,
) -> tuple[float, float] | None:
    """
    Description
    -----------
    Extracts the corrected facing edges of one seam-snapped USV pair from the
    overlap-tiled DAS re-detection of its audio snippet.

    The re-detected segment containing the leading call's midpoint provides the
    new stop of the first call; the segment containing the trailing call's
    midpoint provides the new start of the second call. Because the seam
    artifact only ever *clips* call edges (the true edges lie outward from the
    recorded ones, inside the recorded gap), inward suggestions -- which arise
    from single-channel-versus-multichannel-merge convention differences, not
    from the artifact -- are clamped to the recorded boundary. Pairs the
    re-detection joins into one segment, fails to detect, or would move by more
    than ``max_boundary_shift`` are rejected (returned as None) and left
    untouched by the caller.

    Parameters
    ----------
    annotation_starts (np.ndarray)
        Segment start times (s, snippet-relative) of the snippet re-detection,
        noise rows removed.
    annotation_stops (np.ndarray)
        Matching segment stop times (s, snippet-relative).
    snippet_offset (float)
        Absolute session time (s) of the snippet's first sample; added to the
        snippet-relative annotation times.
    stop_first (float)
        Recorded stop (s) of the leading call in the summary.
    start_second (float)
        Recorded start (s) of the trailing call in the summary.
    mid_first (float)
        Midpoint (s) of the leading call's recorded extent.
    mid_second (float)
        Midpoint (s) of the trailing call's recorded extent.
    max_boundary_shift (float)
        Maximum permitted outward correction (s) per edge; mechanistically one
        legacy DAS window tile.

    Returns
    -------
    new_edges (tuple[float, float] | None)
        ``(new_stop_first, new_start_second)`` in absolute session time, or
        None when the pair cannot be repaired safely.
    """
    seg_starts = annotation_starts + snippet_offset
    seg_stops = annotation_stops + snippet_offset
    first_hits = np.flatnonzero((seg_starts <= mid_first) & (seg_stops >= mid_first))
    second_hits = np.flatnonzero((seg_starts <= mid_second) & (seg_stops >= mid_second))
    if len(first_hits) == 0 or len(second_hits) == 0:
        return None
    if first_hits[0] == second_hits[0]:
        return None
    new_stop_first = max(float(seg_stops[first_hits[0]]), stop_first)
    new_start_second = min(float(seg_starts[second_hits[0]]), start_second)
    if (new_stop_first - stop_first) > max_boundary_shift:
        return None
    if (start_second - new_start_second) > max_boundary_shift:
        return None
    return new_stop_first, new_start_second


class FindMouseVocalizations:
    def __init__(
        self,
        root_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the FindMouseVocalizations class.

        Parameters
        ----------
        root_directory (str)
            Root directory for data; defaults to None.
        input_parameter_dict (dict)
            Processing parameters; defaults to None.
        message_output (function)
            Function to output messages; defaults to None.

        Returns
        -------
        None
        """

        if input_parameter_dict is None:
            with open(
                pathlib.Path(__file__).parent.parent / "_parameter_settings/processing_settings.json"
            ) as json_file:
                _defaults = json.load(json_file)
            self.input_parameter_dict = _defaults["usv_inference"]["FindMouseVocalizations"]
        else:
            self.input_parameter_dict = input_parameter_dict["usv_inference"]["FindMouseVocalizations"]

        self.root_directory = root_directory
        self.message_output = message_output or print

        self.app_context_bool = is_gui_context()

    def das_command_line_inference(self) -> None:
        """
        Description
        -----------
        This method takes WAV files as input and runs DAS inference on them to generate
        tentative USV segments in the recording.

        Parameters
        ----------

        Returns
        -------
        .csv annotation files
            CSV files w/ onsets and offsets of all detected USV segments,
            shape: (N_USV, VOC_TYPE, START_SEC, END_SEC).
        """

        self.message_output(
            f"DAS inference started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}. Please be patient, this can take >5 min/file."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        das_conda_name = self.input_parameter_dict["das_command_line_inference"][
            "das_conda_env_name"
        ]
        model_base = str(pathlib.Path(configure_path(self.input_parameter_dict['das_command_line_inference']['das_model_directory'])) / self.input_parameter_dict['das_command_line_inference']['model_name_base'])
        thresh = self.input_parameter_dict["das_command_line_inference"][
            "segment_confidence_threshold"
        ]
        min_len = self.input_parameter_dict["das_command_line_inference"][
            "segment_minlen"
        ]
        fill_gap = self.input_parameter_dict["das_command_line_inference"][
            "segment_fillgap"
        ]
        save_format = self.input_parameter_dict["das_command_line_inference"][
            "output_file_type"
        ]

        hpss_dir = pathlib.Path(self.root_directory) / "audio" / "hpss_filtered"

        # run inference
        for one_file in sorted(hpss_dir.glob("*.wav*")):
            self.message_output(
                f"Running DAS inference on: {one_file.name}"
            )
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

            conda_exe = os.environ.get('CONDA_EXE', 'conda')
            clean_env = os.environ.copy()
            clean_env.pop('PYTHONHOME', None)
            inference_subp = subprocess.Popen(
                args=[conda_exe, 'run', '--no-capture-output', '-n', das_conda_name, 'das', 'predict', one_file, model_base,
                      '--segment-thres', str(thresh), '--segment-minlen', str(min_len),
                      '--segment-fillgap', str(fill_gap), '--save-format', str(save_format)],
                cwd=hpss_dir,
                env=clean_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                shell=False,
            )

            # DAS inference on a long recording can take hours on CPU, so the
            # per-file budget is generous. A 12 h ceiling still catches a
            # genuinely hung process (GPU lost, file-descriptor deadlock)
            # rather than letting it sit indefinitely.
            wait_for_subprocesses(
                subps=[inference_subp],
                max_seconds=12 * 60 * 60,
                label=f"DAS inference on {pathlib.Path(one_file).name}",
                poll_interval_s=5,
                message_output=self.message_output,
                raise_on_nonzero=False,
                raise_on_timeout=False,
            )

        # create save directory if it doesn't exist
        das_dir = pathlib.Path(self.root_directory) / "audio" / "das_annotations"
        das_dir.mkdir(parents=True, exist_ok=True)

        # move annotation files to save directory
        # NB: materialize the directory listing before moving — moving entries
        # out of `hpss_dir` while its iterator is live can skip files on some
        # filesystems. The suffix is matched with `endswith` (not a substring)
        # so only true `.{save_format}` outputs are moved, never a name that
        # merely contains that token.
        for one_file in sorted(hpss_dir.iterdir()):
            if one_file.name.endswith(f".{save_format}"):
                shutil.move(src=one_file, dst=das_dir / one_file.name)

    def summarize_das_findings(self) -> None:
        """
        Description
        -----------
        This method takes CSV files generated by DAS inference and creates a summary file,
        containing information about individual USV segment start and stop times, duration,
        peak amplitude channel, mean amplitude channel, total number of channels it was
        detected on, list of channels it was detected on, and emitter ID.

        Parameters
        ----------

        Returns
        -------
        .csv summary file
            CSV file w/ information about all detected USV segments,
            shape: (N_USV, USV_ID, START, STOP, DURATION, PEAK_AMP_CH,
            MEAN_AMP_CH, CHS_COUNT, CHS_DETECTED, EMITTER).
        """

        self.message_output(
            f"DAS summary started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        ch_conversion_dict = {
            "m_ch01": 0,
            "m_ch02": 1,
            "m_ch03": 2,
            "m_ch04": 3,
            "m_ch05": 4,
            "m_ch06": 5,
            "m_ch07": 6,
            "m_ch08": 7,
            "m_ch09": 8,
            "m_ch10": 9,
            "m_ch11": 10,
            "m_ch12": 11,
            "s_ch01": 12,
            "s_ch02": 13,
            "s_ch03": 14,
            "s_ch04": 15,
            "s_ch05": 16,
            "s_ch06": 17,
            "s_ch07": 18,
            "s_ch08": 19,
            "s_ch09": 20,
            "s_ch10": 21,
            "s_ch11": 22,
            "s_ch12": 23,
        }

        # Hardware-excluded microphones for this session (per-session metadata
        # record, ``Equipment -> audio_Avisoft -> excluded_channels``; empty
        # for healthy sessions). Their annotation files are skipped in Phase 1,
        # so excluded channels never contribute detections to the merge.
        excluded_channels = read_excluded_audio_channels(self.root_directory, logger=self.message_output)
        if excluded_channels:
            self.message_output(
                f"Excluding audio channel(s) {excluded_channels} per session metadata."
            )

        session_id = pathlib.Path(self.root_directory).name

        annot_dir = pathlib.Path(self.root_directory) / "audio" / "das_annotations"
        das_annotation_files = sorted(annot_dir.glob("*.csv")) if annot_dir.is_dir() else []
        if not das_annotation_files:
            self.message_output(
                f"No DAS annotations found in directory: {self.root_directory}. Skipping summary generation."
            )
            return

        try:
            # Phase 1: load all channel CSVs, filter noise, collect flat segment list
            # Each entry is (start_seconds, stop_seconds, channel_numeric_index).
            all_segments = []
            for one_file in das_annotation_files:
                m = _DAS_ANNOTATION_FILE_RE.match(one_file.name)
                if m is None:
                    self.message_output(
                        f"Skipping {one_file.name}: filename does not match expected "
                        f"DAS annotation pattern '<device>_..._<chXX>_...annotations.csv'."
                    )
                    continue
                file_id = f"{m.group(1)}_{m.group(2)}"
                if file_id not in ch_conversion_dict:
                    self.message_output(
                        f"Skipping {one_file.name}: unrecognized device/channel '{file_id}'."
                    )
                    continue
                if file_id in excluded_channels:
                    self.message_output(
                        f"Skipping {one_file.name}: channel '{file_id}' is excluded per session metadata."
                    )
                    continue
                channel_df = pls.read_csv(source=str(one_file))
                channel_df = channel_df.filter(pls.col("name") != "noise")
                ch_num = ch_conversion_dict[file_id]
                for seg_row in channel_df.iter_rows(named=True):
                    all_segments.append(
                        (seg_row["start_seconds"], seg_row["stop_seconds"], ch_num)
                    )

            # Phase 2: sort all segments by start time
            all_segments.sort(key=lambda seg: seg[0])

            # Phase 3: greedy interval merge across all channels.
            # Two intervals (a_start, a_stop) and (b_start, b_stop) overlap (open-ended) when:
            #   a_start < b_stop  and  b_start < a_stop
            # Since segments are sorted by start, only the running stop needs comparing.
            # Each merged entry is a dict with start, stop, chs_detected (set), and placeholder fields.
            #
            # This replaces the coverage-watershed merge introduced in f807c41. That merge
            # trimmed every USV to the span where at least valley_frac of its peak channel
            # count still agreed, which deletes the faint onsets and offsets heard by only
            # the nearest few microphones: measured against the local noise floor, watershed
            # edges leave 5-19% of a call's own peak energy outside the boundary, where the
            # union leaves none. Its premise -- that a long merged interval is one jittery
            # channel chaining distinct calls -- does not hold either; those intervals are
            # dense runs of genuinely overlapping calls, with no silence to cut at.
            merged = _greedy_merge_segments(all_segments)

            # The union's one failure mode: a channel that fuses a run of calls into
            # a single detection drags everyone else's boundaries out with it, since
            # the outermost edge wins. Only intervals where one channel contradicts a
            # clear majority are reconsidered -- dense runs that every channel agrees
            # are long stay exactly as merged. See _remerge_from_consensus.
            merge_params = self.input_parameter_dict["summarize_das_findings"]
            if merge_params["consensus_remerge_bool"]:
                merged, n_split = _remerge_from_consensus(
                    merged,
                    min_duration_s=merge_params["consensus_remerge_min_duration_s"],
                    span_factor=merge_params["consensus_remerge_span_factor"],
                    max_dissenting_channels=merge_params["consensus_remerge_max_dissenting_channels"],
                    min_agreeing_channels=merge_params["consensus_remerge_min_agreeing_channels"],
                    max_depth=merge_params["consensus_remerge_max_depth"],
                    min_gap_s=merge_params["consensus_remerge_min_gap_s"],
                )
                if n_split > 0:
                    self.message_output(
                        f"Re-merged {n_split} interval(s) from the agreeing channels, where "
                        f"one channel contradicted the majority."
                    )

            # Reject implausibly long intervals before Phase 4 looks at anything.
            # Phase 4 asks whether a detection correlates across channels, which
            # broadband noise does perfectly well: session 20240229_163242 carried six
            # "USVs" with a median duration of 3,074 ms, on all 24 channels, and every
            # one survived the correlation and coherence checks. Duration is the
            # discriminator those checks cannot supply -- no mouse USV lasts a second.
            #
            # The default is set from the cohort rather than assumed: over 439,301 USVs
            # in 246 summarized sessions the median is 86 ms, p99 is 371 ms, p99.9 is
            # 617 ms and p99.99 is 896 ms. A 1,000 ms cutoff is ~5x the p90, past any
            # plausible call or tightly-overlapping cluster, and removes 21 intervals
            # in 439,301 (0.005%) -- of which the 6 longest are that one noise session.
            duration_params = self.input_parameter_dict["summarize_das_findings"]
            if duration_params["max_usv_duration_bool"] and merged:
                max_duration_s = duration_params["max_usv_duration_s"]

                # Before deleting anything, give an over-length interval one more
                # attempt with the dissent limit relaxed. 20240311_143803 at 357.531 s
                # is 1,094 ms of genuine calls that stayed whole only because FOUR of
                # its channels smeared and consensus_remerge_max_dissenting_channels
                # is 2 -- the merge declined, and the gate then destroyed real data.
                # Relaxing the limit globally is not the answer: over twelve intervals
                # from confirmed non-playback sessions, judged against spectrograms, a
                # limit of 4 was better in six and worse in five, so neither value is
                # right on ordinary data. Applied only here it changed 0 of 13,851
                # intervals across eleven such sessions while splitting 357.531 into
                # seven pieces, the longest 303 ms. Deleting real calls is the worst
                # outcome available, so the retry is worth having on that path alone.
                if merge_params["consensus_remerge_bool"]:
                    rescued = []
                    for usv in merged:
                        if usv['stop'] - usv['start'] <= max_duration_s:
                            rescued.append(usv)
                            continue
                        retry, _ = _remerge_from_consensus(
                            [usv], merge_params["consensus_remerge_min_duration_s"],
                            merge_params["consensus_remerge_span_factor"],
                            merge_params["consensus_remerge_rescue_dissenting_channels"],
                            merge_params["consensus_remerge_min_agreeing_channels"],
                            merge_params["consensus_remerge_max_depth"],
                            merge_params["consensus_remerge_min_gap_s"],
                        )
                        rescued.extend(retry)
                    if len(rescued) != len(merged):
                        self.message_output(
                            f"Re-merged {len(rescued) - len(merged)} further piece(s) from "
                            f"interval(s) that would otherwise have exceeded {max_duration_s} s."
                        )
                        merged = rescued

                too_long = [usv for usv in merged if usv['stop'] - usv['start'] > max_duration_s]
                if too_long:
                    merged = [usv for usv in merged if usv['stop'] - usv['start'] <= max_duration_s]
                    self.message_output(
                        f"Rejected {len(too_long)} interval(s) longer than {max_duration_s} s "
                        f"(longest {max((u['stop'] - u['start']) for u in too_long):.2f} s); "
                        f"{len(merged)} remain."
                    )

            # Convert channel sets to sorted lists and compute counts
            for usv in merged:
                del usv['segments']
                usv['chs_detected'] = sorted(usv['chs_detected'])
                usv['chs_count'] = len(usv['chs_detected'])

            n_usv = len(merged)
            self.message_output(
                f"Merged {n_usv} USV intervals from {len(all_segments)} raw detections across {len({seg[2] for seg in all_segments})} channels."
            )
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)

            # Whether to run the Phase-4 putative-noise rejection (amplitude +
            # spectrogram correlation/coherence checks). When False, every merged
            # interval is kept and written to the summary CSV as-is.
            filter_putative_noise_bool = self.input_parameter_dict[
                "summarize_das_findings"
            ]["filter_putative_noise_bool"]

            # Phase 4: amplitude + spectrogram quality checks
            # (skipped entirely when filter_putative_noise_bool is False)
            if filter_putative_noise_bool and n_usv > 0:
                audio_file_loc = first_match_or_raise(
                    root=pathlib.Path(self.root_directory) / "audio" / "hpss_filtered",
                    pattern="*.mmap",
                    label="concatenated audio mmap",
                )
                audio_file_name = audio_file_loc.name
                # The mmap filename encodes its array metadata as the last four
                # underscore-separated tokens, in the trailing layout
                # '..._<sampling_rate>_<sample_num>_<channel_num>_<dtype>.mmap'.
                # Parsing right-to-left: [-1][:-5] is the dtype with the trailing
                # '.mmap' (5 chars) stripped, [-2] the channel count, [-3] the
                # sample count, [-4] the sampling rate.
                data_type, channel_num, sample_num, audio_sampling_rate = (
                    audio_file_name.split("_")[-1][:-5],
                    int(audio_file_name.split("_")[-2]),
                    int(audio_file_name.split("_")[-3]),
                    int(audio_file_name.split("_")[-4]),
                )
                audio_file_data = np.memmap(
                    filename=audio_file_loc,
                    mode="r",
                    dtype=data_type,
                    shape=(sample_num, channel_num),
                )

                len_win_signal = self.input_parameter_dict["summarize_das_findings"][
                    "len_win_signal"
                ]
                low_freq_cutoff = self.input_parameter_dict["summarize_das_findings"][
                    "low_freq_cutoff"
                ]
                noise_corr_cutoff = self.input_parameter_dict["summarize_das_findings"][
                    "noise_corr_cutoff_min"
                ]
                coherence_cutoff = self.input_parameter_dict["summarize_das_findings"][
                    "coherence_cutoff_min"
                ]
                coherence_channel_count = self.input_parameter_dict["summarize_das_findings"][
                    "coherence_channel_count"
                ]
                frequency_resolution = audio_sampling_rate / len_win_signal
                lower_bin = int(np.floor(low_freq_cutoff / frequency_resolution))
                # Defensive: if lower_bin sits past the STFT's freq axis every
                # in-band slice below would be empty and corrcoef would raise an
                # obscure broadcasting error; surface the real problem here.
                if lower_bin >= len_win_signal // 2 + 1:
                    msg = (
                        f"lower_bin ({lower_bin}) exceeds STFT freq-axis "
                        f"length ({len_win_signal // 2 + 1}); "
                        "check `low_freq_cutoff` vs `len_win_signal` / sampling rate"
                    )
                    raise ValueError(msg)

                # Channels eligible for the amplitude scan and the coherence
                # ranking: every audio channel minus the session's
                # hardware-excluded ones (see the metadata read above).
                excluded_channel_indices = {
                    ch_conversion_dict[ch_name]
                    for ch_name in excluded_channels
                    if ch_name in ch_conversion_dict
                }
                eligible_channel_indices = [
                    ch for ch in range(channel_num) if ch not in excluded_channel_indices
                ]

                condition_0_list = np.full(shape=n_usv, fill_value=False)
                mean_signal_correlations = np.full(n_usv, np.nan)
                spatial_coherences = np.full(n_usv, np.nan)

                for i, usv in tqdm(
                    enumerate(merged),
                    desc="Computing spectrogram correlations/coherences in progress...",
                    total=n_usv,
                    position=0,
                    leave=True,
                ):
                    start_usv = int(np.floor(usv['start'] * audio_sampling_rate))
                    stop_usv = int(np.ceil(usv['stop'] * audio_sampling_rate))
                    # Materialize the USV sample window once: each fresh memmap index
                    # re-reads the same byte range from disk, and this window is used
                    # twice (peak/mean amplitude channels + the STFT input below).
                    window = np.asarray(audio_file_data[start_usv:stop_usv, :])[:, eligible_channel_indices]
                    # Data-integrity guard: an all-zero window means the
                    # concatenated mmap has no audio where DAS detected calls
                    # (e.g. a truncated/partial mmap write). Classifying such
                    # windows would silently discard real calls as noise --
                    # fail loud instead so the corrupt intermediate is fixed.
                    if window.size and not window.any():
                        msg = (
                            f"All-zero audio window at USV {i} "
                            f"({usv['start']:.3f}-{usv['stop']:.3f} s) in the concatenated "
                            f"mmap '{audio_file_name}' despite DAS detections there; the "
                            "mmap is likely truncated or corrupt -- regenerate it "
                            "(concatenate-audio-files) before summarizing."
                        )
                        raise ValueError(msg)
                    # Peak/mean amplitude channels are searched over the ELIGIBLE
                    # channels only, then mapped back to absolute channel indices
                    # -- a loud artifact on an excluded channel must not become
                    # the peak-amplitude channel and fail condition_0 below.
                    peak_amp_ch = eligible_channel_indices[
                        int(np.unravel_index(np.argmax(window), window.shape)[1])
                    ]
                    mean_amp_ch = eligible_channel_indices[
                        int(np.argmax(np.abs(window).mean(axis=0)))
                    ]
                    usv['peak_amp_ch'] = int(peak_amp_ch)
                    usv['mean_amp_ch'] = int(mean_amp_ch)
                    usv_detected_chs = usv['chs_detected']

                    # remove USV segments if they don't appear on both peak and mean amplitude channels; this is clearly noise
                    condition_0_list[i] = (
                        peak_amp_ch not in usv_detected_chs
                        or mean_amp_ch not in usv_detected_chs
                    )

                    # One in-band magnitude spectrogram per eligible channel; the
                    # detected-channel correlation and the top-K spatial coherence
                    # are both read off this single STFT.
                    spectrogram_all_eligible = np.abs(
                        librosa.stft(
                            window.astype("float32").T,
                            n_fft=len_win_signal,
                        )
                    )[:, lower_bin:, :]
                    flattened_spectrograms = spectrogram_all_eligible.reshape(
                        len(eligible_channel_indices), -1
                    )

                    # Cross-channel spectral correlation across the DAS-detected
                    # channels (defined for multi-channel detections only).
                    detected_positions = [
                        eligible_channel_indices.index(ch) for ch in usv_detected_chs
                    ]
                    # A zero-variance channel (digital silence in the HPSS output)
                    # makes corrcoef emit NaN for its pairs; such a pair carries no
                    # evidence of a shared pattern, so a NaN aggregate is coerced to
                    # 0.0 (fails the cutoff) rather than silently skipping the check.
                    if len(detected_positions) > 1:
                        with np.errstate(invalid="ignore", divide="ignore"):
                            correlation_matrix = np.corrcoef(flattened_spectrograms[detected_positions])
                        pairwise_correlations = correlation_matrix[np.triu_indices(n=len(detected_positions), k=1)]
                        mean_signal_correlations[i] = (
                            0.0 if np.isnan(pairwise_correlations).any() else float(np.mean(pairwise_correlations))
                        )

                    # Spatial coherence: mean pairwise correlation across the
                    # top-K eligible channels ranked by in-band energy in this
                    # window, detection status ignored. A genuine call dominates
                    # the in-band soundscape at its moment, so its loudest
                    # channels share one time-frequency pattern; a localized
                    # artifact's pattern appears nowhere else on the array.
                    coherence_k = min(coherence_channel_count, len(eligible_channel_indices))
                    if coherence_k > 1:
                        channel_energies = (flattened_spectrograms ** 2).sum(axis=1)
                        top_positions = np.argsort(channel_energies)[::-1][:coherence_k]
                        with np.errstate(invalid="ignore", divide="ignore"):
                            coherence_matrix = np.corrcoef(flattened_spectrograms[top_positions])
                        pairwise_coherences = coherence_matrix[np.triu_indices(n=coherence_k, k=1)]
                        spatial_coherences[i] = (
                            0.0 if np.isnan(pairwise_coherences).any() else float(np.mean(pairwise_coherences))
                        )

                self.message_output(
                    f"Phase-4 cutoffs (absolute): detected-channel correlation >= {noise_corr_cutoff}, "
                    f"top-{coherence_channel_count} spatial coherence >= {coherence_cutoff}."
                )

                # filter noise: drop USVs failing the amplitude-channel check or the
                # correlation-AND-coherence gate. Both cutoffs are ABSOLUTE (no
                # session-relative percentiles, which overshoot on clean candidate
                # pools and undershoot on noise-dominated ones): noise must fool
                # both descriptors to survive, while a real call only needs each
                # at its loose threshold. Single-channel detections have no defined
                # correlation and are gated by coherence alone.
                drop_counter = 0
                kept_merged = []
                for i, usv in enumerate(merged):
                    condition_1 = (
                        not np.isnan(mean_signal_correlations[i])
                        and mean_signal_correlations[i] < noise_corr_cutoff
                    )
                    condition_2 = (
                        not np.isnan(spatial_coherences[i])
                        and spatial_coherences[i] < coherence_cutoff
                    )
                    if condition_0_list[i] or condition_1 or condition_2:
                        drop_counter += 1
                    else:
                        kept_merged.append(usv)
                merged = kept_merged

                self.message_output(
                    f"Number of detections dropped due to low signal correlation/coherence across channels: {drop_counter}"
                )

                if n_usv > 1:
                    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(24, 4), dpi=300)
                    # Correlations/coherences live in [-1, 1]; a fixed range keeps
                    # the figure comparable across sessions and avoids degenerate
                    # bin edges when every value is (near-)identical.
                    ax[0].hist(
                        x=mean_signal_correlations[~np.isnan(mean_signal_correlations)],
                        bins=20,
                        range=(-1.0, 1.0),
                        histtype="stepfilled",
                        color="#BBD5E8",
                        edgecolor="#202020",
                        alpha=0.5,
                    )
                    ax[0].set_xlabel("Mean signal/spectral correlation")
                    ax[0].set_ylabel("Number of putative USVs")
                    ax[0].axvline(x=noise_corr_cutoff, ls="-.", lw=1.2, c="#202020")
                    ax[1].hist(
                        x=spatial_coherences[~np.isnan(spatial_coherences)],
                        bins=20,
                        range=(-1.0, 1.0),
                        histtype="stepfilled",
                        color="#BBD5E8",
                        edgecolor="#202020",
                        alpha=0.5,
                    )
                    ax[1].set_xlabel(f"Spatial coherence (top-{coherence_channel_count} channels)")
                    ax[1].set_ylabel("Number of putative USVs")
                    ax[1].axvline(x=coherence_cutoff, ls="-.", lw=1.2, c="#202020")
                    save_figure(
                        fig,
                        stem=f"{session_id}_usv_signal_correlation_histogram",
                        viz_settings=getattr(self, "visualizations_parameter_dict", None),
                        override_dir=pathlib.Path(self.root_directory) / "audio",
                        timestamp_in_name=False,
                    )
                    plt.close()

                self.message_output(
                    f"In this session, {len(merged)} USVs were detected."
                )

                # save the summary file
                _write_usv_summary_csv(
                    merged,
                    pathlib.Path(self.root_directory) / "audio" / f"{session_id}_usv_summary.csv",
                )

            elif not filter_putative_noise_bool and n_usv >= 1:
                # Putative-noise filtering disabled: keep every merged interval
                # without amplitude/spectrogram rejection. The intervals are
                # already start-sorted (the Phase-3 watershed merge returns its
                # USVs ordered by start), so the peak/mean amplitude channels
                # stay at their 0.0 placeholders and the summary CSV is written
                # directly from the merged list.
                self.message_output(
                    f"Putative-noise filtering disabled; {len(merged)} USVs kept without amplitude/spectrogram checks."
                )

                # save the summary file
                _write_usv_summary_csv(
                    merged,
                    pathlib.Path(self.root_directory) / "audio" / f"{session_id}_usv_summary.csv",
                )

            # load metadata
            metadata, metadata_path = load_session_metadata(
                root_directory=self.root_directory,
                logger=self.message_output
            )
            if metadata is not None:
                metadata['Session']['session_usv_count'] = len(merged)
                save_session_metadata(data=metadata, filepath=metadata_path, logger=self.message_output)

        except (IndexError, FileNotFoundError) as exc:
            self.message_output(
                f"DAS summary skipped for '{self.root_directory}': {type(exc).__name__}: {exc}"
            )
            return

        # Post-summary seam check-and-correct (no-op by construction on
        # sessions inferred with the overlap-tiled model, whose annotations
        # carry no seam-ladder fingerprints).
        if self.input_parameter_dict["summarize_das_findings"]["seam_repair_bool"]:
            self.repair_seam_snapped_boundaries()

    def repair_seam_snapped_boundaries(self) -> None:
        """
        Description
        -----------
        Checks the session's ``*_usv_summary.csv`` for inter-USV gaps snapped to
        the legacy DAS window-stitching seam grid and corrects the affected call
        boundaries by re-detecting each flagged pair's audio snippet with the
        DAS model configured under ``das_command_line_inference``.

        The legacy non-overlapping DAS tiling (stride equal to the window
        length minus a 32-sample trim) judged each stitched span without
        acoustic context from its neighbours, so the faint edges of calls
        flanking a real pause were clipped exactly to the window seams: the
        summary records such a pause with a width of k stride multiples
        (k = 1..``seam_repair_max_rung``) and the flanking calls lose up to one
        stride of faint edge material. This tool

        1. flags consecutive summary pairs whose gap falls inside the
           merged-level tolerance window around a stride multiple
           (``seam_repair_width_tolerance_below_ms`` /
           ``seam_repair_width_tolerance_above_ms``) AND is corroborated by a
           microsecond-exact ladder gap (:func:`_ladder_gap_leading_stops`) in
           at least one channel's raw DAS annotations whose leading stop lies
           within ``seam_repair_raw_stop_tolerance_s`` of the merged stop;
        2. excises a ``seam_repair_snippet_margin_s``-padded snippet around
           each flagged pair from the corroborating channel's HPSS-filtered
           WAV;
        3. re-runs ``das predict`` once on the snippet folder;
        4. replaces each pair's facing edges with the re-detected ones
           (:func:`_repaired_facing_edges`) -- outward-only, capped at
           ``seam_repair_max_boundary_shift_s`` per edge -- and updates the
           affected ``start``/``stop``/``duration`` summary values.

        The summary CSV is rewritten atomically with every other column
        preserved, so the tool is safe on both freshly summarized and
        enrichment-carrying (emitter / acoustic / embedding columns) summaries.
        A sidecar ``*_seam_repair_report.json`` in the session's ``audio``
        directory records the settings used and the outcome for every flagged
        pair. Missing prerequisites (summary, raw annotations, HPSS WAVs) are
        reported through ``message_output`` and skip the check rather than
        raising, mirroring :func:`summarize_das_findings`.

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(
            f"DAS seam check-and-repair started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        repair_params = self.input_parameter_dict["summarize_das_findings"]
        stride_samples = repair_params["seam_repair_legacy_stride_samples"]
        max_rung = repair_params["seam_repair_max_rung"]
        ladder_tolerance_samples = repair_params["seam_repair_ladder_tolerance_samples"]
        width_below_s = repair_params["seam_repair_width_tolerance_below_ms"] / 1000.0
        width_above_s = repair_params["seam_repair_width_tolerance_above_ms"] / 1000.0
        raw_stop_tolerance_s = repair_params["seam_repair_raw_stop_tolerance_s"]
        snippet_margin_s = repair_params["seam_repair_snippet_margin_s"]
        max_boundary_shift_s = repair_params["seam_repair_max_boundary_shift_s"]

        session_id = pathlib.Path(self.root_directory).name
        audio_dir = pathlib.Path(self.root_directory) / "audio"
        annot_dir = audio_dir / "das_annotations"
        hpss_dir = audio_dir / "hpss_filtered"

        try:
            summary_path = first_match_or_raise(
                root=audio_dir,
                pattern="*_usv_summary.csv",
                label="USV summary CSV",
            )
            probe_wav = first_match_or_raise(
                root=hpss_dir,
                pattern="*.wav",
                label="HPSS-filtered WAV",
            )
        except FileNotFoundError as exc:
            self.message_output(
                f"DAS seam check skipped for '{self.root_directory}': {exc}"
            )
            return
        if not annot_dir.is_dir():
            self.message_output(
                f"DAS seam check skipped for '{self.root_directory}': no das_annotations directory."
            )
            return
        try:
            sampling_rate = int(sf.info(str(probe_wav)).samplerate)
        except RuntimeError as exc:
            self.message_output(
                f"DAS seam check skipped for '{self.root_directory}': unreadable WAV '{probe_wav.name}': {exc}"
            )
            return

        # usv_id is zero-padded ("0000") in freshly written summaries; without
        # the override polars re-infers it as Int64 and the atomic rewrite
        # would silently reformat the column.
        summary_df = pls.read_csv(source=str(summary_path), schema_overrides={"usv_id": pls.String})
        starts = summary_df["start"].to_numpy().astype(float)
        stops = summary_df["stop"].to_numpy().astype(float)
        durations = summary_df["duration"].to_numpy().astype(float).copy()
        if np.any(np.diff(starts) < 0):
            msg = (
                f"USV summary '{summary_path.name}' is not sorted by start time; "
                "the seam check pairs consecutive rows and requires sorted input."
            )
            raise ValueError(msg)
        n_gaps = max(len(starts) - 1, 0)

        # Phase 1: merged-level width gate. Merged gap widths can sit slightly
        # off the exact raw ladder values because the interval union takes the
        # outermost edge across channels, hence the tolerance window around each
        # stride multiple.
        stride_s = stride_samples / sampling_rate
        gaps = starts[1:] - stops[:-1] if n_gaps else np.empty(0, dtype=float)
        rung = np.round(gaps / stride_s).astype(np.int64) if n_gaps else np.empty(0, dtype=np.int64)
        width_gate = (
            (rung >= 1)
            & (rung <= max_rung)
            & (gaps >= rung * stride_s - width_below_s)
            & (gaps <= rung * stride_s + width_above_s)
        )
        width_gate_indices = np.flatnonzero(width_gate)

        # Phase 2: raw-annotation corroboration -- the microsecond-exact
        # ladder fingerprint pins the mechanism and rejects width-window
        # innocents.
        channel_ladder_stops: dict[str, np.ndarray] = {}
        for one_file in sorted(annot_dir.glob("*.csv")):
            m = _DAS_ANNOTATION_FILE_RE.match(one_file.name)
            if m is None:
                continue
            file_id = f"{m.group(1)}_{m.group(2)}"
            channel_df = pls.read_csv(source=str(one_file)).filter(pls.col("name") != "noise").sort("start_seconds")
            leading_stops = _ladder_gap_leading_stops(
                start_seconds=channel_df["start_seconds"].to_numpy().astype(float),
                stop_seconds=channel_df["stop_seconds"].to_numpy().astype(float),
                sampling_rate=sampling_rate,
                stride_samples=stride_samples,
                tolerance_samples=ladder_tolerance_samples,
                max_rung=max_rung,
            )
            if len(leading_stops):
                channel_ladder_stops[file_id] = leading_stops

        flagged: list[dict] = []
        for pair_row in width_gate_indices:
            for file_id, leading_stops in channel_ladder_stops.items():
                if np.min(np.abs(leading_stops - stops[pair_row])) <= raw_stop_tolerance_s:
                    flagged.append({"pair_row": int(pair_row), "channel": file_id})
                    break

        self.message_output(
            f"Seam check: {n_gaps} gaps, {len(width_gate_indices)} in the width gate, "
            f"{len(flagged)} corroborated by raw ladder fingerprints."
        )

        report: dict = {
            "session_id": session_id,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "sampling_rate": sampling_rate,
            "settings": {key: repair_params[key] for key in repair_params if key.startswith("seam_repair")},
            "n_gaps_checked": n_gaps,
            "n_width_gate": len(width_gate_indices),
            "n_flagged": len(flagged),
            "n_repaired": 0,
            "n_no_change": 0,
            "n_skipped": 0,
            "pairs": [],
        }
        report_path = audio_dir / f"{session_id}_seam_repair_report.json"

        if not flagged:
            with atomic_output_path(report_path) as tmp_report, tmp_report.open("w") as report_file:
                json.dump(report, report_file, indent=1)
            self.message_output("Seam check: no seam-snapped pairs found; summary left untouched.")
            return

        # Phase 3: excise snippets from the corroborating channels and re-run
        # DAS once on the folder (the model loads once for all snippets).
        snippet_dir = audio_dir / "seam_repair_snippets"
        if snippet_dir.is_dir():
            shutil.rmtree(snippet_dir)
        snippet_dir.mkdir(parents=True)

        wav_path_by_channel: dict[str, pathlib.Path] = {}
        for pair in flagged:
            file_id = pair["channel"]
            if file_id not in wav_path_by_channel:
                device, channel_token = file_id.split("_")
                wav_path_by_channel[file_id] = first_match_or_raise(
                    root=hpss_dir,
                    pattern=f"{device}_*_{channel_token}_*.wav",
                    label=f"HPSS-filtered WAV for channel {file_id}",
                )
            pair_row = pair["pair_row"]
            snippet_offset = max(0.0, starts[pair_row] - snippet_margin_s)
            snippet_stop = stops[pair_row + 1] + snippet_margin_s
            snippet_audio, _ = sf.read(
                file=str(wav_path_by_channel[file_id]),
                start=int(snippet_offset * sampling_rate),
                frames=int((snippet_stop - snippet_offset) * sampling_rate),
                dtype="int16",
            )
            sf.write(
                file=str(snippet_dir / f"pair{pair_row:05d}.wav"),
                data=snippet_audio,
                samplerate=sampling_rate,
                subtype="PCM_16",
            )
            pair["snippet_offset"] = snippet_offset

        das_params = self.input_parameter_dict["das_command_line_inference"]
        model_base = str(
            pathlib.Path(configure_path(das_params["das_model_directory"])) / das_params["model_name_base"]
        )
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        clean_env = os.environ.copy()
        clean_env.pop("PYTHONHOME", None)
        snippet_subp = subprocess.Popen(
            args=[conda_exe, "run", "--no-capture-output", "-n", das_params["das_conda_env_name"],
                  "das", "predict", str(snippet_dir), model_base,
                  "--segment-thres", str(das_params["segment_confidence_threshold"]),
                  "--segment-minlen", str(das_params["segment_minlen"]),
                  "--segment-fillgap", str(das_params["segment_fillgap"]),
                  "--save-format", "csv"],
            cwd=snippet_dir,
            env=clean_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            shell=False,
        )
        # Snippets are sub-second, so even hundreds finish in minutes on CPU;
        # a 4 h ceiling only catches a genuinely hung process.
        wait_for_subprocesses(
            subps=[snippet_subp],
            max_seconds=4 * 60 * 60,
            label=f"DAS seam-repair snippet inference ({len(flagged)} snippets)",
        )

        # Phase 4: harvest the re-detected facing edges and apply them.
        for pair in flagged:
            pair_row = pair["pair_row"]
            record: dict = {
                "pair_row": pair_row,
                "channel": pair["channel"],
                "old_stop_first": float(stops[pair_row]),
                "old_start_second": float(starts[pair_row + 1]),
            }
            snippet_annotations = snippet_dir / f"pair{pair_row:05d}_annotations.csv"
            if not snippet_annotations.is_file():
                record["outcome"] = "skipped:no_snippet_annotations"
                report["n_skipped"] += 1
                report["pairs"].append(record)
                continue
            annotation_df = pls.read_csv(source=str(snippet_annotations)).filter(pls.col("name") != "noise").sort("start_seconds")
            new_edges = _repaired_facing_edges(
                annotation_starts=annotation_df["start_seconds"].to_numpy().astype(float),
                annotation_stops=annotation_df["stop_seconds"].to_numpy().astype(float),
                snippet_offset=pair["snippet_offset"],
                stop_first=float(stops[pair_row]),
                start_second=float(starts[pair_row + 1]),
                mid_first=float(0.5 * (starts[pair_row] + stops[pair_row])),
                mid_second=float(0.5 * (starts[pair_row + 1] + stops[pair_row + 1])),
                max_boundary_shift=max_boundary_shift_s,
            )
            if new_edges is None:
                record["outcome"] = "skipped:unrepairable"
                report["n_skipped"] += 1
                report["pairs"].append(record)
                continue
            new_stop_first, new_start_second = new_edges
            record["new_stop_first"] = new_stop_first
            record["new_start_second"] = new_start_second
            if new_stop_first == stops[pair_row] and new_start_second == starts[pair_row + 1]:
                record["outcome"] = "no_change"
                report["n_no_change"] += 1
            else:
                stops[pair_row] = new_stop_first
                starts[pair_row + 1] = new_start_second
                durations[pair_row] = stops[pair_row] - starts[pair_row]
                durations[pair_row + 1] = stops[pair_row + 1] - starts[pair_row + 1]
                record["outcome"] = "repaired"
                report["n_repaired"] += 1
            report["pairs"].append(record)

        shutil.rmtree(snippet_dir)

        if report["n_repaired"] > 0:
            summary_df = summary_df.with_columns(
                pls.Series(name="start", values=starts),
                pls.Series(name="stop", values=stops),
                pls.Series(name="duration", values=durations),
            )
            with atomic_output_path(summary_path) as tmp_summary:
                summary_df.write_csv(file=str(tmp_summary))

        with atomic_output_path(report_path) as tmp_report, tmp_report.open("w") as report_file:
            json.dump(report, report_file, indent=1)

        self.message_output(
            f"Seam repair: {report['n_repaired']} pairs corrected, {report['n_no_change']} confirmed unchanged, "
            f"{report['n_skipped']} skipped; report saved to '{report_path.name}'."
        )
