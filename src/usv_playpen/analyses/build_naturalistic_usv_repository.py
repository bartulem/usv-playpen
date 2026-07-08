"""
@author: bartulem
Build a naturalistic USV playback repository: one self-contained H5 per sex holding
clean, mask-denoised vocalizations together with their natural bout-sequence structure.

Naturalistic playback should use real vocalizations played in their natural sequences
(bouts, in order, with real timing), split by emitter. This module is part (1) of that
effort -- the repository builder. For every assigned USV in a courtship session it
reconstructs clean audio (recompute the complex STFT of the raw ``hpss_filtered``
segment, apply the stored SAM mask with true phase kept, inverse-STFT), segments the
emitter's USVs into natural bouts, and writes -- per sex -- a single timestamped H5
containing the concatenated int16 audio plus the per-USV and per-bout metadata a
playback function needs to replay real sequences with real timing.

Only courtship (male-female) sessions are used, because the male/female mapping
(``track_names[0]``/``[1]``) is a positional convention that is only reliable there
(same-sex sessions would mislabel sex). Bouts are segmented by the same inter-bout
interval (IBI) rule the modeling layer uses. All gaps stored are the REAL recorded
values: within-bout inter-USV gaps and the real pause preceding each bout.

The reconstruction chain (validated) is: max-variance channel -> tight SAM mask ->
time-only mask feather -> single true-phase ISTFT (no iterations) -> call-adaptive fade
-> peak-normalize -> int16 at the recording sampling rate.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import librosa
import numpy as np
import polars as pls
from click.core import ParameterSource
from scipy.ndimage import binary_dilation, gaussian_filter

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import first_match_or_raise, resolve_experimenter_path
from ..processing.build_qlvm_training_set import build_session_masks
from ..time_utils import is_gui_context, smart_wait
from ._usv_io import extract_session_metadata
from .compute_inter_usv_interval_distributions import _read_session_lists

_INT16_INFO = np.iinfo(np.int16)
_SETTINGS_DIR = pathlib.Path(__file__).resolve().parent.parent / "_parameter_settings"

# Context labels. Each drives (1) how the emitter is handled, (2) which directory the
# database is written to, and (3) the filename context token. Mode "emitter" keeps only
# USVs attributed to the target sex's track (courtship, where the two animals differ in
# sex); "all" keeps every USV labelled by the build's sex (same-sex / lone sessions, which
# have no per-USV attribution); "mixed" keeps every USV with no sex split (sex is None).
_CONTEXT_LABELS = {
    "courtship_male":   ("courtship", "male",   "emitter"),
    "courtship_female": ("courtship", "female", "emitter"),
    "lone_male":       ("lone",     "male",   "all"),
    "lone_female":     ("lone",     "female", "all"),
    "same_sex_male":    ("same_sex",  "male",   "all"),
    "same_sex_female":  ("same_sex",  "female", "all"),
    "mixed":            ("mixed",     None,     "mixed"),
}


def _normalize_emitter(value: object) -> str:
    """
    Description
    -----------
    Normalize an emitter / track-name string for comparison by stripping the NUL
    padding some fixed-width id fields carry and any surrounding whitespace (matches
    the normalization used elsewhere when mapping ``emitter`` to sex).

    Parameters
    ----------
    value (object)
        Raw emitter / id value (``None`` maps to the empty string).

    Returns
    -------
    normalized (str)
        The NUL- and whitespace-stripped string.
    """

    if value is None:
        return ""
    return str(value).strip("\x00").strip()


def _segment_bouts(starts: np.ndarray, stops: np.ndarray, ibi_threshold: float) -> np.ndarray:
    """
    Description
    -----------
    Assign each USV (of one emitter, sorted ascending by start) to a bout. A new bout
    begins at the first USV and whenever the end-to-start gap from the previous USV is
    at least ``ibi_threshold``. Mirrors ``NeuronalTuning._detect_bouts`` but uses ``>=``
    (rather than strict ``>``) to match the modeling bout definition
    (``find_variable_length_bouts``).

    Parameters
    ----------
    starts (np.ndarray, shape (n_usvs,))
        USV start times in seconds, sorted ascending.
    stops (np.ndarray, shape (n_usvs,))
        USV stop times in seconds, paired with ``starts``.
    ibi_threshold (float)
        Inter-USV silence (s) at or above which a new bout begins.

    Returns
    -------
    bout_idx (np.ndarray of int, shape (n_usvs,))
        Bout index in ``[0, n_bouts)`` assigned to each USV.
    """

    if starts.size == 0:
        return np.empty(0, dtype=int)
    is_bout_start = np.concatenate([[True], (starts[1:] - stops[:-1]) >= ibi_threshold])
    return np.cumsum(is_bout_start.astype(int)) - 1


def reconstruct_usv_waveform(
    audio_segment_channels: np.ndarray,
    mask_2d: np.ndarray,
    sampling_rate: int,
    spec_params: dict,
    mask_dilation: int,
    feather_sigma_time: float,
    fade_ms: float,
) -> np.ndarray | None:
    """
    Description
    -----------
    Reconstruct a single clean, denoised USV waveform from a raw multichannel audio
    segment and its stored SAM mask, via true-phase masked inverse STFT.

    The forward pipeline (``generate_spectrograms.compute_usv_spectrogram``) is
    mirrored so the stored mask registers onto the recomputed STFT: the same
    ``n_fft`` / ``hop_length`` / ``window`` produce an STFT whose frame count equals
    the stored ``duration``, and the stored (``num_freq_bins`` x ``num_time_bins``)
    mask is up-mapped onto the full complex-STFT grid. In frequency this inverts the
    forward band-limit + ``np.interp`` resize with a nearest-neighbour map (each
    in-band STFT bin takes the mask row it was interpolated from); in time it is a
    1:1 column correspondence (the forward path pads/crops with ``fix_length``, it
    does not resample time). The mask is applied to the complex STFT (true phase
    retained) and inverted with a single ``librosa.istft``.

    Two cosmetic steps clean the boundaries without distorting the call: the mask is
    feathered in TIME ONLY (Gaussian ``feather_sigma_time``) so the hard time-edge
    does not invert to a broadband onset/offset click, while the frequency edges stay
    tight (a frequency feather would bleed a halo below/above the call); and a short
    call-adaptive raised-cosine fade tapers the waveform ends.

    Parameters
    ----------
    audio_segment_channels (np.ndarray)
        ``(n_samples, n_channels)`` raw audio slice for the USV (the same
        ``hpss_filtered`` mmap the spectrograms/masks were built from).
    mask_2d (np.ndarray)
        ``(num_freq_bins, num_time_bins)`` boolean (or 0/1) SAM region mask for this
        USV, low-frequency-first, as returned by ``build_session_masks``.
    sampling_rate (int)
        Audio sampling rate in Hz (e.g. 250000).
    spec_params (dict)
        The ``generate_spectrograms`` settings block; supplies ``nperseg``,
        ``hop_length``, ``window``, ``min_freq``, ``max_freq``, ``num_freq_bins``,
        ``num_time_bins`` (read directly, never re-derived, so params match the
        forward pipeline exactly).
    mask_dilation (int)
        Morphological dilation (in mask bins) applied to the mask before inversion;
        ``0`` keeps the tight SAM mask (dilation pulls in neighbouring harmonics /
        background and is generally worse).
    feather_sigma_time (float)
        Gaussian sigma (in time bins) of the time-only mask feather.
    fade_ms (float)
        Target raised-cosine onset/offset fade length in milliseconds; the actual
        fade is capped at one third of the call so short calls are not over-eaten.

    Returns
    -------
    waveform (np.ndarray or None)
        The reconstructed float64 waveform (in the input's amplitude scale), or
        ``None`` if the segment is shorter than one STFT window.
    """

    nperseg = int(spec_params['nperseg'])
    hop_length = int(spec_params['hop_length'])
    window = spec_params['window']
    min_freq = spec_params['min_freq']
    max_freq = spec_params['max_freq']
    num_freq_bins = int(spec_params['num_freq_bins'])
    num_time_bins = int(spec_params['num_time_bins'])

    if audio_segment_channels.shape[0] < nperseg:
        return None

    # Max-variance channel: complex STFTs cannot be phase-averaged across channels,
    # so the loudest single channel is used (mirrors the forward variance weighting).
    channel_variances = np.var(audio_segment_channels.astype(np.float64), axis=0)
    best_channel = int(np.argmax(channel_variances))
    signal = audio_segment_channels[:, best_channel].astype(np.float64)
    signal = signal - np.mean(signal)

    complex_stft = librosa.stft(
        signal,
        n_fft=nperseg,
        hop_length=hop_length,
        win_length=nperseg,
        window=window,
        center=True,
    )

    # Up-map the stored mask onto the full (n_fft/2 + 1, T_native) STFT grid.
    freqs = librosa.fft_frequencies(sr=sampling_rate, n_fft=nperseg)
    band = (freqs >= min_freq) & (freqs <= max_freq)
    band_idx = np.where(band)[0]
    # Nearest-neighbour inverse of the forward np.interp(freq_orig -> 128 bins): each
    # in-band STFT bin takes the 128-grid row it would have been interpolated from.
    row_of_bin = np.rint(np.linspace(0, 1, band_idx.size) * (num_freq_bins - 1)).astype(int)

    mask = np.asarray(mask_2d, dtype=bool)
    if mask_dilation > 0:
        mask = binary_dilation(mask, iterations=mask_dilation)
    mask_band = mask[row_of_bin, :]
    time_overlap = min(complex_stft.shape[1], num_time_bins)
    mask_full = np.zeros(complex_stft.shape, dtype=np.float64)
    mask_full[band_idx[:, None], np.arange(time_overlap)[None, :]] = mask_band[:, :time_overlap]

    # Time-only feather: smooth the mask's onset/offset time-edges (removes the
    # broadband click) but keep the frequency edges hard (no below/above halo).
    soft_mask = gaussian_filter(mask_full, sigma=(0.0, feather_sigma_time))

    waveform = librosa.istft(
        complex_stft * soft_mask,
        length=signal.shape[0],
        win_length=nperseg,
        hop_length=hop_length,
        window=window,
        center=True,
    )

    n_fade = min(int(fade_ms * 1e-3 * sampling_rate), waveform.shape[0] // 3)
    if n_fade > 0 and waveform.shape[0] > 2 * n_fade:
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, n_fade)))
        waveform[:n_fade] *= ramp
        waveform[-n_fade:] *= ramp[::-1]
    return waveform


class NaturalisticUsvRepositoryBuilder:
    """
    Description
    -----------
    Builds one naturalistic USV playback repository H5 per sex from a list of session
    root directories: reconstructs clean audio for each assigned courtship USV,
    segments it into natural bouts, and stores the concatenated audio + per-USV +
    per-bout structure needed to replay real sequences with real timing.
    """

    def __init__(
        self,
        root_directories: list[str] | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the NaturalisticUsvRepositoryBuilder.

        Parameters
        ----------
        root_directories (list[str])
            Session root directories to process. Optional: when empty (the CLI path),
            the roots are read from the ``session_lists`` text files named in the
            ``build_naturalistic_usv_repository`` settings block (one session root per
            line, via :func:`_read_session_lists`). Each session's
            ``audio/hpss_filtered/*.mmap``, ``audio/spectrograms/*_spectrograms.h5``
            and ``audio/*_usv_summary.csv`` are located within it. Sessions lacking the
            raw ``hpss_filtered`` mmap or (when gated) not decoding to a courtship
            experiment are skipped.
        input_parameter_dict (dict)
            Analyses settings; the ``build_naturalistic_usv_repository`` block supplies
            the output dirs, courtship gate, bout, filter and reconstruction params.
            The ``generate_spectrograms`` (processing) and ``mixture_model_params``
            (modeling) blocks are read directly from their canonical settings files.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.root_directories = root_directories if root_directories is not None else []
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def build(self) -> None:
        """
        Description
        -----------
        Runs a single (sex, context) repository build. The ``context_label`` setting (e.g.
        ``courtship_male``, ``same_sex_female``, ``lone_male``, ``mixed``) selects which
        USVs to keep (a courtship build keeps only the target sex's attributed emitter, while
        same-sex / lone / mixed builds keep every USV without attribution), the output
        directory (the target sex's, or the mixed dir), and the filename context token. For
        each session root it segments the selected USVs into natural bouts, reconstructs
        every USV of each complete bout, and accumulates audio + per-USV/per-bout metadata;
        after all sessions it writes one timestamped H5 that also records the input session
        lists as provenance. A session that cannot be read is skipped and logged so a large
        batch always completes.

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(
            f"Naturalistic USV repository build started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['build_naturalistic_usv_repository']
        with (_SETTINGS_DIR / "processing_settings.json").open(encoding="utf-8") as processing_file:
            spec_params = json.load(processing_file)['generate_spectrograms']
        with (_SETTINGS_DIR / "modeling_settings.json").open(encoding="utf-8") as modeling_file:
            mixture_params = json.load(modeling_file)['mixture_model_params']

        context_label = cfg['context_label']
        if context_label not in _CONTEXT_LABELS:
            msg = f"context_label must be one of {sorted(_CONTEXT_LABELS)}, got {context_label!r}."
            raise ValueError(msg)
        context_token, target_sex, emitter_mode = _CONTEXT_LABELS[context_label]
        ibi_z_score = cfg['ibi_z_score']
        ibi_component_index = cfg['ibi_component_index']
        min_vocalizations = cfg['min_vocalizations']
        length_threshold = cfg['length_threshold']
        min_duration = cfg['min_duration']
        mask_dilation = cfg['mask_dilation']
        feather_sigma_time = cfg['feather_sigma_time']
        fade_ms = cfg['fade_ms']
        peak_normalize = cfg['peak_normalize']
        peak_target_fraction = cfg['peak_target_fraction']
        session_lists = cfg['session_lists']
        root_directories = self.root_directories or _read_session_lists(session_lists, self.message_output)
        # One repository root (a shared data root); the database is written into its
        # male / female / mixed subdirectory according to the build's sex.
        repository_root = pathlib.Path(resolve_experimenter_path(
            self.input_parameter_dict['data_roots']['naturalistic_usv_repository_dir']))
        target_output_dir = repository_root / (target_sex if target_sex is not None else "mixed")
        # Bout-gap threshold from the target sex's first mixture component; a mixed build
        # (no target sex) falls back to the male component.
        ibi_sex = target_sex if target_sex is not None else "male"
        ibi_threshold = float(np.exp(mixture_params[ibi_sex]['means'][ibi_component_index]
                                     + ibi_z_score * mixture_params[ibi_sex]['sds'][ibi_component_index]))

        # Single accumulator: audio chunks + parallel per-USV / per-bout metadata.
        store = {
            "audio": [], "offset": [], "length": [],
            "session": [], "emitter": [], "usv_row": [],
            "bout_index": [], "position_in_bout": [], "gap_to_next_s": [],
            "feature_frames": [],
            "bout_session": [], "bout_emitter": [], "bout_usv_start": [],
            "bout_usv_count": [], "bout_preceding_isi_s": [],
            "sample_cursor": 0, "usv_cursor": 0, "bout_cursor": 0,
            "sampling_rate_hz": None,
        }

        for root_directory in root_directories:
            root = pathlib.Path(root_directory)
            try:
                audio_file_loc = first_match_or_raise(
                    root=root / "audio" / "hpss_filtered",
                    pattern="*.mmap",
                    label="concatenated audio mmap",
                )

                usv_summary_loc = first_match_or_raise(
                    root=root / "audio",
                    pattern="*_usv_summary.csv",
                    recursive=True,
                    label="USV summary CSV",
                )
                h5_loc = first_match_or_raise(
                    root=root / "audio" / "spectrograms",
                    pattern="*_spectrograms.h5",
                    label="per-session spectrogram H5",
                )

                audio_file_name = audio_file_loc.name
                data_type = audio_file_name.split("_")[-1][:-5]
                channel_num = int(audio_file_name.split("_")[-2])
                sample_num = int(audio_file_name.split("_")[-3])
                sampling_rate = int(audio_file_name.split("_")[-4])
                audio_file_data = np.memmap(
                    filename=audio_file_loc, mode="r", dtype=data_type, shape=(sample_num, channel_num)
                )

                usv_summary_df = pls.read_csv(source=str(usv_summary_loc))
                starts_all = usv_summary_df["start"].to_numpy()
                stops_all = usv_summary_df["stop"].to_numpy()

                with h5py.File(str(h5_loc), "r") as h5_file:
                    session_id = next(iter(h5_file["spectrogram"].keys()))
                    durations = h5_file[f"spectrogram/{session_id}"]["durations"][:]

                    # Select this build's USV rows: a courtship build keeps only the target
                    # sex's attributed emitter; same-sex / lone / mixed builds keep every
                    # USV (no attribution) labelled by the build's sex.
                    if emitter_mode == "emitter":
                        # Courtship: the two animals differ in sex; read the target sex's
                        # track id and keep only USVs attributed to it.
                        metadata = extract_session_metadata(str(root))
                        stored_emitter = _normalize_emitter(metadata[f'{target_sex}_id'])
                        emitters_all = [_normalize_emitter(e) for e in usv_summary_df["emitter"].to_list()]
                        emitter_rows = np.array(
                            [r for r in range(len(emitters_all)) if emitters_all[r] == stored_emitter],
                            dtype=np.int64,
                        )
                    else:
                        # Same-sex / lone / mixed: no attribution; keep every USV.
                        stored_emitter = target_sex if target_sex is not None else "mixed"
                        emitter_rows = np.arange(len(starts_all), dtype=np.int64)
                    if emitter_rows.size == 0:
                        continue

                    order = np.argsort(starts_all[emitter_rows], kind="stable")
                    rows_sorted = emitter_rows[order]
                    starts_sorted = starts_all[rows_sorted]
                    stops_sorted = stops_all[rows_sorted]
                    bout_idx = _segment_bouts(starts_sorted, stops_sorted, ibi_threshold)

                    # Candidate bouts: >= min_vocalizations and every USV in-duration-range.
                    candidate_bout_positions: list[np.ndarray] = []
                    for b in range(int(bout_idx.max()) + 1 if bout_idx.size else 0):
                        positions = np.where(bout_idx == b)[0]
                        if positions.size < min_vocalizations:
                            continue
                        bout_rows = rows_sorted[positions]
                        if np.any(durations[bout_rows] < min_duration) or np.any(durations[bout_rows] > length_threshold):
                            continue
                        candidate_bout_positions.append(positions)

                    if not candidate_bout_positions:
                        continue

                    # One masks pass for all USVs in the candidate bouts.
                    selected_rows = np.unique(np.concatenate([rows_sorted[p] for p in candidate_bout_positions]))
                    n_freq = int(spec_params['num_freq_bins'])
                    n_time = int(spec_params['num_time_bins'])
                    masks, masks_len = build_session_masks(h5_file, session_id, selected_rows, n_freq, n_time)
                    row_to_pos = {int(r): i for i, r in enumerate(selected_rows)}

                    usv_cursor_before = store["usv_cursor"]
                    self._reconstruct_and_store(
                        store, target_sex if target_sex is not None else "mixed", session_id,
                        stored_emitter, candidate_bout_positions,
                        rows_sorted, starts_sorted, stops_sorted, masks, masks_len,
                        row_to_pos, audio_file_data, sample_num, sampling_rate, spec_params,
                        mask_dilation, feather_sigma_time, fade_ms, peak_normalize,
                        peak_target_fraction, spec_params['offset'],
                    )
                    # Carry the FULL usv_summary row for every USV just stored (in the same
                    # order), so the repository holds every per-USV feature column.
                    new_rows = store["usv_row"][usv_cursor_before:]
                    if new_rows:
                        store["feature_frames"].append(usv_summary_df[new_rows])
            except Exception as session_error:  # batch robustness: skip + log a bad session
                self.message_output(f"Skipping {root.name}: {session_error}")
                continue

        timestamp = datetime.today().strftime('%Y%m%d_%H%M%S')
        self._write_repository(
            store, target_sex, context_token, context_label, target_output_dir,
            timestamp, ibi_z_score, ibi_component_index, session_lists, root_directories,
        )
        self.message_output("Naturalistic USV repository build finished.")

    def _reconstruct_and_store(
        self, acc, sex, session_id, emitter_norm, candidate_bout_positions, rows_sorted,
        starts_sorted, stops_sorted, masks, masks_len, row_to_pos,
        audio_file_data, sample_num, sampling_rate, spec_params, mask_dilation,
        feather_sigma_time, fade_ms, peak_normalize, peak_target_fraction, offset,
    ) -> None:
        """
        Description
        -----------
        Reconstruct each USV of every complete candidate bout for one (session, sex)
        and append its audio + per-USV + per-bout metadata to the accumulator. A bout
        is kept only if every USV in it has a detected mask and reconstructs
        successfully (complete-bout rule); incomplete bouts are dropped and logged.
        Records the real within-bout gap to the next USV and the real preceding
        inter-bout gap.

        Parameters
        ----------
        acc (dict)
            The per-sex accumulator being filled.
        sex, session_id, emitter_norm (str)
            Sex label, session id and normalized emitter id.
        candidate_bout_positions (list[np.ndarray])
            Sorted-position index arrays, one per candidate bout.
        rows_sorted, starts_sorted, stops_sorted (np.ndarray)
            The emitter's usv_summary rows / starts / stops, sorted ascending by start.
        masks, masks_len (np.ndarray)
            Per-selected-row SAM masks and instance counts from ``build_session_masks``.
        row_to_pos (dict)
            Maps a usv_summary row to its index into ``masks`` / ``masks_len``.
        audio_file_data (np.memmap)
            The session's raw multichannel audio.
        sample_num, sampling_rate (int)
            Total sample count and sampling rate (Hz) of the mmap.
        spec_params (dict), mask_dilation (int), feather_sigma_time (float),
        fade_ms (float), peak_normalize (bool), peak_target_fraction (float),
        offset (float)
            Reconstruction parameters (``peak_target_fraction`` scales each
            peak-normalized snippet toward the int16 ceiling; see
            :func:`reconstruct_usv_waveform`).

        Returns
        -------
        None
        """

        acc["sampling_rate_hz"] = sampling_rate
        prev_bout_last_stop = None
        dropped = 0
        for positions in candidate_bout_positions:
            bout_rows = rows_sorted[positions]
            bout_starts = starts_sorted[positions]
            bout_stops = stops_sorted[positions]

            bout_waveforms = []
            for local_i, row in enumerate(bout_rows):
                pos = row_to_pos[int(row)]
                if masks_len[pos] == 0:
                    bout_waveforms = None
                    break
                s0 = max(0, int(np.floor((float(bout_starts[local_i]) - offset) * sampling_rate)))
                s1 = min(sample_num, int(np.ceil((float(bout_stops[local_i]) + offset) * sampling_rate)))
                if s1 <= s0:
                    bout_waveforms = None
                    break
                segment = np.asarray(audio_file_data[s0:s1, :])
                waveform = reconstruct_usv_waveform(
                    audio_segment_channels=segment, mask_2d=masks[pos], sampling_rate=sampling_rate,
                    spec_params=spec_params, mask_dilation=mask_dilation,
                    feather_sigma_time=feather_sigma_time, fade_ms=fade_ms,
                )
                if waveform is None:
                    bout_waveforms = None
                    break
                if peak_normalize:
                    peak = float(np.max(np.abs(waveform)))
                    if peak <= 0.0:
                        bout_waveforms = None
                        break
                    waveform = waveform / peak * (peak_target_fraction * _INT16_INFO.max)
                bout_waveforms.append(np.clip(np.round(waveform), _INT16_INFO.min, _INT16_INFO.max).astype(np.int16))

            if bout_waveforms is None:
                dropped += 1
                continue

            # Complete bout kept: record per-bout row + each USV.
            preceding_isi = (float('nan') if prev_bout_last_stop is None
                             else float(bout_starts[0] - prev_bout_last_stop))
            acc["bout_session"].append(session_id)
            acc["bout_emitter"].append(emitter_norm)
            acc["bout_usv_start"].append(acc["usv_cursor"])
            acc["bout_usv_count"].append(len(bout_rows))
            acc["bout_preceding_isi_s"].append(preceding_isi)
            bout_index = acc["bout_cursor"]

            for local_i, row in enumerate(bout_rows):
                waveform = bout_waveforms[local_i]
                acc["audio"].append(waveform)
                acc["offset"].append(acc["sample_cursor"])
                acc["length"].append(int(waveform.shape[0]))
                acc["session"].append(session_id)
                acc["emitter"].append(emitter_norm)
                acc["usv_row"].append(int(row))
                acc["bout_index"].append(bout_index)
                acc["position_in_bout"].append(local_i)
                gap_next = (float(bout_starts[local_i + 1] - bout_stops[local_i])
                            if local_i < len(bout_rows) - 1 else float('nan'))
                acc["gap_to_next_s"].append(gap_next)
                acc["sample_cursor"] += int(waveform.shape[0])
                acc["usv_cursor"] += 1

            acc["bout_cursor"] += 1
            prev_bout_last_stop = float(bout_stops[-1])

        if dropped:
            self.message_output(f"{session_id} [{sex}]: dropped {dropped} incomplete bout(s).")

    def _write_repository(self, acc, sex, context_token, context_label, output_dir, timestamp,
                          ibi_z_score, ibi_component_index, session_lists, root_directories) -> None:
        """
        Description
        -----------
        Write this build's accumulated audio + per-USV + per-bout metadata to a single
        timestamped H5 in ``output_dir``. Audio is one concatenated int16 array indexed
        by per-USV ``offset``/``length``; string metadata uses a variable-length UTF-8
        dtype. The filename carries the context token and the H5 records its social context
        plus the input session lists (provenance). Nothing is written (with a warning) if no
        bouts were accumulated.

        Parameters
        ----------
        acc (dict)
            The accumulator for this (sex, context) build.
        sex (str | None)
            Sex label ('male' / 'female'), or ``None`` for a sex-agnostic ``mixed`` build.
        context_token (str)
            Context filename token ('courtship' / 'same_sex' / 'lone' / 'mixed').
        context_label (str)
            Full context label (e.g. ``courtship_male``), recorded as a root attribute.
        output_dir (pathlib.Path)
            Resolved output directory for this build.
        timestamp (str)
            Build timestamp (``%Y%m%d_%H%M%S``) used in the filename.
        ibi_z_score (float), ibi_component_index (int)
            Bout-segmentation parameters, recorded as root attributes.
        session_lists (list), root_directories (list)
            The input session-list files and resolved session roots, stored as provenance.

        Returns
        -------
        None
        """

        target_label = sex if sex is not None else "mixed"
        if not acc["audio"]:
            self.message_output(f"No {target_label} bouts accumulated; no repository written.")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"naturalistic_usv_repository_{context_token}_{timestamp}.h5"
        str_dtype = h5py.string_dtype(encoding="utf-8")
        audio_concat = np.concatenate(acc["audio"])
        n_usv = len(acc["usv_row"])
        n_bout = len(acc["bout_usv_start"])

        with h5py.File(str(out_path), "w") as h5_file:
            h5_file.attrs["sex"] = target_label
            h5_file.attrs["social_context"] = context_token
            h5_file.attrs["context_label"] = context_label
            h5_file.attrs["sampling_rate_hz"] = int(acc["sampling_rate_hz"])
            h5_file.attrs["build_timestamp"] = timestamp
            h5_file.attrs["ibi_z_score"] = float(ibi_z_score)
            h5_file.attrs["ibi_component_index"] = int(ibi_component_index)
            h5_file.attrs["n_usv"] = n_usv
            h5_file.attrs["n_bout"] = n_bout
            h5_file.attrs["n_sessions"] = len(set(acc["bout_session"]))

            h5_file.create_dataset("audio", data=audio_concat, dtype=np.int16, compression="gzip", compression_opts=4)
            usv_group = h5_file.create_group("usv")
            usv_group.create_dataset("offset", data=np.asarray(acc["offset"], dtype=np.int64))
            usv_group.create_dataset("length", data=np.asarray(acc["length"], dtype=np.int64))
            usv_group.create_dataset("session", data=np.asarray(acc["session"], dtype=object), dtype=str_dtype)
            usv_group.create_dataset("emitter", data=np.asarray(acc["emitter"], dtype=object), dtype=str_dtype)
            usv_group.create_dataset("usv_row", data=np.asarray(acc["usv_row"], dtype=np.int64))
            usv_group.create_dataset("bout_index", data=np.asarray(acc["bout_index"], dtype=np.int64))
            usv_group.create_dataset("position_in_bout", data=np.asarray(acc["position_in_bout"], dtype=np.int64))
            usv_group.create_dataset("gap_to_next_s", data=np.asarray(acc["gap_to_next_s"], dtype=np.float64))
            # Full per-USV usv_summary feature table (every column: mask_number,
            # spectral_entropy, acoustic features, category, qlvm, ...), aligned to the
            # stored USV order, one dataset per column.
            feature_group = usv_group.create_group("features")
            if acc["feature_frames"]:
                # `vertical_relaxed` coerces a column to a common supertype when sessions
                # disagree on its dtype (e.g. Int64 in one usv_summary, Float64 in another).
                feature_df = pls.concat(acc["feature_frames"], how="vertical_relaxed")
                for col in feature_df.columns:
                    values = feature_df[col].to_numpy()
                    if values.dtype.kind in ("O", "U", "S"):
                        encoded = np.asarray(["" if v is None else str(v) for v in values], dtype=object)
                        feature_group.create_dataset(col, data=encoded, dtype=str_dtype)
                    else:
                        feature_group.create_dataset(col, data=values)
            bout_group = h5_file.create_group("bout")
            bout_group.create_dataset("session", data=np.asarray(acc["bout_session"], dtype=object), dtype=str_dtype)
            bout_group.create_dataset("emitter", data=np.asarray(acc["bout_emitter"], dtype=object), dtype=str_dtype)
            bout_group.create_dataset("usv_start", data=np.asarray(acc["bout_usv_start"], dtype=np.int64))
            bout_group.create_dataset("usv_count", data=np.asarray(acc["bout_usv_count"], dtype=np.int64))
            bout_group.create_dataset("preceding_isi_s", data=np.asarray(acc["bout_preceding_isi_s"], dtype=np.float64))

            # Provenance: exactly which session-list files + resolved roots built this file.
            provenance_group = h5_file.create_group("provenance")
            provenance_group.create_dataset("session_lists", data=np.asarray([str(p) for p in session_lists], dtype=object), dtype=str_dtype)
            provenance_group.create_dataset("session_roots", data=np.asarray([str(p) for p in root_directories], dtype=object), dtype=str_dtype)

        self.message_output(
            f"Wrote {target_label} repository: {n_usv} USVs in {n_bout} bouts -> {out_path!s}."
        )


@click.command(name="build-naturalistic-usv-repository")
@click.option('--session-list', 'session_lists', type=click.Path(exists=True, file_okay=True, dir_okay=False), multiple=True, required=False, help='Path to a text file listing session root directories (one per line). Repeatable; overrides the settings default.')
@click.option('--context-label', 'context_label', type=click.Choice(['courtship_male', 'courtship_female', 'lone_male', 'lone_female', 'same_sex_male', 'same_sex_female', 'mixed']), default=None, required=False, help='Which (sex, social context) database to build; drives emitter handling, output directory, and the filename token.')
@click.option('--ibi-z-score', 'ibi_z_score', type=float, default=None, required=False, help='z-score for the inter-bout-interval threshold exp(mu + z*sd).')
@click.option('--ibi-component-index', 'ibi_component_index', type=int, default=None, required=False, help='Mixture component index (per-sex) used for the IBI threshold.')
@click.option('--min-vocalizations', 'min_vocalizations', type=int, default=None, required=False, help='Minimum USVs for a bout to be kept.')
@click.option('--length-threshold', 'length_threshold', type=int, default=None, required=False, help='Drop the whole bout if any of its USVs has duration > this (time bins); the stored 128-column mask truncates longer calls.')
@click.option('--min-duration', 'min_duration', type=int, default=None, required=False, help='Drop the whole bout if any of its USVs has duration < this (time bins).')
@click.option('--mask-dilation', 'mask_dilation', type=int, default=None, required=False, help='Grow the SAM mask by this many bins before inversion (0 = tight).')
@click.option('--feather-sigma-time', 'feather_sigma_time', type=float, default=None, required=False, help='Gaussian sigma (time bins) of the time-only mask feather.')
@click.option('--fade-ms', 'fade_ms', type=float, default=None, required=False, help='Raised-cosine onset/offset fade length in milliseconds (call-adaptive).')
@click.option('--peak-normalize/--no-peak-normalize', 'peak_normalize', default=None, required=False, help='Peak-normalize each USV to a uniform level, or preserve relative amplitude.')
@click.option('--peak-target-fraction', 'peak_target_fraction', type=float, default=None, required=False, help='Fraction of the int16 ceiling each peak-normalized snippet is scaled to.')
@click.pass_context
def build_naturalistic_usv_repository_cli(ctx, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to build one (sex, social context) naturalistic USV playback
    repository H5 (clean reconstructed audio + natural bout-sequence structure) from a
    list of session root directories, selected by ``--context-label``.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    analyses_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='analyses_settings',
        parameters_lists=['session_lists'],
        block='build_naturalistic_usv_repository',
    )

    NaturalisticUsvRepositoryBuilder(
        input_parameter_dict=analyses_settings_dict,
        message_output=print,
    ).build()
