"""
@author: bartulem
Generates playback WAV files and frequency shifts audio segment.
"""
from __future__ import annotations

import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import h5py
import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

from ..os_utils import find_base_path, find_cluster_path, newest_match_or_raise, resolve_data_root
from ..time_utils import is_gui_context, smart_wait

# Maps a playback `context_label` to (context filename token, sex subdirectory / output-name
# prefix). Mirrors the build-side _CONTEXT_LABELS in build_naturalistic_usv_repository; the
# generator opens the newest ``naturalistic_usv_repository_<token>_*.h5`` under
# ``<naturalistic_usv_repository_dir>/<sex>/``.
_PLAYBACK_CONTEXTS = {
    "courtship_male":   ("courtship", "male"),
    "courtship_female": ("courtship", "female"),
    "lone_male":        ("lone",      "male"),
    "lone_female":      ("lone",      "female"),
    "same_sex_male":    ("same_sex",  "male"),
    "same_sex_female":  ("same_sex",  "female"),
    "mixed":            ("mixed",     "mixed"),
}


def _read_int16_snippet(snippet_path: Path, expected_rate_hz: int | None = None) -> np.ndarray:
    """
    Description
    -----------
    Read a playback-snippet WAV and return its sample array, enforcing that it is
    ``int16``. The playback builders concatenate snippets with explicitly ``int16``
    silence and write an ``int16`` WAV; a snippet of any other dtype (``int32`` /
    ``float32`` / ``uint8``) would make ``np.concatenate`` silently upcast the whole
    buffer, changing the output bit depth, and a ``float32`` snippet in ``[-1, 1]``
    mixed with ``int16`` zeros would have nonsensical relative amplitude.

    When ``expected_rate_hz`` is provided, the snippet's native sample rate must also
    match it. The builder writes the output WAV at a fixed rate and does NOT resample
    the concatenated snippets, so a snippet recorded at a different rate would be
    played back at the wrong speed/pitch -- validating the rate here fails loudly
    instead of silently corrupting playback timing.

    Parameters
    ----------
    snippet_path (Path)
        Path to the snippet ``.wav`` file.
    expected_rate_hz (int | None)
        If not ``None``, the snippet's native sample rate (Hz) must equal this value;
        a mismatch raises ``ValueError``. When ``None``, the rate is not checked.

    Returns
    -------
    snippet_data (np.ndarray)
        The snippet samples, guaranteed ``int16``.

    Raises
    ------
    ValueError
        If the snippet WAV is not ``int16``, or its sample rate differs from
        ``expected_rate_hz`` (when that is provided).
    """

    snippet_rate, snippet_data = wavfile.read(snippet_path)
    if snippet_data.dtype != np.int16:
        msg = (
            f"playback snippet {Path(snippet_path).name!r} has dtype "
            f"{snippet_data.dtype} but must be int16 -- the playback silence, seed, "
            f"and output WAV are all int16, so a non-int16 snippet would upcast the "
            f"output bit depth and corrupt relative amplitudes. Re-export the "
            f"snippets as 16-bit PCM."
        )
        raise ValueError(msg)
    if expected_rate_hz is not None and int(snippet_rate) != int(expected_rate_hz):
        msg = (
            f"playback snippet {Path(snippet_path).name!r} has a native sample rate of "
            f"{int(snippet_rate)} Hz but the playback output is written at "
            f"{int(expected_rate_hz)} Hz, and the builder does not resample -- the "
            f"snippet would play back at the wrong speed/pitch. Re-export the snippets "
            f"at {int(expected_rate_hz)} Hz."
        )
        raise ValueError(msg)
    return snippet_data


class AudioGenerator:

    if os.name == 'nt':
        command_addition = 'cmd /c '
        shell_usage_bool = False
    else:
        command_addition = ''
        shell_usage_bool = True

    def __init__(self, **kwargs):
        """
        Description
        -----------
        Initializes the AudioGenerator class.

        Parameters
        ----------
        exp_id (str)
            Base file server directory.
        root_directory (str)
            Root directory for data; defaults to None.
        create_playback_settings_dict (dict)
            Settings for creating USV playback files; defaults to None.
        freq_shift_settings_dict (dict)
            Frequency shift settings; defaults to None.
        message_output (function)
            Defines output messages; defaults to None.

        Returns
        -------
        None
        """

        expected_kwargs = {'exp_id', 'root_directory', 'create_playback_settings_dict',
                           'freq_shift_settings_dict', 'message_output'}
        unexpected_kwargs = set(kwargs) - expected_kwargs
        if unexpected_kwargs:
            raise TypeError(f"{type(self).__name__}() got unexpected keyword argument(s) "
                            f"{', '.join(map(repr, sorted(unexpected_kwargs)))}; expected only "
                            f"{', '.join(map(repr, sorted(expected_kwargs)))}.")
        for kw_arg, kw_val in kwargs.items():
            self.__dict__[kw_arg] = kw_val

        self.app_context_bool = is_gui_context()

    def create_naturalistic_usv_playback_wav(self) -> None:
        """
        Description
        -----------
        Constructs naturalistic USV playback WAV file(s) by replaying REAL vocalization
        bouts drawn from a per-sex naturalistic USV repository H5 (built by
        ``build_naturalistic_usv_repository``). Instead of drawing single USVs at random
        and sampling inter-event gaps from a mixture model, this replays whole recorded
        bouts in their natural emission order (``1-2-3-4-5-6``) with their real timing.

        The ``context_label`` (a (sex, social context) pair, e.g. ``courtship_female``) picks
        the newest repository H5 under ``<naturalistic_usv_repository_dir>/<sex>/`` matching
        that context; it supplies: a concatenated int16 ``audio`` array indexed per USV by
        ``usv/offset`` + ``usv/length``; the per-bout USV runs (``bout/usv_start`` +
        ``bout/usv_count``, stored in emission order); the real within-bout inter-USV
        gaps (``usv/gap_to_next_s``); and the real preceding inter-bout pause
        (``bout/preceding_isi_s``). The output sampling rate is the repository's
        ``sampling_rate_hz`` root attribute.

        The way the code works is as follows:
        (1) it writes a fixed ``edge_silence_seconds`` lead-in silence
        (2) it draws a random intact bout and appends its USVs in order, separated by
            their real within-bout inter-USV intervals (IUI) of silence
        (3) between bouts (not before the first) it inserts the drawn bout's real preceding
            inter-sequence interval (ISI) of silence -- a session-first bout (``NaN``) uses
            a sampled real ISI -- clipped to ``max_isi_seconds`` so no single pause is huge
        (4) it keeps adding whole bouts, skipping any that (with the lead-out) would overrun
            the target, until the remaining time is too small; it then writes a fixed
            ``edge_silence_seconds`` lead-out silence. The file is therefore built of whole
            bouts and is *up to* ``total_acceptable_naturalistic_playback_time``, opening and
            closing on the fixed edge silence

        Two side-car files are written alongside each ``.wav`` (both 1:1 with the audio
        chunks, clamped to the truncated WAV): ``_spacing.txt`` (per-chunk sample counts)
        and ``_usvids.txt`` (per-chunk labels -- ``<session>_usv<row>`` for a USV, or
        ``ISI`` / ``IUI`` for a silence gap).

        ``playback_seed`` (``None`` by default -> fresh entropy) seeds the numpy generator
        driving BOTH the bout draws and the fallback-ISI draws, so an integer seed yields
        a documented, repeatable stimulus set.

        Parameters
        ----------
        None
            Inputs are read from ``self.create_playback_settings_dict`` (the
            ``create_naturalistic_usv_playback_wav`` settings block) and ``self.exp_id``.

        Returns
        -------
        usv_playback (.wav file(s))
            Wave file(s) of naturalistic real-bout USV sequences, plus their
            ``_spacing.txt`` / ``_usvids.txt`` side-cars.
        """

        self.message_output(f"Creating naturalistic USV playback file(s) started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        settings = self.create_playback_settings_dict
        context_label = settings['context_label']
        if context_label not in _PLAYBACK_CONTEXTS:
            msg = f"context_label must be one of {sorted(_PLAYBACK_CONTEXTS)}, got {context_label!r}."
            raise ValueError(msg)
        # `context_label` picks the sex subdirectory + context filename token; playback always
        # uses the newest matching build in that directory (no explicit file path). `sex` also
        # prefixes the output WAV / txt filenames ('male' / 'female' / 'mixed').
        context_token, sex = _PLAYBACK_CONTEXTS[context_label]
        try:
            repository_h5_path = newest_match_or_raise(
                resolve_data_root('naturalistic_usv_repository_dir') / sex,
                f"naturalistic_usv_repository_{context_token}_*.h5",
                key=lambda p: p.stat().st_mtime,
                label=f"{context_label} naturalistic USV repository",
            )
        except FileNotFoundError:
            # No build exists for this context: report it cleanly (the GUI shows this in its
            # output box) and stop, rather than raising an uncaught error / crashing.
            self.message_output(
                f"No naturalistic USV repository has been built for context '{context_label}'; "
                f"nothing to play back. Build one with build-naturalistic-usv-repository first."
            )
            return
        total_acceptable_playback_time = settings['total_acceptable_naturalistic_playback_time']
        complexity_enabled = settings['complexity_enabled']
        complexity_mask_threshold = settings['complexity_mask_threshold']
        complexity_start_fraction = settings['complexity_start_fraction']
        complexity_end_fraction = settings['complexity_end_fraction']
        complexity_bandwidth = settings['complexity_bandwidth']
        edge_silence_seconds = settings['edge_silence_seconds']
        max_isi_seconds = settings['max_isi_seconds']

        output_file_dir = resolve_data_root('naturalistic_usv_playback_dir')
        output_file_dir.mkdir(parents=True, exist_ok=True)

        # `playback_seed` is None by default (fresh entropy, non-reproducible); set it to
        # an integer for a documented, repeatable stimulus set. It seeds the numpy
        # Generator driving both the random bout draws and the fallback-ISI draws.
        rng = np.random.default_rng(settings['playback_seed'])

        with h5py.File(repository_h5_path, 'r') as repository:
            sampling_rate = int(repository.attrs['sampling_rate_hz'])
            bout_usv_start = repository['bout/usv_start'][:]
            bout_usv_count = repository['bout/usv_count'][:]
            bout_preceding_isi = repository['bout/preceding_isi_s'][:]
            usv_offset = repository['usv/offset'][:]
            usv_length = repository['usv/length'][:]
            usv_session = repository['usv/session'].asstr()[:]
            usv_row = repository['usv/usv_row'][:]
            usv_gap_to_next = repository['usv/gap_to_next_s'][:]
            audio_dataset = repository['audio']
            n_bout = int(bout_usv_start.shape[0])
            # Real inter-bout pauses (excluding the NaN of each session's first bout),
            # used as the ISI when a randomly drawn bout is a session's first bout.
            real_isi_pool = bout_preceding_isi[~np.isnan(bout_preceding_isi)]

            # Per-bout complex-fraction: the fraction of the bout's USVs that are "complex"
            # (mask_number >= threshold). Used to steer the bout draw toward a scheduled
            # target complexity; computed only when complexity steering is enabled.
            if complexity_enabled:
                usv_is_complex = repository['usv/features/mask_number'][:] >= complexity_mask_threshold
                bout_complex_fraction = np.array([
                    float(np.mean(usv_is_complex[int(s):int(s) + int(c)]))
                    for s, c in zip(bout_usv_start, bout_usv_count, strict=True)
                ])
            else:
                usv_is_complex = None
                bout_complex_fraction = None

            for _ in range(settings['num_naturalistic_usv_files']):
                smart_wait(app_context_bool=self.app_context_bool, seconds=1)
                current_time = datetime.today().strftime('%Y%m%d_%H%M%S')

                # Accumulate audio chunks + (sample_count, label) metadata, concatenated
                # ONCE after the loop (O(N)); spacing/usvids are written AFTER the loop
                # clamped to target_samples so they describe exactly the sliced WAV.
                replay_chunks: list[np.ndarray] = []
                meta_entries: list[tuple[int, str]] = []
                target_samples = int(total_acceptable_playback_time * sampling_rate)
                total_playback_time_created = 0.0
                last_time_updated = 0.0
                # Complexity-steering bookkeeping (achieved ratio + variety), logged after.
                drawn_bout_indices: list[int] = []
                complex_usv_placed = 0
                total_usv_placed = 0

                replay_txt_path = output_file_dir / f"{sex}_usv_playback_{total_acceptable_playback_time}s_{current_time}_spacing.txt"
                usv_id_txt_path = output_file_dir / f"{sex}_usv_playback_{total_acceptable_playback_time}s_{current_time}_usvids.txt"

                with (replay_txt_path.open('w+') as replay_txt_file,
                      usv_id_txt_path.open('w+') as usv_id_txt_file,
                      tqdm(total=total_acceptable_playback_time, desc="Generating Playback", unit="s") as pbar):

                    # fixed lead-in silence at the very start (stands in for the first bout's
                    # preceding pause, so the file opens on a call after a short, constant gap)
                    edge_silence_samples = int(np.ceil(edge_silence_seconds * sampling_rate))
                    replay_chunks.append(np.zeros(edge_silence_samples, dtype=np.int16))
                    meta_entries.append((edge_silence_samples, 'ISI'))
                    total_playback_time_created += edge_silence_seconds

                    first_bout = True
                    # stop once many consecutive draws no longer fit the remaining time
                    consecutive_skips = 0
                    while consecutive_skips < 100:
                        # draw a bout: uniform by default, or steered toward the scheduled
                        # target complexity (a linear ramp across the file position) when
                        # complexity_enabled -- bouts whose complex-fraction is near the
                        # current target are favoured via a Gaussian weight; repetition (with
                        # replacement) supplies complex bouts as needed
                        if complexity_enabled:
                            file_position = min(1.0, total_playback_time_created / total_acceptable_playback_time)
                            target_fraction = complexity_start_fraction + (complexity_end_fraction - complexity_start_fraction) * file_position
                            draw_weights = np.exp(-((bout_complex_fraction - target_fraction) ** 2) / (2.0 * complexity_bandwidth ** 2))
                            weight_sum = draw_weights.sum()
                            if weight_sum > 0:
                                bout_index = int(rng.choice(n_bout, p=draw_weights / weight_sum))
                            else:
                                # All Gaussian weights underflowed to 0 (a very small
                                # complexity_bandwidth with every bout far from the target
                                # complexity); fall back to a uniform draw instead of dividing
                                # 0/0 -> NaN, which would make rng.choice raise.
                                bout_index = int(rng.integers(n_bout))
                        else:
                            bout_index = int(rng.integers(n_bout))

                        usv_start = int(bout_usv_start[bout_index])
                        usv_count = int(bout_usv_count[bout_index])

                        # inter-sequence interval before this bout: the first placed bout has
                        # none (the fixed lead-in stands in for it); later bouts use the real
                        # recorded pause (NaN -> a sampled real ISI), clipped to max_isi_seconds
                        if first_bout:
                            isi = 0.0
                        else:
                            isi = float(bout_preceding_isi[bout_index])
                            if np.isnan(isi):
                                isi = (float(real_isi_pool[int(rng.integers(real_isi_pool.shape[0]))])
                                       if real_isi_pool.size else 0.0)
                            isi = min(isi, max_isi_seconds)
                        isi_samples = int(np.ceil(isi * sampling_rate))

                        # cost of this whole bout: its ISI + all USV audio + the real within-bout
                        # inter-USV gaps (IUIs are natural + small, so they are not clipped)
                        bout_usv_seconds = sum(int(usv_length[usv_start + p]) for p in range(usv_count)) / sampling_rate
                        bout_iui_seconds = sum(float(usv_gap_to_next[usv_start + p]) for p in range(usv_count - 1))
                        bout_seconds = isi + bout_usv_seconds + bout_iui_seconds

                        # skip a bout that (with the fixed lead-out) would overrun the target and
                        # try another, so the file ends on a whole bout without stopping early on
                        # one large draw; after many consecutive misses the remaining time is too
                        # small for anything and the loop stops. The first bout is always placed.
                        if not first_bout and (total_playback_time_created + bout_seconds + edge_silence_seconds) > total_acceptable_playback_time:
                            consecutive_skips += 1
                            continue
                        consecutive_skips = 0

                        if isi_samples > 0:
                            replay_chunks.append(np.zeros(isi_samples, dtype=np.int16))
                            meta_entries.append((isi_samples, 'ISI'))
                            total_playback_time_created += isi

                        if complexity_enabled:
                            drawn_bout_indices.append(bout_index)
                            bout_complex = usv_is_complex[usv_start:usv_start + usv_count]
                            complex_usv_placed += int(bout_complex.sum())
                            total_usv_placed += int(bout_complex.size)

                        # replay the bout's USVs in their natural order, each separated by
                        # its real within-bout inter-USV interval
                        for position in range(usv_count):
                            usv = usv_start + position
                            start_sample = int(usv_offset[usv])
                            length_samples = int(usv_length[usv])
                            usv_audio = np.asarray(audio_dataset[start_sample:start_sample + length_samples], dtype=np.int16)
                            usv_label = f"{usv_session[usv]}_usv{int(usv_row[usv]):05d}"
                            total_playback_time_created += usv_audio.shape[0] / sampling_rate

                            if position < (usv_count - 1):
                                iui = float(usv_gap_to_next[usv])
                                iui_samples = int(np.ceil(iui * sampling_rate))
                                total_playback_time_created += iui
                                replay_chunks.append(usv_audio)
                                meta_entries.append((usv_audio.shape[0], usv_label))
                                replay_chunks.append(np.zeros(iui_samples, dtype=np.int16))
                                meta_entries.append((iui_samples, 'IUI'))
                            else:
                                replay_chunks.append(usv_audio)
                                meta_entries.append((usv_audio.shape[0], usv_label))

                        first_bout = False

                        update_amount = int(np.floor(total_playback_time_created - last_time_updated))
                        pbar.update(update_amount)
                        last_time_updated = total_playback_time_created

                    # fixed lead-out silence at the very end (so the file closes on a call
                    # followed by a short, constant gap rather than a truncated pause)
                    replay_chunks.append(np.zeros(edge_silence_samples, dtype=np.int16))
                    meta_entries.append((edge_silence_samples, 'ISI'))
                    total_playback_time_created += edge_silence_seconds

                    if pbar.n < pbar.total:
                        pbar.update(pbar.total - pbar.n)

                    # Write spacing/usvids clamped to target_samples so they describe
                    # exactly the samples kept in the sliced WAV: walk the per-chunk
                    # entries, clamp the chunk straddling the truncation boundary, drop
                    # everything past it.
                    offset = 0
                    for count, label in meta_entries:
                        if offset >= target_samples:
                            break
                        kept = min(count, target_samples - offset)
                        replay_txt_file.write(f'{kept} \n')
                        usv_id_txt_file.write(f'{label} \n')
                        offset += count

                replay_wav_arr = (np.concatenate(replay_chunks) if replay_chunks
                                  else np.array([], dtype=np.int16))
                replay_wav_arr = replay_wav_arr[:target_samples]

                actual_total_time_sec = int(np.ceil(replay_wav_arr.shape[0] / sampling_rate))
                self.message_output(f"The total duration of the generated naturalistic playback file is {round(actual_total_time_sec / 60, 2)} min.")

                if complexity_enabled and total_usv_placed:
                    self.message_output(
                        f"Complexity steering (complex = mask_number >= {complexity_mask_threshold}): "
                        f"target {complexity_start_fraction:.2f} -> {complexity_end_fraction:.2f}, "
                        f"achieved {complex_usv_placed / total_usv_placed:.2f} complex USVs using "
                        f"{len(set(drawn_bout_indices))} distinct bouts of {n_bout}."
                    )

                wavfile.write(filename=output_file_dir / f"{sex}_usv_playback_{total_acceptable_playback_time}s_{current_time}.wav",
                              rate=sampling_rate,
                              data=replay_wav_arr)

        self.message_output(f"Creating naturalistic USV playback file(s) ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")

    def create_usv_playback_wav(self) -> None:
        """
        Description
        -----------
        This method takes .wav files containing individual USVs and concatenates them
        together with a known IPI period between each USV.

        NB: Run time for 10k USVs (~19 min .wav file) is ~18 minutes.

        Parameters
        ----------
        None
            Inputs are read from ``self.create_playback_settings_dict`` (the
            ``create_usv_playback_wav`` settings block) and ``self.exp_id``.

        Returns
        -------
        usv_playback (.wav file(s))
            Wave file(s) with concatenated USVs.
        """

        self.message_output(f"Creating USV playback file(s) started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        os_base_path = find_base_path()
        local_cup_mount_bool = os.path.ismount(os_base_path)
        if local_cup_mount_bool:
            playback_snippets_dir = Path(os_base_path) / self.exp_id / 'usv_playback_experiments' / self.create_playback_settings_dict['playback_snippets_dir']
            output_file_dir = Path(os_base_path) / self.exp_id / 'usv_playback_experiments' / 'usv_playback_files'
        else:
            playback_snippets_dir = Path(find_cluster_path()) / self.exp_id / 'usv_playback_experiments' / self.create_playback_settings_dict['playback_snippets_dir']
            output_file_dir = Path(find_cluster_path()) / self.exp_id / 'usv_playback_experiments' / 'usv_playback_files'

        output_file_dir.mkdir(parents=True, exist_ok=True)
        ipi_duration = self.create_playback_settings_dict['ipi_duration']
        wav_sampling_rate = self.create_playback_settings_dict['wav_sampling_rate']
        total_usv_number = self.create_playback_settings_dict['total_usv_number']

        # `playback_seed` is None by default (fresh entropy, non-reproducible);
        # set it to an integer for a documented, repeatable stimulus set. A
        # local random.Random avoids mutating the global `random` module state.
        py_rng = random.Random(self.create_playback_settings_dict['playback_seed'])

        for _ in range(self.create_playback_settings_dict['num_usv_files']):

            smart_wait(app_context_bool=self.app_context_bool, seconds=1)
            current_time = datetime.today().strftime('%Y%m%d_%H%M%S')

            wav_files_list = sorted(playback_snippets_dir.glob('*.wav'))
            if not wav_files_list:
                msg = (
                    f"create_usv_playback_wav: no .wav playback snippets found in "
                    f"{playback_snippets_dir!s}; cannot draw USV files for the playback sequence."
                )
                raise FileNotFoundError(msg)
            # Preload + validate each unique snippet once (snippets are drawn with
            # replacement from a small fixed pool, so the same files would otherwise
            # be re-read thousands of times across the build); the seeded draw
            # sequence is unchanged, so the output is byte-identical.
            _expected_snippet_rate_hz = int(wav_sampling_rate * 1e3)
            wav_cache = {wav_path: _read_int16_snippet(wav_path, expected_rate_hz=_expected_snippet_rate_hz) for wav_path in wav_files_list}

            ipi_duration_samples = int(np.ceil(ipi_duration * wav_sampling_rate * 1e3))

            arr_start_with_ipi = np.zeros(ipi_duration_samples).astype(np.int16)
            # Accumulate chunks in a list and concatenate ONCE after the loop (O(N));
            # re-growing the array with np.concatenate every USV recopies the whole
            # buffer -> O(N^2) (the source of the docstring's "~18 minutes for 10k USVs").
            replay_chunks = [arr_start_with_ipi.copy()]

            with (open(output_file_dir / f"usv_playback_n={total_usv_number}_{current_time}_spacing.txt",
                       'w+') as replay_txt_file,
                  open(output_file_dir / f"usv_playback_n={total_usv_number}_{current_time}_usvids.txt",
                       'w+') as usv_id_txt_file):
                replay_txt_file.write(f'{ipi_duration_samples} \n')
                for _ in tqdm(range(total_usv_number)):
                    random_wav_file = py_rng.choice(wav_files_list)
                    random_wav_file_data = wav_cache[random_wav_file]
                    replay_chunks.append(random_wav_file_data)
                    replay_chunks.append(arr_start_with_ipi)
                    replay_txt_file.write(f'{random_wav_file_data.shape[0]} \n')
                    replay_txt_file.write(f'{ipi_duration_samples} \n')
                    usv_id_txt_file.write(f'{random_wav_file.name} \n')

            replay_wav_arr = np.concatenate(replay_chunks)
            self.message_output(f"The total duration of the generated playback file is {round(replay_wav_arr.shape[0] / (wav_sampling_rate * 1e3) / 60, 2)} min.")

            wavfile.write(filename=output_file_dir / f"usv_playback_n={total_usv_number}_{current_time}.wav",
                          rate=int(wav_sampling_rate * 1e3),
                          data=replay_wav_arr)

        self.message_output(f"Creating USV playback file(s) ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")

    def frequency_shift_audio_segment(self, seq_start: float = None, seq_duration: float = None) -> Path | None:
        """
        Description
        -----------
        This method takes a temporal sequence from an existing USV .wav recording and pitch
        shifts (e.g., shifting down by a tritone) the sequence to human audible range.

        There are several steps in this procedure:
        (1) pitch-shifting occurs on the raw input audio segment
        (2) volume is modulated dynamically on this signal (this is optional):
            the transfer function ('6:-70,...') says that very soft sounds (below -70dB)
            will remain unchanged; sounds in the range -60dB to 0dB (maximum volume)
            will be boosted so that the 60dB dynamic  range of the original music will be
            compressed 3-to-1 into a 20dB range; the -5 (dB) output gain is needed to
            avoid clipping; -90 (dB) for the initial volume will work fine for a clip that
            starts with near silence; the delay of 0.2 (seconds) has the effect of causing
            the compander to react a bit more quickly to sudden volume changes
        (3) stationary noise reduction is applied to the signal
            3 standard deviations above mean to place the threshold between
            signal and noise
        (4) tempo is adjusted to match the duration of the original audio segment

        These audio files are to be used for presentation purposes only.

        NB, relevant term:
        octave: interval between one pitch and another with double its frequency (12 semitones)

        Parameters
        ----------
        seq_start (float)
            Start time (in seconds) of the segment to extract from the source .wav; when None,
            falls back to ``fs_sequence_start`` in the frequency-shift settings dictionary. This
            override lets callers (e.g., the behavioral-video step) align the audible segment to
            an externally-defined window without mutating the settings file.
        seq_duration (float)
            Duration (in seconds) of the segment to extract; when None, falls back to
            ``fs_sequence_duration`` in the frequency-shift settings dictionary.

        Returns
        -------
        final_output_file (pathlib.Path or None)
            Absolute path to the written audible .wav file (saved in the
            'audio/frequency_shifted_audio_segments' directory), or None if the source audio file
            could not be uniquely located.
        """

        self.message_output(f"Frequency shifting of audio segment by {abs(self.freq_shift_settings_dict['fs_octave_shift'])} octaves started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        audio_dir = self.freq_shift_settings_dict['fs_audio_dir']
        device_id = self.freq_shift_settings_dict['fs_device_id']
        channel_id = self.freq_shift_settings_dict['fs_channel_id']

        wav_sampling_rate = self.freq_shift_settings_dict['fs_wav_sampling_rate']
        seq_start = seq_start if seq_start is not None else self.freq_shift_settings_dict['fs_sequence_start']
        seq_duration = seq_duration if seq_duration is not None else self.freq_shift_settings_dict['fs_sequence_duration']
        octave_shift = self.freq_shift_settings_dict['fs_octave_shift']
        volume_adjustment = self.freq_shift_settings_dict['fs_volume_adjustment']
        compand_transfer = self.freq_shift_settings_dict['fs_compand_transfer']
        noise_reduction_std_threshold = self.freq_shift_settings_dict['fs_noise_reduction_std_threshold']
        sinc_upper_cutoff_hz = self.freq_shift_settings_dict['fs_sinc_upper_cutoff_hz']

        audio_file_loc = list((Path(self.root_directory) / 'audio' / audio_dir).glob(f"*{device_id}_*_ch{channel_id:02d}_*.wav"))

        if len(audio_file_loc) != 1:
            self.message_output("Requested audio file not found. Please try again.")
            return None

        output_dir = Path(self.root_directory) / 'audio' / 'frequency_shifted_audio_segments'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file_name = f"{audio_file_loc[0].name}_start={seq_start}s_duration={seq_duration}s_octave_shift={octave_shift}"

        # load audio sequence with Librosa
        original_audio, original_sr = librosa.load(
            audio_file_loc[0],
            sr=int(wav_sampling_rate * 1e3),
            offset=seq_start,
            duration=seq_duration
        )

        # calculate the new sample rate for resampling (changes pitch AND speed)
        new_sr = int(original_sr * (2.0 ** octave_shift))

        # intermediate filenames for the processing pipeline
        temp_resampled_file = output_dir / f"{output_file_name}_temp_resampled.wav"
        temp_audible_file = output_dir / f"{output_file_name}_temp_audible.wav"
        temp_denoised_file = output_dir / f"{output_file_name}_temp_denoised.wav"
        final_output_file = output_dir / f"{output_file_name}_audible_denoised_tempo_adjusted.wav"

        # export the resampled audio (this is the pitch/speed shift)
        sf.write(temp_resampled_file, original_audio, new_sr)

        # perform volume adjustment with SoX (if needed)
        if volume_adjustment:
            subprocess.Popen(args=f'''{self.command_addition}static_sox {temp_resampled_file} {temp_audible_file} compand {compand_transfer}''',
                             cwd=output_dir,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT,
                             shell=self.shell_usage_bool).wait()

            processed_audio, _ = librosa.load(temp_audible_file, sr=new_sr)
        else:
            processed_audio, _ = librosa.load(temp_resampled_file, sr=new_sr)

        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=processed_audio, sr=new_sr, stationary=True, n_std_thresh_stationary=noise_reduction_std_threshold)
        sf.write(temp_denoised_file, reduced_noise, new_sr)

        # correct the tempo back to the original duration using SoX
        tempo_adjustment_factor = original_sr / new_sr

        if 'filtered' not in audio_dir:
            upper_cutoff_freq = int(np.ceil(sinc_upper_cutoff_hz / (2 ** abs(octave_shift))))
            subprocess.Popen(args=f'''{self.command_addition}static_sox {temp_denoised_file} {final_output_file} sinc {upper_cutoff_freq}-0 tempo -s {tempo_adjustment_factor}''',
                             cwd=output_dir,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT,
                             shell=self.shell_usage_bool).wait()
        else:
            subprocess.Popen(args=f'''{self.command_addition}static_sox {temp_denoised_file} {final_output_file} tempo -s {tempo_adjustment_factor}''',
                             cwd=output_dir,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT,
                             shell=self.shell_usage_bool).wait()

        temp_resampled_file.unlink()
        if volume_adjustment:
            temp_audible_file.unlink()
        temp_denoised_file.unlink()

        return final_output_file
