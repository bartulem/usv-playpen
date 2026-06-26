"""
@author: bartulem
Generates playback WAV files and frequency shifts audio segment.
"""

import os
import random
import subprocess
from datetime import datetime
from pathlib import Path

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

from ..os_utils import find_base_path, find_cluster_path
from ..time_utils import is_gui_context, smart_wait
from .mixture_model_utils import TMixture, _sample_from_mixture
from .usv_interval_archive import read_usv_interval_h5, reconstruct_best_model


def _split_iui_isi(model: TMixture) -> tuple[TMixture, TMixture]:
    """
    Description
    -----------
    Splits a per-sex log-space Student-t interval mixture into the two
    sub-models the naturalistic playback generator needs:

    * the within-sequence inter-USV-interval (IUI) pool -- every component
      EXCEPT the slowest, with mixing weights renormalised to sum to 1, and
    * the between-sequence inter-sequence-interval (ISI) model -- the single
      slowest (longest-interval) component.

    The reconstructed model returned by :func:`reconstruct_best_model` is
    pre-sorted in ascending log-mean order, so the slowest component is the
    last one (index ``K - 1``); no re-sorting is performed here. Pooling every
    non-slowest component into the IUI sub-model (rather than using only the
    fastest breathing-expiration component) keeps the full within-sequence
    interval mass -- this matters for sexes whose characteristic gap sits in a
    middle component (e.g. the female ~0.9 s component carrying the bulk of the
    mass) rather than in the fastest one.

    Parameters
    ----------
    model (TMixture)
        A fitted per-sex Student-t mixture (components ascending by log-mean).

    Returns
    -------
    iui_model (TMixture)
        Sub-mixture of all components except the slowest, weights renormalised.
    isi_model (TMixture)
        Single-component model holding the slowest component only.
    """

    K = model.n_components
    if K < 2:
        msg = (
            "create_naturalistic_usv_playback_wav: need >= 2 Student-t "
            f"components to separate IUI from ISI, but the model has K={K}."
        )
        raise ValueError(msg)

    means = model.means_.ravel()
    covariances = model.covariances_.ravel()
    nus = model.nus_.ravel()
    weights = model.weights_.ravel()

    iui_weights = weights[:-1] / weights[:-1].sum()
    iui_model = TMixture(
        weights=iui_weights,
        means=means[:-1],
        covariances=covariances[:-1],
        nus=nus[:-1],
    )
    isi_model = TMixture(
        weights=np.array([1.0]),
        means=means[-1:],
        covariances=covariances[-1:],
        nus=nus[-1:],
    )
    return iui_model, isi_model


def _mixture_log_bounds(model: TMixture, clip_pct: float) -> tuple[float, float]:
    """
    Description
    -----------
    Estimates a symmetric log-space percentile band ``[100 - clip_pct,
    clip_pct]`` for a Student-t mixture by sampling it once. The band is used
    to reject-resample draws so the heavy tails of low-``nu`` components cannot
    emit an absurdly long (or short) interval -- a raw draw from a ``nu ~ 3``
    component, once exponentiated, can otherwise exceed the entire playback
    duration.

    A fixed-seed local generator is used so the bounds are deterministic for a
    given model and do NOT perturb the caller's reproducible draw stream.

    Parameters
    ----------
    model (TMixture)
        The (sub-)mixture to bound.
    clip_pct (float)
        Upper percentile of the retained band (e.g. ``99.5``); the lower edge
        is its complement (``100 - clip_pct``).

    Returns
    -------
    lo_log (float)
        Lower log-space bound (the ``100 - clip_pct`` percentile).
    hi_log (float)
        Upper log-space bound (the ``clip_pct`` percentile).
    """

    bound_rng = np.random.default_rng(0)
    log_samples = _sample_from_mixture(model, 200000, bound_rng)
    lo_log = float(np.percentile(log_samples, 100.0 - clip_pct))
    hi_log = float(np.percentile(log_samples, clip_pct))
    return lo_log, hi_log


def _draw_bounded_seconds(
    model: TMixture,
    rng: np.random.Generator,
    lo_log: float,
    hi_log: float,
) -> float:
    """
    Description
    -----------
    Draws a single interval (in seconds) from a log-space Student-t mixture,
    reject-resampling until the log-space draw falls inside ``[lo_log,
    hi_log]``, then exponentiating. This caps the heavy-tailed components
    without distorting the bulk of the distribution.

    Parameters
    ----------
    model (TMixture)
        The (sub-)mixture to sample from.
    rng (np.random.Generator)
        The caller's random generator (seeded by ``playback_seed`` for
        reproducible stimulus sets).
    lo_log (float)
        Lower log-space acceptance bound.
    hi_log (float)
        Upper log-space acceptance bound.

    Returns
    -------
    interval_seconds (float)
        The accepted draw, exponentiated from log-space into seconds.
    """

    while True:
        draw = float(_sample_from_mixture(model, 1, rng)[0])
        if lo_log <= draw <= hi_log:
            return float(np.exp(draw))


def _read_int16_snippet(snippet_path: Path) -> np.ndarray:
    """
    Description
    -----------
    Read a playback-snippet WAV and return its sample array, enforcing that it is
    ``int16``. The playback builders concatenate snippets with explicitly ``int16``
    silence and write an ``int16`` WAV; a snippet of any other dtype (``int32`` /
    ``float32`` / ``uint8``) would make ``np.concatenate`` silently upcast the whole
    buffer, changing the output bit depth, and a ``float32`` snippet in ``[-1, 1]``
    mixed with ``int16`` zeros would have nonsensical relative amplitude.

    Parameters
    ----------
    snippet_path (Path)
        Path to the snippet ``.wav`` file.

    Returns
    -------
    snippet_data (np.ndarray)
        The snippet samples, guaranteed ``int16``.

    Raises
    ------
    ValueError
        If the snippet WAV is not ``int16``.
    """

    _, snippet_data = wavfile.read(snippet_path)
    if snippet_data.dtype != np.int16:
        msg = (
            f"playback snippet {Path(snippet_path).name!r} has dtype "
            f"{snippet_data.dtype} but must be int16 -- the playback silence, seed, "
            f"and output WAV are all int16, so a non-int16 snippet would upcast the "
            f"output bit depth and corrupt relative amplitudes. Re-export the "
            f"snippets as 16-bit PCM."
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
        Constructs naturalistic USV playback sequences by sampling inter-event
        intervals and sequence lengths from empirically derived distributions.

        Inter-USV intervals (IUI) and inter-sequence intervals (ISI) are sampled
        from a sex-specific log-space Student-t mixture model fit to the
        empirical end-to-start (``e2s``) inter-USV interval distribution. The
        fitted model is read live from the HDF5 interval archive
        (``naturalistic_iui_archive_h5``); the number of components ``K`` is the
        per-sex value selected by the archive's bootstrap-LRT step-up procedure
        (``K_selected_<sex>``), so no component counts or parameters are
        hard-coded here. The reconstructed mixture (components ascending by
        log-mean) is split into two roles:
            - ISI: the slowest (longest-interval) component only -- the long
              quiet pause between sequences.
            - IUI: the pool of all remaining (faster) components, weights
              renormalised -- the short within-sequence gaps. Pooling rather
              than using only the fastest breathing-expiration component keeps
              the full within-sequence interval mass (e.g. the female ~0.9 s
              component, which carries most of her mass).

        Because the low-``nu`` Student-t components have heavy tails, every draw
        is reject-resampled to the ``[100 - clip_pct, clip_pct]`` percentile band
        of its sub-mixture before being exponentiated, so a single draw cannot
        emit an absurdly long silence. The clip percentile is read per sex from
        ``naturalistic_interval_clip_pct`` (a ``{'male': ..., 'female': ...}``
        dict), since the sexes' within-sequence interval spreads differ.

        Sequence length is drawn from N(13, 5) clipped to [3, 23] USVs.
        Sex is inferred from naturalistic_playback_snippets_dir_prefix.

        The way the code works is as follows:
        (1) it finds all .wav files in the specified directory (female or male)
        (2) it draws a long inter-sequence quiet interval (ISI) from the slowest
            t-mixture component (bounded), then exponentiates
        (3) it draws a sequence length from N(13, 5) clipped to [3, 23]
        (4) it plays that many pseudo-randomly chosen USVs, each separated by
            a short inter-USV interval (IUI) drawn from the IUI sub-mixture
            (bounded), then exponentiated
        (5) it goes back to (2) and repeats until exceeding total playback time

        NB: Run time for ~18 min .wav file is ~2 minutes.

        Parameters
        ----------
        None
            Inputs are read from ``self.create_playback_settings_dict`` (the
            ``create_naturalistic_usv_playback_wav`` settings block) and ``self.exp_id``.

        Returns
        -------
        usv_playback (.wav file(s))
            Wave file(s) with naturalistic sequences of USVs.
        """

        self.message_output(f"Creating naturalistic USV playback file(s) started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}")
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        os_base_path = find_base_path()
        local_cup_mount_bool = os.path.ismount(os_base_path)
        prefix = self.create_playback_settings_dict['naturalistic_playback_snippets_dir_prefix']
        if local_cup_mount_bool:
            playback_snippets_dir = Path(os_base_path) / self.exp_id / 'usv_playback_experiments' / f"{prefix}_usv_playback_snippets"
            output_file_dir = Path(os_base_path) / self.exp_id / 'usv_playback_experiments' / 'naturalistic_usv_playback_files'
        else:
            playback_snippets_dir = Path(find_cluster_path()) / self.exp_id / 'usv_playback_experiments' / f"{prefix}_usv_playback_snippets"
            output_file_dir = Path(find_cluster_path()) / self.exp_id / 'usv_playback_experiments' / 'naturalistic_usv_playback_files'

        output_file_dir.mkdir(parents=True, exist_ok=True)

        wav_sampling_rate = self.create_playback_settings_dict['naturalistic_wav_sampling_rate']
        total_acceptable_playback_time = self.create_playback_settings_dict['total_acceptable_naturalistic_playback_time']

        # Reconstruct the sex-specific Student-t interval model live from the
        # HDF5 archive (no hard-coded parameters), using the per-sex component
        # count the archive's bootstrap-LRT selected. Split it into the
        # within-sequence IUI pool and the between-sequence ISI component, then
        # precompute per-sub-model percentile bounds for heavy-tail clipping.
        sex = 'female' if 'female' in prefix.lower() else 'male'
        interval_mode = self.create_playback_settings_dict['naturalistic_interval_mode']
        clip_pct = self.create_playback_settings_dict['naturalistic_interval_clip_pct'][sex]
        interval_archive = read_usv_interval_h5(self.create_playback_settings_dict['naturalistic_iui_archive_h5'])
        mode_node = interval_archive['modes'][interval_mode]
        k_selected = int(mode_node['attrs'][f'K_selected_{sex}'])
        interval_model, _ = reconstruct_best_model(mode_node['gmm_fits'], sex, k_selected)
        iui_model, isi_model = _split_iui_isi(interval_model)
        iui_lo_log, iui_hi_log = _mixture_log_bounds(iui_model, clip_pct)
        isi_lo_log, isi_hi_log = _mixture_log_bounds(isi_model, clip_pct)
        self.message_output(
            f"Loaded {interval_mode} Student-t interval model for '{sex}' (K={k_selected}) from archive; "
            f"IUI pool = {iui_model.n_components} component(s), ISI = slowest component."
        )

        # `playback_seed` is None by default (fresh entropy, non-reproducible);
        # set it to an integer to generate a documented, repeatable stimulus set.
        # It seeds BOTH a numpy Generator (`rng`, which drives the ISI/IUI interval
        # draws and the sequence-length draw -- i.e. the substantive interval/length
        # structure) and a local random.Random (`py_rng`, used only for USV-file
        # selection); the latter avoids mutating the global `random` module state.
        playback_seed = self.create_playback_settings_dict['playback_seed']
        rng = np.random.default_rng(playback_seed)
        py_rng = random.Random(playback_seed)

        for _ in range(self.create_playback_settings_dict['num_naturalistic_usv_files']):
            smart_wait(app_context_bool=self.app_context_bool, seconds=1)
            current_time = datetime.today().strftime('%Y%m%d_%H%M%S')

            wav_files_list = sorted(playback_snippets_dir.glob('*.wav'))
            # Accumulate chunks in a list, concatenated ONCE after the while loop
            # (O(N)); per-iteration np.concatenate recopies the whole buffer -> O(N^2).
            replay_chunks = []
            # Collect the per-chunk (sample_count, label) metadata in parallel with the
            # audio chunks and write the spacing/usvids .txt files AFTER the loop,
            # clamped to target_samples, so they describe exactly the samples kept in
            # the sliced WAV. The inner sequence loop can overshoot the time budget;
            # the WAV is sliced to target_samples but the metadata must match it, or
            # downstream alignment walking spacing.txt desynchronizes past the cut.
            meta_entries: list[tuple[int, str]] = []
            target_samples = int(total_acceptable_playback_time * wav_sampling_rate * 1e3)

            total_playback_time_created = 0
            last_time_updated = 0  # Variable to track progress for tqdm

            replay_txt_path = output_file_dir / f"{prefix}_usv_playback_{total_acceptable_playback_time}s_{current_time}_spacing.txt"
            usv_id_txt_path = output_file_dir / f"{prefix}_usv_playback_{total_acceptable_playback_time}s_{current_time}_usvids.txt"

            # This ensures files are opened only once and not overwritten.
            with (replay_txt_path.open('w+') as replay_txt_file,
                  usv_id_txt_path.open('w+') as usv_id_txt_file,
                  tqdm(total=total_acceptable_playback_time, desc="Generating Playback", unit="s") as pbar):

                while total_playback_time_created < total_acceptable_playback_time:
                    # inter-sequence interval: draw the long between-sequence pause
                    # from the slowest t-mixture component, reject-resampled to the
                    # [100 - clip_pct, clip_pct] percentile band so a heavy tail cannot
                    # emit an absurdly long silence
                    isi = _draw_bounded_seconds(isi_model, rng, isi_lo_log, isi_hi_log)

                    # if this ISI alone would consume all remaining time, stop rather
                    # than filling the tail of the file with silence
                    if total_playback_time_created + isi >= total_acceptable_playback_time:
                        break

                    # `wav_sampling_rate` is stored in kHz (e.g. 250), so multiplying by
                    # 1e3 converts it to samples-per-second (Hz) before scaling by seconds
                    isi_samples = int(np.ceil(isi * wav_sampling_rate * 1e3))

                    replay_chunks.append(np.zeros(isi_samples, dtype=np.int16))
                    meta_entries.append((isi_samples, 'ISI'))

                    total_playback_time_created += isi

                    # sequence length: Gaussian(13, 5) clipped to [3, 23]
                    usv_seq_length = int(np.clip(round(rng.normal(13, 5)), 3, 23))
                    for usv_idx in range(usv_seq_length):
                        # pick USV file
                        random_wav_file = py_rng.choice(wav_files_list)
                        random_wav_file_data = _read_int16_snippet(random_wav_file)
                        total_playback_time_created += (random_wav_file_data.shape[0] / (wav_sampling_rate * 1e3))

                        if usv_idx < (usv_seq_length - 1):
                            # inter-USV interval: draw the within-sequence gap from the
                            # IUI sub-mixture (all components except the slowest),
                            # reject-resampled to the percentile band
                            iui = _draw_bounded_seconds(iui_model, rng, iui_lo_log, iui_hi_log)
                            iui_samples = int(np.ceil(iui * wav_sampling_rate * 1e3))
                            total_playback_time_created += iui

                            replay_chunks.append(random_wav_file_data)
                            meta_entries.append((random_wav_file_data.shape[0], random_wav_file.name))
                            replay_chunks.append(np.zeros(iui_samples, dtype=np.int16))
                            meta_entries.append((iui_samples, 'IUI'))
                        else:
                            replay_chunks.append(random_wav_file_data)
                            meta_entries.append((random_wav_file_data.shape[0], random_wav_file.name))

                    # manually update the progress bar at the end of each loop
                    update_amount = int(np.floor(total_playback_time_created - last_time_updated))
                    pbar.update(update_amount)
                    last_time_updated = total_playback_time_created

                if pbar.n < pbar.total:
                    pbar.update(pbar.total - pbar.n)

                # Write the spacing / usvids metadata clamped to target_samples so it
                # describes exactly the samples kept in the sliced WAV below: walk the
                # per-chunk entries, clamp the single chunk that straddles the
                # truncation boundary to its in-WAV length, and drop everything past it.
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

            actual_total_time_sec = int(np.ceil(replay_wav_arr.shape[0] / (wav_sampling_rate * 1e3)))
            self.message_output(f"The total duration of the generated naturalistic playback file is {round(actual_total_time_sec / 60, 2)} min.")

            wavfile.write(filename=output_file_dir / f"{prefix}_usv_playback_{total_acceptable_playback_time}s_{current_time}.wav",
                          rate=int(wav_sampling_rate * 1e3),
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
                    random_wav_file_data = _read_int16_snippet(random_wav_file)
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
            subprocess.Popen(args=f'''{self.command_addition}static_sox {temp_resampled_file} {temp_audible_file} compand 0.3,1 6:-70,-60,-20 -5 -90 0.2''',
                             cwd=output_dir,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.STDOUT,
                             shell=self.shell_usage_bool).wait()

            processed_audio, _ = librosa.load(temp_audible_file, sr=new_sr)
        else:
            processed_audio, _ = librosa.load(temp_resampled_file, sr=new_sr)

        # perform noise reduction
        reduced_noise = nr.reduce_noise(y=processed_audio, sr=new_sr, stationary=True, n_std_thresh_stationary=3)
        sf.write(temp_denoised_file, reduced_noise, new_sr)

        # correct the tempo back to the original duration using SoX
        tempo_adjustment_factor = original_sr / new_sr

        if 'filtered' not in audio_dir:
            upper_cutoff_freq = int(np.ceil(25000 / (2 ** abs(octave_shift))))
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
