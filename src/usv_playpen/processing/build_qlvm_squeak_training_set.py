"""
@author: bartulem

Build a QLVM training set (``.npz``) of squeaks (broadband vocalizations) from a
list of session root directories, drawn the way the training sets of the phase 3
squeak QLVM cells (``qlvm_models_latest/phase3_BBVs_qlvm``) were drawn.

This is the in-house port of the reference builder ``build_bbv_dataset.py``
(MMMmB repository). It lives in its own module, not in :mod:`build_qlvm_training_set`,
because it reads its spectrograms through :mod:`detect_usv_squeaks` (which itself
imports ``stretch_specs`` from :mod:`build_qlvm_training_set`, so importing it back
there would be circular) and because it shares none of the USV builder's inputs:
no spectrogram H5, no SAM masks, no mask-count strata. It shares the session split
(:func:`build_qlvm_training_set.split_sessions_by_type`), the session typing and
the resize (:func:`build_qlvm_training_set.stretch_specs`). The steps:

1. **Candidates.** Every row of every session's ``*_usv_summary.csv`` that
   ``detect-usv-squeaks`` marked ``squeak`` true (pure squeaks and segments
   holding both a squeak and a USV; with ``exclude_noise`` also not ``noise``, a
   guard, since noise rows carry null booleans) and gave a squeak extent
   (``squeak_start`` / ``squeak_end``, the envelope of the segment's
   above-threshold squeak frames) is a candidate; a row with several squeaks is
   ONE candidate, cropped to that envelope, as
   ``infer-qlvm-squeak-latents`` embeds it. Its audio window is the segment
   widened to hold the envelope plus the context frames
   (:func:`detect_usv_squeaks.squeak_crop_window`: a squeak often extends past its
   segment), its first and last squeak frames are the window frames whose centres
   lie inside the envelope (:func:`detect_usv_squeaks.squeak_extent_frames`), and
   its frame count is that of the window's sonic spectrogram
   (``1 + n_samples // 512``, :func:`detect_usv_squeaks.squeak_window_n_frames`).
   Sessions are taken in sorted id order and rows in summary order, whatever the
   order of the session list.
2. **Crop gates** (:func:`squeak_crop_gates`). With ``crop_window``
   ``"full_length"`` (the default, and the rule ``infer-qlvm-squeak-latents``
   embeds with) the crop may lie anywhere in the segment; with
   ``"first_128_frames"`` (the reference rule, forced by the reference
   builder's 128-frame spectrogram store) the segment is cut to its first 128
   frames and a squeak still going at frame 127 of a longer segment
   (right-censored) is dropped. The extent is then widened by
   ``context_frames`` either side (clipped to the segment), and crops narrower
   than ``min_trimmed_frames`` or wider than the 128-frame frame are dropped.
   The crop width sets the duration stratum (``duration_bin_edges``,
   ``np.digitize``).
3. **Draw** (one generator, ``np.random.default_rng(random_state)``). A
   ``per_session_bin_cap`` above 0 first keeps at most that many rows of any one
   session in any one stratum (:func:`cap_rows_per_session_bin`; the ``session``
   cells; 0 is the ``lumped`` cells), then ``n_total`` rows are drawn from what is
   left, ``"natural"`` (uniformly) or ``"uniform"`` (equal across strata,
   :func:`stratified_duration_draw`). ``full_dataset`` takes every row that passed
   the gates instead and also writes ``full_data.npz``.
4. **Spectrograms.** The drawn rows' sonic spectrograms are rebuilt from the
   session audio over their windows (:func:`detect_usv_squeaks.squeak_window_spectrograms`,
   the front end that reproduces the reference ``_sonic_wav_`` store), each crop is
   normalized (``crop_normalization``: ``"per_crop"``,
   ``(x - min) / (max - min + 1e-6)``, or ``"absolute"``,
   ``(clip(x_dB, -100, 50) + 25) / 75``), written from column 0 of a 128-frame
   zero frame and centred with ``stretch_specs`` (no time stretch).
5. **Split and write.** Whole sessions are held out per type
   (:func:`split_sessions_by_type`), and ``train_data.npz`` / ``val_data.npz``
   (then ``full_data.npz``) and ``metadata.npz`` are written in the format
   ``train-qlvm`` reads: ``spectrograms``, all-zero ``masks`` and ``masks_len``,
   ``durations`` (crop width), ``spec_id`` (``{session}_{row}``), ``session_id``,
   ``session_type``, ``duration_bin``, ``squeak_probability`` (the classifier's
   probability that the segment holds a squeak, ``p_squeak + p_both``),
   ``crop_first``, ``crop_last`` (frames of the audio window),
   ``n_frames_original`` (frames of the window) and ``apply_mask`` False.

Reproducing the phase 3 sets needs ``crop_window`` ``"first_128_frames"`` and
``exclude_metadata_audio_channels`` false (the setting the bit-identical rebuild
of ``bbv-natural_dur-26-40-62_session_N11000_seed42`` was verified with), and
the candidates of the reference squeak index (the retired binary classifier's
segment flags and the extents of its 128-frame pass), which no summary keeps any
more; :meth:`QLVMSqueakTrainingSetBuilder.build_from_candidates` accepts such
externally supplied candidates (with the segment itself as the audio window). See
``docs/Process.rst``.
"""

from __future__ import annotations

import json
import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import numpy as np
import polars as pls
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import (
    SQUEAK_FLAG_COLUMN,
    USV_FLAG_COLUMN,
    first_match_or_raise,
    squeak_bearing_mask,
)
from ..time_utils import is_gui_context, smart_wait
from .build_qlvm_training_set import (
    parse_int_list,
    session_type_from_metadata,
    split_sessions_by_type,
    stretch_specs,
)
from .detect_usv_squeaks import (
    SQUEAK_REFERENCE_WINDOW_FRAMES,
    SQUEAK_SPEC_PARAMS,
    squeak_crop_window,
    squeak_extent_frames,
    squeak_window_n_frames,
    squeak_window_spectrograms,
)

# Width (frames) of the frame every crop is written into before the resize: the
# 128-frame window of the squeak store the phase 3 sets were cut from.
CROP_FRAME_WIDTH = SQUEAK_REFERENCE_WINDOW_FRAMES

# Epsilon of the per-crop min-max (build_bbv_dataset.py --normalization per-crop).
CROP_MINMAX_EPSILON = 1e-6


def squeak_candidates_from_summary(usv_summary: pls.DataFrame, exclude_noise: bool) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    The squeak candidates of one session: rows with ``squeak`` true (pure squeaks
    and "both", :func:`os_utils.squeak_bearing_mask`; and, with ``exclude_noise``,
    ``noise`` not true; noise rows carry null booleans, so this is a guard), with a
    finite squeak extent
    (``squeak_start`` / ``squeak_end``) and an audio window of at least one frame.
    Each candidate's audio window is the segment widened to hold the envelope plus
    its context (:func:`detect_usv_squeaks.squeak_crop_window`), and its extent
    frames are the window frames whose centres lie inside the envelope
    (:func:`detect_usv_squeaks.squeak_extent_frames`). Null flags count as false.

    Parameters
    ----------
    usv_summary (pls.DataFrame)
        The session's USV summary (``start``, ``stop``, ``usv``, ``squeak``,
        ``p_squeak``, ``p_both``, ``squeak_start``, ``squeak_end`` and, with
        ``exclude_noise``, ``noise``).
    exclude_noise (bool)
        Leave out rows flagged as noise.

    Returns
    -------
    candidates (dict[str, np.ndarray])
        Row-aligned arrays: ``row`` (summary row), ``start_s`` / ``stop_s`` (the
        audio window, s), ``n_frames`` (frames of the window), ``first_raw`` /
        ``last_raw`` (first and last squeak frame of the window) and
        ``probability`` (``p_squeak + p_both``, the probability that the segment
        holds a squeak).

    Raises
    ------
    ValueError
        A needed column is missing.
    """

    needed = ["start", "stop", USV_FLAG_COLUMN, SQUEAK_FLAG_COLUMN, "p_squeak", "p_both", "squeak_start", "squeak_end", *(["noise"] if exclude_noise else [])]
    missing = [column for column in needed if column not in usv_summary.columns]
    if missing:
        error_message = f"The USV summary has no {missing} column(s); run detect-usv-noise and detect-usv-squeaks first."
        raise ValueError(error_message)
    keep = squeak_bearing_mask(usv_summary, "the USV summary").to_numpy().copy()
    if exclude_noise:
        keep &= ~usv_summary["noise"].cast(pls.Boolean).fill_null(False).to_numpy()
    start = usv_summary["start"].cast(pls.Float64).to_numpy()
    stop = usv_summary["stop"].cast(pls.Float64).to_numpy()
    squeak_start = usv_summary["squeak_start"].cast(pls.Float64).fill_null(np.nan).to_numpy()
    squeak_end = usv_summary["squeak_end"].cast(pls.Float64).fill_null(np.nan).to_numpy()
    keep &= np.isfinite(squeak_start) & np.isfinite(squeak_end)
    window_start = np.full(start.size, np.nan)
    window_stop = np.full(start.size, np.nan)
    window_start[keep], window_stop[keep] = squeak_crop_window(start[keep], stop[keep], squeak_start[keep], squeak_end[keep])
    n_frames = np.zeros(start.size, dtype=np.int64)
    n_frames[keep] = squeak_window_n_frames(window_start[keep], window_stop[keep])
    keep &= n_frames > 0
    rows = np.flatnonzero(keep).astype(np.int64)
    first_raw, last_raw = squeak_extent_frames(window_start[rows], squeak_start[rows], squeak_end[rows])
    probability = (usv_summary["p_squeak"].cast(pls.Float64).fill_null(np.nan).to_numpy()
                   + usv_summary["p_both"].cast(pls.Float64).fill_null(np.nan).to_numpy())
    return {
        "row": rows,
        "start_s": window_start[rows],
        "stop_s": window_stop[rows],
        "n_frames": n_frames[rows],
        "first_raw": first_raw,
        "last_raw": last_raw,
        "probability": probability[rows],
    }


def squeak_crop_gates(
    n_frames: np.ndarray,
    first_raw: np.ndarray,
    last_raw: np.ndarray,
    crop_window: str,
    context_frames: int,
    min_trimmed_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, int]]]:
    """
    Description
    -----------
    Turns squeak extents into crops and applies the curation gates, in the order
    of ``build_bbv_dataset.py``. Under ``"first_128_frames"`` the segment is cut
    to its first 128 frames and a squeak of a longer segment whose last frame is
    frame 127 or later is dropped (right-censored: its end is not measured);
    under ``"full_length"`` no segment is cut. The extent is widened by
    ``context_frames`` either side and clipped to the (cut) segment; crops
    narrower than ``min_trimmed_frames`` are dropped, and so are crops wider than
    the 128-frame frame (possible only under ``"full_length"``).

    Parameters
    ----------
    n_frames (np.ndarray)
        ``(N,)`` full-length frame count of each segment.
    first_raw (np.ndarray)
        ``(N,)`` first squeak frame.
    last_raw (np.ndarray)
        ``(N,)`` last squeak frame.
    crop_window (str)
        ``"full_length"`` or ``"first_128_frames"``.
    context_frames (int)
        Frames added either side of the extent.
    min_trimmed_frames (int)
        Narrowest crop kept.

    Returns
    -------
    keep (np.ndarray)
        ``(N,)`` boolean, rows that pass every gate.
    first (np.ndarray)
        ``(N,)`` int64 first crop frame (meaningful where ``keep``).
    last (np.ndarray)
        ``(N,)`` int64 last crop frame, inclusive.
    gates (list[tuple[str, int]])
        ``(gate, rows left after it)``, starting with the candidates.
    """

    if crop_window not in ("full_length", "first_128_frames"):
        error_message = f"crop_window must be 'full_length' or 'first_128_frames', got {crop_window!r}."
        raise ValueError(error_message)
    n_frames = np.asarray(n_frames, dtype=np.int64)
    keep = np.ones(n_frames.size, dtype=bool)
    gates = [("squeak candidates", int(keep.sum()))]
    if crop_window == "first_128_frames":
        n_valid = np.minimum(n_frames, CROP_FRAME_WIDTH)
        keep &= ~((n_frames > CROP_FRAME_WIDTH) & (np.asarray(last_raw) >= CROP_FRAME_WIDTH - 1))
        gates.append(("drop right-censored", int(keep.sum())))
    else:
        n_valid = n_frames
    first = np.maximum(np.asarray(first_raw, dtype=np.int64) - context_frames, 0)
    last = np.minimum(np.asarray(last_raw, dtype=np.int64) + context_frames, n_valid - 1)
    trimmed = last - first + 1
    keep &= trimmed >= min_trimmed_frames
    gates.append((f"drop trimmed < {min_trimmed_frames}", int(keep.sum())))
    if crop_window == "full_length":
        keep &= trimmed <= CROP_FRAME_WIDTH
        gates.append((f"drop trimmed > {CROP_FRAME_WIDTH}", int(keep.sum())))
    return keep, first, last, gates


def cap_rows_per_session_bin(
    session_ids: np.ndarray,
    bins: np.ndarray,
    cap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Description
    -----------
    Keeps at most ``cap`` rows of any one session in any one duration stratum.
    The (session, stratum) groups are visited in order of first appearance, and a
    group larger than ``cap`` keeps ``rng.choice(its positions, cap,
    replace=False)`` (pandas ``groupby(sort=False)`` order, as
    ``build_bbv_dataset.py`` visited them).

    Parameters
    ----------
    session_ids (np.ndarray)
        ``(N,)`` session of each row.
    bins (np.ndarray)
        ``(N,)`` duration stratum of each row.
    cap (int)
        Largest number of rows kept per (session, stratum); must be positive.
    rng (np.random.Generator)
        Generator the draw consumes.

    Returns
    -------
    keep (np.ndarray)
        ``(N,)`` boolean.
    """

    keep = np.zeros(len(session_ids), dtype=bool)
    groups: dict[tuple[str, int], list[int]] = {}
    for position, key in enumerate(zip(np.asarray(session_ids).tolist(), np.asarray(bins).tolist(), strict=True)):
        groups.setdefault(key, []).append(position)
    for positions in groups.values():
        group_positions = np.array(positions, dtype=np.int64)
        if group_positions.size > cap:
            group_positions = rng.choice(group_positions, size=cap, replace=False)
        keep[group_positions] = True
    return keep


def stratified_duration_draw(
    bins: np.ndarray,
    n_total: int,
    n_bins: int,
    draw_mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Description
    -----------
    Draws ``n_total`` row positions (all of them when fewer are available).
    ``"natural"``: ``rng.choice`` over the positions listed stratum by stratum.
    ``"uniform"``: per-stratum quotas by repeated even sharing (each pass gives
    every stratum with rows left ``max(remaining // n_live, 1)`` more, capped by
    what it has, until the total is met or every stratum is used up), then
    ``rng.choice`` within each stratum in stratum order. This is
    ``build_bbv_dataset.py``'s ``stratified_draw`` verbatim, so a seed draws the
    same rows.

    Parameters
    ----------
    bins (np.ndarray)
        ``(N,)`` duration stratum of each row, in ``0 .. n_bins - 1``.
    n_total (int)
        Rows to draw.
    n_bins (int)
        Number of strata.
    draw_mode (str)
        ``"natural"`` or ``"uniform"``.
    rng (np.random.Generator)
        Generator the draw consumes.

    Returns
    -------
    positions (np.ndarray)
        Drawn row positions (unsorted, as drawn).
    """

    by_bin = [np.flatnonzero(np.asarray(bins) == stratum) for stratum in range(n_bins)]
    if draw_mode == "natural":
        pool = np.concatenate([members for members in by_bin if members.size])
        return rng.choice(pool, size=min(n_total, pool.size), replace=False)
    if draw_mode != "uniform":
        error_message = f"draw_mode must be 'natural' or 'uniform', got {draw_mode!r}."
        raise ValueError(error_message)

    quotas = np.zeros(n_bins, dtype=np.int64)
    remaining, live = n_total, [stratum for stratum in range(n_bins) if by_bin[stratum].size]
    while remaining > 0 and live:
        share = max(remaining // len(live), 1)
        progressed = False
        for stratum in list(live):
            take = min(share, by_bin[stratum].size - quotas[stratum], remaining)
            if take <= 0:
                live.remove(stratum)
                continue
            quotas[stratum] += take
            remaining -= take
            progressed = True
            if quotas[stratum] == by_bin[stratum].size:
                live.remove(stratum)
            if remaining == 0:
                break
        if not progressed:
            break
    return np.concatenate([rng.choice(by_bin[stratum], size=int(quotas[stratum]), replace=False)
                           for stratum in range(n_bins) if quotas[stratum]])


def normalize_crop(crop: np.ndarray, crop_normalization: str) -> np.ndarray:
    """
    Description
    -----------
    Normalizes one float32 absolute-dB crop: ``"per_crop"`` min-max,
    ``(x - min) / (max - min + 1e-6)`` (the phase 3 sets), or ``"absolute"``,
    ``(clip(x, -100, 50) + 25) / 75`` (the squeak classifier's fixed transform).

    Parameters
    ----------
    crop (np.ndarray)
        ``(F, width)`` float32 crop in dB.
    crop_normalization (str)
        ``"per_crop"`` or ``"absolute"``.

    Returns
    -------
    normalized (np.ndarray)
        ``(F, width)`` float32 crop.
    """

    if crop_normalization == "per_crop":
        low, high = float(crop.min()), float(crop.max())
        return (crop - low) / (high - low + CROP_MINMAX_EPSILON)
    if crop_normalization == "absolute":
        return (np.clip(crop, -100.0, 50.0) + 25.0) / 75.0
    error_message = f"crop_normalization must be 'per_crop' or 'absolute', got {crop_normalization!r}."
    raise ValueError(error_message)


class QLVMSqueakTrainingSetBuilder:
    """
    Description
    -----------
    Builds a QLVM ``.npz`` training set of squeak crops from a list of session
    root directories (see the module docstring).
    """

    def __init__(
        self,
        root_directories: list[str] | None = None,
        output_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the QLVMSqueakTrainingSetBuilder.

        Parameters
        ----------
        root_directories (list[str])
            Session root directories (each with ``audio/hpss`` wavs, a
            ``*_usv_summary.csv`` holding the squeak columns and a metadata YAML).
        output_directory (str)
            Directory to write the ``.npz`` outputs + metadata.
        input_parameter_dict (dict)
            Processing settings; the ``build_qlvm_squeak_training_set`` block
            supplies every parameter.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.root_directories = root_directories if root_directories is not None else []
        self.output_directory = output_directory
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def collect_candidates(self) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, str]]:
        """
        Description
        -----------
        Reads the squeak candidates of every session
        (:func:`squeak_candidates_from_summary`), sessions in sorted id order.

        Parameters
        ----------

        Returns
        -------
        candidates (dict[str, np.ndarray])
            Row-aligned arrays over all sessions: ``session_id`` plus the columns of
            :func:`squeak_candidates_from_summary`.
        session_roots (dict[str, str])
            Session id -> root directory.
        session_type_by_key (dict[str, str])
            Session id -> type (:func:`session_type_from_metadata`).
        """

        cfg = self.input_parameter_dict['build_qlvm_squeak_training_set']
        session_roots = {pathlib.Path(root).name: root for root in self.root_directories}
        columns: dict[str, list[np.ndarray]] = {key: [] for key in ("session_id", "row", "start_s", "stop_s", "n_frames", "first_raw", "last_raw", "probability")}
        session_type_by_key = {}
        for session_id in sorted(session_roots):
            root = session_roots[session_id]
            session_type_by_key[session_id] = session_type_from_metadata(root, self.message_output)
            usv_summary_path = first_match_or_raise(
                root=pathlib.Path(root) / "audio", pattern="*_usv_summary.csv", recursive=True, label="USV summary CSV",
            )
            usv_summary = pls.read_csv(source=str(usv_summary_path), schema_overrides={"usv_id": pls.String})
            session_candidates = squeak_candidates_from_summary(usv_summary, cfg['exclude_noise'])
            for key, values in session_candidates.items():
                columns[key].append(values)
            columns["session_id"].append(np.full(session_candidates["row"].size, session_id))
        candidates = {key: np.concatenate(values) if values else np.empty(0) for key, values in columns.items()}
        return candidates, session_roots, session_type_by_key

    def build(self) -> None:
        """
        Description
        -----------
        Collects the candidates from the session summaries
        (:meth:`collect_candidates`) and builds the set from them
        (:meth:`build_from_candidates`).

        Parameters
        ----------

        Returns
        -------
        None
        """

        self.message_output(
            f"QLVM squeak training-set build started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)
        candidates, session_roots, session_type_by_key = self.collect_candidates()
        self.build_from_candidates(candidates, session_roots, session_type_by_key, "usv_summary rows with squeak true (pure squeak and both)")
        self.message_output(
            f"QLVM squeak training-set build ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )

    def build_from_candidates(
        self,
        candidates: dict[str, np.ndarray],
        session_roots: dict[str, str],
        session_type_by_key: dict[str, str],
        candidate_source: str,
    ) -> None:
        """
        Description
        -----------
        Builds the set from squeak candidates: crop gates
        (:func:`squeak_crop_gates`), duration strata, the per-session cap and the
        draw (or every row under ``full_dataset``), the spectrogram rebuild,
        crop, normalization and resize, the session split, and the ``.npz``
        writes. :meth:`build` feeds it the summary candidates; a caller can feed it
        candidates from another source in the same layout (e.g. the phase 3 sets'
        own squeak index, to rebuild them), in the row order to draw them in.

        Parameters
        ----------
        candidates (dict[str, np.ndarray])
            Row-aligned ``session_id``, ``row``, ``start_s`` / ``stop_s`` (the
            audio window the spectrogram is rebuilt over, s; the segment itself
            for candidates of the reference squeak index), ``n_frames`` (frames of
            that window), ``first_raw`` / ``last_raw`` (squeak frames of the
            window) and ``probability``.
        session_roots (dict[str, str])
            Session id -> root directory, for every session in ``candidates``.
        session_type_by_key (dict[str, str])
            Session id -> type, for every session in ``candidates``.
        candidate_source (str)
            Where the candidates came from, recorded in ``metadata.npz``.

        Returns
        -------
        None
        """

        cfg = self.input_parameter_dict['build_qlvm_squeak_training_set']
        draw_mode = cfg['draw_mode']
        n_total = int(cfg['n_total'])
        cap = int(cfg['per_session_bin_cap'])
        edges = [int(edge) for edge in cfg['duration_bin_edges']]
        context_frames = int(cfg['context_frames'])
        min_trimmed_frames = int(cfg['min_trimmed_frames'])
        crop_window = cfg['crop_window']
        crop_normalization = cfg['crop_normalization']
        exclude_metadata_audio_channels = cfg['exclude_metadata_audio_channels']
        validation_split = cfg['validation_split']
        random_state = cfg['random_state']
        full_dataset = cfg['full_dataset']
        target_shape = tuple(int(v) for v in cfg['target_shape'])
        if not 0.0 < validation_split < 1.0:
            error_message = f"validation_split must be in the open interval (0, 1), got {validation_split}."
            raise ValueError(error_message)
        n_bins = len(edges) + 1
        output_dir = pathlib.Path(self.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(random_state)

        keep, first, last, gates = squeak_crop_gates(
            candidates["n_frames"], candidates["first_raw"], candidates["last_raw"], crop_window, context_frames, min_trimmed_frames,
        )
        pool = {key: values[keep] for key, values in candidates.items()}
        pool["first"], pool["last"] = first[keep], last[keep]
        pool["bin"] = np.digitize(pool["last"] - pool["first"] + 1, edges).astype(np.int64)
        if cap > 0 and not full_dataset:
            capped = cap_rows_per_session_bin(pool["session_id"], pool["bin"], cap, rng)
            pool = {key: values[capped] for key, values in pool.items()}
            gates.append((f"per-session/stratum cap {cap}", int(pool["row"].size)))
        for label, count in gates:
            self.message_output(f"    {label:<32s} {count:>7,d}")
        available = np.bincount(pool["bin"], minlength=n_bins)
        if full_dataset:
            chosen = np.arange(pool["row"].size)
        else:
            chosen = np.sort(stratified_duration_draw(pool["bin"], n_total, n_bins, draw_mode, rng))
        drawn = {key: values[chosen] for key, values in pool.items()}
        realized = np.bincount(drawn["bin"], minlength=n_bins)
        self.message_output(
            f"Drew {drawn['row'].size:,} squeaks; per duration stratum available {available.tolist()}, drawn {realized.tolist()}."
        )

        n_rows = drawn["row"].size
        specs = np.zeros((n_rows, SQUEAK_SPEC_PARAMS["num_freq_bins"], CROP_FRAME_WIDTH), dtype=np.float32)
        widths = np.empty(n_rows, dtype=np.int64)
        for session_id in dict.fromkeys(drawn["session_id"].tolist()):
            positions = np.flatnonzero(drawn["session_id"] == session_id)
            spectrograms = squeak_window_spectrograms(
                session_root=pathlib.Path(session_roots[session_id]),
                window_start_s=drawn["start_s"][positions],
                window_stop_s=drawn["stop_s"][positions],
                exclude_metadata_audio_channels=exclude_metadata_audio_channels,
                message_output=self.message_output,
            )
            for position, spectrogram in zip(positions, spectrograms, strict=True):
                if spectrogram is None or spectrogram.shape[1] != drawn["n_frames"][position]:
                    error_message = (
                        f"{session_id} row {drawn['row'][position]}: the rebuilt spectrogram has "
                        f"{None if spectrogram is None else spectrogram.shape[1]} frames, the candidate {drawn['n_frames'][position]}."
                    )
                    raise ValueError(error_message)
                crop = spectrogram[:, int(drawn["first"][position]):int(drawn["last"][position]) + 1].astype(np.float32)
                crop = normalize_crop(crop, crop_normalization)
                specs[position, :, :crop.shape[1]] = crop
                widths[position] = crop.shape[1]
        self.message_output(f"Cropped and normalized {n_rows:,} squeaks; widths {int(widths.min())}-{int(widths.max())} frames.")

        spec_ids = np.array([f"{session_id}_{int(row)}" for session_id, row in zip(drawn["session_id"], drawn["row"], strict=True)])
        session_types = np.array([session_type_by_key[session_id] for session_id in drawn["session_id"]])
        selected_counts = {session_id: int(count) for session_id, count in zip(*np.unique(drawn["session_id"], return_counts=True), strict=True)}
        type_by_key = {session_id: session_type_by_key[session_id] for session_id in selected_counts}
        train_sessions, val_sessions = split_sessions_by_type(
            list(selected_counts), type_by_key, selected_counts, validation_split, random_state
        )
        is_val = np.isin(drawn["session_id"], val_sessions)
        splits = [("train_data.npz", np.flatnonzero(~is_val)), ("val_data.npz", np.flatnonzero(is_val))]
        if full_dataset:
            splits.append(("full_data.npz", np.arange(n_rows)))
        written: dict[str, int] = {}
        for filename, rows in splits:
            resized = stretch_specs(specs[rows], widths[rows], target_shape, False).astype(np.float32)
            np.savez(
                output_dir / filename,
                spectrograms=resized,
                masks=np.zeros_like(resized, dtype=np.float32),
                masks_len=np.zeros(rows.size, dtype=np.int64),
                durations=widths[rows].astype(np.int64),
                spec_id=spec_ids[rows],
                session_id=drawn["session_id"][rows].astype(str),
                session_type=session_types[rows],
                duration_bin=drawn["bin"][rows].astype(np.int64),
                squeak_probability=drawn["probability"][rows].astype(np.float32),
                crop_first=drawn["first"][rows].astype(np.int64),
                crop_last=drawn["last"][rows].astype(np.int64),
                n_frames_original=drawn["n_frames"][rows].astype(np.int64),
                apply_mask=np.array(False),
            )
            written[filename] = int(rows.size)
            self.message_output(f"  Wrote {rows.size:,} -> {output_dir / filename}.")

        type_counts = {session_type: int(count) for session_type, count in zip(*np.unique(session_types, return_counts=True), strict=True)}
        np.savez(
            output_dir / "metadata.npz",
            root_directories=np.array([session_roots[session_id] for session_id in sorted(session_roots)]),
            candidate_source=candidate_source,
            length_threshold=np.nan,
            validation_split=validation_split,
            random_state=random_state,
            full_dataset=full_dataset,
            target_shape=np.array(target_shape),
            time_stretch=False,
            masking_type="none",
            apply_mask=False,
            require_mask=False,
            split_by_session=True,
            draw=draw_mode,
            duration_bin_edges=np.array(edges),
            per_session_bin_cap=cap,
            n_total_target=n_total,
            bin_available=available,
            bin_realized=realized,
            context_frames=context_frames,
            min_trimmed_frames=min_trimmed_frames,
            crop_window=crop_window,
            normalization=crop_normalization,
            exclude_noise=cfg['exclude_noise'],
            exclude_metadata_audio_channels=exclude_metadata_audio_channels,
            curation_gates=json.dumps(gates),
            session_type_counts=json.dumps(type_counts),
            session_type_by_key=json.dumps(type_by_key),
            split_sessions=json.dumps({"train": train_sessions, "validation": val_sessions}),
            n_train=written["train_data.npz"],
            n_val=written["val_data.npz"],
            n_full=written["full_data.npz"] if full_dataset else 0,
        )


@click.command(name="build-qlvm-squeak-training-set")
@click.option('--root-directories', type=str, required=True, help='Comma-separated string of session root directory paths.')
@click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory to write the .npz training set.')
@click.option('--draw-mode', 'draw_mode', type=click.Choice(['natural', 'uniform']), default=None, required=False, help='Draw uniformly over squeaks ("natural") or equally across duration strata ("uniform").')
@click.option('--n-total', 'n_total', type=int, default=None, required=False, help='Squeaks to draw.')
@click.option('--per-session-bin-cap', 'per_session_bin_cap', type=int, default=None, required=False, help='Most squeaks any one session may give any one duration stratum before the draw (0: no cap, the "lumped" sets).')
@click.option('--duration-bin-edges', 'duration_bin_edges', type=str, default=None, required=False, help='Comma-separated crop-width edges (frames) of the duration strata, e.g. 26,40,62.')
@click.option('--context-frames', 'context_frames', type=int, default=None, required=False, help='Frames added either side of the squeak extent.')
@click.option('--min-trimmed-frames', 'min_trimmed_frames', type=int, default=None, required=False, help='Narrowest crop (frames) kept.')
@click.option('--crop-window', 'crop_window', type=click.Choice(['full_length', 'first_128_frames']), default=None, required=False, help='Crop from the full-length segment ("full_length") or from its first 128 frames, dropping right-censored squeaks ("first_128_frames", the phase 3 sets).')
@click.option('--crop-normalization', 'crop_normalization', type=click.Choice(['per_crop', 'absolute']), default=None, required=False, help='Min-max each crop ("per_crop") or apply the fixed dB transform ("absolute").')
@click.option('--exclude-noise/--no-exclude-noise', 'exclude_noise', default=None, required=False, help='Leave out squeak rows the USV summary flags as noise.')
@click.option('--exclude-metadata-audio-channels/--no-exclude-metadata-audio-channels', 'exclude_metadata_audio_channels', default=None, required=False, help='Drop channels the session metadata marks as excluded from the spectrogram average.')
@click.option('--validation-split', 'validation_split', type=float, default=None, required=False, help='Fraction of each session type\'s squeaks held out (as whole sessions) for validation.')
@click.option('--random-state', 'random_state', type=int, default=None, required=False, help='Seed of the cap, the draw and the session split.')
@click.option('--full-dataset/--no-full-dataset', 'full_dataset', default=None, required=False, help='Take every squeak that passes the gates (no cap, no draw) and also write full_data.npz.')
@click.option('--target-shape', 'target_shape', nargs=2, type=int, default=None, required=False, help='Output spectrogram (freq, time) shape as two ints, e.g. --target-shape 128 128.')
@click.pass_context
def build_qlvm_squeak_training_set_cli(ctx, root_directories, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to build a QLVM squeak training set (``.npz``) from a
    list of session root directories.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]
    if 'duration_bin_edges' in provided_params:
        ctx.params['duration_bin_edges'] = parse_int_list(ctx.params['duration_bin_edges'])

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings',
        parameters_lists=['target_shape'],
        block='build_qlvm_squeak_training_set',
    )

    root_dirs = [p.strip() for p in root_directories.split(",") if p.strip()]

    QLVMSqueakTrainingSetBuilder(
        root_directories=root_dirs,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).build()
