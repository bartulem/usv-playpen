"""
@author: bartulem
Assemble per-session USV spectrogram H5 files into a single curated training set
(``.npz``) for the QLVM vocalization model.

Reads the per-session ``*_spectrograms.h5`` files produced by
:mod:`generate_spectrograms`, drops over-long spectrograms, optionally
subsamples, splits into train/val (or keeps one full set), resizes/time-stretches
every spectrogram to ``target_shape``, and writes ``train_data.npz`` /
``val_data.npz`` (or ``full_data.npz``) plus a ``metadata.npz`` sidecar.

This is the in-house, torch-free (``.npz``) port of the external
``preprocess_monolithic.py`` + ``data_utils`` resize/split logic; the
:mod:`train_qlvm` trainer reads the ``.npz`` directly (no torch on this side of
the dataset boundary).

Masking (``masking_type``):

* ``"sam"`` (default) -- when a session's spectrogram H5 carries a
  ``mask/<session>`` group (from :mod:`generate_masks`), each kept spectrogram's
  2D region is the ``np.any`` union of its instance segmentations, resized with
  the spectrogram and binarized, and the spectrogram is **masked**
  (``spec *= mask``, background zeroed) -- exactly as the external
  ``prepare_masked_datasets`` does. A kept USV with no detected mask (or a
  session with no mask group) falls back to an all-ones mask, so its spectrogram
  is kept unchanged rather than zeroed. ``masks_len`` is the per-spectrogram
  instance count.
* ``"none"`` -- no masking; raw spectrograms with all-zero ``masks``/``masks_len``
  placeholders (kept so the key set is uniform).

Each output ``.npz`` is row-aligned on dim 0 = N samples and holds:
``spectrograms`` (N, F, T) float32 (mask-applied under ``"sam"``), ``masks``
(N, F, T) float32 (binarized region; all-zero under ``"none"``), ``masks_len``
(N,) int64, ``durations`` (N,) int64, and ``spec_id`` (N,) str.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import numpy as np
from click.core import ParameterSource
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import zoom
from sklearn.model_selection import train_test_split

from ..cli_utils import modify_settings_json_for_cli
from ..time_utils import is_gui_context, smart_wait


def compute_selected_indices(
    durations_by_key: dict[str, np.ndarray],
    length_threshold: float,
    dataset_size_constraint: float | None,
    random_state: int,
) -> dict[str, np.ndarray]:
    """
    Description
    -----------
    For each session key, returns the sorted indices of real spectrograms
    (``0 < duration < length_threshold`` — ``duration == 0`` rows are the
    all-zero placeholders for invalid USVs and are excluded), optionally
    subsampled so the total kept count
    approaches ``dataset_size_constraint`` (an absolute count if ``> 1``, a
    proportion if in ``(0, 1]``, all data if ``None``). Lets later phases read
    only the needed rows from disk.

    Parameters
    ----------
    durations_by_key (dict[str, np.ndarray])
        Per-session native spectrogram durations.
    length_threshold (float)
        Drop spectrograms whose duration is ``>= threshold``.
    dataset_size_constraint (float | None)
        Cap on total kept samples (see above).
    random_state (int)
        Seed for reproducible subsampling.

    Returns
    -------
    selected_indices_by_key (dict[str, np.ndarray])
        Per-session sorted index arrays.
    """

    rng = np.random.default_rng(random_state)
    total_filtered = sum(int(np.sum((d > 0) & (d < length_threshold))) for d in durations_by_key.values())

    samples_per_session: int | None = None
    if dataset_size_constraint is not None and durations_by_key:
        if dataset_size_constraint > 1:
            target_total = int(dataset_size_constraint)
        else:
            target_total = int(total_filtered * dataset_size_constraint)
        samples_per_session = target_total // len(durations_by_key)

    selected: dict[str, np.ndarray] = {}
    for key, durations in durations_by_key.items():
        valid_indices = np.where((durations > 0) & (durations < length_threshold))[0]
        if samples_per_session is not None and samples_per_session < len(valid_indices):
            sampled = rng.choice(len(valid_indices), size=samples_per_session, replace=False)
            valid_indices = valid_indices[sampled]
        selected[key] = np.sort(valid_indices)
    return selected


def _apply_time_stretching(spec: np.ndarray, duration: int, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Description
    -----------
    Time-stretches a spectrogram's signal window ``[0, duration]`` to fill the
    full target time axis via linear interpolation (the QLVM time-warp option).

    Parameters
    ----------
    spec (np.ndarray)
        A ``(F, T)`` spectrogram.
    duration (int)
        Native signal length in time bins.
    target_shape (tuple[int, int])
        Output ``(freq, time)`` shape.

    Returns
    -------
    spec_resized (np.ndarray)
        The stretched ``target_shape`` spectrogram.
    """

    freq_orig = np.arange(spec.shape[0])
    time_orig = np.arange(duration)
    spec_signal = spec[:, :duration]
    interpolator = RegularGridInterpolator(
        (freq_orig, time_orig), spec_signal, method="linear", bounds_error=False, fill_value=0.0
    )
    freq_new = np.linspace(0, spec.shape[0] - 1, target_shape[0])
    time_new = np.linspace(0, duration - 1, target_shape[1])
    freq_grid, time_grid = np.meshgrid(freq_new, time_new, indexing="ij")
    points = np.stack([freq_grid.ravel(), time_grid.ravel()], axis=-1)
    return interpolator(points).reshape(target_shape)


def _apply_simple_resize(spec: np.ndarray, duration: int, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Description
    -----------
    Zoom-resizes a spectrogram to ``target_shape`` and center-pads the signal
    window (the QLVM non-warp option).

    Parameters
    ----------
    spec (np.ndarray)
        A ``(F, T)`` spectrogram.
    duration (int)
        Native signal length in time bins.
    target_shape (tuple[int, int])
        Output ``(freq, time)`` shape.

    Returns
    -------
    spec_centered (np.ndarray)
        The resized, centered ``target_shape`` spectrogram.
    """

    zoom_factors = (target_shape[0] / spec.shape[0], target_shape[1] / spec.shape[1])
    spec_interp = zoom(spec, zoom_factors, order=1)
    signal_length = min(duration, target_shape[1])
    signal_portion = spec_interp[:, :signal_length]
    left_pad = (target_shape[1] - signal_length) // 2
    right_pad = target_shape[1] - signal_length - left_pad
    return np.pad(signal_portion, ((0, 0), (left_pad, right_pad)), mode="constant", constant_values=0.0)


def stretch_specs(
    spectrograms: np.ndarray,
    durations: np.ndarray,
    target_shape: tuple[int, int],
    time_stretch: bool,
) -> np.ndarray:
    """
    Description
    -----------
    Resizes every spectrogram to ``target_shape``, either time-stretching the
    signal window to fill the frame or center-resizing it. Zeros the padded tail
    beyond each spectrogram's native ``duration`` before resizing.

    Parameters
    ----------
    spectrograms (np.ndarray)
        ``(N, F, T)`` spectrograms.
    durations (np.ndarray)
        Native time-bin counts, shape ``(N,)``.
    target_shape (tuple[int, int])
        Output ``(freq, time)`` shape.
    time_stretch (bool)
        Time-warp if True, else center-resize.

    Returns
    -------
    resized (np.ndarray)
        ``(N, *target_shape)`` float32 array.
    """

    n = spectrograms.shape[0]
    resized = np.empty((n, *target_shape), dtype=np.float32)
    for idx in range(n):
        spec_work = spectrograms[idx].copy()
        duration = int(min(max(int(durations[idx]), 1), spec_work.shape[1]))
        spec_work[:, duration:] = 0
        if time_stretch:
            resized[idx] = _apply_time_stretching(spec_work, duration, target_shape)
        else:
            resized[idx] = _apply_simple_resize(spec_work, duration, target_shape)
    return resized


def build_session_masks(
    h5_file: h5py.File,
    session_id: str,
    selected_indices: np.ndarray,
    n_freq: int,
    n_time: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Builds per-spectrogram 2D region masks and instance counts for one session's
    selected rows from its ``mask/<session>`` group (written by
    :mod:`generate_masks`). Each selected row's mask is the boolean union
    (``np.any``) of every ``segmentations`` row whose ``spectrogram_index`` equals
    that row; a row with no mask instances (the detector found none) falls back to
    an all-ones mask so its spectrogram is later kept unchanged rather than zeroed.
    When the session has no ``mask/<session>`` group, every selected row gets an
    all-ones mask and a zero instance count.

    Parameters
    ----------
    h5_file (h5py.File)
        Open per-session spectrogram H5.
    session_id (str)
        Session id naming the ``spectrogram/<session>`` / ``mask/<session>`` groups.
    selected_indices (np.ndarray)
        Sorted usv_summary row indices kept for this session.
    n_freq (int)
        Frequency-bin count ``F`` of each mask (mask height).
    n_time (int)
        Time-bin count ``T`` of each mask (mask width).

    Returns
    -------
    masks (np.ndarray)
        A ``(len(selected_indices), F, T)`` float32 array (1.0 inside the region,
        0.0 outside; all-ones for rows that fall back).
    masks_len (np.ndarray)
        A ``(len(selected_indices),)`` int64 array of per-row instance counts.
    """

    n_selected = selected_indices.shape[0]
    masks = np.ones((n_selected, n_freq, n_time), dtype=np.float32)
    masks_len = np.zeros(n_selected, dtype=np.int64)

    mask_group_key = f"mask/{session_id}"
    if mask_group_key not in h5_file:
        return masks, masks_len

    mask_group = h5_file[mask_group_key]
    segmentations = mask_group["segmentations"][:]
    spectrogram_index = mask_group["spectrogram_index"][:]
    for position, summary_row in enumerate(selected_indices):
        mask_rows = np.flatnonzero(spectrogram_index == int(summary_row))
        if mask_rows.size > 0:
            masks[position] = np.any(segmentations[mask_rows], axis=0).astype(np.float32)
            masks_len[position] = int(mask_rows.size)
    return masks, masks_len


class QLVMTrainingSetBuilder:
    """
    Description
    -----------
    Builds a curated ``.npz`` training set for the QLVM model from a list of
    per-session spectrogram H5 files.
    """

    def __init__(
        self,
        spectrogram_h5_paths: list[str] | None = None,
        output_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the QLVMTrainingSetBuilder.

        Parameters
        ----------
        spectrogram_h5_paths (list[str])
            Per-session ``*_spectrograms.h5`` files to combine.
        output_directory (str)
            Directory to write the ``.npz`` outputs + metadata.
        input_parameter_dict (dict)
            Processing settings; the ``build_qlvm_training_set`` block supplies
            the filtering / split / resize parameters.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.spectrogram_h5_paths = spectrogram_h5_paths if spectrogram_h5_paths is not None else []
        self.output_directory = output_directory
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def build(self) -> None:
        """
        Description
        -----------
        Runs the full pipeline: load durations → select indices (length filter +
        optional subsample) → load selected specs → combine → train/val split (or
        full) → resize/time-stretch → write ``.npz`` outputs + ``metadata.npz``.

        Parameters
        ----------

        Returns
        -------
        ``train_data.npz`` + ``val_data.npz`` (or ``full_data.npz``) + ``metadata.npz``.
        """

        self.message_output(
            f"QLVM training-set build started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['build_qlvm_training_set']
        length_threshold = cfg['length_threshold']
        dataset_size_constraint = cfg['dataset_size_constraint']
        validation_split = cfg['validation_split']
        random_state = cfg['random_state']
        full_dataset = cfg['full_dataset']
        target_shape = tuple(int(v) for v in cfg['target_shape'])
        time_stretch = cfg['time_stretch']
        masking_type = cfg['masking_type']

        output_dir = pathlib.Path(self.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: cheap per-session durations, keyed by the session id that
        # names the ``spectrogram/<session>`` group inside each file.
        durations_by_key: dict[str, np.ndarray] = {}
        session_by_path: dict[str, str] = {}
        for h5_path in self.spectrogram_h5_paths:
            with h5py.File(h5_path, "r") as h5_file:
                session_id = next(iter(h5_file["spectrogram"].keys()))
                session_by_path[h5_path] = session_id
                durations_by_key[session_id] = h5_file[f"spectrogram/{session_id}"]["durations"][:]

        # Phase 2: length filter + optional subsample.
        selected = compute_selected_indices(
            durations_by_key, length_threshold,
            None if full_dataset else dataset_size_constraint, random_state,
        )

        # Phase 3: load selected spectrograms/durations from each session's
        # ``spectrogram/<session>`` group, concatenate, and build the
        # globally-unique spec_id. Spectrogram rows are 1:1 with usv_summary.csv,
        # so the selected row index IS the usv index; the cross-session spec_id is
        # f"{session_id}_{row_index}" (the per-sample identifier the trainer carries).
        # Under ``masking_type == "sam"`` the per-row SAM region masks + instance
        # counts are read from the ``mask/<session>`` group at the same time.
        specs_list: list[np.ndarray] = []
        durations_list: list[np.ndarray] = []
        spec_id_list: list[np.ndarray] = []
        masks_list: list[np.ndarray] = []
        masks_len_list: list[np.ndarray] = []
        for h5_path in self.spectrogram_h5_paths:
            session_id = session_by_path[h5_path]
            idx = selected[session_id]
            if idx.size == 0:
                continue
            with h5py.File(h5_path, "r") as h5_file:
                session_group = h5_file[f"spectrogram/{session_id}"]
                specs_list.append(session_group["spectrograms"][idx])
                durations_list.append(session_group["durations"][idx])
                n_freq, n_time = session_group["spectrograms"].shape[1:]
                if masking_type == "sam":
                    session_masks, session_masks_len = build_session_masks(
                        h5_file, session_id, idx, n_freq, n_time
                    )
                    masks_list.append(session_masks)
                    masks_len_list.append(session_masks_len)
            spec_id_list.append(np.array([f"{session_id}_{int(i)}" for i in idx]))

        if not specs_list:
            self.message_output("No spectrograms survived filtering; nothing written.")
            return

        all_specs = np.concatenate(specs_list)
        all_durations = np.concatenate(durations_list)
        all_spec_ids = np.concatenate(spec_id_list)
        if masking_type == "sam":
            all_masks = np.concatenate(masks_list)
            all_masks_len = np.concatenate(masks_len_list)
        else:
            all_masks = np.zeros_like(all_specs, dtype=np.float32)
            all_masks_len = np.zeros(all_specs.shape[0], dtype=np.int64)
        n_with_masks = int(np.count_nonzero(all_masks_len > 0))
        self.message_output(
            f"masking_type='{masking_type}': {n_with_masks}/{all_specs.shape[0]} kept spectrograms have "
            f"a detected SAM mask (the rest keep an all-ones mask)."
        )

        # Phases 5-6: split (or full) + resize + (under "sam") binarize-and-apply + save.
        if full_dataset:
            splits = {"full_data.npz": (all_specs, all_masks, all_masks_len, all_durations, all_spec_ids)}
        else:
            (
                train_specs, val_specs, train_masks, val_masks, train_ml, val_ml,
                train_dur, val_dur, train_ids, val_ids,
            ) = train_test_split(
                all_specs, all_masks, all_masks_len, all_durations, all_spec_ids,
                test_size=validation_split, random_state=random_state,
            )
            splits = {
                "train_data.npz": (train_specs, train_masks, train_ml, train_dur, train_ids),
                "val_data.npz": (val_specs, val_masks, val_ml, val_dur, val_ids),
            }

        written: dict[str, int] = {}
        for filename, (split_specs, split_masks, split_ml, split_dur, split_ids) in splits.items():
            resized = stretch_specs(split_specs, split_dur, target_shape, time_stretch)
            if masking_type == "sam":
                # Resize masks with the SAME per-row transform, binarize at 0.5, then
                # mask the spectrogram (background zeroed) -- mirrors prepare_masked_datasets.
                resized_masks = (stretch_specs(split_masks, split_dur, target_shape, time_stretch) >= 0.5).astype(np.float32)
                out_specs = (resized * resized_masks).astype(np.float32)
                out_masks = resized_masks
            else:
                out_specs = resized.astype(np.float32)
                out_masks = np.zeros_like(resized, dtype=np.float32)
            np.savez(
                output_dir / filename,
                spectrograms=out_specs,
                masks=out_masks,
                masks_len=split_ml.astype(np.int64),
                durations=split_dur.astype(np.int64),
                spec_id=split_ids,
            )
            written[filename] = int(resized.shape[0])
            self.message_output(f"  Wrote {written[filename]} samples -> {output_dir / filename}.")

        np.savez(
            output_dir / "metadata.npz",
            spectrogram_h5_paths=np.array(self.spectrogram_h5_paths),
            length_threshold=length_threshold,
            dataset_size_constraint=np.nan if dataset_size_constraint is None else dataset_size_constraint,
            validation_split=validation_split,
            random_state=random_state,
            full_dataset=full_dataset,
            target_shape=np.array(target_shape),
            time_stretch=time_stretch,
            masking_type=masking_type,
            **{f"n_{name.split('_')[0]}": count for name, count in written.items()},
        )

        self.message_output(
            f"QLVM training-set build ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="build-qlvm-training-set")
@click.option('--spectrogram-h5-paths', type=str, required=True, help='Comma-separated per-session *_spectrograms.h5 paths.')
@click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory to write the .npz training set.')
@click.option('--length-threshold', 'length_threshold', type=float, default=None, required=False, help='Drop spectrograms with duration >= threshold (time bins).')
@click.option('--validation-split', 'validation_split', type=float, default=None, required=False, help='Fraction held out for validation.')
@click.option('--full-dataset/--no-full-dataset', 'full_dataset', default=None, required=False, help='Write a single full_data.npz (no train/val split).')
@click.option('--time-stretch/--no-time-stretch', 'time_stretch', default=None, required=False, help='Time-warp the signal window instead of center-resizing.')
@click.option('--masking-type', 'masking_type', type=click.Choice(['sam', 'none']), default=None, required=False, help='Apply SAM mask regions from the mask/<session> groups ("sam") or keep raw spectrograms ("none").')
@click.pass_context
def build_qlvm_training_set_cli(ctx, spectrogram_h5_paths, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to assemble per-session spectrogram H5 files into a
    curated ``.npz`` QLVM training set.

    Parameters
    ----------

    Returns
    -------
    None
    """

    provided_params = [key for key in kwargs if ctx.get_parameter_source(key) == ParameterSource.COMMANDLINE]

    processing_settings_dict = modify_settings_json_for_cli(
        ctx=ctx,
        provided_params=provided_params,
        settings_dict='processing_settings',
    )

    h5_paths = [p.strip() for p in spectrogram_h5_paths.split(",") if p.strip()]

    QLVMTrainingSetBuilder(
        spectrogram_h5_paths=h5_paths,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).build()
