"""
@author: bartulem
Assemble per-session USV spectrogram H5 files into a single curated training set
(``.npz``) for the QLVM vocalization model.

Reads the per-session ``*_spectrograms.h5`` files produced by
:mod:`generate_spectrograms`, drops over-long spectrograms, optionally
subsamples, splits into train/val (or keeps one full set), resizes/time-stretches
every spectrogram to ``target_shape``, and writes ``train_data.npz`` /
``val_data.npz`` (or ``full_data.npz``) plus a ``metadata.npz`` sidecar.

This is the in-house, **mask-free**, torch-free (``.npz``) port of the external
``preprocess_monolithic.py`` + ``data_utils`` resize/split logic. Models are
retrained without masks, so no SAM/Otsu mask handling is performed; the saved
``masks``/``masks_len`` arrays are all-zero, present only so the external trainer
keeps a uniform key set. The external trainer reads ``.npz`` (no torch on this
side of the boundary).

Each output ``.npz`` is row-aligned on dim 0 = N samples and holds:
``spectrograms`` (N, F, T) float32, ``masks`` (N, F, T) float32 (all-zero),
``masks_len`` (N,) int64 (all-zero), ``durations`` (N,) int64, and ``spec_id``
(N,) str.
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
        # f"{session_id}_{row_index}" (the format the external trainer expects).
        specs_list: list[np.ndarray] = []
        durations_list: list[np.ndarray] = []
        spec_id_list: list[np.ndarray] = []
        for h5_path in self.spectrogram_h5_paths:
            session_id = session_by_path[h5_path]
            idx = selected[session_id]
            if idx.size == 0:
                continue
            with h5py.File(h5_path, "r") as h5_file:
                session_group = h5_file[f"spectrogram/{session_id}"]
                specs_list.append(session_group["spectrograms"][idx])
                durations_list.append(session_group["durations"][idx])
            spec_id_list.append(np.array([f"{session_id}_{int(i)}" for i in idx]))

        if not specs_list:
            self.message_output("No spectrograms survived filtering; nothing written.")
            return

        all_specs = np.concatenate(specs_list)
        all_durations = np.concatenate(durations_list)
        all_spec_ids = np.concatenate(spec_id_list)

        # Phases 5-6: split (or full) + resize + save.
        if full_dataset:
            splits = {"full_data.npz": (all_specs, all_durations, all_spec_ids)}
        else:
            train_specs, val_specs, train_dur, val_dur, train_ids, val_ids = train_test_split(
                all_specs, all_durations, all_spec_ids,
                test_size=validation_split, random_state=random_state,
            )
            splits = {
                "train_data.npz": (train_specs, train_dur, train_ids),
                "val_data.npz": (val_specs, val_dur, val_ids),
            }

        written: dict[str, int] = {}
        for filename, (split_specs, split_dur, split_ids) in splits.items():
            resized = stretch_specs(split_specs, split_dur, target_shape, time_stretch)
            np.savez(
                output_dir / filename,
                spectrograms=resized.astype(np.float32),
                masks=np.zeros_like(resized, dtype=np.float32),
                masks_len=np.zeros(resized.shape[0], dtype=np.int64),
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
            masking_type="none",
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
