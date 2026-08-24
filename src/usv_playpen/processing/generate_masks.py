"""
@author: bartulem
Generate per-USV instance masks (SAM2 box-prompt) for one session.

This is the usv-playpen-native orchestrator around the vendored mask-segmentation
kernels in ``processing/masks/`` (ported from the external ``spec_gen_full_pipeline``
box-prompt path). For one session it reads the per-USV spectrograms written by
``generate_spectrograms`` (``audio/spectrograms/<session>_spectrograms.h5``), runs a
YOLO box detector to propose one box per call candidate, prompts SAM2 once per box to
segment each call, and writes the resulting instance masks back INTO the SAME H5 under
a ``mask/<session>`` group that the spectrogram viewers and acoustic-feature steps read.

Default method is ``boxprompt`` with the ``yolo`` detector: the learned box detector is
faster, more accurate, and more stringent than the grid (``SAM2AutomaticMaskGenerator``)
path -- it emits one box per detected call instead of over-proposing and re-fusing
temporally overlapping calls, so overlapping/stacked USVs stay distinct instances. The
connected-component detector (``cc``) is the no-weights fallback.

Mask-array index convention (matches the consolidated-store readers in
``visualizations/make_usv_spectrograms.py`` and the masked-feature path):
``mask/<session>/segmentations`` is ``(M, F, T)`` boolean and ``mask/<session>/
spectrogram_index`` is ``(M,)`` int, where each row's ``spectrogram_index`` is the
usv_summary.csv row (== the ``spectrogram/<session>/spectrograms`` row) the mask belongs
to. A spec's masks are selected with ``spectrogram_index == spec_row`` and combined via
``np.any(segmentations[rows, :, :duration], axis=0)``. ``duration == 0`` placeholder
rows get no masks, and masks are stored at the full ``(F, T)`` size -- the reader crops
each to the spec's ``duration``.

The heavy compute dependencies (``torch``, ``sam2``, ``ultralytics``) are imported
lazily inside :meth:`MaskGenerator.generate_session_masks` -- importing this module
never pulls them in, and a missing SAM2 checkpoint / YOLO weights raises a clear,
actionable error BEFORE any GPU library is touched.
"""

from __future__ import annotations

import gc
import logging
import os
import pathlib
import sys
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import matplotlib
import numpy as np
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import configure_path, derive_spectrogram_model_paths, first_match_or_raise
from ..time_utils import is_gui_context, smart_wait


def flatten_session_masks(
    processed_masks: dict[int, list[dict]],
    num_freq_bins: int,
    num_time_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Flattens the per-spec mask dictionaries returned by
    ``process_session_batch_boxprompt`` into the two consolidated-store mask
    datasets written under ``mask/<session>``.

    The kernel returns ``{spec_index: [ {'segmentation': bool[F, w], ...}, ... ]}``,
    where ``spec_index`` is the usv_summary.csv row of the spectrogram and each
    mask's ``segmentation`` spans only the spec's valid ``w == min(duration, T)``
    columns. This packs every mask into a fixed ``(F, T)`` boolean frame (the
    segmentation placed in the first ``w`` columns, the rest left ``False``) so the
    readers can slice ``segmentations[rows, :, :duration]`` back out, and records the
    owning spec row for each mask in a parallel ``spectrogram_index`` array.

    Parameters
    ----------
    processed_masks (dict[int, list[dict]])
        Mapping ``spec_row -> list of mask dicts`` (each with a boolean
        ``'segmentation'`` of shape ``(F, w)``). Spec rows with no masks map to an
        empty list and contribute no rows.
    num_freq_bins (int)
        Frequency-bin count ``F`` of the stored frame (mask row height).
    num_time_bins (int)
        Time-bin count ``T`` of the stored frame (mask row width); each mask is
        padded out to this width.

    Returns
    -------
    segmentations (np.ndarray)
        A ``(M, F, T)`` boolean array of all masks across the session (``M`` is the
        total mask count); shape ``(0, F, T)`` when no masks were produced.
    spectrogram_index (np.ndarray)
        A ``(M,)`` int64 array giving, for each mask row, the usv_summary.csv row of
        the spectrogram it belongs to.
    """

    total_masks = sum(len(processed_masks[spec_row]) for spec_row in processed_masks)

    if total_masks == 0:
        segmentations = np.zeros((0, num_freq_bins, num_time_bins), dtype=bool)
        spectrogram_index = np.zeros((0,), dtype=np.int64)
        return segmentations, spectrogram_index

    # Preallocate the full ``(M, F, T)`` output (zero-initialised, so any
    # columns beyond ``valid_cols`` stay ``False``) and fill each mask row in
    # place, avoiding the previous per-mask zero frame plus the second full
    # ``np.stack`` copy. ``spectrogram_index`` is filled in lockstep.
    segmentations = np.zeros((total_masks, num_freq_bins, num_time_bins), dtype=bool)
    spectrogram_index = np.empty((total_masks,), dtype=np.int64)

    row_idx = 0
    for spec_row in sorted(processed_masks.keys()):
        for mask in processed_masks[spec_row]:
            seg = np.asarray(mask['segmentation'], dtype=bool)
            valid_cols = min(seg.shape[1], num_time_bins)
            # The box-prompt path detects boxes and prompts SAM2 on np.flipud(spec)
            # (image orientation, high-freq-at-top), so each returned mask's frequency
            # axis is inverted relative to the stored spectrogram and the ascending
            # ``frequency_bins`` axis the acoustic-feature / QLVM code index against.
            # Flip it back to natural (low-freq-first) orientation here so masked
            # features and latents land at the correct frequency.
            segmentations[row_idx, :, :valid_cols] = np.flipud(seg)[:, :valid_cols]
            spectrogram_index[row_idx] = int(spec_row)
            row_idx += 1

    return segmentations, spectrogram_index


class MaskGenerator:
    """
    Description
    -----------
    Generates per-USV instance masks for one session via the vendored SAM2
    box-prompt kernels and writes them into the session's spectrogram HDF5 file
    under a ``mask/<session>`` group consumed by the spectrogram viewers and the
    masked acoustic-feature path.
    """

    def __init__(
        self,
        root_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the MaskGenerator.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree with the
            spectrogram H5 written by ``generate_spectrograms``).
        input_parameter_dict (dict)
            Processing settings; the ``generate_masks`` block supplies the SAM2
            checkpoint/config paths, the YOLO weights + thresholds, and the
            box-prompt levers.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.root_directory = root_directory
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def _merge_patched_session_masks(
        self,
        h5_loc: pathlib.Path,
        mask_group_key: str,
        row_subset: set[int],
        new_segmentations: np.ndarray,
        new_spectrogram_index: np.ndarray,
        num_freq_bins: int,
        num_time_bins: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Description
        -----------
        Splices freshly segmented masks for a subset of USV rows into the
        session's existing mask arrays, preserving the masks of every row
        outside the subset.

        Instance masks are stored as one flat ``(M, F, T)`` stack with a
        parallel ``spectrogram_index`` array rather than per row, so patching a
        row means dropping its old masks, appending the new ones, and restoring
        the index-sorted order :func:`flatten_session_masks` guarantees. When
        the session has no mask group yet, the new masks are returned unchanged.

        Parameters
        ----------
        h5_loc (pathlib.Path)
            Per-session spectrogram HDF5 holding the ``mask/<session>`` group.
        mask_group_key (str)
            Key of that group.
        row_subset (set[int])
            USV rows whose masks were recomputed; their previous masks are
            dropped.
        new_segmentations (np.ndarray)
            ``(m, F, T)`` boolean masks freshly produced for the subset rows.
        new_spectrogram_index (np.ndarray)
            ``(m,)`` native usv_summary rows owning each new mask.
        num_freq_bins (int)
            Frequency-bin count ``F`` (used to shape an empty result).
        num_time_bins (int)
            Time-bin count ``T`` (used to shape an empty result).

        Returns
        -------
        segmentations, spectrogram_index (tuple[np.ndarray, np.ndarray])
            The merged mask stack and its owning-row index, sorted ascending by
            row so the merged arrays are indistinguishable from a full re-run.
        """

        with h5py.File(h5_loc, "r") as h5_file:
            if mask_group_key not in h5_file:
                return new_segmentations, new_spectrogram_index
            existing_segmentations = h5_file[mask_group_key]["segmentations"][:]
            existing_spectrogram_index = h5_file[mask_group_key]["spectrogram_index"][:]

        retained = ~np.isin(existing_spectrogram_index, np.fromiter(row_subset, dtype=np.int64, count=len(row_subset)))
        retained_segmentations = existing_segmentations[retained]
        retained_spectrogram_index = existing_spectrogram_index[retained]

        if retained_segmentations.shape[0] == 0 and new_segmentations.shape[0] == 0:
            return (
                np.zeros((0, num_freq_bins, num_time_bins), dtype=bool),
                np.zeros((0,), dtype=np.int64),
            )

        merged_segmentations = np.concatenate([retained_segmentations, new_segmentations], axis=0)
        merged_spectrogram_index = np.concatenate([retained_spectrogram_index, new_spectrogram_index], axis=0)
        # ``flatten_session_masks`` emits masks grouped by ascending owning row;
        # a stable sort restores exactly that order (and keeps each row's masks
        # in their original within-row sequence) after the splice.
        sort_order = np.argsort(merged_spectrogram_index, kind="stable")
        self.message_output(
            f"Patched masks for {len(row_subset)} USV row(s): "
            f"{retained_segmentations.shape[0]} retained + {new_segmentations.shape[0]} new."
        )
        return merged_segmentations[sort_order], merged_spectrogram_index[sort_order]

    def generate_session_masks(self, row_subset: set[int] | None = None) -> None:
        """
        Description
        -----------
        Reads the session's per-USV spectrograms, runs the selected box detector +
        SAM2 box-prompt segmenter over the valid (``duration > 0``) USVs, and writes
        the resulting instance masks into the SAME
        ``audio/spectrograms/<session>_spectrograms.h5`` under a ``mask/<session>``
        group: ``segmentations`` ``(M, F, T)`` boolean and ``spectrogram_index``
        ``(M,)`` int, where each mask's ``spectrogram_index`` is the usv_summary.csv
        row it belongs to.

        The full ``(N, F, T)`` spectrogram stack and ``(N,)`` durations are passed
        to the kernel unchanged: it skips rows shorter than ``duration_min`` (which
        includes every ``duration == 0`` placeholder), so the returned dictionary is
        already keyed by the native usv_summary row index. An existing
        ``mask/<session>`` group is overwritten so re-runs are idempotent.

        The detector is selectable: ``detector='yolo'`` builds a learned ``detect_fn``
        that proposes one box per call, while ``detector='cc'`` builds no ``detect_fn``
        and the kernel falls back to its connected-component detector (the no-weights
        path).

        The heavy compute libraries (``torch``, ``sam2``, ``ultralytics``) are
        imported here, not at module load. The SAM2 checkpoint/config and YOLO
        weights are validated up front and a missing path raises ``FileNotFoundError``
        with the offending setting before any GPU library is imported.

        With ``row_subset`` the method switches to patch mode: only those rows
        are segmented (the kernel receives a sub-stack and its output is remapped
        back to native rows), and the result is spliced into the existing mask
        arrays so every other row keeps the masks it already had. This pairs with
        the ``row_subset`` mode of ``SpectrogramGenerator`` for refreshing the
        USVs whose boundaries changed, without re-segmenting the whole session.

        Parameters
        ----------
        row_subset (set[int] | None)
            When None (default), the whole session is segmented and any existing
            ``mask/<session>`` group is replaced. When a set of
            ``usv_summary.csv`` row indices, only those rows are re-segmented and
            merged into the existing masks; an out-of-range index raises
            ``ValueError``.

        Returns
        -------
        .h5 ``mask/<session>`` group
            Instance masks appended to the per-session spectrogram H5.
        """

        self.message_output(
            f"Mask generation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        derive_spectrogram_model_paths(self.input_parameter_dict)
        cfg = self.input_parameter_dict['generate_masks']
        method = cfg['method']
        detector = cfg['detector']
        # The three filesystem paths are stored in the canonical /mnt/falkner form
        # and translated to the host OS's mount (e.g. /Volumes/falkner on macOS) via
        # configure_path, matching das_inference; sam2_model_cfg is a config NAME
        # resolved inside the SAM2 install, not a mount path, so it is left as-is.
        sam2_model_dir = configure_path(cfg['sam2_model_dir'])
        sam2_model_cfg = cfg['sam2_model_cfg']
        sam2_model_path = configure_path(cfg['sam2_model_path'])
        # Only translate yolo_weights when the YOLO detector is actually selected and a
        # path is configured: configure_path raises on None, so the documented cc no-weights
        # fallback (yolo_weights set to JSON null) must skip the translation entirely.
        yolo_weights = (
            configure_path(cfg['yolo_weights'])
            if (detector == 'yolo' and cfg['yolo_weights'])
            else cfg['yolo_weights']
        )

        if method != 'boxprompt':
            error_message = (
                f"generate_masks currently supports method='boxprompt' only, got {method!r}. "
                f"Set processing_settings['generate_masks']['method'] = 'boxprompt'."
            )
            raise ValueError(error_message)

        # Validate the model paths BEFORE importing torch/sam2/ultralytics so a
        # misconfiguration surfaces immediately with the offending setting rather than
        # as an opaque failure deep inside the GPU stack.
        if not sam2_model_dir or not pathlib.Path(sam2_model_dir).is_dir():
            error_message = (
                f"SAM2 model directory not configured or missing: {sam2_model_dir!r}. "
                f"Set processing_settings['generate_masks']['sam2_model_dir']."
            )
            raise FileNotFoundError(error_message)
        if not sam2_model_path or not pathlib.Path(sam2_model_path).is_file():
            error_message = (
                f"SAM2 checkpoint not configured or missing: {sam2_model_path!r}. "
                f"Set processing_settings['generate_masks']['sam2_model_path']."
            )
            raise FileNotFoundError(error_message)
        if not sam2_model_cfg:
            error_message = (
                "SAM2 model config not configured: set "
                "processing_settings['generate_masks']['sam2_model_cfg'] (a config name/path "
                "resolvable from sam2_model_dir)."
            )
            raise FileNotFoundError(error_message)
        if detector == 'yolo' and (not yolo_weights or not pathlib.Path(yolo_weights).is_file()):
            error_message = (
                f"YOLO weights not configured or missing: {yolo_weights!r}. "
                f"Set processing_settings['generate_masks']['yolo_weights'] (a trained best.pt), "
                f"or switch processing_settings['generate_masks']['detector'] to 'cc'."
            )
            raise FileNotFoundError(error_message)

        root = pathlib.Path(self.root_directory)
        session_id = root.name

        # Session-keyed, NOT "*_spectrograms.h5": sessions can hold other files
        # ending in that suffix (e.g. a separate sonic-band
        # "<session>_3_30khz_spectrograms.h5"), and a wildcard picks whichever
        # sorts first -- silently reading a file with no spectrogram/<session>
        # group. The per-session name written by generate_spectrograms is exact.
        h5_loc = first_match_or_raise(
            root=root / "audio" / "spectrograms",
            pattern=f"{session_id}_spectrograms.h5",
            label="per-session spectrogram H5",
        )
        with h5py.File(h5_loc, "r") as h5_file:
            session_group = h5_file[f"spectrogram/{session_id}"]
            specs = session_group["spectrograms"][:]
            durations = session_group["durations"][:]

        num_specs, num_freq_bins, num_time_bins = specs.shape

        # Patch mode: segment only the requested rows. The kernel is handed a
        # sub-stack, so its returned dictionary is keyed by sub-stack position
        # and is remapped back to native usv_summary rows below; masks of rows
        # outside the subset are carried over from the existing group untouched.
        subset_indices: list[int] = []
        if row_subset is None:
            kernel_specs = specs
            kernel_durations = durations
        else:
            out_of_range = [idx for idx in row_subset if idx < 0 or idx >= num_specs]
            if out_of_range:
                error_message = (
                    f"row_subset contains out-of-range rows {sorted(out_of_range)[:5]} for session "
                    f"'{session_id}' with {num_specs} spectrogram rows"
                )
                raise ValueError(error_message)
            subset_indices = sorted(row_subset)
            kernel_specs = specs[subset_indices]
            kernel_durations = durations[subset_indices]

        # Match the kernel's segmentation threshold (it skips rows shorter than
        # duration_min, not just duration==0 placeholders) so this up-front count
        # reflects exactly what will be segmented.
        valid_count = int(np.count_nonzero(kernel_durations >= cfg['duration_min']))
        self.message_output(
            f"Segmenting {valid_count} valid USVs (of {len(kernel_durations)}) for session {session_id}."
        )

        # Heavy, GPU-only kernels: import lazily so `import generate_masks` and the
        # CLI plumbing stay torch-free until an actual run is requested.
        from .masks._common_memory import (  # noqa: PLC0415 (lazy: keeps torch out of import)
            setup_device,
        )
        from .masks.box_detectors.detect import (  # noqa: PLC0415 (lazy: pulls ultralytics)
            get_detector,
        )
        from .masks.boxprompt_utils import (  # noqa: PLC0415 (lazy: pulls torch + sam2)
            build_predictor,
            process_session_batch_boxprompt,
        )

        logger = logging.getLogger("usv_playpen.generate_masks")

        cmap = matplotlib.colormaps[cfg['mask_cmap']]
        # cuDNN's autotuner picks algorithms from the GPU state at process
        # start, and with the bf16 autocast + TF32 this module enables that
        # shifts borderline detections by more than the margin separating a
        # faint call from the detector's confidence gate. Two runs in ONE
        # process agree bit-for-bit, but runs in DIFFERENT processes do not:
        # the cohort's original mask pass left 9-26% of rows with no mask,
        # while a rerun of the identical code on the identical spectrograms
        # left 1.4%. Disabling the autotuner (deterministic) trades a little
        # speed for masks that reproduce across processes.
        device = setup_device(deterministic=cfg['deterministic'], logger=logger)

        # SAM2's hydra config + (possibly relative) checkpoint resolve against the
        # model directory, so build the predictor from there (mirroring specgen), then
        # restore the original working directory.
        original_dir = pathlib.Path.cwd()
        try:
            os.chdir(sam2_model_dir)
            predictor = build_predictor(
                model_cfg=sam2_model_cfg,
                checkpoint=sam2_model_path,
                device=device,
                logger=logger,
            )
        finally:
            os.chdir(original_dir)

        detect_fn = None
        if detector == 'yolo':
            detect_fn = get_detector(
                "yolo",
                weights=yolo_weights,
                conf=cfg['yolo_conf'],
                iou=cfg['yolo_iou'],
                imgsz=cfg['yolo_imgsz'],
                device=None,
            )

        # Pass the FULL spectrogram stack + durations: the kernel skips rows shorter
        # than duration_min (including every duration==0 placeholder), so the returned
        # dict is already keyed by the native usv_summary row index.
        _, processed_masks = process_session_batch_boxprompt(
            session_id,
            kernel_specs,
            kernel_durations,
            predictor,
            None,  # detector_cfg: cc connected-component config is unused here (yolo uses detect_fn below)
            cmap,
            duration_min=cfg['duration_min'],
            batch_size=cfg['batch_size'],
            multimask_output=cfg['multimask_output'],
            iou_floor=cfg['iou_floor'],
            drop_below_iou=cfg['drop_below_iou'],
            split_disconnected=cfg['split_disconnected'],
            max_iters=cfg['max_iters'],
            merge_instances=cfg['merge_instances'],
            merge_iou=cfg['merge_iou'],
            merge_containment=cfg['merge_containment'],
            detect_fn=detect_fn,
            mask_intensity_floor=cfg['mask_intensity_floor'],
            tiny_mask_floor_px=cfg['tiny_mask_floor_px'],
            min_box_area=cfg['min_box_area'],
            logger=logger,
        )

        if row_subset is not None:
            processed_masks = {
                subset_indices[sub_position]: masks for sub_position, masks in processed_masks.items()
            }

        segmentations, spectrogram_index = flatten_session_masks(
            processed_masks=processed_masks,
            num_freq_bins=num_freq_bins,
            num_time_bins=num_time_bins,
        )

        mask_group_key = f"mask/{session_id}"
        if row_subset is not None:
            segmentations, spectrogram_index = self._merge_patched_session_masks(
                h5_loc=h5_loc,
                mask_group_key=mask_group_key,
                row_subset=row_subset,
                new_segmentations=segmentations,
                new_spectrogram_index=spectrogram_index,
                num_freq_bins=num_freq_bins,
                num_time_bins=num_time_bins,
            )

        with h5py.File(h5_loc, "a") as h5_file:
            if mask_group_key in h5_file:
                del h5_file[mask_group_key]
            mask_group = h5_file.create_group(mask_group_key)
            mask_group.attrs["created"] = "generate_masks"
            mask_group.attrs["method"] = method
            mask_group.attrs["detector"] = detector
            mask_group.attrs["total_masks"] = int(segmentations.shape[0])
            if segmentations.shape[0] > 0:
                mask_group.create_dataset(
                    "segmentations", data=segmentations, compression="gzip", compression_opts=6
                )
                mask_group.create_dataset(
                    "spectrogram_index", data=spectrogram_index, compression="gzip", compression_opts=6
                )
            else:
                mask_group.create_dataset("segmentations", data=segmentations)
                mask_group.create_dataset("spectrogram_index", data=spectrogram_index)

        # Drop this call's predictor/detector and return the allocator's free
        # blocks to the driver. This REDUCES but does not eliminate the device
        # memory a process retains per session: SAM2/Hydra/ultralytics keep
        # references of their own that we cannot reach from here (measured
        # ~250 MiB retained per session on a 16 GiB card). A caller that walks
        # many sessions must therefore CHUNK ITS WORK ACROSS PROCESSES rather
        # than rely on this cleanup -- one CLI invocation per session, as the
        # shipped command does, is already safe. torch is only consulted when
        # it is already loaded, so this stays free of a heavy import (and of a
        # GPU dependency) when the kernels are stubbed out.
        del predictor
        del detect_fn
        gc.collect()
        if "torch" in sys.modules:
            torch_module = sys.modules["torch"]
            if torch_module.cuda.is_available():
                torch_module.cuda.empty_cache()

        self.message_output(
            f"Wrote {segmentations.shape[0]} instance masks across {len(np.unique(spectrogram_index))} USVs "
            f"to {mask_group_key} in {h5_loc}."
        )
        self.message_output(
            f"Mask generation ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="generate-usv-masks")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--detector', 'detector', type=click.Choice(['yolo', 'cc']), default=None, required=False, help='Box detector backend (yolo learned detector or cc baseline).')
@click.option('--sam2-model-dir', 'sam2_model_dir', type=str, default=None, required=False, help='SAM2 model directory (config/checkpoint resolve against it).')
@click.option('--sam2-model-cfg', 'sam2_model_cfg', type=str, default=None, required=False, help='SAM2 model config name/path (resolvable from sam2_model_dir).')
@click.option('--sam2-model-path', 'sam2_model_path', type=str, default=None, required=False, help='SAM2 checkpoint path.')
@click.option('--yolo-weights', 'yolo_weights', type=str, default=None, required=False, help='Trained YOLO best.pt weights path.')
@click.option('--yolo-conf', 'yolo_conf', type=float, default=None, required=False, help='YOLO confidence threshold (lower => more recall).')
@click.option('--yolo-iou', 'yolo_iou', type=float, default=None, required=False, help='YOLO NMS IoU (raise to keep stacked calls).')
@click.option('--method', 'method', type=str, default=None, required=False, help="Mask-generation method; only 'boxprompt' (SAM2 box-prompt path) is supported.")
@click.option('--yolo-imgsz', 'yolo_imgsz', type=int, default=None, required=False, help='YOLO detector input image size in px (native spectrogram size is 128).')
@click.option('--deterministic/--no-deterministic', 'deterministic', default=None, required=False, help="Disable cuDNN's autotuner so masks reproduce across processes (default: enabled). "
     "With the autotuner on, algorithm choice varies with the GPU state at process start and "
     "borderline faint calls fall on either side of the detector's confidence gate.")
@click.option('--mask-cmap', 'mask_cmap', type=str, default=None, required=False, help='Matplotlib colormap used to render each spectrogram to RGB before SAM2 prompting.')
@click.option('--duration-min', 'duration_min', type=int, default=None, required=False, help='Minimum USV duration (time bins) to segment; shorter/placeholder (duration==0) rows are skipped.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Number of spectrograms processed per batch before a memory-cleanup pass.')
@click.option('--multimask-output/--no-multimask-output', 'multimask_output', default=None, required=False, help='Let SAM2 emit multiple candidate masks per box and keep the highest-IoU one (vs a single mask).')
@click.option('--iou-floor', 'iou_floor', type=float, default=None, required=False, help='Predicted-IoU threshold below which a mask is flagged low-IoU (see --drop-below-iou).')
@click.option('--drop-below-iou/--no-drop-below-iou', 'drop_below_iou', default=None, required=False, help='Discard masks whose SAM2 predicted IoU is below --iou-floor (default keeps them).')
@click.option('--split-disconnected/--no-split-disconnected', 'split_disconnected', default=None, required=False, help='Split a SAM2 mask with multiple 8-connected components into separate instances.')
@click.option('--max-iters', 'max_iters', type=int, default=None, required=False, help='Residual re-prompting passes for the cc detector (the yolo detector is always single-pass).')
@click.option('--merge-instances/--no-merge-instances', 'merge_instances', default=None, required=False, help='Post-merge near-duplicate / contained instances to correct over-segmentation.')
@click.option('--merge-iou', 'merge_iou', type=float, default=None, required=False, help='IoU above which two overlapping instances are fused in the post-merge step.')
@click.option('--merge-containment', 'merge_containment', type=float, default=None, required=False, help='Containment fraction above which a smaller instance is merged into a larger enclosing one.')
@click.option('--mask-intensity-floor', 'mask_intensity_floor', type=float, default=None, required=False, help='Normalized-intensity floor; keep only mask pixels at/above it (drops faint harmonics/tails). 0 disables.')
@click.option('--tiny-mask-floor-px', 'tiny_mask_floor_px', type=int, default=None, required=False, help='Minimum mask area in px; smaller masks / split components are dropped.')
@click.option('--min-box-area', 'min_box_area', type=int, default=None, required=False, help='Minimum detector box area in px before SAM2 prompting; 0 disables the gate.')
@click.pass_context
def generate_masks_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to generate per-USV instance masks for one session and
    write them into the session's spectrogram H5.

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
        block='generate_masks',
    )

    MaskGenerator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).generate_session_masks()
