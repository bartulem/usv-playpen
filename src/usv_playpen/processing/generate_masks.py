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

import logging
import os
import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import matplotlib
import numpy as np
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import configure_path, first_match_or_raise
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

    seg_rows: list[np.ndarray] = []
    index_rows: list[int] = []

    for spec_row in sorted(processed_masks.keys()):
        for mask in processed_masks[spec_row]:
            seg = np.asarray(mask['segmentation'], dtype=bool)
            frame = np.zeros((num_freq_bins, num_time_bins), dtype=bool)
            valid_cols = min(seg.shape[1], num_time_bins)
            frame[:, :valid_cols] = seg[:, :valid_cols]
            seg_rows.append(frame)
            index_rows.append(int(spec_row))

    if seg_rows:
        segmentations = np.stack(seg_rows, axis=0)
        spectrogram_index = np.asarray(index_rows, dtype=np.int64)
    else:
        segmentations = np.zeros((0, num_freq_bins, num_time_bins), dtype=bool)
        spectrogram_index = np.zeros((0,), dtype=np.int64)

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

    def generate_session_masks(self) -> None:
        """
        Description
        -----------
        Reads the session's per-USV spectrograms, runs the YOLO box detector +
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

        The heavy compute libraries (``torch``, ``sam2``, ``ultralytics``) are
        imported here, not at module load. The SAM2 checkpoint/config and YOLO
        weights are validated up front and a missing path raises ``FileNotFoundError``
        with the offending setting before any GPU library is imported.

        Parameters
        ----------

        Returns
        -------
        .h5 ``mask/<session>`` group
            Instance masks appended to the per-session spectrogram H5.
        """

        self.message_output(
            f"Mask generation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

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
        yolo_weights = configure_path(cfg['yolo_weights'])

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

        h5_loc = first_match_or_raise(
            root=root / "audio" / "spectrograms",
            pattern="*_spectrograms.h5",
            label="per-session spectrogram H5",
        )
        with h5py.File(h5_loc, "r") as h5_file:
            session_group = h5_file[f"spectrogram/{session_id}"]
            specs = session_group["spectrograms"][:]
            durations = session_group["durations"][:]

        num_specs, num_freq_bins, num_time_bins = specs.shape
        valid_count = int(np.count_nonzero(durations > 0))
        self.message_output(
            f"Segmenting {valid_count} valid USVs (of {num_specs}) for session {session_id}."
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
        device = setup_device(deterministic=False, logger=logger)

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
            specs,
            durations,
            predictor,
            None,
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

        segmentations, spectrogram_index = flatten_session_masks(
            processed_masks=processed_masks,
            num_freq_bins=num_freq_bins,
            num_time_bins=num_time_bins,
        )

        mask_group_key = f"mask/{session_id}"
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

        self.message_output(
            f"Wrote {segmentations.shape[0]} instance masks across {len(np.unique(spectrogram_index))} USVs "
            f"to {mask_group_key} in {h5_loc}."
        )
        self.message_output(
            f"Mask generation ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="generate-masks")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--detector', 'detector', type=click.Choice(['yolo', 'cc']), default=None, required=False, help='Box detector backend (yolo learned detector or cc baseline).')
@click.option('--sam2-model-dir', 'sam2_model_dir', type=str, default=None, required=False, help='SAM2 model directory (config/checkpoint resolve against it).')
@click.option('--sam2-model-cfg', 'sam2_model_cfg', type=str, default=None, required=False, help='SAM2 model config name/path (resolvable from sam2_model_dir).')
@click.option('--sam2-model-path', 'sam2_model_path', type=str, default=None, required=False, help='SAM2 checkpoint path.')
@click.option('--yolo-weights', 'yolo_weights', type=str, default=None, required=False, help='Trained YOLO best.pt weights path.')
@click.option('--yolo-conf', 'yolo_conf', type=float, default=None, required=False, help='YOLO confidence threshold (lower => more recall).')
@click.option('--yolo-iou', 'yolo_iou', type=float, default=None, required=False, help='YOLO NMS IoU (raise to keep stacked calls).')
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
    )

    MaskGenerator(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).generate_session_masks()
