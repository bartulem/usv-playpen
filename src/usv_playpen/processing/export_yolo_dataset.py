"""
@author: bartulem
Export a YOLO box-detector training dataset from per-session USV spectrograms.

To train the YOLO box detector that :mod:`generate_masks` uses (method
``boxprompt``, detector ``yolo``), the detector needs ground-truth bounding boxes
around the calls on each spectrogram image. This module renders every valid USV
spectrogram to an image EXACTLY the way the inference path renders it
(``np.flipud`` -> slice to the signal window -> ``render_spec_image`` with the
same colormap) and writes an Ultralytics-format dataset (``images/{train,val}``,
``labels/{train,val}`` with one ``class xc yc w h`` line per box, normalized, and
a ``data.yaml``), so the trained detector sees the same distribution at train and
inference time.

Label source (``label_source``):

* ``"cc"`` (default, zero manual work) -- pseudo-label boxes with the unlearned
  connected-component detector (``cc_box_detector`` via the unified
  ``get_detector("cc")``). This bootstraps a learned YOLO from the baseline with
  no annotation effort.
* ``"manual"`` -- use hand-verified YOLO-format label files (``{spec_id}.txt``)
  from ``manual_labels_directory``; spectrograms with no label file get an empty
  label file (treated as background).
* ``"merge"`` -- cc pseudo-labels, overridden by a manual ``{spec_id}.txt`` for
  every spectrogram that has one.

The connected-component detector is pure numpy/scipy/skimage, so building a
``cc``/``merge`` dataset needs no GPU and no torch.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import numpy as np
from click.core import ParameterSource
from PIL import Image

from ..cli_utils import modify_settings_json_for_cli
from ..time_utils import is_gui_context, smart_wait
from .masks.box_detectors.common.boxes import render_spec_image, tlbr_to_xywhn
from .masks.box_detectors.detect import get_detector

# Single-class detector: every box is a USV call.
_CLASS_ID = 0
_CLASS_NAME = "usv"


def spec_to_yolo_image(spec: np.ndarray, duration: int, colormap: str) -> tuple[np.ndarray, int, int]:
    """
    Description
    -----------
    Renders one spectrogram to the YOLO training image exactly as the inference
    detector renders it: flip vertically (``np.flipud``), slice to the signal
    window ``[0, min(duration, T))``, and colour-map to a uint8 RGB image. Returns
    the image plus its ``(width, height)`` for label normalization.

    Parameters
    ----------
    spec (np.ndarray)
        A ``(F, T)`` spectrogram (unflipped, as stored in the H5).
    duration (int)
        Native signal length in time bins (the window width is ``min(duration, T)``).
    colormap (str)
        Matplotlib colormap name (must match the detector's ``colormap``).

    Returns
    -------
    image (np.ndarray)
        A uint8 ``(H, W, 3)`` RGB image (``H == F``, ``W == min(duration, T)``).
    width (int)
        Image width ``W`` (for normalizing box x/width).
    height (int)
        Image height ``H`` (for normalizing box y/height).
    """

    n_time = spec.shape[1]
    eff_duration = max(1, min(int(duration), n_time))
    window = np.flipud(spec)[:, :eff_duration]
    image = render_spec_image(window, colormap=colormap)
    height, width = image.shape[0], image.shape[1]
    return image, width, height


class YOLODatasetExporter:
    """
    Description
    -----------
    Exports an Ultralytics-format YOLO box-detector dataset (images + normalized
    box labels + ``data.yaml``) from a list of per-session spectrogram H5 files,
    pseudo-labeling boxes with the connected-component detector and/or ingesting
    hand-verified labels.
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
        Initializes the YOLODatasetExporter.

        Parameters
        ----------
        spectrogram_h5_paths (list[str])
            Per-session ``*_spectrograms.h5`` files to draw spectrograms from.
        output_directory (str)
            Directory to write the YOLO dataset (``images/``, ``labels/``,
            ``data.yaml``); created if missing.
        input_parameter_dict (dict)
            Processing settings; the ``export_yolo_dataset`` block supplies the
            label source, split, colormap, and manual-labels directory.
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

    def _cc_labels(self, detect_fn, spec: np.ndarray, duration: int, width: int, height: int) -> list[str]:
        """
        Description
        -----------
        Runs the connected-component detector on one spectrogram and returns its
        boxes as YOLO label lines (``"0 xc yc w h"``, normalized). ``detect_fn``
        takes the UNFLIPPED spec and returns flipped-space ``(t, l, b, r)`` boxes
        (same orientation as the rendered window image), which are normalized by
        the rendered image's ``(width, height)``.

        Parameters
        ----------
        detect_fn (Callable)
            The ``get_detector("cc")`` closure ``detect(spec, duration) -> boxes``.
        spec (np.ndarray)
            The ``(F, T)`` spectrogram (unflipped).
        duration (int)
            Native signal length in time bins.
        width (int)
            Rendered-image width (box x/width normalizer).
        height (int)
            Rendered-image height (box y/height normalizer).

        Returns
        -------
        lines (list[str])
            YOLO label lines, one per detected box.
        """

        boxes = detect_fn(spec, duration)
        lines: list[str] = []
        for box in boxes:
            xc, yc, box_w, box_h = tlbr_to_xywhn(tuple(int(v) for v in box), width, height)
            lines.append(f"{_CLASS_ID} {xc:.6f} {yc:.6f} {box_w:.6f} {box_h:.6f}")
        return lines

    def export(self) -> None:
        """
        Description
        -----------
        Renders every valid (``duration > 0``) spectrogram across all input
        sessions to a YOLO image, attaches box labels per ``label_source``, splits
        the spectrograms into train/val as an exact, reproducible
        ``validation_split`` fraction (a seeded permutation over the whole dataset,
        so the val set is never accidentally empty), and writes the Ultralytics
        dataset (``images/{train,val}/{spec_id}.png``,
        ``labels/{train,val}/{spec_id}.txt``, ``data.yaml``) to the output
        directory.

        Parameters
        ----------

        Returns
        -------
        A YOLO dataset directory (``images/``, ``labels/``, ``data.yaml``).
        """

        self.message_output(
            f"YOLO dataset export started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['export_yolo_dataset']
        label_source = cfg['label_source']
        validation_split = cfg['validation_split']
        random_state = cfg['random_state']
        colormap = cfg['colormap']
        manual_labels_directory = cfg['manual_labels_directory']

        if label_source in ("manual", "merge") and not manual_labels_directory:
            error_message = (
                f"label_source='{label_source}' requires "
                f"processing_settings['export_yolo_dataset']['manual_labels_directory'] to be set."
            )
            raise ValueError(error_message)
        manual_dir = pathlib.Path(manual_labels_directory) if manual_labels_directory else None

        output_dir = pathlib.Path(self.output_directory)
        for split in ("train", "val"):
            (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        detect_fn = get_detector("cc") if label_source in ("cc", "merge") else None

        # Phase 1: enumerate every valid (duration > 0) spectrogram across all
        # sessions in all files, in a stable order (file list order, then H5 group
        # insertion order, then row order). An H5 may hold more than one session
        # group, so iterate them all rather than assuming a single session.
        catalog: list[tuple[str, str, int]] = []
        for h5_path in self.spectrogram_h5_paths:
            with h5py.File(h5_path, "r") as h5_file:
                for session_id in h5_file["spectrogram"]:
                    durations = h5_file[f"spectrogram/{session_id}"]["durations"][:]
                    for row in np.flatnonzero(durations > 0):
                        catalog.append((h5_path, session_id, int(row)))

        # The val set is an EXACT, reproducible fraction of the whole dataset (a
        # seeded permutation, not a per-image coin flip — the latter does not
        # guarantee the fraction and can leave the val split empty on small sets).
        n_total = len(catalog)
        n_val = round(n_total * validation_split)
        rng = np.random.default_rng(random_state)
        val_positions = {int(i) for i in rng.permutation(n_total)[:n_val]}

        # Phase 2: render + label + write, opening each file once in the SAME order.
        n_boxes = 0
        position = 0
        for h5_path in self.spectrogram_h5_paths:
            with h5py.File(h5_path, "r") as h5_file:
                for session_id in h5_file["spectrogram"]:
                    session_group = h5_file[f"spectrogram/{session_id}"]
                    specs = session_group["spectrograms"]
                    durations = session_group["durations"][:]
                    for row in np.flatnonzero(durations > 0):
                        split = "val" if position in val_positions else "train"
                        position += 1
                        spec = specs[row].astype(np.float32)
                        duration = int(durations[row])
                        spec_id = f"{session_id}_{int(row)}"

                        image, width, height = spec_to_yolo_image(spec, duration, colormap)

                        manual_file = manual_dir / f"{spec_id}.txt" if manual_dir is not None else None
                        has_manual = manual_file is not None and manual_file.is_file()
                        if label_source == "manual" or (label_source == "merge" and has_manual):
                            # Drop blank lines (trailing newline / accidental double
                            # newlines) -- Ultralytics rejects empty label lines.
                            raw_lines = manual_file.read_text().splitlines() if has_manual else []
                            lines = [line for line in raw_lines if line.strip()]
                        else:
                            lines = self._cc_labels(detect_fn, spec, duration, width, height)

                        Image.fromarray(image).save(output_dir / "images" / split / f"{spec_id}.png")
                        (output_dir / "labels" / split / f"{spec_id}.txt").write_text("\n".join(lines))
                        n_boxes += len(lines)

        # Quote the path so a value with spaces / a Windows drive letter stays valid YAML.
        data_yaml = (
            f'path: "{output_dir}"\n'
            f"train: images/train\n"
            f"val: images/val\n"
            f"nc: 1\n"
            f"names:\n  {_CLASS_ID}: {_CLASS_NAME}\n"
        )
        (output_dir / "data.yaml").write_text(data_yaml)

        self.message_output(
            f"Exported {n_total} spectrogram images ({n_boxes} boxes, label_source='{label_source}'; "
            f"{n_val} val / {n_total - n_val} train) -> {output_dir} (data.yaml written)."
        )
        self.message_output(
            f"YOLO dataset export ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="export-yolo-dataset")
@click.option('--spectrogram-h5-paths', type=str, required=True, help='Comma-separated per-session *_spectrograms.h5 paths.')
@click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory to write the YOLO dataset.')
@click.option('--label-source', 'label_source', type=click.Choice(['cc', 'manual', 'merge']), default=None, required=False, help='Box label source: cc pseudo-labels, manual files, or merge.')
@click.option('--validation-split', 'validation_split', type=float, default=None, required=False, help='Fraction of images held out for validation.')
@click.option('--manual-labels-directory', 'manual_labels_directory', type=str, default=None, required=False, help='Directory of hand-verified {spec_id}.txt YOLO labels (manual/merge).')
@click.pass_context
def export_yolo_dataset_cli(ctx, spectrogram_h5_paths, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to export a YOLO box-detector training dataset from
    per-session spectrogram H5 files.

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

    YOLODatasetExporter(
        spectrogram_h5_paths=h5_paths,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).export()
