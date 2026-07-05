"""
@author: bartulem
Train the YOLO box detector used by the in-house mask-generation step.

:mod:`generate_masks` (method ``boxprompt``, detector ``yolo``) proposes one box
per call with a trained Ultralytics YOLO model, then prompts SAM2 per box. This
module fine-tunes that YOLO detector from a COCO-pretrained checkpoint
(``yolo11n.pt`` by default) on the single-class USV dataset written by
:mod:`export_yolo_dataset`, and copies the resulting ``best.pt`` to a stable path
that ``generate_masks``' ``yolo_weights`` setting can point at.

Only the YOLO *box detector* is trained here -- SAM2 is used pretrained (it is
not fine-tuned in usv-playpen). Ultralytics + torch are imported lazily inside
:meth:`MaskDetectorTrainer.train`, so importing this module never pulls them in;
a missing ``data.yaml`` raises a clear error before any GPU library is touched.
"""

from __future__ import annotations

import pathlib
import shutil
from collections.abc import Callable
from datetime import datetime

import click
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..time_utils import is_gui_context, smart_wait


class MaskDetectorTrainer:
    """
    Description
    -----------
    Fine-tunes the Ultralytics YOLO box detector on an
    :mod:`export_yolo_dataset` dataset and exposes the trained ``best.pt`` at a
    stable output path for :mod:`generate_masks`.
    """

    def __init__(
        self,
        dataset_directory: str | None = None,
        output_directory: str | None = None,
        input_parameter_dict: dict | None = None,
        message_output: Callable | None = None,
    ) -> None:
        """
        Description
        -----------
        Initializes the MaskDetectorTrainer.

        Parameters
        ----------
        dataset_directory (str)
            Directory holding the YOLO dataset (must contain ``data.yaml``) from
            :mod:`export_yolo_dataset`.
        output_directory (str)
            Directory for the Ultralytics run and the copied ``best.pt`` (created
            if missing).
        input_parameter_dict (dict)
            Processing settings; the ``train_masks`` block supplies the base
            weights and the training hyperparameters.
        message_output (Callable)
            Logging callback; defaults to ``print``.

        Returns
        -------
        None
        """

        self.dataset_directory = dataset_directory
        self.output_directory = output_directory
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output if message_output is not None else print
        self.app_context_bool = is_gui_context()

    def train(self) -> None:
        """
        Description
        -----------
        Fine-tunes the YOLO detector on the dataset's ``data.yaml`` for
        ``n_epochs`` at image size ``imgsz`` (128, the native spectrogram size),
        then copies the run's ``best.pt`` to ``<output_directory>/best.pt`` -- the
        path to set as ``generate_masks``' ``yolo_weights``.

        Ultralytics + torch are imported here, not at module load; a missing
        ``data.yaml`` raises ``FileNotFoundError`` before any GPU library is
        imported.

        Parameters
        ----------

        Returns
        -------
        ``<output_directory>/best.pt``
            The trained YOLO detector weights (plus the full Ultralytics run dir).
        """

        self.message_output(
            f"YOLO box-detector training started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['train_masks']
        base_weights = cfg['base_weights']
        n_epochs = cfg['n_epochs']
        imgsz = cfg['imgsz']
        batch_size = cfg['batch_size']
        device = cfg['device']
        run_name = cfg['run_name']

        # Guard against None directories: reachable only via direct programmatic
        # construction (the CLI marks both required=True). pathlib.Path(None) would
        # otherwise raise an opaque TypeError; raise the clear error the docstrings promise.
        if self.dataset_directory is None:
            error_message = "dataset_directory must be provided (got None)."
            raise ValueError(error_message)
        if self.output_directory is None:
            error_message = "output_directory must be provided (got None)."
            raise ValueError(error_message)

        dataset_dir = pathlib.Path(self.dataset_directory)
        data_yaml = dataset_dir / "data.yaml"
        if not data_yaml.is_file():
            error_message = (
                f"No data.yaml in {dataset_dir}; run export-yolo-dataset first."
            )
            raise FileNotFoundError(error_message)

        output_dir = pathlib.Path(self.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Heavy, GPU-only library: import lazily so `import train_masks` and the CLI
        # plumbing stay torch/ultralytics-free until an actual training run starts.
        from ultralytics import YOLO  # noqa: PLC0415 (lazy: pulls torch + ultralytics)

        model = YOLO(base_weights)
        results = model.train(
            data=str(data_yaml),
            epochs=n_epochs,
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=str(output_dir),
            name=run_name,
            exist_ok=True,
        )

        # Copy the run's best.pt to a stable, predictable path for generate_masks.
        run_best = pathlib.Path(results.save_dir) / "weights" / "best.pt"
        stable_best = output_dir / "best.pt"
        if not run_best.is_file():
            error_message = (
                f"Training finished but {run_best} was not found; inspect the Ultralytics run at "
                f"{results.save_dir}. Not copying a stale best.pt."
            )
            raise FileNotFoundError(error_message)
        shutil.copy2(run_best, stable_best)
        self.message_output(
            f"Trained YOLO detector; best weights -> {stable_best} "
            f"(set processing_settings['generate_masks']['yolo_weights'] to this path)."
        )

        self.message_output(
            f"YOLO box-detector training ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="train-masks")
@click.option('--dataset-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='YOLO dataset directory (export-yolo-dataset output, with data.yaml).')
@click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory for the Ultralytics run + copied best.pt.')
@click.option('--base-weights', 'base_weights', type=str, default=None, required=False, help='Base YOLO checkpoint to fine-tune from (e.g. yolo11n.pt).')
@click.option('--n-epochs', 'n_epochs', type=int, default=None, required=False, help='Number of training epochs.')
@click.option('--imgsz', 'imgsz', type=int, default=None, required=False, help='Square image size (px) the detector trains at; 128 is the native spectrogram size.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Training batch size (imgs/batch).')
@click.option('--device', 'device', type=str, default=None, required=False, help='Compute device: a GPU index (e.g. "0"), "cpu", or omit for Ultralytics auto-select (null).')
@click.option('--run-name', 'run_name', type=str, default=None, required=False, help='Ultralytics run name (subdir under the output directory holding the run artifacts).')
@click.pass_context
def train_masks_cli(ctx, dataset_directory, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to fine-tune the YOLO box detector for mask generation
    on an exported YOLO dataset.

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
        block='train_masks',
    )

    MaskDetectorTrainer(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).train()
