"""
@author: bartulem
Train the QLVM (QMC latent-variable model) decoder on a curated USV-spectrogram
training set and export the weights the JAX inference path consumes.

This is the usv-playpen-native orchestrator around the vendored torch QLVM
training kernels in ``processing/qlvm_training/`` (ported from the external
``specgen`` ``spec_gen_full_pipeline`` package). It reads the ``.npz`` training
set written by :mod:`build_qlvm_training_set` (``train_data.npz`` + optional
``val_data.npz``), builds the fixed quasi-random lattice + ConvTranspose decoder,
trains the decoder under the Bernoulli evidence objective, and writes two
artifacts into the output directory:

* ``qmc_train_qlvm.tar`` -- the full torch checkpoint (model + optimizer +
  per-batch loss trajectory), for resuming / diagnostics; and
* ``qmc_decoder_weights.npz`` -- the decoder ``state_dict`` as one array per
  layer (keys ``"0.weight"``/``"0.bias"``/.../``"9.weight"``), which is EXACTLY
  what the torch-free JAX inference path (``analyses/qlvm_model.py`` via
  ``analyses/qlvm_latents.py``'s ``weights_npz_path``) loads. Training and
  inference therefore meet only at this ``.npz`` boundary -- the decoder
  architecture here (:func:`build_qmc_decoder`) is the one
  ``qlvm_model.decoder_forward`` reconstructs.

The model has no learned encoder: the torus is a fixed lattice and only the
decoder is trained. Each epoch applies a fresh random torus shift to the whole
lattice (the QMC integration trick). The heavy compute dependencies (``torch``,
the vendored kernels) are imported lazily inside :meth:`QLVMTrainer.train` so
importing this module never pulls in torch.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import numpy as np
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..time_utils import is_gui_context, smart_wait

# Output artifact names (the .npz is the train -> JAX-inference bridge file).
_CHECKPOINT_NAME = "qmc_train_qlvm.tar"
_DECODER_WEIGHTS_NAME = "qmc_decoder_weights.npz"


def build_qmc_decoder(latent_dim: int):
    """
    Description
    -----------
    Builds the ConvTranspose2d decoder for the QMCLVM, identical to the external
    trainer's ``build_qmc_decoder`` so the exported weights load unchanged into
    the JAX inference decoder (``analyses/qlvm_model.decoder_forward``).

    The input width is ``2 * latent_dim`` because ``TorusBasis`` expands each
    latent coordinate ``z`` to a ``(cos(2*pi*z), sin(2*pi*z))`` pair before
    decoding. The body is two linear layers up to a ``64x8x8`` feature map
    followed by four stride-2 ``ConvTranspose2d`` blocks (64 -> 32 -> 16 -> 8 ->
    1) and a final ``Sigmoid`` so outputs lie in ``[0, 1]`` (a ``128x128``
    spectrogram). The ``nn.Sequential`` layer indices (``0,1`` Linear; ``3,5,7,9``
    ConvTranspose) are the ``state_dict`` keys the inference path expects.

    Parameters
    ----------
    latent_dim (int)
        Torus latent dimensionality; the decoder input is ``2 * latent_dim``.

    Returns
    -------
    decoder (torch.nn.Sequential)
        The untrained decoder module.
    """

    from torch import nn  # noqa: PLC0415 (lazy: keeps torch off the import path)

    return nn.Sequential(
        nn.Linear(2 * latent_dim, 2048),
        nn.Linear(2048, 64 * 8 * 8),
        nn.Unflatten(1, (64, 8, 8)),
        nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        nn.Sigmoid(),
    )


def build_lattice(lattice_type: str, latent_dim: int, korobov_a: int, n_points: int, fib_m: int):
    """
    Description
    -----------
    Builds the fixed quasi-random latent lattice (``base_sequence``) the decoder
    is trained over, selecting the generator by ``lattice_type``. Mirrors the
    external trainer's ``build_lattice_pair`` (and the lattice
    ``analyses/qlvm_model.py`` rebuilds at inference, so the same settings keep
    train and inference on the same torus).

    Parameters
    ----------
    lattice_type (str)
        ``"korobov"`` (Korobov lattice), ``"roberts"`` (Roberts low-discrepancy
        sequence), or anything else (Fibonacci lattice, 2D only).
    latent_dim (int)
        Torus latent dimensionality.
    korobov_a (int)
        Korobov generating integer (used only for ``"korobov"``).
    n_points (int)
        Number of lattice points for korobov/roberts.
    fib_m (int)
        Fibonacci lattice order (used only for the Fibonacci fallback).

    Returns
    -------
    base_sequence (torch.Tensor)
        A ``(n_lattice_points, latent_dim)`` lattice tensor.
    """

    from .qlvm_training.sampling import (  # noqa: PLC0415 (lazy: pulls torch + scipy)
        gen_fib_basis,
        gen_korobov_basis,
        roberts_sequence,
    )

    if lattice_type == "korobov":
        return gen_korobov_basis(korobov_a, latent_dim, n_points)
    if lattice_type == "roberts":
        return roberts_sequence(n_points, latent_dim)
    return gen_fib_basis(m=fib_m)


class QLVMTrainer:
    """
    Description
    -----------
    Trains the QLVM decoder on a ``build_qlvm_training_set`` ``.npz`` set and
    writes the torch checkpoint plus the decoder-weights ``.npz`` that the
    JAX inference path consumes.
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
        Initializes the QLVMTrainer.

        Parameters
        ----------
        dataset_directory (str)
            Directory holding the ``.npz`` training set (``train_data.npz`` and,
            unless full-dataset, ``val_data.npz``) from
            :mod:`build_qlvm_training_set`.
        output_directory (str)
            Directory to write the checkpoint + decoder-weights ``.npz`` (created
            if missing).
        input_parameter_dict (dict)
            Processing settings; the ``train_qlvm`` block supplies the lattice /
            model / optimization parameters.
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

    def _load_split(self, npz_path: pathlib.Path):
        """
        Description
        -----------
        Loads one ``.npz`` split into a torch ``TensorDataset`` of
        ``(spectrogram, masks_len)`` items, where each spectrogram is shaped
        ``(1, F, T)`` float32 (the channel axis the decoder/likelihood expect)
        and ``masks_len`` is carried as the per-item label (unused by the
        unconditional evidence objective, kept for parity/diagnostics).

        Parameters
        ----------
        npz_path (pathlib.Path)
            Path to a ``train_data.npz`` / ``val_data.npz`` / ``full_data.npz``.

        Returns
        -------
        dataset (torch.utils.data.TensorDataset)
            The spectrogram/label dataset.
        n_samples (int)
            Number of spectrograms in the split.
        """

        import torch  # noqa: PLC0415 (lazy: keeps torch off the import path)

        with np.load(npz_path, allow_pickle=True) as data:
            specs = data["spectrograms"].astype(np.float32)
            masks_len = data["masks_len"].astype(np.int64) if "masks_len" in data else np.zeros(specs.shape[0], dtype=np.int64)
        specs_tensor = torch.from_numpy(specs).unsqueeze(1).to(torch.float32)  # (N, 1, F, T)
        labels_tensor = torch.from_numpy(masks_len)
        return torch.utils.data.TensorDataset(specs_tensor, labels_tensor), int(specs.shape[0])

    def train(self) -> None:
        """
        Description
        -----------
        Reads the ``.npz`` training set, builds the lattice + decoder, trains the
        QLVM decoder under the Bernoulli evidence objective for ``n_epochs``
        (running validation evidence every ``val_freq`` epochs when a
        ``val_data.npz`` is present), and writes the checkpoint
        (``qmc_train_qlvm.tar``) and the decoder-weights bridge file
        (``qmc_decoder_weights.npz``) into the output directory.

        The full ``(N, 1, F, T)`` spectrogram stack is trained as-is; the lattice
        is given a fresh random torus shift every batch (the QMC trick). After
        training the decoder ``state_dict`` is dumped one array per layer so the
        torch-free JAX inference path can reload it without torch.

        Parameters
        ----------

        Returns
        -------
        ``qmc_train_qlvm.tar`` + ``qmc_decoder_weights.npz``
            Checkpoint and decoder-weights bridge written to the output directory.
        """

        self.message_output(
            f"QLVM training started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['train_qlvm']
        n_epochs = cfg['n_epochs']
        latent_dim = cfg['latent_dim']
        lattice_type = cfg['lattice_type']
        korobov_a = cfg['korobov_a']
        train_n_points = cfg['train_n_points']
        test_n_points = cfg['test_n_points']
        fib_m = cfg['fib_m']
        batch_size = cfg['batch_size']
        learning_rate = cfg['learning_rate']
        val_freq = cfg['val_freq']
        seed = cfg['seed']
        num_workers = cfg['num_workers']

        dataset_dir = pathlib.Path(self.dataset_directory)
        train_npz = dataset_dir / "train_data.npz"
        full_npz = dataset_dir / "full_data.npz"
        if train_npz.is_file():
            train_path = train_npz
        elif full_npz.is_file():
            train_path = full_npz
        else:
            error_message = (
                f"No training set found in {dataset_dir} (expected train_data.npz or full_data.npz). "
                f"Run build-qlvm-training-set first."
            )
            raise FileNotFoundError(error_message)
        val_path = dataset_dir / "val_data.npz"

        output_dir = pathlib.Path(self.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Heavy, torch-only kernels: import lazily so `import train_qlvm` and the
        # CLI plumbing stay torch-free until an actual training run is requested.
        import torch  # noqa: PLC0415 (lazy: keeps torch off the import path)
        from torch.optim import Adam  # noqa: PLC0415 (lazy)

        from .qlvm_training.checkpoint import save  # noqa: PLC0415 (lazy: pulls torch)
        from .qlvm_training.loop import test_epoch, train_epoch  # noqa: PLC0415 (lazy)
        from .qlvm_training.losses import binary_evidence  # noqa: PLC0415 (lazy)
        from .qlvm_training.qmc_base import QMCLVM, TorusBasis  # noqa: PLC0415 (lazy)

        torch.manual_seed(seed)
        np.random.seed(seed)

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        self.message_output(f"Training QLVM (latent_dim={latent_dim}, lattice={lattice_type}) on device: {device}.")

        train_dataset, n_train = self._load_split(train_path)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        self.message_output(f"Loaded {n_train} training spectrograms from {train_path.name}.")

        val_loader = None
        if val_path.is_file():
            val_dataset, n_val = self._load_split(val_path)
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
            self.message_output(f"Loaded {n_val} validation spectrograms from {val_path.name}.")

        decoder = build_qmc_decoder(latent_dim)
        model = QMCLVM(latent_dim=latent_dim, device=device, decoder=decoder, basis=TorusBasis())
        train_base_sequence = build_lattice(lattice_type, latent_dim, korobov_a, train_n_points, fib_m).to(device)
        test_base_sequence = build_lattice(lattice_type, latent_dim, korobov_a, test_n_points, fib_m).to(device)

        # binary_evidence(samples, data) with its defaults IS the training objective
        # (Bernoulli log-evidence over the lattice); pass it directly.
        loss_function = binary_evidence
        optimizer = Adam(model.parameters(), lr=learning_rate)

        all_losses: list[float] = []
        for epoch in range(n_epochs):
            batch_losses, model, optimizer = train_epoch(
                model, optimizer, train_loader, train_base_sequence, loss_function
            )
            all_losses += batch_losses

            if val_loader is not None and ((epoch + 1) % val_freq == 0 or epoch == n_epochs - 1):
                val_losses = test_epoch(model, val_loader, test_base_sequence, loss_function)
                self.message_output(
                    f"  Epoch {epoch + 1}/{n_epochs}: train evidence loss {np.mean(batch_losses):.4f}, "
                    f"val evidence loss {np.mean(val_losses):.4f}."
                )
            elif (epoch + 1) % val_freq == 0 or epoch == n_epochs - 1:
                self.message_output(
                    f"  Epoch {epoch + 1}/{n_epochs}: train evidence loss {np.mean(batch_losses):.4f}."
                )

        # Checkpoint (model + optimizer + loss trajectory) for resuming/diagnostics.
        checkpoint_path = output_dir / _CHECKPOINT_NAME
        save(model.to("cpu"), optimizer, all_losses, fn=str(checkpoint_path))

        # Decoder-weights bridge: one array per nn.Sequential layer (keys
        # "0.weight"/"0.bias"/.../"9.weight"), exactly what the JAX inference path
        # (analyses/qlvm_model.decoder_forward) reloads without torch.
        decoder_state = model.decoder.state_dict()
        weights_path = output_dir / _DECODER_WEIGHTS_NAME
        np.savez(
            weights_path,
            **{key: value.detach().cpu().numpy() for key, value in decoder_state.items()},
        )

        self.message_output(f"Wrote checkpoint -> {checkpoint_path} and decoder weights -> {weights_path}.")
        self.message_output(
            f"QLVM training ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="train-qlvm")
@click.option('--dataset-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Directory holding the .npz training set (build-qlvm-training-set output).')
@click.option('--output-directory', type=click.Path(file_okay=False, dir_okay=True), required=True, help='Directory to write the checkpoint + decoder-weights .npz.')
@click.option('--n-epochs', 'n_epochs', type=int, default=None, required=False, help='Number of training epochs.')
@click.option('--latent-dim', 'latent_dim', type=int, default=None, required=False, help='Torus latent dimensionality.')
@click.option('--lattice-type', 'lattice_type', type=click.Choice(['korobov', 'roberts', 'fib']), default=None, required=False, help='Quasi-random lattice generator.')
@click.option('--korobov-a', 'korobov_a', type=int, default=None, required=False, help='Korobov generating integer.')
@click.option('--batch-size', 'batch_size', type=int, default=None, required=False, help='Training batch size.')
@click.option('--learning-rate', 'learning_rate', type=float, default=None, required=False, help='Adam learning rate.')
@click.pass_context
def train_qlvm_cli(ctx, dataset_directory, output_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to train the QLVM decoder on a curated ``.npz`` training
    set and export the decoder weights for the JAX inference path.

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

    QLVMTrainer(
        dataset_directory=dataset_directory,
        output_directory=output_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).train()
