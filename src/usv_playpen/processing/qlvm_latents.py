"""
@author: bartulem
Embed a session's USV spectrograms into the trained QLVM toroidal latent space
and merge the latent coordinates + watershed categories into its
``*_usv_summary.csv``.

This is the in-house, JAX (torch-free) inference driver. It loads the frozen
decoder weights (a ``.npz`` converted once from the training checkpoint's
``state_dict``), rebuilds the fixed lattice, embeds the session's spectrograms
via :func:`qlvm_model.embed_data`, and assigns each USV a cluster by **spatial
lookup into two fixed reference watershed grids** — a FINE grid and a COARSE grid
(the torus-periodic ``ws_labels_periodic`` field of a fine and a coarse reference
``arrays.npz``) — NOT a per-session re-watershed, so clusters are comparable
across every session embedded into the same torus.

Columns written into ``usv_summary.csv`` (the ones the visualizations/tuning
code already consume): ``qlvm_dim1``, ``qlvm_dim2`` (torus coordinates),
``qlvm_category`` (FINE cluster label, e.g. 12 classes) and ``qlvm_supercategory``
(COARSE cluster label, e.g. 7 classes; 0 = background/noise).

Fidelity: the session spectrograms are preprocessed with the SAME resize /
time-stretch used to build the training set (:func:`stretch_specs`), so they are
in-distribution for the decoder.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from datetime import datetime

import click
import h5py
import jax.numpy as jnp
import numpy as np
import polars as pls
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import configure_path, first_match_or_raise
from ..processing.build_qlvm_training_set import stretch_specs
from ..time_utils import is_gui_context, smart_wait
from .qlvm_model import embed_data, gen_fib_basis, gen_korobov_basis, roberts_sequence

# QLVM columns written into the USV summary CSV (consumed downstream).
QLVM_COLUMNS = ("qlvm_dim1", "qlvm_dim2", "qlvm_category", "qlvm_supercategory")


def load_decoder_params(weights_npz_path: str) -> dict[str, jnp.ndarray]:
    """
    Description
    -----------
    Loads the frozen decoder weights from the converted ``.npz`` (one array per
    ``state_dict`` entry) into the key form :func:`qlvm_model.decoder_forward`
    expects. A leading ``decoder.`` prefix (present when the converter dumps the
    full QMCLVM ``state_dict``) is stripped.

    Parameters
    ----------
    weights_npz_path (str)
        Path to the ``.npz`` of decoder weights.

    Returns
    -------
    params (dict[str, jnp.ndarray])
        Decoder weights keyed by ``"<layer_idx>.weight"`` / ``"<layer_idx>.bias"``.
    """
    raw = np.load(configure_path(weights_npz_path))
    params: dict[str, jnp.ndarray] = {}
    for key in raw.files:
        clean = key[len("decoder."):] if key.startswith("decoder.") else key
        params[clean] = jnp.asarray(raw[key])
    return params


def build_lattice(cfg: dict) -> jnp.ndarray:
    """
    Description
    -----------
    Rebuilds the fixed QLVM lattice from the training configuration so inference
    uses the exact grid the model was trained on.

    Parameters
    ----------
    cfg (dict)
        The ``infer_qlvm_latents`` settings block (``lattice_type``,
        ``latent_dim``, ``n_points``, ``korobov_a``, ``fib_m``).

    Returns
    -------
    lattice (jnp.ndarray)
        Lattice points, shape ``(n_points, latent_dim)``.
    """
    lattice_type = cfg["lattice_type"]
    if lattice_type == "korobov":
        return gen_korobov_basis(cfg["korobov_a"], cfg["latent_dim"], cfg["n_points"])
    if lattice_type == "roberts":
        return roberts_sequence(cfg["n_points"], cfg["latent_dim"])
    if lattice_type == "fibonacci":
        return gen_fib_basis(cfg["fib_m"])
    msg = f"build_lattice: unknown lattice_type {lattice_type!r} (expected korobov|roberts|fibonacci)."
    raise ValueError(msg)


def labels_for_coords(
    coords: np.ndarray,
    fine_grid: np.ndarray,
    coarse_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Description
    -----------
    Looks up each torus coordinate's cluster label in the FINE and COARSE
    reference watershed grids, matching ``inference_latents.py``'s convention
    ``label = grid[int(y * res), int(x * res)]`` (with clipping to each grid's
    resolution). Each grid is the torus-periodic ``ws_labels_periodic`` field of
    its reference ``arrays.npz`` (periodic = correct for the native-torus QLVM
    coordinates, which wrap at the seam). The fine grid yields the per-USV
    ``qlvm_category`` (e.g. 12 clusters); the coarse grid yields the broader
    ``qlvm_supercategory`` (e.g. 7 clusters).

    Parameters
    ----------
    coords (np.ndarray)
        Torus coordinates in ``[0, 1)``, shape ``(N, 2)`` ordered ``(x, y)``.
    fine_grid (np.ndarray)
        Fine-granularity periodic watershed label grid, shape ``(res, res)``.
    coarse_grid (np.ndarray)
        Coarse-granularity periodic watershed label grid, shape ``(res, res)``.

    Returns
    -------
    category (np.ndarray)
        Fine cluster labels, shape ``(N,)``.
    supercategory (np.ndarray)
        Coarse cluster labels, shape ``(N,)``.
    """

    def _lookup(grid: np.ndarray) -> np.ndarray:
        res = grid.shape[0]
        px = np.clip((coords[:, 0] * res).astype(int), 0, res - 1)
        py = np.clip((coords[:, 1] * res).astype(int), 0, res - 1)
        return grid[py, px]

    return _lookup(fine_grid), _lookup(coarse_grid)


class QLVMLatentInference:
    """
    Description
    -----------
    Embeds one session's spectrograms into the trained QLVM torus and merges the
    coordinates + watershed categories into its ``*_usv_summary.csv``.
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
        Initializes the QLVMLatentInference.

        Parameters
        ----------
        root_directory (str)
            Session root directory (contains the ``audio`` tree).
        input_parameter_dict (dict)
            Processing settings; the ``infer_qlvm_latents`` block supplies the
            weight/reference paths and lattice configuration.
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

    def infer_and_merge(self) -> None:
        """
        Description
        -----------
        Loads the decoder weights, lattice and reference watershed grids; reads
        the session spectrogram H5, preprocesses identically to training, embeds
        into the torus, assigns categories by reference lookup, and merges
        ``qlvm_*`` columns into the matching USV summary rows (joined on the
        per-USV index in each ``spec_id``; USVs absent from the H5 get nulls;
        any pre-existing ``qlvm_*`` columns are replaced).

        Parameters
        ----------

        Returns
        -------
        Updated ``*_usv_summary.csv`` with the ``qlvm_*`` columns.
        """
        self.message_output(
            f"QLVM latent inference started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        cfg = self.input_parameter_dict['infer_qlvm_latents']
        params = load_decoder_params(cfg['weights_npz_path'])
        lattice = build_lattice(cfg)

        # Fine grid -> qlvm_category; coarse grid -> qlvm_supercategory. Both are
        # the torus-periodic watershed (ws_labels_periodic) of their reference file.
        fine_ref = np.load(configure_path(cfg['reference_arrays_fine_npz_path']))
        coarse_ref = np.load(configure_path(cfg['reference_arrays_coarse_npz_path']))
        fine_grid = fine_ref['ws_labels_periodic']
        coarse_grid = coarse_ref['ws_labels_periodic']

        root = pathlib.Path(self.root_directory)
        h5_loc = first_match_or_raise(
            root=root / "audio" / "spectrograms",
            pattern="*_spectrograms.h5",
            label="per-session spectrogram H5",
        )
        with h5py.File(h5_loc, "r") as h5_file:
            session_group = h5_file[f"spectrogram/{root.name}"]
            specs = session_group["spectrograms"][:]
            durations = session_group["durations"][:]
        # spectrogram rows are 1:1 with usv_summary.csv; embed only the real
        # (duration > 0) USVs and remember their row positions for the merge.
        usv_indices = np.flatnonzero(durations > 0).astype(np.uint32)
        specs = specs[usv_indices]
        durations = durations[usv_indices]

        # Preprocess identically to the training set (same resize/time-stretch).
        resized = stretch_specs(specs.astype(np.float32), durations, (128, 128), cfg['time_stretch'])
        data = jnp.asarray(resized[:, None, :, :])

        coords = np.asarray(embed_data(lattice, data, params))           # (N, 2)
        category, supercategory = labels_for_coords(coords, fine_grid, coarse_grid)

        qlvm_df = pls.DataFrame({
            "_usv_row": usv_indices,
            "qlvm_dim1": coords[:, 0].astype(np.float64),
            "qlvm_dim2": coords[:, 1].astype(np.float64),
            "qlvm_category": category.astype(np.int64),
            "qlvm_supercategory": supercategory.astype(np.int64),
        })

        usv_summary_loc = first_match_or_raise(
            root=root / "audio",
            pattern="*_usv_summary.csv",
            recursive=True,
            label="USV summary CSV",
        )
        usv_df = pls.read_csv(source=str(usv_summary_loc))
        usv_df = usv_df.drop([c for c in QLVM_COLUMNS if c in usv_df.columns])
        usv_df = usv_df.with_row_index(name="_usv_row")
        merged = usv_df.join(qlvm_df, on="_usv_row", how="left").drop("_usv_row")
        merged.write_csv(file=str(usv_summary_loc))

        self.message_output(
            f"Merged QLVM latents/categories for {len(usv_indices)} USVs into {usv_summary_loc.name}."
        )
        self.message_output(
            f"QLVM latent inference ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="infer-qlvm-latents")
@click.option('--root-directory', type=click.Path(exists=True, file_okay=False, dir_okay=True), required=True, help='Session root directory path.')
@click.option('--weights-npz-path', 'weights_npz_path', type=str, default=None, required=False, help='Path to the converted decoder weights .npz.')
@click.option('--reference-arrays-fine-npz-path', 'reference_arrays_fine_npz_path', type=str, default=None, required=False, help='Path to the FINE reference arrays.npz (ws_labels_periodic -> qlvm_category).')
@click.option('--reference-arrays-coarse-npz-path', 'reference_arrays_coarse_npz_path', type=str, default=None, required=False, help='Path to the COARSE reference arrays.npz (ws_labels_periodic -> qlvm_supercategory).')
@click.pass_context
def infer_qlvm_latents_cli(ctx, root_directory, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to embed a session's USV spectrograms into the QLVM
    torus and merge the latents/categories into its USV summary CSV.

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

    QLVMLatentInference(
        root_directory=root_directory,
        input_parameter_dict=processing_settings_dict,
        message_output=print,
    ).infer_and_merge()
