"""
@author: bartulem
Consolidate per-session spectrogram/mask H5 files into one multi-session store.

Each session's ``audio/spectrograms/<session>_spectrograms.h5`` (written by
``generate-usv-spectrograms`` + ``generate-usv-masks``) already uses the
consolidated-store layout -- a shared top-level ``frequency_bins`` axis plus
per-session ``spectrogram/<session>`` and ``mask/<session>`` groups -- so
consolidation is a validated group-copy over a list of sessions, plus a
per-session ``qlvm_dim`` (n, 2) dataset injected from the session's
``*_usv_summary.csv`` ``qlvm1``/``qlvm2`` columns (the latent coordinates the
torus-traversal video and the embedding explorer read).

The output is written to ``spectrograms_root`` as
``spectrograms_sam2masks_<S>sessions_<N>vocalizations_<UTC timestamp>.h5``;
``os_utils.resolve_consolidated_h5_path`` picks the newest such store, so a
fresh consolidation activates on the next read with no configuration change.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from datetime import UTC, datetime

import click
import h5py
import numpy as np
import polars as pls
from click.core import ParameterSource

from ..cli_utils import modify_settings_json_for_cli
from ..os_utils import atomic_output_path, first_match_or_raise
from ..time_utils import is_gui_context, smart_wait


class SpectrogramStoreConsolidator:

    def __init__(self,
                 root_directories: list[str] | None = None,
                 input_parameter_dict: dict | None = None,
                 message_output: Callable = print) -> None:
        """
        Description
        -----------
        Initializes the SpectrogramStoreConsolidator class.

        Parameters
        ----------
        root_directories (list[str])
            Session root directories whose per-session spectrogram H5 files are
            consolidated, in the order they should appear in the store.
        input_parameter_dict (dict)
            Processing settings; ``spectrograms_root`` names the output
            directory the store is written to.
        message_output (Callable)
            Defines output messages; defaults to ``print``.

        Returns
        -------
        None
        """

        self.root_directories = root_directories if root_directories is not None else []
        self.input_parameter_dict = input_parameter_dict if input_parameter_dict is not None else {}
        self.message_output = message_output
        self.app_context_bool = is_gui_context()

    def consolidate_spectrogram_store(self) -> None:
        """
        Description
        -----------
        Copies every session's ``spectrogram/<session>`` and ``mask/<session>``
        groups into one multi-session store under ``spectrograms_root``,
        validating that all sessions share one ``frequency_bins`` axis and that
        each session's spectrogram row count matches its ``*_usv_summary.csv``
        row count. A per-session ``spectrogram/<session>/qlvm_dim`` (n, 2)
        dataset is injected from the summary's ``qlvm1``/``qlvm2`` columns when
        both are present and non-null; sessions without QLVM latents (e.g.
        playback) are copied without it. Sessions missing the per-session H5
        are reported and skipped. File-level attrs record ``n_sessions``,
        ``n_vocalizations``, ``created_by`` and ``created_date``.

        Parameters
        ----------

        Returns
        -------
        .h5 consolidated store
            One ``spectrograms_sam2masks_<S>sessions_<N>vocalizations_<ts>.h5``
            under ``spectrograms_root``, written atomically.
        """

        self.message_output(
            f"Spectrogram-store consolidation started at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )
        smart_wait(app_context_bool=self.app_context_bool, seconds=1)

        spectrograms_root = pathlib.Path(self.input_parameter_dict['spectrograms_root'])
        if not spectrograms_root.is_dir():
            err_msg = f"spectrograms_root directory '{spectrograms_root}' does not exist."
            raise FileNotFoundError(err_msg)

        session_entries = []
        frequency_bins = None
        n_vocalizations_total = 0
        for one_root in self.root_directories:
            root = pathlib.Path(one_root)
            session_id = root.name
            h5_candidates = sorted((root / "audio" / "spectrograms").glob(f"{session_id}_spectrograms.h5"))
            if not h5_candidates:
                self.message_output(
                    f"Skipping {session_id}: no per-session spectrograms H5 found."
                )
                continue
            summary_path = first_match_or_raise(
                root=root / "audio",
                pattern="*_usv_summary.csv",
                label="USV summary CSV",
            )
            summary_df = pls.read_csv(source=str(summary_path))
            with h5py.File(h5_candidates[0], mode="r") as session_h5:
                session_bins = session_h5["frequency_bins"][:]
                n_spectrograms = session_h5[f"spectrogram/{session_id}/spectrograms"].shape[0]
            if frequency_bins is None:
                frequency_bins = session_bins
            elif not np.allclose(frequency_bins, session_bins):
                err_msg = (
                    f"Session '{session_id}' has a frequency_bins axis that differs from the "
                    f"store's shared axis; all sessions must share one spectrogram frequency grid."
                )
                raise ValueError(err_msg)
            if n_spectrograms != summary_df.height:
                err_msg = (
                    f"Session '{session_id}' is inconsistent: {n_spectrograms} spectrogram rows "
                    f"but {summary_df.height} USV summary rows; regenerate its spectrograms."
                )
                raise ValueError(err_msg)
            qlvm_coords = None
            if 'qlvm1' in summary_df.columns and 'qlvm2' in summary_df.columns:
                qlvm1 = summary_df['qlvm1'].to_numpy()
                qlvm2 = summary_df['qlvm2'].to_numpy()
                if summary_df.height > 0 and not (np.all(np.isnan(qlvm1.astype(np.float64))) or np.all(np.isnan(qlvm2.astype(np.float64)))):
                    qlvm_coords = np.column_stack([qlvm1, qlvm2]).astype(np.float64)
            session_entries.append((session_id, h5_candidates[0], qlvm_coords))
            n_vocalizations_total += n_spectrograms

        if not session_entries:
            self.message_output("No sessions with per-session spectrogram H5 files; nothing to consolidate.")
            return

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%SZ")
        store_name = (
            f"spectrograms_sam2masks_{len(session_entries)}sessions_"
            f"{n_vocalizations_total}vocalizations_{timestamp}.h5"
        )
        store_path = spectrograms_root / store_name

        with atomic_output_path(store_path) as tmp_store_path, h5py.File(tmp_store_path, mode="w") as store_h5:
            store_h5.attrs["created_by"] = "consolidate-spectrogram-store"
            store_h5.attrs["created_date"] = datetime.now(UTC).isoformat()
            store_h5.attrs["n_sessions"] = len(session_entries)
            store_h5.attrs["n_vocalizations"] = n_vocalizations_total
            store_h5.create_dataset("frequency_bins", data=frequency_bins, compression="gzip", compression_opts=6)
            for session_id, session_h5_path, qlvm_coords in session_entries:
                with h5py.File(session_h5_path, mode="r") as session_h5:
                    session_h5.copy(f"spectrogram/{session_id}", store_h5, name=f"spectrogram/{session_id}")
                    if f"mask/{session_id}" in session_h5:
                        session_h5.copy(f"mask/{session_id}", store_h5, name=f"mask/{session_id}")
                    else:
                        self.message_output(
                            f"Session '{session_id}' has no mask group; copied without masks."
                        )
                if qlvm_coords is not None:
                    store_h5[f"spectrogram/{session_id}"].create_dataset(
                        "qlvm_dim", data=qlvm_coords, compression="gzip", compression_opts=6
                    )
                self.message_output(f"Consolidated session '{session_id}'.")

        self.message_output(
            f"Consolidated {len(session_entries)} sessions / {n_vocalizations_total} vocalizations -> {store_path}."
        )
        self.message_output(
            f"Spectrogram-store consolidation ended at: {datetime.now().hour:02d}:{datetime.now().minute:02d}:{datetime.now().second:02d}."
        )


@click.command(name="consolidate-spectrogram-store")
@click.option('--root-directories', type=str, required=True, help='Comma-separated string of session root directory paths, in store order.')
@click.option('--spectrograms-root', 'spectrograms_root', type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None, required=False, help='Output directory the consolidated store is written to.')
@click.pass_context
def consolidate_spectrogram_store_cli(ctx, root_directories, **kwargs) -> None:
    """
    Description
    -----------
    A command-line tool to consolidate per-session spectrogram/mask H5 files
    into one multi-session store (with per-session ``qlvm_dim`` latents).

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
        settings_dict='processing_settings'
    )

    all_paths = [one_dir.strip() for one_dir in root_directories.split(',')]
    valid_dirs = [one_path for one_path in all_paths if pathlib.Path(one_path).is_dir()]

    SpectrogramStoreConsolidator(
        root_directories=valid_dirs,
        input_parameter_dict=processing_settings_dict,
        message_output=print
    ).consolidate_spectrogram_store()
