"""
@author: bartulem
HDF5 archive layer for inter-USV interval / mixture-model analyses.

Consolidates the five per-mode CSV side-effects of
:class:`compute_inter_usv_interval_distributions.InterUSVIntervalCalculator` into a single
self-describing HDF5 file, structured as::

    usv_interval_analysis_<YYYYMMDD>_<HHMMSS>.h5
    ├── /attrs                  analysis-level provenance (every JSON
    │                           parameter that drove the run, plus
    │                           created_at / git_sha / source_lists /
    │                           n_sessions_loaded)
    ├── /<mode>/                mode group (``s2s`` and/or ``e2s``)
    │   ├── /attrs              mode-level provenance
    │   │                       (alpha_effective, K_selected_male,
    │   │                        K_selected_female)
    │   ├── intervals           tidy one-row-per-inter-USV interval table
    │   ├── drop_counts         dropped non-positive intervals per sex
    │   ├── gmm_fits            full IC sweep (every K x every rep,
    │   │                       including per-component params -- this
    │   │                       table doubles as the model-parameter
    │   │                       store; pick the best rep at load time)
    │   ├── bootstrap_lrt       per-pair LRT summary plus per-sex
    │   │                       step-up-selected K
    │   └── bootstrap_lrt_null  long-form null draws used by the
    │                           bootstrap-LRT panel plot

The corresponding loaders return shapes identical to the in-memory
objects produced by the compute path, so plot helpers can run off-line
re-renders months after the data was generated without refitting.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import polars as pls

from ..os_utils import configure_path
from .gmm_utils import TMixture


# Internal helpers: polars <-> HDF5 dataset translation


def _polars_to_h5(group: h5py.Group, name: str, df: pls.DataFrame) -> None:
    """
    Description
    Writes a polars DataFrame as a single compound HDF5 dataset under
    ``group[name]``. Column dtypes are mapped to numpy / fixed-width
    string types; empty frames are preserved by writing a zero-length
    dataset with the schema's column names recorded as a JSON attribute
    (``schema_json``) so the loader can rebuild an empty frame with the
    right columns.

    Parameters
    group (h5py.Group)
        Destination group.
    name (str)
        Dataset name.
    df (pls.DataFrame)
        DataFrame to serialise. May be empty.

    Returns
    """

    schema = {col: str(dtype) for col, dtype in df.schema.items()}

    if df.height == 0:
        # Zero-length compound dataset isn't trivially constructable from
        # an empty polars frame, so we record the schema as JSON and
        # write an empty placeholder dataset; the reader uses
        # ``schema_json`` to rebuild the empty frame.
        ds = group.create_dataset(name, data=np.zeros((0,), dtype=np.uint8))
        ds.attrs["empty"] = True
        ds.attrs["schema_json"] = json.dumps(schema)
        return

    # Build a numpy structured array. Strings are encoded as variable-
    # length UTF-8 (h5py special_dtype) so we don't truncate session_id
    # or path strings of unknown length.
    str_dtype = h5py.string_dtype(encoding="utf-8")
    np_fields = []
    for col, dtype in df.schema.items():
        if dtype in (pls.Utf8,):
            np_fields.append((col, str_dtype))
        elif dtype in (pls.Int8, pls.Int16, pls.Int32, pls.Int64):
            np_fields.append((col, np.int64))
        elif dtype in (pls.UInt8, pls.UInt16, pls.UInt32, pls.UInt64):
            np_fields.append((col, np.uint64))
        elif dtype in (pls.Float32, pls.Float64):
            np_fields.append((col, np.float64))
        elif dtype == pls.Boolean:
            np_fields.append((col, np.bool_))
        else:
            # Fallback: stringify
            np_fields.append((col, str_dtype))

    arr = np.empty(df.height, dtype=np_fields)
    for col, _ in np_fields:
        col_data = df[col].to_list()
        arr[col] = col_data

    ds = group.create_dataset(name, data=arr, compression="gzip")
    ds.attrs["empty"] = False
    ds.attrs["schema_json"] = json.dumps(schema)


def _h5_to_polars(ds: h5py.Dataset) -> pls.DataFrame:
    """
    Description
    Reverse of :func:`_polars_to_h5`: rebuilds a polars DataFrame from
    a compound dataset, restoring the original schema recorded in the
    ``schema_json`` attribute.

    Parameters
    ds (h5py.Dataset)
        Source dataset previously written by :func:`_polars_to_h5`.

    Returns
    df (pls.DataFrame)
        Reconstructed DataFrame; empty if the original was empty.
    """

    schema_json = ds.attrs.get("schema_json", "{}")
    schema_raw = json.loads(schema_json) if isinstance(schema_json, str) else json.loads(schema_json.decode("utf-8"))

    # Map polars dtype string back to a polars dtype object so the
    # rebuilt empty frame has the same schema as the source frame.
    dtype_lookup = {
        "Utf8": pls.Utf8, "String": pls.Utf8,
        "Int8": pls.Int8, "Int16": pls.Int16, "Int32": pls.Int32, "Int64": pls.Int64,
        "UInt8": pls.UInt8, "UInt16": pls.UInt16, "UInt32": pls.UInt32, "UInt64": pls.UInt64,
        "Float32": pls.Float32, "Float64": pls.Float64,
        "Boolean": pls.Boolean,
    }
    schema = {col: dtype_lookup.get(dtype_str, pls.Utf8) for col, dtype_str in schema_raw.items()}

    is_empty = bool(ds.attrs.get("empty", False))
    if is_empty:
        return pls.DataFrame(schema=schema)

    arr = ds[()]
    cols: dict[str, list] = {}
    for col, _ in arr.dtype.descr:
        col_data = arr[col]
        if col_data.dtype.kind == "O":  # variable-length strings stored as object
            cols[col] = [
                v.decode("utf-8") if isinstance(v, (bytes, bytearray)) else str(v)
                for v in col_data.tolist()
            ]
        else:
            cols[col] = col_data.tolist()

    return pls.DataFrame(cols, schema=schema)


def _try_git_sha(repo_root: Path) -> str:
    """
    Description
    Best-effort short git SHA at ``repo_root``; returns the literal
    string ``"unknown"`` when the directory is not a git checkout, the
    ``git`` binary is absent, or any other invocation error occurs.
    Pure provenance metadata; no functional consequences if it fails.

    Parameters
    repo_root (pathlib.Path)
        Directory to interrogate.

    Returns
    sha (str)
        Short SHA, or ``"unknown"``.
    """

    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=2.0, check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "unknown"


# Public writer


def write_ivi_h5(
    h5_path: str | Path,
    *,
    analysis_attrs: dict,
    per_mode: dict[str, dict[str, Any]],
) -> Path:
    """
    Description
    Writes a single inter-USV interval-analysis HDF5 file consolidating every artifact
    that was previously distributed across five CSVs per mode.

    The caller is responsible for assembling the ``per_mode`` dict
    keyed on interval-type strings (``"s2s"`` / ``"e2s"``); modes for
    which the compute path produced no data should simply be omitted
    from the dict, in which case the corresponding HDF5 group is not
    created.

    Parameters
    h5_path (str | Path)
        Output file path. The parent directory is created if missing.
        The path is run through :func:`configure_path` for cross-OS
        path-prefix translation.
    analysis_attrs (dict)
        Provenance attributes attached to the file root. Should include
        every JSON parameter that drove the run (so a re-render months
        later is fully self-describing) plus ``created_at_iso``,
        ``git_sha``, ``source_lists`` (list of resolved paths), and
        ``n_sessions_loaded``.
    per_mode (dict)
        Mapping ``mode -> {'attrs': {...}, 'intervals': pls.DataFrame,
        'drop_counts': pls.DataFrame, 'gmm_fits': pls.DataFrame |
        None, 'bootstrap_lrt': pls.DataFrame | None,
        'bootstrap_lrt_null': pls.DataFrame | None}``. Tables that the
        compute path skipped (e.g. when ``fit_gmm`` is false) may be
        ``None`` and are then not written.

    Returns
    out_path (pathlib.Path)
        Resolved path of the written file.
    """

    out_path = Path(configure_path(str(h5_path)))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as f:
        for k, v in analysis_attrs.items():
            f.attrs[k] = _attr_value(v)

        for mode, payload in per_mode.items():
            grp = f.create_group(mode)
            for k, v in payload.get("attrs", {}).items():
                grp.attrs[k] = _attr_value(v)

            tables = (
                "intervals",
                "drop_counts",
                "gmm_fits",
                "bootstrap_lrt",
                "bootstrap_lrt_null",
            )
            for table_name in tables:
                df = payload.get(table_name)
                if df is None:
                    continue
                _polars_to_h5(grp, table_name, df)

    return out_path


def _attr_value(v: Any) -> Any:
    """
    Description
    Coerces a Python value into something h5py can store as an
    attribute. Lists / dicts are JSON-encoded; ``None`` becomes the
    literal string ``"null"`` (h5py does not accept Python ``None``
    as an attribute value). Scalars and numpy scalars pass through.

    Parameters
    v (Any)
        Value to coerce.

    Returns
    coerced (Any)
        h5py-storable representation of ``v``.
    """

    if v is None:
        return "null"
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v)
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float, str, np.integer, np.floating, np.bool_)):
        return v
    return json.dumps(v, default=str)


# Public reader


def read_usv_interval_h5(h5_path: str | Path) -> dict:
    """
    Description
    Reads an HDF5 archive previously written by :func:`write_ivi_h5`
    and returns its contents as a structured dict::

        {
          "attrs": {...},
          "modes": {
            "<mode>": {
              "attrs": {...},
              "intervals": pls.DataFrame,
              "drop_counts": pls.DataFrame,
              "gmm_fits": pls.DataFrame | None,
              "bootstrap_lrt": pls.DataFrame | None,
              "bootstrap_lrt_null": pls.DataFrame | None,
            },
            ...
          }
        }

    JSON-encoded attribute values (lists / dicts written by the writer)
    are re-decoded into Python objects so the caller doesn't have to
    know which attributes were structured.

    Parameters
    h5_path (str | Path)
        Path to the HDF5 file.

    Returns
    archive (dict)
        Nested dict described above.
    """

    in_path = Path(configure_path(str(h5_path)))
    out: dict = {"attrs": {}, "modes": {}}

    with h5py.File(in_path, "r") as f:
        for k, v in f.attrs.items():
            out["attrs"][k] = _decode_attr(v)

        for mode_key in f.keys():
            grp = f[mode_key]
            mode_payload: dict = {"attrs": {}}
            for k, v in grp.attrs.items():
                mode_payload["attrs"][k] = _decode_attr(v)

            for table_name in (
                "intervals",
                "drop_counts",
                "gmm_fits",
                "bootstrap_lrt",
                "bootstrap_lrt_null",
            ):
                if table_name in grp:
                    mode_payload[table_name] = _h5_to_polars(grp[table_name])
                else:
                    mode_payload[table_name] = None

            out["modes"][mode_key] = mode_payload

    return out


def _decode_attr(v: Any) -> Any:
    """
    Description
    Reverse of :func:`_attr_value`: tries to JSON-decode string
    attributes; passes through scalars unchanged. The literal string
    ``"null"`` is restored to Python ``None``.

    Parameters
    v (Any)
        Raw attribute value as returned by h5py.

    Returns
    decoded (Any)
        Python-native value.
    """

    if isinstance(v, bytes):
        v = v.decode("utf-8")
    if isinstance(v, str):
        if v == "null":
            return None
        # Best-effort JSON decode; non-JSON strings (most attrs) pass through.
        try:
            return json.loads(v)
        except (ValueError, TypeError):
            return v
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# Best-rep model reconstruction


def reconstruct_best_model(
    gmm_fits: pls.DataFrame,
    sex: str,
    K: int,
    ic_col: str = "cv_neg_loglik",
):
    """
    Description
    Picks the lowest-IC rep at ``(sex, K)`` from a sweep DataFrame and
    rebuilds the fitted mixture model from the per-component parameters
    archived in that row, without re-running EM. Returns a
    ``(model, gmm_order)`` pair shaped exactly like the live compute
    path returns, so any downstream plot helper consumes it without
    modification.

    For Gaussian mixtures, an ``sklearn.mixture.GaussianMixture`` is
    instantiated with ``weights_``, ``means_`` and ``covariances_`` set
    from the row, plus a freshly computed ``precisions_cholesky_``
    (1-D analytic form: ``1 / sqrt(cov)``) so ``score_samples`` works
    immediately. For Student-t mixtures, a :class:`gmm_utils.TMixture`
    is constructed directly.

    Components are returned in ascending log-mean order, matching the
    ``model_order`` convention used elsewhere; the reconstructed model
    is "pre-sorted" (its component slots are already in canonical
    order) so the returned ``gmm_order`` is simply ``arange(K)``.

    Parameters
    gmm_fits (pls.DataFrame)
        Sweep DataFrame as written to ``gmm_fits``. Must contain the
        per-component columns ``logmean_k``, ``logsd_k``, ``weight_k``
        and ``nu_k`` for ``k=1..K`` (NaN-padded for k > K), plus a
        ``model_class`` column whose unique value within the
        ``(sex, K)`` slice is one of ``"gauss"`` / ``"t"``.
    sex (str)
        Filter value (``"male"`` / ``"female"``).
    K (int)
        Number of components for the row to extract.
    ic_col (str)
        Information criterion used to pick the "best" rep within
        ``(sex, K)``. Defaults to ``"cv_neg_loglik"``; falls back to
        ``"bic"`` when CV values are NaN (e.g. tiny samples).

    Returns
    model (GaussianMixture | TMixture)
        Reconstructed mixture; ``score_samples`` etc. work directly.
    gmm_order (np.ndarray)
        ``np.arange(K)`` -- the model is pre-sorted by ascending
        log-mean.
    """

    sub = gmm_fits.filter(
        (pls.col("sex") == sex) & (pls.col("n_comp") == int(K))
    )
    if sub.height == 0:
        msg = f"reconstruct_best_model: no rows for sex={sex!r}, K={K}."
        raise ValueError(
            msg
        )

    chosen_ic = ic_col
    # Fall back if every CV value is NaN (small-sample CV-skip path).
    if chosen_ic == "cv_neg_loglik":
        cv_vals = sub["cv_neg_loglik"].to_numpy()
        if np.all(np.isnan(cv_vals)):
            chosen_ic = "bic"

    best = sub.sort(chosen_ic, nulls_last=True).head(1).row(0, named=True)
    model_class = str(best["model_class"]).strip()

    weights = np.array(
        [float(best[f"weight_{k+1}"]) for k in range(K)], dtype=float
    )
    means = np.array(
        [float(best[f"logmean_{k+1}"]) for k in range(K)], dtype=float
    )
    logsds = np.array(
        [float(best[f"logsd_{k+1}"]) for k in range(K)], dtype=float
    )
    covs = (logsds ** 2)

    if model_class == "gauss":
        # Lazy import: keeps sklearn off the import-time critical path
        # for callers that only ever load t-mixture archives.
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(n_components=K, covariance_type="full")
        gmm.weights_ = weights
        gmm.means_ = means.reshape(K, 1)
        gmm.covariances_ = covs.reshape(K, 1, 1)
        # 1-D precisions_cholesky_: (1 / sqrt(cov)) for each component,
        # shape (K, 1, 1). sklearn's scoring routines require this
        # attribute even when covariances_ is fully specified.
        gmm.precisions_cholesky_ = (1.0 / np.sqrt(covs)).reshape(K, 1, 1)
        return gmm, np.arange(K)

    if model_class == "t":
        nus = np.array(
            [float(best[f"nu_{k+1}"]) for k in range(K)], dtype=float
        )
        tmix = TMixture(
            weights=weights, means=means, covariances=covs, nus=nus,
        )
        return tmix, np.arange(K)

    msg = (
        f"reconstruct_best_model: unknown model_class={model_class!r} "
        "(expected 'gauss' or 't')."
    )
    raise ValueError(
        msg
    )


def detect_repo_root_for_provenance(start: Path | str) -> Path:
    """
    Description
    Walks upward from ``start`` looking for a ``.git`` directory; falls
    back to ``start`` itself when no enclosing repo is found. Used by
    the writer to populate the ``git_sha`` provenance attribute without
    requiring callers to know where the repository root lives.

    Parameters
    start (pathlib.Path | str)
        Directory to start the upward search from.

    Returns
    repo_root (pathlib.Path)
        The first ancestor containing ``.git``; otherwise ``start``.
    """

    p = Path(start).resolve()
    for cand in (p, *p.parents):
        if (cand / ".git").exists():
            return cand
    return p


def git_sha_for_provenance(start: Path | str) -> str:
    """
    Description
    Convenience wrapper combining :func:`detect_repo_root_for_provenance`
    and :func:`_try_git_sha` so callers can read a single short SHA
    string with one call.

    Parameters
    start (pathlib.Path | str)
        Any directory inside (or near) the repo to identify.

    Returns
    sha (str)
        Short SHA or ``"unknown"``.
    """

    return _try_git_sha(detect_repo_root_for_provenance(start))
