"""
One-shot recovery utility for consolidated bout-onset selection
artifacts produced BEFORE the per-fold-final-refit fix on the
``model`` branch (commit ``56efb08``).

Such pickles carry a single-entry ``filter_shapes`` list because the
pre-fix selector collapsed the final refit to one global-balanced
GAM fit. Downstream the plotter's percentile band over CV folds
needs >= 2 fold samples to render anything visible -- a single
entry produces a zero-width band on top of the line and looks like
"no error bars".

This module re-creates the original run's CV-fold partitioning
(reading every seed / strategy knob from the pickle's
``_run_metadata`` and ``_input_metadata`` blocks), refits the final
multivariate GAM **once per CV fold** for the already-chosen
feature set, and writes the multi-fold ``filter_shapes`` list back
into the pickle's last step.

The 6-day selection itself is NOT re-run. Only the post-selection
visualisation refit -- one GAM fit per CV fold for the final feature
set -- is recomputed (~minutes per fit; total wall time scales with
``n_splits_selection`` rather than candidates × steps × splits).

By construction this calls
``model_selection.compute_filter_shapes_per_fold_bout_onset``, the
same helper the production selector now uses, so the recovered
``filter_shapes`` is byte-for-byte equivalent to what a fresh
re-run on the fixed code would have produced (given the
metadata-recorded random seed).

Backups
-------
The original pickle is moved to
``<basename>.pre_filter_shape_fix.bak`` before the in-place write,
so the recovery is reversible. Calling the utility a second time
detects the existing ``.bak`` and refuses to overwrite it.

Usage
-----
Programmatic::

    from usv_playpen.modeling.recompute_filter_shapes import recompute_filter_shapes
    recompute_filter_shapes(
        '/.../selection_male_intact_partners_onsets_bout_mixed_20260511_203829Z.pkl',
    )

CLI::

    recompute-filter-shapes /.../selection_*.pkl
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from .load_input_files import load_pickle_modeling_data
from .model_selection import (
    compute_filter_shapes_per_fold_bout_onset,
    pool_session_arrays,
)
from .modeling_metadata import RESERVED_METADATA_KEYS
from ..os_utils import configure_path


# Keys the recovery needs to find in ``_run_metadata`` /
# ``_input_metadata`` to reproduce the original CV-fold partitioning
# bit-for-bit. Missing any of these aborts the recovery with a
# diagnostic naming the missing key, so we never silently use a
# different fold partition than the selection actually saw.
REQUIRED_RUN_META = (
    'random_seed', 'split_strategy', 'n_splits_selection',
    'test_proportion', 'gam_kwargs', 'anchor_feature',
)
REQUIRED_EXTRA_KNOBS = ('n_splines_time', 'n_splines_value')
REQUIRED_INPUT_META = ('filter_history_frames', 'session_ids')
BACKUP_SUFFIX = '.pre_filter_shape_fix.bak'


def _bounded_test_proportion(requested: float, n: int) -> float:
    """
    Replicate the simple safety clamp the selector applies to the
    session-split ``test_size`` so we don't depend on importing a
    private helper.
    """

    if n < 2:
        msg = f"need >= 2 sessions to split; got {n}."
        raise ValueError(msg)
    lo = 1.0 / n
    hi = (n - 1) / n
    return max(min(requested, hi), lo)


def _require_keys(block: dict, keys, block_name: str, missing_out: list) -> None:
    """Append a dotted missing-key path to ``missing_out`` per absent key."""

    for k in keys:
        if k not in block:
            missing_out.append(f"{block_name}.{k}")


def _validate_metadata(run_md: dict, input_md: dict) -> None:
    """
    Description
    -----------
    Verify every metadata key needed for exact CV-fold reproduction
    is present in the consolidated pickle. Raise ``ValueError``
    with a dotted-path list naming every missing key so the user
    can decide whether to populate them by hand or accept that
    exact reproduction isn't possible for this artifact.

    Parameters
    ----------
    run_md (dict)
        The ``_run_metadata`` block hoisted by the consolidator.
    input_md (dict)
        The ``_input_metadata`` block hoisted by the consolidator.

    Returns
    -------
    None
        Returns ``None`` on success; raises ``ValueError`` on
        missing required keys.
    """

    missing: list[str] = []
    _require_keys(run_md, REQUIRED_RUN_META, '_run_metadata', missing)
    extra = run_md.get('extra_knobs', {})
    _require_keys(extra, REQUIRED_EXTRA_KNOBS, '_run_metadata.extra_knobs', missing)
    _require_keys(input_md, REQUIRED_INPUT_META, '_input_metadata', missing)
    if missing:
        msg = (
            "Cannot exactly reproduce CV-fold partitioning: required "
            "metadata keys missing from the consolidated pickle:\n  "
            + "\n  ".join(missing)
            + "\n\n"
            "Exact reproduction is required by this utility so the "
            "recomputed filter shapes correspond to the train data each "
            "CV fold actually saw during the original selection."
        )
        raise ValueError(msg)


def _resolve_local_path(recorded_path: str) -> Path:
    """
    Run a path recorded under one OS's mount convention through
    ``configure_path`` so it resolves on the current machine.
    Errors clearly if the resolved file is not present, since the
    recovery cannot proceed without the input feature data.
    """

    resolved = Path(configure_path(recorded_path))
    if not resolved.exists():
        msg = (
            f"recorded path does not resolve on this machine: {recorded_path}\n"
            f"  (tried: {resolved})\n"
            "Was the file moved or deleted? The recovery utility cannot "
            "re-load the design matrix without it."
        )
        raise FileNotFoundError(msg)
    return resolved


def _rebuild_cv_folds(
        *,
        split_strategy: str,
        all_sessions: list,
        anchor_feature_data: dict,
        history_frames: int,
        n_splits: int,
        test_proportion: float,
        random_seed: int,
) -> list[dict]:
    """
    Description
    -----------
    Re-create the ``cv_folds`` list of dicts the original
    bout-onset selector used, given the metadata-recorded knobs and
    the freshly-loaded anchor feature data.

    Mirrors the cv-fold construction inside
    ``bout_onset_model_selection`` (sees ``model_selection.py``).
    The ``StratifiedShuffleSplit`` / ``ShuffleSplit`` random_state
    is the same as the original; provided the input data hasn't
    changed shape, the resulting partitions are bit-for-bit
    identical to the selection run's folds.

    Parameters
    ----------
    split_strategy (str)
        Either ``'session'`` or ``'mixed'`` -- determines which
        scikit-learn splitter is used and which fold-dict layout
        is emitted.
    all_sessions (list of str)
        Session ids the selection saw. For ``'session'`` strategy
        these are the items that get split; for ``'mixed'`` they
        determine the row counts on the anchor feature.
    anchor_feature_data (dict)
        Per-session positive / negative trial arrays for the
        anchor feature (e.g. ``all_feature_data['nose-nose']``).
        Used to count rows for the mixed-strategy
        ``StratifiedShuffleSplit``.
    history_frames (int)
        Frames per filter; forwarded to ``pool_session_arrays``.
    n_splits (int)
        Number of CV folds.
    test_proportion (float)
        Test fraction. For ``'session'`` it is clamped to the
        feasible range via ``_bounded_test_proportion``; for
        ``'mixed'`` it is forwarded as-is to ``StratifiedShuffleSplit``.
    random_seed (int)
        ``random_state`` argument to the splitter.

    Returns
    -------
    list of dict
        ``cv_folds`` ready to feed
        ``compute_filter_shapes_per_fold_bout_onset``.
    """

    cv_folds: list[dict] = []
    if split_strategy == 'session':
        all_sessions_arr = np.array(all_sessions)
        ss = ShuffleSplit(
            n_splits=n_splits,
            test_size=_bounded_test_proportion(test_proportion, len(all_sessions)),
            random_state=random_seed,
        )
        for train_idx, test_idx in ss.split(all_sessions_arr):
            cv_folds.append({
                'train_sessions': all_sessions_arr[train_idx],
                'test_sessions': all_sessions_arr[test_idx],
                'type': 'session',
            })
    elif split_strategy == 'mixed':
        X_p_all, X_n_all = pool_session_arrays(
            anchor_feature_data,
            all_sessions,
            pos_key="usv_feature_arr",
            neg_key="no_usv_feature_arr",
            n_frames=history_frames,
        )
        n_pos_total = X_p_all.shape[0]
        n_neg_total = X_n_all.shape[0]
        y_full = np.concatenate(
            (np.ones(n_pos_total), np.zeros(n_neg_total))
        )
        sss = StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_proportion,
            random_state=random_seed,
        )
        for train_ix, test_ix in sss.split(np.zeros(len(y_full)), y_full):
            cv_folds.append({
                'train_idx': train_ix,
                'test_idx': test_ix,
                'n_pos_total': n_pos_total,
                'n_neg_total': n_neg_total,
                'type': 'mixed',
            })
    else:
        msg = (
            f"Unknown split_strategy {split_strategy!r}; expected "
            "'session' or 'mixed'."
        )
        raise ValueError(msg)
    return cv_folds


def recompute_filter_shapes(consolidated_pkl_path: str) -> Path:
    """
    Description
    -----------
    Recompute and persist multi-fold ``filter_shapes`` for a legacy
    bout-onset consolidated pickle whose final-refit produced only a
    single global-fit entry.

    Steps:
      1. Load the consolidated pickle and validate that every
         metadata key needed for exact CV-fold reproduction is
         present. Hard-fail otherwise.
      2. Backup the original to ``<basename>.pre_filter_shape_fix.bak``
         (refuses to overwrite an existing backup -- a second run
         on the already-recovered file is a no-op-with-error).
      3. Re-load the modeling input pickle pointed at by
         ``steps[-1]['input_data_path']`` (run through
         ``configure_path`` for cross-OS mounts).
      4. Re-build ``pooled_feature_cache`` for the selected
         features ONLY (no need to cache the entire ranked set
         since selection is already done).
      5. Re-create ``cv_folds`` with the metadata-recorded
         ``random_seed`` / ``split_strategy`` /
         ``n_splits_selection`` / ``test_proportion`` so the
         partitions match the original run.
      6. Call ``compute_filter_shapes_per_fold_bout_onset`` -- the
         very same helper the fixed production selector uses -- to
         get one filter-shape dict per CV fold.
      7. Hard-fail if 0 folds produced filter shapes; warn if < 2
         (plotter's percentile band needs >= 2). Otherwise write
         the new list back into ``steps[-1]['filter_shapes']``
         and re-pickle.

    Parameters
    ----------
    consolidated_pkl_path (str)
        Absolute or ``configure_path``-able path to a consolidated
        ``selection_*.pkl`` produced by
        ``consolidate_model_selection_results``.

    Returns
    -------
    pathlib.Path
        Absolute path of the (re-)written consolidated pickle
        (same path as the input, after the backup was placed).
    """

    pkl_path = Path(consolidated_pkl_path).resolve()
    if not pkl_path.is_file():
        msg = f"consolidated pickle not found: {pkl_path}"
        raise FileNotFoundError(msg)
    backup_path = pkl_path.with_suffix(pkl_path.suffix + BACKUP_SUFFIX)
    if backup_path.exists():
        msg = (
            f"backup file already exists: {backup_path}\n"
            "This usually means the recovery utility has already been run "
            "on this pickle. Delete the .bak (or rename it out of the way) "
            "if you want to re-run the recovery."
        )
        raise FileExistsError(msg)

    print(f"[recompute_filter_shapes] Loading {pkl_path}")
    with pkl_path.open('rb') as fh:
        consolidated = pickle.load(fh)

    if 'steps' not in consolidated or not isinstance(consolidated['steps'], list):
        msg = (
            f"{pkl_path} is not a consolidated selection artifact "
            "(missing 'steps' list)."
        )
        raise ValueError(msg)

    run_md = consolidated.get('_run_metadata', {})
    input_md = consolidated.get('_input_metadata', {})
    _validate_metadata(run_md, input_md)

    if run_md.get('selection_function') != 'bout_onset_model_selection':
        msg = (
            f"This recovery utility only supports bout-onset selections; "
            f"the artifact's _run_metadata.selection_function is "
            f"{run_md.get('selection_function')!r}."
        )
        raise ValueError(msg)

    steps = consolidated['steps']
    last_step = steps[-1]
    current_model_features = last_step['final_model_features']
    if not current_model_features:
        msg = (
            "last step has no 'final_model_features'; cannot determine "
            "which features to refit."
        )
        raise ValueError(msg)

    input_data_path = last_step['input_data_path']
    print(f"[recompute_filter_shapes] Resolving input data path: {input_data_path}")
    input_data_resolved = _resolve_local_path(input_data_path)

    print(f"[recompute_filter_shapes] Loading modeling input data ...")
    all_feature_data = load_pickle_modeling_data(str(input_data_resolved))

    anchor_feature = run_md['anchor_feature']
    history_frames = int(input_md['filter_history_frames'])
    n_splits = int(run_md['n_splits_selection'])
    test_proportion = float(run_md['test_proportion'])
    random_seed = int(run_md['random_seed'])
    split_strategy = run_md['split_strategy']
    gam_kwargs = dict(run_md['gam_kwargs'])
    n_splines_value = int(run_md['extra_knobs']['n_splines_value'])
    n_splines_time = int(run_md['extra_knobs']['n_splines_time'])
    all_sessions = list(input_md['session_ids'])

    print(
        f"[recompute_filter_shapes] Recreating {n_splits} {split_strategy}-strategy "
        f"folds at random_seed={random_seed}"
    )
    cv_folds = _rebuild_cv_folds(
        split_strategy=split_strategy,
        all_sessions=all_sessions,
        anchor_feature_data=all_feature_data[anchor_feature],
        history_frames=history_frames,
        n_splits=n_splits,
        test_proportion=test_proportion,
        random_seed=random_seed,
    )

    print(
        f"[recompute_filter_shapes] Pre-pooling {len(current_model_features)} feature(s) "
        "for the final refit"
    )
    pooled_feature_cache: dict = {}
    for feat in current_model_features:
        X_p, X_n = pool_session_arrays(
            all_feature_data[feat],
            all_sessions,
            pos_key="usv_feature_arr",
            neg_key="no_usv_feature_arr",
            n_frames=history_frames,
        )
        pooled_feature_cache[feat] = {
            'X_pos': X_p,
            'X_neg': X_n,
            'X_full': np.concatenate((X_p, X_n), axis=0),
        }

    time_indices = np.arange(history_frames, dtype=float)
    print(
        f"[recompute_filter_shapes] Refitting final {len(current_model_features)}-feature "
        f"GAM across {len(cv_folds)} fold(s) ..."
    )
    final_fold_shapes = compute_filter_shapes_per_fold_bout_onset(
        cv_folds=cv_folds,
        current_model_features=current_model_features,
        all_feature_data=all_feature_data,
        pooled_feature_cache=pooled_feature_cache,
        history_frames=history_frames,
        n_splines_value=n_splines_value,
        n_splines_time=n_splines_time,
        gam_kwargs=gam_kwargs,
        random_seed=random_seed,
        time_indices=time_indices,
    )

    if not final_fold_shapes:
        msg = (
            f"All {len(cv_folds)} fold(s) failed during the recovery "
            "refit. The pickle has NOT been modified."
        )
        raise RuntimeError(msg)
    if len(final_fold_shapes) < 2:
        print(
            f"[recompute_filter_shapes] WARNING: only "
            f"{len(final_fold_shapes)} fold(s) produced filter shapes; "
            "the plotter's percentile band will collapse to a single "
            f"line (expected {len(cv_folds)} folds)."
        )

    # Back up the original BEFORE any write. Use copy2 so mtime
    # survives the backup and the user can identify when the
    # pre-recovery pickle was originally written.
    print(f"[recompute_filter_shapes] Backing up original to {backup_path.name}")
    shutil.copy2(pkl_path, backup_path)

    last_step['filter_shapes'] = final_fold_shapes
    last_step.setdefault('final_model_features', current_model_features)

    with pkl_path.open('wb') as fh:
        pickle.dump(consolidated, fh)
    print(
        f"[recompute_filter_shapes] Wrote {len(final_fold_shapes)} fold(s) of "
        f"filter_shapes back to {pkl_path.name}"
    )
    return pkl_path


def main(argv: list[str] | None = None) -> int:
    """
    Description
    -----------
    Console-script entry point for ``recompute-filter-shapes``.
    Wraps ``recompute_filter_shapes`` so the recovery is one line
    from a venv-active shell.

    Parameters
    ----------
    argv (list[str] | None)
        Argument vector. ``None`` reads from ``sys.argv[1:]``.

    Returns
    -------
    exit_code (int)
        ``0`` on success, ``1`` on caught failure (FileNotFoundError /
        ValueError / RuntimeError surfaced from the recovery).
    """

    parser = argparse.ArgumentParser(
        description=(
            "Recompute per-fold filter shapes for a legacy bout-onset "
            "consolidated selection pickle (one that was written before "
            "commit 56efb08 and therefore carries a single-entry "
            "filter_shapes list)."
        )
    )
    parser.add_argument(
        'consolidated_pkl',
        help="Path to the consolidated selection_*.pkl",
    )
    args = parser.parse_args(argv)
    try:
        recompute_filter_shapes(args.consolidated_pkl)
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        print(f"[recompute_filter_shapes] FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
