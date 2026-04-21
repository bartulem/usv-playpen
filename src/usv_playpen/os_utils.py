"""
@author: bartulem
Configure path to the OS in use and small subprocess/glob helpers shared
across the codebase.
"""

from __future__ import annotations

import pathlib
import platform
import subprocess
from collections.abc import Callable, Iterable
from typing import Optional


def find_base_path() -> str | None:
    """
    Description
    ----------
    This function converts the CUP path between OSs.
    ----------

    Parameters
    ----------
    ----------

    Returns
    -------
     base_path (str)
        OS-converted CUP path.
    """

    if platform.system() == "Windows":
        base_path = "F:\\"
    elif platform.system() == "Darwin":
        base_path = "/Volumes/falkner"
    elif platform.system() == "Linux":
        base_path = "/mnt/falkner"
    else:
        base_path = None

    return base_path


def configure_path(pa: str) -> str:
    """
    Description
    ----------
    This function converts path names between OSs.
    ----------

    Parameters
    ----------
    pa (str)
        Original path.

    Returns
    -------
     pa (str)
        OS-converted path.
    """

    if pa.startswith("F:\\"):
        if platform.system() == "Darwin":
            pa = pa.replace("\\", "/").replace("F:", "/Volumes/falkner")
        elif platform.system() == "Linux":
            pa = pa.replace("\\", "/").replace("F:", "/mnt/falkner")
    elif pa.startswith("/mnt"):
        if platform.system() == "Windows":
            pa = pa.replace("/mnt/falkner", "F:").replace("/", "\\")
        elif platform.system() == "Darwin":
            pa = pa.replace("mnt", "Volumes")
    elif pa.startswith("/Volumes"):
        if platform.system() == "Windows":
            pa = pa.replace("/Volumes/falkner", "F:").replace("/", "\\")
        elif platform.system() == "Linux":
            pa = pa.replace("Volumes", "mnt")

    return pa


def wait_for_subprocesses(
    subps: Iterable[subprocess.Popen],
    max_seconds: float,
    label: str,
    poll_interval_s: float = 1.0,
    message_output: Optional[Callable] = None,
    raise_on_nonzero: bool = False,
    raise_on_timeout: bool = True,
) -> list[Optional[int]]:
    """
    Description
    ----------
    Polls a collection of subprocess.Popen handles until every one has
    terminated or a timeout is reached. Replaces the previous 'while True:
    poll()' idiom that appeared across the codebase (behavioral_experiments,
    synchronize_files, modify_files, das_inference, anipose_operations), which
    had no timeout and silently ignored non-zero return codes.

    On timeout, still-running subprocesses are terminated (SIGTERM) and given
    a short grace period before being killed (SIGKILL), so the parent process
    does not leave orphaned Popen handles.
    ----------

    Parameters
    ----------
    subps (Iterable[subprocess.Popen])
        The subprocess handles to wait on. Empty iterables are a no-op.
    max_seconds (float)
        Hard timeout for the entire group. A TimeoutError is raised if the
        group does not finish within this budget (unless raise_on_timeout is
        False, in which case the still-running subprocesses are terminated and
        their slots in the returned list are left as None).
    label (str)
        Short human-readable label used in log / exception messages so the
        caller knows which phase timed out (e.g., 'audio file copy').
    poll_interval_s (float)
        Seconds between successive poll() calls.
    message_output (Callable, optional)
        Function used to surface progress and failure messages. Defaults to
        the built-in print() when None.
    raise_on_nonzero (bool)
        If True, raises RuntimeError when any subprocess exits with a
        non-zero return code.
    raise_on_timeout (bool)
        If True, raises TimeoutError when the group exceeds max_seconds.
    ----------

    Returns
    -------
    return_codes (list[Optional[int]])
        The return code of each subprocess, in the same order as the input.
        Slots are None for subprocesses that had to be terminated on timeout.
    -------
    """

    log = message_output or print

    subps_list = list(subps)
    if not subps_list:
        return []

    import time as _time

    deadline = _time.monotonic() + max_seconds

    while True:
        status = [p.poll() for p in subps_list]
        if all(s is not None for s in status):
            break
        if _time.monotonic() >= deadline:
            still_running_idx = [i for i, s in enumerate(status) if s is None]
            log(
                f"[{label}] timed out after {max_seconds:.0f} s with "
                f"{len(still_running_idx)}/{len(subps_list)} subprocess(es) still running; terminating."
            )
            for i in still_running_idx:
                try:
                    subps_list[i].terminate()
                except OSError:
                    pass
            # Brief grace period for terminate() to take effect
            grace_end = _time.monotonic() + 3
            while _time.monotonic() < grace_end and any(p.poll() is None for p in subps_list):
                _time.sleep(0.25)
            for i in still_running_idx:
                if subps_list[i].poll() is None:
                    try:
                        subps_list[i].kill()
                    except OSError:
                        pass
            if raise_on_timeout:
                raise TimeoutError(
                    f"{label}: {len(still_running_idx)} subprocess(es) did not finish within {max_seconds:.0f} s."
                )
            # refresh status after termination
            status = [p.poll() for p in subps_list]
            break
        _time.sleep(poll_interval_s)

    failed = [(i, s) for i, s in enumerate(status) if s is not None and s != 0]
    if failed:
        failures_str = ", ".join(f"#{i}(rc={s})" for i, s in failed)
        log(f"[{label}] {len(failed)}/{len(subps_list)} subprocess(es) exited with non-zero status: {failures_str}.")
        if raise_on_nonzero:
            raise RuntimeError(
                f"{label}: {len(failed)} subprocess(es) failed — {failures_str}."
            )

    return status


def first_match_or_raise(
    root: pathlib.Path,
    pattern: str,
    recursive: bool = False,
    label: Optional[str] = None,
) -> pathlib.Path:
    """
    Description
    ----------
    Returns the first path matching a glob pattern under ``root``, or raises
    a FileNotFoundError with a clear, debuggable message naming both the
    pattern and the root that produced zero matches. Replaces the common
    'sorted(root.glob(...))[0]' / 'list(root.rglob(...))[0]' idiom, which
    surfaced as bare IndexError with no hint about which pattern failed.
    ----------

    Parameters
    ----------
    root (pathlib.Path)
        The directory to search.
    pattern (str)
        The glob pattern (relative to root) to match.
    recursive (bool)
        If True, uses rglob to walk the directory tree; otherwise uses glob.
    label (str, optional)
        Short context label to include in the error message (e.g.,
        'camera frame count JSON'). Defaults to the pattern itself.
    ----------

    Returns
    -------
    match (pathlib.Path)
        The first match. For deterministic callers that cared about ordering,
        note that this returns the first match in whatever order the
        underlying glob produces; wrap with sorted() on the caller side if
        needed.
    -------
    """

    root = pathlib.Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{label or pattern}: search root '{root}' does not exist."
        )
    matches_iter = root.rglob(pattern) if recursive else root.glob(pattern)
    match = next(iter(matches_iter), None)
    if match is None:
        kind = "rglob" if recursive else "glob"
        raise FileNotFoundError(
            f"{label or pattern}: no match for {kind} pattern '{pattern}' under '{root}'."
        )
    return match


def newest_match_or_raise(
    root: pathlib.Path,
    pattern: str,
    key: Optional[Callable[[pathlib.Path], float]] = None,
    recursive: bool = False,
    label: Optional[str] = None,
) -> pathlib.Path:
    """
    Description
    ----------
    Returns the single "largest" (newest, by default) path matching a glob
    pattern under ``root``, or raises FileNotFoundError with a clear, named
    message when the glob produces zero matches. Replaces the common
    'max(root.glob(...), key=lambda p: p.stat().st_ctime)' idiom, which
    raised a bare ValueError ("max() arg is an empty sequence") with no
    hint about which directory or pattern produced the empty result.
    ----------

    Parameters
    ----------
    root (pathlib.Path)
        The directory to search.
    pattern (str)
        The glob pattern (relative to root) to match.
    key (Callable[[pathlib.Path], float], optional)
        Ordering key passed to max(). Defaults to creation time
        (lambda p: p.stat().st_ctime).
    recursive (bool)
        If True, uses rglob to walk the directory tree; otherwise uses glob.
    label (str, optional)
        Short context label to include in the error message (e.g.,
        'most recent Avisoft .wav'). Defaults to the pattern itself.
    ----------

    Returns
    -------
    match (pathlib.Path)
        The maximum-key match.
    -------
    """

    root = pathlib.Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{label or pattern}: search root '{root}' does not exist."
        )
    if key is None:
        key = lambda p: p.stat().st_ctime
    matches = list(root.rglob(pattern) if recursive else root.glob(pattern))
    if not matches:
        kind = "rglob" if recursive else "glob"
        raise FileNotFoundError(
            f"{label or pattern}: no match for {kind} pattern '{pattern}' under '{root}'."
        )
    return max(matches, key=key)
