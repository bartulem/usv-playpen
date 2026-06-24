"""
@author: bartulem
Configure path to the OS in use and small subprocess/glob helpers shared
across the codebase.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import platform
import subprocess
import time as _time
from collections.abc import Callable, Iterable, Iterator
from typing import Optional

import toml


# The lab CUP shares are defined ONCE, in the ``lab_shares`` / ``file_server``
# entries of the host config (``_config/behavioral_experiments_settings.toml``),
# read by ``_host_lab_shares`` below -- the single source also consumed by the
# recording GUI and behavioral_experiments. Each share stores only the
# irreducible tokens: the share ``name`` (falkner), the Windows drive LETTER
# (``F``), and the per-OS mount PARENT (``/Volumes``, ``/mnt``, ``/mnt/cup/labs``
# for ``darwin``/``linux``/``cluster``). ``expand_lab_share`` turns those into
# the full leading mount roots (``F:``, ``/Volumes/falkner``, ...) and the
# ``\\<file_server>\<name>`` UNC. The first share (falkner) is the default
# returned by ``find_base_path``. Only a leading mount root is ever rewritten, so
# look-alike substrings elsewhere in the path are never touched, and additional
# shares (murthy, ...) are handled by construction, not special-casing.
#
# These tokens are read once and expanded by ``_host_lab_shares``; if that config
# is missing/unparseable or lacks a ``lab_shares`` / ``file_server`` entry, path
# translation raises rather than guessing -- a broken or incomplete host config
# should fail loud, not silently fall back to assumed shares.
_OS_KEYS = {"Windows": "windows", "Darwin": "darwin", "Linux": "linux"}


def expand_lab_share(share: dict, file_server: str) -> dict:
    """
    Description
    -----------
    Expands a token-form lab share into its full leading mount roots. The host
    config stores only the irreducible tokens per share (the drive LETTER and the
    per-OS mount PARENT, with the share ``name`` factored out); this appends the
    name (and the ``:`` for Windows) to build each OS's leading mount root, plus
    the ``\\<file_server>\\<name>`` UNC path. It is the single place this
    derivation happens, used by ``_host_lab_shares`` (path translation), the
    recording GUI, and behavioral_experiments.

    Parameters
    ----------
    share (dict)
        Token-form share: ``name`` + ``windows`` (drive letter, e.g. ``F``) +
        ``darwin``/``linux``/``cluster`` mount parents (e.g. ``/Volumes``,
        ``/mnt``, ``/mnt/cup/labs``).
    file_server (str)
        The SMB server name (e.g. ``cup``) -- the ``\\<file_server>\\...`` host.

    Returns
    -------
    expanded (dict)
        ``name`` plus the full leading roots ``windows`` (``F:``), ``darwin``
        (``/Volumes/falkner``), ``linux`` (``/mnt/falkner``), ``cluster``
        (``/mnt/cup/labs/falkner``), and ``unc`` (``\\cup\\falkner``).
    """

    name = share["name"]
    return {
        "name": name,
        "windows": f"{share['windows']}:",
        "darwin": f"{share['darwin']}/{name}",
        "linux": f"{share['linux']}/{name}",
        "cluster": f"{share['cluster']}/{name}",
        "unc": rf"\\{file_server}\{name}",
    }


def recording_destinations(lab_shares, file_server: str, selected_labs, experimenter: str) -> tuple[list[str], list[str]]:
    """
    Description
    -----------
    Builds the per-OS recording destination lists for the selected labs --
    ``<mount root>/<experimenter>/Data`` for each selected share, in both Linux
    and Windows forms. The single place recording destinations are composed,
    used by the recording GUI and behavioral_experiments so the destination
    layout lives in one spot and is never persisted as hardcoded full paths.

    Parameters
    ----------
    lab_shares (iterable of dict)
        Token-form shares (see ``expand_lab_share``).
    file_server (str)
        The SMB server name.
    selected_labs (iterable of str)
        The ``name`` of each lab this host should write recordings to.
    experimenter (str)
        The experimenter folder placed under the share root.

    Returns
    -------
    (linux_destinations, win_destinations) (tuple[list[str], list[str]])
        Parallel lists of full destination directories in Linux and Windows
        form, one per selected lab, in ``lab_shares`` order.
    """

    selected = set(selected_labs)
    linux_destinations = []
    win_destinations = []
    for share in lab_shares:
        if share["name"] in selected:
            roots = expand_lab_share(share, file_server)
            linux_destinations.append(f"{roots['linux']}/{experimenter}/Data")
            win_destinations.append(f"{roots['windows']}\\{experimenter}\\Data")
    return linux_destinations, win_destinations

_HOST_CONFIG_PATH = pathlib.Path(__file__).parent / "_config" / "behavioral_experiments_settings.toml"
# One-element cache for the resolved (shares, file_server); a list so it can be
# populated by mutation (no ``global``). Cleared in tests to force a re-read.
_HOST_SHARES_CACHE: list[tuple[tuple[dict[str, str], ...], str]] = []


def _host_lab_shares() -> tuple[tuple[dict[str, str], ...], str]:
    """
    Description
    -----------
    Returns the host's EXPANDED lab share table and file-server name, resolved
    once and cached for the process. The token-form values are read from the
    ``lab_shares`` and ``file_server`` entries of the host config
    (``_config/behavioral_experiments_settings.toml``) -- the single place the
    drive letters / mount roots are defined, also consumed by the recording GUI
    and behavioral_experiments -- and expanded into full leading roots via
    ``expand_lab_share`` so ``configure_path``/``find_base_path``/
    ``to_cluster_path`` see ready-to-use mount roots. If that file is missing or
    unparseable, or lacks a ``lab_shares`` / ``file_server`` entry, a clear error
    is raised rather than falling back to assumed shares -- a broken or incomplete
    host config fails loud at the first path translation.

    Parameters
    ----------
    None

    Returns
    -------
    (shares, file_server) (tuple[tuple[dict[str, str], ...], str])
        ``shares`` is the ordered per-lab table of EXPANDED shares (first entry =
        default for ``find_base_path``), each a dict with full
        ``name``/``windows``/``darwin``/``linux``/``cluster``/``unc`` roots;
        ``file_server`` is the SMB server name (e.g. ``cup``).
    """

    if _HOST_SHARES_CACHE:
        return _HOST_SHARES_CACHE[0]

    try:
        host_config = toml.load(_HOST_CONFIG_PATH)
    except (OSError, toml.TomlDecodeError) as exc:
        msg = (
            f"Cannot read the host config '{_HOST_CONFIG_PATH}' required to resolve "
            f"the lab CUP shares for path translation: {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(msg) from exc

    if "lab_shares" not in host_config or not host_config["lab_shares"]:
        msg = (
            f"Host config '{_HOST_CONFIG_PATH}' has no non-empty 'lab_shares' table; "
            "cannot resolve the lab CUP share mount roots."
        )
        raise KeyError(msg)
    if "file_server" not in host_config:
        msg = (
            f"Host config '{_HOST_CONFIG_PATH}' has no 'file_server' entry; "
            "cannot form the file-server UNC roots."
        )
        raise KeyError(msg)

    raw_shares = tuple(host_config["lab_shares"])
    file_server = host_config["file_server"]

    expanded = tuple(expand_lab_share(share, file_server) for share in raw_shares)
    _HOST_SHARES_CACHE.append((expanded, file_server))
    return _HOST_SHARES_CACHE[0]


# One-element cache for the resolved experimenter id; a list so it can be
# populated by mutation (no ``global``). Cleared in tests to force a re-read.
_HOST_EXPERIMENTER_CACHE: list[str] = []


def _host_experimenter() -> str:
    """
    Description
    -----------
    Returns the canonical experimenter id from the host config TOML (the
    top-level ``experimenter`` key of
    ``_config/behavioral_experiments_settings.toml``), resolved once and cached
    for the process. This is the single source the ``{experimenter}`` placeholder
    in the analysis data / model paths resolves to (via :func:`configure_path`),
    so the experimenter is declared in exactly one place -- the same key the
    recording GUI writes and ``exp_id`` is derived from. A missing or unparseable
    host config, or one lacking an ``experimenter`` entry, raises rather than
    guessing, so a broken config fails loud at the first templated-path read.

    Parameters
    ----------
    None

    Returns
    -------
    experimenter (str)
        The experimenter id (e.g. ``Bartul``).
    """

    if _HOST_EXPERIMENTER_CACHE:
        return _HOST_EXPERIMENTER_CACHE[0]

    try:
        host_config = toml.load(_HOST_CONFIG_PATH)
    except (OSError, toml.TomlDecodeError) as exc:
        msg = (
            f"Cannot read the host config '{_HOST_CONFIG_PATH}' required to resolve "
            f"the experimenter id for path templating: {type(exc).__name__}: {exc}"
        )
        raise RuntimeError(msg) from exc

    if "experimenter" not in host_config:
        msg = (
            f"Host config '{_HOST_CONFIG_PATH}' has no 'experimenter' entry; "
            "cannot fill the '{experimenter}' placeholder in data paths."
        )
        raise KeyError(msg)

    _HOST_EXPERIMENTER_CACHE.append(host_config["experimenter"])
    return _HOST_EXPERIMENTER_CACHE[0]


def find_base_path() -> str | None:
    """
    Description
    -----------
    Returns the primary (falkner) CUP share's mount root for the OS currently
    in use: ``F:\\`` on Windows, ``/Volumes/falkner`` on macOS, ``/mnt/falkner``
    on Linux. Derived from the first entry of the resolved lab-share table
    (``_host_lab_shares``) so the roots are defined in exactly one place.

    Parameters
    ----------
    None

    Returns
    -------
    base_path (str | None)
        The falkner mount root for the host OS, or ``None`` on an unrecognised
        platform (callers must handle the ``None`` case before using the value).
    """

    system = platform.system()
    if system not in _OS_KEYS:
        return None
    base = _host_lab_shares()[0][0][_OS_KEYS[system]]
    return f"{base}\\" if system == "Windows" else base


def configure_path(pa: str) -> str:
    """
    Description
    -----------
    Translates a CUP-share path from whichever OS form it was written in into
    the form expected by the OS currently in use, for any share listed in
    the resolved lab-share table (falkner, murthy, ...).

    A ``{experimenter}`` placeholder -- used in the analysis data / model paths
    in the ``*_settings.json`` files so those paths follow the experimenter
    rather than hard-coding a name -- is first filled with the canonical
    experimenter id (:func:`_host_experimenter`, declared in exactly one place).
    Paths without the placeholder (e.g. the user-entered root/arena directories)
    are unaffected.

    Only the leading mount root is rewritten; the remainder of the path is kept
    verbatim apart from normalising the path separator to the target OS
    (``\\`` on Windows, ``/`` elsewhere). A root must be followed by a path
    separator (or be the whole string) to match, so:

    * embedded look-alike substrings are never corrupted -- e.g.
      ``/mnt/falkner/exp_mnt_2025`` on macOS becomes
      ``/Volumes/falkner/exp_mnt_2025`` (the inner ``mnt`` is left alone), and
    * a path that is already in the host-OS form, or that does not begin with
      any known share root, is returned unchanged (passthrough).

    Parameters
    ----------
    pa (str)
        Original path, in any OS's form for a known CUP share, or an unrelated
        path (returned unchanged).

    Returns
    -------
    pa (str)
        OS-converted path, or the original string if no known share root
        matched or the host OS is unrecognised.
    """

    if "{experimenter}" in pa:
        pa = pa.replace("{experimenter}", _host_experimenter())

    system = platform.system()
    if system not in _OS_KEYS:
        return pa
    target_key = _OS_KEYS[system]

    for share in _host_lab_shares()[0]:
        # Only the host-OS forms are translation sources; the non-OS ``cluster``
        # form is handled by ``to_cluster_path`` and must never match here.
        for src_key in _OS_KEYS.values():
            if src_key == target_key:
                continue
            root = share[src_key]
            if pa == root or (pa.startswith(root) and pa[len(root):len(root) + 1] in ("/", "\\")):
                remainder = pa[len(root):]
                remainder = remainder.replace("/", "\\") if target_key == "windows" else remainder.replace("\\", "/")
                return f"{share[target_key]}{remainder}"

    return pa


def find_cluster_path() -> str:
    """
    Description
    -----------
    Returns the primary (falkner) CUP share's **cluster** mount root
    (e.g. ``/mnt/cup/labs/falkner``), regardless of the host OS. This is the
    location of the share when a job runs ON the HPC cluster (where the share
    is not locally mounted under the host-OS root from :func:`find_base_path`).
    Derived from the resolved lab-share table so the cluster root is defined in
    exactly one place (``_config/behavioral_experiments_settings.toml``).

    Parameters
    ----------
    None

    Returns
    -------
    cluster_path (str)
        The falkner cluster mount root.
    """

    return _host_lab_shares()[0][0]["cluster"]


_ANALYSES_SETTINGS_PATH = pathlib.Path(__file__).parent / "_parameter_settings" / "analyses_settings.json"


def resolve_data_root(key: str) -> pathlib.Path:
    """
    Description
    -----------
    Reads a canonical data-location path from
    ``analyses_settings.json['data_roots'][key]`` and resolves it via
    :func:`configure_path`, which fills the ``{experimenter}`` placeholder with
    the canonical experimenter id and translates the leading mount root to the
    host OS. This is the single place the analysis data roots (the EPHYS /
    histology / Data trees, the unit catalog, the aggregator output directory,
    ...) are defined, so they are user-editable configuration that is also
    OS-portable and experimenter-keyed -- rather than hard-coded
    ``/mnt/<experimenter>/...`` constants.

    Parameters
    ----------
    key (str)
        A key under the ``data_roots`` block of ``analyses_settings.json``
        (e.g. ``'ephys_root'``, ``'histology_root'``, ``'catalog_path'``).

    Returns
    -------
    data_root (pathlib.Path)
        The configured path, with ``{experimenter}`` filled in and its leading
        mount root translated to the host OS.
    """

    with _ANALYSES_SETTINGS_PATH.open() as settings_file:
        data_roots = json.load(settings_file)["data_roots"]
    return pathlib.Path(configure_path(data_roots[key]))


def ephys_base_for_data_root(data_root_directory: str) -> pathlib.Path:
    """
    Description
    -----------
    Maps a session's ``Data``-tree root directory to the parent of its sibling
    ``EPHYS``-tree directory. The lab mirrors every ``Data`` directory with an
    ``EPHYS`` directory at the same level, so a session stored at
    ``<base>/Data/<session_id>`` keeps its electrophysiology recordings under
    ``<base>/EPHYS/...``.

    Only the final path *component* that is exactly ``Data`` in the **parent**
    of ``data_root_directory`` is swapped for ``EPHYS``; look-alike substrings
    elsewhere in the path -- an experimenter directory such as ``Database``, or
    a session id that merely contains the text ``Data`` -- are therefore never
    corrupted. This replaces the previous unanchored
    ``str(parent).replace('Data', 'EPHYS')`` idiom, which rewrote every
    occurrence of the substring and so was prone to silently mangling such
    paths.

    Parameters
    ----------
    data_root_directory (str)
        Absolute path to a single session's ``Data``-tree root directory, e.g.
        ``/mnt/falkner/Bartul/Data/20230101_120000``.

    Returns
    -------
    ephys_base (pathlib.Path)
        The session root's parent with its final ``Data`` component replaced by
        ``EPHYS``, e.g. ``/mnt/falkner/Bartul/EPHYS``. If the parent contains no
        ``Data`` component it is returned unchanged.
    """

    parent = pathlib.Path(data_root_directory).parent
    parts = list(parent.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "Data":
            parts[index] = "EPHYS"
            break
    return pathlib.Path(*parts)


def to_cluster_path(pa: str) -> str:
    """
    Description
    -----------
    Translates a CUP-share path written in any host-OS form (Windows drive
    letter, macOS ``/Volumes`` mount, or Linux ``/mnt`` mount) into the form the
    compute cluster (spock/della) uses, where every lab share is mounted under
    ``/mnt/cup/labs/<lab>``. Unlike ``configure_path``, whose target is the host
    OS, the target here is fixed (the cluster), because this is used when
    *submitting* jobs from a workstation to run remotely.

    Both lab shares in the resolved table (falkner, murthy) are handled from every
    host form, so a ``M:\\...`` / ``/Volumes/murthy/...`` / ``/mnt/murthy/...``
    path maps correctly to ``/mnt/cup/labs/murthy/...`` -- the previous
    per-OS ``.replace`` only special-cased falkner on Windows and silently left
    murthy paths unconverted.

    Matching is anchored on a full mount root followed by a separator (or the
    whole string), so embedded look-alike substrings are never corrupted
    (the previous Linux ``.replace('mnt', 'mnt/cup/labs')`` mangled any inner
    ``mnt`` token). Separators in the remainder are normalised to ``/`` because
    the cluster is Linux.

    Parameters
    ----------
    pa (str)
        Original path in any host-OS form for a known CUP share. A path that
        does not begin with a known share root is returned with its separators
        normalised to ``/`` but otherwise unchanged.

    Returns
    -------
    pa (str)
        Cluster-form path (``/mnt/cup/labs/<lab>/...``), or the separator-
        normalised original if no known share root matched.
    """

    normalised = pa.replace("\\", "/")
    for share in _host_lab_shares()[0]:
        cluster_root = share["cluster"]
        for src_key in _OS_KEYS.values():
            root = share[src_key].replace("\\", "/")
            if normalised == root or (normalised.startswith(root) and normalised[len(root):len(root) + 1] == "/"):
                return f"{cluster_root}{normalised[len(root):]}"
    return normalised


@contextlib.contextmanager
def atomic_output_path(final_path: str | pathlib.Path) -> Iterator[pathlib.Path]:
    """
    Description
    -----------
    Context manager for crash-safe, atomic publishing of precious files. It
    yields a temporary sibling path to write into and, on clean exit, replaces
    ``final_path`` with it via ``os.replace`` (an atomic rename on the same
    filesystem).

    Writing irreplaceable data (session metadata, h5 archives, pickles)
    straight to its final path with mode ``'w'`` truncates the existing file
    before the new bytes land, so a crash, kill, or full disk mid-write leaves
    a corrupt or empty file -- and the original is already gone. Writing to a
    sibling temp file and renaming makes the publish all-or-nothing: a reader
    sees either the complete old file or the complete new file, never a partial
    one.

    The temp file is created in the same directory as ``final_path`` so the
    rename stays on one filesystem (cross-filesystem ``os.replace`` is not
    atomic and may fail). If the body raises, the temp file is removed and the
    exception re-raised, leaving any existing ``final_path`` untouched.

    Parameters
    ----------
    final_path (str | pathlib.Path)
        Destination path to publish atomically. Its parent directory must
        already exist (this helper does not create it, matching the callers'
        existing assumption).

    Yields
    ------
    tmp_path (pathlib.Path)
        Sibling temporary path the caller writes its bytes to. After a clean
        exit it has been renamed onto ``final_path`` and no longer exists under
        the temporary name.
    """

    final = pathlib.Path(final_path)
    tmp = final.with_name(f".{final.name}.tmp-{os.getpid()}")
    try:
        yield tmp
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    os.replace(tmp, final)


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
    -----------
    Polls a collection of subprocess.Popen handles until every one has
    terminated or a timeout is reached. Replaces the previous 'while True:
    poll()' idiom that appeared across the codebase (behavioral_experiments,
    synchronize_files, modify_files, das_inference, anipose_operations), which
    had no timeout and silently ignored non-zero return codes.

    On timeout, still-running subprocesses are terminated (SIGTERM) and given
    a short grace period before being killed (SIGKILL), so the parent process
    does not leave orphaned Popen handles.

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

    Returns
    -------
    return_codes (list[Optional[int]])
        The return code of each subprocess, in the same order as the input.
        Slots are None for subprocesses that had to be terminated on timeout.
    """

    log = message_output or print

    subps_list = list(subps)
    if not subps_list:
        return []

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
    -----------
    Returns the alphabetically-first path matching a glob pattern under
    ``root``, or raises a FileNotFoundError with a clear, debuggable message
    naming both the pattern and the root that produced zero matches. Replaces
    the common 'sorted(root.glob(...))[0]' / 'list(root.rglob(...))[0]' idiom,
    which surfaced as bare IndexError with no hint about which pattern failed.

    Matches are sorted deterministically before selection so that callers get
    the same answer across runs, platforms, and filesystems. Earlier revisions
    used 'next(iter(glob))' which returned matches in directory-entry order;
    that caused non-deterministic behavior when the pattern matched multiple
    files (ext4's hash-order directory listing, in particular, is effectively
    random). Every known caller was the post-refactor equivalent of
    'sorted(glob)[0]', so sorting is restored as the default — there is no
    opt-out because non-deterministic first-match is not a feature we want.

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

    Returns
    -------
    match (pathlib.Path)
        The alphabetically-first match.
    """

    root = pathlib.Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{label or pattern}: search root '{root}' does not exist."
        )
    matches = sorted(root.rglob(pattern) if recursive else root.glob(pattern))
    if not matches:
        kind = "rglob" if recursive else "glob"
        raise FileNotFoundError(
            f"{label or pattern}: no match for {kind} pattern '{pattern}' under '{root}'."
        )
    return matches[0]


def newest_match_or_raise(
    root: pathlib.Path,
    pattern: str,
    key: Optional[Callable[[pathlib.Path], float]] = None,
    recursive: bool = False,
    label: Optional[str] = None,
) -> pathlib.Path:
    """
    Description
    -----------
    Returns the single "largest" (newest, by default) path matching a glob
    pattern under ``root``, or raises FileNotFoundError with a clear, named
    message when the glob produces zero matches. Replaces the common
    'max(root.glob(...), key=lambda p: p.stat().st_ctime)' idiom, which
    raised a bare ValueError ("max() arg is an empty sequence") with no
    hint about which directory or pattern produced the empty result.

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

    Returns
    -------
    match (pathlib.Path)
        The maximum-key match.
    """

    root = pathlib.Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"{label or pattern}: search root '{root}' does not exist."
        )
    if key is None:
        def key(p):
            return p.stat().st_ctime
    matches = list(root.rglob(pattern) if recursive else root.glob(pattern))
    if not matches:
        kind = "rglob" if recursive else "glob"
        raise FileNotFoundError(
            f"{label or pattern}: no match for {kind} pattern '{pattern}' under '{root}'."
        )
    return max(matches, key=key)


# Embedding-landscape resolution. The visualization layer reads its precomputed
# cohort artifacts from a single base directory (``shared_resources.spectrograms_dir``)
# by convention, rather than from several hard-coded file paths:
#   <dir>/qlvm/arrays_{coarse,fine}.npz      QLVM torus density + watershed labels
#   <dir>/vae/vae_density_{coarse,fine}.npz  VAE umap density + category labels
#   <dir>/spectrograms_*.h5                   consolidated spectrogram/mask/latent store
def resolve_embedding_arrays_path(spectrograms_dir: str, embedding: str, clustering: str) -> str:
    """
    Description
    -----------
    Build the path to a precomputed embedding-landscape ``.npz`` under the
    spectrograms base directory, by convention -- QLVM at
    ``<dir>/qlvm/arrays_{coarse,fine}.npz`` and VAE at
    ``<dir>/vae/vae_density_{coarse,fine}.npz``. This is a pure path builder (run
    through ``configure_path``); whether the file exists is the caller's concern
    (the sequence figure falls back to a bare panel, the torus video requires it).

    Parameters
    ----------
    spectrograms_dir (str)
        Base directory where the ``qlvm`` / ``vae`` subdirectories branch off.
    embedding (str)
        ``"qlvm"`` or ``"vae"``.
    clustering (str)
        ``"fine"`` selects the fine map; anything else selects the coarse map.

    Returns
    -------
    path (str)
        The OS-resolved ``.npz`` path (not checked for existence).
    """

    base = pathlib.Path(configure_path(spectrograms_dir))
    tag = "fine" if clustering == "fine" else "coarse"
    if embedding == "vae":
        return str(base / "vae" / f"vae_density_{tag}.npz")
    return str(base / "qlvm" / f"arrays_{tag}.npz")


def resolve_consolidated_h5_path(spectrograms_dir: str) -> str:
    """
    Description
    -----------
    Resolve the consolidated spectrogram/mask/latent ``.h5`` store: the most
    recently modified ``spectrograms_*.h5`` directly under the base directory. The
    name pattern is deliberate -- it selects the consolidated store and skips other
    ``.h5`` siblings (e.g. ``qlvm_clusters_*.h5``). Raises ``FileNotFoundError`` with
    a clear message if the directory is missing or holds no matching store.

    Parameters
    ----------
    spectrograms_dir (str)
        Base spectrograms directory (run through ``configure_path``).

    Returns
    -------
    path (str)
        The OS-resolved path to the newest ``spectrograms_*.h5``.
    """

    base = pathlib.Path(configure_path(spectrograms_dir))
    return str(
        newest_match_or_raise(
            base,
            "spectrograms_*.h5",
            key=lambda p: p.stat().st_mtime,
            label="consolidated spectrogram .h5",
        )
    )


def resolve_pooled_embeddings_cache(spectrograms_dir: str) -> str:
    """
    Description
    -----------
    Build the path to the cohort pooled-embeddings parquet cache under the
    spectrograms base directory, by convention:
    ``<dir>/embeddings/pooled_embeddings.parquet``. This is a pure path builder
    (run through ``configure_path``); whether the file exists is the caller's
    concern -- the embedding figures pass it as ``embeddings_cache_path`` so that
    ``build_pooled_embeddings_df`` loads it when present (one combined table that
    carries both the VAE and QLVM coordinates and the coarse + fine labels), and
    otherwise pools the cohort and writes it there.

    Parameters
    ----------
    spectrograms_dir (str)
        Base directory where the precomputed embedding artifacts live.

    Returns
    -------
    path (str)
        The OS-resolved parquet path (not checked for existence).
    """

    base = pathlib.Path(configure_path(spectrograms_dir))
    return str(base / "embeddings" / "pooled_embeddings.parquet")
