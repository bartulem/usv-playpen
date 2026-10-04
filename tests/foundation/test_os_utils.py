"""
Tests for usv_playpen.os_utils: cross-OS path translation (find_base_path /
configure_path), the Data->EPHYS sibling-tree mapping (ephys_base_for_data_root),
the deterministic glob helpers (first_match_or_raise / newest_match_or_raise),
the band-explicit audio memmap lookup (find_audio_mmap / parse_audio_mmap_name,
including that every reader gets the 'usv' memmap when both bands exist)
and the subprocess-group waiter (wait_for_subprocesses).

`configure_path`/`find_base_path` are exercised under all three target OSs by
monkeypatching `os_utils.platform.system`; the cases include the two bugs the
mount-mapping table was introduced to fix (all-occurrence substring corruption,
and the previously-unhandled `murthy` share).
"""

import os
import pathlib
import re
import time

import numpy as np
import polars as pls
import pytest

from usv_playpen import os_utils
from usv_playpen.processing.generate_spectrograms import open_hpss_audio
from usv_playpen.visualizations.make_behavioral_videos import load_audio_data
from usv_playpen.processing.qlvm_latents import (
    model_cell_reserved_columns,
    validate_model_cells,
)
from tests.conftest import write_qlvm_category_bundle


@pytest.fixture
def as_os(monkeypatch):
    """Return a setter that pins os_utils.platform.system() to a given value."""
    def _set(system_name):
        monkeypatch.setattr(os_utils.platform, "system", lambda: system_name)
    return _set


# find_base_path

@pytest.mark.parametrize("system,expected", [
    ("Windows", "F:\\"),
    ("Darwin", "/Volumes/falkner"),
    ("Linux", "/mnt/falkner"),
    ("SunOS", None),
])
def test_find_base_path_per_os(as_os, system, expected):
    as_os(system)
    assert os_utils.find_base_path() == expected


# find_cluster_path

@pytest.mark.parametrize("system", ["Windows", "Darwin", "Linux"])
def test_find_cluster_path_is_os_independent(as_os, system):
    """The cluster mount root (where a job sees the share when running ON the
    HPC cluster) is the same regardless of the host OS."""
    as_os(system)
    assert os_utils.find_cluster_path() == "/mnt/cup/labs/falkner"


# _host_experimenter + resolve_experimenter_path re-keying

def test_host_experimenter_reads_from_host_config(monkeypatch):
    """The experimenter id is read from the `experimenter` key of the host
    config TOML (the same key `exp_id` derives from)."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.delenv("EXPERIMENTER_ID", raising=False)
    assert os_utils._host_experimenter() == "Bartul"


def test_host_experimenter_raises_when_config_missing(monkeypatch, tmp_path):
    """A missing host config raises rather than guessing the experimenter."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.delenv("EXPERIMENTER_ID", raising=False)
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", tmp_path / "absent.toml")
    with pytest.raises(RuntimeError, match="Cannot read the host config"):
        os_utils._host_experimenter()


def test_host_experimenter_env_var_overrides_toml(monkeypatch):
    """An EXPERIMENTER_ID environment variable overrides the host TOML, so a
    cluster / headless run selects the experimenter without editing the TOML."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.setenv("EXPERIMENTER_ID", "Liza")
    assert os_utils._host_experimenter() == "Liza"


def test_host_experimenter_blank_env_var_falls_back_to_toml(monkeypatch):
    """A blank / whitespace EXPERIMENTER_ID is ignored; the host TOML wins."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.setenv("EXPERIMENTER_ID", "   ")
    assert os_utils._host_experimenter() == "Bartul"


def test_host_experimenter_env_var_used_when_config_missing(monkeypatch, tmp_path):
    """With EXPERIMENTER_ID set the host TOML is never read -- the override wins
    even when the config is absent (the cluster case: shipped TOML untouched)."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", tmp_path / "absent.toml")
    monkeypatch.setenv("EXPERIMENTER_ID", "Charlie")
    assert os_utils._host_experimenter() == "Charlie"


@pytest.mark.parametrize("system,expected", [
    ("Linux", "/mnt/falkner/Liza/EPHYS"),
    ("Darwin", "/Volumes/falkner/Liza/EPHYS"),
    ("Windows", "F:\\Liza\\EPHYS"),
])
def test_resolve_experimenter_path_rekeys_then_translates(as_os, monkeypatch, system, expected):
    """`resolve_experimenter_path` re-keys the shipped experimenter name in the
    path to the host experimenter, THEN translates the leading mount root to the
    host OS. A non-default host ('Liza') proves the re-key actually happens."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", ["Liza"])
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_LIST_CACHE", [["Bartul", "Liza"]])
    as_os(system)
    assert os_utils.resolve_experimenter_path("/mnt/falkner/Bartul/EPHYS") == expected


def test_configure_path_os_translates_only(as_os, monkeypatch, tmp_path):
    """`configure_path` only OS-translates the leading mount root; it never reads
    the experimenter id (re-keying is `resolve_experimenter_path`'s job) — proven
    by pointing the host config at an absent file and asserting no error."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", tmp_path / "absent.toml")
    as_os("Darwin")
    assert os_utils.configure_path("/mnt/falkner/SomeUser/Data/s1") == "/Volumes/falkner/SomeUser/Data/s1"


# resolve_data_root

def test_resolve_data_root_reads_key_and_delegates(as_os, monkeypatch, tmp_path):
    """`resolve_data_root` reads the requested key from the `data_roots` block
    and resolves it via `resolve_experimenter_path` (which re-keys the shipped
    experimenter name and OS-translates). A non-default host ('Liza') proves the
    re-key flows through."""
    settings = tmp_path / "analyses_settings.json"
    settings.write_text('{"data_roots": {"ephys_root": "/mnt/falkner/Bartul/EPHYS"}}')
    monkeypatch.setattr(os_utils, "_ANALYSES_SETTINGS_PATH", settings)
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", ["Liza"])
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_LIST_CACHE", [["Bartul", "Liza"]])
    as_os("Darwin")
    assert os_utils.resolve_data_root("ephys_root") == pathlib.Path("/Volumes/falkner/Liza/EPHYS")
    as_os("Linux")
    assert os_utils.resolve_data_root("ephys_root") == pathlib.Path("/mnt/falkner/Liza/EPHYS")


def test_resolve_data_root_reads_shipped_block(as_os):
    """The shipped analyses_settings.json defines the `data_roots` keys the
    analysis tools rely on; spot-check one resolves on the host OS (the shipped
    `Bartul` re-keyed to the host experimenter, which is `Bartul` here)."""
    as_os("Linux")
    assert os_utils.resolve_data_root("catalog_path") == pathlib.Path(
        "/mnt/falkner/Bartul/EPHYS/unit_catalog.csv"
    )


# _host_lab_shares (single-source table read from the host config TOML)

def test_host_lab_shares_reads_from_host_config(monkeypatch):
    """The live table is read from the lab_shares/file_server entries of the host
    config TOML and expanded to full roots, with falkner first."""
    monkeypatch.setattr(os_utils, "_HOST_SHARES_CACHE", [])
    shares, file_server = os_utils._host_lab_shares()
    assert file_server == "cup"
    assert [s["name"] for s in shares][:2] == ["falkner", "murthy"]
    # the TOML stores tokens ('F', '/mnt'); _host_lab_shares returns full roots
    assert shares[0]["windows"] == "F:" and shares[0]["linux"] == "/mnt/falkner"
    assert shares[0]["unc"] == r"\\cup\falkner"


def test_host_lab_shares_raises_when_config_missing(monkeypatch, tmp_path):
    """A missing host config raises (no silent fallback): path translation fails
    loud on an absent/broken config rather than guessing assumed shares."""
    monkeypatch.setattr(os_utils, "_HOST_SHARES_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", tmp_path / "absent.toml")
    with pytest.raises(RuntimeError, match="Cannot read the host config"):
        os_utils._host_lab_shares()


def test_host_lab_shares_raises_when_lab_shares_missing(monkeypatch, tmp_path):
    """A host config that parses but has no 'lab_shares' table raises KeyError."""
    cfg = tmp_path / "host.toml"
    cfg.write_text('file_server = "cup"\n')
    monkeypatch.setattr(os_utils, "_HOST_SHARES_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", cfg)
    with pytest.raises(KeyError, match="lab_shares"):
        os_utils._host_lab_shares()


def test_host_lab_shares_raises_when_file_server_missing(monkeypatch, tmp_path):
    """A host config with 'lab_shares' but no 'file_server' raises KeyError."""
    cfg = tmp_path / "host.toml"
    cfg.write_text(
        '[[lab_shares]]\n'
        'name = "falkner"\n'
        'windows = "F"\n'
        'darwin = "/Volumes"\n'
        'linux = "/mnt"\n'
        'cluster = "/mnt/cup/labs"\n'
    )
    monkeypatch.setattr(os_utils, "_HOST_SHARES_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", cfg)
    with pytest.raises(KeyError, match="file_server"):
        os_utils._host_lab_shares()


def test_expand_lab_share_builds_full_roots_from_tokens():
    """A token-form share expands to full per-OS roots + UNC (name appended,
    ':' added for Windows)."""
    expanded = os_utils.expand_lab_share(
        {"name": "falkner", "windows": "F", "darwin": "/Volumes", "linux": "/mnt", "cluster": "/mnt/cup/labs"},
        "cup")
    assert expanded == {
        "name": "falkner", "windows": "F:", "darwin": "/Volumes/falkner",
        "linux": "/mnt/falkner", "cluster": "/mnt/cup/labs/falkner", "unc": r"\\cup\falkner",
    }


def test_recording_destinations_derives_selected_only():
    """Destinations are <root>/<experimenter>/Data for each SELECTED lab, in both
    OS forms, in table order; unselected labs are skipped."""
    lab_shares = [
        {"name": "falkner", "windows": "F", "darwin": "/Volumes", "linux": "/mnt", "cluster": "/mnt/cup/labs"},
        {"name": "murthy", "windows": "M", "darwin": "/Volumes", "linux": "/mnt", "cluster": "/mnt/cup/labs"},
    ]
    lin, win = os_utils.recording_destinations(lab_shares, "cup", ["murthy"], "Bartul")
    assert lin == ["/mnt/murthy/Bartul/Data"]
    assert win == ["M:\\Bartul\\Data"]


# configure_path

@pytest.mark.parametrize("system,pa,expected", [
    # to Linux
    ("Linux", "F:\\Bartul\\Data\\s1", "/mnt/falkner/Bartul/Data/s1"),
    ("Linux", "/Volumes/falkner/Bartul/Data/s1", "/mnt/falkner/Bartul/Data/s1"),
    ("Linux", "M:\\a\\b", "/mnt/murthy/a/b"),
    ("Linux", "/Volumes/murthy/a", "/mnt/murthy/a"),
    # to Windows
    ("Windows", "/mnt/falkner/Bartul/x", "F:\\Bartul\\x"),
    ("Windows", "/Volumes/falkner/Bartul/x", "F:\\Bartul\\x"),
    # murthy -> Windows: BROKEN before the mount-table fix (produced \\mnt\\murthy)
    ("Windows", "/mnt/murthy/a", "M:\\a"),
    # to Darwin
    ("Darwin", "/mnt/falkner/Bartul/x", "/Volumes/falkner/Bartul/x"),
    ("Darwin", "F:\\Bartul\\x", "/Volumes/falkner/Bartul/x"),
])
def test_configure_path_translations(as_os, system, pa, expected):
    as_os(system)
    assert os_utils.configure_path(pa) == expected


def test_configure_path_does_not_corrupt_embedded_substring(as_os):
    # The old all-occurrence `.replace("mnt", "Volumes")` mangled inner tokens.
    as_os("Darwin")
    assert (os_utils.configure_path("/mnt/falkner/exp_mnt_2025/data")
            == "/Volumes/falkner/exp_mnt_2025/data")


@pytest.mark.parametrize("system,pa", [
    ("Linux", "/mnt/falkner/already/native"),     # already in host-OS form
    ("Darwin", "/Volumes/murthy/already"),         # already in host-OS form
    ("Linux", "/home/user/not/a/share"),           # unrelated path
    ("Windows", "C:\\Windows\\System32"),          # unrelated drive
    ("Linux", "/mnt/other_lab/x"),                 # /mnt but not a known share
])
def test_configure_path_passthrough(as_os, system, pa):
    as_os(system)
    assert os_utils.configure_path(pa) == pa


def test_configure_path_unknown_os_passthrough(as_os):
    as_os("SunOS")
    assert os_utils.configure_path("F:\\Bartul\\x") == "F:\\Bartul\\x"


def test_configure_path_bare_root(as_os):
    as_os("Linux")
    assert os_utils.configure_path("F:") == "/mnt/falkner"


# ephys_base_for_data_root
#
# Path-component logic only (no platform.system() branch), so these run under
# the host OS's pathlib; POSIX inputs are used since the suite runs on
# POSIX CI/dev machines, and results are compared Path-to-Path for robustness.

@pytest.mark.parametrize("data_root,expected", [
    # canonical single-'Data' session roots -> sibling EPHYS base (byte-identical
    # to the old str(parent).replace('Data','EPHYS') on these real paths)
    ("/mnt/falkner/Bartul/Data/20230101_120000", "/mnt/falkner/Bartul/EPHYS"),
    ("/Volumes/falkner/Bartul/Data/20230101_120000", "/Volumes/falkner/Bartul/EPHYS"),
    ("/mnt/murthy/Lab/Data/sess_01", "/mnt/murthy/Lab/EPHYS"),
])
def test_ephys_base_for_data_root_maps_data_to_ephys(data_root, expected):
    assert os_utils.ephys_base_for_data_root(data_root) == pathlib.Path(expected)


def test_ephys_base_for_data_root_does_not_corrupt_lookalike_component():
    # 'Database' must survive: only the exact 'Data' component is swapped, where
    # the old unanchored replace would have produced '.../EPHYSbase/EPHYS'.
    assert (os_utils.ephys_base_for_data_root("/mnt/falkner/Database/Data/sess")
            == pathlib.Path("/mnt/falkner/Database/EPHYS"))


def test_ephys_base_for_data_root_ignores_data_inside_session_id():
    # The session id lives below the parent, so a 'Data'-containing id is never
    # rewritten; the parent's 'Data' component still maps to 'EPHYS'.
    assert (os_utils.ephys_base_for_data_root("/mnt/falkner/Bartul/Data/Data_collection_01")
            == pathlib.Path("/mnt/falkner/Bartul/EPHYS"))


def test_ephys_base_for_data_root_swaps_only_final_data_component():
    # Old unanchored replace turned BOTH 'Data's into 'EPHYS'; the anchored
    # helper rewrites only the last, leaving the upper one intact.
    assert (os_utils.ephys_base_for_data_root("/mnt/falkner/Data/Bartul/Data/sess")
            == pathlib.Path("/mnt/falkner/Data/Bartul/EPHYS"))


def test_ephys_base_for_data_root_no_data_component_returns_parent_unchanged():
    # No 'Data' component -> parent returned unchanged (no accidental rewrite).
    assert (os_utils.ephys_base_for_data_root("/mnt/falkner/Bartul/Other/sess")
            == pathlib.Path("/mnt/falkner/Bartul/Other"))


# to_cluster_path
#
# Host-independent by design (a job may be submitted from any workstation), so
# no platform monkeypatching: every host-OS source form is matched regardless
# of the running OS.

@pytest.mark.parametrize("pa,expected", [
    # falkner from every host form -> identical to the old per-OS .replace
    ("F:\\Bartul\\Data\\s1", "/mnt/cup/labs/falkner/Bartul/Data/s1"),
    ("/Volumes/falkner/Bartul/Data/s1", "/mnt/cup/labs/falkner/Bartul/Data/s1"),
    ("/mnt/falkner/Bartul/Data/s1", "/mnt/cup/labs/falkner/Bartul/Data/s1"),
    # murthy from every host form -> the Windows M: case was BROKEN before
    # (left as 'M:/...'); Linux/macOS murthy already worked and stay the same
    ("M:\\a\\b", "/mnt/cup/labs/murthy/a/b"),
    ("/Volumes/murthy/a/b", "/mnt/cup/labs/murthy/a/b"),
    ("/mnt/murthy/a/b", "/mnt/cup/labs/murthy/a/b"),
])
def test_to_cluster_path_maps_every_host_form(pa, expected):
    assert os_utils.to_cluster_path(pa) == expected


def test_to_cluster_path_windows_murthy_regression():
    # Explicit regression for the dropped-murthy bug: a Windows murthy path must
    # resolve to the murthy cluster mount, not stay an unconverted drive path.
    assert os_utils.to_cluster_path("M:\\Lab\\Data\\s1") == "/mnt/cup/labs/murthy/Lab/Data/s1"


def test_to_cluster_path_does_not_corrupt_inner_mnt_token():
    # The old Linux .replace('mnt', 'mnt/cup/labs') mangled any inner 'mnt';
    # anchoring on the leading root leaves 'mnt_backup' intact.
    assert (os_utils.to_cluster_path("/mnt/falkner/mnt_backup/x")
            == "/mnt/cup/labs/falkner/mnt_backup/x")


def test_to_cluster_path_bare_root():
    assert os_utils.to_cluster_path("/mnt/murthy") == "/mnt/cup/labs/murthy"


@pytest.mark.parametrize("pa,expected", [
    ("", ""),                                       # empty -> unchanged
    ("/home/user/not/a/share", "/home/user/not/a/share"),
    ("C:\\Windows\\System32", "C:/Windows/System32"),   # only separators normalised
    ("/mnt/other_lab/x", "/mnt/other_lab/x"),       # /mnt but not a known share
])
def test_to_cluster_path_passthrough(pa, expected):
    assert os_utils.to_cluster_path(pa) == expected


# _on_cluster + cluster-aware configure_path

def test_on_cluster_true_when_cluster_mount_present(monkeypatch, tmp_path):
    """_on_cluster is a filesystem check (cluster-mount presence), not a host-name
    match: it is True exactly when the primary share's cluster root exists as a
    directory."""
    cluster_root = tmp_path / "cup" / "labs" / "falkner"
    cluster_root.mkdir(parents=True)
    monkeypatch.setattr(
        os_utils, "_HOST_SHARES_CACHE",
        [(({"name": "falkner", "cluster": str(cluster_root)},), "cup")],
    )
    monkeypatch.setattr(os_utils, "_ON_CLUSTER_CACHE", [])
    assert os_utils._on_cluster() is True


def test_on_cluster_false_when_cluster_mount_absent(monkeypatch, tmp_path):
    """No cluster mount on this filesystem -> not on the cluster; the host-OS form
    is used downstream."""
    monkeypatch.setattr(
        os_utils, "_HOST_SHARES_CACHE",
        [(({"name": "falkner", "cluster": str(tmp_path / "absent")},), "cup")],
    )
    monkeypatch.setattr(os_utils, "_ON_CLUSTER_CACHE", [])
    assert os_utils._on_cluster() is False


def test_configure_path_resolves_cluster_form_on_cluster(monkeypatch):
    """On the cluster (mount present) configure_path resolves ANY host form to the
    cluster mount via to_cluster_path, regardless of platform.system() -- this is
    what makes the canonical /mnt/falkner settings paths work inside a cluster job.
    An already-cluster-form path (e.g. an explicitly-passed session root) is left
    unchanged."""
    monkeypatch.setattr(os_utils, "_ON_CLUSTER_CACHE", [True])
    assert (os_utils.configure_path("/mnt/falkner/Bartul/spectrograms/sam")
            == "/mnt/cup/labs/falkner/Bartul/spectrograms/sam")
    assert (os_utils.configure_path("/mnt/cup/labs/murthy/Bartul/Data/s1")
            == "/mnt/cup/labs/murthy/Bartul/Data/s1")


def test_configure_path_keeps_host_form_off_cluster(as_os, monkeypatch):
    """Off the cluster (no cluster mount) configure_path keeps its host-OS
    behavior untouched -- the cluster branch is a strict addition."""
    monkeypatch.setattr(os_utils, "_ON_CLUSTER_CACHE", [False])
    as_os("Darwin")
    assert os_utils.configure_path("/mnt/falkner/Bartul/x") == "/Volumes/falkner/Bartul/x"


# atomic_output_path

def test_atomic_output_path_publishes_only_on_clean_exit(tmp_path):
    # The final file must not appear until the with-block exits cleanly; once
    # it does, it holds exactly what was written and no temp sibling remains.
    final = tmp_path / "data.txt"
    with os_utils.atomic_output_path(final) as tmp:
        tmp.write_text("new content")
        assert not final.exists()
    assert final.read_text() == "new content"
    assert list(tmp_path.glob(".data.txt.tmp*")) == []


def test_atomic_output_path_preserves_original_on_error(tmp_path):
    # A crash mid-write must leave the pre-existing file untouched (the whole
    # point of the helper) and clean up the temp sibling.
    final = tmp_path / "data.txt"
    final.write_text("original")
    with pytest.raises(RuntimeError):
        with os_utils.atomic_output_path(final) as tmp:
            tmp.write_text("half-written, never published")
            raise RuntimeError("boom")
    assert final.read_text() == "original"
    assert list(tmp_path.glob(".data.txt.tmp*")) == []


# first_match_or_raise

def test_first_match_returns_alphabetically_first(tmp_path):
    for name in ("c.txt", "a.txt", "b.txt"):
        (tmp_path / name).touch()
    assert os_utils.first_match_or_raise(tmp_path, "*.txt").name == "a.txt"


def test_first_match_recursive(tmp_path):
    sub = tmp_path / "deep"
    sub.mkdir()
    (sub / "found.json").touch()
    assert os_utils.first_match_or_raise(tmp_path, "*.json", recursive=True).name == "found.json"


def test_first_match_raises_with_label(tmp_path):
    with pytest.raises(FileNotFoundError, match="my-thing"):
        os_utils.first_match_or_raise(tmp_path, "*.nope", label="my-thing")


def test_first_match_raises_on_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        os_utils.first_match_or_raise(tmp_path / "absent", "*.txt")


def test_first_match_digit_prefix_excludes_speaker(tmp_path):
    # Mirrors assign_vocalizations: track files are timestamp-prefixed, so a
    # '[0-9]*' pattern selects the session track h5 and excludes 'speaker_*'.
    (tmp_path / "speaker_points3d_translated_rotated_metric.h5").touch()
    (tmp_path / "20230207213549_points3d_translated_rotated_metric.h5").touch()
    chosen = os_utils.first_match_or_raise(
        tmp_path, "[0-9]*_points3d_translated_rotated_metric.h5"
    )
    assert chosen.name == "20230207213549_points3d_translated_rotated_metric.h5"


# find_audio_mmap / parse_audio_mmap_name

def _write_band_mmap(root, band, value, n_samples=40, n_channels=3, session_id="230101120000"):
    """
    Description
    -----------
    Writes a constant-valued int16 memmap with the canonical name of one band
    into that band's folder under ``<root>/audio``.

    Parameters
    ----------
    root (pathlib.Path)
        Session root.
    band (str)
        ``'usv'`` or ``'broadband'``.
    value (int)
        Constant sample value (identifies the file when read back).
    n_samples, n_channels (int)
        Memmap shape.
    session_id (str)
        Recording id token of the name.

    Returns
    -------
    path (pathlib.Path)
        The written memmap.
    """

    folder = root / "audio" / os_utils.AUDIO_MMAP_BAND_FOLDERS[band]
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{session_id}_concatenated_audio_{folder.name}_250000_{n_samples}_{n_channels}_int16.mmap"
    np.full((n_samples, n_channels), value, dtype=np.int16).tofile(path)
    return path


def test_find_audio_mmap_returns_the_band_file(tmp_path):
    usv = _write_band_mmap(tmp_path, "usv", 1)
    broadband = _write_band_mmap(tmp_path, "broadband", 2)
    assert os_utils.find_audio_mmap(tmp_path, "usv") == usv
    assert os_utils.find_audio_mmap(str(tmp_path), "broadband") == broadband


def test_find_audio_mmap_ignores_strays_temporaries_and_other_folders(tmp_path):
    usv = _write_band_mmap(tmp_path, "usv", 1)
    # a stray unfiltered memmap that sorts first under audio/, a temporary
    # sibling, a non-canonical name and another band's name in the usv folder
    stray_dir = tmp_path / "audio" / "cropped_to_video"
    stray_dir.mkdir(parents=True)
    (stray_dir / "230101120000_concatenated_audio_cropped_to_video_250000_40_3_int16.mmap").write_bytes(b"\x00\x00")
    (usv.parent / f".{usv.name}.tmp-123").write_bytes(b"\x00\x00")
    (usv.parent / "sess_250000_40_3_int16.mmap").write_bytes(b"\x00\x00")
    (usv.parent / "230101120000_concatenated_audio_broadband_filtered_250000_40_3_int16.mmap").write_bytes(b"\x00\x00")
    assert os_utils.find_audio_mmap(tmp_path, "usv") == usv
    with pytest.raises(FileNotFoundError, match="broadband audio memmap"):
        os_utils.find_audio_mmap(tmp_path, "broadband")


def test_find_audio_mmap_raises_unless_exactly_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        os_utils.find_audio_mmap(tmp_path, "usv")
    (tmp_path / "audio" / "hpss_filtered").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no file matching"):
        os_utils.find_audio_mmap(tmp_path, "usv")
    _write_band_mmap(tmp_path, "usv", 1, n_samples=40)
    _write_band_mmap(tmp_path, "usv", 1, n_samples=41)
    with pytest.raises(RuntimeError, match="exactly one"):
        os_utils.find_audio_mmap(tmp_path, "usv")
    with pytest.raises(ValueError, match="Unknown audio band"):
        os_utils.find_audio_mmap(tmp_path, "sonic")


def test_parse_audio_mmap_name_reads_layout(tmp_path):
    path = _write_band_mmap(tmp_path, "broadband", 0, n_samples=123, n_channels=24)
    assert os_utils.parse_audio_mmap_name(path) == {
        "band": "broadband", "id": "230101120000", "sampling_rate": 250000,
        "n_samples": 123, "n_channels": 24, "dtype": "int16"}
    with pytest.raises(ValueError, match="not a concatenated audio memmap"):
        os_utils.parse_audio_mmap_name("sess_250000_40_3_int16.mmap")


def test_usv_readers_get_the_usv_file_when_both_bands_exist(tmp_path):
    # usv band = 1, broadband = 2, stray recursive-glob trap = 3
    _write_band_mmap(tmp_path, "usv", 1)
    _write_band_mmap(tmp_path, "broadband", 2)
    stray_dir = tmp_path / "audio" / "cropped_to_video"
    stray_dir.mkdir(parents=True)
    np.full((40, 3), 3, dtype=np.int16).tofile(stray_dir / "230101120000_concatenated_audio_cropped_to_video_250000_40_3_int16.mmap")
    audio, sampling_rate = open_hpss_audio(tmp_path)
    assert sampling_rate == 250000
    assert np.all(np.asarray(audio) == 1)
    video_audio, video_rate = load_audio_data(str(tmp_path))
    assert video_rate == 250000
    assert np.all(np.asarray(video_audio) == 1)


def test_every_audio_mmap_reader_asks_for_the_usv_band():
    # Static guard over the package: no reader may locate a memmap with a glob
    # ('*.mmap' patterns, recursive or not) and every find_audio_mmap call names
    # the 'usv' band, so no USV reader can be handed the broadband memmap; the
    # broadband writer (modify_files) is the only module allowed to name it.
    package_root = pathlib.Path(os_utils.__file__).parent
    readers = []
    for source_path in sorted(package_root.rglob("*.py")):
        source = source_path.read_text(encoding="utf-8")
        relative = source_path.relative_to(package_root).as_posix()
        assert not re.search(r"pattern\s*=\s*['\"][^'\"]*\.mmap", source), f"{relative} globs for a memmap"
        assert not re.search(r"glob\(\s*['\"][^'\"]*\.mmap", source), f"{relative} globs for a memmap"
        for band in re.findall(r"find_audio_mmap\([^)]*band\s*=\s*['\"](\w+)['\"]", source):
            readers.append(relative)
            assert band == "usv", f"{relative} asks for the {band!r} band"
    assert sorted(set(readers)) == [
        "analyses/build_naturalistic_usv_repository.py",
        "analyses/neuronal_coactivity_engine.py",
        "processing/assign_vocalizations.py",
        "processing/das_inference.py",
        "processing/generate_spectrograms.py",
        "visualizations/make_behavioral_videos.py",
        "visualizations/make_usv_spectrograms.py",
    ]


# newest_match_or_raise

def test_newest_match_picks_max_key(tmp_path):
    older = tmp_path / "old.bin"
    newer = tmp_path / "new.bin"
    older.touch()
    newer.touch()
    # explicit key avoids depending on filesystem ctime/mtime resolution
    chosen = os_utils.newest_match_or_raise(tmp_path, "*.bin", key=lambda p: p.name)
    assert chosen.name == "old.bin"  # "old" > "new" lexicographically


def test_newest_match_raises_when_empty(tmp_path):
    with pytest.raises(FileNotFoundError, match="newest"):
        os_utils.newest_match_or_raise(tmp_path, "*.bin", label="newest")


# QLVM category bundle / production cells / resolve_consolidated_h5_path

def test_qlvm_map_display_names_cover_every_map():
    """The GUI names every QLVM map, and only those: the regular map as 'QLVM',
    each conditional map by its conditioning property."""
    assert tuple(os_utils.QLVM_MAP_DISPLAY_NAMES) == os_utils.QLVM_MAPS
    assert list(os_utils.QLVM_MAP_DISPLAY_NAMES.values()) == ["QLVM", "duration", "entropy", "bandwidth", "loudness"]


def test_qlvm_category_constants_and_production_cells():
    """The category bundle is one constant on the regular map, its column is the one
    category column of every map, the qlvm_v3 reference-arrays convention is gone, and
    the production cells resolve from the package root (a map outside QLVM_MAPS raises)."""
    assert os_utils.QLVM_MAPS == ("qlvm", "qlvm_duration", "qlvm_entropy", "qlvm_bandwidth", "qlvm_loudness")
    assert os_utils.QLVM_REGULAR_MAP == os_utils.QLVM_CATEGORY_MAP == "qlvm"
    assert os_utils.QLVM_CATEGORY_COLUMN == "qlvm_category"
    assert os_utils.QLVM_CATEGORY_COLUMNS == ("qlvm_category",)
    assert os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY == (
        "/mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/regions/clustering_clean/category_bundle"
    )
    assert not hasattr(os_utils, "resolve_embedding_arrays_path")
    assert not hasattr(os_utils, "QLVM_REFERENCE_ARRAYS_DIRECTORY_NAME")
    package = "/mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/masked_clean"
    assert os_utils.qlvm_production_cell_directory("qlvm") == f"{package}/cell/masked"
    assert os_utils.qlvm_production_cell_directory("qlvm_duration") == f"{package}/conditionals/cell/duration"
    assert os_utils.qlvm_production_cell_directory("qlvm_entropy") == f"{package}/conditionals/cell/spectral_entropy"
    assert os_utils.qlvm_production_cell_directory("qlvm_bandwidth") == f"{package}/conditionals/cell/bandwidth"
    assert os_utils.qlvm_production_cell_directory("qlvm_loudness") == f"{package}/conditionals/cell/loudness"
    with pytest.raises(ValueError, match="qlvm_map must be one of"):
        os_utils.qlvm_production_cell_directory("qlvm_dur")
    with pytest.raises(ValueError, match="qlvm_map must be one of"):
        os_utils.qlvm_production_cell_directory("vae")
    assert os_utils.qlvm_cell_model_id(f"{package}/cell/masked") == "masked_clean/cell/masked"
    assert os_utils.qlvm_cell_model_id("F:\\Bartul\\x\\masked_clean\\cell\\masked") == "masked_clean/cell/masked"


def test_load_qlvm_category_bundle_reads_the_one_bundle(qlvm_category_bundle):
    """load_qlvm_category_bundle reads the bundle the module constant names: the label grid
    (1..k, [y, x]), density, pixel-centre axis, label positions as centres, names, the map
    and production cell it partitions, and a provenance line from build_config.json."""
    bundle = os_utils.load_qlvm_category_bundle()
    res = bundle["resolution"]
    assert bundle["directory"] == str(qlvm_category_bundle)
    assert bundle["label_grid"].shape == (res, res) and bundle["label_grid"].dtype == np.int64
    assert sorted(np.unique(bundle["label_grid"]).tolist()) == [1, 2, 3, 4]
    assert bundle["label_grid"][0, -1] == 2 and bundle["label_grid"][-1, 0] == 3  # [y, x]
    assert bundle["density"].shape == (res, res)
    np.testing.assert_allclose(bundle["axis"], (np.arange(res) + 0.5) / res)
    np.testing.assert_allclose(bundle["centers"], [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]])
    assert bundle["names"] == ["R-1", "R-2", "R-3", "R-4"]
    assert bundle["map"] == "qlvm"
    assert bundle["model_id"] == "masked_clean/cell/masked"
    assert "built 2026-10-03T12:00:00" in bundle["identity"] and "0123456789ab" in bundle["identity"]


def test_read_qlvm_category_bundle_refuses_a_malformed_bundle(tmp_path):
    """A missing file names the bundle; a grid whose labels are not the nomenclature's
    1..k, or grids of different shapes, are refused."""
    with pytest.raises(FileNotFoundError, match="category_grids.npz"):
        os_utils.read_qlvm_category_bundle(str(tmp_path / "empty"))
    directory = write_qlvm_category_bundle(tmp_path / "bad_labels")
    with np.load(directory / "category_grids.npz") as grids:
        arrays = {key: grids[key] for key in grids.files}
    arrays["label_grid"] = np.where(arrays["label_grid"] == 4, 5, arrays["label_grid"]).astype(np.int16)
    np.savez_compressed(directory / "category_grids.npz", **arrays)
    with pytest.raises(ValueError, match="not 1..4"):
        os_utils.read_qlvm_category_bundle(str(directory))
    directory = write_qlvm_category_bundle(tmp_path / "bad_shape")
    with np.load(directory / "category_grids.npz") as grids:
        arrays = {key: grids[key] for key in grids.files}
    arrays["density"] = arrays["density"][:-1]
    np.savez_compressed(directory / "category_grids.npz", **arrays)
    with pytest.raises(ValueError, match="must have the label grid's"):
        os_utils.read_qlvm_category_bundle(str(directory))


def test_resolve_consolidated_h5_picks_newest_and_skips_other_h5(tmp_path):
    (tmp_path / "qlvm_clusters_20260506.h5").write_bytes(b"x")  # different store -> must be ignored
    old = tmp_path / "spectrograms_old.h5"
    new = tmp_path / "spectrograms_new.h5"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    os.utime(old, (1, 1))
    os.utime(new, (10 ** 9, 10 ** 9))  # strictly newer mtime
    assert os_utils.resolve_consolidated_h5_path(str(tmp_path)) == str(new)


# order_usv_summary_columns

def test_order_usv_summary_columns_puts_known_columns_in_canonical_order():
    """Known columns follow USV_SUMMARY_COLUMN_ORDER whatever order they arrive in; unknown
    columns keep their relative order after them; values are untouched; absent canonical
    columns are not created."""
    table = pls.DataFrame({
        "custom_b": [1], "mean_freq_hz": [2.0], "usv_id": ["000000"], "squeak": [True],
        "start": [0.1], "custom_a": [3], "emitter": [None], "stop": [0.2], "qlvm1": [0.5],
        "squeak_end": [0.19], "chs_count": [3.0], "noise": [False],
    })
    ordered = os_utils.order_usv_summary_columns(table)
    assert ordered.columns == [
        "usv_id", "start", "stop", "noise", "squeak", "squeak_end", "emitter", "chs_count", "mean_freq_hz",
        "qlvm1", "custom_b", "custom_a",
    ]
    assert ordered.equals(table.select(ordered.columns))


def test_usv_summary_column_order_is_the_agreed_layout():
    """The agreed layout, in full: DAS event -> noise and vocal-class block -> emitter and DAS
    channel statistics -> acoustic features -> the regular QLVM map with its category -> the
    duration, spectral-entropy, bandwidth and loudness conditional maps (coordinates only) ->
    the squeak torus.
    No obsolete column is canonical."""
    assert list(os_utils.USV_SUMMARY_COLUMN_ORDER) == [
        "usv_id", "start", "stop", "duration",
        "noise", "noise_probability", "usv", "squeak", "p_usv", "p_squeak", "p_both", "squeak_start", "squeak_end",
        "emitter", "peak_amp_ch", "mean_amp_ch", "chs_count", "chs_detected",
        "mean_freq_hz", "peak_freq_hz", "freq_bandwidth_hz", "mean_amplitude", "max_amplitude", "loudness_db",
        "spectral_entropy", "mask_number",
        "qlvm1", "qlvm2", "qlvm_category",
        "qlvm_duration1", "qlvm_duration2", "qlvm_entropy1", "qlvm_entropy2",
        "qlvm_bandwidth1", "qlvm_bandwidth2", "qlvm_loudness1", "qlvm_loudness2",
        "qlvm_squeak1", "qlvm_squeak2",
    ]
    assert len(set(os_utils.USV_SUMMARY_COLUMN_ORDER)) == len(os_utils.USV_SUMMARY_COLUMN_ORDER)
    assert not set(os_utils.USV_SUMMARY_OBSOLETE_COLUMNS) & set(os_utils.USV_SUMMARY_COLUMN_ORDER)
    assert os_utils.QLVM_SUMMARY_MAP_PREFIXES == ("qlvm", "qlvm_duration", "qlvm_entropy", "qlvm_bandwidth", "qlvm_loudness")
    assert {"squeak_probability", "squeak_frame_runs", "call_class", "squeak_spans", "n_squeaks"} <= set(
        os_utils.USV_SUMMARY_OBSOLETE_COLUMNS
    )


def test_obsolete_usv_summary_columns_picks_listed_and_confidence_columns():
    """The listed obsolete columns and every *_category_agreement / *_category_uncertain column
    are picked, in the given order; canonical and unknown columns are not."""
    columns = [
        "usv_id", "qlvm_supercategory", "qlvm_category", "qlvm_dur_category", "qlvm_mf1", "custom",
        "qlvm_x_category_uncertain", "qlvm_category_agreement", "qlvm_loud_supercategory", "qlvm_model",
        "qlvm_ent1", "qlvm_entropy1",
    ]
    assert os_utils.obsolete_usv_summary_columns(columns) == [
        "qlvm_supercategory", "qlvm_dur_category", "qlvm_mf1", "qlvm_x_category_uncertain",
        "qlvm_category_agreement", "qlvm_loud_supercategory", "qlvm_model", "qlvm_ent1",
    ]


def test_obsolete_columns_never_match_the_current_conditional_maps():
    """The retired short-prefix maps (qlvm_dur*, qlvm_ent*) and the retired v3 maps
    (qlvm_bw*, qlvm_loud*, qlvm_mf*) are obsolete, but the current conditional maps whose
    names merely START with a retired prefix (qlvm_duration*, qlvm_entropy*,
    qlvm_bandwidth*, qlvm_loudness*) are canonical and never picked: obsolete names are
    matched exactly, not as prefixes."""
    current = [f"{prefix}{axis}" for prefix in os_utils.QLVM_SUMMARY_MAP_PREFIXES for axis in (1, 2)]
    retired = [
        "qlvm_dur1", "qlvm_dur2", "qlvm_ent1", "qlvm_ent2", "qlvm_bw1", "qlvm_bw2",
        "qlvm_loud1", "qlvm_loud2", "qlvm_mf1", "qlvm_mf2",
    ]
    assert os_utils.obsolete_usv_summary_columns(current) == []
    assert os_utils.obsolete_usv_summary_columns(retired + current) == retired
    assert not set(current) & set(os_utils.USV_SUMMARY_OBSOLETE_COLUMNS)
    assert set(current) <= set(os_utils.USV_SUMMARY_COLUMN_ORDER)


def test_tidy_usv_summary_columns_drops_obsolete_and_reorders():
    """An old-layout summary loses every obsolete column, the rest comes out in canonical
    order with unknown columns last, the values of kept columns are unchanged, and the report
    says what changed."""
    table = pls.DataFrame({
        "usv_id": ["000000", "000001"], "start": [0.1, 1.0], "stop": [0.2, 1.1], "duration": [0.1, 0.1],
        "peak_amp_ch": [3.0, 4.0], "emitter": ["m", None], "noise": [False, True],
        "squeak_probability": [0.1, None], "mean_freq_hz": [50000.0, 60000.0],
        "qlvm1": [0.1, 0.2], "qlvm2": [0.3, 0.4], "qlvm_category": [1, 2], "qlvm_supercategory": [1, 1],
        "qlvm_dur1": [0.5, 0.6], "qlvm_dur2": [0.7, 0.8], "qlvm_dur_category": [3, 4],
        "qlvm_duration1": [0.5, 0.6], "qlvm_duration2": [0.7, 0.8], "qlvm_bandwidth1": [0.1, 0.2],
        "qlvm_loudness2": [0.3, 0.4],
        "qlvm_dur_supercategory": [1, 2], "qlvm_mf1": [0.0, 0.0], "qlvm_mf2": [0.0, 0.0],
        "qlvm_mf_category": [1, 1], "qlvm_bw1": [0.0, 0.0], "qlvm_loud_supercategory": [2, 2],
        "qlvm_category_agreement": [0.9, 0.5], "qlvm_category_uncertain": [False, True],
        "qlvm_squeak1": [None, 0.2], "qlvm_squeak2": [None, 0.3], "custom": [1, 2],
    })
    tidied, report = os_utils.tidy_usv_summary_columns(table)
    assert tidied.columns == [
        "usv_id", "start", "stop", "duration", "noise", "emitter", "peak_amp_ch", "mean_freq_hz",
        "qlvm1", "qlvm2", "qlvm_category", "qlvm_duration1", "qlvm_duration2", "qlvm_bandwidth1",
        "qlvm_loudness2", "qlvm_squeak1", "qlvm_squeak2", "custom",
    ]
    assert tidied.equals(table.select(tidied.columns))
    assert report["dropped"] == [
        "squeak_probability", "qlvm_supercategory", "qlvm_dur1", "qlvm_dur2", "qlvm_dur_category",
        "qlvm_dur_supercategory", "qlvm_mf1", "qlvm_mf2",
        "qlvm_mf_category", "qlvm_bw1", "qlvm_loud_supercategory", "qlvm_category_agreement",
        "qlvm_category_uncertain",
    ]
    assert report["unknown"] == ["custom"]
    assert report["columns_before"] == table.columns
    assert report["columns_after"] == tidied.columns
    assert report["changed"] is True


def test_tidy_usv_summary_columns_leaves_a_canonical_summary_alone():
    """A summary already in the canonical layout comes back identical and unchanged."""
    table = pls.DataFrame({"usv_id": ["000000"], "start": [0.1], "stop": [0.2], "qlvm1": [0.5], "qlvm2": [0.6]})
    tidied, report = os_utils.tidy_usv_summary_columns(table)
    assert tidied.equals(table)
    assert report["dropped"] == []
    assert report["changed"] is False


# derive_spectrogram_model_paths

def test_derive_spectrogram_model_paths_fills_empties_from_root():
    settings = {
        "spectrograms_root": "/mnt/falkner/Bartul/spectrograms",
        "generate_masks": {
            "sam2_model_dir": "", "sam2_model_cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
            "sam2_model_path": "", "yolo_weights": "",
        },
        "infer_qlvm_latents": {"model_cells": {}, "masking_type": "none"},
        "infer_qlvm_squeak_latents": {"model_cell_directory": ""},
        "assign_qlvm_categories": {"category_directory": "", "coordinate_prefix": ""},
        "detect_usv_squeaks": {"squeak_model_path": ""},
        "detect_usv_noise": {"noise_model_path": ""},
    }
    returned = os_utils.derive_spectrogram_model_paths(settings)
    root = "/mnt/falkner/Bartul/spectrograms"
    assert returned is settings  # mutates in place and returns the same dict
    assert settings["generate_masks"]["sam2_model_dir"] == f"{root}/sam"
    assert settings["generate_masks"]["sam2_model_path"] == f"{root}/sam/checkpoint.pt"
    assert settings["generate_masks"]["yolo_weights"] == f"{root}/sam/best.pt"
    # the SAM2 config NAME is never derived
    assert settings["generate_masks"]["sam2_model_cfg"] == "configs/sam2.1/sam2.1_hiera_b+.yaml"
    # the OLD in-house QLVM model under <root>/qlvm is never derived (no key of it is
    # written); the production mapping is, with the masking and time stretch its cells
    # were trained with
    assert set(settings["infer_qlvm_latents"]) == {"model_cells", "masking_type", "time_stretch"}
    package = "/mnt/falkner/Bartul/PC_transfer/qlvm_time_stretch/masked_clean"
    assert settings["infer_qlvm_latents"]["model_cells"] == {
        "qlvm": f"{package}/cell/masked",
        "qlvm_duration": f"{package}/conditionals/cell/duration",
        "qlvm_entropy": f"{package}/conditionals/cell/spectral_entropy",
        "qlvm_bandwidth": f"{package}/conditionals/cell/bandwidth",
        "qlvm_loudness": f"{package}/conditionals/cell/loudness",
    }
    assert list(settings["infer_qlvm_latents"]["model_cells"]) == [
        "qlvm", "qlvm_duration", "qlvm_entropy", "qlvm_bandwidth", "qlvm_loudness",
    ]
    assert settings["infer_qlvm_latents"]["masking_type"] == "sam"
    assert settings["infer_qlvm_latents"]["time_stretch"] is True
    # the production squeak QLVM cell (time-stretched, unmasked, no floor)
    assert settings["infer_qlvm_squeak_latents"]["model_cell_directory"] == (
        "/mnt/falkner/Bartul/PC_transfer/qlvm_final/squeaks/cell/stretch_nofloor"
    )
    assert settings["detect_usv_squeaks"]["squeak_model_path"] == f"{root}/squeak/usv_squeak_timemil_ens5_n2476_20260930_reviewed.pt"
    assert settings["detect_usv_noise"]["noise_model_path"] == f"{root}/noise/noise_timemil_ens5_n4680_20260926.pt"
    # assign-qlvm-categories labels with the category bundle every figure draws, on the regular map
    assert settings["assign_qlvm_categories"] == {
        "category_directory": os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY,
        "coordinate_prefix": "qlvm",
    }


def test_derive_spectrogram_model_paths_preserves_explicit_overrides():
    settings = {
        "spectrograms_root": "/mnt/falkner/Bartul/spectrograms",
        "generate_masks": {
            "sam2_model_dir": "", "sam2_model_cfg": "cfg.yaml",
            "sam2_model_path": "/custom/elsewhere/checkpoint.pt", "yolo_weights": "",
        },
        "infer_qlvm_latents": {"model_cells": {"qlvm_x": "/custom/cell"}, "masking_type": "none", "time_stretch": False},
        "infer_qlvm_squeak_latents": {"model_cell_directory": "/custom/squeak_cell"},
        "assign_qlvm_categories": {"category_directory": "/custom/bundle", "coordinate_prefix": "qlvm_x"},
        "detect_usv_squeaks": {"squeak_model_path": "/custom/squeak.pt"},
        "detect_usv_noise": {"noise_model_path": ""},
    }
    os_utils.derive_spectrogram_model_paths(settings)
    # explicit (non-empty) paths win
    assert settings["generate_masks"]["sam2_model_path"] == "/custom/elsewhere/checkpoint.pt"
    assert settings["detect_usv_squeaks"]["squeak_model_path"] == "/custom/squeak.pt"
    assert settings["detect_usv_noise"]["noise_model_path"] == "/mnt/falkner/Bartul/spectrograms/noise/noise_timemil_ens5_n4680_20260926.pt"
    # empty siblings are still derived from the root
    assert settings["generate_masks"]["sam2_model_dir"] == "/mnt/falkner/Bartul/spectrograms/sam"
    # explicit model cells are left entirely alone, their masking type and time stretch included
    assert settings["infer_qlvm_latents"]["model_cells"] == {"qlvm_x": "/custom/cell"}
    assert settings["infer_qlvm_latents"]["masking_type"] == "none"
    assert settings["infer_qlvm_latents"]["time_stretch"] is False
    assert settings["infer_qlvm_squeak_latents"]["model_cell_directory"] == "/custom/squeak_cell"
    assert settings["assign_qlvm_categories"] == {"category_directory": "/custom/bundle", "coordinate_prefix": "qlvm_x"}


@pytest.mark.parametrize("configured", [
    {"qlvm_x": "/custom/cell"},
    {"qlvm": "/custom/regular_cell", "qlvm_duration": "/custom/duration_cell"},
])
def test_derive_spectrogram_model_paths_keeps_configured_qlvm_models(configured):
    """Configured model_cells (one cell or several) win over the derived production
    mapping, and their masking_type is not overridden."""
    settings = {
        "spectrograms_root": "/mnt/falkner/Bartul/spectrograms",
        "generate_masks": {"sam2_model_dir": "", "sam2_model_path": "", "yolo_weights": ""},
        "infer_qlvm_latents": {"model_cells": dict(configured), "masking_type": "sam"},
        "infer_qlvm_squeak_latents": {"model_cell_directory": ""},
        "assign_qlvm_categories": {"category_directory": "", "coordinate_prefix": ""},
        "detect_usv_squeaks": {"squeak_model_path": ""},
        "detect_usv_noise": {"noise_model_path": ""},
    }
    os_utils.derive_spectrogram_model_paths(settings)
    assert settings["infer_qlvm_latents"]["model_cells"] == configured
    assert settings["infer_qlvm_latents"]["masking_type"] == "sam"


def test_derived_qlvm_model_cells_pass_model_cell_validation():
    """The derived production prefixes write their coordinates only, which the model_cells
    validator (which forbids overwriting other summary columns) accepts. Every canonical
    map coordinate column (qlvm, qlvm_duration, qlvm_entropy, qlvm_bandwidth, qlvm_loudness)
    is writable by a model_cells run and nothing else is: the regular map's qlvm_category
    (written by assign-qlvm-categories) and the squeak torus coordinates are not."""
    settings = {
        "spectrograms_root": "/mnt/falkner/Bartul/spectrograms",
        "generate_masks": {"sam2_model_dir": "", "sam2_model_path": "", "yolo_weights": ""},
        "infer_qlvm_latents": {"model_cells": {}, "masking_type": "sam"},
        "infer_qlvm_squeak_latents": {"model_cell_directory": ""},
        "assign_qlvm_categories": {"category_directory": "", "coordinate_prefix": ""},
        "detect_usv_squeaks": {"squeak_model_path": ""},
        "detect_usv_noise": {"noise_model_path": ""},
    }
    os_utils.derive_spectrogram_model_paths(settings)
    validate_model_cells(settings["infer_qlvm_latents"]["model_cells"].items())
    reserved = model_cell_reserved_columns()
    canonical_qlvm = {c for c in os_utils.USV_SUMMARY_COLUMN_ORDER if c.startswith("qlvm")}
    map_coordinates = {f"{prefix}{axis}" for prefix in os_utils.QLVM_SUMMARY_MAP_PREFIXES for axis in (1, 2)}
    assert canonical_qlvm - reserved == map_coordinates
    assert {"qlvm_category", "qlvm_squeak1", "qlvm_squeak2"} <= reserved
    validate_model_cells([(prefix, "/cell") for prefix in os_utils.QLVM_SUMMARY_MAP_PREFIXES])


def test_derive_spectrogram_model_paths_noop_when_root_absent():
    # legacy settings: granular paths set directly, no spectrograms_root key -> must not KeyError
    settings = {
        "generate_masks": {
            "sam2_model_dir": "/legacy/sam", "sam2_model_path": "/legacy/sam/ck.pt",
            "yolo_weights": "/legacy/sam/best.pt",
        },
        "infer_qlvm_latents": {"model_cells": {"qlvm": "/legacy/cell"}},
    }
    returned = os_utils.derive_spectrogram_model_paths(settings)
    assert returned is settings
    assert settings["generate_masks"]["sam2_model_dir"] == "/legacy/sam"
    assert settings["infer_qlvm_latents"]["model_cells"] == {"qlvm": "/legacy/cell"}
    # an empty root is likewise a no-op (no garbage "/sam" paths)
    s2 = {"spectrograms_root": "", "generate_masks": {"sam2_model_dir": "x"}}
    assert os_utils.derive_spectrogram_model_paths(s2)["generate_masks"]["sam2_model_dir"] == "x"


def test_resolve_consolidated_h5_raises_when_no_store(tmp_path):
    (tmp_path / "qlvm_clusters_x.h5").write_bytes(b"x")  # only a non-consolidated .h5
    with pytest.raises(FileNotFoundError, match="consolidated"):
        os_utils.resolve_consolidated_h5_path(str(tmp_path))


def test_resolve_pooled_embeddings_cache_convention(tmp_path):
    """The cache name is versioned by the QLVM model, so the v3 cache never overwrites
    the old model's pooled_embeddings.parquet."""
    base = tmp_path / "spectrograms"
    assert os_utils.resolve_pooled_embeddings_cache(str(base)) == str(
        base / "embeddings" / "pooled_embeddings_qlvmv3.parquet"
    )


# wait_for_subprocesses

class FakePopen:
    """Minimal subprocess.Popen stand-in with a scriptable poll()."""

    def __init__(self, returncode=0, finish_after=0, terminate_code=-15):
        self._returncode = returncode
        self._finish_after = finish_after
        self._polls = 0
        self._terminated = False
        self._terminate_code = terminate_code
        self.terminated = False
        self.killed = False

    def poll(self):
        if self._terminated:
            return self._terminate_code
        if self._polls >= self._finish_after:
            return self._returncode
        self._polls += 1
        return None

    def terminate(self):
        self.terminated = True
        self._terminated = True

    def kill(self):
        self.killed = True
        self._terminated = True


def test_wait_empty_is_noop():
    assert os_utils.wait_for_subprocesses([], max_seconds=1, label="x") == []


def test_wait_all_success_returns_codes():
    procs = [FakePopen(returncode=0), FakePopen(returncode=0)]
    assert os_utils.wait_for_subprocesses(procs, max_seconds=5, label="ok") == [0, 0]


def test_wait_nonzero_logged_and_optionally_raised():
    logs = []
    procs = [FakePopen(returncode=0), FakePopen(returncode=2)]
    codes = os_utils.wait_for_subprocesses(procs, max_seconds=5, label="mix",
                                           message_output=logs.append)
    assert codes == [0, 2]
    assert any("non-zero" in m for m in logs)
    with pytest.raises(RuntimeError, match="failed"):
        os_utils.wait_for_subprocesses([FakePopen(returncode=2)], max_seconds=5,
                                       label="mix", raise_on_nonzero=True)


def test_wait_timeout_terminates_and_raises():
    hung = FakePopen(finish_after=10 ** 9)  # never finishes on its own
    with pytest.raises(TimeoutError, match="did not finish"):
        os_utils.wait_for_subprocesses([hung], max_seconds=0.2, label="hang",
                                       poll_interval_s=0.05)
    assert hung.terminated  # was asked to terminate on timeout


def test_wait_timeout_no_raise_returns_status():
    hung = FakePopen(finish_after=10 ** 9)
    start = time.monotonic()
    status = os_utils.wait_for_subprocesses([hung], max_seconds=0.2, label="hang",
                                            poll_interval_s=0.05, raise_on_timeout=False)
    assert time.monotonic() - start < 5  # grace period did not block (terminate killed it)
    assert hung.terminated
    assert len(status) == 1


def test_resolve_analyses_setting_reads_block_key():
    """`resolve_analyses_setting` returns a value straight from
    `analyses_settings.json[block][key]` (used by the neuropixels tools to source
    the probe→hemisphere map and Kilosort version from settings)."""

    assert os_utils.resolve_analyses_setting(
        'npx_histology_ibl_alignment_export', 'probe_to_hemisphere'
    ) == {'imec0': 'R', 'imec1': 'L'}
    assert int(os_utils.resolve_analyses_setting('npx_spike_quality_metrics', 'kilosort_version')) == 4


def test_resolve_modeling_setting_reads_block_key():
    """`resolve_modeling_setting` returns a value straight from
    `modeling_settings.json[block][key]` (used to source the former hard-coded
    modeling defaults — selection p-value, ECE bins, session-split tolerance)."""

    assert float(os_utils.resolve_modeling_setting('model_params', 'selection_p_val')) == 0.01
    assert int(os_utils.resolve_modeling_setting('diagnostics', 'ece_n_bins')) == 10
    assert float(os_utils.resolve_modeling_setting('model_validation', 'session_split_initial_tolerance')) == 0.05


def test_rebase_experimenter_in_paths_rewrites_bounded_components_only():
    """`rebase_experimenter_in_paths` rewrites an experimenter name only where it
    is a full path component or the entire string, re-keying it to `exp_id`;
    unbounded substrings and non-matching leaves are left untouched, and it
    recurses through dicts/lists while leaving non-strings alone."""

    obj = {
        "ephys": "/mnt/falkner/Bartul/EPHYS",     # bounded component -> rebased
        "who": "Bartul",                          # entire string -> rebased
        "lookalike": "/mnt/Bartuli/data",         # 'Bartul' is an unbounded substring -> untouched
        "label": "A84I Linux",                    # unrelated non-path string -> untouched
        "roots": ["/mnt/falkner/Bartul/Data", 42],  # recurse list; ints pass through
    }
    out = os_utils.rebase_experimenter_in_paths(obj, experimenter_list=["Bartul"], exp_id="Annegret")
    assert out["ephys"] == "/mnt/falkner/Annegret/EPHYS"
    assert out["who"] == "Annegret"
    assert out["lookalike"] == "/mnt/Bartuli/data"
    assert out["label"] == "A84I Linux"
    assert out["roots"] == ["/mnt/falkner/Annegret/Data", 42]


def test_rebase_experimenter_in_paths_is_idempotent():
    """Re-keying a path that already uses `exp_id` is a no-op (safe to apply on
    every load / experimenter change)."""

    already = {"p": "/mnt/falkner/Annegret/EPHYS"}
    out = os_utils.rebase_experimenter_in_paths(already, experimenter_list=["Bartul"], exp_id="Annegret")
    assert out == already


# call classes


def test_call_class_mask_derives_classes_from_the_two_booleans():
    """Classes come from both booleans, never from usv alone: pure USV = usv & ~squeak, pure squeak =
    squeak & ~usv, both = usv & squeak; null rows (noise) are in no class; the flags may arrive as
    Boolean, as text, or as an all-null column; an unknown class or a summary without the booleans
    raises."""
    table = pls.DataFrame(
        {"usv": [True, None, True, False, True], "squeak": [False, None, True, True, False]},
        schema={"usv": pls.Boolean, "squeak": pls.Boolean},
    )
    assert os_utils.call_class_mask(table, ["usv"], "s").to_list() == [True, False, False, False, True]
    assert os_utils.pure_usv_mask(table, "s").to_list() == [True, False, False, False, True]
    assert os_utils.pure_squeak_mask(table, "s").to_list() == [False, False, False, True, False]
    assert os_utils.both_mask(table, "s").to_list() == [False, False, True, False, False]
    assert os_utils.squeak_bearing_mask(table, "s").to_list() == [False, False, True, True, False]
    assert os_utils.call_class_mask(table, os_utils.squeak_class_selection("squeak+both"), "s").to_list() == [False, False, True, True, False]
    assert os_utils.call_class_mask(table, os_utils.squeak_class_selection("both"), "s").to_list() == [False, False, True, False, False]
    as_text = table.with_columns(pls.col("usv").cast(pls.String), pls.col("squeak").cast(pls.String))
    assert os_utils.pure_usv_mask(as_text, "s").to_list() == [True, False, False, False, True]
    all_null = pls.DataFrame({"usv": [None, None], "squeak": [None, None]})
    assert os_utils.call_class_mask(all_null, ["usv", "squeak", "both"], "s").to_list() == [False, False]
    with pytest.raises(ValueError, match="unknown call class"):
        os_utils.call_class_mask(table, ["squeaks"], "s")
    with pytest.raises(KeyError, match="detect-usv-squeaks"):
        os_utils.call_class_mask(pls.DataFrame({"squeak": [True]}), ["usv"], "old_summary.csv")
    with pytest.raises(ValueError, match="squeak class selection"):
        os_utils.squeak_class_selection("usv")
