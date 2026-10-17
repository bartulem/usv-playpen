"""
Tests for usv_playpen.os_utils: cross-OS path translation (find_base_path /
configure_path), the Data->EPHYS sibling-tree mapping (ephys_base_for_data_root),
the deterministic glob helpers (first_match_or_raise / newest_match_or_raise),
the band-explicit audio memmap lookup (find_audio_mmap / parse_audio_mmap_name,
including that every reader gets the 'ultrasonic' memmap when both bands exist)
and the subprocess-group waiter (wait_for_subprocesses).

`configure_path`/`find_base_path` are exercised under all three target OSs by
monkeypatching `os_utils.platform.system`; the cases include the two bugs the
mount-mapping table was introduced to fix (all-occurrence substring corruption,
and the previously-unhandled `murthy` share).
"""

import json
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
        ``'ultrasonic'`` or ``'broadband'``.
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
    usv = _write_band_mmap(tmp_path, "ultrasonic", 1)
    broadband = _write_band_mmap(tmp_path, "broadband", 2)
    assert os_utils.find_audio_mmap(tmp_path, "ultrasonic") == usv
    assert os_utils.find_audio_mmap(str(tmp_path), "broadband") == broadband


def test_find_audio_mmap_ignores_strays_temporaries_and_other_folders(tmp_path):
    usv = _write_band_mmap(tmp_path, "ultrasonic", 1)
    # a stray unfiltered memmap that sorts first under audio/, a temporary
    # sibling, a non-canonical name and another band's name in the usv folder
    stray_dir = tmp_path / "audio" / "cropped_to_video"
    stray_dir.mkdir(parents=True)
    (stray_dir / "230101120000_concatenated_audio_cropped_to_video_250000_40_3_int16.mmap").write_bytes(b"\x00\x00")
    (usv.parent / f".{usv.name}.tmp-123").write_bytes(b"\x00\x00")
    (usv.parent / "sess_250000_40_3_int16.mmap").write_bytes(b"\x00\x00")
    (usv.parent / "230101120000_concatenated_audio_broadband_filtered_250000_40_3_int16.mmap").write_bytes(b"\x00\x00")
    assert os_utils.find_audio_mmap(tmp_path, "ultrasonic") == usv
    with pytest.raises(FileNotFoundError, match="broadband audio memmap"):
        os_utils.find_audio_mmap(tmp_path, "broadband")


def test_find_audio_mmap_raises_unless_exactly_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        os_utils.find_audio_mmap(tmp_path, "ultrasonic")
    (tmp_path / "audio" / "hpss_filtered").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no file matching"):
        os_utils.find_audio_mmap(tmp_path, "ultrasonic")
    _write_band_mmap(tmp_path, "ultrasonic", 1, n_samples=40)
    _write_band_mmap(tmp_path, "ultrasonic", 1, n_samples=41)
    with pytest.raises(RuntimeError, match="exactly one"):
        os_utils.find_audio_mmap(tmp_path, "ultrasonic")
    with pytest.raises(ValueError, match="Unknown audio band"):
        os_utils.find_audio_mmap(tmp_path, "sonic")


def test_parse_audio_mmap_name_reads_layout(tmp_path):
    path = _write_band_mmap(tmp_path, "broadband", 0, n_samples=123, n_channels=24)
    assert os_utils.parse_audio_mmap_name(path) == {
        "band": "broadband", "id": "230101120000", "sampling_rate": 250000,
        "n_samples": 123, "n_channels": 24, "dtype": "int16"}
    with pytest.raises(ValueError, match="not a concatenated audio memmap"):
        os_utils.parse_audio_mmap_name("sess_250000_40_3_int16.mmap")


def test_usv_readers_get_the_ultrasonic_file_when_both_bands_exist(tmp_path):
    # ultrasonic band = 1, broadband = 2, stray recursive-glob trap = 3
    _write_band_mmap(tmp_path, "ultrasonic", 1)
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


def test_every_audio_mmap_reader_asks_for_the_ultrasonic_band():
    # Static guard over the package: no reader may locate a memmap with a glob
    # ('*.mmap' patterns, recursive or not) and every find_audio_mmap call names
    # the 'ultrasonic' band, so no USV reader can be handed the broadband memmap; the
    # broadband writer (modify_files) is the only module allowed to name it. The
    # vocalocator (assign_vocalizations) is the one reader whose band is a setting,
    # vocalocator.vcl_audio_band ('ultrasonic' by default), checked separately below.
    package_root = pathlib.Path(os_utils.__file__).parent
    readers = []
    for source_path in sorted(package_root.rglob("*.py")):
        source = source_path.read_text(encoding="utf-8")
        relative = source_path.relative_to(package_root).as_posix()
        assert not re.search(r"pattern\s*=\s*['\"][^'\"]*\.mmap", source), f"{relative} globs for a memmap"
        assert not re.search(r"glob\(\s*['\"][^'\"]*\.mmap", source), f"{relative} globs for a memmap"
        for band in re.findall(r"find_audio_mmap\([^)]*band\s*=\s*['\"](\w+)['\"]", source):
            readers.append(relative)
            assert band == "ultrasonic", f"{relative} asks for the {band!r} band"
    assert sorted(set(readers)) == [
        "analyses/build_naturalistic_usv_repository.py",
        "analyses/neuronal_coactivity_engine.py",
        "processing/das_inference.py",
        "processing/generate_spectrograms.py",
        "visualizations/make_behavioral_videos.py",
        "visualizations/make_usv_spectrograms.py",
    ]
    vocalocator_source = (package_root / "processing" / "assign_vocalizations.py").read_text(encoding="utf-8")
    assert "find_audio_mmap(root_directory=self.root_directory, band=vcl_audio_band)" in vocalocator_source
    assert "self.input_parameter_dict['vocalocator']['vcl_audio_band']" in vocalocator_source


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


def test_newest_match_searches_subfolders_only_when_recursive(tmp_path):
    """The default glob sees the top level only, so a newer file in a subfolder is
    not picked; recursive=True walks the tree and picks it; a missing root raises
    naming the pattern."""
    top = tmp_path / "top.bin"
    nested = tmp_path / "sub" / "zz_nested.bin"
    nested.parent.mkdir()
    top.touch()
    nested.touch()
    assert os_utils.newest_match_or_raise(tmp_path, "*.bin", key=lambda p: p.name) == top
    assert os_utils.newest_match_or_raise(tmp_path, "*.bin", key=lambda p: p.name, recursive=True) == nested
    with pytest.raises(FileNotFoundError, match="no match for glob"):
        os_utils.newest_match_or_raise(tmp_path / "sub", "top.bin")
    with pytest.raises(FileNotFoundError, match=r"\*\.bin: search root"):
        os_utils.newest_match_or_raise(tmp_path / "absent", "*.bin")


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
    assert os_utils.QLVM_CATEGORY_BUNDLE_DIRECTORY == "/mnt/falkner/Bartul/spectrograms/categories"
    assert not hasattr(os_utils, "resolve_embedding_arrays_path")
    assert not hasattr(os_utils, "QLVM_REFERENCE_ARRAYS_DIRECTORY_NAME")
    package = "/mnt/falkner/Bartul/spectrograms/qlvm"
    for qlvm_map in os_utils.QLVM_MAPS:
        # one folder per map, named by its column prefix
        assert os_utils.qlvm_production_cell_directory(qlvm_map) == f"{package}/{qlvm_map}"
    with pytest.raises(ValueError, match="qlvm_map must be one of"):
        os_utils.qlvm_production_cell_directory("qlvm_dur")
    with pytest.raises(ValueError, match="qlvm_map must be one of"):
        os_utils.qlvm_production_cell_directory("vae")
    assert os_utils.qlvm_cell_model_id(f"{package}/qlvm") == "spectrograms/qlvm/qlvm"
    assert os_utils.qlvm_cell_model_id("F:\\Bartul\\x\\masked_clean\\cell\\masked") == "masked_clean/cell/masked"


def test_qlvm_map_cell_directory_covers_the_usv_and_squeak_maps():
    """Every map a decode can work on resolves to its own production cell: the USV maps
    through qlvm_production_cell_directory, the squeak map to the squeak package cell;
    a prefix naming no map raises."""
    assert os_utils.QLVM_SQUEAK_MAP == "qlvm_squeak"
    assert os_utils.QLVM_DECODER_MAPS == os_utils.QLVM_MAPS + ("qlvm_squeak",)
    for qlvm_map in os_utils.QLVM_MAPS:
        assert os_utils.qlvm_map_cell_directory(qlvm_map) == os_utils.qlvm_production_cell_directory(qlvm_map)
    assert os_utils.qlvm_map_cell_directory("qlvm_squeak") == "/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_squeak"
    with pytest.raises(ValueError, match="qlvm_map must be one of"):
        os_utils.qlvm_map_cell_directory("vae")


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
    assert bundle["model_id"] == "spectrograms/qlvm/qlvm"
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
    directory = write_qlvm_category_bundle(tmp_path / "not_square")
    with np.load(directory / "category_grids.npz") as grids:
        arrays = {key: grids[key] for key in grids.files}
    for key in ("label_grid", "agreement", "density"):
        arrays[key] = arrays[key][:, :-2]
    np.savez_compressed(directory / "category_grids.npz", **arrays)
    with pytest.raises(ValueError, match="must be a square grid"):
        os_utils.read_qlvm_category_bundle(str(directory))


def test_read_qlvm_category_bundle_returns_every_file_it_reads(tmp_path):
    """The agreement grid, the descriptions and both JSON files come back as read,
    and the provenance line carries the positions hash cut at 12 characters and the
    vocalization and category counts."""
    directory = write_qlvm_category_bundle(tmp_path / "bundle")
    with np.load(directory / "category_grids.npz") as grids:
        agreement = grids["agreement"]
    with (directory / "category_nomenclature.json").open() as handle:
        nomenclature = json.load(handle)
    with (directory / "build_config.json").open() as handle:
        build_config = json.load(handle)
    bundle = os_utils.read_qlvm_category_bundle(str(directory))
    np.testing.assert_array_equal(bundle["agreement"], agreement)
    assert bundle["agreement"].dtype == np.float64
    assert bundle["descriptions"] == [str(category["description"]) for category in nomenclature["categories"]]
    assert bundle["nomenclature"] == nomenclature
    assert bundle["build_config"] == build_config
    sha = build_config["positions_file_sha256"]
    assert f"positions sha256 {sha[:12]}," in bundle["identity"]
    assert f"{int(build_config['n_vocalizations'])} vocalizations, 4 categories)" in bundle["identity"]


def test_read_qlvm_category_bundle_reads_a_bundle_built_before_the_rename(tmp_path):
    """A bundle built before the count was renamed records it as n_calls; it still reads,
    with the same count in the provenance line."""
    directory = write_qlvm_category_bundle(tmp_path / "legacy")
    config = json.loads((directory / "build_config.json").read_text())
    config["n_calls"] = config.pop("n_vocalizations")
    (directory / "build_config.json").write_text(json.dumps(config))
    bundle = os_utils.read_qlvm_category_bundle(str(directory))
    assert f"{config['n_calls']} vocalizations, 4 categories)" in bundle["identity"]


def test_resolve_consolidated_h5_picks_newest_and_skips_other_h5(tmp_path):
    (tmp_path / "qlvm_clusters_20260506.h5").write_bytes(b"x")  # different store -> must be ignored
    old = tmp_path / "spectrograms_old.h5"
    new = tmp_path / "spectrograms_new.h5"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    os.utime(old, (1, 1))
    os.utime(new, (10 ** 9, 10 ** 9))  # strictly newer mtime
    assert os_utils.resolve_consolidated_h5_path(str(tmp_path)) == str(new)


def test_resolve_squeak_spectrogram_store_picks_newest_squeak_store_only(tmp_path):
    """The squeak store resolver takes the newest squeak_spectrograms_*.h5 by mtime and
    never the consolidated USV store beside it, even when that one is newer; with no
    squeak store it raises naming the store."""
    usv_store = tmp_path / "spectrograms_production.h5"
    old = tmp_path / "squeak_spectrograms_old.h5"
    new = tmp_path / "squeak_spectrograms_new.h5"
    for path, stamp in ((old, 10 ** 8), (new, 10 ** 9), (usv_store, 2 * 10 ** 9)):
        path.write_bytes(b"x")
        os.utime(path, (stamp, stamp))
    assert os_utils.resolve_squeak_spectrogram_store_path(str(tmp_path)) == str(new)
    os.utime(old, (3 * 10 ** 9, 3 * 10 ** 9))
    assert os_utils.resolve_squeak_spectrogram_store_path(str(tmp_path)) == str(old)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="squeak spectrogram store"):
        os_utils.resolve_squeak_spectrogram_store_path(str(empty))


def test_cell_cluster_directory_finds_either_layout(tmp_path):
    """A v3 cell's inference/clusters_<level> is found, and preferred over a v2
    cluster/<level> in the same cell; a v2-only cell gives cluster/<level>; a cell
    with neither raises naming the level and both candidates."""
    v3 = tmp_path / "v3"
    (v3 / "inference" / "clusters_fine").mkdir(parents=True)
    (v3 / "cluster" / "fine").mkdir(parents=True)
    assert os_utils.cell_cluster_directory(v3, "fine") == v3 / "inference" / "clusters_fine"
    v2 = tmp_path / "v2"
    (v2 / "cluster" / "coarse").mkdir(parents=True)
    assert os_utils.cell_cluster_directory(v2, "coarse") == v2 / "cluster" / "coarse"
    with pytest.raises(FileNotFoundError, match=r"no fine cluster folder \(.*clusters_fine or .*cluster/fine\)"):
        os_utils.cell_cluster_directory(v2, "fine")


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
        "noise", "usv", "squeak",
        "noise_probability", "usv_probability", "squeak_probability", "overlap_probability",
        "squeak_start", "squeak_end",
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
    assert {"squeak_frame_runs", "call_class", "squeak_spans", "n_squeaks"} <= set(os_utils.USV_SUMMARY_OBSOLETE_COLUMNS)
    assert os_utils.USV_SUMMARY_RENAMED_COLUMNS == {
        "p_usv": "usv_probability", "p_squeak": "squeak_probability", "p_both": "overlap_probability",
    }
    assert set(os_utils.USV_SUMMARY_RENAMED_COLUMNS.values()) <= set(os_utils.USV_SUMMARY_COLUMN_ORDER)
    assert not set(os_utils.USV_SUMMARY_RENAMED_COLUMNS) & set(os_utils.USV_SUMMARY_COLUMN_ORDER)


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
        "squeak_frame_runs": [1, None], "mean_freq_hz": [50000.0, 60000.0],
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
        "squeak_frame_runs", "qlvm_supercategory", "qlvm_dur1", "qlvm_dur2", "qlvm_dur_category",
        "qlvm_dur_supercategory", "qlvm_mf1", "qlvm_mf2",
        "qlvm_mf_category", "qlvm_bw1", "qlvm_loud_supercategory", "qlvm_category_agreement",
        "qlvm_category_uncertain",
    ]
    assert report["unknown"] == ["custom"]
    assert report["columns_before"] == table.columns
    assert report["columns_after"] == tidied.columns
    assert report["changed"] is True


def test_tidy_usv_summary_columns_renames_the_first_probability_names():
    """A summary written with p_usv / p_squeak / p_both gets them renamed to usv_probability /
    squeak_probability / overlap_probability with every value kept, placed after
    noise_probability in the canonical vocal-class block; nothing is dropped."""
    table = pls.DataFrame({
        "usv_id": ["000000", "000001"], "start": [0.1, 1.0], "stop": [0.2, 1.1], "duration": [0.1, 0.1],
        "noise": [False, False], "noise_probability": [0.01, 0.02], "usv": [True, False], "squeak": [False, True],
        "p_usv": [0.9, 0.1], "p_squeak": [0.05, 0.8], "p_both": [0.05, 0.1], "squeak_start": [None, 1.0],
        "squeak_end": [None, 1.05], "emitter": ["m", None],
    })
    tidied, report = os_utils.tidy_usv_summary_columns(table)
    assert tidied.columns == [
        "usv_id", "start", "stop", "duration", "noise", "usv", "squeak", "noise_probability",
        "usv_probability", "squeak_probability", "overlap_probability", "squeak_start", "squeak_end", "emitter",
    ]
    assert report["renamed"] == {"p_usv": "usv_probability", "p_squeak": "squeak_probability", "p_both": "overlap_probability"}
    assert report["dropped"] == []
    assert report["changed"] is True
    for old, new in os_utils.USV_SUMMARY_RENAMED_COLUMNS.items():
        assert tidied[new].to_list() == table[old].to_list()
    again, second = os_utils.tidy_usv_summary_columns(tidied)
    assert again.equals(tidied)
    assert second["renamed"] == {}
    assert second["changed"] is False


def test_tidy_usv_summary_columns_drops_an_old_name_whose_new_name_is_present():
    """When a summary holds both an old probability name and its new name, the new column is
    kept as it is and the old one is dropped (reported under dropped, not renamed)."""
    table = pls.DataFrame({
        "usv_id": ["000000"], "start": [0.1], "stop": [0.2], "p_squeak": [0.3], "squeak_probability": [0.6],
    })
    tidied, report = os_utils.tidy_usv_summary_columns(table)
    assert tidied.columns == ["usv_id", "start", "stop", "squeak_probability"]
    assert tidied["squeak_probability"].to_list() == [0.6]
    assert report["renamed"] == {}
    assert report["dropped"] == ["p_squeak"]


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
    # the cells are not derived from spectrograms_root (no single-model key is written):
    # the production mapping comes from the module constants, one folder per prefix
    # under the lab's spectrograms/qlvm folder, with the masking and time stretch its
    # cells were trained with
    assert set(settings["infer_qlvm_latents"]) == {"model_cells", "masking_type", "time_stretch"}
    package = "/mnt/falkner/Bartul/spectrograms/qlvm"
    assert settings["infer_qlvm_latents"]["model_cells"] == {
        "qlvm": f"{package}/qlvm",
        "qlvm_duration": f"{package}/qlvm_duration",
        "qlvm_entropy": f"{package}/qlvm_entropy",
        "qlvm_bandwidth": f"{package}/qlvm_bandwidth",
        "qlvm_loudness": f"{package}/qlvm_loudness",
    }
    assert list(settings["infer_qlvm_latents"]["model_cells"]) == [
        "qlvm", "qlvm_duration", "qlvm_entropy", "qlvm_bandwidth", "qlvm_loudness",
    ]
    assert settings["infer_qlvm_latents"]["masking_type"] == "sam"
    assert settings["infer_qlvm_latents"]["time_stretch"] is True
    # the production squeak QLVM cell (time-stretched, unmasked, no floor)
    assert settings["infer_qlvm_squeak_latents"]["model_cell_directory"] == "/mnt/falkner/Bartul/spectrograms/qlvm/qlvm_squeak"
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
    """The cache is the production maps' pooled_embeddings_production.parquet under
    embeddings/ (os_utils.POOLED_EMBEDDINGS_CACHE_NAME), so a cache of other maps under
    another name is never read or overwritten."""
    base = tmp_path / "spectrograms"
    assert os_utils.resolve_pooled_embeddings_cache(str(base)) == str(
        base / "embeddings" / "pooled_embeddings_production.parquet"
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


def test_rebase_experimenter_in_paths_rewrites_every_bounded_occurrence():
    """Every bounded occurrence is rewritten, not only the first: an unbounded
    occurrence earlier in the path does not hide a later component, a name in
    two components is rewritten in both, backslash separators bound a name like
    slashes, a name at the start of a path is bounded by the string start, and
    every listed name is handled (one that does not occur does not stop the
    others)."""

    names = ["Charlie", "Bartul", "Dexter"]
    rebase = os_utils.rebase_experimenter_in_paths
    assert rebase("/mnt/Bartul_old/Bartul/data", names, "Ivan") == "/mnt/Bartul_old/Ivan/data"
    assert rebase("/mnt/Bartul/x/Bartul", names, "Ivan") == "/mnt/Ivan/x/Ivan"
    assert rebase("F:\\Bartul\\EPHYS", names, "Ivan") == "F:\\Ivan\\EPHYS"
    assert rebase("F:\\xBartul\\EPHYS", names, "Ivan") == "F:\\xBartul\\EPHYS"
    assert rebase("Bartul/EPHYS", names, "Ivan") == "Ivan/EPHYS"
    assert rebase("/mnt/Dexter/a/Bartul", names, "Ivan") == "/mnt/Ivan/a/Ivan"
    assert rebase("/mnt/Bartul/a", names, "Bartul") == "/mnt/Bartul/a"
    assert rebase("/mnt/Bart/a", ["Bart"], "Bartul") == "/mnt/Bartul/a"


def test_rebase_experimenter_in_paths_is_idempotent():
    """Re-keying a path that already uses `exp_id` is a no-op (safe to apply on
    every load / experimenter change)."""

    already = {"p": "/mnt/falkner/Annegret/EPHYS"}
    out = os_utils.rebase_experimenter_in_paths(already, experimenter_list=["Bartul"], exp_id="Annegret")
    assert out == already


# vocal classes


def _timed_summary() -> pls.DataFrame:
    """Four segments: a pure USV, a pure squeak whose extent starts before its segment, an
    overlap segment whose squeak starts 40 ms after its USV onset and outlasts it, and a noise
    row (null flags). Rows are deliberately out of onset order once re-timed."""

    return pls.DataFrame({
        "start": [1.00, 2.00, 0.50, 3.00],
        "stop": [1.10, 2.10, 0.60, 3.10],
        "usv": [True, False, True, None],
        "squeak": [False, True, True, None],
        "squeak_start": [None, 1.98, 0.54, None],
        "squeak_end": [None, 2.05, 0.65, None],
    })


@pytest.mark.parametrize(("classes", "overlap_times"), [
    (("squeak",), (0.54, 0.65)),
    (("squeak", "overlap"), (0.54, 0.65)),
    (("usv", "overlap"), (0.50, 0.60)),
    (("usv",), (0.50, 0.60)),
    (("overlap",), (0.50, 0.65)),
    (("usv", "squeak"), (0.50, 0.65)),
    (("usv", "squeak", "overlap"), (0.50, 0.65)),
])
def test_vocal_event_times_follow_the_class_group(classes, overlap_times):
    """A pure squeak takes its squeak extent and a pure USV its segment; an overlap segment takes
    the squeak extent in a squeak-only group, its segment in a USV-only group, and the union
    otherwise; noise keeps its segment; rows come back sorted by the new onset. A USV-only group
    re-times nothing (its rows never need the extent), so a pure squeak the caller did not filter
    out keeps its segment there."""

    timed = os_utils.vocal_event_times(_timed_summary(), classes, "s")
    rows = dict(zip(zip(timed["usv"].to_list(), timed["squeak"].to_list(), strict=True),
                    zip(timed["start"].to_list(), timed["stop"].to_list(), strict=True), strict=True))
    assert rows[(True, False)] == (1.00, 1.10)
    usv_only = "squeak" not in classes and overlap_times == (0.50, 0.60)
    assert rows[(False, True)] == ((2.00, 2.10) if usv_only else (1.98, 2.05))
    assert rows[(True, True)] == overlap_times
    assert rows[(None, None)] == (3.00, 3.10)
    assert timed["start"].to_list() == sorted(timed["start"].to_list())


def test_vocal_event_times_fail_loudly_without_the_squeak_extent():
    """A group that can time a row by the squeak extent raises without ``squeak_start`` /
    ``squeak_end`` (even when no such row is present) or when a squeak row lacks its extent; a
    USV-only group needs neither."""

    no_extent = _timed_summary().drop("squeak_start", "squeak_end")
    for classes in (("squeak",), ("overlap",), ("usv", "squeak")):
        with pytest.raises(KeyError, match="squeak_start"):
            os_utils.vocal_event_times(no_extent.filter(pls.col("squeak") == False), classes, "s")  # noqa: E712
    for classes in (("usv",), ("usv", "overlap")):
        assert os_utils.vocal_event_times(no_extent, classes, "s")["start"].to_list() == [0.50, 1.00, 2.00, 3.00]
    holed = _timed_summary().with_columns(pls.lit(None, dtype=pls.Float64).alias("squeak_start"))
    with pytest.raises(ValueError, match="squeak-bearing row"):
        os_utils.vocal_event_times(holed, ("squeak",), "s")
    with pytest.raises(ValueError, match="unknown vocal class"):
        os_utils.vocal_event_times(_timed_summary(), ("chirp",), "s")


def test_vocal_occupancy_times_span_segment_and_squeak():
    """The silence reference blocks every squeak-bearing row over the union of its segment and
    its squeak extent, every other row over its segment, and needs the extent columns."""

    timed = os_utils.vocal_occupancy_times(_timed_summary(), "s")
    assert timed["start"].to_list() == [0.50, 1.00, 1.98, 3.00]
    assert timed["stop"].to_list() == [0.65, 1.10, 2.10, 3.10]


def test_event_times_keep_duration_on_the_event():
    """A re-timed row's ``duration`` is the length of its new times; an untouched row keeps the
    stored value exactly."""

    table = _timed_summary().with_columns((pls.col("stop") - pls.col("start")).alias("duration"))
    timed = os_utils.vocal_event_times(table, ("squeak",), "s")
    by_flags = dict(zip(zip(timed["usv"].to_list(), timed["squeak"].to_list(), strict=True),
                        timed["duration"].to_list(), strict=True))
    assert by_flags[(False, True)] == pytest.approx(2.05 - 1.98)
    assert by_flags[(True, True)] == pytest.approx(0.65 - 0.54)
    assert by_flags[(True, False)] == table["duration"][0]
    occupied = os_utils.vocal_occupancy_times(table, "s")
    assert (occupied["duration"] - (occupied["stop"] - occupied["start"])).abs().max() == pytest.approx(0.0)
    with pytest.raises(KeyError, match="silence reference"):
        os_utils.vocal_occupancy_times(_timed_summary().drop("squeak_end"), "s")


def test_vocal_class_mask_derives_classes_from_the_two_booleans():
    """Classes come from both booleans, never from usv alone: pure USV = usv & ~squeak, pure squeak =
    squeak & ~usv, overlap = usv & squeak; null rows (noise) are in no class; the flags may arrive as
    Boolean, as text, or as an all-null column; an unknown class or a summary without the booleans
    raises."""
    table = pls.DataFrame(
        {"usv": [True, None, True, False, True], "squeak": [False, None, True, True, False]},
        schema={"usv": pls.Boolean, "squeak": pls.Boolean},
    )
    assert os_utils.vocal_class_mask(table, ["usv"], "s").to_list() == [True, False, False, False, True]
    assert os_utils.pure_usv_mask(table, "s").to_list() == [True, False, False, False, True]
    assert os_utils.pure_squeak_mask(table, "s").to_list() == [False, False, False, True, False]
    assert os_utils.overlap_mask(table, "s").to_list() == [False, False, True, False, False]
    assert os_utils.squeak_bearing_mask(table, "s").to_list() == [False, False, True, True, False]
    assert os_utils.vocal_class_mask(table, os_utils.squeak_class_selection("squeak+overlap"), "s").to_list() == [False, False, True, True, False]
    assert os_utils.vocal_class_mask(table, os_utils.squeak_class_selection("overlap"), "s").to_list() == [False, False, True, False, False]
    as_text = table.with_columns(pls.col("usv").cast(pls.String), pls.col("squeak").cast(pls.String))
    assert os_utils.pure_usv_mask(as_text, "s").to_list() == [True, False, False, False, True]
    all_null = pls.DataFrame({"usv": [None, None], "squeak": [None, None]})
    assert os_utils.vocal_class_mask(all_null, ["usv", "squeak", "overlap"], "s").to_list() == [False, False]
    with pytest.raises(ValueError, match="unknown vocal class"):
        os_utils.vocal_class_mask(table, ["squeaks"], "s")
    with pytest.raises(KeyError, match="detect-usv-squeaks"):
        os_utils.vocal_class_mask(pls.DataFrame({"squeak": [True]}), ["usv"], "old_summary.csv")
    with pytest.raises(ValueError, match="squeak class selection"):
        os_utils.squeak_class_selection("usv")


def test_noise_mask_and_drop_noise_usvs_take_any_csv_encoding(tmp_path):
    """The noise flag is read through text, like the usv / squeak booleans: a Boolean column, the
    strings "true" / "false" (what polars.read_csv infers when the schema-inference rows hold no
    value) and an all-null column all work, a null counts as not noise, and a summary without the
    column raises. The CSV round trip reproduces the String typing a plain cast(Boolean) or ``~``
    refused."""
    as_bool = pls.DataFrame({"noise": [True, False, None]}, schema={"noise": pls.Boolean})
    assert os_utils.noise_mask(as_bool, "s").to_list() == [True, False, False]
    as_text = as_bool.with_columns(pls.col("noise").cast(pls.String))
    assert os_utils.noise_mask(as_text, "s").to_list() == [True, False, False]
    summary_path = tmp_path / "s_usv_summary.csv"
    pls.DataFrame(
        {"start": [0.1, 0.2, 0.3], "noise": [None, None, None]},
        schema={"start": pls.Float64, "noise": pls.Boolean},
    ).write_csv(summary_path)
    all_null = pls.read_csv(summary_path)
    assert all_null["noise"].dtype == pls.String
    kept, n_dropped = os_utils.drop_noise_usvs(all_null, "s", message_output=lambda *_args: None)
    assert (kept.height, n_dropped) == (3, 0)
    kept, n_dropped = os_utils.drop_noise_usvs(as_text, "s", message_output=lambda *_args: None)
    assert (kept.height, n_dropped) == (2, 1)
    with pytest.raises(KeyError, match="detect-usv-noise"):
        os_utils.noise_mask(pls.DataFrame({"start": [0.1]}), "old_summary.csv")


def _random_timed_summary(rng: np.random.Generator, n_rows: int) -> pls.DataFrame:
    """A random summary: segments with random classes (pure USV, pure squeak, overlap, noise) and, on
    every squeak-bearing row, a squeak extent that may start before, inside or after its segment."""

    start = np.sort(rng.uniform(0.0, 100.0, n_rows))
    stop = start + rng.uniform(0.005, 0.3, n_rows)
    kind = rng.integers(0, 4, n_rows)  # 0 pure USV, 1 pure squeak, 2 overlap, 3 noise
    usv = [True if k in (0, 2) else (False if k == 1 else None) for k in kind]
    squeak = [True if k in (1, 2) else (False if k == 0 else None) for k in kind]
    bearing = np.isin(kind, (1, 2))
    squeak_start = np.where(bearing, start + rng.uniform(-0.01, 0.2, n_rows), np.nan)
    squeak_end = np.where(bearing, squeak_start + rng.uniform(0.002, 0.2, n_rows), np.nan)
    return pls.DataFrame({
        "start": start, "stop": stop, "duration": stop - start,
        "usv": pls.Series(usv, dtype=pls.Boolean), "squeak": pls.Series(squeak, dtype=pls.Boolean),
        "squeak_start": pls.Series(squeak_start).fill_nan(None),
        "squeak_end": pls.Series(squeak_end).fill_nan(None),
        "row": np.arange(n_rows),
    })


@pytest.mark.parametrize("seed", range(20))
def test_event_and_occupancy_times_are_self_consistent_on_random_summaries(seed):
    """Property check over random summaries and every class group: re-timing keeps every row
    exactly once, sorts by onset, never ends an event before it starts, keeps `duration` equal
    to `stop - start` on re-timed rows, leaves pure USVs and noise rows untouched, and the
    occupancy span of every row contains both its segment and its event times."""

    rng = np.random.default_rng(seed)
    table = _random_timed_summary(rng, 200)
    by_row = {row: (start, stop) for row, start, stop in zip(table["row"], table["start"], table["stop"], strict=True)}
    occupancy = os_utils.vocal_occupancy_times(table, "s")
    occupancy_by_row = {row: (start, stop) for row, start, stop in
                        zip(occupancy["row"], occupancy["start"], occupancy["stop"], strict=True)}
    for classes in (("usv",), ("squeak",), ("overlap",), ("usv", "overlap"), ("squeak", "overlap"),
                    ("usv", "squeak"), ("usv", "squeak", "overlap")):
        timed = os_utils.vocal_event_times(table, classes, "s")
        assert sorted(timed["row"].to_list()) == list(range(table.height))
        assert timed["start"].to_list() == sorted(timed["start"].to_list())
        assert (timed["stop"] >= timed["start"]).all()
        np.testing.assert_allclose(timed["duration"], timed["stop"] - timed["start"], atol=1e-12)
        for row, squeak, start, stop in zip(timed["row"], timed["squeak"], timed["start"], timed["stop"], strict=True):
            if not squeak:
                assert (start, stop) == by_row[row]
            occupancy_start, occupancy_stop = occupancy_by_row[row]
            assert occupancy_start <= min(start, by_row[row][0]) + 1e-12
            assert occupancy_stop >= max(stop, by_row[row][1]) - 1e-12


# host-config error paths, cluster detection fallback


def test_host_lab_shares_raises_when_lab_shares_table_is_empty(monkeypatch, tmp_path):
    """A host config whose ``lab_shares`` entry is present but EMPTY raises the same
    KeyError as a missing one: an empty table would leave every path untranslated
    (and ``find_base_path`` without a default share), so it must fail loud rather
    than return an empty share tuple."""
    cfg = tmp_path / "host.toml"
    cfg.write_text('lab_shares = []\nfile_server = "cup"\n')
    monkeypatch.setattr(os_utils, "_HOST_SHARES_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", cfg)
    with pytest.raises(KeyError, match="has no non-empty 'lab_shares' table"):
        os_utils._host_lab_shares()
    assert os_utils._HOST_SHARES_CACHE == []


def test_host_experimenter_raises_naming_the_missing_entry(monkeypatch, tmp_path):
    """A host config that parses but has no ``experimenter`` key raises a KeyError
    whose message names the missing entry and the consequence (experimenter-scoped
    paths cannot be re-keyed), and nothing is cached."""
    cfg = tmp_path / "host.toml"
    cfg.write_text('file_server = "cup"\n')
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.delenv("EXPERIMENTER_ID", raising=False)
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", cfg)
    with pytest.raises(KeyError, match="has no 'experimenter' entry; cannot re-key experimenter-scoped data paths"):
        os_utils._host_experimenter()
    assert os_utils._HOST_EXPERIMENTER_CACHE == []


def test_host_experimenter_list_reads_and_fails_loud(monkeypatch, tmp_path):
    """The experimenter roster is read from the ``experimenter_list`` key and cached
    as a list; an unreadable host config raises RuntimeError naming the roster
    lookup, and a config without the key raises KeyError naming the entry."""
    cfg = tmp_path / "host.toml"
    cfg.write_text('experimenter_list = ["Ann", "Bob"]\n')
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_LIST_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", cfg)
    assert os_utils._host_experimenter_list() == ["Ann", "Bob"]
    assert os_utils._HOST_EXPERIMENTER_LIST_CACHE == [["Ann", "Bob"]]

    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_LIST_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", tmp_path / "absent.toml")
    with pytest.raises(RuntimeError, match=r"Cannot read the host config .* experimenter roster for path re-keying: FileNotFoundError"):
        os_utils._host_experimenter_list()

    no_key = tmp_path / "no_key.toml"
    no_key.write_text('experimenter = "Ann"\n')
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", no_key)
    with pytest.raises(KeyError, match="has no 'experimenter_list' entry; cannot re-key experimenter-scoped paths"):
        os_utils._host_experimenter_list()


def test_on_cluster_false_and_cached_when_host_config_unreadable(monkeypatch, tmp_path):
    """When the share table cannot be resolved (no host config on this machine) the
    host is treated as a workstation: ``_on_cluster`` returns exactly False (not a
    falsy None) and caches that False, so a second query answers False without
    re-reading the config."""
    monkeypatch.setattr(os_utils, "_HOST_SHARES_CACHE", [])
    monkeypatch.setattr(os_utils, "_ON_CLUSTER_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", tmp_path / "absent.toml")
    assert os_utils._on_cluster() is False
    assert os_utils._ON_CLUSTER_CACHE == [False]
    assert os_utils._on_cluster() is False


# rebase_experimenter_in_paths: occurrence scanning


def test_rebase_experimenter_in_paths_scans_every_occurrence_in_order():
    """The occurrence scan walks forward one occurrence at a time: a name at the very
    start of a path is rewritten and the scan resumes right after the substituted id
    (so a later component is still found), three bounded occurrences are all
    rewritten (none skipped by jumping to the last), and an unbounded occurrence
    followed by two bounded ones rewrites both bounded ones."""
    rebase = os_utils.rebase_experimenter_in_paths
    assert rebase("Bartul/x/Bartul", ["Bartul"], "Ivan") == "Ivan/x/Ivan"
    assert rebase("/B/x/B/y/B", ["B"], "Q") == "/Q/x/Q/y/Q"
    assert rebase("/Bx/B/y/B", ["B"], "Q") == "/Bx/Q/y/Q"


# read_qlvm_category_bundle: pickle refusal and nomenclature labels


def test_read_qlvm_category_bundle_refuses_pickled_grids(tmp_path):
    """The grids are loaded with ``allow_pickle=False``: a bundle whose label grid
    was saved as an object array (which would need unpickling) is refused with
    NumPy's pickle error instead of executing pickled content."""
    directory = write_qlvm_category_bundle(tmp_path / "pickled")
    with np.load(directory / "category_grids.npz") as grids:
        arrays = {key: grids[key] for key in grids.files}
    arrays["label_grid"] = arrays["label_grid"].astype(object)
    np.savez_compressed(directory / "category_grids.npz", **arrays)
    with pytest.raises(ValueError, match="allow_pickle"):
        os_utils.read_qlvm_category_bundle(str(directory))


def test_read_qlvm_category_bundle_refuses_out_of_order_nomenclature(tmp_path):
    """A nomenclature whose categories do not carry ``grid_label`` 1..k in order is
    refused, with a message naming the labels it holds and the expected range."""
    directory = write_qlvm_category_bundle(tmp_path / "swapped")
    nomenclature = json.loads((directory / "category_nomenclature.json").read_text())
    nomenclature["categories"][0]["grid_label"] = 2
    nomenclature["categories"][1]["grid_label"] = 1
    (directory / "category_nomenclature.json").write_text(json.dumps(nomenclature))
    with pytest.raises(ValueError, match=r"the nomenclature's grid labels \[2, 1, 3, 4\] are not 1\.\.4\."):
        os_utils.read_qlvm_category_bundle(str(directory))


# ephys_base_for_data_root: components near the path root


@pytest.mark.parametrize(("data_root", "expected"), [
    ("/Data/s1", "/EPHYS"),
    ("Data/s1", "EPHYS"),
    ("/Data/a/s1", "/EPHYS/a"),
    ("/x/Data/a/b/s1", "/x/EPHYS/a/b"),
])
def test_ephys_base_for_data_root_checks_every_component_down_to_the_root(data_root, expected):
    """The backward scan visits every component of the parent, including the first
    one(s) next to the root and components separated from the end by an odd number
    of others, so a ``Data`` component anywhere in the parent is found."""
    assert os_utils.ephys_base_for_data_root(data_root) == pathlib.Path(expected)


# wait_for_subprocesses on a simulated clock


class FakeClock:
    """A deterministic stand-in for the ``time`` module ``wait_for_subprocesses``
    uses (``os_utils._time``): ``monotonic()`` returns a simulated time that only
    advances through ``sleep(seconds)``, and every sleep is recorded."""

    def __init__(self, start=100.0):
        self.now = start
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class StubbornPopen:
    """A Popen stand-in that never exits on its own and, once terminated, keeps
    running for ``exit_delay`` simulated seconds before reporting -15; ``kill()``
    makes it exit at once with -9 (or never, when ``survives_kill``)."""

    def __init__(self, clock, exit_delay, survives_kill=False):
        self.clock = clock
        self.exit_delay = exit_delay
        self.survives_kill = survives_kill
        self.terminated_at = None
        self.killed = False

    def poll(self):
        if self.killed and not self.survives_kill:
            return -9
        if self.terminated_at is not None and self.clock.now >= self.terminated_at + self.exit_delay:
            return -15
        return None

    def terminate(self):
        self.terminated_at = self.clock.now

    def kill(self):
        self.killed = True


def test_wait_waits_until_the_deadline_before_timing_out(monkeypatch):
    """The deadline is ``now + max_seconds``: a process that finishes after a few
    polls inside the budget is waited for (one sleep of ``poll_interval_s`` per
    unfinished poll) and its code returned, with no timeout."""
    clock = FakeClock()
    monkeypatch.setattr(os_utils, "_time", clock)
    codes = os_utils.wait_for_subprocesses([FakePopen(returncode=0, finish_after=2)], max_seconds=5,
                                           label="slow", poll_interval_s=0.5, message_output=lambda *_: None)
    assert codes == [0]
    assert clock.sleeps == [0.5, 0.5]


def test_wait_times_out_exactly_at_the_deadline(monkeypatch):
    """The timeout fires when the clock reaches the deadline (``>=``): with a zero
    budget an unfinished process is terminated on the very first poll, before any
    poll-interval sleep, even though it would have finished on the next poll."""
    clock = FakeClock()
    monkeypatch.setattr(os_utils, "_time", clock)
    proc = FakePopen(returncode=0, finish_after=1)
    with pytest.raises(TimeoutError, match="1 subprocess"):
        os_utils.wait_for_subprocesses([proc], max_seconds=0, label="zero", poll_interval_s=1.0,
                                       message_output=lambda *_: None)
    assert proc.terminated
    assert 1.0 not in clock.sleeps


def test_wait_timeout_logs_and_spares_a_process_that_exits_in_the_grace_period(monkeypatch):
    """On timeout the group logs which share is still running, terminates it, and
    polls every 0.25 s while it is still running; a process that exits 1 s after
    terminate() is never killed, and the grace loop stops as soon as it exits
    (four 0.25 s sleeps), not at the end of the 3 s window."""
    clock = FakeClock()
    monkeypatch.setattr(os_utils, "_time", clock)
    logs = []
    proc = StubbornPopen(clock, exit_delay=1.0)
    status = os_utils.wait_for_subprocesses([proc], max_seconds=2, label="grace", poll_interval_s=1.0,
                                            message_output=logs.append, raise_on_timeout=False)
    assert status == [-15]
    assert proc.terminated_at == 102.0
    assert not proc.killed
    assert clock.sleeps == [1.0, 1.0, 0.25, 0.25, 0.25, 0.25]
    assert logs[0] == "[grace] timed out after 2 s with 1/1 subprocess(es) still running; terminating."
    assert logs[1] == "[grace] 1/1 subprocess(es) exited with non-zero status: #0(rc=-15)."


def test_wait_timeout_kills_a_process_still_running_after_three_seconds(monkeypatch):
    """The grace period is exactly 3 s: a process that would need 3.5 s to exit
    after terminate() is still running when the grace loop ends (at 3.00 s, twelve
    0.25 s sleeps, not one more) and is killed; a process already exited by then is
    not killed."""
    clock = FakeClock()
    monkeypatch.setattr(os_utils, "_time", clock)
    stubborn = StubbornPopen(clock, exit_delay=3.5)
    status = os_utils.wait_for_subprocesses([stubborn], max_seconds=0, label="kill", poll_interval_s=1.0,
                                            message_output=lambda *_: None, raise_on_timeout=False)
    assert stubborn.killed
    assert status == [-9]
    assert clock.sleeps == [0.25] * 12
    assert clock.now == 103.0


def test_wait_timeout_status_none_is_not_reported_as_a_failure(monkeypatch):
    """A process that survives even kill() leaves a None slot in the returned status;
    None means 'still running', not a non-zero exit, so no failure line is logged."""
    clock = FakeClock()
    monkeypatch.setattr(os_utils, "_time", clock)
    logs = []
    immortal = StubbornPopen(clock, exit_delay=10 ** 9, survives_kill=True)
    status = os_utils.wait_for_subprocesses([immortal], max_seconds=0, label="immortal", poll_interval_s=1.0,
                                            message_output=logs.append, raise_on_timeout=False)
    assert status == [None]
    assert immortal.killed
    assert not any("non-zero" in message for message in logs)


def test_wait_reports_exactly_the_nonzero_codes():
    """Only codes other than 0 count as failures: an all-zero group logs nothing,
    and a mixed group's failure line lists each failing slot with its code (an exit
    code of 1 is a failure like any other)."""
    logs = []
    assert os_utils.wait_for_subprocesses([FakePopen(returncode=0), FakePopen(returncode=0)], max_seconds=5,
                                          label="ok", message_output=logs.append) == [0, 0]
    assert logs == []
    codes = os_utils.wait_for_subprocesses([FakePopen(returncode=1), FakePopen(returncode=0), FakePopen(returncode=3)],
                                           max_seconds=5, label="mix", message_output=logs.append)
    assert codes == [1, 0, 3]
    assert logs == ["[mix] 2/3 subprocess(es) exited with non-zero status: #0(rc=1), #2(rc=3)."]


# drop_noise_usvs / vocal-class helpers: messages name the source


def test_drop_noise_usvs_messages_name_the_source():
    """A summary without ``noise`` raises naming the source and the remedies, and a
    drop is logged as one line naming the source, the dropped count and the total."""
    with pytest.raises(KeyError, match=r"old\.csv has no 'noise' column, so its noise segments cannot be excluded"):
        os_utils.drop_noise_usvs(pls.DataFrame({"start": [0.1]}), "old.csv", message_output=lambda *_: None)
    logs = []
    kept, n_dropped = os_utils.drop_noise_usvs(
        pls.DataFrame({"start": [0.1, 0.2, 0.3], "noise": [True, False, True]}), "s1", message_output=logs.append)
    assert kept["start"].to_list() == [0.2]
    assert n_dropped == 2
    assert logs == ["    s1: dropped 2 noise segment(s) of 3."]


@pytest.mark.parametrize("helper", [
    lambda table: os_utils.vocal_class_mask(table, ["usv"], "old_summary.csv"),
    lambda table: os_utils.pure_usv_mask(table, "old_summary.csv"),
    lambda table: os_utils.pure_squeak_mask(table, "old_summary.csv"),
    lambda table: os_utils.overlap_mask(table, "old_summary.csv"),
    lambda table: os_utils.squeak_bearing_mask(table, "old_summary.csv"),
    lambda table: os_utils.vocal_event_times(table, ["usv"], "old_summary.csv"),
    lambda table: os_utils.vocal_occupancy_times(table, "old_summary.csv"),
])
def test_vocal_helpers_name_the_source_when_the_flags_are_missing(helper):
    """Every vocal-class helper passes its caller's source on to the flag check, so
    the KeyError for a summary without ``usv`` names the file it came from."""
    table = pls.DataFrame({"start": [0.1], "stop": [0.2], "squeak": [True]})
    with pytest.raises(KeyError, match=r"old_summary\.csv has no \['usv'\] column"):
        helper(table)


def test_squeak_extent_errors_name_the_source_and_the_consumer():
    """The missing-extent KeyError and the empty-extent ValueError name the source
    and what needed the extent: the class group's event times, or the silence
    reference."""
    no_extent = _timed_summary().drop("squeak_start", "squeak_end")
    with pytest.raises(KeyError, match=r"s7 has no \['squeak_start', 'squeak_end'\] column\(s\), which the \['squeak'\] event times needs"):
        os_utils.vocal_event_times(no_extent, ("squeak",), "s7")
    holed = _timed_summary().with_columns(pls.lit(None, dtype=pls.Float64).alias("squeak_start"))
    with pytest.raises(ValueError, match=r"s7: 2 squeak-bearing row\(s\) have no squeak_start / squeak_end, which the \['squeak'\] event times needs"):
        os_utils.vocal_event_times(holed, ("squeak",), "s7")
    with pytest.raises(KeyError, match=r"s8 has no \['squeak_start', 'squeak_end'\] column\(s\), which the silence reference needs"):
        os_utils.vocal_occupancy_times(no_extent, "s8")
    with pytest.raises(ValueError, match=r"s8: 2 squeak-bearing row\(s\) have no squeak_start / squeak_end, which the silence reference needs"):
        os_utils.vocal_occupancy_times(holed, "s8")


def test_squeak_extent_tolerates_unparseable_text_off_squeak_rows():
    """The extent is cast non-strictly: a text column holding a non-numeric value on
    a row that is not timed by the squeak (a pure USV) does not raise, and the
    numeric text on the squeak rows is used."""
    table = _timed_summary().with_columns(
        pls.Series("squeak_start", ["n/a", "1.98", "0.54", None]),
        pls.Series("squeak_end", ["n/a", "2.05", "0.65", None]),
    )
    timed = os_utils.vocal_event_times(table, ("squeak",), "s")
    assert timed["start"].to_list() == [0.54, 1.00, 1.98, 3.00]
    assert timed["stop"].to_list() == [0.65, 1.10, 2.05, 3.10]


def test_vocal_event_times_union_takes_the_earlier_squeak_onset():
    """In a union group an overlap segment whose squeak starts BEFORE its segment
    onset takes the squeak onset (the earlier of the two) and the later offset."""
    table = _timed_summary().with_columns(pls.Series("squeak_start", [None, 1.98, 0.45, None]))
    timed = os_utils.vocal_event_times(table, ("overlap",), "s")
    assert timed["start"].to_list()[0] == 0.45
    assert timed["stop"].to_list()[0] == 0.65


def test_vocal_event_times_usv_only_group_casts_both_times_to_float():
    """A USV-only group returns every row with its segment times, both ``start`` and
    ``stop`` replaced by Float64 copies (integer-typed times included)."""
    table = pls.DataFrame({"start": [3, 1], "stop": [4, 2], "usv": [True, True], "squeak": [False, False]})
    timed = os_utils.vocal_event_times(table, ("usv",), "s")
    assert timed["start"].dtype == pls.Float64
    assert timed["stop"].dtype == pls.Float64
    assert timed["start"].to_list() == [1.0, 3.0]
    assert timed["stop"].to_list() == [2.0, 4.0]


# glob helpers: messages and ordering keys


def test_first_match_message_names_the_glob_kind(tmp_path):
    """The no-match error names the glob kind used: ``glob`` for a flat search and
    ``rglob`` for a recursive one."""
    with pytest.raises(FileNotFoundError, match=r"thing: no match for glob pattern '\*\.nope'"):
        os_utils.first_match_or_raise(tmp_path, "*.nope", label="thing")
    with pytest.raises(FileNotFoundError, match=r"thing: no match for rglob pattern '\*\.nope'"):
        os_utils.first_match_or_raise(tmp_path, "*.nope", recursive=True, label="thing")


def test_resolve_consolidated_h5_error_names_the_store(tmp_path):
    """With no consolidated store the error opens with the store's label, not the
    bare glob pattern."""
    with pytest.raises(FileNotFoundError, match=r"^consolidated spectrogram \.h5: no match for glob pattern"):
        os_utils.resolve_consolidated_h5_path(str(tmp_path))


def test_resolve_squeak_spectrogram_store_orders_by_mtime_not_ctime(tmp_path):
    """The squeak store resolver orders by modification time: the store with the
    later mtime wins even when the other one's metadata (ctime) changed last."""
    old = tmp_path / "squeak_spectrograms_old.h5"
    new = tmp_path / "squeak_spectrograms_new.h5"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    os.utime(new, (10 ** 9, 10 ** 9))
    time.sleep(0.05)
    os.utime(old, (10 ** 8, 10 ** 8))
    assert old.stat().st_ctime >= new.stat().st_ctime
    assert os_utils.resolve_squeak_spectrogram_store_path(str(tmp_path)) == str(new)
