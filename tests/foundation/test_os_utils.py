"""
Tests for usv_playpen.os_utils: cross-OS path translation (find_base_path /
configure_path), the Data->EPHYS sibling-tree mapping (ephys_base_for_data_root),
the deterministic glob helpers (first_match_or_raise / newest_match_or_raise)
and the subprocess-group waiter (wait_for_subprocesses).

`configure_path`/`find_base_path` are exercised under all three target OSs by
monkeypatching `os_utils.platform.system`; the cases include the two bugs the
mount-mapping table was introduced to fix (all-occurrence substring corruption,
and the previously-unhandled `murthy` share).
"""

import os
import pathlib
import time

import pytest

from usv_playpen import os_utils


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


# _host_experimenter + configure_path {experimenter} fill

def test_host_experimenter_reads_from_host_config(monkeypatch):
    """The experimenter id is read from the `experimenter` key of the host
    config TOML (the same key `exp_id` derives from)."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    assert os_utils._host_experimenter() == "Bartul"


def test_host_experimenter_raises_when_config_missing(monkeypatch, tmp_path):
    """A missing host config raises rather than guessing the experimenter."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", tmp_path / "absent.toml")
    with pytest.raises(RuntimeError, match="Cannot read the host config"):
        os_utils._host_experimenter()


@pytest.mark.parametrize("system,expected", [
    ("Linux", "/mnt/falkner/Liza/EPHYS"),
    ("Darwin", "/Volumes/falkner/Liza/EPHYS"),
    ("Windows", "F:\\Liza\\EPHYS"),
])
def test_configure_path_fills_experimenter_then_translates(as_os, monkeypatch, system, expected):
    """`configure_path` fills the `{experimenter}` placeholder from the host
    experimenter id, THEN translates the leading mount root to the host OS. A
    non-default experimenter ('Liza') proves the substitution actually happens."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", ["Liza"])
    as_os(system)
    assert os_utils.configure_path("/mnt/falkner/{experimenter}/EPHYS") == expected


def test_configure_path_without_placeholder_never_reads_experimenter(as_os, monkeypatch, tmp_path):
    """A path with no `{experimenter}` (e.g. a user-entered root/arena dir) is
    OS-translated only; the experimenter id is never read — proven by pointing
    the host config at an absent file and asserting no error is raised."""
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", [])
    monkeypatch.setattr(os_utils, "_HOST_CONFIG_PATH", tmp_path / "absent.toml")
    as_os("Darwin")
    assert os_utils.configure_path("/mnt/falkner/SomeUser/Data/s1") == "/Volumes/falkner/SomeUser/Data/s1"


# resolve_data_root

def test_resolve_data_root_reads_key_and_delegates(as_os, monkeypatch, tmp_path):
    """`resolve_data_root` reads the requested key from the `data_roots` block
    and resolves it via `configure_path` (which fills `{experimenter}` and
    OS-translates). A non-default experimenter ('Liza') proves it flows through."""
    settings = tmp_path / "analyses_settings.json"
    settings.write_text('{"data_roots": {"ephys_root": "/mnt/falkner/{experimenter}/EPHYS"}}')
    monkeypatch.setattr(os_utils, "_ANALYSES_SETTINGS_PATH", settings)
    monkeypatch.setattr(os_utils, "_HOST_EXPERIMENTER_CACHE", ["Liza"])
    as_os("Darwin")
    assert os_utils.resolve_data_root("ephys_root") == pathlib.Path("/Volumes/falkner/Liza/EPHYS")
    as_os("Linux")
    assert os_utils.resolve_data_root("ephys_root") == pathlib.Path("/mnt/falkner/Liza/EPHYS")


def test_resolve_data_root_reads_shipped_block(as_os):
    """The shipped analyses_settings.json defines the `data_roots` keys the
    analysis tools rely on; spot-check one resolves on the host OS (with the
    shipped `experimenter` filled into `{experimenter}`)."""
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


# resolve_embedding_arrays_path / resolve_consolidated_h5_path

def test_resolve_embedding_arrays_path_conventions(tmp_path):
    base = tmp_path / "spectrograms"
    assert os_utils.resolve_embedding_arrays_path(str(base), "qlvm", "coarse") == str(base / "qlvm" / "arrays_coarse.npz")
    assert os_utils.resolve_embedding_arrays_path(str(base), "qlvm", "fine") == str(base / "qlvm" / "arrays_fine.npz")
    assert os_utils.resolve_embedding_arrays_path(str(base), "vae", "coarse") == str(base / "vae" / "vae_density_coarse.npz")
    assert os_utils.resolve_embedding_arrays_path(str(base), "vae", "fine") == str(base / "vae" / "vae_density_fine.npz")


def test_resolve_consolidated_h5_picks_newest_and_skips_other_h5(tmp_path):
    (tmp_path / "qlvm_clusters_20260506.h5").write_bytes(b"x")  # different store -> must be ignored
    old = tmp_path / "spectrograms_old.h5"
    new = tmp_path / "spectrograms_new.h5"
    old.write_bytes(b"x")
    new.write_bytes(b"x")
    os.utime(old, (1, 1))
    os.utime(new, (10 ** 9, 10 ** 9))  # strictly newer mtime
    assert os_utils.resolve_consolidated_h5_path(str(tmp_path)) == str(new)


def test_resolve_consolidated_h5_raises_when_no_store(tmp_path):
    (tmp_path / "qlvm_clusters_x.h5").write_bytes(b"x")  # only a non-consolidated .h5
    with pytest.raises(FileNotFoundError, match="consolidated"):
        os_utils.resolve_consolidated_h5_path(str(tmp_path))


def test_resolve_pooled_embeddings_cache_convention(tmp_path):
    base = tmp_path / "spectrograms"
    assert os_utils.resolve_pooled_embeddings_cache(str(base)) == str(
        base / "embeddings" / "pooled_embeddings.parquet"
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
