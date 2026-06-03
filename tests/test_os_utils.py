"""
Tests for usv_playpen.os_utils: cross-OS path translation (find_base_path /
configure_path), the deterministic glob helpers (first_match_or_raise /
newest_match_or_raise) and the subprocess-group waiter (wait_for_subprocesses).

`configure_path`/`find_base_path` are exercised under all three target OSs by
monkeypatching `os_utils.platform.system`; the cases include the two bugs the
mount-mapping table was introduced to fix (all-occurrence substring corruption,
and the previously-unhandled `murthy` share).
"""

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
