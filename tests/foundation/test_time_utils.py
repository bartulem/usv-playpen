"""
Tests for usv_playpen.time_utils: is_gui_context() detection and smart_wait()
dispatch (QTest.qWait in a GUI context vs time.sleep otherwise), including the
negative-seconds clamp. smart_wait is called ~75 times across the codebase but
previously had no test coverage.
"""

import pytest

from usv_playpen import time_utils


def test_is_gui_context_false_when_no_qapplication(monkeypatch):
    monkeypatch.setattr(time_utils.QCoreApplication, "instance", lambda: None)
    assert time_utils.is_gui_context() is False


def test_is_gui_context_true_with_qapplication(qapp):
    # pytest-qt's qapp fixture provides a real QApplication instance
    assert time_utils.is_gui_context() is True


def test_smart_wait_non_gui_uses_time_sleep(monkeypatch):
    recorded = []
    monkeypatch.setattr(time_utils.time, "sleep", lambda s: recorded.append(s))
    time_utils.smart_wait(app_context_bool=False, seconds=1.5)
    assert recorded == [1.5]


def test_smart_wait_gui_uses_qwait_in_milliseconds(monkeypatch):
    recorded = []
    monkeypatch.setattr(time_utils.QTest, "qWait", lambda ms: recorded.append(ms))
    time_utils.smart_wait(app_context_bool=True, seconds=2)
    assert recorded == [2000]


def test_smart_wait_defaults_to_zero_sleep(monkeypatch):
    recorded = []
    monkeypatch.setattr(time_utils.time, "sleep", lambda s: recorded.append(s))
    time_utils.smart_wait()
    assert recorded == [0.0]


@pytest.mark.parametrize("app_ctx,expected", [(False, 0.0), (True, 0)])
def test_smart_wait_clamps_negative_seconds(monkeypatch, app_ctx, expected):
    recorded = []
    if app_ctx:
        monkeypatch.setattr(time_utils.QTest, "qWait", lambda ms: recorded.append(ms))
    else:
        monkeypatch.setattr(time_utils.time, "sleep", lambda s: recorded.append(s))
    time_utils.smart_wait(app_context_bool=app_ctx, seconds=-5)
    assert recorded == [expected]
