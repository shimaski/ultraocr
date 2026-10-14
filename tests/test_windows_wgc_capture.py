from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np

from manga_rt_translator.capture.windows_wgc import WindowsGraphicsCapture


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_wgc_stream_uses_hwnd_copies_latest_and_converts_bgra(monkeypatch):
    instances = []

    class FakeControl:
        stopped = False

        def stop(self):
            self.stopped = True

    class FakeCapture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.handlers = {}
            self.control = FakeControl()
            instances.append(self)

        def event(self, handler):
            self.handlers[handler.__name__] = handler

        def start_free_threaded(self):
            return self.control

    monkeypatch.setitem(
        sys.modules,
        "windows_capture",
        SimpleNamespace(WindowsCapture=FakeCapture),
    )

    stream = WindowsGraphicsCapture(window_handle=42)
    stream.start()
    assert wait_until(lambda: stream.state == "running")
    capture = instances[0]
    assert capture.kwargs == {
        "cursor_capture": False,
        "draw_border": None,
        "minimum_update_interval": None,
        "dirty_region": None,
        "window_hwnd": 42,
        "monitor_index": None,
    }

    source = np.zeros((2, 3, 4), dtype=np.uint8)
    source[0, 0] = (10, 20, 30, 255)
    capture.handlers["on_frame_arrived"](
        SimpleNamespace(frame_buffer=source), capture.control
    )
    source.fill(0)
    status, frame = stream.read_latest()
    assert status == "ok"
    assert frame is not None
    assert frame.image[0, 0].tolist() == [30, 20, 10]
    assert stream.read_latest() == ("nochange", None)

    next_frame = np.full((2, 3, 4), (1, 2, 3, 255), dtype=np.uint8)
    capture.handlers["on_frame_arrived"](
        SimpleNamespace(frame_buffer=next_frame), capture.control
    )
    assert stream.read_latest()[0] == "ok"
    stream.stop()
    assert capture.control.stopped
    assert stream.read_latest() == ("unavailable", None)


def test_wgc_stream_can_target_monitor(monkeypatch):
    instances = []

    class FakeCapture:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.handlers = {}
            instances.append(self)

        def event(self, handler):
            self.handlers[handler.__name__] = handler

        def start_free_threaded(self):
            return SimpleNamespace(stop=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "windows_capture",
        SimpleNamespace(WindowsCapture=FakeCapture),
    )
    stream = WindowsGraphicsCapture(monitor_index=2)
    stream.start()
    assert wait_until(lambda: stream.state == "running")
    assert instances[0].kwargs["monitor_index"] == 2
    assert instances[0].kwargs["window_hwnd"] is None
    stream.stop()


def test_wgc_stream_reports_start_failure(monkeypatch):
    class BrokenCapture:
        def __init__(self, **_kwargs):
            raise RuntimeError("window target was rejected")

    monkeypatch.setitem(
        sys.modules,
        "windows_capture",
        SimpleNamespace(WindowsCapture=BrokenCapture),
    )
    stream = WindowsGraphicsCapture(window_handle=99)
    stream.start()
    assert wait_until(lambda: stream.state == "failed")
    assert "window target was rejected" in stream.last_error
    assert stream.read_latest() == ("unavailable", None)
