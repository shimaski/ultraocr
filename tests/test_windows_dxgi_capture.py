from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np

from manga_rt_translator.capture.windows_dxgi import (
    WindowsDxgiRegionCapture,
    WindowsDxgiStream,
)


class FakeCamera:
    def __init__(self):
        self.starts = []
        self.stops = 0
        self.frames = []
        self.current_timestamp = None
        self.releases = 0

    def start(self, *, region, target_fps, video_mode):
        self.starts.append((region, target_fps, video_mode))

    def get_latest_frame(self):
        return np.zeros((100, 200, 3), dtype=np.uint8)

    def stop(self):
        self.stops += 1

    def release(self):
        self.releases += 1

    def grab(self, *, copy):
        assert copy
        if not self.frames:
            return None
        image, self.current_timestamp = self.frames.pop(0)
        return image

    @property
    def latest_frame_time(self):
        if self.frames:
            return self.frames[0][1]
        return self.current_timestamp


def wait_until(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def wait_for_frame(stream, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = stream.read_latest()
        if frame is not None:
            return frame
        time.sleep(0.005)
    return None


def test_region_capture_stream_deduplicates_frames_and_reconfigures(monkeypatch):
    cameras = []

    def create_camera(*, output_idx, output_color):
        assert output_color == "RGB"
        camera = FakeCamera()
        camera.output_idx = output_idx
        cameras.append(camera)
        return camera

    monkeypatch.setitem(
        sys.modules,
        "dxcam",
        SimpleNamespace(create=create_camera),
    )
    stream = WindowsDxgiRegionCapture(fps=30)
    first_region = (20, 30, 220, 180)

    stream.set_region(first_region)
    stream.set_region(first_region)
    camera = cameras[0] if cameras else None
    assert wait_until(lambda: bool(cameras))
    camera = cameras[0]
    assert wait_until(lambda: len(camera.starts) == 1)
    camera.frames.append((np.zeros((150, 200, 3), dtype=np.uint8), 1.0))
    first = wait_for_frame(stream)
    assert first is not None and first.index == 1 and first.timestamp == 1.0
    camera.frames.append((np.ones((150, 200, 3), dtype=np.uint8), 1.0))
    time.sleep(0.03)
    assert stream.read_latest() is None
    camera.frames.clear()
    camera.frames.append((np.full((150, 200, 3), 2, dtype=np.uint8), 1.04))
    second = wait_for_frame(stream)
    assert second is not None and second.index == 2 and second.timestamp == 1.04

    second_region = (40, 50, 240, 200)
    stream.set_region(second_region)
    assert wait_until(lambda: len(camera.starts) == 2)
    assert camera.starts == [
        (first_region, 30, True),
        (second_region, 30, True),
    ]
    assert wait_until(lambda: camera.stops >= 1)

    stream.stop()
    assert wait_until(lambda: camera.stops >= 2)
    stream.release()
    assert camera.releases == 1


def test_region_capture_recreates_camera_when_output_changes(monkeypatch):
    cameras = []

    def create_camera(*, output_idx, output_color):
        assert output_color == "RGB"
        camera = FakeCamera()
        camera.output_idx = output_idx
        cameras.append(camera)
        return camera

    monkeypatch.setitem(
        sys.modules,
        "dxcam",
        SimpleNamespace(create=create_camera),
    )
    stream = WindowsDxgiRegionCapture(fps=30, output_idx=0)
    region = (0, 0, 100, 100)
    stream.set_region(region, output_idx=0)
    assert wait_until(lambda: len(cameras) == 1 and len(cameras[0].starts) == 1)

    stream.set_region(region, output_idx=1)
    assert wait_until(lambda: len(cameras) == 2 and len(cameras[1].starts) == 1)
    assert cameras[0].releases == 1
    assert cameras[1].output_idx == 1
    assert cameras[1].starts == [(region, 30, True)]
    stream.set_region((10, 10, 110, 110))
    assert wait_until(lambda: len(cameras[1].starts) == 2)
    assert cameras[1].starts[-1] == ((10, 10, 110, 110), 30, True)
    stream.release()


def test_region_request_does_not_wait_for_camera_initialization(monkeypatch):
    entered = threading.Event()
    allow_create = threading.Event()

    def create_camera(**kwargs):
        entered.set()
        assert allow_create.wait(2)
        return FakeCamera()

    monkeypatch.setitem(
        sys.modules,
        "dxcam",
        SimpleNamespace(create=create_camera),
    )
    stream = WindowsDxgiRegionCapture(fps=30)
    started = time.monotonic()
    stream.set_region((0, 0, 100, 100))
    assert time.monotonic() - started < 0.1
    assert entered.wait(1)
    allow_create.set()
    stream.release()


def test_window_stream_tracks_move_monitor_change_and_minimize(monkeypatch):
    current = {
        "window": SimpleNamespace(
            handle=55, rect=SimpleNamespace(x=10, y=20, x2=210, y2=120)
        ),
        "monitor": 0,
        "offscreen": False,
    }
    first_camera = FakeCamera()
    second_camera = FakeCamera()
    created = []

    def create_camera(*, output_idx, output_color):
        assert output_color == "RGB"
        camera = first_camera if output_idx == 0 else second_camera
        created.append(output_idx)
        return camera

    monkeypatch.setitem(
        sys.modules,
        "dxcam",
        SimpleNamespace(create=create_camera),
    )
    def resolve_geometry(window):
        if current["offscreen"]:
            raise RuntimeError("window is outside monitor bounds")
        monitor = SimpleNamespace(
            left=current["monitor"] * 1920,
            top=0,
            device_name=f"DISPLAY{current['monitor'] + 1}",
        )
        return (
            current["monitor"],
            (window.rect.x, window.rect.y, window.rect.x2, window.rect.y2),
            (monitor.left + window.rect.x, monitor.top + window.rect.y),
            monitor,
        )

    def find_target(**kwargs):
        window = current["window"]
        if window is None:
            return None
        handle = kwargs.get("handle")
        if handle is not None and window.handle != handle:
            return None
        return window

    monkeypatch.setattr(
        "manga_rt_translator.capture.windows_dxgi._capture_log",
        lambda _message: None,
    )
    monkeypatch.setitem(
        sys.modules,
        "manga_rt_translator.capture.windows_windows",
        SimpleNamespace(
            find_window=find_target,
            resolve_capture_geometry_for_window=resolve_geometry,
        ),
    )

    stream = WindowsDxgiStream(
        fps=30,
        window_title="Browser",
        window_handle=55,
    )
    stream.start()
    assert first_camera.starts == [((10, 20, 210, 120), 30, True)]

    current["window"] = SimpleNamespace(
        handle=55, rect=SimpleNamespace(x=30, y=45, x2=230, y2=145)
    )
    stream._refresh_window_region()
    assert first_camera.starts[-1] == ((30, 45, 230, 145), 30, True)

    current["monitor"] = 1
    current["window"] = SimpleNamespace(
        handle=55, rect=SimpleNamespace(x=5, y=8, x2=205, y2=108)
    )
    stream._refresh_window_region()
    assert first_camera.releases == 1
    assert second_camera.starts == [((5, 8, 205, 108), 30, True)]
    assert created == [0, 1]

    current["window"] = None
    stream._refresh_window_region()
    assert stream._window_missing
    assert second_camera.stops == 1

    current["window"] = SimpleNamespace(
        handle=77, rect=SimpleNamespace(x=15, y=18, x2=215, y2=118)
    )
    stream._refresh_window_region()
    assert not stream._window_missing
    assert stream._resolved_window_handle == 77
    assert second_camera.starts[-1] == ((15, 18, 215, 118), 30, True)

    current["offscreen"] = True
    stream._refresh_window_region()
    assert stream._window_missing
    assert second_camera.stops == 3
    current["offscreen"] = False
    stream._refresh_window_region()
    assert not stream._window_missing
    assert second_camera.starts[-1] == ((15, 18, 215, 118), 30, True)
    stream.stop()


def test_window_stream_refreshes_geometry_while_frames_are_consumed(monkeypatch):
    current = {"window": SimpleNamespace(handle=66, rect=SimpleNamespace(x=0, y=0, x2=200, y2=100))}
    camera = FakeCamera()
    monkeypatch.setitem(
        sys.modules,
        "dxcam",
        SimpleNamespace(create=lambda **_kwargs: camera),
    )
    monkeypatch.setitem(
        sys.modules,
        "manga_rt_translator.capture.windows_windows",
        SimpleNamespace(
            find_window=lambda **_kwargs: current["window"],
            resolve_capture_geometry_for_window=lambda window: (
                0,
                (window.rect.x, window.rect.y, window.rect.x2, window.rect.y2),
                (window.rect.x, window.rect.y),
                SimpleNamespace(left=0, top=0, device_name="DISPLAY1"),
            ),
        ),
    )
    monkeypatch.setattr(
        "manga_rt_translator.capture.windows_dxgi._capture_log",
        lambda _message: None,
    )
    stream = WindowsDxgiStream(fps=30, window_handle=66)
    stream._window_refresh_interval = 0.002
    frames = stream.frames()
    first = next(frames)
    assert first.capture_viewport is not None
    assert first.capture_viewport.capture_origin == (0, 0)
    current["window"] = SimpleNamespace(
        handle=66, rect=SimpleNamespace(x=25, y=35, x2=225, y2=135)
    )

    deadline = time.monotonic() + 1
    last_frame = first
    while len(camera.starts) < 2 and time.monotonic() < deadline:
        last_frame = next(frames)

    assert camera.starts == [
        ((0, 0, 200, 100), 30, True),
        ((25, 35, 225, 135), 30, True),
    ]
    assert last_frame.capture_viewport is not None
    assert last_frame.capture_viewport.capture_origin == (25, 35)

    current["window"] = None
    deadline = time.monotonic() + 1
    lost_frame = None
    while time.monotonic() < deadline:
        candidate = next(frames)
        if candidate.capture_lost:
            lost_frame = candidate
            break
    assert lost_frame is not None
    assert lost_frame.image.shape == (0, 0, 3)

    current["window"] = SimpleNamespace(
        handle=66, rect=SimpleNamespace(x=40, y=50, x2=240, y2=150)
    )
    deadline = time.monotonic() + 1
    restored_frame = None
    while len(camera.starts) < 3 and time.monotonic() < deadline:
        restored_frame = next(frames)
    assert len(camera.starts) == 3
    assert restored_frame is not None and not restored_frame.capture_lost
    assert restored_frame.capture_viewport is not None
    assert restored_frame.capture_viewport.capture_origin == (40, 50)
    stream.stop()
    frames.close()
