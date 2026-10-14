from __future__ import annotations

import time
import threading
from typing import Iterator

import numpy as np

from manga_rt_translator.models import CaptureViewport, Frame


class WindowsDxgiStream:
    """Windows frame stream using DXGI Desktop Duplication through dxcam.

    The important part is that `start(video_mode=True)` creates a continuous
    capture loop. The pipeline consumes the latest video frame instead of
    calling a screenshot API repeatedly.
    """

    def __init__(
        self,
        fps: int = 30,
        output_idx: int = 0,
        region: tuple[int, int, int, int] | None = None,
        window_title: str | None = None,
        window_handle: int | None = None,
    ) -> None:
        self.fps = fps
        self.output_idx = output_idx
        self.region = region
        self.window_title = window_title
        self.window_handle = window_handle
        self._camera = None
        self._dxcam_module = None
        self._running = False
        self._index = 0
        self._window_tracking = False
        self._resolved_window_handle: int | None = None
        self._window_missing = False
        self._capture_loss_pending = False
        self._active_region: tuple[int, int, int, int] | None = None
        self._active_output_idx = output_idx
        self._capture_origin: tuple[int, int] = (0, 0)
        self._monitor_info = None
        self._window_refresh_interval = 0.5

    def start(self) -> None:
        if self._running:
            return
        try:
            import dxcam as dxcam_module
        except ImportError as exc:
            raise RuntimeError(
                "Windows capture requires dxcam. Install with: "
                'pip install -e ".[windows]"'
            ) from exc

        self._dxcam_module = dxcam_module
        region = self.region
        output_idx = self.output_idx
        if region is None and (self.window_title or self.window_handle is not None):
            from manga_rt_translator.capture.windows_windows import (
                find_window,
                resolve_capture_geometry_for_window,
            )

            window = find_window(
                title_contains=self.window_title,
                handle=self.window_handle,
            )
            if window is None:
                selector = (
                    f"handle={self.window_handle}"
                    if self.window_handle is not None
                    else f"title contains {self.window_title!r}"
                )
                raise RuntimeError(f"Unable to find visible window for {selector}.")
            output_idx, region, self._capture_origin, self._monitor_info = (
                resolve_capture_geometry_for_window(window)
            )
            self._window_tracking = True
            self._resolved_window_handle = window.handle
        else:
            from manga_rt_translator.capture.windows_windows import monitor_for_output

            self._monitor_info = monitor_for_output(output_idx)
            if region is None:
                self._capture_origin = (
                    self._monitor_info.left,
                    self._monitor_info.top,
                )
            else:
                self._capture_origin = (
                    self._monitor_info.left + region[0],
                    self._monitor_info.top + region[1],
                )

        self._camera = self._dxcam_module.create(
            output_idx=output_idx,
            output_color="RGB",
        )
        self._camera.start(
            target_fps=self.fps,
            video_mode=True,
            region=region,
        )
        self._active_region = region
        self._active_output_idx = output_idx
        self._window_missing = False
        self._running = True

    def frames(self) -> Iterator[Frame]:
        if not self._running:
            self.start()
        assert self._camera is not None

        sleep_seconds = max(0.001, 1.0 / max(1, self.fps * 2))
        next_window_check = 0.0
        while self._running:
            now = time.monotonic()
            if self._window_tracking and now >= next_window_check:
                self._refresh_window_region()
                next_window_check = now + self._window_refresh_interval
            if self._window_missing:
                if self._capture_loss_pending:
                    self._capture_loss_pending = False
                    self._index += 1
                    yield Frame(
                        image=np.empty((0, 0, 3), dtype=np.uint8),
                        timestamp=time.time(),
                        index=self._index,
                        capture_lost=True,
                    )
                    continue
                time.sleep(min(0.1, sleep_seconds * 2))
                continue
            image = self._camera.get_latest_frame()
            if image is None:
                time.sleep(sleep_seconds)
                continue
            self._index += 1
            viewport = None
            if self._monitor_info is not None:
                viewport = CaptureViewport(
                    capture_origin=self._capture_origin,
                    capture_size=(image.shape[1], image.shape[0]),
                    monitor_origin=(self._monitor_info.left, self._monitor_info.top),
                    monitor_device_name=self._monitor_info.device_name,
                    monitor_index=self._active_output_idx,
                )
            yield Frame(
                image=image,
                timestamp=time.time(),
                index=self._index,
                capture_viewport=viewport,
            )

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
        self._running = False

    def _refresh_window_region(self) -> None:
        from manga_rt_translator.capture.windows_windows import (
            find_window,
            resolve_capture_geometry_for_window,
        )

        window = find_window(
            title_contains=self.window_title,
            handle=self._resolved_window_handle or self.window_handle,
        )
        if (
            window is None
            and self.window_title
            and self._resolved_window_handle is not None
        ):
            window = find_window(title_contains=self.window_title)
        if window is None:
            if not self._window_missing:
                if self._camera is not None:
                    self._camera.stop()
                _capture_log("window_capture.target_missing")
                self._capture_loss_pending = True
            self._window_missing = True
            return

        try:
            output_idx, region, capture_origin, monitor_info = (
                resolve_capture_geometry_for_window(window)
            )
        except RuntimeError as exc:
            if not self._window_missing:
                if self._camera is not None:
                    self._camera.stop()
                _capture_log(f"window_capture.target_unavailable: {exc}")
                self._capture_loss_pending = True
            self._window_missing = True
            return
        if (
            not self._window_missing
            and output_idx == self._active_output_idx
            and region == self._active_region
            and capture_origin == self._capture_origin
            and monitor_info == self._monitor_info
        ):
            return

        camera = self._camera
        if camera is not None:
            camera.stop()
        if output_idx != self._active_output_idx:
            if camera is not None:
                camera.release()
            assert self._dxcam_module is not None
            camera = self._dxcam_module.create(
                output_idx=output_idx,
                output_color="RGB",
            )
            self._camera = camera

        try:
            assert camera is not None
            camera.start(
                target_fps=self.fps,
                video_mode=True,
                region=region,
            )
        except Exception as exc:
            if not self._window_missing:
                _capture_log(
                    f"window_capture.restart_error {type(exc).__name__}: {exc}"
                )
                self._capture_loss_pending = True
            self._window_missing = True
            return

        if self._window_missing:
            _capture_log("window_capture.target_restored")
        self._resolved_window_handle = window.handle
        self._active_output_idx = output_idx
        self._active_region = region
        self.region = region
        self.output_idx = output_idx
        self._capture_origin = capture_origin
        self._monitor_info = monitor_info
        self._window_missing = False


class WindowsDxgiRegionCapture:
    """Long-lived DXGI capture whose initialization never blocks the UI thread."""

    def __init__(self, fps: int = 30, output_idx: int = 0) -> None:
        self.fps = max(1, int(fps))
        self.output_idx = output_idx
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._desired_region: tuple[int, int, int, int] | None = None
        self._desired_output_idx = self.output_idx
        self._latest: Frame | None = None
        self._consumed_index = 0
        self._index = 0
        self.last_error = ""

    def set_region(
        self,
        region: tuple[int, int, int, int],
        output_idx: int | None = None,
    ) -> None:
        normalized = tuple(int(value) for value in region)
        with self._lock:
            next_output_idx = self.output_idx if output_idx is None else int(output_idx)
            if (
                normalized != self._desired_region
                or next_output_idx != self._desired_output_idx
            ):
                self._desired_region = normalized
                self._desired_output_idx = next_output_idx
                self.output_idx = next_output_idx
                self._latest = None
                self._consumed_index = self._index
            if self._thread is None or not self._thread.is_alive():
                self._shutdown.clear()
                self._thread = threading.Thread(
                    target=self._capture_loop,
                    name="reader-dxgi-capture",
                    daemon=True,
                )
                self._thread.start()
        self._wake.set()

    def read_latest(self) -> Frame | None:
        with self._lock:
            if self._latest is None or self._latest.index <= self._consumed_index:
                return None
            self._consumed_index = self._latest.index
            return self._latest

    def stop(self) -> None:
        with self._lock:
            self._desired_region = None
            self._latest = None
            self._consumed_index = self._index
        self._wake.set()

    def release(self) -> None:
        self._shutdown.set()
        self.stop()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)
        self._thread = None

    def _capture_loop(self) -> None:
        camera = None
        active_region: tuple[int, int, int, int] | None = None
        active_output_idx = self.output_idx
        last_timestamp: float | None = None
        try:
            while not self._shutdown.is_set():
                with self._lock:
                    desired = self._desired_region
                    desired_output_idx = self._desired_output_idx
                if desired != active_region or desired_output_idx != active_output_idx:
                    if camera is not None and active_region is not None:
                        camera.stop()
                        active_region = None
                    if desired_output_idx != active_output_idx:
                        if camera is not None:
                            camera.release()
                        camera = None
                        active_output_idx = desired_output_idx
                    if desired is None:
                        self._wake.wait(0.05)
                        self._wake.clear()
                        continue
                    try:
                        if camera is None:
                            import dxcam

                            camera = dxcam.create(
                                output_idx=active_output_idx,
                                output_color="RGB",
                            )
                        camera.start(
                            region=desired,
                            target_fps=self.fps,
                            video_mode=True,
                        )
                        active_region = desired
                        last_timestamp = None
                        self.last_error = ""
                    except Exception as exc:
                        self.last_error = f"{type(exc).__name__}: {exc}"
                        _capture_log(f"dxgi.start_error {self.last_error}")
                        if camera is not None:
                            try:
                                camera.stop()
                            except Exception:
                                pass
                        self._wake.wait(0.5)
                        self._wake.clear()
                        continue

                if camera is None or active_region is None:
                    continue
                try:
                    timestamp = camera.latest_frame_time
                    if timestamp is None or (
                        last_timestamp is not None and timestamp <= last_timestamp
                    ):
                        self._wake.wait(min(0.01, 1 / self.fps))
                        self._wake.clear()
                        continue
                    image = camera.grab(copy=True)
                    if image is None:
                        self._wake.wait(min(0.01, 1 / self.fps))
                        self._wake.clear()
                        continue
                    timestamp = camera.latest_frame_time or timestamp
                    timestamp = float(timestamp)
                    if last_timestamp is None or timestamp > last_timestamp:
                        last_timestamp = timestamp
                        with self._lock:
                            self._index += 1
                            self._latest = Frame(
                                image=np.ascontiguousarray(image),
                                timestamp=timestamp,
                                index=self._index,
                            )
                except Exception as exc:
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    _capture_log(f"dxgi.read_error {self.last_error}")
                    camera.stop()
                    active_region = None
                    self._wake.wait(0.5)
                    self._wake.clear()
        finally:
            if camera is not None:
                try:
                    camera.stop()
                finally:
                    camera.release()


def _capture_log(message: str) -> None:
    try:
        with open(".reader_debug.log", "a", encoding="utf-8") as handle:
            handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    except OSError:
        pass
