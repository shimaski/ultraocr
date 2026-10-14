from __future__ import annotations

import threading
import time

import numpy as np

from manga_rt_translator.models import Frame


class WindowsGraphicsCapture:
    """Event-driven Windows Graphics Capture stream for one HWND."""

    def __init__(
        self,
        window_handle: int | None = None,
        monitor_index: int | None = None,
    ) -> None:
        self.window_handle = int(window_handle) if window_handle is not None else None
        self.monitor_index = monitor_index
        self._lock = threading.Lock()
        self._state = "stopped"
        self._latest: Frame | None = None
        self._index = 0
        self._consumed_index = 0
        self._capture = None
        self._control = None
        self._start_thread: threading.Thread | None = None
        self.startup_phase = "idle"
        self.last_error = ""

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    def start(self) -> None:
        with self._lock:
            if self._state in {"starting", "running"}:
                return
            self._state = "starting"
            self.startup_phase = "starting capture thread"
            self._latest = None
            self.last_error = ""
        self._start_thread = threading.Thread(
            target=self._start_capture,
            name="reader-wgc-start",
            daemon=True,
        )
        self._start_thread.start()

    def read_latest(self) -> tuple[str, Frame | None]:
        with self._lock:
            if (
                self._latest is not None
                and self._latest.index > self._consumed_index
            ):
                self._consumed_index = self._latest.index
                return "ok", self._latest
            if self._state == "starting":
                return "starting", None
            if self._state == "running":
                return "nochange", None
            return "unavailable", None

    def peek_latest(self) -> Frame | None:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        with self._lock:
            control = self._control
            self._state = "stopped"
            self._latest = None
        if control is not None:
            try:
                control.stop()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"

    def on_frame_arrived(self, frame: object, _control: object) -> None:
        with self._lock:
            if self._state not in {"starting", "running"}:
                return
        try:
            pixels = np.asarray(frame.frame_buffer)
            if pixels.ndim != 3 or pixels.shape[2] < 4:
                raise ValueError(f"Unexpected WGC frame shape: {pixels.shape}")
            rgb = np.ascontiguousarray(pixels[:, :, 2::-1])
            with self._lock:
                if self._state not in {"starting", "running"}:
                    return
                self._index += 1
                self._latest = Frame(
                    image=rgb,
                    timestamp=time.time(),
                    index=self._index,
                )
        except Exception as exc:
            with self._lock:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self._state = "failed"

    def on_closed(self) -> None:
        with self._lock:
            if self._state != "stopped":
                self._state = "closed"

    def _start_capture(self) -> None:
        try:
            self.startup_phase = "importing Windows Capture backend"
            from windows_capture import WindowsCapture

            self.startup_phase = "creating capture session"
            capture = WindowsCapture(
                cursor_capture=False,
                draw_border=None,
                minimum_update_interval=None,
                dirty_region=None,
                monitor_index=self.monitor_index,
                window_hwnd=(
                    self.window_handle if self.monitor_index is None else None
                ),
            )
            capture.event(self.on_frame_arrived)
            capture.event(self.on_closed)
            self.startup_phase = "starting native capture thread"
            control = capture.start_free_threaded()
            with self._lock:
                self._capture = capture
                self._control = control
                stopped = self._state == "stopped"
                if not stopped:
                    self._state = "running"
                    self.startup_phase = "running"
            if stopped:
                control.stop()
        except Exception as exc:
            with self._lock:
                if self._state != "stopped":
                    self.last_error = f"{type(exc).__name__}: {exc}"
                    self._state = "failed"
                    self.startup_phase = "failed"
