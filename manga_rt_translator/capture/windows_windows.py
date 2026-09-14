from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

from manga_rt_translator.models import Rect

user32 = ctypes.windll.user32


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    class_name: str
    rect: Rect


@dataclass(frozen=True)
class MonitorInfo:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return max(0, self.right - self.left)

    @property
    def height(self) -> int:
        return max(0, self.bottom - self.top)


def list_visible_windows() -> list[WindowInfo]:
    windows: list[WindowInfo] = []

    enum_proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @enum_proc
    def _callback(hwnd: int, lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        if user32.IsIconic(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True

        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        title = title_buffer.value.strip()
        if not title:
            return True

        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return True
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width < 200 or height < 120:
            return True

        class_buffer = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, class_buffer, 255)
        windows.append(
            WindowInfo(
                handle=int(hwnd),
                title=title,
                class_name=class_buffer.value,
                rect=Rect(rect.left, rect.top, width, height),
            )
        )
        return True

    user32.EnumWindows(_callback, 0)
    windows.sort(key=lambda item: item.title.lower())
    return windows


def list_monitors() -> list[MonitorInfo]:
    monitors: list[MonitorInfo] = []
    monitor_enum = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )

    @monitor_enum
    def _callback(
        hmonitor: int,
        hdc: int,
        rect_ptr: ctypes.POINTER(wintypes.RECT),
        lparam: int,
    ) -> bool:
        rect = rect_ptr.contents
        monitors.append(
            MonitorInfo(
                left=rect.left,
                top=rect.top,
                right=rect.right,
                bottom=rect.bottom,
            )
        )
        return True

    user32.EnumDisplayMonitors(0, 0, _callback, 0)
    return monitors


def resolve_capture_region_for_window(window: WindowInfo) -> tuple[int, tuple[int, int, int, int]]:
    monitors = list_monitors()
    if not monitors:
        raise RuntimeError("Unable to enumerate monitors for capture.")

    best_index = 0
    best_overlap = -1
    for index, monitor in enumerate(monitors):
        overlap = _overlap_area(window.rect, monitor)
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index

    monitor = monitors[best_index]
    clipped = _clip_rect_to_monitor(window.rect, monitor)
    if clipped.width <= 1 or clipped.height <= 1:
        raise RuntimeError(
            f"Selected window is outside the visible bounds of monitor {best_index}."
        )

    region = (
        clipped.x - monitor.left,
        clipped.y - monitor.top,
        clipped.x2 - monitor.left,
        clipped.y2 - monitor.top,
    )
    return best_index, region


def find_window(
    title_contains: str | None = None,
    handle: int | None = None,
) -> WindowInfo | None:
    windows = list_visible_windows()
    if handle is not None:
        for window in windows:
            if window.handle == handle:
                return window
        return None

    if title_contains:
        needle = title_contains.casefold()
        exact = [window for window in windows if needle == window.title.casefold()]
        if exact:
            return exact[0]

        partial = [window for window in windows if needle in window.title.casefold()]
        if partial:
            partial.sort(key=lambda item: (len(item.title), item.title.lower()))
            return partial[0]

    return None


def _clip_rect_to_monitor(rect: Rect, monitor: MonitorInfo) -> Rect:
    left = max(rect.x, monitor.left)
    top = max(rect.y, monitor.top)
    right = min(rect.x2, monitor.right)
    bottom = min(rect.y2, monitor.bottom)
    return Rect(left, top, max(0, right - left), max(0, bottom - top))


def _overlap_area(rect: Rect, monitor: MonitorInfo) -> int:
    clipped = _clip_rect_to_monitor(rect, monitor)
    return clipped.area
