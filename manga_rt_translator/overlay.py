from __future__ import annotations

import time
import logging
import platform
from dataclasses import dataclass
from typing import Iterable

from manga_rt_translator.config import OverlayConfig
from manga_rt_translator.capture.windows_capture_affinity import (
    exclude_window_from_dxgi_capture,
)
from manga_rt_translator.models import CaptureViewport, OverlayItem, Rect
from manga_rt_translator.overlay_background import foreground_color
from manga_rt_translator.overlay_geometry import logical_capture_rect


class Overlay:
    def set_capture_viewport(self, viewport: CaptureViewport | None) -> None:
        return None

    def update(self, items: Iterable[OverlayItem]) -> None:
        raise NotImplementedError

    def run_forever(self) -> int:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class ConsoleOverlay(Overlay):
    def __init__(self) -> None:
        self.items: list[OverlayItem] = []

    def update(self, items: Iterable[OverlayItem]) -> None:
        self.items = list(items)
        for item in self.items:
            print(f"[overlay] {item.bbox.as_tuple()} {item.text}")

    def run_forever(self) -> int:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            return 0

    def stop(self) -> None:
        return None


def create_overlay(config: OverlayConfig, enabled: bool = True) -> Overlay:
    if not enabled or not config.enabled:
        return ConsoleOverlay()
    try:
        return create_qt_overlay(config)
    except ImportError as exc:
        raise RuntimeError(
            "Qt overlay requires PySide6. Install with: "
            'pip install -e ".[overlay]"'
        ) from exc


def create_qt_overlay(config: OverlayConfig) -> Overlay:
    from PySide6.QtCore import QObject, QRectF, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
    from PySide6.QtWidgets import QApplication, QWidget

    class OverlayBridge(QObject):
        items_changed = Signal(object)
        viewport_changed = Signal(object)
        stop_requested = Signal()

    class OverlayWidget(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.items: list[OverlayItem] = []
            self.capture_scale = 1.0
            self.setWindowTitle("Manga Real-Time Translator Overlay")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowTransparentForInput
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

            screen = QApplication.primaryScreen()
            geometry = screen.geometry() if screen else None
            if geometry:
                self.setGeometry(geometry)
            else:
                self.resize(1920, 1080)

        @Slot(object)
        def set_capture_viewport(self, viewport: CaptureViewport | None) -> None:
            if viewport is None:
                self.capture_scale = 1.0
                screen = QApplication.primaryScreen()
                if screen is not None:
                    self.setGeometry(screen.geometry())
                self.update()
                return

            screens = QApplication.screens()
            screen = next(
                (
                    candidate
                    for candidate in screens
                    if viewport.monitor_device_name
                    and candidate.name().casefold()
                    == viewport.monitor_device_name.casefold()
                ),
                None,
            )
            if screen is None and 0 <= viewport.monitor_index < len(screens):
                screen = screens[viewport.monitor_index]
            if screen is None:
                screen = QApplication.primaryScreen()
            if screen is None:
                return

            self.capture_scale = max(1.0, float(screen.devicePixelRatio()))
            geometry = screen.geometry()
            logical_geometry = logical_capture_rect(
                viewport,
                Rect(geometry.x(), geometry.y(), geometry.width(), geometry.height()),
                self.capture_scale,
            )
            self.setGeometry(
                logical_geometry.x,
                logical_geometry.y,
                logical_geometry.width,
                logical_geometry.height,
            )
            self.update()

        @Slot(object)
        def set_items(self, items: list[OverlayItem]) -> None:
            now = time.time()
            self.items = [
                item
                for item in items
                if item.expires_at is None or item.expires_at > now
            ]
            self.update()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            scale = self.capture_scale
            if scale != 1.0:
                painter.scale(1 / scale, 1 / scale)
            font = QFont(config.font_family, max(1, round(config.font_size * scale)))
            font.setBold(True)
            painter.setFont(font)

            for item in self.items:
                patch = item.background_patch
                patch_rect = item.background_rect
                if patch is None or patch_rect is None or item.text == item.source_text:
                    continue
                image = QImage(
                    patch.data,
                    patch.shape[1],
                    patch.shape[0],
                    int(patch.strides[0]),
                    QImage.Format.Format_RGB888,
                ).copy()
                painter.drawImage(
                    QRectF(
                        patch_rect.x,
                        patch_rect.y,
                        patch_rect.width,
                        patch_rect.height,
                    ),
                    image,
                )

            for item in self.items:
                rect = _fit_rect(
                    item.bbox,
                    max(1, round(self.width() * scale)),
                    max(1, round(self.height() * scale)),
                )
                bubble = QPainterPath()
                bubble.addRoundedRect(rect.x, rect.y, rect.width, rect.height, 10, 10)
                has_restored_background = (
                    item.background_patch is not None and item.text != item.source_text
                )
                if not has_restored_background:
                    painter.fillPath(bubble.translated(0, 2), QColor(0, 0, 0, 48))

                sampled_background = item.background_color
                bg = (
                    QColor(
                        *sampled_background,
                        int(255 * config.opacity)
                        if has_restored_background
                        else 255,
                    )
                    if sampled_background is not None
                    else QColor(
                        255,
                        252,
                        245,
                        int(255 * config.opacity),
                    )
                )
                fg = QColor(*foreground_color(sampled_background), 245)
                border = (
                    QColor(36, 31, 28, 110)
                    if fg.red() < 128
                    else QColor(245, 245, 245, 150)
                )
                painter.fillPath(bubble, bg)
                if not has_restored_background:
                    painter.setPen(QPen(border, 1.2))
                    painter.drawPath(bubble)
                painter.setPen(fg)
                painter.drawText(
                    rect.x + 10,
                    rect.y + 8,
                    max(1, rect.width - 20),
                    max(1, rect.height - 16),
                    Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap,
                    item.text,
                )

    class QtOverlay(Overlay):
        def __init__(self) -> None:
            self.app = QApplication.instance() or QApplication([])
            self.bridge = OverlayBridge()
            self.widget = OverlayWidget()
            self._capture_viewport: CaptureViewport | None = None
            self.capture_exclusion: tuple[bool, str, int] | None = None
            self.bridge.items_changed.connect(self.widget.set_items)
            self.bridge.viewport_changed.connect(self.widget.set_capture_viewport)
            self.bridge.stop_requested.connect(self.app.quit)
            self.widget.show()
            if platform.system().lower() == "windows":
                self.capture_exclusion = exclude_window_from_dxgi_capture(
                    int(self.widget.winId())
                )
                excluded, method, error = self.capture_exclusion
                if not excluded:
                    logging.getLogger(__name__).warning(
                        "Could not exclude translation overlay from DXGI capture "
                        "(method=%s, WinError=%s).",
                        method,
                        error,
                    )

        def update(self, items: Iterable[OverlayItem]) -> None:
            self.bridge.items_changed.emit(list(items))

        def set_capture_viewport(
            self, viewport: CaptureViewport | None
        ) -> None:
            if viewport == self._capture_viewport:
                return
            self._capture_viewport = viewport
            self.bridge.viewport_changed.emit(viewport)

        def run_forever(self) -> int:
            return int(self.app.exec())

        def stop(self) -> None:
            self.bridge.stop_requested.emit()

    return QtOverlay()


@dataclass
class PaintRect:
    x: int
    y: int
    width: int
    height: int


def _fit_rect(rect: Rect, max_width: int, max_height: int) -> PaintRect:
    raw_width = max(rect.width, 110)
    raw_height = max(rect.height, 52)
    # Cover the whole text block instead of a fixed-size card.
    width = min(raw_width + 16, max_width - 4)
    height = min(raw_height + 20, max_height - 4)
    x = min(max(0, rect.x - 8), max(0, max_width - width))
    y = min(max(0, rect.y - 10), max(0, max_height - height))
    return PaintRect(x=x, y=y, width=width, height=height)
