from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

from manga_rt_translator.config import OverlayConfig
from manga_rt_translator.models import OverlayItem, Rect


class Overlay:
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
    from PySide6.QtCore import QObject, Qt, Signal, Slot
    from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
    from PySide6.QtWidgets import QApplication, QWidget

    class OverlayBridge(QObject):
        items_changed = Signal(object)
        stop_requested = Signal()

    class OverlayWidget(QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.items: list[OverlayItem] = []
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
            font = QFont(config.font_family, config.font_size)
            font.setBold(True)
            painter.setFont(font)

            for item in self.items:
                rect = _fit_rect(item.bbox, self.width(), self.height())
                bubble = QPainterPath()
                bubble.addRoundedRect(rect.x, rect.y, rect.width, rect.height, 10, 10)
                shadow = QColor(0, 0, 0, 48)
                painter.fillPath(bubble.translated(0, 2), shadow)

                bg = QColor(255, 252, 245, int(255 * config.opacity))
                fg = QColor(24, 20, 18, 245)
                border = QColor(36, 31, 28, 110)
                painter.setPen(QPen(border, 1.2))
                painter.fillPath(bubble, bg)
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
            self.bridge.items_changed.connect(self.widget.set_items)
            self.bridge.stop_requested.connect(self.app.quit)
            self.widget.show()

        def update(self, items: Iterable[OverlayItem]) -> None:
            self.bridge.items_changed.emit(list(items))

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
