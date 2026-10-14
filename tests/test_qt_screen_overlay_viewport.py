from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from manga_rt_translator.config import AppConfig
from manga_rt_translator.models import CaptureViewport, OverlayItem, Rect
from manga_rt_translator.overlay import create_qt_overlay
from manga_rt_translator.overlay_geometry import logical_capture_rect


def test_qt_overlay_moves_and_resizes_to_captured_window_viewport():
    app = QApplication.instance() or QApplication([])
    overlay = create_qt_overlay(AppConfig().overlay)
    screen = app.primaryScreen()
    assert screen is not None
    geometry = screen.geometry()
    scale = max(1.0, float(screen.devicePixelRatio()))
    viewport = CaptureViewport(
        capture_origin=(100, 80),
        capture_size=(320, 200),
        monitor_origin=(0, 0),
        monitor_device_name=screen.name(),
        monitor_index=0,
    )
    expected = logical_capture_rect(
        viewport,
        Rect(geometry.x(), geometry.y(), geometry.width(), geometry.height()),
        scale,
    )

    try:
        overlay.set_capture_viewport(viewport)
        overlay.update(
            [
                OverlayItem(
                    bbox=Rect(20, 30, 70, 24),
                    text="Translated",
                    source_text="Source",
                )
            ]
        )
        app.processEvents()

        assert overlay.widget.geometry().getRect() == (
            expected.x,
            expected.y,
            expected.width,
            expected.height,
        )
        assert overlay.widget.capture_scale == scale
        assert overlay.widget.isVisible()
    finally:
        overlay.widget.close()
        app.processEvents()
