from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage, QPainter
from PySide6.QtWidgets import QApplication, QWidget

import manga_rt_translator.reader as reader_module
from manga_rt_translator.config import AppConfig
from manga_rt_translator.models import OverlayItem, Rect
from manga_rt_translator.work_queue import ViewState


class _NoHotkeys:
    translate_registered = False
    clear_region_registered = False
    errors: list[str] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def test_real_reader_overlay_paints_translation_at_detected_bubble_anchor(monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QApplication, "exec", lambda _self: 0)
    monkeypatch.setattr(reader_module, "load_config", lambda _path: AppConfig())
    monkeypatch.setattr(reader_module, "resolve_glossary_path", lambda _path: None)
    monkeypatch.setattr(reader_module, "GlobalHotkeyManager", _NoHotkeys)
    warmed_configs = []
    monkeypatch.setattr(
        "manga_rt_translator.ocr.warmup_ocr",
        lambda config: warmed_configs.append(config),
    )

    assert reader_module.run_reader_app(None, None, "about:blank") == 0
    window = reader_module._ACTIVE_WINDOW
    assert window is not None
    try:
        assert window.source_combo.currentData() == "Auto"
        assert warmed_configs[0].extra_langs == ["japan"]
        assert window.launch.config.ocr.extra_langs == []
        overlay = window.overlay_widget
        overlay.setGeometry(0, 0, 400, 300)
        overlay.show()
        overlay.update_items([
            OverlayItem(
                bbox=Rect(20, 20, 68, 24),
                display_bbox=Rect(100, 80, 160, 120),
                text="Tradução aqui",
                source_text="Translation here",
                background_color=(249, 249, 249),
            )
        ])
        app.processEvents()

        image = QImage(400, 300, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        overlay.render(painter, QPoint())
        painter.end()

        assert not image.isNull()
        first_caption = overlay._placements[
            reader_module._overlay_item_key(overlay.items[0])
        ]
        first_bubble = reader_module.PaintRect(100, 80, 160, 120)
        assert reader_module._paint_rect_contains(first_bubble, first_caption)
        assert first_caption.width < first_bubble.width
        assert first_caption.height < first_bubble.height
        assert abs(
            first_caption.x + first_caption.width / 2
            - (first_bubble.x + first_bubble.width / 2)
        ) <= 1
        assert abs(
            first_caption.y + first_caption.height / 2
            - (first_bubble.y + first_bubble.height / 2)
        ) <= 1
        assert image.pixelColor(30, 30).alpha() == 0

        captions = [
            OverlayItem(
                bbox=Rect(62, 124, 90, 24),
                display_bbox=Rect(20, 80, 170, 140),
                text="A porta foi completamente bloqueada.",
                source_text="The door was completely blocked.",
                background_color=(249, 249, 249),
            ),
            OverlayItem(
                bbox=Rect(236, 120, 86, 25),
                display_bbox=Rect(210, 80, 170, 140),
                text="Eu vou cuidar disso depois.",
                source_text="I will take care of that later.",
                background_color=(249, 249, 249),
            ),
        ]
        overlay.update_items(captions)
        app.processEvents()
        placed = [
            overlay._placements[reader_module._overlay_item_key(item)]
            for item in captions
        ]
        for bubble, rect in zip(
            (item.display_bbox for item in captions), placed, strict=True
        ):
            assert reader_module._paint_rect_contains(
                reader_module.PaintRect(
                    bubble.x, bubble.y, bubble.width, bubble.height
                ),
                rect,
            )
        assert not reader_module._rects_overlap(*placed)

        repaired = OverlayItem(
            bbox=Rect(120, 100, 80, 30),
            display_bbox=Rect(100, 80, 160, 120),
            text="Tradução",
            source_text="Original",
            background_color=(249, 249, 249),
            background_patch=np.full((200, 200, 3), (70, 70, 70), dtype=np.uint8),
            background_rect=Rect(50, 50, 200, 200),
        )
        overlay.update_items([repaired])
        app.processEvents()
        repaired_image = QImage(
            400, 300, QImage.Format.Format_ARGB32_Premultiplied
        )
        repaired_image.fill(0)
        painter = QPainter(repaired_image)
        overlay.render(painter, QPoint())
        painter.end()
        repaired_rect = overlay._placements[
            reader_module._overlay_item_key(repaired)
        ]
        fill_pixel = repaired_image.pixelColor(
            repaired_rect.x + 7, repaired_rect.y + repaired_rect.height // 2
        )
        assert fill_pixel.red() > 200

        overlay.suspend_presentation()
        app.processEvents()
        view = ViewState()
        fresh_token = view.advance()
        repaint_calls = []
        original_update = QWidget.update

        def track_overlay_update(widget, *args):
            if widget is overlay:
                repaint_calls.append(True)
            return original_update(widget, *args)

        monkeypatch.setattr(QWidget, "update", track_overlay_update)
        overlay.update_items((fresh_token, overlay.items))
        assert not overlay._suspended
        assert overlay._items_token is fresh_token
        assert repaint_calls

        scroll_view = ViewState()
        prior_token = scroll_view.token()
        retained_items = list(overlay.items)
        overlay.update_items((prior_token, retained_items))
        next_token = scroll_view.advance()
        assert not prior_token.current()

        class _Pipeline:
            active_items = retained_items

        class _Runtime:
            view = scroll_view
            pipeline = _Pipeline()

        window.runtime = _Runtime()
        window._refresh_scroll_overlay()
        assert overlay._items_token.current()
        assert overlay.items == retained_items
        assert not overlay._suspended
        assert overlay._items_token.generation == next_token.generation
        window.runtime = None
    finally:
        window.runtime = None
        window.close()
        app.processEvents()
        reader_module._ACTIVE_WINDOW = None
