"""Render real OCR detections through the reader overlay for visual QA."""
from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

import cv2
import numpy as np
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFontDatabase, QImage, QPainter
from PySide6.QtWidgets import QApplication

from manga_rt_translator.cache import TranslationMemory
from manga_rt_translator.config import AppConfig
from manga_rt_translator.diff import FrameDiffer
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.models import Frame, OverlayItem
from manga_rt_translator.ocr import create_ocr_engine
from manga_rt_translator.overlay import Overlay
from manga_rt_translator.pipeline import MangaTranslationPipeline
from manga_rt_translator.reader import _rects_overlap, run_reader_app
from manga_rt_translator.text_regions import TextRegionDetector
from manga_rt_translator.translator import TranslationResult, Translator


class _CaptureOverlay(Overlay):
    def __init__(self) -> None:
        self.items: list[OverlayItem] = []

    def update(self, items) -> None:
        self.items = list(items)

    def run_forever(self) -> int:
        return 0

    def stop(self) -> None:
        return None


class _LayoutTranslator(Translator):
    def translate(self, text, source_lang, target_lang, glossary, context=""):
        normalized = " ".join(text.split()).upper()
        if normalized == "CHIEF!":
            translated = "CHEFE!"
        elif normalized.startswith("BARLZEN"):
            translated = text.strip()
        elif normalized.startswith("HAS THERE EVER BEEN"):
            translated = "Alguém já ouviu nossa história de verdade antes?"
        else:
            translated = "Tradução de teste em português"
        return TranslationResult(text=translated, provider="layout-qa")


class _NoHotkeys:
    translate_registered = False
    clear_region_registered = False
    errors: list[str] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reader_debug_output/overlay-capture-validation.png"),
    )
    parser.add_argument(
        "--region",
        help="Optional x,y,width,height crop matching the user's selected area.",
    )
    args = parser.parse_args()

    if platform.system().lower() != "windows":
        raise SystemExit("This visual check uses the Windows reader widget.")
    raw = cv2.imread(str(args.image))
    if raw is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    frame_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    if args.region:
        try:
            x, y, crop_width, crop_height = (int(value) for value in args.region.split(","))
        except (TypeError, ValueError) as exc:
            raise SystemExit("--region must be x,y,width,height") from exc
        image_height, image_width = frame_rgb.shape[:2]
        if (
            x < 0 or y < 0 or crop_width <= 0 or crop_height <= 0
            or x + crop_width > image_width or y + crop_height > image_height
        ):
            raise SystemExit("--region lies outside the source image")
        frame_rgb = frame_rgb[y : y + crop_height, x : x + crop_width].copy()
    height, width = frame_rgb.shape[:2]

    config = AppConfig()
    config.ocr.source_lang = "en"
    config.ocr.extra_langs = []
    overlay = _CaptureOverlay()
    memory = TranslationMemory(":memory:", fuzzy_threshold=config.translation.fuzzy_threshold)
    pipeline = MangaTranslationPipeline(
        config=config,
        stream=None,
        differ=FrameDiffer(config.diff),
        detector=TextRegionDetector(ignore_top_pixels=0),
        ocr=create_ocr_engine(config.ocr),
        memory=memory,
        glossary=Glossary.default(),
        translator=_LayoutTranslator(),
        overlay=overlay,
    )
    started = time.perf_counter()
    try:
        pipeline.process_frame(Frame(frame_rgb, time.time(), 0), force=True)
        items = list(overlay.items)
        if not items:
            raise SystemExit("PaddleOCR returned no translated text blocks.")
    finally:
        pipeline.close(wait=True)
        memory.close()
    ocr_ms = (time.perf_counter() - started) * 1000

    app = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "segoeui.ttf"
    if font_path.is_file():
        QFontDatabase.addApplicationFont(str(font_path))
    from manga_rt_translator import ocr as ocr_module
    import manga_rt_translator.reader as reader_module

    reader_module.load_config = lambda _path: config
    reader_module.resolve_glossary_path = lambda _path: None
    reader_module.GlobalHotkeyManager = _NoHotkeys
    ocr_module.warmup_ocr = lambda _config: None
    QApplication.exec = lambda _self: 0
    if run_reader_app(None, None, "about:blank") != 0:
        raise SystemExit("Could not initialize the reader overlay widget.")
    window = reader_module._ACTIVE_WINDOW
    if window is None:
        raise SystemExit("Reader did not create its main window.")

    try:
        overlay_widget = window.overlay_widget
        window.overlay_raise_timer.stop()
        window._sync_overlay_geometry = lambda: None
        overlay_widget.set_viewport_region(None)
        overlay_widget.setGeometry(0, 0, width, height)
        overlay_widget.show()
        app.processEvents()
        overlay_widget.setGeometry(0, 0, width, height)
        overlay_widget.update_items(items)
        app.processEvents()

        source = QImage(
            frame_rgb.data,
            width,
            height,
            int(frame_rgb.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        canvas = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(0xFF000000)
        painter = QPainter(canvas)
        painter.drawImage(0, 0, source)
        overlay_widget.render(painter, QPoint())
        painter.end()

        placed = list(overlay_widget._placements.values())
        if not placed:
            raise SystemExit("The production overlay placed no captions.")
        for index, rect in enumerate(placed):
            if rect.x < 0 or rect.y < 0 or rect.x + rect.width > width or rect.y + rect.height > height:
                raise SystemExit(f"Caption outside viewport: {rect}")
            for other in placed[index + 1 :]:
                if _rects_overlap(rect, other):
                    raise SystemExit(f"Caption collision: {rect} overlaps {other}")

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not canvas.save(str(args.output)):
            raise SystemExit(f"Could not write visual result: {args.output}")
        print(
            f"OK: OCR={len(items)} placed={len(placed)} omitted={len(items) - len(placed)} "
            f"overlaps=0 ocr_ms={ocr_ms:.0f} output={args.output.resolve()}"
        )
    finally:
        window.close()
        app.processEvents()
        reader_module._ACTIVE_WINDOW = None


if __name__ == "__main__":
    main()
