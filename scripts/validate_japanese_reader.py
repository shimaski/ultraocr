"""Run the real reader against a local Japanese image and Google Translate."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("QT_OPENGL", "angle")

from PySide6.QtCore import QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication

from manga_rt_translator import reader as reader_module
from manga_rt_translator.config import AppConfig
from manga_rt_translator.models import Rect


def _has_japanese(text: str) -> bool:
    return any("\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff" for char in text)


def _japanese_char_count(text: str) -> int:
    return sum(
        "\u3040" <= char <= "\u30ff" or "\u3400" <= char <= "\u9fff"
        for char in text
    )


def _qualifying_translations(
    items: list[object], minimum_chars: int, expected_source: str
) -> list[object]:
    return [
        item
        for item in items
        if _japanese_char_count(item.source_text) >= minimum_chars
        and (not expected_source or expected_source in item.source_text)
        and item.text.strip() != item.source_text.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", type=Path)
    source_group.add_argument("--vertical-fixture", action="store_true")
    parser.add_argument("--region", default="430,80,580,440")
    parser.add_argument("--scale-factor", type=float, default=1.0)
    parser.add_argument("--rec-model", choices=("mobile", "server"), default="mobile")
    parser.add_argument("--min-japanese-chars", type=int, default=4)
    parser.add_argument(
        "--expected-source", default=""
    )
    args = parser.parse_args()
    if args.vertical_fixture and not args.expected_source:
        args.expected_source = "\u5927\u4e08\u592b"
    if args.min_japanese_chars < 1:
        parser.error("--min-japanese-chars must be at least 1")

    image_path = args.image.expanduser().resolve() if args.image else None
    if image_path is not None and not image_path.is_file():
        parser.error(f"Image not found: {image_path}")
    try:
        x, y, width, height = (int(value) for value in args.region.split(","))
    except (TypeError, ValueError):
        parser.error("--region must be x,y,width,height")
    if min(width, height) <= 0:
        parser.error("--region width and height must be positive")

    os.environ["QT_SCALE_FACTOR"] = str(args.scale_factor)
    root = Path(__file__).resolve().parents[1]
    output = root / "reader_debug_output" / f"japanese-qa-{int(time.time())}"
    output.mkdir(parents=True, exist_ok=True)
    page = output / "fixture.html"
    if args.vertical_fixture:
        page_html = (
            "<!doctype html><html><body style='margin:0;background:#202426'>"
            "<div style='position:absolute;left:420px;top:55px;width:330px;height:470px;"
            "box-sizing:border-box;border:4px solid #111;border-radius:48%;"
            "background:#fff;display:flex;align-items:center;justify-content:center'>"
            "<div style='writing-mode:vertical-rl;text-orientation:upright;"
            "font-family:Yu Gothic,Meiryo,sans-serif;font-size:38px;line-height:1.25;"
            "color:#111'>"
            "&#x672c;&#x5f53;&#x306b;&#x5927;&#x4e08;&#x592b;&#x3067;&#x3059;&#x304b;&#xff1f;"
            "</div></div></body></html>"
        )
    else:
        page_html = (
            "<!doctype html><html><body style='margin:0;background:#191a1b'>"
            f"<img style='display:block;width:1050px;height:auto' src='{image_path.as_uri()}'>"
            "<img style='display:block;width:1050px;height:auto' "
            f"src='{image_path.as_uri()}'></body></html>"
        )
    page.write_text(page_html, encoding="utf-8")

    config = AppConfig()
    config.ocr.source_lang = "japan"
    config.ocr.extra_langs = []
    config.ocr.rec_model = args.rec_model
    config.translation.provider = "google"
    config.cache.path = str(output / "memory.sqlite")
    reader_module.load_config = lambda _path: config

    app = QApplication.instance() or QApplication([])
    state: dict[str, object] = {
        "source": "japan",
        "fixture": "vertical-dialogue" if args.vertical_fixture else str(image_path),
        "provider": "google",
        "rec_model": args.rec_model,
        "target": "pt-BR",
        "region": [x, y, width, height],
        "scale_factor": args.scale_factor,
        "min_japanese_chars": args.min_japanese_chars,
        "expected_source": args.expected_source,
        "success": False,
        "items": [],
    }
    started = time.monotonic()
    poll = QTimer()
    worker = None

    def finish(error: str = "") -> None:
        nonlocal worker
        poll.stop()
        state["error"] = error
        window = reader_module._ACTIVE_WINDOW
        if window is not None:
            if window.runtime is not None:
                worker = window.runtime.worker
            window.close()
        (output / "result.json").write_text(
            json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        print(json.dumps(state, ensure_ascii=True), flush=True)
        QTimer.singleShot(100, app.quit)

    def inspect() -> None:
        window = reader_module._ACTIVE_WINDOW
        if window is None or window.runtime is None or window.runtime.pipeline is None:
            return
        pipeline = window.runtime.pipeline
        with pipeline.view.lock:
            items = list(pipeline.active_items)
        japanese = [item for item in items if _has_japanese(item.source_text)]
        translated = [
            item
            for item in japanese
            if item.text.strip() != item.source_text.strip()
        ]
        state["items"] = [
            {
                "source": item.source_text,
                "translated": item.text,
                "japanese_chars": _japanese_char_count(item.source_text),
                "confidence": round(item.confidence, 3),
                "bbox": item.bbox.as_tuple(),
            }
            for item in translated
        ]
        state["ocr_regions"] = pipeline.stats.ocr_regions
        state["translation_ms"] = round(pipeline.stats.last_translation_ms)
        valid = _qualifying_translations(
            translated, args.min_japanese_chars, args.expected_source
        )
        if valid:
            base = window.webview.grab()
            overlay = window.overlay_widget.grab()
            painter = QPainter(base)
            painter.drawPixmap(0, 0, overlay)
            painter.end()
            base.save(str(output / "translated.png"))
            state["elapsed_s"] = round(time.monotonic() - started, 2)
            state["success"] = True
            finish()
            return
        if translated:
            reason = (
                "OCR text did not contain --expected-source"
                if args.expected_source
                else "OCR returned translated fragments below --min-japanese-chars"
            )
            advice = (
                "check the selected image and region."
                if args.expected_source
                else "select a larger text region or lower the threshold."
            )
            finish(f"{reason}; {advice}")
            return
        if pipeline.stats.last_error:
            finish(pipeline.stats.last_error)

    def select_region() -> None:
        window = reader_module._ACTIVE_WINDOW
        if window is None:
            return finish("Reader window was not created")
        window.resize(1150, 920)
        window.source_combo.setCurrentText("japan")
        window.mode_combo.setCurrentIndex(0)
        window._handle_region_selected(Rect(x, y, width, height))
        poll.start(100)

    def bind_page() -> None:
        window = reader_module._ACTIVE_WINDOW
        if window is None:
            return finish("Reader window was not created")
        window.webview.loadFinished.connect(
            lambda ok: QTimer.singleShot(
                400,
                select_region if ok else lambda: finish("Japanese fixture failed to load"),
            )
        )

    poll.timeout.connect(inspect)
    QTimer.singleShot(0, bind_page)
    QTimer.singleShot(120_000, lambda: finish("Timed out waiting for OCR/translation"))
    reader_module.run_reader_app(None, None, page.as_uri())
    if worker is not None:
        worker.join(30)
    return 0 if state["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
