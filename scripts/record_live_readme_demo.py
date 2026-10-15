from __future__ import annotations

import http.server
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
FIXTURE = MEDIA / "reader-demo-page.html"
OUTPUT = MEDIA / "reader-live-translation.gif"
FRAME_INTERVAL_MS = 120
MAX_RECORD_SECONDS = 38


class _NoLogHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *_args: object) -> None:
        pass


def _start_demo_server() -> tuple[http.server.ThreadingHTTPServer, threading.Thread]:
    handler = lambda *args, **kwargs: _NoLogHandler(  # noqa: E731
        *args, directory=str(FIXTURE.parent), **kwargs
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _translated_items(runtime: object) -> list[object]:
    pipeline = getattr(runtime, "pipeline", None)
    items = list(getattr(pipeline, "active_items", ()) or ())
    return [
        item
        for item in items
        if getattr(item, "text", "").strip()
        and getattr(item, "text", "").strip()
        != getattr(item, "source_text", "").strip()
    ]


def _make_gif(frames: list[Image.Image]) -> None:
    if len(frames) < 15:
        raise RuntimeError("A gravação ficou curta demais para uma demonstração.")
    max_width = 1120
    resized = []
    for frame in frames:
        if frame.width > max_width:
            height = round(frame.height * max_width / frame.width)
            frame = frame.resize((max_width, height), Image.Resampling.LANCZOS)
        resized.append(frame.convert("RGB"))

    palette = resized[0].quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    encoded = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in resized
    ]
    encoded[0].save(
        OUTPUT,
        save_all=True,
        append_images=encoded[1:],
        duration=FRAME_INTERVAL_MS,
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    if not FIXTURE.is_file():
        raise FileNotFoundError(FIXTURE)

    os.environ.setdefault("QT_QPA_PLATFORM", "windows")
    server, server_thread = _start_demo_server()
    frames: list[Image.Image] = []
    failure: list[str] = []
    validated: list[tuple[str, str]] = []
    success = threading.Event()

    from PySide6.QtCore import QPoint, QSettings, QTimer, Qt
    from PySide6.QtGui import QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    import manga_rt_translator.reader as reader_module

    settings = QSettings("UltraOCR", "MangaRealTimeTranslator")
    had_saved_theme = settings.contains("reader/dark_mode")
    saved_theme = settings.value("reader/dark_mode", False, type=bool)
    original_exec = QApplication.exec

    def recording_exec(app: QApplication) -> int:
        window = reader_module._ACTIVE_WINDOW
        if window is None:
            raise RuntimeError("O leitor não abriu a janela principal.")

        window.resize(1480, 940)
        window.move(170, 35)
        window.show()
        window.raise_()
        window.activateWindow()
        window.theme_action.setChecked(False)

        started_at = time.monotonic()
        translation_started_at: float | None = None
        finish_scheduled = False

        def capture_frame() -> None:
            # Capture only the application's own Qt surfaces. Never record
            # the desktop, which can contain unrelated windows or private data.
            pixmap = window.grab()
            overlay_pixmap = window.overlay_widget.grab()
            if pixmap.isNull() or overlay_pixmap.isNull():
                return
            image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            width, height, stride = image.width(), image.height(), image.bytesPerLine()
            frame = Image.frombytes(
                "RGBA", (width, height), image.bits().tobytes(), "raw", "RGBA", stride
            )
            overlay = overlay_pixmap.toImage().convertToFormat(
                QImage.Format.Format_RGBA8888
            )
            overlay_image = Image.frombytes(
                "RGBA",
                (overlay.width(), overlay.height()),
                overlay.bits().tobytes(),
                "raw",
                "RGBA",
                overlay.bytesPerLine(),
            )
            webview_origin = window.webview.mapTo(window, QPoint(0, 0))
            frame.alpha_composite(
                overlay_image, (webview_origin.x(), webview_origin.y())
            )
            frame = frame.convert("RGB")
            frames.append(frame)

        recorder = QTimer(window)
        recorder.timeout.connect(capture_frame)

        def fail(message: str) -> None:
            if not failure:
                failure.append(message)
            recorder.stop()
            window.close()
            app.quit()

        def finish_after_translation() -> None:
            nonlocal finish_scheduled
            if finish_scheduled:
                return
            finish_scheduled = True
            success.set()
            QTimer.singleShot(2600, lambda: (recorder.stop(), window.close(), app.quit()))

        def click_start() -> None:
            if window.start_button.isEnabled():
                QTest.mouseClick(window.start_button, Qt.MouseButton.LeftButton)
                QTimer.singleShot(900, click_translate)
            else:
                QTimer.singleShot(350, click_start)

        def click_translate() -> None:
            nonlocal translation_started_at
            translation_started_at = time.monotonic()
            QTest.mouseClick(window.analyze_button, Qt.MouseButton.LeftButton)

        def poll_translation() -> None:
            if failure or finish_scheduled:
                return
            runtime = window.runtime
            translated = _translated_items(runtime) if runtime is not None else []
            if len(translated) >= 2:
                phrases = " ".join(
                    getattr(item, "source_text", "").lower()
                    for item in translated
                )
                if "clock" in phrases and "sunset" in phrases:
                    validated[:] = [
                        (
                            getattr(item, "source_text", "").strip(),
                            getattr(item, "text", "").strip(),
                        )
                        for item in translated
                        if "clock" in getattr(item, "source_text", "").lower()
                        or "sunset" in getattr(item, "source_text", "").lower()
                    ]
                    finish_after_translation()
                    return
            if runtime is not None and runtime.last_error:
                fail(f"O runtime do leitor falhou: {runtime.last_error}")
                return
            deadline = translation_started_at or started_at
            if time.monotonic() - deadline > 30:
                stats = runtime.stats_text() if runtime is not None else "runtime ausente"
                fail(f"A tradução real não concluiu dois balões. Estado: {stats}")

        def page_loaded(ok: bool) -> None:
            if not ok:
                fail("A página local de demonstração não carregou.")
                return
            window.source_combo.setCurrentIndex(window.source_combo.findData("en"))
            window.mode_combo.setCurrentText("Traduzir PT-BR")
            window.provider_combo.setCurrentText("Google")
            window.analysis_combo.setCurrentText("Manual")
            window.overlay_checkbox.setChecked(True)
            recorder.start(FRAME_INTERVAL_MS)
            QTimer.singleShot(1600, click_start)
            poller.start(300)

        poller = QTimer(window)
        poller.timeout.connect(poll_translation)
        window.webview.loadFinished.connect(page_loaded)
        deadline_timer = QTimer(window)
        deadline_timer.setSingleShot(True)
        deadline_timer.timeout.connect(
            lambda: fail("Tempo máximo da gravação excedido sem tradução validada.")
        )
        deadline_timer.start(MAX_RECORD_SECONDS * 1000)

        return int(original_exec())

    try:
        temp_path = Path(tempfile.mkdtemp(prefix="ultraocr-readme-demo-"))
        try:
            config_path = temp_path / "demo.toml"
            config_path.write_text(
                """
[capture]
backend = "auto"
fps = 8

[ocr]
enabled = true
source_lang = "en"
extra_langs = []
min_confidence = 0.55
ignore_top_pixels = 0
det_model = "mobile"
rec_model = "mobile"
detector = "paddle"

[translation]
target_lang = "pt-BR"
provider = "google"
timeout_seconds = 20
fuzzy_threshold = 99

[overlay]
enabled = true
font_family = "Segoe UI"
font_size = 20
opacity = 0.94
ttl_seconds = 900

[cache]
path = """ + repr(str(temp_path / "demo-cache.sqlite3")) + """
""",
                encoding="utf-8",
            )
            url = f"http://127.0.0.1:{server.server_port}/reader-demo-page.html"
            old_cwd = Path.cwd()
            try:
                os.chdir(temp_path)
                with (
                    patch.object(QApplication, "exec", recording_exec),
                    patch.object(
                        reader_module,
                        "default_profile_path",
                        lambda: temp_path / "demo-profiles.json",
                    ),
                ):
                    result = reader_module.run_reader_app(str(config_path), None, url)
            finally:
                os.chdir(old_cwd)
            if result != 0:
                raise RuntimeError(f"O leitor encerrou com código {result}.")
        finally:
            shutil.rmtree(temp_path, ignore_errors=True)
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)
        if had_saved_theme:
            settings.setValue("reader/dark_mode", saved_theme)
        else:
            settings.remove("reader/dark_mode")

    if failure:
        raise RuntimeError(failure[0])
    if not success.is_set():
        raise RuntimeError("A gravação terminou sem validar traduções no overlay.")

    _make_gif(frames)
    print(f"Demonstração gravada: {OUTPUT}")
    print(f"Frames capturados: {len(frames)}")
    for source, translated in validated:
        print(f"OCR: {source}\nPT-BR: {translated}")


if __name__ == "__main__":
    main()
