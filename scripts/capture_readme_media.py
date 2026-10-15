from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import patch

os.environ["QT_QPA_PLATFORM"] = "offscreen"

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"


class _NoHotkeys:
    translate_registered = False
    clear_region_registered = False
    errors: list[str] = []

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> None:
    from PySide6.QtCore import QSettings
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtWidgets import QApplication, QDockWidget

    import manga_rt_translator.ocr as ocr_module
    import manga_rt_translator.reader as reader_module
    from manga_rt_translator.config import AppConfig

    MEDIA.mkdir(parents=True, exist_ok=True)
    settings = QSettings("UltraOCR", "MangaRealTimeTranslator")
    had_saved_theme = settings.contains("reader/dark_mode")
    saved_theme = settings.value("reader/dark_mode", False, type=bool)
    settings.setValue("reader/dark_mode", False)
    def capture_event_loop(app: QApplication) -> int:
        font_id = QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        if font_id >= 0:
            app.setFont(QFont("Segoe UI", 9))
        window = reader_module._ACTIVE_WINDOW
        if window is None:
            raise RuntimeError("Reader window was not created")
        dock = window.findChild(QDockWidget, "translationDock")
        if dock is None:
            raise RuntimeError("Reader controls dock was not found")
        try:
            window.resize(1440, 960)
            window.show()
            app.processEvents()
            for dark, filename in (
                (True, "reader-controls-dark.png"),
                (False, "reader-controls.png"),
                (True, "reader-controls-dark.png"),
                (False, "reader-controls.png"),
            ):
                window.theme_action.setChecked(dark)
                app.processEvents()
                time.sleep(0.2)
                app.processEvents()
                path = MEDIA / filename
                if not dock.grab().save(str(path), "PNG"):
                    raise RuntimeError(f"Could not save capture: {path}")

            from PIL import Image

            frames = [
                Image.open(MEDIA / name).convert("RGB")
                for name in (
                    "reader-controls.png",
                    "reader-controls-dark.png",
                    "reader-controls-dark.png",
                    "reader-controls.png",
                )
            ]
            frames[0].save(
                MEDIA / "reader-theme-toggle.gif",
                save_all=True,
                append_images=frames[1:],
                duration=[850, 850, 500, 850],
                loop=0,
                optimize=True,
                disposal=2,
            )
            return 0
        finally:
            window.close()
            app.processEvents()

    try:
        with (
            patch.object(reader_module, "load_config", lambda _path: AppConfig()),
            patch.object(reader_module, "resolve_glossary_path", lambda _path: None),
            patch.object(reader_module, "GlobalHotkeyManager", _NoHotkeys),
            patch.object(ocr_module, "warmup_ocr", lambda _config: None),
            patch.object(QApplication, "exec", capture_event_loop),
        ):
            result = reader_module.run_reader_app(None, None, "about:blank")
            if result != 0:
                raise RuntimeError(f"Reader exited with code {result}")
    finally:
        if had_saved_theme:
            settings.setValue("reader/dark_mode", saved_theme)
        else:
            settings.remove("reader/dark_mode")


if __name__ == "__main__":
    main()
