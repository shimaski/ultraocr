"""Verify that the reader's real overlay is omitted from DXGI capture."""
from __future__ import annotations

import os
import platform
import time
import ctypes

os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

from PySide6.QtWidgets import QApplication

from manga_rt_translator.config import AppConfig
from manga_rt_translator.models import OverlayItem, Rect
from manga_rt_translator.reader import run_reader_app


class _NoHotkeys:
    translate_registered = False
    clear_region_registered = False
    errors: list[str] = []

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def close(self) -> None:
        pass


def main() -> None:
    if platform.system().lower() != "windows":
        raise SystemExit("DXGI capture validation is available on Windows only.")

    import manga_rt_translator.ocr as ocr_module
    import manga_rt_translator.reader as reader_module

    config = AppConfig()
    reader_module.load_config = lambda _path: config
    reader_module.resolve_glossary_path = lambda _path: None
    reader_module.GlobalHotkeyManager = _NoHotkeys
    ocr_module.warmup_ocr = lambda _config: None
    app_exec = QApplication.exec
    QApplication.exec = lambda _self: 0

    stream = None
    window = None
    app = None
    try:
        run_reader_app(None, None, "about:blank")
        window = reader_module._ACTIVE_WINDOW
        app = QApplication.instance()
        if window is None or app is None:
            raise SystemExit("Reader did not initialize.")

        window.showMaximized()
        window.activateWindow()
        app.processEvents()
        window._sync_overlay_geometry()
        overlay = window.overlay_widget
        if not overlay.isVisible():
            overlay.show()
            app.processEvents()

        overlay.update_items(
            [
                OverlayItem(
                    bbox=Rect(36, 36, 90, 42),
                    text="DXGI PROBE",
                    source_text="source",
                    background_color=(255, 0, 255),
                )
            ]
        )
        app.processEvents()
        placements = list(overlay._placements.values())
        if not placements:
            raise SystemExit("The production overlay did not paint the probe.")

        affinity_ok = window._exclude_overlay_from_capture()
        if not affinity_ok:
            error_code = window._overlay_capture_error
            message = ctypes.FormatError(error_code).strip()
            raise SystemExit(
                "Reader could not exclude its overlay from capture: "
                f"method={window._overlay_capture_method} WinError {error_code} ({message})"
            )
        exclusion_method = window._overlay_capture_method

        screen = overlay.windowHandle().screen()
        if screen is None:
            raise SystemExit("Could not determine the overlay's display.")
        screen_geometry = screen.geometry()
        scale = float(screen.devicePixelRatio())
        global_geometry = overlay.geometry()
        placed = placements[0]
        roi_x = int((global_geometry.x() + placed.x + 4 - screen_geometry.x()) * scale)
        roi_y = int((global_geometry.y() + placed.y + 4 - screen_geometry.y()) * scale)
        roi_width = max(12, int(16 * scale))
        roi_height = max(12, int(12 * scale))
        capture_region = (
            roi_x,
            roi_y,
            roi_x + roi_width,
            roi_y + roi_height,
        )

        from manga_rt_translator.capture.windows_dxgi import WindowsDxgiRegionCapture

        stream = WindowsDxgiRegionCapture(fps=30, output_idx=0)
        stream.set_region(capture_region)
        deadline = time.monotonic() + 5
        frame = None
        while time.monotonic() < deadline:
            app.processEvents()
            frame = stream.read_latest()
            if frame is not None:
                break
            if stream.last_error:
                raise SystemExit(f"DXGI stream error: {stream.last_error}")
            time.sleep(0.02)
        if frame is None:
            raise SystemExit("DXGI stream produced no frame within five seconds.")

        pixels = frame.image[:, :, :3].astype("int16")
        magenta = (
            (pixels[:, :, 0] > 220)
            & (pixels[:, :, 1] < 40)
            & (pixels[:, :, 2] > 220)
        )
        magenta_ratio = float(magenta.mean())
        if magenta_ratio >= 0.25:
            raise SystemExit(
                f"Overlay leaked into DXGI frame: {magenta_ratio:.0%} magenta pixels."
            )
        print(
            f"OK: exclusion={exclusion_method} frame={frame.image.shape[1]}x{frame.image.shape[0]} "
            f"overlay_pixels={magenta_ratio:.1%} roi={capture_region}"
        )
    finally:
        if stream is not None:
            stream.release()
        if window is not None:
            window.close()
        if app is not None:
            app.processEvents()
        reader_module._ACTIVE_WINDOW = None
        QApplication.exec = app_exec


if __name__ == "__main__":
    main()
