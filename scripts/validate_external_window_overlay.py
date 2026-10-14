"""Exercise the external-window DXGI-to-overlay path without saving frames."""
from __future__ import annotations

import platform
import time

if platform.system().lower() != "windows":
    raise SystemExit("External window overlay validation is Windows-only.")

from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from manga_rt_translator.capture.windows_dxgi import WindowsDxgiStream
from manga_rt_translator.capture.windows_windows import list_visible_windows
from manga_rt_translator.config import AppConfig
from manga_rt_translator.models import OverlayItem, Rect
from manga_rt_translator.overlay import create_qt_overlay
from manga_rt_translator.overlay_geometry import logical_capture_rect


def main() -> None:
    app = QApplication.instance() or QApplication([])
    target_widget = QWidget()
    target_widget.setWindowTitle("Manga Translator Capture Probe")
    target_widget.setStyleSheet("background: #d8e4ed; color: #152330;")
    target_layout = QVBoxLayout(target_widget)
    target_layout.addWidget(QLabel("Temporary capture target for overlay validation"))
    target_widget.setGeometry(180, 140, 720, 520)
    target_widget.show()
    app.processEvents()

    windows = list_visible_windows()
    target_handle = int(target_widget.winId())
    target = next((window for window in windows if window.handle == target_handle), None)
    if target is None:
        target_widget.close()
        raise SystemExit("Qt's temporary target window was not enumerated.")

    stream = WindowsDxgiStream(fps=30, window_handle=target.handle)
    overlay = None
    frames = None
    try:
        frames = stream.frames()
        captured = next(frames)
        viewport = captured.capture_viewport
        if viewport is None:
            raise SystemExit("DXGI frame did not include capture viewport metadata.")
        if viewport.capture_origin == viewport.monitor_origin:
            raise SystemExit("Temporary target did not exercise a nonzero screen origin.")

        overlay = create_qt_overlay(AppConfig().overlay)
        overlay.set_capture_viewport(viewport)
        overlay.update(
            [
                OverlayItem(
                    bbox=Rect(40, 30, 100, 42),
                    text="OVERLAY TEST",
                    source_text="SOURCE TEXT",
                    background_color=(255, 0, 255),
                )
            ]
        )
        app.processEvents()

        screens = QApplication.screens()
        screen = next(
            (
                candidate
                for candidate in screens
                if viewport.monitor_device_name
                and candidate.name().casefold()
                == viewport.monitor_device_name.casefold()
            ),
            screens[viewport.monitor_index]
            if 0 <= viewport.monitor_index < len(screens)
            else QApplication.primaryScreen(),
        )
        if screen is None:
            raise SystemExit("Qt did not expose the captured monitor.")
        geometry = screen.geometry()
        expected = logical_capture_rect(
            viewport,
            Rect(geometry.x(), geometry.y(), geometry.width(), geometry.height()),
            screen.devicePixelRatio(),
        )
        app.processEvents()
        if overlay.widget.geometry().getRect() != (
            expected.x,
            expected.y,
            expected.width,
            expected.height,
        ):
            raise SystemExit(
                "Overlay placement mismatch: "
                f"actual={overlay.widget.geometry().getRect()} "
                f"expected={(expected.x, expected.y, expected.width, expected.height)}"
            )

        time.sleep(0.15)
        captured = next(frames)
        probe = captured.image[24:36, 42:58, :3].astype("int16")
        magenta = (
            (probe[:, :, 0] > 220)
            & (probe[:, :, 1] < 60)
            & (probe[:, :, 2] > 220)
        )
        leak_ratio = float(magenta.mean())
        if leak_ratio >= 0.25:
            raise SystemExit(
                f"Translation overlay leaked into DXGI capture ({leak_ratio:.0%})."
            )

        exclusion = overlay.capture_exclusion
        if exclusion is not None and not exclusion[0]:
            raise SystemExit(
                "Windows rejected overlay exclusion: "
                f"method={exclusion[1]} error={exclusion[2]}"
            )
        print(
            "OK: "
            f"window={target.title!r} monitor={viewport.monitor_device_name or viewport.monitor_index} "
            f"capture={viewport.capture_size[0]}x{viewport.capture_size[1]} "
            f"overlay={overlay.widget.geometry().getRect()} "
            f"scale={screen.devicePixelRatio():g} "
            f"exclusion={exclusion} leaked_pixels={leak_ratio:.1%}"
        )
    finally:
        if frames is not None:
            frames.close()
        stream.stop()
        if overlay is not None:
            overlay.widget.close()
        target_widget.close()
        app.processEvents()


if __name__ == "__main__":
    main()
