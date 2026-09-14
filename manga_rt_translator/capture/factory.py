from __future__ import annotations

import platform

from manga_rt_translator.capture.base import ScreenStream
from manga_rt_translator.config import CaptureConfig


def create_screen_stream(config: CaptureConfig) -> ScreenStream:
    backend = config.backend.lower()
    system = platform.system().lower()

    if backend in {"auto", "windows-dxgi", "dxgi"} and system == "windows":
        from manga_rt_translator.capture.windows_dxgi import WindowsDxgiStream

        return WindowsDxgiStream(
            fps=config.fps,
            output_idx=config.monitor,
            region=config.region,
            window_title=config.window_title,
            window_handle=config.window_handle,
        )

    if backend in {"auto", "macos", "screencapturekit"} and system == "darwin":
        raise NotImplementedError(
            "macOS ScreenCaptureKit backend is planned but not implemented yet."
        )

    if backend in {"auto", "linux", "pipewire"} and system == "linux":
        raise NotImplementedError(
            "Linux PipeWire backend is planned but not implemented yet."
        )

    raise RuntimeError(
        f"No capture backend available for backend={config.backend!r}, system={system!r}."
    )
