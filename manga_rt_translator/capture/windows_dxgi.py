from __future__ import annotations

import time
from typing import Iterator

from manga_rt_translator.models import Frame
from manga_rt_translator.capture.windows_windows import (
    find_window,
    resolve_capture_region_for_window,
)


class WindowsDxgiStream:
    """Windows frame stream using DXGI Desktop Duplication through dxcam.

    The important part is that `start(video_mode=True)` creates a continuous
    capture loop. The pipeline consumes the latest video frame instead of
    calling a screenshot API repeatedly.
    """

    def __init__(
        self,
        fps: int = 30,
        output_idx: int = 0,
        region: tuple[int, int, int, int] | None = None,
        window_title: str | None = None,
        window_handle: int | None = None,
    ) -> None:
        self.fps = fps
        self.output_idx = output_idx
        self.region = region
        self.window_title = window_title
        self.window_handle = window_handle
        self._camera = None
        self._running = False
        self._index = 0

    def start(self) -> None:
        if self._running:
            return
        try:
            import dxcam
        except ImportError as exc:
            raise RuntimeError(
                "Windows capture requires dxcam. Install with: "
                'pip install -e ".[windows]"'
            ) from exc

        region = self.region
        output_idx = self.output_idx
        if region is None and (self.window_title or self.window_handle is not None):
            window = find_window(
                title_contains=self.window_title,
                handle=self.window_handle,
            )
            if window is None:
                selector = (
                    f"handle={self.window_handle}"
                    if self.window_handle is not None
                    else f"title contains {self.window_title!r}"
                )
                raise RuntimeError(f"Unable to find visible window for {selector}.")
            output_idx, region = resolve_capture_region_for_window(window)

        self._camera = dxcam.create(
            output_idx=output_idx,
            output_color="RGB",
        )
        self._camera.start(
            target_fps=self.fps,
            video_mode=True,
            region=region,
        )
        self._running = True

    def frames(self) -> Iterator[Frame]:
        if not self._running:
            self.start()
        assert self._camera is not None

        sleep_seconds = max(0.001, 1.0 / max(1, self.fps * 2))
        while self._running:
            image = self._camera.get_latest_frame()
            if image is None:
                time.sleep(sleep_seconds)
                continue
            self._index += 1
            yield Frame(image=image, timestamp=time.time(), index=self._index)

    def stop(self) -> None:
        if self._camera is not None:
            self._camera.stop()
        self._running = False
