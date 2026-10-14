from __future__ import annotations

from manga_rt_translator.models import CaptureViewport, Rect


def logical_capture_rect(
    viewport: CaptureViewport,
    monitor_geometry: Rect,
    device_pixel_ratio: float,
) -> Rect:
    scale = max(1.0, float(device_pixel_ratio))
    return Rect(
        x=monitor_geometry.x
        + round((viewport.capture_origin[0] - viewport.monitor_origin[0]) / scale),
        y=monitor_geometry.y
        + round((viewport.capture_origin[1] - viewport.monitor_origin[1]) / scale),
        width=max(1, round(viewport.capture_size[0] / scale)),
        height=max(1, round(viewport.capture_size[1] / scale)),
    )
