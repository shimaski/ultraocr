from manga_rt_translator.models import CaptureViewport, Rect
from manga_rt_translator.overlay_geometry import logical_capture_rect


def test_physical_capture_origin_maps_to_logical_secondary_display_coordinates():
    viewport = CaptureViewport(
        capture_origin=(2160, 120),
        capture_size=(1200, 800),
        monitor_origin=(1920, 0),
        monitor_device_name=r"\\.\DISPLAY2",
        monitor_index=1,
    )

    rect = logical_capture_rect(
        viewport,
        monitor_geometry=Rect(1920, 0, 1280, 800),
        device_pixel_ratio=1.25,
    )

    assert rect == Rect(2112, 96, 960, 640)


def test_capture_viewport_handles_negative_monitor_coordinates():
    viewport = CaptureViewport(
        capture_origin=(-1840, 90),
        capture_size=(700, 500),
        monitor_origin=(-1920, 0),
        monitor_index=0,
    )

    rect = logical_capture_rect(
        viewport,
        monitor_geometry=Rect(-1536, 0, 1536, 864),
        device_pixel_ratio=1.25,
    )

    assert rect == Rect(-1472, 72, 560, 400)
