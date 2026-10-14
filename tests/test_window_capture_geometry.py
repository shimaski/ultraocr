from manga_rt_translator.capture import windows_windows
from manga_rt_translator.models import Rect


def test_window_capture_geometry_returns_crop_origin_and_monitor(monkeypatch):
    monitors = [
        windows_windows.MonitorInfo(-1920, 0, 0, 1080, r"\\.\DISPLAY1"),
        windows_windows.MonitorInfo(0, 0, 1920, 1080, r"\\.\DISPLAY2"),
    ]
    monkeypatch.setattr(windows_windows, "list_monitors", lambda: monitors)
    window = windows_windows.WindowInfo(
        handle=10,
        title="Reader",
        class_name="Browser",
        rect=Rect(-500, 100, 700, 300),
    )

    output, region, origin, monitor = (
        windows_windows.resolve_capture_geometry_for_window(window)
    )

    assert output == 0
    assert region == (1420, 100, 1920, 400)
    assert origin == (-500, 100)
    assert monitor == monitors[0]
    assert windows_windows.resolve_capture_region_for_window(window) == (
        output,
        region,
    )


def test_monitor_index_maps_qt_device_name_to_dxgi_enumeration(monkeypatch):
    monkeypatch.setattr(
        windows_windows,
        "list_monitors",
        lambda: [
            windows_windows.MonitorInfo(-1920, 0, 0, 1080, r"\\.\DISPLAY1"),
            windows_windows.MonitorInfo(0, 0, 1920, 1080, r"\\.\DISPLAY2"),
        ],
    )

    assert windows_windows.monitor_index_for_device_name("DISPLAY2") == 1
    assert windows_windows.monitor_index_for_device_name(r"\\.\DISPLAY1") == 0
