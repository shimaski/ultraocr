from __future__ import annotations

import ctypes

from manga_rt_translator.capture import windows_capture_affinity as affinity


class _ApiFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


class _User32:
    def __init__(self, display_affinity, composition=None):
        self.SetWindowDisplayAffinity = _ApiFunction(display_affinity)
        if composition is not None:
            self.SetWindowCompositionAttribute = _ApiFunction(composition)


def _mock_windows(monkeypatch, user32, error=8):
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_args, **_kwargs: user32, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: error, raising=False)


def test_prefers_window_display_affinity(monkeypatch):
    calls = []
    user32 = _User32(lambda hwnd, flags: calls.append((hwnd, flags)) or True)
    _mock_windows(monkeypatch, user32)

    assert affinity.exclude_window_from_dxgi_capture(123) == (
        True,
        "WDA_EXCLUDEFROMCAPTURE",
        0,
    )
    assert calls == [(123, 0x11)]


def test_falls_back_to_desktop_duplication_exclusion(monkeypatch):
    def set_composition(hwnd, pointer):
        data = ctypes.cast(
            pointer, ctypes.POINTER(affinity._WindowCompositionAttributeData)
        ).contents
        enabled = ctypes.cast(data.data, ctypes.POINTER(ctypes.c_int)).contents.value
        assert hwnd == 456
        assert data.attribute == 24
        assert data.size == ctypes.sizeof(ctypes.c_int)
        assert enabled == 1
        return True

    user32 = _User32(lambda *_args: False, set_composition)
    _mock_windows(monkeypatch, user32)

    assert affinity.exclude_window_from_dxgi_capture(456) == (
        True,
        "WCA_EXCLUDED_FROM_DDA",
        8,
    )


def test_reports_failure_when_both_apis_reject_exclusion(monkeypatch):
    user32 = _User32(lambda *_args: False, lambda *_args: False)
    _mock_windows(monkeypatch, user32, error=87)

    assert affinity.exclude_window_from_dxgi_capture(789) == (
        False,
        "WCA_EXCLUDED_FROM_DDA",
        87,
    )
