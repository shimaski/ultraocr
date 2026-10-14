"""Windows APIs for keeping the reader overlay out of DXGI captures."""
from __future__ import annotations

import ctypes


class _WindowCompositionAttributeData(ctypes.Structure):
    _fields_ = [
        ("attribute", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_size_t),
    ]


def exclude_window_from_dxgi_capture(hwnd: int) -> tuple[bool, str, int]:
    """Return (excluded, method, error); prefer the documented WDA API."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    setter = user32.SetWindowDisplayAffinity
    setter.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    setter.restype = ctypes.c_bool
    if setter(hwnd, 0x11):
        return True, "WDA_EXCLUDEFROMCAPTURE", 0

    affinity_error = ctypes.get_last_error()
    composition_setter = getattr(user32, "SetWindowCompositionAttribute", None)
    if composition_setter is None:
        return False, "WDA_EXCLUDEFROMCAPTURE", affinity_error

    composition_setter.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowCompositionAttributeData),
    ]
    composition_setter.restype = ctypes.c_bool
    enabled = ctypes.c_int(1)
    data = _WindowCompositionAttributeData(
        24,
        ctypes.cast(ctypes.byref(enabled), ctypes.c_void_p),
        ctypes.sizeof(enabled),
    )
    if composition_setter(hwnd, ctypes.byref(data)):
        return True, "WCA_EXCLUDED_FROM_DDA", affinity_error
    return False, "WCA_EXCLUDED_FROM_DDA", ctypes.get_last_error()
