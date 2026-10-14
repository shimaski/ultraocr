from __future__ import annotations

import ctypes
import platform
from ctypes import wintypes
from typing import Callable


WM_HOTKEY = 0x0312
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_NOREPEAT = 0x4000
HOTKEY_TRANSLATE = 0x4D51
HOTKEY_CLEAR_REGION = 0x4D52


class _Point(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class WinMessage(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("message", wintypes.UINT),
        ("wParam", wintypes.WPARAM),
        ("lParam", wintypes.LPARAM),
        ("time", wintypes.DWORD),
        ("pt", _Point),
    ]


class GlobalHotkeyManager:
    """Registers thread-wide Windows hotkeys and dispatches WM_HOTKEY via Qt."""

    def __init__(
        self,
        app: object,
        callbacks: dict[int, Callable[[], None]],
        *,
        user32: object | None = None,
        get_last_error: Callable[[], int] | None = None,
    ) -> None:
        if user32 is None and platform.system().lower() != "windows":
            raise RuntimeError("Global hotkeys are only implemented on Windows.")
        self.app = app
        self.callbacks = dict(callbacks)
        self.user32 = user32 or ctypes.WinDLL("user32", use_last_error=True)
        self._get_last_error = get_last_error or ctypes.get_last_error
        self.registered: set[int] = set()
        self.errors: dict[int, int] = {}
        self._bind_functions()

        from PySide6.QtCore import QAbstractNativeEventFilter

        manager = self

        class HotkeyEventFilter(QAbstractNativeEventFilter):
            def nativeEventFilter(self, event_type: object, message: object) -> bool:  # noqa: N802
                event_name = bytes(event_type)
                if event_name not in (b"windows_dispatcher_MSG", b"windows_generic_MSG"):
                    return False
                try:
                    address = int(message)
                    native_message = ctypes.cast(
                        address, ctypes.POINTER(WinMessage)
                    ).contents
                except (TypeError, ValueError, OSError):
                    return False
                if native_message.message != WM_HOTKEY:
                    return False
                manager._dispatch(int(native_message.wParam))
                return True

        self.event_filter = HotkeyEventFilter()
        self.app.installNativeEventFilter(self.event_filter)
        self._register(HOTKEY_TRANSLATE, ord("T"))
        self._register(HOTKEY_CLEAR_REGION, ord("R"))

    def _bind_functions(self) -> None:
        if getattr(self.user32, "_is_real_win32", False):
            return
        if isinstance(self.user32, ctypes.WinDLL):
            self.user32.RegisterHotKey.argtypes = [
                wintypes.HWND,
                ctypes.c_int,
                wintypes.UINT,
                wintypes.UINT,
            ]
            self.user32.RegisterHotKey.restype = wintypes.BOOL
            self.user32.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
            self.user32.UnregisterHotKey.restype = wintypes.BOOL

    def _register(self, hotkey_id: int, virtual_key: int) -> None:
        modifiers = MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT
        if self.user32.RegisterHotKey(None, hotkey_id, modifiers, virtual_key):
            self.registered.add(hotkey_id)
        else:
            self.errors[hotkey_id] = int(self._get_last_error())

    def _dispatch(self, hotkey_id: int) -> None:
        callback = self.callbacks.get(hotkey_id)
        if callback is not None:
            callback()

    def close(self) -> None:
        for hotkey_id in tuple(self.registered):
            self.user32.UnregisterHotKey(None, hotkey_id)
        self.registered.clear()
        self.app.removeNativeEventFilter(self.event_filter)

    @property
    def translate_registered(self) -> bool:
        return HOTKEY_TRANSLATE in self.registered

    @property
    def clear_region_registered(self) -> bool:
        return HOTKEY_CLEAR_REGION in self.registered
