from __future__ import annotations

import ctypes

from manga_rt_translator.global_hotkeys import (
    HOTKEY_CLEAR_REGION,
    HOTKEY_TRANSLATE,
    MOD_CONTROL,
    MOD_NOREPEAT,
    MOD_SHIFT,
    WM_HOTKEY,
    GlobalHotkeyManager,
    WinMessage,
)


class FakeApp:
    def installNativeEventFilter(self, event_filter):
        self.event_filter = event_filter

    def removeNativeEventFilter(self, event_filter):
        assert self.event_filter is event_filter
        self.event_filter = None


class FakeUser32:
    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.registered = []
        self.unregistered = []

    def RegisterHotKey(self, hwnd, hotkey_id, modifiers, virtual_key):
        self.registered.append((hwnd, hotkey_id, modifiers, virtual_key))
        return int(hotkey_id not in self.fail_ids)

    def UnregisterHotKey(self, hwnd, hotkey_id):
        self.unregistered.append((hwnd, hotkey_id))
        return 1


def test_thread_global_hotkeys_register_dispatch_and_unregister():
    app = FakeApp()
    user32 = FakeUser32()
    invoked = []
    manager = GlobalHotkeyManager(
        app,
        {
            HOTKEY_TRANSLATE: lambda: invoked.append("translate"),
            HOTKEY_CLEAR_REGION: lambda: invoked.append("clear"),
        },
        user32=user32,
    )
    try:
        assert manager.registered == {HOTKEY_TRANSLATE, HOTKEY_CLEAR_REGION}
        assert user32.registered[0] == (
            None,
            HOTKEY_TRANSLATE,
            MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT,
            ord("T"),
        )
        message = WinMessage()
        message.message = WM_HOTKEY
        message.wParam = HOTKEY_TRANSLATE
        handled = manager.event_filter.nativeEventFilter(
            b"windows_dispatcher_MSG", ctypes.addressof(message)
        )
        assert handled is True
        assert invoked == ["translate"]

        message.message = 0x1234
        assert manager.event_filter.nativeEventFilter(
            b"windows_dispatcher_MSG", ctypes.addressof(message)
        ) is False
    finally:
        manager.close()

    assert app.event_filter is None
    assert user32.unregistered == [
        (None, HOTKEY_TRANSLATE),
        (None, HOTKEY_CLEAR_REGION),
    ]


def test_registration_conflict_is_reported_without_losing_other_hotkey():
    app = FakeApp()
    user32 = FakeUser32(fail_ids={HOTKEY_CLEAR_REGION})
    manager = GlobalHotkeyManager(
        app,
        {HOTKEY_TRANSLATE: lambda: None, HOTKEY_CLEAR_REGION: lambda: None},
        user32=user32,
        get_last_error=lambda: 1409,
    )
    try:
        assert manager.translate_registered
        assert not manager.clear_region_registered
        assert manager.errors == {HOTKEY_CLEAR_REGION: 1409}
    finally:
        manager.close()
