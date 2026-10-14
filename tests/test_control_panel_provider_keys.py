from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMainWindow

import manga_rt_translator.ui as ui_module
from manga_rt_translator.capture.windows_windows import WindowInfo
from manga_rt_translator.config import AppConfig
from manga_rt_translator.models import Rect


def test_control_panel_keeps_translation_keys_separate(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    config = AppConfig()
    config.translation.deepl_api_key = "deepl-from-config"
    config.translation.gemini_api_key = "gemini-from-config"
    config.translation.libre_api_key = "libre-from-config"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(ui_module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ui_module, "load_config", lambda _path: config)
    monkeypatch.setattr(
        ui_module,
        "list_visible_windows",
        lambda: [WindowInfo(123, "Manga", "Browser", Rect(0, 0, 800, 600))],
    )

    def exercise_panel(_app):
        panel = next(
            widget
            for widget in app.topLevelWidgets()
            if isinstance(widget, QMainWindow)
            and widget.windowTitle() == "Manga Real-Time Translator"
        )
        key_input = panel.deepl_key_input
        panel.mode_combo.setCurrentText("DeepL")
        assert key_input.isEnabled()
        assert key_input.text() == "deepl-from-config"
        key_input.setText("deepl-entered")

        panel.mode_combo.setCurrentText("Gemini")
        assert key_input.isEnabled()
        assert key_input.text() == "gemini-from-config"
        key_input.setText("gemini-entered")

        panel.mode_combo.setCurrentText("Libre")
        assert key_input.isEnabled()
        assert key_input.text() == "libre-from-config"
        key_input.setText("libre-entered")

        panel.mode_combo.setCurrentText("DeepL")
        assert key_input.text() == "deepl-entered"
        resolved = panel._build_config()
        assert resolved.translation.deepl_api_key == "deepl-entered"
        assert resolved.translation.gemini_api_key == "gemini-entered"
        assert resolved.translation.libre_api_key == "libre-entered"
        panel.mode_combo.setCurrentText("Libre")
        resolved = panel._build_config()
        monkeypatch.setattr(
            ui_module.subprocess,
            "Popen",
            lambda args, **_kwargs: args,
        )
        command = panel._spawn_runtime(resolved)
        assert "--libre-api-key" in command
        assert "libre-entered" in command
        panel.mode_combo.setCurrentText("DeepL")
        panel._save_state()
        state = json.loads(
            (tmp_path / ".manga_rt_translator_ui_state.json").read_text(
                encoding="utf-8"
            )
        )
        assert state["provider_api_keys"] == {
            "deepl": "deepl-entered",
            "gemini": "gemini-entered",
            "libre": "libre-entered",
        }
        panel.saved_state = state
        panel._restore_state_to_widgets()
        assert key_input.text() == "deepl-entered"
        panel.mode_combo.setCurrentText("Gemini")
        assert key_input.text() == "gemini-entered"
        panel.mode_combo.setCurrentText("Libre")
        assert key_input.text() == "libre-entered"
        panel.saved_state = {
            "provider": "DeepL",
            "deepl_api_key": "legacy-deepl-key",
        }
        panel._restore_state_to_widgets()
        assert key_input.text() == "legacy-deepl-key"
        panel.close()
        app.processEvents()
        return 0

    monkeypatch.setattr(QApplication, "exec", exercise_panel)
    assert ui_module.run_control_panel(None, None) == 0

