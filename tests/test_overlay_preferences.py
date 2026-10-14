import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialogButtonBox, QDoubleSpinBox, QFontComboBox, QSpinBox

from manga_rt_translator.config import OverlayConfig
from manga_rt_translator.overlay_preferences import edit_overlay_appearance


def test_overlay_appearance_dialog_applies_values_on_accept():
    app = QApplication.instance() or QApplication([])
    config = OverlayConfig(font_size=18, opacity=0.94)

    def accept_dialog() -> None:
        dialog = app.activeModalWidget()
        assert dialog is not None
        dialog.findChild(QFontComboBox, "overlayFontFamily").setCurrentIndex(0)
        dialog.findChild(QSpinBox, "overlayFontSize").setValue(26)
        dialog.findChild(QDoubleSpinBox, "overlayOpacity").setValue(0.7)
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Ok).click()

    QTimer.singleShot(100, accept_dialog)
    assert edit_overlay_appearance(None, config)
    assert config.font_size == 26
    assert config.opacity == 0.7
    assert config.font_family


def test_overlay_appearance_dialog_cancel_keeps_existing_values():
    app = QApplication.instance() or QApplication([])
    config = OverlayConfig(font_family="Segoe UI", font_size=18, opacity=0.94)

    def cancel_dialog() -> None:
        dialog = app.activeModalWidget()
        assert dialog is not None
        dialog.findChild(QSpinBox, "overlayFontSize").setValue(30)
        dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.StandardButton.Cancel).click()

    QTimer.singleShot(100, cancel_dialog)
    assert not edit_overlay_appearance(None, config)
    assert (config.font_family, config.font_size, config.opacity) == (
        "Segoe UI",
        18,
        0.94,
    )
