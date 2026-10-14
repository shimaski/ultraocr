from __future__ import annotations

from manga_rt_translator.config import OverlayConfig


def edit_overlay_appearance(parent: object, config: OverlayConfig) -> bool:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFontComboBox,
        QFormLayout,
        QSpinBox,
    )

    dialog = QDialog(parent)
    dialog.setWindowTitle("Aparência das traduções")
    form = QFormLayout(dialog)

    family = QFontComboBox(dialog)
    family.setObjectName("overlayFontFamily")
    family.setCurrentFont(QFont(config.font_family))
    form.addRow("Fonte", family)

    size = QSpinBox(dialog)
    size.setObjectName("overlayFontSize")
    size.setRange(8, 36)
    size.setValue(max(8, min(36, int(config.font_size))))
    form.addRow("Tamanho", size)

    opacity = QDoubleSpinBox(dialog)
    opacity.setObjectName("overlayOpacity")
    opacity.setRange(0.35, 1.0)
    opacity.setSingleStep(0.05)
    opacity.setDecimals(2)
    opacity.setValue(max(0.35, min(1.0, float(config.opacity))))
    form.addRow("Opacidade", opacity)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Ok
        | QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    form.addRow(buttons)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return False

    config.font_family = family.currentFont().family()
    config.font_size = size.value()
    config.opacity = opacity.value()
    return True
