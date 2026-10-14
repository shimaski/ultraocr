import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPushButton,
)

from manga_rt_translator.glossary import Glossary
from manga_rt_translator.ui import edit_glossary


def _button(dialog: QDialog, label: str) -> QPushButton:
    return next(
        child for child in dialog.findChildren(QPushButton)
        if child.text() == label
    )


def test_glossary_editor_import_replaces_draft_only_after_confirmation():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = Glossary(
            names={"Aki": "Aki"},
            terms={"rental girlfriend": "namorada de aluguel"},
            instructions=["Use português brasileiro natural."],
        )
        source_path = root / "import.toml"
        source.save(source_path)
        active_path = root / "active.toml"
        original = Glossary(names={"Mika": "Mika"})
        original.save(active_path)

        def import_and_save() -> None:
            dialog = app.activeModalWidget()
            assert isinstance(dialog, QDialog)
            _button(dialog, "Importar TOML…").click()
            button_box = dialog.findChild(QDialogButtonBox)
            assert button_box is not None
            button_box.button(QDialogButtonBox.StandardButton.Save).click()

        with (
            patch(
                "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                return_value=(str(source_path), "Glossário TOML (*.toml)"),
            ),
            patch(
                "PySide6.QtWidgets.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
        ):
            QTimer.singleShot(100, import_and_save)
            assert edit_glossary(None, original, active_path, root / "cache.sqlite3")

        assert original == source
        assert Glossary.load(active_path) == source


def test_glossary_editor_exports_current_draft_without_saving_it():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        exported_path = root / "portable.toml"
        draft = Glossary(
            names={"Rin": "Rin"},
            terms={"guild master": "mestre da guilda"},
            instructions=["Preserve the character's tone."],
        )

        def export_and_cancel() -> None:
            dialog = app.activeModalWidget()
            assert isinstance(dialog, QDialog)
            _button(dialog, "Exportar TOML…").click()
            button_box = dialog.findChild(QDialogButtonBox)
            assert button_box is not None
            button_box.button(QDialogButtonBox.StandardButton.Cancel).click()

        with patch(
            "PySide6.QtWidgets.QFileDialog.getSaveFileName",
            return_value=(str(exported_path), "Glossário TOML (*.toml)"),
        ):
            QTimer.singleShot(100, export_and_cancel)
            assert not edit_glossary(None, draft, root / "active.toml", root / "cache.sqlite3")

        assert Glossary.load(exported_path) == draft
