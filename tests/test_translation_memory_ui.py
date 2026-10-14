from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox, QPushButton

from manga_rt_translator.cache import TranslationMemory
from manga_rt_translator.ui import manage_translation_memory


def _button(dialog: QDialog, label: str) -> QPushButton:
    return next(
        child for child in dialog.findChildren(QPushButton)
        if child.text() == label
    )


def test_memory_dialog_exports_json_from_the_reader_database():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        db_path = root / "cache.sqlite3"
        memory = TranslationMemory(db_path)
        memory.save("Good morning", "Bom dia", "en->pt-BR")
        memory.close()
        export_path = root / "portable.json"

        def export_and_close() -> None:
            dialog = app.activeModalWidget()
            assert isinstance(dialog, QDialog)
            _button(dialog, "Exportar JSON…").click()
            dialog.accept()

        with (
            patch(
                "PySide6.QtWidgets.QFileDialog.getSaveFileName",
                return_value=(str(export_path), "Memória JSON (*.json)"),
            ),
            patch("PySide6.QtWidgets.QMessageBox.information"),
        ):
            QTimer.singleShot(100, export_and_close)
            assert not manage_translation_memory(None, db_path)

        imported = TranslationMemory(root / "check.sqlite3")
        try:
            assert imported.import_json(export_path).inserted == 1
            assert imported.lookup("Good morning", "en->pt-BR").translated_text == "Bom dia"
        finally:
            imported.close()


def test_memory_dialog_imports_and_updates_the_selected_database():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = TranslationMemory(root / "source.sqlite3")
        source.save("I am here", "Estou aqui", "en->pt-BR")
        export_path = root / "portable.json"
        source.export_json(export_path)
        source.close()

        db_path = root / "cache.sqlite3"
        target = TranslationMemory(db_path)
        target.save("I am here", "Eu estou aqui", "en->pt-BR")
        target.save("Welcome", "Bem-vindo", "en->pt-BR")
        target.close()

        def import_and_close() -> None:
            dialog = app.activeModalWidget()
            assert isinstance(dialog, QDialog)
            _button(dialog, "Importar JSON…").click()
            dialog.accept()

        with (
            patch(
                "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                return_value=(str(export_path), "Memória JSON (*.json)"),
            ),
            patch(
                "PySide6.QtWidgets.QMessageBox.question",
                return_value=QMessageBox.StandardButton.Yes,
            ),
            patch("PySide6.QtWidgets.QMessageBox.information"),
        ):
            QTimer.singleShot(100, import_and_close)
            assert manage_translation_memory(None, db_path)

        target = TranslationMemory(db_path)
        try:
            assert target.lookup("I am here", "en->pt-BR").translated_text == "Estou aqui"
            assert target.lookup("Welcome", "en->pt-BR").translated_text == "Bem-vindo"
        finally:
            target.close()


def test_memory_dialog_can_keep_existing_translations_on_conflict():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = TranslationMemory(root / "source.sqlite3")
        source.save("Thank you", "Muito obrigado", "en->pt-BR")
        export_path = root / "portable.json"
        source.export_json(export_path)
        source.close()

        db_path = root / "cache.sqlite3"
        target = TranslationMemory(db_path)
        target.save("Thank you", "Obrigado", "en->pt-BR")
        target.close()

        def import_and_close() -> None:
            dialog = app.activeModalWidget()
            assert isinstance(dialog, QDialog)
            _button(dialog, "Importar JSON…").click()
            dialog.accept()

        with (
            patch(
                "PySide6.QtWidgets.QFileDialog.getOpenFileName",
                return_value=(str(export_path), "Memória JSON (*.json)"),
            ),
            patch(
                "PySide6.QtWidgets.QMessageBox.question",
                return_value=QMessageBox.StandardButton.No,
            ),
            patch("PySide6.QtWidgets.QMessageBox.information"),
        ):
            QTimer.singleShot(100, import_and_close)
            assert not manage_translation_memory(None, db_path)

        target = TranslationMemory(db_path)
        try:
            assert target.lookup("Thank you", "en->pt-BR").translated_text == "Obrigado"
        finally:
            target.close()
