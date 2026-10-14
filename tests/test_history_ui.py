import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QTableWidget

from manga_rt_translator.cache import TranslationMemory
from manga_rt_translator.ui import show_translation_history


def test_history_dialog_edits_translation_and_returns_changed():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "history-ui.sqlite3"
        memory = TranslationMemory(db_path)
        memory.save("I will protect you", "Eu vou proteger você", "en->pt-BR")
        memory.record_history(
            "I will protect you", "Eu vou proteger você", "en->pt-BR", "google"
        )
        memory.close()

        def select_and_edit() -> None:
            dialog = app.activeModalWidget()
            assert isinstance(dialog, QDialog)
            table = dialog.findChild(QTableWidget)
            assert table is not None and table.rowCount() == 1
            table.selectRow(0)
            button = next(
                child for child in dialog.findChildren(QPushButton)
                if child.text() == "Corrigir tradução"
            )
            button.click()
            QTimer.singleShot(0, dialog.accept)

        with patch(
            "PySide6.QtWidgets.QInputDialog.getMultiLineText",
            return_value=("Eu vou salvar você", True),
        ):
            QTimer.singleShot(100, select_and_edit)
            assert show_translation_history(None, db_path)

        memory = TranslationMemory(db_path)
        assert memory.lookup("I will protect you", "en->pt-BR").translated_text == "Eu vou salvar você"
        assert memory.history()[0].provider == "manual"
        memory.close()
