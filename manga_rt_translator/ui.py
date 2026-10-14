from __future__ import annotations

import copy
import json
import platform
import sqlite3
import subprocess
import sys
from pathlib import Path

from manga_rt_translator.capture.windows_windows import WindowInfo, list_visible_windows
from manga_rt_translator.cache import TranslationMemory
from manga_rt_translator.config import AppConfig, load_config
from manga_rt_translator.glossary import Glossary, resolve_glossary_path


def show_translation_history(parent: object, db_path: str | Path) -> bool:
    from datetime import datetime

    from PySide6.QtWidgets import (
        QDialog,
        QHBoxLayout,
        QHeaderView,
        QInputDialog,
        QLineEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
    )

    memory = TranslationMemory(db_path)
    dialog = QDialog(parent)
    dialog.setWindowTitle("Histórico de traduções")
    dialog.resize(920, 480)
    layout = QVBoxLayout(dialog)
    search = QLineEdit(dialog)
    search.setPlaceholderText("Buscar no original ou na tradução")
    layout.addWidget(search)

    table = QTableWidget(0, 5, dialog)
    table.setHorizontalHeaderLabels(["Data", "Original", "Português", "Idioma", "Provedor"])
    table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    table.setSortingEnabled(False)
    header = table.horizontalHeader()
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
    header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
    layout.addWidget(table)

    actions = QHBoxLayout()
    edit_button = QPushButton("Corrigir tradução", dialog)
    edit_button.setEnabled(False)
    actions.addWidget(edit_button)
    actions.addStretch(1)
    layout.addLayout(actions)
    entries = []
    changed = False

    def refresh(query: str = "") -> None:
        nonlocal entries
        entries = memory.history(query)
        table.setRowCount(len(entries))
        for row_index, entry in enumerate(entries):
            values = (
                datetime.fromtimestamp(entry.created_at).strftime("%d/%m/%Y %H:%M:%S"),
                entry.original_text,
                entry.translated_text,
                entry.language_pair,
                entry.provider,
            )
            for column, value in enumerate(values):
                table.setItem(row_index, column, QTableWidgetItem(value))

    def edit_selected() -> None:
        nonlocal changed
        row_index = table.currentRow()
        if row_index < 0 or row_index >= len(entries):
            return
        entry = entries[row_index]
        text, accepted = QInputDialog.getMultiLineText(
            dialog,
            "Corrigir tradução",
            entry.original_text,
            entry.translated_text,
        )
        if not accepted or not text.strip():
            return
        try:
            changed = memory.edit_history_translation(entry.id, text) or changed
        except Exception as exc:
            from PySide6.QtWidgets import QMessageBox

            QMessageBox.warning(dialog, "Falha ao salvar correção", str(exc))
            return
        refresh(search.text())

    table.itemSelectionChanged.connect(
        lambda: edit_button.setEnabled(table.currentRow() >= 0)
    )
    edit_button.clicked.connect(edit_selected)
    table.cellDoubleClicked.connect(
        lambda row, column: edit_selected() if column == 2 else None
    )

    search.textChanged.connect(refresh)
    refresh()
    try:
        dialog.exec()
    finally:
        memory.close()
    return changed


def manage_translation_memory(parent: object, db_path: str | Path) -> bool:
    from PySide6.QtWidgets import (
        QDialog,
        QFileDialog,
        QHBoxLayout,
        QMessageBox,
        QPushButton,
        QVBoxLayout,
    )

    dialog = QDialog(parent)
    dialog.setWindowTitle("Memória de tradução")
    dialog.setModal(True)
    layout = QVBoxLayout(dialog)
    buttons = QHBoxLayout()
    import_button = QPushButton("Importar JSON…", dialog)
    export_button = QPushButton("Exportar JSON…", dialog)
    close_button = QPushButton("Fechar", dialog)
    buttons.addWidget(import_button)
    buttons.addWidget(export_button)
    buttons.addStretch(1)
    buttons.addWidget(close_button)
    layout.addLayout(buttons)
    imported = False

    def export_file() -> None:
        filename, _ = QFileDialog.getSaveFileName(
            dialog, "Exportar memória de tradução", "translation-memory.json",
            "Memória JSON (*.json)",
        )
        if not filename:
            return
        memory = TranslationMemory(db_path)
        try:
            count = memory.export_json(filename)
        except (OSError, sqlite3.Error) as exc:
            QMessageBox.warning(dialog, "Não foi possível exportar", str(exc))
        else:
            QMessageBox.information(
                dialog, "Memória exportada", f"Entradas exportadas: {count}"
            )
        finally:
            memory.close()

    def import_file() -> None:
        nonlocal imported
        filename, _ = QFileDialog.getOpenFileName(
            dialog, "Importar memória de tradução", "", "Memória JSON (*.json)"
        )
        if not filename:
            return
        choice = QMessageBox.question(
            dialog,
            "Importar memória",
            "Atualizar entradas existentes com a mesma frase e par de idiomas?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        memory = TranslationMemory(db_path)
        try:
            result = memory.import_json(
                filename, conflict="replace" if choice == QMessageBox.StandardButton.Yes else "keep"
            )
        except (OSError, ValueError, sqlite3.Error) as exc:
            QMessageBox.warning(dialog, "Memória inválida", str(exc))
        else:
            imported = result.inserted + result.updated > 0
            QMessageBox.information(
                dialog,
                "Memória importada",
                f"Novas: {result.inserted} · Atualizadas: {result.updated} · "
                f"Preservadas: {result.kept_existing}",
            )
        finally:
            memory.close()

    import_button.clicked.connect(import_file)
    export_button.clicked.connect(export_file)
    close_button.clicked.connect(dialog.accept)
    dialog.exec()
    return imported


def edit_glossary(
    parent: object,
    glossary: Glossary,
    glossary_path: str | Path | None,
    cache_path: str | Path,
    *,
    cache_scope: str | None = None,
) -> bool:
    from PySide6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QHBoxLayout,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QVBoxLayout,
        QWidget,
    )

    dialog = QDialog(parent)
    dialog.setWindowTitle("Glossário")
    dialog.resize(680, 500)
    layout = QVBoxLayout(dialog)
    tabs = QTabWidget(dialog)
    tables: dict[str, QTableWidget] = {}
    for title, key, values in (
        ("Nomes protegidos", "names", glossary.names),
        ("Termos", "terms", glossary.terms),
    ):
        page = QWidget(tabs)
        page_layout = QVBoxLayout(page)
        table = QTableWidget(0, 2, page)
        table.setHorizontalHeaderLabels(["Original", "Usar na tradução"])
        table.horizontalHeader().setStretchLastSection(True)
        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        for source, target in values.items():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(source))
            table.setItem(row, 1, QTableWidgetItem(target))
        buttons = QHBoxLayout()
        add_button = QPushButton("Adicionar")
        remove_button = QPushButton("Remover")
        add_button.clicked.connect(
            lambda _checked=False, grid=table: _add_glossary_row(grid)
        )
        remove_button.clicked.connect(
            lambda _checked=False, grid=table: grid.removeRow(grid.currentRow())
            if grid.currentRow() >= 0 else None
        )
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addStretch(1)
        page_layout.addWidget(table)
        page_layout.addLayout(buttons)
        tabs.addTab(page, title)
        tables[key] = table
    layout.addWidget(tabs)

    instructions = QPlainTextEdit(dialog)
    instructions.setPlaceholderText("Uma regra de tradução por linha")
    instructions.setPlainText("\n".join(glossary.instructions))
    layout.addWidget(instructions)

    glossary_files = QHBoxLayout()
    import_button = QPushButton("Importar TOML…", dialog)
    export_button = QPushButton("Exportar TOML…", dialog)
    glossary_files.addWidget(import_button)
    glossary_files.addWidget(export_button)
    glossary_files.addStretch(1)
    layout.addLayout(glossary_files)

    def replace_table(table: QTableWidget, values: dict[str, str]) -> None:
        table.setRowCount(0)
        for source, target in values.items():
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(source))
            table.setItem(row, 1, QTableWidgetItem(target))

    def import_file() -> None:
        filename, _selected_filter = QFileDialog.getOpenFileName(
            dialog, "Importar glossário", "", "Glossário TOML (*.toml)"
        )
        if not filename:
            return
        answer = QMessageBox.question(
            dialog,
            "Substituir rascunho",
            "O conteúdo atual do editor será substituído. Continuar?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            imported = Glossary.load(filename)
        except Exception as exc:
            QMessageBox.warning(dialog, "Glossário inválido", str(exc))
            return
        replace_table(tables["names"], imported.names)
        replace_table(tables["terms"], imported.terms)
        instructions.setPlainText("\n".join(imported.instructions))

    def export_file() -> None:
        filename, _selected_filter = QFileDialog.getSaveFileName(
            dialog, "Exportar glossário", "glossary.toml", "Glossário TOML (*.toml)"
        )
        if not filename:
            return
        try:
            exported = Glossary(
                _glossary_table_values(tables["names"]),
                _glossary_table_values(tables["terms"]),
                [line.strip() for line in instructions.toPlainText().splitlines() if line.strip()],
            )
            exported.save(filename)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(dialog, "Não foi possível exportar", str(exc))

    import_button.clicked.connect(import_file)
    export_button.clicked.connect(export_file)

    buttons = QDialogButtonBox(
        QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
        parent=dialog,
    )
    layout.addWidget(buttons)

    def save() -> None:
        try:
            names = _glossary_table_values(tables["names"])
            terms = _glossary_table_values(tables["terms"])
        except ValueError as exc:
            QMessageBox.warning(dialog, "Glossário inválido", str(exc))
            return
        next_instructions = [
            line.strip() for line in instructions.toPlainText().splitlines() if line.strip()
        ]
        instructions_changed = next_instructions != glossary.instructions
        if instructions_changed:
            answer = QMessageBox.question(
                dialog,
                "Atualizar cache",
                "As regras gerais mudaram. As traduções em cache serão invalidadas para que a nova regra seja aplicada; o histórico será mantido. Continuar?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        updated = Glossary(names, terms, next_instructions)
        target_path = Path(glossary_path) if glossary_path else Path.cwd() / "glossary.toml"
        try:
            updated.save(target_path)
        except OSError as exc:
            QMessageBox.critical(dialog, "Não foi possível salvar", str(exc))
            return

        changed_sources = _changed_glossary_sources(glossary, updated)
        glossary.names = updated.names
        glossary.terms = updated.terms
        glossary.instructions = updated.instructions
        try:
            memory = TranslationMemory(cache_path)
            try:
                invalidated = memory.invalidate_glossary_cache(
                    changed_sources,
                    clear_all=instructions_changed,
                    language_pair_prefix=cache_scope,
                )
            finally:
                memory.close()
        except Exception as exc:
            QMessageBox.warning(
                dialog,
                "Glossário salvo",
                f"O glossário foi salvo, mas não consegui atualizar o cache: {exc}",
            )
        else:
            if invalidated:
                QMessageBox.information(
                    dialog,
                    "Glossário salvo",
                    f"{invalidated} tradução(ões) em cache serão refeitas com as novas regras. O histórico foi mantido.",
                )
        dialog.accept()

    buttons.accepted.connect(save)
    buttons.rejected.connect(dialog.reject)
    return dialog.exec() == QDialog.DialogCode.Accepted


def _add_glossary_row(table: object) -> None:
    from PySide6.QtWidgets import QTableWidgetItem

    row = table.rowCount()
    table.insertRow(row)
    table.setItem(row, 0, QTableWidgetItem(""))
    table.setItem(row, 1, QTableWidgetItem(""))
    table.setCurrentCell(row, 0)
    table.editItem(table.item(row, 0))


def _glossary_table_values(table: object) -> dict[str, str]:
    values: dict[str, str] = {}
    for row in range(table.rowCount()):
        source_item = table.item(row, 0)
        target_item = table.item(row, 1)
        source = source_item.text().strip() if source_item else ""
        target = target_item.text().strip() if target_item else ""
        if not source and not target:
            continue
        if not source or not target:
            raise ValueError("Preencha os dois campos ou remova a linha vazia.")
        if source in values:
            raise ValueError(f"Termo duplicado: {source}")
        values[source] = target
    return values


def _changed_glossary_sources(old: Glossary, new: Glossary) -> list[str]:
    changed: list[str] = []
    for before, after in ((old.names, new.names), (old.terms, new.terms)):
        changed.extend(
            source for source in before.keys() | after.keys()
            if before.get(source) != after.get(source)
        )
    return changed


def run_control_panel(config_path: str | None, glossary_path: str | None) -> int:
    if platform.system().lower() != "windows":
        raise RuntimeError("The control panel is currently implemented for Windows only.")

    from PySide6.QtCore import QTimer, Qt
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QGridLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    class ControlPanel(QMainWindow):
        def __init__(self, base_config: AppConfig, glossary: Glossary) -> None:
            super().__init__()
            self.base_config = base_config
            self.glossary = glossary
            self.glossary_path = Path(glossary_path) if glossary_path else Path.cwd() / "glossary.toml"
            self.process: subprocess.Popen[str] | None = None
            self._provider_api_keys = {
                "deepl": base_config.translation.deepl_api_key,
                "gemini": base_config.translation.gemini_api_key,
                "libre": base_config.translation.libre_api_key,
            }
            self._active_key_provider = "google"
            self.windows: list[WindowInfo] = []
            self.state_path = Path.cwd() / ".manga_rt_translator_ui_state.json"
            self.stats_path = Path.cwd() / ".manga_rt_translator_runtime_stats.json"
            self.saved_state = self._load_state()
            self.setWindowTitle("Manga Real-Time Translator")
            self.resize(860, 580)
            self._build_ui()
            self._refresh_windows()
            self.timer = QTimer(self)
            self.timer.timeout.connect(self._poll_stats)
            self.timer.start(1000)

        def _build_ui(self) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(16, 16, 16, 16)
            layout.setSpacing(12)

            title = QLabel("Escolha uma janela e inicie a tradução.")
            title.setStyleSheet("font-size: 18px; font-weight: 600;")
            layout.addWidget(title)

            controls = QGridLayout()
            controls.setHorizontalSpacing(12)
            controls.setVerticalSpacing(10)

            self.refresh_button = QPushButton("Atualizar janelas")
            self.refresh_button.clicked.connect(self._refresh_windows)
            controls.addWidget(self.refresh_button, 0, 0)

            self.check_button = QPushButton("Verificar")
            self.check_button.clicked.connect(self._check_readiness)
            controls.addWidget(self.check_button, 0, 1)

            controls.addWidget(QLabel("FPS de captura"), 0, 2)
            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(5, 60)
            self.fps_spin.setValue(self.base_config.capture.fps)
            controls.addWidget(self.fps_spin, 0, 3)

            controls.addWidget(QLabel("Source"), 0, 4)
            self.source_combo = QComboBox()
            self.source_combo.addItems(["en", "japan", "korean"])
            self.source_combo.setCurrentText(self.base_config.ocr.source_lang)
            controls.addWidget(self.source_combo, 0, 5)

            controls.addWidget(QLabel("Provedor"), 1, 0)
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Google", "DeepL", "Gemini", "Libre", "Argos offline", "Echo"])
            self.mode_combo.currentTextChanged.connect(self._sync_mode_ui)
            controls.addWidget(self.mode_combo, 1, 1)

            self.overlay_checkbox = QCheckBox("Overlay")
            self.overlay_checkbox.setChecked(True)
            controls.addWidget(self.overlay_checkbox, 1, 2)

            self.japanese_checkbox = QCheckBox("OCR japonês")
            self.japanese_checkbox.setChecked("japan" in self.base_config.ocr.extra_langs)
            controls.addWidget(self.japanese_checkbox, 1, 3, 1, 3)

            self.korean_checkbox = QCheckBox("OCR coreano")
            self.korean_checkbox.setChecked("korean" in self.base_config.ocr.extra_langs)
            controls.addWidget(self.korean_checkbox, 2, 0, 1, 2)

            controls.addWidget(QLabel("Chave de API"), 2, 2)
            self.deepl_key_input = QLineEdit()
            self.deepl_key_input.textChanged.connect(self._remember_active_api_key)
            controls.addWidget(self.deepl_key_input, 2, 3, 1, 3)

            layout.addLayout(controls)

            self.window_list = QListWidget()
            self.window_list.setAlternatingRowColors(True)
            layout.addWidget(self.window_list, stretch=1)

            buttons = QHBoxLayout()
            self.start_button = QPushButton("Iniciar")
            self.start_button.clicked.connect(self._start_session)
            buttons.addWidget(self.start_button)

            self.stop_button = QPushButton("Parar")
            self.stop_button.setEnabled(False)
            self.stop_button.clicked.connect(self._stop_session)
            buttons.addWidget(self.stop_button)

            self.history_button = QPushButton("Histórico")
            self.history_button.clicked.connect(
                lambda: show_translation_history(self, self.base_config.cache.path)
            )
            buttons.addWidget(self.history_button)

            self.glossary_button = QPushButton("Glossário")
            self.glossary_button.clicked.connect(self._edit_glossary)
            buttons.addWidget(self.glossary_button)
            buttons.addStretch(1)
            layout.addLayout(buttons)

            self.status_label = QLabel("Parado")
            self.status_label.setWordWrap(True)
            self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(self.status_label)
            self._restore_state_to_widgets()
            self._sync_mode_ui(self.mode_combo.currentText())

        def closeEvent(self, event: object) -> None:  # noqa: N802
            self._save_state()
            self._stop_session()
            return super().closeEvent(event)

        def _refresh_windows(self) -> None:
            self.windows = list_visible_windows()
            self.window_list.clear()
            for window in self.windows:
                text = (
                    f"{window.title}  |  0x{window.handle:08X}  |  "
                    f"{window.rect.crop_tuple()}"
                )
                item = QListWidgetItem(_safe_text(text))
                item.setData(Qt.ItemDataRole.UserRole, window.handle)
                self.window_list.addItem(item)
            self._restore_window_selection()
            if self.window_list.count() > 0 and self.window_list.currentRow() < 0:
                self.window_list.setCurrentRow(0)
            self._set_status(f"Janelas visíveis: {len(self.windows)}")

        def _selected_window(self) -> WindowInfo | None:
            item = self.window_list.currentItem()
            if item is None:
                return None
            handle = int(item.data(Qt.ItemDataRole.UserRole))
            for window in self.windows:
                if window.handle == handle:
                    return window
            return None

        def _edit_glossary(self) -> None:
            edit_glossary(
                self,
                self.glossary,
                self.glossary_path,
                self.base_config.cache.path,
            )

        def _build_config(self) -> AppConfig:
            selected = self._selected_window()
            if selected is None:
                raise RuntimeError("Escolha uma janela visível primeiro.")

            config = copy.deepcopy(self.base_config)
            config.capture.window_handle = selected.handle
            config.capture.window_title = None
            config.capture.region = None
            config.capture.fps = int(self.fps_spin.value())
            config.ocr.source_lang = self.source_combo.currentText()
            extra_langs: list[str] = []
            if self.japanese_checkbox.isChecked():
                extra_langs.append("japan")
            if self.korean_checkbox.isChecked():
                extra_langs.append("korean")
            config.ocr.extra_langs = extra_langs
            selected_provider = self.mode_combo.currentText().strip().lower()
            config.translation.provider = (
                "argos" if selected_provider == "argos offline" else selected_provider
            )
            self._remember_active_api_key(self.deepl_key_input.text())
            config.translation.deepl_api_key = self._provider_api_keys.get("deepl", "")
            config.translation.gemini_api_key = self._provider_api_keys.get("gemini", "")
            config.translation.libre_api_key = self._provider_api_keys.get("libre", "")
            return config

        def _start_session(self) -> None:
            if self.process and self.process.poll() is None:
                return
            try:
                config = self._build_config()
                ready, message = self._check_runtime_readiness(config)
                if not ready:
                    raise RuntimeError(message)
                self._write_runtime_files(config)
                self.process = self._spawn_runtime(config)
                self._save_state()
            except Exception as exc:
                self.process = None
                QMessageBox.critical(self, "Falha ao iniciar", str(exc))
                self._set_status(f"Falhou: {exc}")
                return

            selected = self._selected_window()
            target = selected.title if selected else "desconhecida"
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.glossary_button.setEnabled(False)
            self._set_status(
                f"Traduzindo: {target}",
                f"provedor={self.mode_combo.currentText()} destino={config.translation.target_lang}",
            )

        def _stop_session(self) -> None:
            if self.process is None:
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.glossary_button.setEnabled(True)
                return
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
            self.process = None
            self._save_state()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.glossary_button.setEnabled(True)
            self._set_status("Parado")

        def _poll_stats(self) -> None:
            if not self.process:
                return
            stats = self._read_runtime_stats()
            running = self.process.poll() is None
            state = "Em execução" if running else "Parou inesperadamente"
            details = (
                f"frames={stats.get('frames_seen', 0)} "
                f"roi_frames={stats.get('frames_with_rois', 0)} "
                f"ocr={stats.get('ocr_regions', 0)} "
                f"cache={stats.get('cache_hits', 0)} "
                f"api={stats.get('translate_calls', 0)}"
            )
            last_error = str(stats.get("last_error", "")).strip()
            if last_error:
                details = f"{details}\nerror={last_error}"
            self._set_status(state, details)
            if not running:
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                self.glossary_button.setEnabled(True)
                self.process = None

        def _check_readiness(self) -> None:
            try:
                config = self._build_config()
                ready, message = self._check_runtime_readiness(config)
                if ready:
                    self._set_status("Pronto", self._session_summary(config, message))
                else:
                    self._set_status("Não pronto", self._session_summary(config, message))
            except Exception as exc:
                self._set_status("Falha na verificação", str(exc))

        def _sync_mode_ui(self, mode: str) -> None:
            self._remember_active_api_key(self.deepl_key_input.text())
            provider = _provider_key_id(mode)
            if provider != self._active_key_provider:
                self._active_key_provider = provider
                self.deepl_key_input.setText(
                    self._provider_api_keys.get(provider, "")
                )
            self.deepl_key_input.setEnabled(
                provider in {"deepl", "gemini", "libre"}
            )
            placeholders = {
                "deepl": "Chave de API DeepL",
                "gemini": "Chave Gemini AI Studio",
                "libre": "Chave LibreTranslate (opcional)",
            }
            self.deepl_key_input.setPlaceholderText(
                placeholders.get(provider, "Este provedor não usa chave")
            )

        def _remember_active_api_key(self, value: str) -> None:
            if self._active_key_provider in {"deepl", "gemini", "libre"}:
                self._provider_api_keys[self._active_key_provider] = value.strip()

        def _set_status(self, message: str, extra: str | None = None) -> None:
            text = message if not extra else f"{message}\n{extra}"
            self.status_label.setText(_safe_text(text))

        def _session_summary(self, config: AppConfig, detail: str) -> str:
            selected = self._selected_window()
            target = selected.title if selected else "none"
            provider = self.mode_combo.currentText()
            return (
                f"window={target}\n"
                f"provider={provider}\n"
                f"target_lang={config.translation.target_lang}\n"
                f"fps={config.capture.fps}\n"
                f"detail={detail}"
            )

        def _restore_state_to_widgets(self) -> None:
            state = self.saved_state
            provider = str(state.get("provider", "")) or str(state.get("mode", "Google"))
            # Migrate saved states from the old Ollama-based builds.
            if provider.strip().lower() == "ollama":
                provider = "Google"
            provider_index = self.mode_combo.findText(provider)
            if provider_index >= 0:
                self.mode_combo.setCurrentIndex(provider_index)

            stored_keys = state.get("provider_api_keys", {})
            if isinstance(stored_keys, dict):
                for provider_id in ("deepl", "gemini", "libre"):
                    if provider_id in stored_keys:
                        self._provider_api_keys[provider_id] = str(
                            stored_keys[provider_id]
                        )
            if "deepl" not in stored_keys and "deepl_api_key" in state:
                self._provider_api_keys["deepl"] = str(state["deepl_api_key"])
            self._active_key_provider = _provider_key_id(
                self.mode_combo.currentText()
            )
            self.deepl_key_input.setText(
                self._provider_api_keys.get(self._active_key_provider, "")
            )

            source = str(state.get("source_lang", self.base_config.ocr.source_lang))
            source_index = self.source_combo.findText(source)
            if source_index >= 0:
                self.source_combo.setCurrentIndex(source_index)

            fps = int(state.get("fps", self.base_config.capture.fps))
            fps = max(self.fps_spin.minimum(), min(self.fps_spin.maximum(), fps))
            self.fps_spin.setValue(fps)

            self.overlay_checkbox.setChecked(bool(state.get("overlay_enabled", True)))
            self.japanese_checkbox.setChecked(
                bool(state.get("japanese_enabled", self.japanese_checkbox.isChecked()))
            )
            self.korean_checkbox.setChecked(
                bool(state.get("korean_enabled", self.korean_checkbox.isChecked()))
            )

        def _restore_window_selection(self) -> None:
            handle = self.saved_state.get("window_handle")
            if handle is None:
                return
            for row in range(self.window_list.count()):
                item = self.window_list.item(row)
                if int(item.data(Qt.ItemDataRole.UserRole)) == int(handle):
                    self.window_list.setCurrentRow(row)
                    return

        def _save_state(self) -> None:
            selected = self._selected_window()
            self._remember_active_api_key(self.deepl_key_input.text())
            payload = {
                "window_handle": selected.handle if selected else self.saved_state.get("window_handle"),
                "provider": self.mode_combo.currentText(),
                "source_lang": self.source_combo.currentText(),
                "deepl_api_key": self._provider_api_keys.get("deepl", ""),
                "provider_api_keys": self._provider_api_keys,
                "fps": int(self.fps_spin.value()),
                "overlay_enabled": self.overlay_checkbox.isChecked(),
                "japanese_enabled": self.japanese_checkbox.isChecked(),
                "korean_enabled": self.korean_checkbox.isChecked(),
            }
            self.saved_state = payload
            try:
                self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            except OSError:
                return

        def _load_state(self) -> dict[str, object]:
            try:
                if not self.state_path.exists():
                    return {}
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}

        def _check_runtime_readiness(self, config: AppConfig) -> tuple[bool, str]:
            if self.mode_combo.currentText().strip().lower() == "echo":
                return True, "Modo Echo pronto"
            from manga_rt_translator.translator import create_translator

            translator = create_translator(config)
            return translator.check_health()

        def _write_runtime_files(self, config: AppConfig) -> None:
            runtime_config = {
                "capture": {
                    "backend": config.capture.backend,
                    "fps": config.capture.fps,
                    "monitor": config.capture.monitor,
                    "window_handle": config.capture.window_handle,
                },
                "diff": {
                    "tile_size": config.diff.tile_size,
                    "pixel_threshold": config.diff.pixel_threshold,
                    "min_changed_ratio": config.diff.min_changed_ratio,
                    "roi_padding": config.diff.roi_padding,
                    "stable_ms": config.diff.stable_ms,
                    "max_rois_per_tick": config.diff.max_rois_per_tick,
                    "large_change_tile_ratio": config.diff.large_change_tile_ratio,
                    "scroll_cooldown_ms": config.diff.scroll_cooldown_ms,
                },
                "ocr": {
                    "enabled": config.ocr.enabled,
                    "source_lang": config.ocr.source_lang,
                    "extra_langs": config.ocr.extra_langs,
                    "min_confidence": config.ocr.min_confidence,
                    "ignore_top_pixels": config.ocr.ignore_top_pixels,
                },
                "translation": {
                    "target_lang": config.translation.target_lang,
                    "provider": config.translation.provider,
                    "deepl_free": config.translation.deepl_free,
                    "timeout_seconds": config.translation.timeout_seconds,
                    "fuzzy_threshold": config.translation.fuzzy_threshold,
                },
                "overlay": {
                    "enabled": config.overlay.enabled,
                    "font_family": config.overlay.font_family,
                    "font_size": config.overlay.font_size,
                    "opacity": config.overlay.opacity,
                    "ttl_seconds": config.overlay.ttl_seconds,
                },
                "cache": {
                    "path": config.cache.path,
                },
            }
            Path("config.runtime.json").write_text(json.dumps(runtime_config), encoding="utf-8")
            self.stats_path.write_text("{}", encoding="utf-8")

        def _spawn_runtime(self, config: AppConfig) -> subprocess.Popen[str]:
            args = [
                str(Path.cwd() / ".venv311" / "Scripts" / "python.exe"),
                "-u",
                "-m",
                "manga_rt_translator",
                "run",
                "--config",
                "config.toml",
                "--window-handle",
                str(config.capture.window_handle),
                "--source-lang",
                config.ocr.source_lang,
                "--extra-langs",
                ",".join(config.ocr.extra_langs),
                "--fps",
                str(config.capture.fps),
                "--stats-file",
                str(self.stats_path),
            ]
            if self.glossary_path.exists():
                args.extend(["--glossary", str(self.glossary_path)])
            if self.mode_combo.currentText().strip().lower() == "echo":
                args.append("--echo-translator")
            else:
                args.extend(["--provider", config.translation.provider])
                if config.translation.provider == "deepl" and config.translation.deepl_api_key:
                    args.extend(["--deepl-api-key", config.translation.deepl_api_key])
                if config.translation.provider == "gemini" and config.translation.gemini_api_key:
                    args.extend(["--gemini-api-key", config.translation.gemini_api_key])
                if config.translation.provider == "libre" and config.translation.libre_api_key:
                    args.extend(["--libre-api-key", config.translation.libre_api_key])
            if not self.overlay_checkbox.isChecked():
                args.append("--no-overlay")
            return subprocess.Popen(
                args,
                cwd=str(Path.cwd()),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )

        def _read_runtime_stats(self) -> dict[str, object]:
            try:
                if not self.stats_path.exists():
                    return {}
                return json.loads(self.stats_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return {}

    app = QApplication.instance() or QApplication([])
    glossary_path = resolve_glossary_path(glossary_path)
    panel = ControlPanel(load_config(config_path), Glossary.load(glossary_path))
    panel.show()
    return int(app.exec())


def _safe_text(text: str) -> str:
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def _provider_key_id(mode: str) -> str:
    provider = mode.strip().casefold()
    return "argos" if provider == "argos offline" else provider
