from __future__ import annotations

import copy
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from manga_rt_translator.config import AppConfig, load_config
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.models import Frame, OverlayItem, Rect
from manga_rt_translator.overlay import Overlay
from manga_rt_translator.translator import Translator, create_translator
from manga_rt_translator.reader_runtime import ReaderRuntime
from manga_rt_translator.work_queue import ViewState

# Strong module-level reference to the live window. PySide6 only weakly
# references Python wrappers for C++-owned QObjects; losing the wrapper
# mid-session turns every timer/slot invocation into
# "Slot 'ReaderWindow::...' not found" and freezes the overlay.
_ACTIVE_WINDOW: "ReaderWindow | None" = None


@dataclass(slots=True)
class ReaderLaunchContext:
    config: AppConfig
    glossary: Glossary
    initial_url: str


class NullStream:
    def start(self) -> None:
        return None

    def frames(self) -> list[Frame]:
        return []

    def stop(self) -> None:
        return None


def _reader_log(message: str) -> None:
    try:
        path = Path.cwd() / ".reader_debug.log"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")
            handle.flush()
        print(f"[reader] {message}", flush=True)
    except OSError:
        return


def run_reader_app(
    config_path: str | None,
    glossary_path: str | None,
    initial_url: str | None = None,
) -> int:
    if platform.system().lower() != "windows":
        raise RuntimeError("The desktop reader is currently implemented for Windows only.")

    from PySide6.QtCore import QObject, QPoint, QRect, QSize, Qt, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStatusBar,
        QToolBar,
        QVBoxLayout,
        QWidget,
    )

    class ReaderPage(QWebEnginePage):
        def __init__(self, parent: object | None = None) -> None:
            super().__init__(parent)
            self.last_console_error = ""

        def javaScriptConsoleMessage(  # noqa: N802
            self,
            level: QWebEnginePage.JavaScriptConsoleMessageLevel,
            message: str,
            line_number: int,
            source_id: str,
        ) -> None:
            text = message.strip()
            if not text:
                return
            noisy_markers = (
                "SITEWIDE_APRIL_FOOLS",
                "addEventListener",
                "reading 'remove'",
                "Swiper Loop Warning",
                "Failed to fetch",
                "delaybll is not defined",
            )
            if any(marker in text for marker in noisy_markers):
                return
            self.last_console_error = (
                f"js[{level.name}] {text} ({source_id}:{line_number})"
            )

    class ReaderOverlayWidget(QWidget):
        def __init__(self, overlay_config: object) -> None:
            # Top-level frameless window (Gaminik-style): a child widget of
            # the webview can hit PySide6 6.11 lifecycle bugs (paint
            # SystemError / "Slot not found") when Chromium recomposites the
            # page. An owned top-level window is immune to that.
            super().__init__(None)
            self.overlay_config = overlay_config
            self.items: list[OverlayItem] = []
            self.viewport_region: Rect | None = None
            self._last_signature = None
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
                | Qt.WindowType.WindowTransparentForInput
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
            self.show()

        def update_items(self, items: list[OverlayItem]) -> None:
            if isinstance(items, tuple):
                token, items = items
                if not token.current():
                    return
                self._items_token = token
            else:
                self._items_token = None
            signature = tuple(
                (item.bbox.as_tuple(), item.text, item.expires_at) for item in items
            )
            if signature == self._last_signature:
                return  # nothing changed; skip the repaint
            self._last_signature = signature
            self.items = list(items)
            self.update()

        def set_viewport_region(self, region: Rect | None) -> None:
            self.viewport_region = region
            self.update()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
            font = QFont(
                self.overlay_config.font_family,
                self.overlay_config.font_size,
            )
            font.setBold(True)
            painter.setFont(font)

            now = time.time()
            live = [
                item for item in self.items
                if item.expires_at is None or item.expires_at > now
            ]
            token = getattr(self, '_items_token', None)
            if token is not None and not token.current():
                live = []
            configured_font_size = int(self.overlay_config.font_size)
            if len(live) >= 8:
                base_font_size = min(configured_font_size, 10)
            elif len(live) >= 5:
                base_font_size = min(configured_font_size, 12)
            else:
                base_font_size = configured_font_size
            bounds = (
                PaintRect(
                    self.viewport_region.x,
                    self.viewport_region.y,
                    self.viewport_region.width,
                    self.viewport_region.height,
                )
                if self.viewport_region is not None
                else PaintRect(2, 2, max(1, self.width() - 4), max(1, self.height() - 4))
            )
            occupied: list[PaintRect] = []
            layout: list[tuple[OverlayItem, PaintRect, QFont, bool]] = []
            # Place larger source regions first. They usually contain the main
            # dialogue and leave the small captions more freedom to move.
            for item in sorted(live, key=lambda value: value.bbox.area, reverse=True):
                provisional = bool(item.source_text) and item.text == item.source_text
                source_rect = _offset_rect(item.bbox, self.viewport_region)
                chosen: tuple[PaintRect, QFont] | None = None
                # Dense comic pages need smaller type. Try the configured size
                # first, then compact only as much as collision avoidance needs.
                for point_size in range(base_font_size, 8, -1):
                    font = QFont(self.overlay_config.font_family, point_size)
                    font.setBold(True)
                    font.setItalic(provisional)
                    painter.setFont(font)
                    rect = _fit_rect(source_rect, self.width(), self.height(), item.text)
                    rect = _fit_text_rect(
                        rect,
                        item.text,
                        painter,
                        bounds.y + bounds.height,
                        int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignCenter),
                        QRect,
                    )
                    if rect.width > bounds.width or rect.height > bounds.height:
                        continue
                    placed = _place_without_overlap(rect, occupied, bounds)
                    if placed is not None:
                        chosen = (placed, font)
                        break
                if chosen is None:
                    # An exceptionally crowded viewport is still more useful
                    # with one caption omitted than with several unreadable
                    # captions painted on top of each other.
                    continue
                rect, font = chosen
                occupied.append(rect)
                layout.append((item, rect, font, provisional))

            for item, rect, font, provisional in layout:
                painter.setFont(font)
                bubble = QPainterPath()
                bubble.addRoundedRect(rect.x, rect.y, rect.width, rect.height, 10, 10)
                painter.fillPath(bubble.translated(0, 2), QColor(0, 0, 0, 48))
                if provisional:
                    painter.fillPath(
                        bubble,
                        QColor(250, 250, 250, int(255 * self.overlay_config.opacity * 0.75)),
                    )
                    painter.setPen(QPen(QColor(120, 120, 120, 110), 1.0))
                    painter.drawPath(bubble)
                    painter.setPen(QColor(96, 96, 96, 235))
                else:
                    painter.fillPath(
                        bubble,
                        QColor(255, 252, 245, int(255 * self.overlay_config.opacity)),
                    )
                    painter.setPen(QPen(QColor(36, 31, 28, 110), 1.2))
                    painter.drawPath(bubble)
                    painter.setPen(QColor(24, 20, 18, 245))
                painter.drawText(
                    rect.x + 10,
                    rect.y + 8,
                    max(1, rect.width - 20),
                    max(1, rect.height - 16),
                    Qt.AlignmentFlag.AlignCenter
                    | Qt.AlignmentFlag.AlignVCenter
                    | Qt.TextFlag.TextWordWrap,
                    item.text,
                )

            if self.viewport_region is not None:
                painter.setPen(QPen(QColor(68, 186, 255, 190), 1.5, Qt.PenStyle.DashLine))
                painter.drawRect(
                    self.viewport_region.x,
                    self.viewport_region.y,
                    self.viewport_region.width,
                    self.viewport_region.height,
                )

    class RegionSelectionOverlay(QWidget):
        region_selected = Signal(object)

        def __init__(self, parent: QWidget) -> None:
            super().__init__(parent)
            self.start_point: QPoint | None = None
            self.current_point: QPoint | None = None
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
            self.hide()

        def begin(self, rect: QRect) -> None:
            self.setGeometry(rect)
            self.start_point = None
            self.current_point = None
            self.show()
            self.raise_()
            self.setCursor(Qt.CursorShape.CrossCursor)

        def mousePressEvent(self, event: object) -> None:  # noqa: N802
            if hasattr(event, "position"):
                pos = event.position().toPoint()
            else:
                pos = event.pos()
            self.start_point = pos
            self.current_point = pos
            self.update()

        def mouseMoveEvent(self, event: object) -> None:  # noqa: N802
            if self.start_point is None:
                return
            if hasattr(event, "position"):
                self.current_point = event.position().toPoint()
            else:
                self.current_point = event.pos()
            self.update()

        def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802
            if self.start_point is None:
                self._finish(None)
                return
            if hasattr(event, "position"):
                end = event.position().toPoint()
            else:
                end = event.pos()
            rect = QRect(self.start_point, end).normalized()
            region = None
            if rect.width() >= 24 and rect.height() >= 24:
                region = Rect(rect.x(), rect.y(), rect.width(), rect.height())
            self._finish(region)

        def keyPressEvent(self, event: object) -> None:  # noqa: N802
            if hasattr(event, "key") and event.key() == Qt.Key.Key_Escape:
                self._finish(None)
                return
            return super().keyPressEvent(event)

        def _finish(self, region: Rect | None) -> None:
            self.unsetCursor()
            self.hide()
            self.start_point = None
            self.current_point = None
            self.update()
            self.region_selected.emit(region)

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 70))
            if self.start_point is None or self.current_point is None:
                return
            rect = QRect(self.start_point, self.current_point).normalized()
            painter.fillRect(rect, QColor(255, 255, 255, 18))
            painter.setPen(QPen(QColor(68, 186, 255, 220), 2))
            painter.drawRect(rect)

    class ReaderOverlayBridge(QObject):
        items_changed = Signal(object)
        status_changed = Signal(str)

    class ReaderOverlay(Overlay):
        def __init__(self, widget: ReaderOverlayWidget, bridge: ReaderOverlayBridge) -> None:
            self.widget = widget
            self.bridge = bridge
            self.view = ViewState()

        def update(self, items: list[OverlayItem]) -> None:
            # Qt may deliver a queued signal after another scroll has started.
            # The widget checks the token again at delivery and paint time.
            self.bridge.items_changed.emit((self.view.token(), list(items)))

        def run_forever(self) -> int:
            return 0

        def stop(self) -> None:
            self.bridge.items_changed.emit([])

    class ReaderWindow(QMainWindow):
        def __init__(self, launch: ReaderLaunchContext) -> None:
            super().__init__()
            self.launch = launch
            self.runtime: ReaderRuntime | None = None
            self.selected_region: Rect | None = None
            self.capture_timer = QTimer(self)
            self.capture_timer.timeout.connect(self._capture_reader_frame)
            self.scroll_poll_timer = QTimer(self)
            self.scroll_poll_timer.timeout.connect(self._poll_scroll)
            self.scroll_settle_timer = QTimer(self)
            self.scroll_settle_timer.setSingleShot(True)
            self.scroll_settle_timer.timeout.connect(self._capture_settled_scroll)
            self._last_poll_y: float | None = None
            self._settle_count = 0
            self._captured_y: float | None = None
            self._last_scroll_move_at = 0.0
            self._last_settle_capture = 0.0
            self._scroll_velocity_slow = True
            self.sweep_timer = QTimer(self)
            self.sweep_timer.timeout.connect(self._gui_sweep)
            self.refresh_timer = QTimer(self)
            self.refresh_timer.timeout.connect(self._translate_current_region)
            self.status_timer = QTimer(self)
            self.status_timer.timeout.connect(self._refresh_runtime_status)
            self.status_timer.start(800)
            self.bridge = ReaderOverlayBridge()
            self.setWindowTitle("Manga Real-Time Translator Reader")
            self.resize(1440, 960)
            self._build_ui()
            self._connect_events()
            # Chromium raises its render widget whenever it repaints; keep the
            # overlays on top so translations never disappear behind the page.
            self.overlay_raise_timer = QTimer(self)
            self.overlay_raise_timer.timeout.connect(self._sync_overlay_geometry)
            self.overlay_raise_timer.start(750)
            self._load_initial_page()

        def _build_ui(self) -> None:
            self.webview = QWebEngineView(self)
            self.page = ReaderPage(self.webview)
            self.webview.setPage(self.page)
            self.webview.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)

            root = QWidget(self)
            self.setCentralWidget(root)
            layout = QVBoxLayout(root)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)

            self.toolbar = QToolBar("Navigation", self)
            self.toolbar.setMovable(False)
            self.addToolBar(self.toolbar)

            back_action = QAction("Voltar", self)
            back_action.triggered.connect(self.webview.back)
            self.toolbar.addAction(back_action)

            forward_action = QAction("Avançar", self)
            forward_action.triggered.connect(self.webview.forward)
            self.toolbar.addAction(forward_action)

            reload_action = QAction("Recarregar", self)
            reload_action.triggered.connect(self.webview.reload)
            self.toolbar.addAction(reload_action)

            self.url_input = QLineEdit(self.launch.initial_url)
            self.url_input.returnPressed.connect(self._open_url)
            self.toolbar.addWidget(self.url_input)

            open_button = QPushButton("Abrir")
            open_button.clicked.connect(self._open_url)
            self.toolbar.addWidget(open_button)

            controls = QWidget(self)
            controls_layout = QHBoxLayout(controls)
            controls_layout.setContentsMargins(12, 8, 12, 8)
            controls_layout.setSpacing(10)

            controls_layout.addWidget(QLabel("Origem"))
            self.source_combo = QComboBox()
            self.source_combo.addItems(["Auto", "en", "japan", "korean"])
            default_source = self.launch.config.ocr.source_lang
            self.source_combo.setCurrentText(
                default_source
                if default_source in {"en", "japan", "korean"}
                else "Auto"
            )
            controls_layout.addWidget(self.source_combo)

            controls_layout.addWidget(QLabel("Modo"))
            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Traduzir PT-BR", "Somente OCR (rápido)"])
            self.mode_combo.setCurrentText("Traduzir PT-BR")
            controls_layout.addWidget(self.mode_combo)

            controls_layout.addWidget(QLabel("Provedor"))
            self.provider_combo = QComboBox()
            self.provider_combo.addItems(["Google", "DeepL", "Gemini", "Libre", "Echo"])
            saved_provider = str(self.launch.config.translation.provider or "google").strip().lower()
            provider_label = {"google": "Google", "deepl": "DeepL", "gemini": "Gemini", "libre": "Libre", "echo": "Echo"}.get(
                saved_provider, "Google"
            )
            provider_index = self.provider_combo.findText(provider_label)
            if provider_index >= 0:
                self.provider_combo.setCurrentIndex(provider_index)
            controls_layout.addWidget(self.provider_combo)

            controls_layout.addWidget(QLabel("Chave de API"))
            # Per-provider key memory: the field shows the active provider's
            # key and never clobbers the keys of the others (nor the values
            # loaded from config.toml).
            self._api_keys = {
                "deepl": self.launch.config.translation.deepl_api_key,
                "gemini": self.launch.config.translation.gemini_api_key,
                "libre": self.launch.config.translation.libre_api_key,
            }
            self.deepl_key_input = QLineEdit(self._api_keys.get(saved_provider, ""))
            self.deepl_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.deepl_key_input.setPlaceholderText("DeepL, Gemini ou Libre")
            self.deepl_key_input.setMaximumWidth(200)
            self.deepl_key_input.textChanged.connect(self._store_api_key)
            self.provider_combo.currentTextChanged.connect(self._sync_provider_ui)
            controls_layout.addWidget(self.deepl_key_input)

            controls_layout.addWidget(QLabel("Análise"))
            self.analysis_combo = QComboBox()
            self.analysis_combo.addItems(["Manual", "Contínuo"])
            self.analysis_combo.setCurrentText("Contínuo")
            controls_layout.addWidget(self.analysis_combo)

            controls_layout.addWidget(QLabel("FPS"))
            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(1, 12)
            self.fps_spin.setValue(8)
            self.fps_spin.setToolTip("Sensibilidade ao rolar: maior = reage mais rápido ao parar o scroll")
            controls_layout.addWidget(self.fps_spin)

            self.overlay_checkbox = QCheckBox("Overlay")
            self.overlay_checkbox.setChecked(True)
            controls_layout.addWidget(self.overlay_checkbox)

            self.start_button = QPushButton("Preparar")
            controls_layout.addWidget(self.start_button)

            self.analyze_button = QPushButton("Traduzir região")
            controls_layout.addWidget(self.analyze_button)

            self.inspect_button = QPushButton("Inspecionar OCR")
            controls_layout.addWidget(self.inspect_button)

            self.select_region_button = QPushButton("Selecionar região")
            controls_layout.addWidget(self.select_region_button)

            self.clear_region_button = QPushButton("Limpar região")
            self.clear_region_button.setEnabled(False)
            controls_layout.addWidget(self.clear_region_button)

            self.auto_refresh_checkbox = QCheckBox("Auto atualizar")
            controls_layout.addWidget(self.auto_refresh_checkbox)

            controls_layout.addWidget(QLabel("A cada (s)"))
            self.refresh_spin = QSpinBox()
            self.refresh_spin.setRange(1, 10)
            self.refresh_spin.setValue(2)
            controls_layout.addWidget(self.refresh_spin)

            self.stop_button = QPushButton("Parar")
            controls_layout.addWidget(self.stop_button)
            controls_layout.addStretch(1)

            layout.addWidget(controls)
            layout.addWidget(self.webview, stretch=1)

            self.overlay_widget = ReaderOverlayWidget(self.launch.config.overlay)
            self.overlay_widget.setGeometry(self.webview.geometry())
            self.selection_overlay = RegionSelectionOverlay(root)
            self.selection_overlay.setGeometry(self.webview.geometry())
            self.selection_overlay.region_selected.connect(self._handle_region_selected)

            self.status = QStatusBar(self)
            self.setStatusBar(self.status)
            self.status.showMessage(
                "Leitor pronto. Selecione uma região para traduzir automaticamente, ou use Traduzir região (Ctrl+Shift+T)."
            )

        def resizeEvent(self, event: object) -> None:  # noqa: N802
            super().resizeEvent(event)
            self._sync_overlay_geometry()

        def _sync_overlay_geometry(self) -> None:
            # The translation overlay is a top-level transparent window
            # pinned over the webview (immune to Chromium recompositing).
            # It follows the reader's visibility and focus: hidden when the
            # reader is minimized, hidden or covered by another app.
            global_top_left = self.webview.mapToGlobal(QPoint(0, 0))
            self.overlay_widget.setGeometry(
                QRect(global_top_left, self.webview.size())
            )
            focused = QApplication.activeWindow() is self
            self.overlay_widget.setVisible(
                focused and self.webview.isVisible() and not self.isMinimized()
                and self.overlay_checkbox.isChecked()
            )
            self.selection_overlay.setGeometry(self.webview.geometry())
            if self.selection_overlay.isVisible():
                self.selection_overlay.raise_()

        def _connect_events(self) -> None:
            self.bridge.items_changed.connect(self.overlay_widget.update_items)
            self.webview.urlChanged.connect(self._sync_url)
            self.webview.loadStarted.connect(self._handle_load_started)
            self.webview.loadFinished.connect(self._handle_load_finished)
            self.start_button.clicked.connect(self._handle_start_clicked)
            self.analyze_button.clicked.connect(self._handle_analyze_clicked)
            self.inspect_button.clicked.connect(self._handle_inspect_clicked)
            self.select_region_button.clicked.connect(self._begin_region_selection)
            self.clear_region_button.clicked.connect(self._clear_region_selection)
            self.stop_button.clicked.connect(self._handle_stop_clicked)
            self.auto_refresh_checkbox.toggled.connect(self._sync_refresh_timer)
            self.refresh_spin.valueChanged.connect(self._sync_refresh_timer)
            self.analysis_combo.currentTextChanged.connect(lambda _: self._apply_capture_schedule())
            self.fps_spin.valueChanged.connect(lambda _: self._apply_capture_schedule())
            self.bridge.status_changed.connect(self.status.showMessage)
            self.overlay_checkbox.toggled.connect(self._sync_overlay_geometry)
            QShortcut(QKeySequence("Ctrl+Shift+T"), self, activated=self._translate_current_region)

        def _load_initial_page(self) -> None:
            self._open_url()

        def _open_url(self) -> None:
            raw = self.url_input.text().strip()
            if not raw:
                return
            if "://" not in raw:
                raw = f"https://{raw}"
                self.url_input.setText(raw)
            self.webview.setUrl(QUrl(raw))

        def _sync_url(self, url: QUrl) -> None:
            self.url_input.setText(url.toString())

        def _handle_load_started(self) -> None:
            self.scroll_settle_timer.stop()
            self._last_poll_y = None
            self._captured_y = None
            if self.runtime:
                self.runtime.invalidate_view()
            self.status.showMessage("Carregando página...")

        def _handle_load_finished(self, ok: bool) -> None:
            self._sync_overlay_geometry()
            if ok:
                message = "Página carregada. Selecione a região de leitura para começar."
                if self.page.last_console_error:
                    message = f"{message} Aviso do site: {self.page.last_console_error}"
                self.status.showMessage(message)
            else:
                self.status.showMessage("Falha ao carregar a página.")
            if ok and self.runtime and self.selected_region is not None:
                QTimer.singleShot(350, self._translate_current_region)

        def _store_api_key(self, text: str) -> None:
            provider = self.provider_combo.currentText().strip().lower()
            if provider in self._api_keys:
                self._api_keys[provider] = text.strip()

        def _sync_provider_ui(self, provider_text: str) -> None:
            provider = provider_text.strip().lower()
            if provider in self._api_keys:
                self.deepl_key_input.setText(self._api_keys[provider])
            needs_key = provider in {"deepl", "gemini", "libre"}
            self.deepl_key_input.setEnabled(needs_key)

        def _handle_start_clicked(self) -> None:
            _reader_log("UI: Start Translation clicked")
            self._ensure_runtime_started()

        def _ensure_runtime_started(self) -> bool:
            if self.runtime and self.runtime.worker and self.runtime.worker.is_alive():
                return True
            try:
                config = copy.deepcopy(self.launch.config)
                selected_source = self.source_combo.currentText()
                config.ocr.source_lang = "en" if selected_source == "Auto" else selected_source
                if selected_source == "Auto":
                    # Auto keeps English as primary and falls back to the
                    # dedicated Japanese and Korean models when the score or
                    # script says the page is not Latin text.
                    config.ocr.extra_langs = ["japan", "korean"]
                else:
                    config.ocr.extra_langs = []
                config.capture.fps = int(self.fps_spin.value())
                provider = self.provider_combo.currentText().strip().lower()
                config.translation.provider = provider
                # Keys come from the per-provider memory (seeded from config
                # and updated as the user edits the field).
                config.translation.deepl_api_key = self._api_keys.get("deepl", "")
                config.translation.gemini_api_key = self._api_keys.get("gemini", "")
                config.translation.libre_api_key = self._api_keys.get("libre", "")
                overlay = ReaderOverlay(self.overlay_widget, self.bridge)
                runtime = ReaderRuntime(
                    config=config,
                    glossary=self.launch.glossary,
                    overlay=overlay,
                    status_callback=self._push_status,
                    echo_translator=(self.mode_combo.currentText() == "Somente OCR (rápido)"),
                )
                runtime.set_analysis_mode(self.analysis_combo.currentText())
                runtime.start()
                self.runtime = runtime
            except Exception as exc:
                QMessageBox.critical(self, "Falha ao iniciar", str(exc))
                return False
            self.capture_timer.stop()
            self.refresh_timer.stop()
            self.start_button.setEnabled(False)
            if self.mode_combo.currentText() == "Somente OCR (rápido)":
                self.status.showMessage(
                    "Somente OCR ativo. O texto permanece no idioma original."
                )
            else:
                provider_label = self.provider_combo.currentText()
                self.status.showMessage(
                    f"Traduzindo com {provider_label}. "
                    "Selecione a região de leitura para tradução automática."
                )
            self._apply_capture_schedule()
            return True

        def _handle_analyze_clicked(self) -> None:
            _reader_log("UI: Analyze View clicked")
            self._translate_current_region()

        def _translate_current_region(self) -> None:
            if not self._ensure_runtime_started():
                return
            if self.runtime is None:
                return
            rgb = self._grab_reader_rgb()
            if rgb is None:
                self.status.showMessage("Não foi possível capturar a região atual.")
                return
            self.runtime.push_frame(rgb, force_analyze=True)
            self.status.showMessage("Traduzindo a região atual...")

        def _handle_inspect_clicked(self) -> None:
            _reader_log("UI: Inspect OCR clicked")
            if not self._ensure_runtime_started():
                return
            if self.runtime is None:
                _reader_log("UI: Inspect OCR failed - runtime is None")
                return
            rgb = self._grab_reader_rgb()
            if rgb is not None:
                self.runtime.push_frame(rgb, inspect=True)

        def closeEvent(self, event: object) -> None:  # noqa: N802
            self._handle_stop_clicked()
            self.overlay_raise_timer.stop()
            self.status_timer.stop()
            self.overlay_widget.close()
            self.selection_overlay.close()
            super().closeEvent(event)

        def _handle_stop_clicked(self) -> None:
            self.capture_timer.stop()
            self.scroll_poll_timer.stop()
            self.scroll_settle_timer.stop()
            self.sweep_timer.stop()
            self.refresh_timer.stop()
            if self.runtime is not None:
                self.runtime.stop()
                self.runtime = None
            self.overlay_widget.update_items([])
            self.start_button.setEnabled(True)
            self.status.showMessage("Tradução parada.")

        def _begin_region_selection(self) -> None:
            self.selection_overlay.begin(self.webview.geometry())
            self.status.showMessage("Arraste sobre a área de leitura da página.")

        def _handle_region_selected(self, region: Rect | None) -> None:
            if self.runtime:
                self.runtime.invalidate_view()
            self._last_poll_y = None
            self._captured_y = None
            self.scroll_settle_timer.stop()
            self.selected_region = region
            self.overlay_widget.set_viewport_region(region)
            self.clear_region_button.setEnabled(region is not None)
            if region is None:
                self.status.showMessage(
                    "Região removida; tradução automática parada."
                )
                self.analysis_combo.setCurrentText("Manual")
                self._apply_capture_schedule()
                self._sync_refresh_timer()
                return
            # Selecting a region is the whole setup: from now on everything
            # that scrolls into the region is captured, OCR'd, and translated
            # automatically.
            self.status.showMessage(
                "Região ativa: tudo que aparecer nela será traduzido "
                "automaticamente. Só role e leia."
            )
            self.analysis_combo.setCurrentText("Contínuo")
            self._ensure_runtime_started()
            self._apply_capture_schedule()
            self._sync_refresh_timer()
            # The first selected view is a full-region change. FrameDiffer
            # intentionally suppresses scroll-like changes, so explicitly
            # analyze this stable view once instead of waiting for a second
            # click from the user.
            QTimer.singleShot(
                0,
                lambda: self._capture_reader_frame(force_analyze=True),
            )

        def _clear_region_selection(self) -> None:
            self._handle_region_selected(None)

        def _apply_capture_schedule(self) -> None:
            mode = self.analysis_combo.currentText()
            if self.runtime is not None:
                self.runtime.set_analysis_mode(mode)
            if mode == "Contínuo":
                # Polling the scroll position is an async JavaScript call —
                # cheap and does not block the UI. The expensive full capture
                # runs when scrolling settles (watcher) or on the sweep timer,
                # which also covers slow reading scrolls.
                poll_ms = max(80, int(1000 / max(1, self.fps_spin.value())))
                self.scroll_poll_timer.start(poll_ms)
                # One capture coordinator is enough. The worker processes the
                # fresh frame with FrameDiffer; running another full-frame
                # sweep there caused duplicate OCR and overlay churn.
                self.sweep_timer.start(750)
                self.capture_timer.stop()
            else:
                self.scroll_poll_timer.stop()
                self.scroll_settle_timer.stop()
                self.sweep_timer.stop()
                self.capture_timer.stop()
                self._settle_count = 0

        def _gui_sweep(self) -> None:
            """Capture a fresh frame on the continuous-analysis cadence.
            balloons translate during slow reading scrolls — it grabs a NEW
            frame instead of re-analyzing the last one, and it is suppressed
            during fast scrolling (mid-scroll frames have cut-off balloons)."""
            if self.runtime is None or self.analysis_combo.currentText() != "Contínuo":
                return
            if self.isMinimized() or not self.webview.isVisible():
                return
            if self.selected_region is not None:
                # Fixed-region mode is event driven: one capture on selection
                # and one after each settled scroll. Periodic sweeps would
                # waste OCR and risk sampling our own translation overlay.
                return
            now = time.time()
            if now - self._last_settle_capture < 1.0:
                return  # the settle watcher just ran a fresh analysis
            if not self._scroll_velocity_slow and now - self._last_scroll_move_at < 0.8:
                return  # fast scrolling: mid-scroll frames are garbage
            self._last_settle_capture = now
            self._capture_reader_frame()

        def _poll_scroll(self) -> None:
            if self.runtime is None or self.analysis_combo.currentText() != "Contínuo":
                return
            if self.isMinimized() or not self.webview.isVisible():
                return
            self._poll_calls = getattr(self, "_poll_calls", 0) + 1
            if self._poll_calls % 40 == 1:
                _reader_log(f"scroll.poll running (calls={self._poll_calls})")
            try:
                self.webview.page().runJavaScript(
                    "window.scrollY || 0", 0, self._on_scroll_poll
                )
            except RuntimeError:
                return

        def _on_scroll_poll(self, value: object) -> None:
            self._poll_results = getattr(self, "_poll_results", 0) + 1
            if self._poll_results % 40 == 1:
                _reader_log(f"scroll.poll callback ok (results={self._poll_results})")
            if self.analysis_combo.currentText() != "Contínuo":
                return
            try:
                y = float(value or 0)
            except (TypeError, ValueError):
                return
            if self._last_poll_y is None:
                self._last_poll_y = y
                self._captured_y = y
                return
            delta = abs(y - self._last_poll_y)
            # Slow drift (~reading speed) is translate-worthy; fast scrolling
            # produces frames with cut-off balloons.
            self._scroll_velocity_slow = delta <= 25.0
            if delta > 0.5:
                self._last_poll_y = y
                # A trip away and back still invalidates the previous capture.
                self._captured_y = None
                self._settle_count = 0
                self._last_scroll_move_at = time.time()
                # Smooth scrolling can pause briefly at several intermediate
                # offsets. Restarting this one-shot timer guarantees one OCR
                # pass only after the final position has remained still.
                settle_ms = max(300, int(self.launch.config.diff.scroll_settle_ms))
                self.scroll_settle_timer.start(settle_ms)
                # Scrolling invalidates every viewport-anchored box on the
                # spot — during scroll no frames are processed, so without
                # this the old translations would linger over moved content.
                if self.runtime:
                    self.runtime.invalidate_view()
                self.overlay_widget.update()
                return

        def _capture_settled_scroll(self) -> None:
            if self.runtime is None or self.analysis_combo.currentText() != "Contínuo":
                return
            y = self._last_poll_y
            if y is None or self._captured_y == y:
                return
            self._last_settle_capture = time.time()
            _reader_log(f"scroll.settled y={y:.0f}; forcing region analysis")
            self.status.showMessage(
                "Novo conteúdo detectado; lendo e traduzindo..."
            )
            if self._capture_reader_frame(force_analyze=True):
                self._captured_y = y
            else:
                self.scroll_settle_timer.start(350)

        def _sync_refresh_timer(self) -> None:
            if not self.auto_refresh_checkbox.isChecked() or self.selected_region is None:
                self.refresh_timer.stop()
                return
            self.refresh_timer.start(int(self.refresh_spin.value()) * 1000)

        def _dxcam_camera(self) -> object | None:
            """Desktop Duplication camera (GPU frame capture, ~10x faster
            than widget rendering). Only used while the reader is the
            foreground window on the primary screen — dxcam captures the
            physical screen, so anything on top would leak into the frame."""
            cam = getattr(self, "_dxcam_instance", "uninitialized")
            if cam != "uninitialized":
                return cam
            self._dxcam_instance = None
            try:
                import dxcam

                screen = self.windowHandle().screen()
                primary = QApplication.primaryScreen()
                if screen is not primary:
                    _reader_log("dxcam.skip non-primary screen")
                    return None
                self._dxcam_instance = dxcam.create(output_color="RGB")
            except Exception as exc:
                _reader_log(f"dxcam.unavailable {type(exc).__name__}: {exc}")
            return self._dxcam_instance

        def _grab_dxcam(self) -> tuple[str, np.ndarray | None]:
            """Returns (status, frame): ok / nochange / unavailable."""
            cam = self._dxcam_camera()
            if cam is None:
                return "unavailable", None
            try:
                # Only capture while the reader is the foreground window:
                # dxcam reads the physical screen, so an overlapping window
                # would leak into the frame.
                import ctypes

                hwnd = int(self.winId())
                if ctypes.windll.user32.GetForegroundWindow() != hwnd:
                    return "unavailable", None
                screen = self.windowHandle().screen()
                if screen is None:
                    return "unavailable", None
                origin = self.webview.mapToGlobal(QPoint(0, 0))
                dpr = self.devicePixelRatioF() or 1.0
                left = int(origin.x() * dpr)
                top = int(origin.y() * dpr)
                width = int(self.webview.width() * dpr)
                height = int(self.webview.height() * dpr)
                x, y, w, h = left, top, width, height
                if self.selected_region is not None:
                    region = self.selected_region
                    x = left + int(region.x * dpr)
                    y = top + int(region.y * dpr)
                    w = int(region.width * dpr)
                    h = int(region.height * dpr)
                # dxcam validates against the physical output size; if the
                # window drifted off it (second monitor, negative coords),
                # fall back to the widget grab instead of erroring per tick.
                screen_geo = screen.geometry()
                out_w = int(screen_geo.width() * dpr)
                out_h = int(screen_geo.height() * dpr)
                if x < 0 or y < 0 or x + w > out_w or y + h > out_h:
                    return "unavailable", None
                if w <= 0 or h <= 0:
                    return "unavailable", None
                frame = cam.grab(region=(x, y, x + w, y + h))
                if frame is None:
                    return "nochange", None  # screen unchanged since last grab
                if frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                return "ok", np.ascontiguousarray(frame)
            except Exception as exc:
                now = time.time()
                if now - getattr(self, "_last_dxcam_err", 0.0) > 30:
                    self._last_dxcam_err = now
                    _reader_log(f"dxcam.error {type(exc).__name__}: {exc}")
                return "unavailable", None

        def _grab_reader_rgb(self) -> np.ndarray | None:
            if not self.webview.isVisible():
                return None
            # Fast path for full-view capture: Desktop Duplication frame of
            # the on-screen area. A selected-region session uses the clean
            # Chromium surface below instead. DXCam sees our top-level
            # translation bubbles and would feed Portuguese text back into
            # OCR, progressively contaminating both detection and cache.
            if self.selected_region is None:
                status, frame = self._grab_dxcam()
                if status == "ok":
                    return frame
                if status == "nochange":
                    # Screen untouched since the last grab: genuinely nothing
                    # new to analyze; skip the widget render entirely.
                    return None
            # dxcam unavailable: fall back to rendering the widget. With a
            # reading region selected, render only that sub-rect.
            grab_rect = QRect()
            if self.selected_region is not None:
                region = self.selected_region
                grab_rect = QRect(
                    max(0, region.x),
                    max(0, region.y),
                    min(region.width, max(1, self.webview.width() - max(0, region.x))),
                    min(region.height, max(1, self.webview.height() - max(0, region.y))),
                )
            # The GPU compositor occasionally returns an empty pixmap right
            # after page events; retry briefly before giving up. Note: the
            # no-argument grab() is the historically reliable path — only
            # pass a rect when a region is actually selected.
            overlay_was_visible = (
                self.selected_region is not None and self.overlay_widget.isVisible()
            )
            if overlay_was_visible:
                self.overlay_widget.hide()
            try:
                # Failed captures are retried by the settle timer, never by
                # sleeping on the GUI thread.
                pixmap = self.webview.grab() if grab_rect.isNull() else self.webview.grab(grab_rect)
                if pixmap is None or pixmap.isNull():
                    _reader_log("capture.pixmap_null")
                    return None
                image = pixmap.toImage().convertToFormat(pixmap.toImage().Format.Format_RGBA8888)
                width = image.width()
                height = image.height()
                if width <= 0 or height <= 0:
                    _reader_log(f"capture.empty size={width}x{height}")
                    return None
                ptr = image.bits()
                byte_count = image.sizeInBytes()
                try:
                    ptr.setsize(byte_count)
                except AttributeError:
                    pass
                buffer = bytes(ptr[:byte_count])
                frame = np.frombuffer(buffer, dtype=np.uint8).reshape((height, width, 4))
                rgb = np.ascontiguousarray(frame[:, :, :3])
                logical = self.webview.size() if grab_rect.isNull() else grab_rect.size()
                if (width, height) != (logical.width(), logical.height()):
                    rgb = cv2.resize(rgb, (logical.width(), logical.height()),
                                     interpolation=cv2.INTER_AREA)
                # When a region is selected, grab_rect already rendered
                # exactly the region content; cropping again would shift OCR
                # coordinates and paint translations in the wrong place.
                return rgb
            finally:
                if overlay_was_visible:
                    self.overlay_widget.show()
                    self.overlay_widget.raise_()

        def _capture_reader_frame(self, force_analyze: bool = False) -> bool:
            if self.runtime is None:
                return False
            # A minimized window renders stale or empty content; grabbing it
            # wastes CPU and feeds garbage frames to the pipeline.
            if self.isMinimized():
                return False
            # While the user is actively scrolling the content is unstable
            # and the synchronous grab stutters the UI — the scroll watcher
            # captures as soon as scrolling settles anyway.
            if (
                not force_analyze
                and self.analysis_combo.currentText() == "Contínuo"
                and time.time() - self._last_scroll_move_at < 1.0
            ):
                return False
            rgb = self._grab_reader_rgb()
            if rgb is None:
                return False
            if self.runtime.frames_captured == 0:
                sample = int(rgb.mean()) if rgb.size else -1
                _reader_log(
                    f"capture.first_frame size={rgb.shape[1]}x{rgb.shape[0]} mean={sample}"
                )
            self.runtime.push_frame(rgb, force_analyze=force_analyze)
            self._sync_overlay_geometry()
            return True

        def _refresh_runtime_status(self) -> None:
            if self.runtime is None:
                return
            if self.runtime.last_error:
                error = self.runtime.last_error
                self._handle_stop_clicked()
                self.status.showMessage(error)
                return
            self.status.showMessage(self.runtime.stats_text())

        def _push_status(self, text: str) -> None:
            self.bridge.status_changed.emit(text)

    def _create_app() -> QApplication:
        return QApplication.instance() or QApplication([])

    app = _create_app()
    launch = ReaderLaunchContext(
        config=load_config(config_path),
        glossary=Glossary.load(glossary_path),
        initial_url=initial_url or "https://mangadex.org",
    )
    from manga_rt_translator.ocr import warmup_ocr

    warmup_ocr(launch.config.ocr)
    global _ACTIVE_WINDOW
    window = ReaderWindow(launch)
    _ACTIVE_WINDOW = window
    window.show()
    return int(app.exec())


@dataclass
class PaintRect:
    x: int
    y: int
    width: int
    height: int


def _fit_rect(rect: object, max_width: int, max_height: int, text: str = "") -> PaintRect:
    raw_width = max(getattr(rect, "width", 110), 110)
    raw_height = max(getattr(rect, "height", 52), 52)
    # Hug the translated text block (plus a little padding) so the bubble
    # covers the original text completely, instead of a fixed-size card.
    # Portuguese often needs more horizontal space than the OCR source box.
    # Expanding width prevents the very tall one-word columns seen on dense pages.
    text_width = min(320, max(110, int(max(1, len(text)) ** 0.5 * 42)))
    width = min(max(raw_width + 16, text_width), max_width - 4)
    # OCR boxes may span a tall stack of source lines. The translated caption
    # should follow its own wrapped text height instead of inheriting that
    # entire column and blocking neighboring balloons.
    height = min(max(52, min(raw_height + 20, 140)), max_height - 4)
    source_x = getattr(rect, "x", 0)
    x = min(
        max(0, source_x + (raw_width - width) // 2),
        max(0, max_width - width),
    )
    y = min(max(0, getattr(rect, "y", 0) - 10), max(0, max_height - height))
    return PaintRect(x=x, y=y, width=width, height=height)


def _rects_overlap(left: PaintRect, right: PaintRect, gap: int = 5) -> bool:
    return (
        left.x < right.x + right.width + gap
        and left.x + left.width + gap > right.x
        and left.y < right.y + right.height + gap
        and left.y + left.height + gap > right.y
    )


def _clamp_paint_rect(rect: PaintRect, bounds: PaintRect) -> PaintRect:
    width = min(rect.width, bounds.width)
    height = min(rect.height, bounds.height)
    return PaintRect(
        x=min(max(bounds.x, rect.x), bounds.x + bounds.width - width),
        y=min(max(bounds.y, rect.y), bounds.y + bounds.height - height),
        width=width,
        height=height,
    )


def _place_without_overlap(
    desired: PaintRect,
    occupied: list[PaintRect],
    bounds: PaintRect,
    gap: int = 5,
) -> PaintRect | None:
    """Place close to the OCR anchor using free edges of existing captions."""
    anchor = _clamp_paint_rect(desired, bounds)
    if not any(_rects_overlap(anchor, other, gap) for other in occupied):
        return anchor

    xs = {anchor.x, bounds.x, bounds.x + bounds.width - anchor.width}
    ys = {anchor.y, bounds.y, bounds.y + bounds.height - anchor.height}
    for other in occupied:
        xs.update((other.x - anchor.width - gap, other.x + other.width + gap))
        ys.update((other.y - anchor.height - gap, other.y + other.height + gap))
    candidates = {
        (max(bounds.x, min(x, bounds.x + bounds.width - anchor.width)),
         max(bounds.y, min(y, bounds.y + bounds.height - anchor.height)))
        for x in xs for y in ys
    }
    candidates.discard((anchor.x, anchor.y))
    for x, y in sorted(
        candidates,
        key=lambda point: (point[0] - anchor.x) ** 2 + (point[1] - anchor.y) ** 2,
    ):
        candidate = PaintRect(x, y, anchor.width, anchor.height)
        if not any(_rects_overlap(candidate, other, gap) for other in occupied):
            return candidate
    return None


def _fit_text_rect(
    rect: PaintRect,
    text: str,
    painter: object,
    max_height: int,
    text_flags: int,
    rect_type: object,
) -> PaintRect:
    """Grow a bubble vertically when PT-BR needs more wrapped lines."""
    content_width = max(1, rect.width - 20)
    available_height = max(1, max_height - 20)
    bounds = painter.fontMetrics().boundingRect(
        rect_type(0, 0, content_width, available_height),
        text_flags,
        text,
    )
    desired_height = min(
        max(rect.height, bounds.height() + 20),
        max(1, max_height - 4),
    )
    growth = desired_height - rect.height
    y = min(
        max(0, rect.y - growth // 2),
        max(0, max_height - desired_height),
    )
    return PaintRect(
        x=rect.x,
        y=y,
        width=rect.width,
        height=desired_height,
    )


def _offset_rect(rect: Rect, viewport_region: Rect | None) -> Rect:
    if viewport_region is None:
        return rect
    return Rect(
        x=rect.x + viewport_region.x,
        y=rect.y + viewport_region.y,
        width=rect.width,
        height=rect.height,
    )


def _create_translator(config: AppConfig, echo_translator: bool) -> Translator:
    return create_translator(config, echo=echo_translator)
