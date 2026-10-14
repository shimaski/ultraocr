from __future__ import annotations

import copy
import json
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import cv2

from manga_rt_translator.config import AppConfig, load_config
from manga_rt_translator.capture.windows_capture_affinity import (
    exclude_window_from_dxgi_capture,
)
from manga_rt_translator.glossary import (
    Glossary,
    combine_glossaries,
    resolve_glossary_path,
    work_glossary_identity,
)
from manga_rt_translator.global_hotkeys import (
    HOTKEY_CLEAR_REGION,
    HOTKEY_TRANSLATE,
    GlobalHotkeyManager,
)
from manga_rt_translator.models import Frame, OverlayItem, Rect
from manga_rt_translator.overlay_background import foreground_color
from manga_rt_translator.overlay import Overlay
from manga_rt_translator.profiles import (
    SiteProfile,
    SiteProfileStore,
    default_profile_path,
    normalized_region,
    profile_key,
    scaled_region,
)
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
    glossary_path: str | None = None


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


def _reader_ocr_languages(source: str) -> tuple[str, list[str]]:
    if source == "Auto":
        return "en", ["japan"]
    return source, []


def run_reader_app(
    config_path: str | None,
    glossary_path: str | None,
    initial_url: str | None = None,
) -> int:
    if platform.system().lower() != "windows":
        raise RuntimeError("The desktop reader is currently implemented for Windows only.")

    from PySide6.QtCore import QObject, QPoint, QRect, QRectF, QSize, Qt, QTimer, QUrl, Signal, Slot
    from PySide6.QtGui import QAction, QColor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
    from PySide6.QtWebEngineCore import QWebEnginePage
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDockWidget,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMenu,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QSizePolicy,
        QScrollArea,
        QSpinBox,
        QStatusBar,
        QToolBar,
        QToolButton,
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
            self.capture_scale = 1.0
            self._last_signature = None
            self._suspended = False
            self._placements: dict[str, PaintRect] = {}
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
            # A valid publication is the atomic handoff to the newest view.
            was_suspended = self._suspended
            self._suspended = False
            signature = tuple(
                (
                    item.bbox.as_tuple(),
                    item.display_bbox.as_tuple() if item.display_bbox else None,
                    item.text,
                    item.expires_at,
                    item.visual_hash,
                )
                for item in items
            )
            if signature == self._last_signature:
                if was_suspended:
                    self.update()
                return  # nothing changed; skip the repaint
            self._last_signature = signature
            self.items = list(items)
            live_keys = {_overlay_item_key(item) for item in self.items}
            self._placements = {
                key: rect for key, rect in self._placements.items() if key in live_keys
            }
            self.update()

        def suspend_presentation(self) -> None:
            self._suspended = True
            self.update()

        def set_viewport_region(self, region: Rect | None) -> None:
            self.viewport_region = region
            self.update()

        def set_capture_scale(self, scale: float) -> None:
            scale = max(1.0, float(scale))
            if scale == self.capture_scale:
                return
            self.capture_scale = scale
            self._placements.clear()
            self.update()

        def paintEvent(self, event: object) -> None:  # noqa: N802
            painter = QPainter(self)
            if self._suspended:
                return
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
            for item in sorted(
                live,
                key=lambda value: (-value.bbox.area, _overlay_item_key(value)),
            ):
                provisional = bool(item.source_text) and item.text == item.source_text
                source_rect = _offset_rect(
                    _scale_capture_rect(
                        item.bbox, self.capture_scale
                    ),
                    self.viewport_region,
                )
                bubble_bounds = None
                if item.display_bbox is not None:
                    bubble_rect = _offset_rect(
                        _scale_capture_rect(item.display_bbox, self.capture_scale),
                        self.viewport_region,
                    )
                    bubble_bounds = _intersect_paint_rect(
                        PaintRect(
                            bubble_rect.x,
                            bubble_rect.y,
                            bubble_rect.width,
                            bubble_rect.height,
                        ),
                        bounds,
                    )
                chosen: tuple[PaintRect, QFont] | None = None
                displaced_fallback: tuple[PaintRect, QFont] | None = None
                # Dense comic pages need smaller type. Try the configured size
                # first, then compact only as much as collision avoidance needs.
                for point_size in range(base_font_size, 8, -1):
                    font = QFont(self.overlay_config.font_family, point_size)
                    font.setBold(True)
                    font.setItalic(provisional)
                    painter.setFont(font)
                    rect = _fit_rect(source_rect, self.width(), self.height(), item.text)
                    if bubble_bounds is not None:
                        width = min(rect.width, bubble_bounds.width)
                        height = min(rect.height, bubble_bounds.height)
                        rect = PaintRect(
                            x=bubble_bounds.x + (bubble_bounds.width - width) // 2,
                            y=bubble_bounds.y + (bubble_bounds.height - height) // 2,
                            width=width,
                            height=height,
                        )
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
                    key = _overlay_item_key(item)
                    retained = self._placements.get(key)
                    placed = None
                    if retained is not None:
                        retained = _clamp_paint_rect(retained, bounds)
                        close_to_anchor = (
                            abs(retained.x - rect.x) <= 80
                            and abs(retained.y - rect.y) <= 80
                        )
                        within_bubble = (
                            bubble_bounds is None
                            or _paint_rect_contains(bubble_bounds, retained)
                        )
                        if close_to_anchor and within_bubble and not any(
                            _rects_overlap(retained, other) for other in occupied
                        ):
                            placed = retained
                    if placed is None and bubble_bounds is not None:
                        if (
                            rect.width <= bubble_bounds.width
                            and rect.height <= bubble_bounds.height
                        ):
                            placed = _place_without_overlap(
                                rect, occupied, bubble_bounds
                            )
                    if placed is not None:
                        chosen = (placed, font)
                        break
                    if displaced_fallback is None:
                        outside_bubble = _place_without_overlap(
                            rect, occupied, bounds
                        )
                        if outside_bubble is not None:
                            displaced_fallback = (outside_bubble, font)
                if chosen is None:
                    chosen = displaced_fallback
                if chosen is None:
                    # An exceptionally crowded viewport is still more useful
                    # with one caption omitted than with several unreadable
                    # captions painted on top of each other.
                    continue
                rect, font = chosen
                self._placements[_overlay_item_key(item)] = rect
                occupied.append(rect)
                layout.append((item, rect, font, provisional))

            for item, _, _, provisional in layout:
                patch = item.background_patch
                patch_rect = item.background_rect
                if not provisional and patch is not None and patch_rect is not None:
                    image = QImage(
                        patch.data,
                        patch.shape[1],
                        patch.shape[0],
                        int(patch.strides[0]),
                        QImage.Format.Format_RGB888,
                    ).copy()
                    scaled_patch_rect = _scale_capture_rect(patch_rect, self.capture_scale)
                    positioned_patch_rect = _offset_rect(
                        scaled_patch_rect,
                        self.viewport_region,
                    )
                    painter.drawImage(
                        QRectF(
                            positioned_patch_rect.x,
                            positioned_patch_rect.y,
                            positioned_patch_rect.width,
                            positioned_patch_rect.height,
                        ),
                        image,
                    )

            for item, rect, font, provisional in layout:
                painter.setFont(font)
                sampled_background = item.background_color
                text_color = QColor(*foreground_color(sampled_background), 245)
                has_restored_background = (
                    item.background_patch is not None and not provisional
                )
                bubble = QPainterPath()
                bubble.addRoundedRect(rect.x, rect.y, rect.width, rect.height, 10, 10)
                if not has_restored_background:
                    painter.fillPath(bubble.translated(0, 2), QColor(0, 0, 0, 48))
                if provisional:
                    provisional_background = sampled_background or (250, 250, 250)
                    provisional_alpha = (
                        255
                        if sampled_background is not None
                        else int(255 * self.overlay_config.opacity * 0.75)
                    )
                    painter.fillPath(
                        bubble,
                        QColor(*provisional_background, provisional_alpha),
                    )
                    painter.setPen(QPen(QColor(120, 120, 120, 110), 1.0))
                    painter.drawPath(bubble)
                    painter.setPen(
                        text_color
                        if sampled_background is not None
                        else QColor(96, 96, 96, 235)
                    )
                else:
                    background_color = (
                        QColor(
                            *sampled_background,
                            int(255 * self.overlay_config.opacity)
                            if has_restored_background
                            else 255,
                        )
                        if sampled_background is not None
                        else QColor(
                            255,
                            252,
                            245,
                            int(255 * self.overlay_config.opacity),
                        )
                    )
                    border_color = (
                        QColor(36, 31, 28, 110)
                        if text_color.red() < 128
                        else QColor(245, 245, 245, 150)
                    )
                    painter.fillPath(bubble, background_color)
                    if not has_restored_background:
                        painter.setPen(QPen(border_color, 1.2))
                        painter.drawPath(bubble)
                    painter.setPen(text_color)
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
            self._default_overlay_appearance = (
                launch.config.overlay.font_family,
                launch.config.overlay.font_size,
                launch.config.overlay.opacity,
            )
            self._base_glossary = copy.deepcopy(launch.glossary)
            self._active_work_identity: tuple[str, str] | None = None
            self._active_work_glossary = Glossary()
            self.runtime: ReaderRuntime | None = None
            self.selected_region: Rect | None = None
            self.site_profiles = SiteProfileStore(default_profile_path())
            self._active_profile_host: str | None = None
            self._dxcam_instance = None
            self._dxcam_region: tuple[int, int, int, int] | None = None
            self._wgc_instance = None
            self._overlay_capture_excluded = False
            self._overlay_capture_hwnd = 0
            self._overlay_capture_attempted = False
            self._overlay_capture_method = ""
            self._overlay_capture_error = 0
            self.capture_timer = QTimer(self)
            self.capture_timer.timeout.connect(self._capture_reader_frame)
            self.scroll_poll_timer = QTimer(self)
            self.scroll_poll_timer.timeout.connect(self._poll_scroll)
            self.scroll_settle_timer = QTimer(self)
            self.scroll_settle_timer.setSingleShot(True)
            self.scroll_settle_timer.timeout.connect(self._capture_settled_scroll)
            self._last_poll_y: float | None = None
            self._last_poll_x: float | None = None
            self._last_view_metrics: tuple[float, ...] | None = None
            self._last_shifted_scroll_px: int | None = None
            self._last_capture_size: tuple[int, int] | None = None
            self._scroll_moved_since_capture = False
            self._last_scroll_target_id: str | None = None
            self._settle_count = 0
            self._captured_y: float | None = None
            self._captured_x: float | None = None
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
            self.profile_save_timer = QTimer(self)
            self.profile_save_timer.setSingleShot(True)
            self.profile_save_timer.timeout.connect(self._save_site_profile)
            self.bridge = ReaderOverlayBridge()
            self.setWindowTitle("Leitor de Quadrinhos | Manga Real-Time Translator")
            self.resize(1440, 960)
            self._build_ui()
            self._connect_events()
            self.global_hotkeys = GlobalHotkeyManager(
                QApplication.instance(),
                {
                    HOTKEY_TRANSLATE: self._translate_from_global_hotkey,
                    HOTKEY_CLEAR_REGION: self._clear_region_selection,
                },
            )
            self.translate_shortcut.setEnabled(
                not self.global_hotkeys.translate_registered
            )
            self.clear_shortcut.setEnabled(
                not self.global_hotkeys.clear_region_registered
            )
            if self.global_hotkeys.errors:
                _reader_log(
                    "hotkeys.registration_failed "
                    + repr(self.global_hotkeys.errors)
                )
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

            self.toolbar = QToolBar("Navegação", self)
            self.toolbar.setMovable(False)
            self.toolbar.setObjectName("navigationToolbar")
            self.addToolBar(self.toolbar)

            back_action = QAction("‹", self)
            back_action.setToolTip("Voltar")
            back_action.triggered.connect(self.webview.back)
            self.toolbar.addAction(back_action)

            forward_action = QAction("›", self)
            forward_action.setToolTip("Avançar")
            forward_action.triggered.connect(self.webview.forward)
            self.toolbar.addAction(forward_action)

            reload_action = QAction("⟳", self)
            reload_action.setToolTip("Recarregar página")
            reload_action.triggered.connect(self.webview.reload)
            self.toolbar.addAction(reload_action)

            history_action = QAction("Histórico", self)
            history_action.setToolTip("Ver traduções anteriores")
            history_action.triggered.connect(self._show_translation_history)
            self.toolbar.addAction(history_action)

            self.url_input = QLineEdit(self.launch.initial_url)
            self.url_input.setObjectName("chapterUrlInput")
            self.url_input.setPlaceholderText("Cole o link do capítulo ou da página…")
            self.url_input.setClearButtonEnabled(True)
            self.url_input.setMinimumWidth(280)
            self.url_input.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
            self.url_input.returnPressed.connect(self._open_url)
            self.toolbar.addWidget(self.url_input)

            open_button = QPushButton("Abrir link")
            open_button.setObjectName("openChapterButton")
            open_button.setDefault(True)
            open_button.clicked.connect(self._open_url)
            self.toolbar.addWidget(open_button)

            tools_button = QToolButton(self)
            tools_button.setText("Ferramentas")
            tools_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
            tools_button.setObjectName("toolsMenuButton")
            tools_menu = QMenu(tools_button)

            memory_action = QAction("Memória de tradução…", self)
            memory_action.triggered.connect(self._manage_translation_memory)
            tools_menu.addAction(memory_action)

            glossary_action = QAction("Glossário geral…", self)
            glossary_action.triggered.connect(self._edit_glossary)
            tools_menu.addAction(glossary_action)

            work_glossary_action = QAction("Glossário da obra…", self)
            work_glossary_action.triggered.connect(self._edit_work_glossary)
            tools_menu.addAction(work_glossary_action)

            appearance_action = QAction("Aparência do overlay…", self)
            appearance_action.triggered.connect(self._edit_overlay_appearance)
            tools_menu.addAction(appearance_action)

            tools_menu.addSeparator()
            image_action = QAction("Abrir imagens locais…", self)
            image_action.triggered.connect(self._open_local_image)
            tools_menu.addAction(image_action)

            export_action = QAction("Exportar página traduzida…", self)
            export_action.setToolTip(
                "Salvar a captura visível com as traduções sobrepostas em PNG."
            )
            export_action.triggered.connect(self._export_translated_view)
            tools_menu.addAction(export_action)

            tools_menu.addSeparator()
            tools_button.setMenu(tools_menu)
            self.toolbar.addWidget(tools_button)

            dock = QDockWidget("Tradução e leitura", self)
            dock.setObjectName("translationDock")
            dock.setAllowedAreas(
                Qt.DockWidgetArea.LeftDockWidgetArea
                | Qt.DockWidgetArea.RightDockWidgetArea
            )
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                | QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
            dock.setMinimumWidth(280)
            dock.setMaximumWidth(360)
            panel = QWidget(dock)
            panel_layout = QVBoxLayout(panel)
            panel_layout.setContentsMargins(12, 12, 12, 12)
            panel_layout.setSpacing(12)

            self.start_button = QPushButton("Preparar captura")
            self.start_button.setToolTip("Inicia OCR e tradução para a página atual.")
            panel_layout.addWidget(self.start_button)

            self.analyze_button = QPushButton("Traduzir agora")
            self.analyze_button.setObjectName("translateNowButton")
            self.analyze_button.setToolTip("Analisa e traduz a página visível agora.")
            panel_layout.addWidget(self.analyze_button)

            action_grid = QGridLayout()
            action_grid.setHorizontalSpacing(8)
            action_grid.setVerticalSpacing(8)
            self.select_region_button = QPushButton("Selecionar área")
            self.inspect_button = QPushButton("Inspecionar OCR")
            self.clear_region_button = QPushButton("Limpar área")
            self.clear_region_button.setEnabled(False)
            self.stop_button = QPushButton("Parar")
            action_grid.addWidget(self.select_region_button, 0, 0)
            action_grid.addWidget(self.inspect_button, 0, 1)
            action_grid.addWidget(self.clear_region_button, 1, 0)
            action_grid.addWidget(self.stop_button, 1, 1)
            panel_layout.addLayout(action_grid)

            translation_group = QGroupBox("Tradução")
            translation_form = QFormLayout(translation_group)
            translation_form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            self.source_combo = QComboBox()
            for label, language in (
                ("Automático (inglês/japonês)", "Auto"),
                ("Inglês", "en"),
                ("Japonês", "japan"),
                ("Coreano", "korean"),
                ("Chinês simplificado", "ch"),
            ):
                self.source_combo.addItem(label, language)
            self.source_combo.setCurrentIndex(0)
            self.source_combo.setToolTip(
                "Detectar automaticamente ou definir o idioma de origem."
            )
            translation_form.addRow("Idioma de origem", self.source_combo)

            self.mode_combo = QComboBox()
            self.mode_combo.addItems(["Traduzir PT-BR", "Somente OCR (rápido)"])
            self.mode_combo.setCurrentText("Traduzir PT-BR")
            translation_form.addRow("Modo", self.mode_combo)

            self.provider_combo = QComboBox()
            self.provider_combo.addItems(
                ["Google", "DeepL", "Gemini", "Libre", "Argos offline", "Echo"]
            )
            saved_provider = str(
                self.launch.config.translation.provider or "google"
            ).strip().lower()
            provider_label = {
                "google": "Google", "deepl": "DeepL", "gemini": "Gemini",
                "libre": "Libre", "argos": "Argos offline", "echo": "Echo",
            }.get(saved_provider, "Google")
            provider_index = self.provider_combo.findText(provider_label)
            if provider_index >= 0:
                self.provider_combo.setCurrentIndex(provider_index)

            # Keep provider credentials isolated so switching providers never
            # overwrites another provider's key or the value loaded from config.
            self._api_keys = {
                "deepl": self.launch.config.translation.deepl_api_key,
                "gemini": self.launch.config.translation.gemini_api_key,
                "libre": self.launch.config.translation.libre_api_key,
            }
            self.deepl_key_input = QLineEdit(self._api_keys.get(saved_provider, ""))
            self.deepl_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.deepl_key_input.setPlaceholderText("DeepL, Gemini ou Libre")
            self.deepl_key_input.textChanged.connect(self._store_api_key)
            self.deepl_key_input.editingFinished.connect(
                self._sync_runtime_translation_settings
            )
            self.provider_combo.currentTextChanged.connect(self._sync_provider_ui)
            translation_form.addRow("Provedor", self.provider_combo)
            translation_form.addRow("Chave de API", self.deepl_key_input)
            panel_layout.addWidget(translation_group)

            capture_group = QGroupBox("Captura e overlay")
            capture_form = QFormLayout(capture_group)
            capture_form.setFieldGrowthPolicy(
                QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
            )
            self.analysis_combo = QComboBox()
            self.analysis_combo.addItems(["Manual", "Contínuo"])
            self.analysis_combo.setCurrentText("Contínuo")
            capture_form.addRow("Análise", self.analysis_combo)

            self.fps_spin = QSpinBox()
            self.fps_spin.setRange(1, 12)
            self.fps_spin.setValue(8)
            self.fps_spin.setToolTip(
                "Sensibilidade ao rolar: maior = reage mais rápido ao parar o scroll"
            )
            capture_form.addRow("Sensibilidade", self.fps_spin)

            self.overlay_checkbox = QCheckBox("Mostrar tradução sobre a página")
            self.overlay_checkbox.setChecked(True)
            capture_form.addRow(self.overlay_checkbox)

            self.auto_refresh_checkbox = QCheckBox(
                "Repetir tradução automaticamente"
            )
            self.refresh_spin = QSpinBox()
            self.refresh_spin.setRange(1, 10)
            self.refresh_spin.setValue(2)
            self.refresh_spin.setSuffix(" s")
            refresh_row = QWidget(capture_group)
            refresh_layout = QHBoxLayout(refresh_row)
            refresh_layout.setContentsMargins(0, 0, 0, 0)
            refresh_layout.addWidget(self.auto_refresh_checkbox, 1)
            refresh_layout.addWidget(self.refresh_spin)
            capture_form.addRow(refresh_row)
            panel_layout.addWidget(capture_group)
            panel_layout.addStretch(1)

            scroll_area = QScrollArea(dock)
            scroll_area.setWidgetResizable(True)
            scroll_area.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll_area.setWidget(panel)
            dock.setWidget(scroll_area)
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
            dock.toggleViewAction().setText("Mostrar painel de tradução")
            tools_menu.addAction(dock.toggleViewAction())

            self.setStyleSheet("""
                QMainWindow { background: #f4f6f6; }
                QToolBar#navigationToolbar {
                    background: #ffffff; border: 0; border-bottom: 1px solid #dce3e4;
                    spacing: 6px; padding: 8px;
                }
                QToolBar#navigationToolbar QToolButton,
                QToolBar#navigationToolbar QPushButton {
                    min-height: 32px; padding: 0 10px; border: 1px solid #d5dede;
                    border-radius: 5px; background: #ffffff; color: #263437;
                }
                QToolBar#navigationToolbar QToolButton:hover,
                QToolBar#navigationToolbar QPushButton:hover { background: #edf5f5; }
                QLineEdit#chapterUrlInput {
                    min-height: 32px; padding: 0 10px; border: 1px solid #cbd6d8;
                    border-radius: 5px; background: #fbfcfc; color: #182529;
                }
                QLineEdit#chapterUrlInput:focus { border: 1px solid #07838a; }
                QDockWidget#translationDock { background: #f7f9f9; }
                QDockWidget#translationDock::title {
                    background: #edf2f2; padding: 9px 12px; font-weight: 600;
                    border-bottom: 1px solid #dce3e4;
                }
                QGroupBox {
                    font-weight: 600; border: 1px solid #dce3e4; border-radius: 5px;
                    margin-top: 10px; padding: 12px 8px 8px; background: #ffffff;
                }
                QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 4px; }
                QPushButton {
                    min-height: 32px; padding: 0 9px; border: 1px solid #d2dddd;
                    border-radius: 5px; background: #ffffff; color: #263437;
                }
                QPushButton:hover { background: #edf5f5; }
                QPushButton#translateNowButton, QPushButton#openChapterButton {
                    background: #087f86; border-color: #087f86; color: #ffffff;
                    font-weight: 600;
                }
                QPushButton#translateNowButton:hover, QPushButton#openChapterButton:hover {
                    background: #066d74; border-color: #066d74;
                }
                QComboBox, QSpinBox {
                    min-height: 29px; padding: 0 6px; border: 1px solid #d2dddd;
                    border-radius: 4px; background: #ffffff;
                }
                QStatusBar { background: #ffffff; border-top: 1px solid #dce3e4; }
            """)

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

        def _refresh_scroll_overlay(self) -> None:
            """Rebind retained captions to the new view token during scroll."""
            if self.runtime is None:
                return
            pipeline = self.runtime.pipeline
            with self.runtime.view.lock:
                items = (
                    list(pipeline.active_items)
                    if pipeline is not None
                    else list(self.overlay_widget.items)
                )
                token = self.runtime.view.token()
            self.overlay_widget.update_items((token, items))

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
            for signal in (
                self.source_combo.currentTextChanged,
                self.mode_combo.currentTextChanged,
                self.provider_combo.currentTextChanged,
                self.analysis_combo.currentTextChanged,
                self.overlay_checkbox.toggled,
                self.fps_spin.valueChanged,
            ):
                signal.connect(self._schedule_profile_save)
            self.mode_combo.currentTextChanged.connect(
                self._sync_runtime_translation_settings
            )
            self.translate_shortcut = QShortcut(
                QKeySequence("Ctrl+Shift+T"),
                self,
                activated=self._translate_current_region,
            )
            self.clear_shortcut = QShortcut(
                QKeySequence("Ctrl+Shift+R"),
                self,
                activated=self._clear_region_selection,
            )

        def _load_initial_page(self) -> None:
            self._open_url()

        def _show_translation_history(self) -> None:
            from manga_rt_translator.ui import show_translation_history

            changed = show_translation_history(
                self, self.launch.config.cache.path
            )
            if changed and self.runtime is not None and self.selected_region is not None:
                self._translate_current_region()

        def _manage_translation_memory(self) -> None:
            from manga_rt_translator.ui import manage_translation_memory

            changed = manage_translation_memory(
                self, self.launch.config.cache.path
            )
            if changed and self.runtime is not None and self.selected_region is not None:
                self._translate_current_region()

        def _edit_glossary(self) -> None:
            from manga_rt_translator.ui import edit_glossary

            path = Path(self.launch.glossary_path) if self.launch.glossary_path else Path.cwd() / "glossary.toml"
            if edit_glossary(
                self,
                self._base_glossary,
                path,
                self.launch.config.cache.path,
            ):
                self.launch.glossary_path = str(path)
                self._activate_work_glossary(self.webview.url().toString(), force=True)
                self.status.showMessage("Glossário salvo e aplicado às próximas traduções.")
                if self.runtime is not None and self.selected_region is not None:
                    self._translate_current_region()

        def _edit_overlay_appearance(self) -> None:
            from manga_rt_translator.overlay_preferences import edit_overlay_appearance

            if not edit_overlay_appearance(self, self.launch.config.overlay):
                return
            self.overlay_widget.update()
            self._schedule_profile_save()

        def _work_glossary_path(self, identity: tuple[str, str]) -> Path:
            root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
            return root / "MangaRealTimeTranslator" / "glossaries" / f"{identity[0]}.toml"

        def _activate_work_glossary(self, url: str, *, force: bool = False) -> None:
            identity = work_glossary_identity(url)
            if identity == self._active_work_identity and not force:
                return
            self._active_work_identity = identity
            self._active_work_glossary = Glossary()
            scope = ""
            if identity is not None:
                path = self._work_glossary_path(identity)
                if path.is_file():
                    try:
                        self._active_work_glossary = Glossary.load(path)
                    except (OSError, ValueError, RuntimeError):
                        self.status.showMessage("Glossário da obra inválido; usando apenas o global.")
                scope = f"work:{identity[0]}"
            self.launch.glossary = combine_glossaries(
                self._base_glossary, self._active_work_glossary, scope
            )
            if self.runtime is not None:
                self.runtime.glossary = self.launch.glossary
                if self.runtime.pipeline is not None:
                    self.runtime.pipeline.glossary = self.launch.glossary

        def _edit_work_glossary(self) -> None:
            from manga_rt_translator.ui import edit_glossary

            if self._active_work_identity is None:
                QMessageBox.information(
                    self, "Glossário da obra",
                    "Abra uma página de capítulo que identifique a obra para editar o glossário contextual.",
                )
                return
            identity = self._active_work_identity
            scope = f"work:{identity[0]}|"
            path = self._work_glossary_path(identity)
            if edit_glossary(
                self,
                self._active_work_glossary,
                path,
                self.launch.config.cache.path,
                cache_scope=scope,
            ):
                self.launch.glossary = combine_glossaries(
                    self._base_glossary, self._active_work_glossary,
                    f"work:{identity[0]}",
                )
                if self.runtime is not None:
                    self.runtime.glossary = self.launch.glossary
                    if self.runtime.pipeline is not None:
                        self.runtime.pipeline.glossary = self.launch.glossary
                self.status.showMessage(
                    f"Glossário contextual salvo para {identity[1]}."
                )
                if self.runtime is not None and self.selected_region is not None:
                    self._translate_current_region()

        def _open_url(self) -> None:
            raw = self.url_input.text().strip()
            if not raw:
                return
            if "://" not in raw:
                raw = f"https://{raw}"
                self.url_input.setText(raw)
            self.webview.setUrl(QUrl(raw))

        def _open_local_image(self) -> None:
            from manga_rt_translator.image_viewer import choose_local_images

            choose_local_images(self, self.webview, self.status.showMessage)

        def _export_translated_view(self) -> None:
            from PySide6.QtWidgets import QFileDialog

            from manga_rt_translator.image_export import composite_overlay_image

            items = self.overlay_widget.items
            translated = [
                item for item in items
                if item.text.strip() and item.text != item.source_text
            ]
            if not translated:
                self.status.showMessage("Ainda não há traduções para exportar.")
                return
            pipeline = self.runtime.pipeline if self.runtime is not None else None
            if pipeline is not None:
                with pipeline.view.lock:
                    pending = any(
                        not future.done() for future in pipeline._futures
                    )
                if pending:
                    self.status.showMessage(
                        "Aguarde as traduções terminarem para exportar."
                    )
                    return
            if not self.overlay_widget.isVisible():
                self.status.showMessage("O overlay está oculto; mostre-o antes de exportar.")
                return

            QApplication.processEvents()
            try:
                page_image = self.webview.grab().toImage()
                overlay_image = self.overlay_widget.grab().toImage()
                result = composite_overlay_image(page_image, overlay_image)
            except (OSError, ValueError) as exc:
                self.status.showMessage(f"Falha ao compor a exportação: {exc}")
                return

            filename, _selected_filter = QFileDialog.getSaveFileName(
                self,
                "Exportar página visível traduzida",
                str(Path.home() / "manga-translated.png"),
                "Imagem PNG (*.png)",
            )
            if not filename:
                return
            path = Path(filename).expanduser()
            if path.suffix.lower() != ".png":
                path = path.with_suffix(".png")
            if not result.save(str(path), "PNG"):
                self.status.showMessage("Falha ao exportar: o Qt não conseguiu gravar o PNG.")
                return
            self.status.showMessage(f"Captura traduzida salva: {path}")

        def _sync_url(self, url: QUrl) -> None:
            self.url_input.setText(url.toString())
            host = profile_key(url.toString())
            if host == self._active_profile_host:
                return
            if self._active_profile_host is not None:
                self._save_site_profile(self._active_profile_host)
                self._handle_stop_clicked()
                self.selected_region = None
                self.overlay_widget.set_viewport_region(None)
                self.clear_region_button.setEnabled(False)
            self._active_profile_host = None

        def _handle_load_started(self) -> None:
            self.scroll_settle_timer.stop()
            self._last_poll_y = None
            self._last_poll_x = None
            self._last_view_metrics = None
            self._last_scroll_target_id = None
            self._last_shifted_scroll_px = None
            self._captured_y = None
            self._captured_x = None
            self._last_capture_size = None
            self._scroll_moved_since_capture = False
            if self.runtime:
                self.runtime.invalidate_view()
            self.status.showMessage("Carregando página...")

        def _handle_load_finished(self, ok: bool) -> None:
            self._sync_overlay_geometry()
            restored_region = False
            if ok:
                self._activate_work_glossary(self.webview.url().toString())
                host = profile_key(self.webview.url().toString())
                if host != self._active_profile_host:
                    self._active_profile_host = host
                    restored_region = self._restore_site_profile(
                        self.webview.url().toString()
                    )
                message = "Página carregada. Selecione a região de leitura para começar."
                if self.page.last_console_error:
                    message = f"{message} Aviso do site: {self.page.last_console_error}"
                if self.global_hotkeys.errors:
                    message += " Atalho global ocupado; os atalhos funcionam com o leitor em foco."
                self.status.showMessage(message)
            else:
                self.status.showMessage("Falha ao carregar a página.")
            if ok and not restored_region and self.runtime and self.selected_region is not None:
                QTimer.singleShot(350, self._translate_current_region)

        def _store_api_key(self, text: str) -> None:
            provider = self.provider_combo.currentText().strip().lower()
            if provider in self._api_keys:
                self._api_keys[provider] = text.strip()

        def _schedule_profile_save(self, *_args: object) -> None:
            if self._active_profile_host:
                self.profile_save_timer.start(250)

        def _save_site_profile(self, host: str | None = None) -> None:
            host = host or self._active_profile_host
            if not host:
                return
            width, height = self.webview.width(), self.webview.height()
            profile = SiteProfile(
                source_lang=str(
                    self.source_combo.currentData() or self.source_combo.currentText()
                ),
                provider=self.provider_combo.currentText(),
                analysis_mode=self.analysis_combo.currentText(),
                fps=int(self.fps_spin.value()),
                overlay_enabled=self.overlay_checkbox.isChecked(),
                ocr_only=self.mode_combo.currentText() == "Somente OCR (rápido)",
                region=normalized_region(self.selected_region, width, height),
                overlay_font_family=self.launch.config.overlay.font_family,
                overlay_font_size=self.launch.config.overlay.font_size,
                overlay_opacity=self.launch.config.overlay.opacity,
            )
            self.site_profiles.save(f"https://{host}", profile)

        def _restore_site_profile(self, url: str) -> bool:
            profile = self.site_profiles.load(url)
            self._apply_overlay_appearance(
                profile.overlay_font_family if profile else None,
                profile.overlay_font_size if profile else None,
                profile.overlay_opacity if profile else None,
            )
            if profile is None:
                self.source_combo.setCurrentIndex(0)
                provider = str(self.launch.config.translation.provider or "google").lower()
                labels = {
                    "google": "Google", "deepl": "DeepL", "gemini": "Gemini",
                    "libre": "Libre", "argos": "Argos offline", "echo": "Echo",
                }
                self.provider_combo.setCurrentText(labels.get(provider, "Google"))
                self.mode_combo.setCurrentText("Traduzir PT-BR")
                self.analysis_combo.setCurrentText("Contínuo")
                self.fps_spin.setValue(8)
                self.overlay_checkbox.setChecked(True)
                self._handle_region_selected(None)
                return False

            source_index = self.source_combo.findData(profile.source_lang)
            self.source_combo.setCurrentIndex(source_index if source_index >= 0 else 0)
            self.provider_combo.setCurrentText(profile.provider)
            self.mode_combo.setCurrentText(
                "Somente OCR (rápido)" if profile.ocr_only else "Traduzir PT-BR"
            )
            self.analysis_combo.setCurrentText(profile.analysis_mode)
            self.fps_spin.setValue(profile.fps)
            self.overlay_checkbox.setChecked(profile.overlay_enabled)
            region = scaled_region(profile.region, self.webview.width(), self.webview.height())
            if region is not None:
                self._handle_region_selected(region)
            else:
                self.selected_region = None
                self.overlay_widget.set_viewport_region(None)
                self.clear_region_button.setEnabled(False)
                self._apply_capture_schedule()
                self._sync_refresh_timer()
            return region is not None

        def _apply_overlay_appearance(
            self,
            font_family: str | None,
            font_size: int | None,
            opacity: float | None,
        ) -> None:
            config = self.launch.config.overlay
            default_family, default_size, default_opacity = self._default_overlay_appearance
            config.font_family = font_family or default_family
            config.font_size = font_size if font_size is not None else default_size
            config.opacity = opacity if opacity is not None else default_opacity
            self.overlay_widget.update()

        def _sync_provider_ui(self, provider_text: str) -> None:
            provider = provider_text.strip().lower()
            if provider in self._api_keys:
                self.deepl_key_input.setText(self._api_keys[provider])
            needs_key = provider in {"deepl", "gemini", "libre"}
            self.deepl_key_input.setEnabled(needs_key)
            self._sync_runtime_translation_settings()

        def _sync_runtime_translation_settings(self) -> None:
            runtime = self.runtime
            if runtime is None or runtime.worker is None or not runtime.worker.is_alive():
                return
            config = copy.deepcopy(runtime.config)
            provider = self.provider_combo.currentText().strip().lower()
            if provider == "argos offline":
                provider = "argos"
            config.translation.provider = provider
            config.translation.deepl_api_key = self._api_keys.get("deepl", "")
            config.translation.gemini_api_key = self._api_keys.get("gemini", "")
            config.translation.libre_api_key = self._api_keys.get("libre", "")
            echo = self.mode_combo.currentText() == "Somente OCR (rápido)"
            active_translation = runtime.config.translation
            if (
                provider == active_translation.provider
                and config.translation.deepl_api_key == active_translation.deepl_api_key
                and config.translation.gemini_api_key == active_translation.gemini_api_key
                and config.translation.libre_api_key == active_translation.libre_api_key
                and echo == runtime.echo_translator
            ):
                return
            try:
                runtime.replace_translation_config(config, echo_translator=echo)
            except Exception as exc:
                active = runtime.config.translation.provider
                self.status.showMessage(
                    f"Não foi possível ativar {self.provider_combo.currentText()}; "
                    f"continua ativo {active}: {exc}"
                )
                return
            if echo:
                self.status.showMessage(
                    "Modo Somente OCR ativo; o provedor será usado ao ativar a tradução."
                )
            else:
                self.status.showMessage(
                    f"Provedor {self.provider_combo.currentText()} ativo; "
                    "reprocessando a região visível..."
                )
            QTimer.singleShot(0, self._translate_current_region)

        def _handle_start_clicked(self) -> None:
            _reader_log("UI: Start Translation clicked")
            self._ensure_runtime_started()

        def _ensure_runtime_started(self) -> bool:
            if self.runtime and self.runtime.worker and self.runtime.worker.is_alive():
                return True
            try:
                config = copy.deepcopy(self.launch.config)
                config.ocr.source_lang, config.ocr.extra_langs = _reader_ocr_languages(
                    str(self.source_combo.currentData() or "Auto")
                )
                config.capture.fps = int(self.fps_spin.value())
                provider = self.provider_combo.currentText().strip().lower()
                if provider == "argos offline":
                    provider = "argos"
                config.translation.provider = provider
                # Keys come from the per-provider memory (seeded from config
                # and updated as the user edits the field).
                config.translation.deepl_api_key = self._api_keys.get("deepl", "")
                config.translation.gemini_api_key = self._api_keys.get("gemini", "")
                config.translation.libre_api_key = self._api_keys.get("libre", "")
                if provider == "argos":
                    ready, message = create_translator(config).check_health()
                    if not ready:
                        QMessageBox.information(self, "Argos offline", message)
                        return False
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

        def _translate_from_global_hotkey(self) -> None:
            if QApplication.activeWindow() is self and not self.isMinimized():
                self._translate_current_region()
                return
            if self.isMinimized():
                self.showNormal()
            else:
                self.show()
            self.raise_()
            self.activateWindow()
            QTimer.singleShot(150, self._translate_current_region)

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
            self.profile_save_timer.stop()
            self._save_site_profile()
            self._handle_stop_clicked()
            self.global_hotkeys.close()
            self.overlay_raise_timer.stop()
            self.status_timer.stop()
            self.overlay_widget.close()
            self.selection_overlay.close()
            camera = getattr(self, "_dxcam_instance", None)
            release = getattr(camera, "release", None)
            if callable(release):
                try:
                    release()
                except Exception as exc:
                    _reader_log(f"dxcam.release_error {type(exc).__name__}: {exc}")
            self._stop_wgc()
            super().closeEvent(event)

        def _handle_stop_clicked(self) -> None:
            self.capture_timer.stop()
            self.scroll_poll_timer.stop()
            self.scroll_settle_timer.stop()
            self.sweep_timer.stop()
            self.refresh_timer.stop()
            self._stop_wgc()
            self._stop_dxcam()
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
            self._last_poll_x = None
            self._last_view_metrics = None
            self._last_scroll_target_id = None
            self._captured_y = None
            self._captured_x = None
            self._scroll_moved_since_capture = False
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
                self._schedule_profile_save()
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
            self._schedule_profile_save()
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
                self.sweep_timer.start(
                    max(125, int(1000 / max(1, self.fps_spin.value())))
                )
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
                from manga_rt_translator.scroll_regions import scroll_position_script

                script = scroll_position_script(
                    self.selected_region,
                    self.webview.width(),
                    self.webview.height(),
                )
                self.webview.page().runJavaScript(
                    script, 0, self._on_scroll_poll
                )
            except RuntimeError:
                return

        def _on_scroll_poll(self, value: object) -> None:
            self._poll_results = getattr(self, "_poll_results", 0) + 1
            if self._poll_results % 40 == 1:
                _reader_log(f"scroll.poll callback ok (results={self._poll_results})")
            if self.analysis_combo.currentText() != "Contínuo":
                return
            from manga_rt_translator.scroll_regions import viewport_metrics_changed

            scroll_target_id = "window"
            try:
                if isinstance(value, str):
                    value = json.loads(value)
                if isinstance(value, dict):
                    y = float(value.get("y") or 0)
                    x = float(value.get("x") or 0)
                    scroll_target_id = str(value.get("id") or "window")
                    raw_metrics = value.get("metrics")
                    metrics = (
                        tuple(float(raw_metrics[key]) for key in (
                            "width", "height", "dpr", "documentWidth", "documentHeight"
                        ))
                        if isinstance(raw_metrics, dict)
                        and all(key in raw_metrics for key in (
                            "width", "height", "dpr", "documentWidth", "documentHeight"
                        ))
                        else None
                    )
                else:
                    y = float(value or 0)
                    x = 0.0
                    metrics = None
            except (TypeError, ValueError):
                return
            if self._last_poll_y is None:
                self._last_scroll_target_id = scroll_target_id
                self._last_poll_y = y
                self._last_poll_x = x
                self._last_view_metrics = metrics
                self._captured_y = y
                self._captured_x = x
                self._last_shifted_scroll_px = round(
                    y * self.overlay_widget.capture_scale
                )
                return
            if viewport_metrics_changed(self._last_view_metrics, metrics):
                self._last_view_metrics = metrics
                self._last_poll_x = x
                self._last_poll_y = y
                self._captured_x = None
                self._captured_y = None
                self._scroll_moved_since_capture = True
                self._last_scroll_move_at = time.time()
                if self.runtime:
                    self.runtime.invalidate_view(clear_overlay=False, scrolling=True)
                self._refresh_scroll_overlay()
                self.scroll_settle_timer.start(
                    max(300, int(self.launch.config.diff.scroll_settle_ms))
                )
                _reader_log(f"scroll.viewport_changed metrics={metrics}")
                return
            self._last_view_metrics = metrics or self._last_view_metrics
            if scroll_target_id != self._last_scroll_target_id:
                self._last_scroll_target_id = scroll_target_id
                self._last_poll_y = y
                self._last_poll_x = x
                self._captured_y = None
                self._captured_x = None
                self._scroll_moved_since_capture = True
                self._last_shifted_scroll_px = round(
                    y * self.overlay_widget.capture_scale
                )
                self._last_scroll_move_at = time.time()
                self.scroll_settle_timer.start(
                    max(300, int(self.launch.config.diff.scroll_settle_ms))
                )
                if self.runtime:
                    self.runtime.invalidate_view(clear_overlay=False, scrolling=True)
                self._refresh_scroll_overlay()
                return
            previous_y = self._last_poll_y
            previous_x = self._last_poll_x if self._last_poll_x is not None else x
            delta_y = abs(y - previous_y)
            delta_x = abs(x - previous_x)
            delta = max(delta_x, delta_y)
            # Slow drift (~reading speed) is translate-worthy; fast scrolling
            # produces frames with cut-off balloons.
            self._scroll_velocity_slow = delta <= 25.0
            if delta > 0.5:
                self._last_poll_y = y
                self._last_poll_x = x
                # Keep the last analyzed offset for the strip calculation;
                # movement is tracked separately for an away-and-back trip.
                self._scroll_moved_since_capture = True
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
                    scale = self.overlay_widget.capture_scale
                    delta_y_px = round((previous_y - y) * scale)
                    delta_x_px = round((previous_x - x) * scale)
                    self._last_shifted_scroll_px = round(y * scale)
                    viewport_size = self._last_capture_size
                    self.runtime.invalidate_view(
                        clear_overlay=False,
                        scroll_delta_x=delta_x_px,
                        scroll_delta_y=delta_y_px,
                        viewport_size=viewport_size,
                        scrolling=True,
                    )
                self._refresh_scroll_overlay()
                return

        def _capture_settled_scroll(self) -> None:
            if self.runtime is None or self.analysis_combo.currentText() != "Contínuo":
                return
            y = self._last_poll_y
            x = self._last_poll_x
            from manga_rt_translator.scroll_regions import (
                plan_scroll_analysis,
                promote_interrupted_scroll_to_full_view,
            )

            pipeline = self.runtime.pipeline
            analyze, dirty = plan_scroll_analysis(
                self._captured_y,
                y,
                self._scroll_moved_since_capture,
                baseline_ready=bool(
                    pipeline is not None and pipeline.differ.has_baseline
                ),
                viewport_size=self._last_capture_size,
                capture_scale=self.overlay_widget.capture_scale,
                captured_x=self._captured_x,
                current_x=x,
                max_partial_fraction=0.7,
            )
            retrying_interrupted_work = self.runtime.needs_full_refresh_after_scroll
            analyze, dirty = promote_interrupted_scroll_to_full_view(
                analyze, dirty, retrying_interrupted_work
            )
            if not analyze:
                return
            self._last_settle_capture = time.time()
            _reader_log(
                f"scroll.settled y={y:.0f}; scheduling "
                f"{'full retry' if retrying_interrupted_work else 'region'} analysis"
            )
            self.status.showMessage(
                "Novo conteúdo detectado; lendo e traduzindo..."
            )
            dirty_rects = (dirty,) if dirty is not None else ()
            partial = bool(dirty_rects)
            if self._capture_reader_frame(
                force_analyze=not partial,
                dirty_rects=dirty_rects,
                dirty_rects_stable=partial,
            ):
                if retrying_interrupted_work:
                    self.runtime.mark_scroll_refresh_submitted()
                self._captured_y = y
                self._captured_x = x
                self._scroll_moved_since_capture = False
            else:
                self.scroll_settle_timer.start(350)

        def _sync_refresh_timer(self) -> None:
            if not self.auto_refresh_checkbox.isChecked() or self.selected_region is None:
                self.refresh_timer.stop()
                return
            self.refresh_timer.start(int(self.refresh_spin.value()) * 1000)

        def _dxcam_camera(self) -> object | None:
            """Return the persistent Desktop Duplication capture session."""
            cam = getattr(self, "_dxcam_instance", "uninitialized")
            if cam != "uninitialized":
                return cam
            self._dxcam_instance = None
            try:
                from manga_rt_translator.capture.windows_dxgi import (
                    WindowsDxgiRegionCapture,
                )

                self._dxcam_instance = WindowsDxgiRegionCapture(
                    fps=max(30, int(self.launch.config.capture.fps)),
                )
            except Exception as exc:
                _reader_log(f"dxcam.unavailable {type(exc).__name__}: {exc}")
            return self._dxcam_instance

        def _stop_dxcam(self) -> None:
            camera = getattr(self, "_dxcam_instance", None)
            if camera is not None:
                try:
                    camera.stop()
                except Exception as exc:
                    _reader_log(f"dxcam.stop_error {type(exc).__name__}: {exc}")
            self._dxcam_region = None

        def _wgc_camera(self) -> object | None:
            camera = getattr(self, "_wgc_instance", "uninitialized")
            if camera != "uninitialized":
                return camera
            try:
                from manga_rt_translator.capture.windows_wgc import (
                    WindowsGraphicsCapture,
                )

                camera = WindowsGraphicsCapture(
                    window_handle=int(self.winId()),
                )
                self._wgc_instance = camera
                camera.start()
                _reader_log(
                    f"capture.wgc_start hwnd=0x{camera.window_handle:X}"
                )
                return camera
            except Exception as exc:
                self._wgc_instance = None
                _reader_log(f"capture.wgc_unavailable {type(exc).__name__}: {exc}")
                return None

        def _stop_wgc(self) -> None:
            camera = getattr(self, "_wgc_instance", None)
            if camera is not None:
                try:
                    camera.stop()
                except Exception as exc:
                    _reader_log(f"capture.wgc_stop_error {type(exc).__name__}: {exc}")
            self._wgc_instance = None

        def _grab_wgc(self) -> tuple[str, np.ndarray | None]:
            camera = self._wgc_camera()
            if camera is None:
                return "unavailable", None
            status, frame = camera.read_latest()
            if status != "ok" or frame is None:
                if status == "unavailable" and getattr(camera, "last_error", ""):
                    _reader_log(f"capture.wgc_error {camera.last_error}")
                return status, None

            rgb = frame.image
            client_width = max(1, self.width())
            client_height = max(1, self.height())
            outer = self.frameGeometry()
            outer_width = max(1, outer.width())
            outer_height = max(1, outer.height())
            dpr = self.devicePixelRatioF() or 1.0
            client_error = abs(rgb.shape[1] - client_width * dpr) + abs(
                rgb.shape[0] - client_height * dpr
            )
            outer_error = abs(rgb.shape[1] - outer_width * dpr) + abs(
                rgb.shape[0] - outer_height * dpr
            )
            if outer_error < client_error:
                scale_x = rgb.shape[1] / outer_width
                scale_y = rgb.shape[0] / outer_height
                origin = self.webview.mapToGlobal(QPoint(0, 0)) - outer.topLeft()
            else:
                scale_x = rgb.shape[1] / client_width
                scale_y = rgb.shape[0] / client_height
                origin = self.webview.mapTo(self, QPoint(0, 0))
            if scale_x <= 0 or scale_y <= 0 or abs(scale_x - scale_y) > 0.05:
                _reader_log(
                    "capture.wgc_size_mismatch "
                    f"frame={rgb.shape[1]}x{rgb.shape[0]} "
                    f"view={self.webview.width()}x{self.webview.height()}"
                )
                return "unavailable", None
            view_x = round(origin.x() * scale_x)
            view_y = round(origin.y() * scale_y)
            view_x2 = round((origin.x() + self.webview.width()) * scale_x)
            view_y2 = round((origin.y() + self.webview.height()) * scale_y)
            if (
                view_x < 0
                or view_y < 0
                or view_x2 > rgb.shape[1]
                or view_y2 > rgb.shape[0]
            ):
                return "unavailable", None
            rgb = rgb[view_y:view_y2, view_x:view_x2]
            if self.selected_region is not None:
                region = self.selected_region
                x1 = round(region.x * scale_x)
                y1 = round(region.y * scale_y)
                x2 = round((region.x + region.width) * scale_x)
                y2 = round((region.y + region.height) * scale_y)
                if x1 < 0 or y1 < 0 or x2 > rgb.shape[1] or y2 > rgb.shape[0]:
                    return "unavailable", None
                rgb = rgb[y1:y2, x1:x2]
            if rgb.size == 0:
                return "unavailable", None
            self.overlay_widget.set_capture_scale((scale_x + scale_y) / 2)
            return "ok", rgb

        def _exclude_overlay_from_capture(self) -> bool:
            if not self.overlay_widget.isVisible():
                return True
            try:
                hwnd = int(self.overlay_widget.winId())
                if hwnd == self._overlay_capture_hwnd and self._overlay_capture_attempted:
                    return self._overlay_capture_excluded
                (
                    self._overlay_capture_excluded,
                    self._overlay_capture_method,
                    self._overlay_capture_error,
                ) = exclude_window_from_dxgi_capture(hwnd)
                self._overlay_capture_hwnd = hwnd
                self._overlay_capture_attempted = True
                _reader_log(
                    "capture.overlay_exclusion="
                    f"{self._overlay_capture_excluded} "
                    f"method={self._overlay_capture_method} "
                    f"error={self._overlay_capture_error} hwnd=0x{hwnd:X}"
                )
                return self._overlay_capture_excluded
            except Exception as exc:
                _reader_log(f"capture.overlay_exclusion_error {type(exc).__name__}: {exc}")
                return False

        def _grab_dxcam(self) -> tuple[str, np.ndarray | None]:
            """Returns (status, frame): ok / nochange / unavailable."""
            cam = self._dxcam_camera()
            if cam is None:
                return "unavailable", None
            try:
                # Only capture while the reader is the foreground window:
                # DXcam reads the physical screen, never a hidden WebView image.
                import ctypes

                hwnd = int(self.winId())
                user32 = ctypes.WinDLL("user32", use_last_error=True)
                get_foreground = user32.GetForegroundWindow
                get_foreground.restype = ctypes.c_void_p
                if get_foreground() != hwnd:
                    self._stop_dxcam()
                    return "unavailable", None
                screen = self.windowHandle().screen()
                if screen is None:
                    return "unavailable", None
                from manga_rt_translator.capture.windows_windows import (
                    monitor_index_for_device_name,
                )

                output_idx = monitor_index_for_device_name(screen.name())
                if not self._exclude_overlay_from_capture():
                    self._stop_dxcam()
                    return "unavailable", None
                origin = self.webview.mapToGlobal(QPoint(0, 0))
                dpr = self.devicePixelRatioF() or 1.0
                screen_geo = screen.geometry()
                left = int((origin.x() - screen_geo.x()) * dpr)
                top = int((origin.y() - screen_geo.y()) * dpr)
                width = int(self.webview.width() * dpr)
                height = int(self.webview.height() * dpr)
                x, y, w, h = left, top, width, height
                if self.selected_region is not None:
                    region = self.selected_region
                    x = left + int(region.x * dpr)
                    y = top + int(region.y * dpr)
                    w = int(region.width * dpr)
                    h = int(region.height * dpr)
                # DXGI coordinates are relative to the selected monitor.
                out_w = int(screen_geo.width() * dpr)
                out_h = int(screen_geo.height() * dpr)
                if x < 0 or y < 0 or x + w > out_w or y + h > out_h:
                    return "unavailable", None
                if w <= 0 or h <= 0:
                    return "unavailable", None
                capture_region = (x, y, x + w, y + h)
                cam.set_region(capture_region, output_idx=output_idx)
                self._dxcam_region = capture_region
                frame = cam.read_latest()
                if frame is None:
                    if getattr(cam, "last_error", ""):
                        return "unavailable", None
                    return "nochange", None
                self.overlay_widget.set_capture_scale(dpr)
                return "ok", frame.image
            except Exception as exc:
                now = time.time()
                if now - getattr(self, "_last_dxcam_err", 0.0) > 30:
                    self._last_dxcam_err = now
                    _reader_log(f"dxcam.error {type(exc).__name__}: {exc}")
                return "unavailable", None

        def _grab_reader_rgb(self) -> np.ndarray | None:
            if not self.webview.isVisible():
                return None
            # WGC follows the WebView HWND and is independent of foreground
            # focus; its event-driven frames do not include the separate overlay.
            wgc_status, wgc_frame = self._grab_wgc()
            if wgc_status == "ok":
                return wgc_frame
            if wgc_status in {"starting", "nochange"}:
                return None
            # DXGI remains a streaming fallback when WGC is unavailable.
            status, frame = self._grab_dxcam()
            if status == "ok":
                return frame
            if status == "nochange":
                return None
            # dxcam unavailable: fall back to rendering the widget. With a
            # reading region selected, render only that sub-rect.
            self.overlay_widget.set_capture_scale(1.0)
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

        def _capture_reader_frame(
            self,
            force_analyze: bool = False,
            dirty_rects: tuple[Rect, ...] = (),
            dirty_rects_stable: bool = False,
        ) -> bool:
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
                and not dirty_rects_stable
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
            self._last_capture_size = (rgb.shape[1], rgb.shape[0])
            self.runtime.push_frame(
                rgb,
                force_analyze=force_analyze,
                dirty_rects=dirty_rects,
                dirty_rects_stable=dirty_rects_stable,
            )
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
    resolved_glossary_path = resolve_glossary_path(glossary_path)
    launch = ReaderLaunchContext(
        config=load_config(config_path),
        glossary=Glossary.load(resolved_glossary_path),
        initial_url=initial_url or "https://mangadex.org",
        glossary_path=str(resolved_glossary_path) if resolved_glossary_path else None,
    )
    from manga_rt_translator.ocr import warmup_ocr

    warmup_config = copy.deepcopy(launch.config.ocr)
    if warmup_config.source_lang != "japan" and "japan" not in warmup_config.extra_langs:
        warmup_config.extra_langs.append("japan")
    warmup_ocr(warmup_config)
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


def _intersect_paint_rect(left: PaintRect, right: PaintRect) -> PaintRect | None:
    x = max(left.x, right.x)
    y = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    if x2 <= x or y2 <= y:
        return None
    return PaintRect(x, y, x2 - x, y2 - y)


def _paint_rect_contains(bounds: PaintRect, rect: PaintRect) -> bool:
    return (
        bounds.x <= rect.x
        and bounds.y <= rect.y
        and bounds.x + bounds.width >= rect.x + rect.width
        and bounds.y + bounds.height >= rect.y + rect.height
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


def _scale_capture_rect(rect: Rect, scale: float) -> Rect:
    if scale <= 1.0:
        return rect
    return Rect(
        x=round(rect.x / scale),
        y=round(rect.y / scale),
        width=max(1, round(rect.width / scale)),
        height=max(1, round(rect.height / scale)),
    )


def _overlay_item_key(item: OverlayItem) -> str:
    if item.visual_hash:
        return f"visual:{item.visual_hash}"
    source = " ".join(item.source_text.casefold().split())
    if source:
        return f"source:{source}:{item.bbox.x // 24}:{item.bbox.y // 24}"
    return f"box:{item.bbox.as_tuple()}:{item.text}"


def _create_translator(config: AppConfig, echo_translator: bool) -> Translator:
    return create_translator(config, echo=echo_translator)
