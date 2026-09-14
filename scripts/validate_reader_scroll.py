"""Exercise real Qt capture, OCR, scroll polling and overlay with one selection."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('QT_OPENGL', 'angle')

from PySide6.QtCore import QTimer
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QApplication

from manga_rt_translator import reader
from manga_rt_translator.config import AppConfig
from manga_rt_translator.models import Rect
from manga_rt_translator.translator import TranslationResult, Translator
import manga_rt_translator.reader_runtime as runtime_module
import manga_rt_translator.ocr as ocr_module


class FixtureTranslator(Translator):
    def translate_batch(self, texts, source_lang, target_lang, glossary, context=''):
        assert target_lang == 'pt-BR'
        return [TranslationResult('Chefe!' if 'CHIEF' in t else 'Texto traduzido para teste.',
                                  'fixture') for t in texts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--live-translation', action='store_true')
    parser.add_argument('--detector', choices=['paddle', 'hybrid'], default='paddle')
    parser.add_argument('--scale-factor', type=float, default=1.0)
    args = parser.parse_args()
    os.environ['QT_SCALE_FACTOR'] = str(args.scale_factor)
    root = Path(__file__).resolve().parents[1]
    output = root / 'reader_debug_output' / f'qa-{int(time.time())}'
    output.mkdir(parents=True)
    capture = root / 'reader_debug_output/last_capture.png'
    page = output / 'fixture.html'
    page.write_text('<!doctype html><html><body style="margin:0;background:#191a1b">'
                    + f'<img style="display:block" src="{capture.as_uri()}" width="977" height="861">'*3
                    + '</body></html>', encoding='utf-8')
    config = AppConfig()
    config.ocr.detector = args.detector
    config.cache.path = str(output / 'memory.sqlite')
    reader.load_config = lambda _: config
    ocr_module.warmup_ocr = lambda _: None
    if not args.live_translation:
        runtime_module.create_translator = lambda config, echo: FixtureTranslator()
    app = QApplication.instance() or QApplication([])
    state = {'stage': 'loading', 'selection_count': 0, 'results': [],
             'live_translation': args.live_translation, 'detector': args.detector,
             'scale_factor': args.scale_factor,
             'success': False}
    started = time.monotonic()
    timer = QTimer()
    worker = None

    def finish(error=''):
        nonlocal worker
        timer.stop()
        state['error'] = error
        window = reader._ACTIVE_WINDOW
        if window and window.runtime:
            worker = window.runtime.worker
        if window:
            window.close()
        (output / 'result.json').write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(state, ensure_ascii=True), flush=True)
        QTimer.singleShot(100, app.quit)

    def snapshot(window, name):
        base = window.webview.grab()
        overlay = window.overlay_widget.grab()
        painter = QPainter(base)
        painter.drawPixmap(0, 0, overlay)
        painter.end()
        base.save(str(output / f'{name}.png'))

    def step():
        try:
            window = reader._ACTIVE_WINDOW
            if time.monotonic()-started > 100:
                return finish('Timed out waiting for translated overlay')
            if window is None or not window.runtime or not window.runtime.pipeline:
                return
            pipeline = window.runtime.pipeline
            with pipeline.view.lock:
                items = list(pipeline.active_items)
            translated = [item for item in items if item.text != item.source_text]
            expected = 'HAS THERE' if state['stage'] == 'scrolled' else 'CHIEF'
            if not any(expected in item.source_text.upper() for item in translated):
                return
            assert window.selected_region == Rect(50, 260, 740, 430)
            assert window.overlay_widget.isVisible(), 'Overlay hidden'
            assert window.runtime.last_capture_shape == '740x430', 'DPI coordinates changed'
            state['results'].append(dict(stage=state['stage'], elapsed_s=round(time.monotonic()-started, 2),
                                         captures=window.runtime.frames_captured,
                                         ocr=pipeline.stats.ocr_regions,
                                         texts=[dict(source=i.source_text, translated=i.text) for i in translated]))
            snapshot(window, state['stage'])
            if state['stage'] == 'first':
                state['stage'] = 'scrolled'
                window.webview.page().runJavaScript('window.scrollTo(0,400)')
            elif state['stage'] == 'scrolled':
                state['stage'] = 'returned'
                window.webview.page().runJavaScript('window.scrollTo(0,0)')
            elif state['stage'] == 'returned':
                # Invalidate at an intermediate scroll then return before debounce fires.
                state['stage'] = 'rapid-return'
                window.webview.page().runJavaScript('window.scrollTo(0,400)')
                timer.stop()
                QTimer.singleShot(450, lambda: window.webview.page().runJavaScript('window.scrollTo(0,0)'))
                QTimer.singleShot(900, timer.start)
            elif state['stage'] == 'rapid-return':
                state['success'] = True
                finish()
        except Exception as exc:
            finish(f'{type(exc).__name__}: {exc}')

    def select():
        window = reader._ACTIVE_WINDOW
        window.source_combo.setCurrentText('en')
        window.mode_combo.setCurrentIndex(0)
        window._handle_region_selected(Rect(50, 260, 740, 430))
        state['selection_count'] += 1
        state['stage'] = 'first'
        timer.start(150)

    def ready(ok):
        if not ok:
            finish('Fixture failed to load')
            return
        QTimer.singleShot(350, select)

    def bind():
        window = reader._ACTIVE_WINDOW
        window.resize(1150, 920)
        window.activateWindow()
        window.webview.loadFinished.connect(ready)

    timer.timeout.connect(step)
    QTimer.singleShot(0, bind)
    QTimer.singleShot(110000, lambda: finish('Test deadline exceeded'))
    reader.run_reader_app(None, None, page.as_uri())
    if worker:
        worker.join(30)
    if not state['success']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
