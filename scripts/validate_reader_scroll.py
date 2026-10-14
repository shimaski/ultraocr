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
    page = output / 'fixture.html'
    page.write_text('''<!doctype html><html><head><meta charset="utf-8"><style>
      html,body{margin:0;background:#eee;color:#111;font-family:Arial,sans-serif}
      .panel{height:900px;width:977px;position:relative;border-bottom:8px solid #222}
      .bubble{position:absolute;left:145px;width:580px;height:180px;top:300px;
        box-sizing:border-box;border:5px solid #111;border-radius:48%;background:#fff;
        display:flex;align-items:center;justify-content:center;text-align:center;
        font-size:52px;font-weight:700}
      .panel:nth-child(2) .bubble{top:60px;font-size:34px}
      #horizontal{display:none;position:fixed;left:145px;top:360px;width:580px;height:180px;
        overflow-x:auto;overflow-y:hidden;border:5px solid #111;background:#fff;box-sizing:border-box}
      #horizontal-content{position:relative;width:1250px;height:100%;font-size:44px;font-weight:700}
      #horizontal-content span{position:absolute;top:55px;white-space:nowrap}
      #horizontal-left{left:20px} #horizontal-right{left:700px}
    </style></head><body>
      <section class="panel"><div class="bubble">CHIEF!</div></section>
      <section class="panel"><div class="bubble">HAS THERE EVER BEEN A HUMAN WHO ACTUALLY LISTENED TO OUR</div></section>
      <section class="panel"><div class="bubble">CHIEF!</div></section>
      <div id="horizontal"><div id="horizontal-content">
        <span id="horizontal-left">LEFT SIDE</span><span id="horizontal-right">RIGHT SIDE</span>
      </div></div>
    </body></html>''', encoding='utf-8')
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
            if state['stage'] == 'zoom-check':
                zoom_changed = window._last_view_metrics != state['zoom_metrics_before']
                reread = pipeline.stats.ocr_regions > state['ocr_before_zoom']
                if zoom_changed and reread:
                    state['results'].append(dict(
                        stage='zoom-check',
                        elapsed_s=round(time.monotonic()-started, 2),
                        captures=window.runtime.frames_captured,
                        ocr=pipeline.stats.ocr_regions,
                        metrics_changed=True,
                    ))
                    state['success'] = True
                    finish()
                return
            expected = {
                'first': 'CHIEF', 'scrolled': 'HAS THERE',
                'horizontal-left': 'LEFT SIDE', 'horizontal-right': 'RIGHT SIDE',
            }.get(state['stage'], '')
            if not any(expected in item.source_text.upper() for item in translated):
                return
            assert window.selected_region == Rect(50, 260, 740, 430)
            active_window = QApplication.activeWindow()
            state['visibility'] = {
                'reader_visible': window.isVisible(),
                'reader_active': window.isActiveWindow(),
                'overlay_visible': window.overlay_widget.isVisible(),
                'active_window': active_window.windowTitle() if active_window else None,
            }
            if window.isActiveWindow():
                assert window.overlay_widget.isVisible(), f"Overlay hidden: {state['visibility']}"
            else:
                state['focus_note'] = 'OS did not grant foreground focus; overlay visibility is not asserted'
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
                state['stage'] = 'horizontal-left'
                window.webview.page().runJavaScript(
                    "document.getElementById('horizontal').style.display='block'"
                )
            elif state['stage'] == 'horizontal-left':
                state['stage'] = 'horizontal-right'
                window.webview.page().runJavaScript(
                    "document.getElementById('horizontal').scrollLeft=600"
                )
            elif state['stage'] == 'horizontal-right':
                if args.scale_factor != 1.0:
                    state['success'] = True
                    finish()
                    return
                state['stage'] = 'zoom-check'
                state['zoom_metrics_before'] = window._last_view_metrics
                state['ocr_before_zoom'] = pipeline.stats.ocr_regions
                window.webview.setZoomFactor(1.25)
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
        window = reader._ACTIVE_WINDOW
        window.webview.page().runJavaScript(
            'window.scrollTo(0,0)', lambda _: QTimer.singleShot(350, select)
        )

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
