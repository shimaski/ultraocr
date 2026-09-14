from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from manga_rt_translator.cache import TranslationMemory
from manga_rt_translator.config import AppConfig
from manga_rt_translator.diff import FrameDiffer
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.models import Frame
from manga_rt_translator.ocr import create_ocr_engine, lower_thread_priority
from manga_rt_translator.pipeline import MangaTranslationPipeline
from manga_rt_translator.text_regions import TextRegionDetector
from manga_rt_translator.translator import create_translator
from manga_rt_translator.work_queue import LatestFrameQueue, ViewState, WorkCancelled


def reader_log(message: str) -> None:
    try:
        with Path('.reader_debug.log').open('a', encoding='utf-8') as handle:
            handle.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {message}\n')
        print(f'[reader] {message}', flush=True)
    except OSError:
        pass


class ReaderRuntime:
    def __init__(self, config: AppConfig, glossary: Glossary, overlay: object,
                 status_callback: Callable[[str], None], echo_translator: bool) -> None:
        self.config = config
        self.glossary = glossary
        self.overlay = overlay
        self.status_callback = status_callback
        self.echo_translator = echo_translator
        self.view = getattr(overlay, 'view', None) or ViewState()
        self.frame_queue = LatestFrameQueue(self.view)
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.pipeline: MangaTranslationPipeline | None = None
        self.last_error = ''
        self.frames_captured = 0
        self.frames_submitted = 0
        self.last_capture_shape = ''
        self.analysis_mode = 'Manual'
        self.phase = 'Preparando OCR...'

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        reader_log('runtime.start')
        self.worker = threading.Thread(target=self._run, daemon=True, name='reader-ocr')
        self.worker.start()

    def stop(self, join_timeout: float = 0.5) -> None:
        self.stop_event.set()
        self.view.close()
        self.overlay.stop()
        if self.worker:
            self.worker.join(timeout=join_timeout)

    def invalidate_view(self) -> None:
        with self.view.lock:
            self.frame_queue.invalidate()
            if self.pipeline:
                self.pipeline.discard_obsolete()
            else:
                self.overlay.update([])
        self.phase = 'Aguardando a rolagem parar...'

    def set_analysis_mode(self, mode: str) -> None:
        self.analysis_mode = mode

    def push_frame(self, image: np.ndarray, *, force_analyze: bool = False,
                   inspect: bool = False) -> None:
        if self.stop_event.is_set() or not self.worker or not self.worker.is_alive():
            return
        # The request and its pixels travel together; no separate manual flag
        # can wake the worker against the previous frame.
        frame = Frame(image.copy(), time.time(), self.frames_submitted)
        self.frames_captured += 1
        self.frames_submitted += 1
        self.last_capture_shape = f'{image.shape[1]}x{image.shape[0]}'
        with self.view.lock:
            self.frame_queue.put(frame, force=force_analyze or inspect, inspect=inspect)
            if (force_analyze or inspect) and self.pipeline:
                self.pipeline.discard_obsolete()
        self.phase = 'Lendo o conteudo atual...'
        reader_log(f'capture.queued frame={frame.index} force={force_analyze} size={self.last_capture_shape}')

    def stats_text(self) -> str:
        if self.last_error:
            return self.last_error
        if not self.pipeline:
            return self.phase
        stats = self.pipeline.stats
        if stats.last_error:
            return stats.last_error
        with self.view.lock:
            items = list(self.pipeline.active_items)
            pending = any(not future.done() for future in self.pipeline._futures)
        translated = sum(item.text != item.source_text for item in items)
        phase = 'Traduzindo...' if pending else self.phase
        return (f'{phase} | traduzidos={translated} ocr={stats.ocr_regions} '
                f'cache={stats.cache_hits} api={stats.translate_calls} '
                f'descartados={self.frame_queue.dropped + stats.stale_results}')

    def _run(self) -> None:
        lower_thread_priority()
        memory = None
        try:
            memory = TranslationMemory(self.config.cache.path,
                                       fuzzy_threshold=self.config.translation.fuzzy_threshold)
            self.pipeline = MangaTranslationPipeline(
                config=self.config, stream=None, differ=FrameDiffer(self.config.diff),
                detector=TextRegionDetector(ignore_top_pixels=0),
                ocr=create_ocr_engine(self.config.ocr), memory=memory,
                glossary=self.glossary, translator=create_translator(self.config, echo=self.echo_translator),
                overlay=self.overlay, async_translate=True, view=self.view,
            )
            reader_log('pipeline.ready')
            while not self.stop_event.is_set():
                try:
                    request = self.frame_queue.get()
                except queue.Empty:
                    continue
                if not request.token.current():
                    continue
                try:
                    before = self.pipeline.stats.ocr_regions
                    started = time.perf_counter()
                    if request.inspect:
                        self._inspect(request)
                    elif request.force or self.analysis_mode != 'Manual':
                        self.pipeline.process_frame(request.frame, force=request.force,
                                                    token=request.token)
                    request.token.check()
                    count = self.pipeline.stats.ocr_regions - before
                    self.phase = ('Aguardando novo conteudo' if count else 'Nenhum texto novo encontrado')
                    reader_log(f'analysis.done frame={request.frame.index} regions={count} '
                               f'elapsed_ms={(time.perf_counter()-started)*1000:.0f}')
                    request.token.commit(lambda: self.status_callback(self.stats_text()))
                except WorkCancelled:
                    reader_log(f'analysis.obsolete frame={request.frame.index}')
                except Exception as exc:
                    reader_log(f'analysis.error {type(exc).__name__}: {exc}')
                    request.token.commit(lambda: setattr(self, 'phase', f'Falha na leitura: {exc}'))
        except Exception as exc:
            self.last_error = f'{type(exc).__name__}: {exc}'
            reader_log(f'runtime.error {self.last_error}')
        finally:
            # Only the background worker waits for a running HTTP call before
            # closing SQLite. The GUI's stop operation stays bounded.
            if self.pipeline:
                self.pipeline.close(wait=True)
            if memory:
                memory.close()
            reader_log('runtime.stop')

    def _inspect(self, request: object) -> None:
        import cv2
        from manga_rt_translator.models import Rect
        image = request.frame.image
        full = Rect(0, 0, image.shape[1], image.shape[0])
        regions = self.pipeline._ocr_cached(image, [full], request.token)
        request.token.check()
        directory = Path('reader_debug_output')
        directory.mkdir(exist_ok=True)
        cv2.imwrite(str(directory / 'last_capture.png'), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        (directory / 'last_ocr.txt').write_text('\n'.join(
            f'{r.bbox.as_tuple()} | conf={r.confidence:.2f} | {r.text}' for r in regions
        ), encoding='utf-8')
