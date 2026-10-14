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
        self._full_refresh_after_scroll = False

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

    def invalidate_view(self, *, clear_overlay: bool = True,
                        scroll_delta_x: int = 0,
                        scroll_delta_y: int = 0,
                        viewport_size: tuple[int, int] | None = None,
                        scrolling: bool = False) -> None:
        with self.view.lock:
            if scrolling:
                interrupted = self.frame_queue.has_work()
                if self.pipeline is not None:
                    interrupted = interrupted or self.pipeline.has_pending_translations
                self._full_refresh_after_scroll |= interrupted
            self.frame_queue.invalidate()
            if self.pipeline:
                self.pipeline.discard_obsolete(clear_items=clear_overlay)
                if (scroll_delta_x or scroll_delta_y) and viewport_size:
                    self.pipeline.shift_viewport_items(
                        scroll_delta_y, viewport_size[0], viewport_size[1],
                        dx=scroll_delta_x,
                    )
            else:
                if clear_overlay:
                    self.overlay.update([])
        self.phase = 'Aguardando a rolagem parar...'

    @property
    def needs_full_refresh_after_scroll(self) -> bool:
        with self.view.lock:
            return self._full_refresh_after_scroll

    def mark_scroll_refresh_submitted(self) -> None:
        with self.view.lock:
            self._full_refresh_after_scroll = False

    def set_analysis_mode(self, mode: str) -> None:
        self.analysis_mode = mode

    def replace_translation_config(self, config: AppConfig, *,
                                   echo_translator: bool) -> None:
        translator = create_translator(config, echo=echo_translator)
        if not echo_translator and config.translation.provider == 'argos':
            ready, message = translator.check_health()
            if not ready:
                raise RuntimeError(message)
        with self.view.lock:
            self.frame_queue.invalidate()
            self.config = config
            self.echo_translator = echo_translator
            if self.pipeline is not None:
                self.pipeline.discard_obsolete(clear_items=False)
                self.pipeline.config = config
                self.pipeline.replace_translator(translator)

    def push_frame(self, image: np.ndarray, *, force_analyze: bool = False,
                   inspect: bool = False, dirty_rects=(),
                   dirty_rects_stable: bool = False) -> None:
        if self.stop_event.is_set() or not self.worker or not self.worker.is_alive():
            return
        # The request and its pixels travel together; no separate manual flag
        # can wake the worker against the previous frame.
        frame = Frame(
            image.copy(), time.time(), self.frames_submitted,
            dirty_rects=tuple(dirty_rects),
            dirty_rects_stable=dirty_rects_stable,
        )
        self.frames_captured += 1
        self.frames_submitted += 1
        self.last_capture_shape = f'{image.shape[1]}x{image.shape[0]}'
        with self.view.lock:
            incremental = dirty_rects_stable and bool(dirty_rects)
            prioritized = force_analyze or inspect or incremental
            self.frame_queue.put(
                frame,
                force=prioritized,
                inspect=inspect,
                incremental=incremental,
            )
            if prioritized and self.pipeline:
                # A forced frame represents a complete viewport generation.
                # Preserve the old translated generation until the new one is
                # committed atomically by the pipeline.
                self.pipeline.discard_obsolete(clear_items=False)
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
                f'trad={stats.last_translation_ms:.0f}ms/{stats.last_translated_regions} '
                f'descartados={self.frame_queue.dropped + stats.stale_results}')

    def _run(self) -> None:
        lower_thread_priority()
        memory = None
        try:
            memory = TranslationMemory(self.config.cache.path,
                                       fuzzy_threshold=self.config.translation.fuzzy_threshold)
            with self.view.lock:
                config = self.config
                translator = create_translator(config, echo=self.echo_translator)
                self.pipeline = MangaTranslationPipeline(
                    config=config, stream=None, differ=FrameDiffer(config.diff),
                    detector=TextRegionDetector(ignore_top_pixels=0),
                    ocr=create_ocr_engine(config.ocr), memory=memory,
                    glossary=self.glossary, translator=translator,
                    overlay=self.overlay, async_translate=True, view=self.view,
                )
            reader_log('pipeline.ready')
            while not self.stop_event.is_set():
                try:
                    request = self.frame_queue.get()
                except queue.Empty:
                    continue
                try:
                    if not request.token.current():
                        continue
                    before = self.pipeline.stats.ocr_regions
                    started = time.perf_counter()
                    if request.inspect:
                        self._inspect(request)
                    elif request.incremental:
                        self.pipeline.process_frame(
                            request.frame, force=False, token=request.token
                        )
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
                finally:
                    self.frame_queue.complete(request)
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
