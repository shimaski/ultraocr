from __future__ import annotations

import threading
import traceback

from manga_rt_translator.cache import TranslationMemory
from manga_rt_translator.capture import create_screen_stream
from manga_rt_translator.config import AppConfig
from manga_rt_translator.diff import FrameDiffer
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.ocr import create_ocr_engine
from manga_rt_translator.overlay import Overlay, create_overlay
from manga_rt_translator.pipeline import MangaTranslationPipeline, PipelineStats
from manga_rt_translator.text_regions import TextRegionDetector
from manga_rt_translator.translator import Translator, create_translator


class NullOverlay(Overlay):
    def update(self, items: object) -> None:
        return None

    def run_forever(self) -> int:
        return 0

    def stop(self) -> None:
        return None


class TranslationSession:
    def __init__(
        self,
        config: AppConfig,
        glossary: Glossary,
        echo_translator: bool = False,
        overlay_enabled: bool = True,
    ) -> None:
        self.config = config
        self.glossary = glossary
        self.echo_translator = echo_translator
        self.overlay_enabled = overlay_enabled
        self.stop_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.last_error: str = ""
        self.overlay: Overlay = NullOverlay()
        self.pipeline: MangaTranslationPipeline | None = None
        self.stats = PipelineStats()
        self._memory: TranslationMemory | None = None
        self._stream = None

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        self.stop_event.clear()
        self.last_error = ""
        self.overlay = create_overlay(self.config.overlay, enabled=self.overlay_enabled)
        self.worker = threading.Thread(target=self._run_pipeline, daemon=True)
        self.worker.start()

    def stop(self, join_timeout: float = 2.0) -> None:
        self.stop_event.set()
        self.overlay.stop()
        if self._stream is not None:
            self._stream.stop()
        if self.worker:
            self.worker.join(timeout=join_timeout)
        self._memory = None

    def readiness(self) -> tuple[bool, str]:
        translator = _create_translator(self.config, self.echo_translator)
        checker = getattr(translator, "check_health", None)
        if callable(checker):
            return checker()
        return True, "Translator ready"

    def _run_pipeline(self) -> None:
        try:
            self._memory = TranslationMemory(
                self.config.cache.path,
                fuzzy_threshold=self.config.translation.fuzzy_threshold,
            )
            self._stream = create_screen_stream(self.config.capture)
            differ = FrameDiffer(self.config.diff)
            detector = TextRegionDetector(
                ignore_top_pixels=self.config.ocr.ignore_top_pixels,
            )
            ocr = create_ocr_engine(self.config.ocr)
            translator = _create_translator(self.config, self.echo_translator)
            self.pipeline = MangaTranslationPipeline(
                config=self.config,
                stream=self._stream,
                differ=differ,
                detector=detector,
                ocr=ocr,
                memory=self._memory,
                glossary=self.glossary,
                translator=translator,
                overlay=self.overlay,
            )
            self.stats = self.pipeline.stats
            self.pipeline.run(self.stop_event)
        except Exception as exc:
            self.last_error = "".join(
                traceback.format_exception_only(type(exc), exc)
            ).strip()
            self.stats.last_error = self.last_error
        finally:
            if self._memory is not None:
                self._memory.close()


def _create_translator(config: AppConfig, echo_translator: bool) -> Translator:
    return create_translator(config, echo=echo_translator)
