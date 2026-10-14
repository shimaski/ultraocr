from __future__ import annotations

import time
import threading

import numpy as np

from manga_rt_translator.cache import TranslationMemory
from manga_rt_translator.config import load_config
from manga_rt_translator.diff import FrameDiffer
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.models import CaptureViewport, Frame, OverlayItem, Rect, TextRegion
from manga_rt_translator.overlay import Overlay
from manga_rt_translator.pipeline import MangaTranslationPipeline
from manga_rt_translator.translator import EchoTranslator, TranslationResult, Translator


class FixedTranslator(Translator):
    """Returns a real pt-BR translation so visual-cache hits are accepted."""

    def translate(self, text, source_lang, target_lang, glossary, context=""):
        return TranslationResult(text="Ola aventura", provider="stub")


class BatchCountingTranslator(Translator):
    def __init__(self) -> None:
        self.batch_calls = 0
        self.single_calls = 0

    def translate(self, text, source_lang, target_lang, glossary, context=""):
        self.single_calls += 1
        return TranslationResult(text=f"trad: {text}", provider="stub")

    def translate_batch(self, texts, source_lang, target_lang, glossary, context=""):
        self.batch_calls += 1
        return [TranslationResult(text=f"trad: {t}", provider="stub") for t in texts]


class EchoBatchTranslator(Translator):
    def translate(self, text, source_lang, target_lang, glossary, context=""):
        return TranslationResult(text=text, provider="echo")

    def translate_batch(self, texts, source_lang, target_lang, glossary, context=""):
        return [TranslationResult(text=text, provider="echo") for text in texts]


class TwoBubbleOcr:
    def recognize(self, image, regions):
        return [
            TextRegion(
                bbox=Rect(x=100, y=80, width=120, height=40),
                text="Hello there",
                confidence=0.9,
                source_lang="en",
                visual_hash="hash-a",
            ),
            TextRegion(
                bbox=Rect(x=560, y=380, width=120, height=40),
                text="General Kenobi",
                confidence=0.9,
                source_lang="en",
                visual_hash="hash-b",
            ),
        ]


class NullStream:
    def start(self) -> None:
        return None

    def frames(self):
        return iter(())

    def stop(self) -> None:
        return None


class StubDetector:
    def detect(self, image, rois):
        return list(rois)


class StubOcr:
    def recognize(self, image, regions):
        return [
            TextRegion(
                bbox=Rect(x=100, y=80, width=120, height=40),
                text="Hello adventure",
                confidence=0.9,
                source_lang="en",
                visual_hash="stub-visual-hash",
            )
        ]


class StubOverlay(Overlay):
    def __init__(self) -> None:
        self.items: list[OverlayItem] = []
        self.viewports: list[CaptureViewport | None] = []

    def set_capture_viewport(self, viewport):
        self.viewports.append(viewport)

    def update(self, items) -> None:
        self.items = list(items)

    def run_forever(self) -> int:
        return 0

    def stop(self) -> None:
        return None


def _build_pipeline(tmp_path, translator=None, ocr=None):
    config = load_config(None)
    memory = TranslationMemory(str(tmp_path / "memory.sqlite3"))
    overlay = StubOverlay()
    pipeline = MangaTranslationPipeline(
        config=config,
        stream=NullStream(),
        differ=FrameDiffer(config.diff),
        detector=StubDetector(),
        ocr=ocr or StubOcr(),
        memory=memory,
        glossary=Glossary.default(),
        translator=translator or FixedTranslator(),
        overlay=overlay,
    )
    return config, pipeline, overlay, memory


def test_pipeline_translates_stable_region_and_caches_visual_hits(tmp_path):
    config, pipeline, overlay, memory = _build_pipeline(tmp_path)
    image = np.full((300, 400, 3), 240, dtype=np.uint8)
    bubble = Rect(x=100, y=80, width=120, height=40)
    stable_seconds = config.diff.stable_ms / 1000.0 + 0.05

    # First frame only seeds the differ baseline; nothing is ready yet.
    pipeline.process_frame(Frame(image=image, timestamp=1.0, index=0, dirty_rects=(bubble,)))
    assert overlay.items == []

    # Same image after the stable window releases the pending ROI for OCR.
    items = pipeline.process_frame(
        Frame(image=image.copy(), timestamp=1.0 + stable_seconds, index=1)
    )
    assert len(items) == 1
    assert items[0].text == "Ola aventura"
    assert pipeline.stats.translate_calls == 1
    assert pipeline.stats.ocr_regions == 1
    assert len(memory.history()) == 1
    assert memory.history()[0].provider == "stub"

    # A new changed region that is visually identical to the previous one
    # must be served from the visual cache instead of translating again.
    # The previous overlay item stays visible until the region is re-analyzed.
    pipeline.process_frame(
        Frame(image=image, timestamp=3.0, index=2, dirty_rects=(Rect(x=110, y=90, width=40, height=30),))
    )
    assert len(overlay.items) == 1

    items = pipeline.process_frame(Frame(image=image.copy(), timestamp=3.0 + stable_seconds, index=3))
    assert len(items) == 1
    assert items[0].text == "Ola aventura"
    assert pipeline.stats.translate_calls == 1
    assert pipeline.stats.cache_hits == 1

    memory.close()


def test_pipeline_propagates_capture_viewport_to_overlay(tmp_path):
    _config, pipeline, overlay, memory = _build_pipeline(tmp_path)
    viewport = CaptureViewport(
        capture_origin=(147, 93),
        capture_size=(600, 400),
        monitor_origin=(0, 0),
        monitor_device_name=r"\\.\DISPLAY1",
        monitor_index=0,
    )

    pipeline.process_frame(
        Frame(
            image=np.zeros((400, 600, 3), dtype=np.uint8),
            timestamp=1.0,
            index=0,
            capture_viewport=viewport,
        )
    )

    assert overlay.viewports == [viewport]
    pipeline.close()
    memory.close()


def test_lost_capture_clears_existing_overlay_without_running_ocr(tmp_path):
    _config, pipeline, overlay, memory = _build_pipeline(tmp_path)
    stale = OverlayItem(
        bbox=Rect(20, 30, 80, 30),
        text="Tradução antiga",
        source_text="Old translation",
    )
    pipeline.active_items = [stale]
    overlay.items = [stale]

    result = pipeline.process_frame(
        Frame(
            image=np.empty((0, 0, 3), dtype=np.uint8),
            timestamp=2.0,
            index=1,
            capture_lost=True,
        )
    )

    assert result == []
    assert pipeline.active_items == []
    assert overlay.items == []
    assert pipeline.stats.frames_seen == 0
    memory.close()


def test_work_scoped_glossary_partitions_textual_and_visual_memory(tmp_path):
    _config, pipeline, _overlay, memory = _build_pipeline(tmp_path)
    pipeline.glossary.cache_scope = "work:series-1"
    pair = pipeline._language_pair("en", "pt-BR")

    assert pair == "work:series-1|en->pt-BR"
    memory.save("Hello there", "Olá", pair)
    memory.save_visual("bubble-1", "Hello there", "Olá", pair, (1, 2, 20, 10))
    assert memory.lookup("Hello there", pair).translated_text == "Olá"
    assert memory.lookup("Hello there", "en->pt-BR") is None
    assert memory.get_visual("bubble-1", pair).translated_text == "Olá"
    assert memory.get_visual("bubble-1", "en->pt-BR") is None
    pipeline.close()
    memory.close()


def test_scroll_like_change_clears_stale_overlay(tmp_path):
    class ScrollTickDiffer(FrameDiffer):
        def __init__(self, config):
            super().__init__(config)
            self.force_large_change = False

        def update(self, frame):
            rois = super().update(frame)
            if self.force_large_change:
                self.last_tick_had_large_change = True
                return []
            return rois

    config = load_config(None)
    memory = TranslationMemory(str(tmp_path / "memory_scroll.sqlite3"))
    overlay = StubOverlay()
    pipeline = MangaTranslationPipeline(
        config=config,
        stream=NullStream(),
        differ=ScrollTickDiffer(config.diff),
        detector=StubDetector(),
        ocr=StubOcr(),
        memory=memory,
        glossary=Glossary.default(),
        translator=FixedTranslator(),
        overlay=overlay,
    )
    image = np.full((300, 400, 3), 240, dtype=np.uint8)
    bubble = Rect(x=100, y=80, width=120, height=40)
    stable_seconds = config.diff.stable_ms / 1000.0 + 0.05

    pipeline.process_frame(Frame(image=image, timestamp=1.0, index=0, dirty_rects=(bubble,)))
    pipeline.process_frame(Frame(image=image.copy(), timestamp=1.0 + stable_seconds, index=1))
    assert len(overlay.items) == 1

    # The user scrolls: the differ flags a scroll-like change and the stale
    # overlay must fade out quickly (short TTL) — not persist for the full
    # TTL nor vanish instantly mid-analysis.
    pipeline.differ.force_large_change = True
    items = pipeline.process_frame(Frame(image=image.copy(), timestamp=2.0, index=2))
    assert len(items) == 1
    assert items[0].expires_at is not None
    assert items[0].expires_at - time.time() <= 1.3
    memory.close()


class CountingOcr:
    """Same contract as StubOcr but counts model invocations."""

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image, regions):
        self.calls += 1
        return [
            TextRegion(
                bbox=Rect(x=100, y=80, width=120, height=40),
                text="Hello adventure",
                confidence=0.9,
                source_lang="en",
                visual_hash="stub-visual-hash",
            )
        ]


def test_roi_cache_skips_repeated_ocr_of_identical_content(tmp_path):
    ocr = CountingOcr()
    translator = FixedTranslator()
    config, pipeline, overlay, memory = _build_pipeline(
        tmp_path, translator=translator, ocr=ocr
    )
    image = np.full((300, 400, 3), 240, dtype=np.uint8)
    bubble = Rect(x=100, y=80, width=120, height=40)
    stable_seconds = config.diff.stable_ms / 1000.0 + 0.05

    pipeline.process_frame(Frame(image=image, timestamp=1.0, index=0, dirty_rects=(bubble,)))
    pipeline.process_frame(Frame(image=image.copy(), timestamp=1.0 + stable_seconds, index=1))
    assert ocr.calls == 1
    assert len(overlay.items) == 1

    # Initial analysis covers the full frame; a smaller ROI of the same flat
    # color is NOT the same OCR input. Dimensions must be part of the key.
    pipeline.process_frame(
        Frame(image=image, timestamp=5.0, index=2, dirty_rects=(bubble,))
    )
    items = pipeline.process_frame(
        Frame(image=image.copy(), timestamp=5.0 + stable_seconds, index=3)
    )
    assert ocr.calls == 2
    assert len(items) == 1
    assert items[0].text == "Ola aventura"
    # Replaying that exact ROI now must hit the cache.
    pipeline.process_frame(Frame(image=image, timestamp=7, index=4, dirty_rects=(bubble,)))
    pipeline.process_frame(Frame(image=image.copy(), timestamp=7+stable_seconds, index=5))
    assert ocr.calls == 2
    memory.close()


def test_stable_scroll_reads_only_exposed_strip_and_keeps_shifted_items(tmp_path):
    class ScrollAwareOcr:
        def __init__(self):
            self.calls = []

        def recognize(self, image, regions):
            self.calls.append(list(regions))
            initial = len(self.calls) == 1
            return [
                TextRegion(
                    bbox=(
                        Rect(100, 80, 90, 30)
                        if initial
                        else Rect(220, 360, 90, 25)
                    ),
                    text="Previous dialogue" if initial else "New dialogue",
                    confidence=0.95,
                    source_lang="en",
                    visual_hash="previous" if initial else "new",
                )
            ]

    ocr = ScrollAwareOcr()
    config, pipeline, overlay, memory = _build_pipeline(tmp_path, ocr=ocr)
    image = np.full((400, 500, 3), 240, dtype=np.uint8)
    pipeline.process_frame(Frame(image=image, timestamp=1.0, index=0), force=True)
    assert len(overlay.items) == 1

    pipeline.shift_viewport_items(-50, width=500, height=400)
    strip = Rect(0, 350, 500, 50)
    items = pipeline.process_frame(
        Frame(
            image=image.copy(),
            timestamp=1.01,
            index=1,
            dirty_rects=(strip,),
            dirty_rects_stable=True,
        )
    )

    assert len(ocr.calls) == 2
    assert len(ocr.calls[1]) == 1
    assert ocr.calls[1][0].area < image.shape[0] * image.shape[1] / 4
    assert {item.source_text for item in items} == {
        "Previous dialogue",
        "New dialogue",
    }
    previous = next(item for item in items if item.source_text == "Previous dialogue")
    assert previous.bbox.y == 30
    pipeline.close()
    memory.close()


def test_horizontal_scroll_shifts_existing_overlay_items(tmp_path):
    _config, pipeline, overlay, memory = _build_pipeline(tmp_path)
    image = np.full((400, 500, 3), 240, dtype=np.uint8)
    pipeline.process_frame(Frame(image=image, timestamp=1.0, index=0), force=True)
    initial = pipeline.active_items[0]

    pipeline.shift_viewport_items(0, width=500, height=400, dx=-45)

    assert pipeline.active_items[0].bbox.x == initial.bbox.x - 45
    pipeline.close()
    memory.close()


def test_stable_scroll_processes_known_strip_larger_than_generic_change_guard(tmp_path):
    class BandOcr:
        def __init__(self):
            self.calls = []

        def recognize(self, image, regions):
            self.calls.append(list(regions))
            return [
                TextRegion(
                    bbox=Rect(100, 260, 100, 30),
                    text="Fresh page text",
                    confidence=0.95,
                    source_lang="en",
                    visual_hash=f"band-{len(self.calls)}",
                )
            ]

    ocr = BandOcr()
    config, pipeline, _overlay, memory = _build_pipeline(tmp_path, ocr=ocr)
    image = np.full((400, 500, 3), 240, dtype=np.uint8)
    pipeline.process_frame(Frame(image=image, timestamp=1.0, index=0), force=True)
    band = Rect(0, 230, 500, 170)
    result = pipeline.process_frame(
        Frame(
            image=image.copy(),
            timestamp=1.01,
            index=1,
            dirty_rects=(band,),
            dirty_rects_stable=True,
        )
    )

    assert len(ocr.calls) == 2
    assert ocr.calls[-1][0].area > image.shape[0] * image.shape[1] * 0.45
    assert any(item.source_text == "Fresh page text" for item in result)
    pipeline.close()
    memory.close()


def test_two_bubbles_in_one_frame_use_a_single_batch_call(tmp_path):
    translator = BatchCountingTranslator()
    config, pipeline, overlay, memory = _build_pipeline(
        tmp_path, translator=translator, ocr=TwoBubbleOcr()
    )
    image = np.full((600, 800, 3), 240, dtype=np.uint8)
    stable_seconds = config.diff.stable_ms / 1000.0 + 0.05

    pipeline.process_frame(
        Frame(
            image=image,
            timestamp=1.0,
            index=0,
            dirty_rects=(Rect(x=100, y=80, width=120, height=40), Rect(x=560, y=380, width=120, height=40)),
        )
    )
    items = pipeline.process_frame(
        Frame(image=image.copy(), timestamp=1.0 + stable_seconds, index=1)
    )

    assert len(items) == 2
    assert translator.batch_calls == 1
    assert translator.single_calls == 0
    assert pipeline.stats.translate_calls == 1
    memory.close()


def test_echo_fallback_is_not_saved_as_ptbr_translation(tmp_path):
    config, pipeline, overlay, memory = _build_pipeline(
        tmp_path, translator=EchoBatchTranslator()
    )
    region = TextRegion(
        bbox=Rect(x=100, y=80, width=120, height=40),
        text="This is still English",
        confidence=0.9,
        source_lang="en",
        visual_hash="echo-visual-hash",
    )

    translated = pipeline._translate_misses(
        [region], {"en": [0]}, config.translation.target_lang
    )

    assert translated == {}
    assert memory.lookup("This is still English", "en->pt-BR") is None
    assert memory.get_visual("echo-visual-hash", "en->pt-BR") is None
    assert memory.conn.execute("select count(*) from translations").fetchone()[0] == 0
    assert memory.conn.execute("select count(*) from visual_regions").fetchone()[0] == 0
    assert memory.history() == []
    memory.close()


def test_visual_identity_replaces_a_balloon_after_it_moves(tmp_path):
    _, pipeline, _, memory = _build_pipeline(tmp_path)
    try:
        old = OverlayItem(
            Rect(20, 30, 100, 40), "Texto antigo", "Old text",
            visual_hash="same-source-pixels",
        )
        moved = OverlayItem(
            Rect(250, 190, 100, 40), "Texto novo", "Old text",
            visual_hash="same-source-pixels",
        )
        pipeline.active_items = [old]
        pipeline._upsert_overlay_items([moved])

        assert pipeline.active_items == [moved]
    finally:
        memory.close()


def test_forced_generation_keeps_old_items_until_translation_finishes(tmp_path):
    class BlockingTranslator(Translator):
        def __init__(self):
            self.started = threading.Event()
            self.release = threading.Event()

        def translate_batch(self, texts, source_lang, target_lang, glossary, context=""):
            self.started.set()
            assert self.release.wait(5)
            return [TranslationResult("Nova traducao", "stub") for _ in texts]

    translator = BlockingTranslator()
    _, pipeline, overlay, memory = _build_pipeline(tmp_path, translator=translator)
    pipeline.async_translate = True
    old = OverlayItem(
        Rect(10, 10, 90, 30), "Traducao anterior", "Previous",
        visual_hash="old-view",
    )
    pipeline.active_items = [old]
    overlay.items = [old]
    token = pipeline.view.token()
    region = TextRegion(
        Rect(120, 80, 110, 40), "New balloon", .99, "en", "new-view"
    )
    try:
        pipeline._begin_replacement(token)
        assert pipeline._translate_regions([region], token) == []
        assert translator.started.wait(5)
        assert pipeline.active_items == [old]
        assert overlay.items == [old]

        translator.release.set()
        deadline = time.time() + 5
        while time.time() < deadline and overlay.items == [old]:
            time.sleep(.01)

        assert len(overlay.items) == 1
        assert overlay.items[0].text == "Nova traducao"
        assert overlay.items[0].visual_hash == "new-view"
    finally:
        translator.release.set()
        pipeline.close(wait=True)
        memory.close()
