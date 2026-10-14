from __future__ import annotations

import numpy as np

from manga_rt_translator.cache import TranslationMemory
from manga_rt_translator.config import load_config
from manga_rt_translator.diff import FrameDiffer
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.models import Frame, Rect, TextRegion
from manga_rt_translator.overlay import Overlay
from manga_rt_translator.pipeline import MangaTranslationPipeline
from manga_rt_translator.translator import TranslationResult, Translator


class OneBubbleOcr:
    def __init__(self):
        self.calls = 0

    def recognize(self, image, regions):
        self.calls += 1
        return [TextRegion(
            Rect(30, 25, 26, 6), "Hello", .99, "en", "bubble",
            polygons=(((30, 25), (42, 25), (42, 31), (30, 31)),),
        )]


class OneBubbleDetector:
    def detect(self, image, regions):
        return list(regions)


class OneBubbleTranslator(Translator):
    def translate(self, text, source_lang, target_lang, glossary, context=""):
        return TranslationResult("Olá", "test")


class CaptureOverlay(Overlay):
    def __init__(self):
        self.items = []

    def update(self, items):
        self.items = list(items)

    def run_forever(self):
        return 0

    def stop(self):
        pass


def test_pipeline_carries_sampled_color_to_translated_overlay(tmp_path):
    config = load_config(None)
    memory = TranslationMemory(str(tmp_path / "memory.sqlite3"))
    overlay = CaptureOverlay()
    pipeline = MangaTranslationPipeline(
        config=config,
        stream=None,
        differ=FrameDiffer(config.diff),
        detector=OneBubbleDetector(),
        ocr=OneBubbleOcr(),
        memory=memory,
        glossary=Glossary.default(),
        translator=OneBubbleTranslator(),
        overlay=overlay,
    )
    image = np.full((60, 80, 3), (244, 241, 232), dtype=np.uint8)
    image[26:30, 34:36] = (10, 10, 10)
    image[26:28, 34:40] = (10, 10, 10)

    try:
        items = pipeline.process_frame(Frame(image, 1.0, 0), force=True)
        assert len(items) == 1
        assert items[0].text == "Olá"
        assert items[0].background_color == (244, 241, 232)
        assert items[0].background_patch is not None
        assert items[0].background_rect is not None
    finally:
        pipeline.close()
        memory.close()


def test_pipeline_uses_detected_bubble_as_translation_anchor(tmp_path):
    class BubbleOcr:
        def recognize(self, image, regions):
            return [TextRegion(Rect(124, 99, 68, 24), "Hello there", .99, "en", "speech-bubble")]

    config = load_config(None)
    memory = TranslationMemory(str(tmp_path / "bubble-anchor.sqlite3"))
    overlay = CaptureOverlay()
    pipeline = MangaTranslationPipeline(
        config=config,
        stream=None,
        differ=FrameDiffer(config.diff),
        detector=OneBubbleDetector(),
        ocr=BubbleOcr(),
        memory=memory,
        glossary=Glossary.default(),
        translator=OneBubbleTranslator(),
        overlay=overlay,
    )
    image = np.full((260, 340, 3), (120, 120, 120), dtype=np.uint8)
    image[56:204, 76:244] = (249, 249, 249)
    image[56:60, 76:244] = (15, 15, 15)
    image[200:204, 76:244] = (15, 15, 15)
    image[56:204, 76:80] = (15, 15, 15)
    image[56:204, 240:244] = (15, 15, 15)
    image[103:118, 132:136] = (12, 12, 12)
    image[103:107, 132:150] = (12, 12, 12)

    try:
        items = pipeline.process_frame(Frame(image, 1.0, 0), force=True)
        assert len(items) == 1
        assert items[0].bbox == Rect(124, 99, 68, 24)
        assert items[0].display_bbox is not None
        assert items[0].display_bbox.contains(items[0].bbox)
        assert items[0].display_bbox.width > items[0].bbox.width * 2
    finally:
        pipeline.close()
        memory.close()


def test_ocr_roi_cache_restores_polygon_coordinates(tmp_path):
    config = load_config(None)
    memory = TranslationMemory(str(tmp_path / "memory.sqlite3"))
    ocr = OneBubbleOcr()
    pipeline = MangaTranslationPipeline(
        config=config,
        stream=None,
        differ=FrameDiffer(config.diff),
        detector=OneBubbleDetector(),
        ocr=ocr,
        memory=memory,
        glossary=Glossary.default(),
        translator=OneBubbleTranslator(),
        overlay=CaptureOverlay(),
    )
    image = np.full((60, 80, 3), 245, dtype=np.uint8)
    candidate = Rect(10, 10, 50, 30)

    try:
        first = pipeline._ocr_cached(image, [candidate])
        second = pipeline._ocr_cached(image, [candidate])
        assert ocr.calls == 1
        assert first[0].bbox == second[0].bbox
        assert first[0].polygons == second[0].polygons
    finally:
        pipeline.close()
        memory.close()
