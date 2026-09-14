from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from manga_rt_translator.config import OcrConfig
from manga_rt_translator.image_hash import crop_image, perceptual_region_hash
from manga_rt_translator.models import Rect, TextRegion
from manga_rt_translator.ocr import (
    ComicTextOcr, OcrEngine, PaddleOcrEngine, _ENGINE_LOCK, _INFERENCE_LOCK,
    _no_cancel, group_text_regions,
)
from manga_rt_translator.work_queue import WorkCancelled


_RECOGNIZERS: dict[str, object] = {}


class HybridOcrEngine(OcrEngine):
    """Comic line detection + batched recognition; Paddle repairs weak blocks."""

    detects_text = True
    batch_size = 8
    reliable_confidence = 0.94

    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        self.comic = ComicTextOcr(config)
        self._fallback: PaddleOcrEngine | None = None
        self.last_fallback_reason = ""
        self._fast_disabled = False

    def _paddle(self) -> PaddleOcrEngine:
        if self._fallback is None:
            self._fallback = PaddleOcrEngine(self.config)
        return self._fallback

    def _recognizer(self) -> object:
        model = 'en_PP-OCRv5_mobile_rec'
        with _ENGINE_LOCK:
            if model not in _RECOGNIZERS:
                from paddleocr import TextRecognition
                _RECOGNIZERS[model] = TextRecognition(
                    model_name=model, device='cpu', enable_mkldnn=False, cpu_threads=4,
                )
            return _RECOGNIZERS[model]

    def warmup(self) -> None:
        if self.config.source_lang == 'en' and self.comic.MODEL_PATH.exists():
            self._recognizer()
        for language in self._paddle().languages:
            self._paddle()._get_engine(language)

    def recognize(self, image: np.ndarray, regions: Iterable[Rect]) -> list[TextRegion]:
        return self.recognize_cancellable(image, regions, _no_cancel)

    def recognize_cancellable(self, image: np.ndarray, regions: Iterable[Rect],
                              check: Callable[[], None]) -> list[TextRegion]:
        output: list[TextRegion] = []
        # Vertical/multilingual recognition keeps Paddle's established path.
        fast_supported = (self.config.source_lang == 'en' and not self.config.extra_langs
                          and self.config.rec_model == 'mobile'
                          and self.comic.MODEL_PATH.exists() and not self._fast_disabled)
        for region in regions:
            check()
            if not fast_supported:
                output.extend(self._paddle().recognize_cancellable(image, [region], check))
                continue
            try:
                lines = self._fast_lines(image, region, check)
            except WorkCancelled:
                raise
            except Exception as exc:
                self.last_fallback_reason = f'{type(exc).__name__}: {exc}'
                print(f'[ocr] hybrid fallback: {self.last_fallback_reason}', flush=True)
                self._fast_disabled = True
                fast_supported = False
                lines = []
            check()
            if not lines:
                output.extend(self._paddle().recognize_cancellable(image, [region], check))
                continue
            blocks = group_text_regions(lines, image)
            for block in blocks:
                check()
                at_edge = region.y2 - block.bbox.y2 < max(32, block.bbox.height)
                if block.confidence >= self.reliable_confidence and not at_edge:
                    output.append(block)
                    continue
                # Retry the whole phrase with a small margin, not each word.
                box = block.bbox
                margin = max(12, int(box.height * 0.6))
                x, y = max(region.x, box.x-margin), max(region.y, box.y-margin)
                roi = Rect(x, y, min(region.x2, box.x2+margin)-x,
                           (region.y2 if at_edge else min(region.y2, box.y2+margin))-y)
                repaired = self._paddle().recognize_cancellable(image, [roi], check)
                if repaired and min(r.confidence for r in repaired) > block.confidence:
                    output.extend(repaired)
                elif block.confidence >= self.config.min_confidence:
                    output.append(block)
        return output

    def _fast_lines(self, image: np.ndarray, region: Rect,
                    check: Callable[[], None]) -> list[TextRegion]:
        check()
        boxes = self.comic.detect_lines(crop_image(image, region), merge_vertical=False,
                                       min_area=120)
        check()
        if not boxes:
            return []
        recognizer = self._recognizer()
        lines: list[TextRegion] = []
        for start in range(0, len(boxes), self.batch_size):
            batch = boxes[start:start+self.batch_size]
            absolute = [Rect(region.x+b.x, region.y+b.y, b.width, b.height) for b in batch]
            crops = []
            for box in absolute:
                # DB contours hug the central ink; retain letter endings and punctuation.
                pad = 12
                x, y = max(region.x, box.x-pad), max(region.y, box.y-pad)
                crops.append(crop_image(image, Rect(
                    x, y, min(region.x2, box.x2+pad)-x, min(region.y2, box.y2+pad)-y,
                )))
            with _INFERENCE_LOCK:
                check()
                results = list(recognizer.predict(input=crops, batch_size=self.batch_size))
            check()
            if len(results) != len(batch):
                raise ValueError('Recognition returned a different number of lines')
            for box, result in zip(absolute, results):
                text = str(result.get('rec_text', '')).strip()
                # Keep failed lines as weak evidence so their block gets retried.
                lines.append(TextRegion(
                    bbox=box, text=text or '?', confidence=float(result.get('rec_score', 0)) if text else 0,
                    source_lang=self.config.source_lang,
                    visual_hash=perceptual_region_hash(image, box),
                ))
        return lines
