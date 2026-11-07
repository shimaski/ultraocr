"""RapidOCR (ONNX Runtime) recognition backend.

An optional alternative to the PaddleOCR path, enabled with
``[ocr] detector = "rapid"``. It runs the same PP-OCR detection/recognition
models through ONNX Runtime instead of PaddlePaddle. ONNX Runtime is far
lighter than Paddle on Windows: engine start-up is near-instant (no ~40s model
build, no oneDNN crash workaround) and per-crop recognition is typically
faster. The models are shipped prebuilt by the ``rapidocr_onnxruntime``
package, so there is nothing to convert (the broken ``paddle2onnx`` export path
is avoided entirely).

Scope: the bundled models cover Latin/English text (the project's primary
source language). Japanese and Korean still belong on the Paddle path, which
ships dedicated recognizers for them. This engine keeps Paddle's public
contract — ``recognize`` / ``recognize_cancellable`` with the same cancellation
protocol, ``warmup`` and ``languages`` — and reuses ``group_text_regions`` so
reading order, balloon grouping and geometry behave exactly as before.
"""
from __future__ import annotations

import threading
from typing import Callable, Iterable

import cv2
import numpy as np

from manga_rt_translator.config import OcrConfig
from manga_rt_translator.image_hash import crop_image, perceptual_region_hash
from manga_rt_translator.models import Rect, TextRegion
from manga_rt_translator.ocr import (
    OcrEngine,
    _INFERENCE_LOCK,
    _no_cancel,
    group_text_regions,
    resolve_cpu_threads,
)

# Process-wide engine cache, keyed by the settings that change the model, so a
# restarted reader and the background warm-up share one loaded instance.
_RAPID_CACHE: dict[tuple, object] = {}
_RAPID_LOCK = threading.Lock()


class RapidOcrEngine(OcrEngine):
    detects_text = True

    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        self.languages = list(dict.fromkeys([config.source_lang, *config.extra_langs]))

    def _build_engine(self) -> object:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "detector = 'rapid' requires rapidocr-onnxruntime. Install with: "
                "pip install rapidocr-onnxruntime"
            ) from exc

        threads = resolve_cpu_threads(self.config)
        score = float(getattr(self.config, "min_confidence", 0.5))
        # Older/newer builds accept different kwargs; fall back progressively.
        for kwargs in (
            {"text_score": score, "intra_op_num_threads": threads},
            {"text_score": score},
            {},
        ):
            try:
                return RapidOCR(**kwargs)
            except TypeError:
                continue
        return RapidOCR()

    def _get_engine(self, lang: str | None = None) -> object:
        key = (
            resolve_cpu_threads(self.config),
            round(float(getattr(self.config, "min_confidence", 0.5)), 3),
        )
        engine = _RAPID_CACHE.get(key)
        if engine is not None:
            return engine
        with _RAPID_LOCK:
            engine = _RAPID_CACHE.get(key)
            if engine is None:
                engine = self._build_engine()
                _RAPID_CACHE[key] = engine
        return engine

    def warmup(self) -> None:
        engine = self._get_engine()
        with _INFERENCE_LOCK:
            engine(np.full((64, 64, 3), 240, dtype=np.uint8))

    def recognize(self, image: np.ndarray, regions: Iterable[Rect]) -> list[TextRegion]:
        return self.recognize_cancellable(image, regions, _no_cancel)

    def recognize_cancellable(
        self,
        image: np.ndarray,
        regions: Iterable[Rect],
        check: Callable[[], None],
    ) -> list[TextRegion]:
        engine = self._get_engine()
        max_side = max(1, int(getattr(self.config, "max_input_side", 640)))
        output: list[TextRegion] = []
        for region in regions:
            check()
            crop = crop_image(image, region)
            if crop.size == 0:
                continue
            # Same large-crop cap as the Paddle path: shrink before inference
            # and map the boxes back to source coordinates.
            scale = min(1.0, max_side / max(crop.shape[:2]))
            run_img = (
                crop
                if scale >= 1.0
                else cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            )
            lines = self._recognize_crop(engine, run_img, region, scale, image, check)
            output.extend(group_text_regions(lines, image))
        return group_text_regions(output, image)

    def _recognize_crop(
        self,
        engine: object,
        run_img: np.ndarray,
        region: Rect,
        scale: float,
        full_image: np.ndarray,
        check: Callable[[], None],
    ) -> list[TextRegion]:
        with _INFERENCE_LOCK:
            check()
            result, _elapse = engine(run_img)
        check()
        min_conf = float(getattr(self.config, "min_confidence", 0.5))
        parsed: list[TextRegion] = []
        for row in result or []:
            box, text, score = row[0], row[1], row[2]
            text = str(text).strip()
            if not text:
                continue
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                confidence = 0.0
            if confidence < min_conf:
                continue
            points = [
                (region.x + float(px) / scale, region.y + float(py) / scale)
                for px, py in box
            ]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x = int(min(xs))
            y = int(min(ys))
            bbox = Rect(
                x=x,
                y=y,
                width=max(1, int(max(xs)) - x),
                height=max(1, int(max(ys)) - y),
            )
            parsed.append(
                TextRegion(
                    bbox=bbox,
                    text=text,
                    confidence=confidence,
                    source_lang=self.languages[0],
                    visual_hash=perceptual_region_hash(full_image, bbox),
                    polygons=(tuple(points),),
                )
            )
        return parsed
