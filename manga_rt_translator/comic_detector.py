"""Comic text + speech-balloon detector (RT-DETR-v2, ONNX).

Model: ``detector-v4-s_int8.onnx`` from
https://huggingface.co/ogkalu/comic-text-and-bubble-detector (Apache-2.0),
the detector used by comic-translate. Labels: 0 = bubble, 1 = text inside a
bubble, 2 = free text (captions, sound effects).

It replaces geometric guessing about which OCR lines belong together: OCR
lines inside the same detected text block form one caption, lines in
different blocks never merge, and blocks the OCR missed are re-read from a
tight crop. Everything is optional — without the model file or onnxruntime
the reader keeps its previous behaviour.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from manga_rt_translator.models import Rect

_LOGGER = logging.getLogger(__name__)

MODEL_FILENAME = "detector-v4-s_int8.onnx"
INPUT_SIZE = 640
CONFIDENCE = 0.30
# Pages taller than this (height / width) are split into overlapping slices so
# small lettering survives the 640x640 resize (webtoon strips).
SLICE_RATIO = 2.4


@dataclass
class ComicDetections:
    bubbles: list[Rect] = field(default_factory=list)
    texts: list[Rect] = field(default_factory=list)


def model_candidates() -> list[Path]:
    project_root = Path(__file__).resolve().parent.parent
    return [
        project_root / ".models" / MODEL_FILENAME,
        project_root / MODEL_FILENAME,
        Path.cwd() / ".models" / MODEL_FILENAME,
        Path.cwd() / MODEL_FILENAME,
    ]


class ComicDetector:
    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # CPU keeps the int8 graph exact and leaves the GPU to the browser.
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> ComicDetections:
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            return ComicDetections()
        if height / max(1, width) <= SLICE_RATIO:
            return self._detect_single(image, 0)
        slice_height = int(width * 1.6)
        step = max(1, int(slice_height * 0.8))
        merged = ComicDetections()
        for top in range(0, height, step):
            bottom = min(height, top + slice_height)
            part = self._detect_single(image[top:bottom], top)
            merged.bubbles.extend(part.bubbles)
            merged.texts.extend(part.texts)
            if bottom >= height:
                break
        return ComicDetections(
            bubbles=_dedupe_boxes(merged.bubbles), texts=_dedupe_boxes(merged.texts)
        )

    def _detect_single(self, image: np.ndarray, offset_y: int) -> ComicDetections:
        import cv2

        height, width = image.shape[:2]
        rgb = np.ascontiguousarray(image[:, :, :3])
        tensor = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        tensor = (tensor.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        with self._lock:
            labels, boxes, scores = self.session.run(
                None,
                {
                    "images": tensor,
                    "orig_target_sizes": np.array([[width, height]], dtype=np.int64),
                },
            )[:3]
        result = ComicDetections()
        for label, box, score in zip(labels[0], boxes[0], scores[0]):
            if float(score) < CONFIDENCE:
                continue
            x1, y1, x2, y2 = (int(round(float(v))) for v in box)
            x1, x2 = max(0, x1), min(width, x2)
            y1, y2 = max(0, y1), min(height, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            rect = Rect(x1, y1 + offset_y, x2 - x1, y2 - y1)
            if int(label) == 0:
                result.bubbles.append(rect)
            else:
                result.texts.append(rect)
        return result


def _dedupe_boxes(boxes: list[Rect]) -> list[Rect]:
    """Merge duplicates produced by overlapping slices."""
    kept: list[Rect] = []
    for box in sorted(boxes, key=lambda r: r.area, reverse=True):
        duplicate = False
        for index, other in enumerate(kept):
            inter_w = min(box.x2, other.x2) - max(box.x, other.x)
            inter_h = min(box.y2, other.y2) - max(box.y, other.y)
            if inter_w > 0 and inter_h > 0 and inter_w * inter_h >= 0.6 * box.area:
                kept[index] = other.union(box)
                duplicate = True
                break
        if not duplicate:
            kept.append(box)
    return kept


_INSTANCE: ComicDetector | None = None
_INSTANCE_FAILED = False
_INSTANCE_LOCK = threading.Lock()
_CACHE: dict[str, object] = {"key": None, "value": None}
_CACHE_LOCK = threading.Lock()


def get_detector() -> ComicDetector | None:
    """The shared detector, or None when the model/onnxruntime is missing."""
    global _INSTANCE, _INSTANCE_FAILED
    if _INSTANCE is not None or _INSTANCE_FAILED:
        return _INSTANCE
    with _INSTANCE_LOCK:
        if _INSTANCE is not None or _INSTANCE_FAILED:
            return _INSTANCE
        path = next((p for p in model_candidates() if p.is_file()), None)
        if path is None:
            _INSTANCE_FAILED = True
            print("[detector] comic text detector not found; using geometric grouping", flush=True)
            return None
        try:
            _INSTANCE = ComicDetector(path)
            print(f"[detector] comic text detector loaded: {path}", flush=True)
        except Exception as exc:  # onnxruntime missing or bad model
            _INSTANCE_FAILED = True
            print(f"[detector] unavailable ({type(exc).__name__}: {exc})", flush=True)
            return None
    return _INSTANCE


def detect_cached(image: np.ndarray | None) -> ComicDetections | None:
    """Detect once per frame: OCR grouping calls this for every tile."""
    if image is None or image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    detector = get_detector()
    if detector is None:
        return None
    key = (id(image), image.shape, image.__array_interface__["data"][0])
    with _CACHE_LOCK:
        if _CACHE["key"] == key:
            return _CACHE["value"]  # type: ignore[return-value]
    try:
        value = detector.detect(image)
    except Exception as exc:
        _LOGGER.warning("comic detector failed: %s", exc)
        return None
    with _CACHE_LOCK:
        _CACHE["key"] = key
        _CACHE["value"] = value
    return value


def block_index_for(box: Rect, texts: list[Rect]) -> int | None:
    """Index of the detected text block that holds most of this OCR line."""
    best: int | None = None
    best_ratio = 0.0
    area = max(1, box.area)
    for index, text in enumerate(texts):
        inter_w = min(box.x2, text.x2) - max(box.x, text.x)
        inter_h = min(box.y2, text.y2) - max(box.y, text.y)
        if inter_w <= 0 or inter_h <= 0:
            continue
        ratio = inter_w * inter_h / area
        if ratio > best_ratio:
            best_ratio = ratio
            best = index
    return best if best_ratio >= 0.5 else None
