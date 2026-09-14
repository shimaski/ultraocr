from __future__ import annotations

import os
import platform
import threading
import time as _time_module
import urllib.request
from pathlib import Path
from typing import Callable, Iterable

# Must be set BEFORE paddle imports (setdefault is too late once PaddleX is
# in the process): the oneDNN backend of paddle 3.3.1 on Windows crashes with
# `ConvertPirAttribute2RuntimeAttribute not supported` on the SECOND inference
# call. Disabling MKLDNN keeps det/rec stable and, with the doc preprocessor
# off, fast (measured 1643ms -> 59ms per crop).
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_use_onednn"] = "0"

import cv2
import numpy as np

from manga_rt_translator.config import OcrConfig
from manga_rt_translator.image_hash import crop_image, perceptual_region_hash
from manga_rt_translator.models import Rect, TextRegion

# Process-wide engine cache: once a language's models are loaded they are
# reused by every later runtime start (and by the startup warm-up).
_ENGINE_CACHE: dict[tuple[str, str, str], object] = {}
_ENGINE_LOCK = threading.Lock()
_INFERENCE_LOCK = threading.RLock()


def _no_cancel() -> None:
    pass


class OcrEngine:
    def recognize(self, image: np.ndarray, regions: Iterable[Rect]) -> list[TextRegion]:
        raise NotImplementedError


class DisabledOcrEngine(OcrEngine):
    def recognize(self, image: np.ndarray, regions: Iterable[Rect]) -> list[TextRegion]:
        return []


class PaddleOcrEngine(OcrEngine):
    detects_text = True

    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "OCR requires PaddleOCR. Install with: "
                'pip install -e ".[ocr]"'
            ) from exc

        self._paddle_ocr_cls = PaddleOCR
        self.languages = list(dict.fromkeys([config.source_lang, *config.extra_langs]))
        self.engines: dict[str, object] = {}

    def recognize(self, image: np.ndarray, regions: Iterable[Rect]) -> list[TextRegion]:
        return self.recognize_cancellable(image, regions, _no_cancel)

    def recognize_cancellable(self, image: np.ndarray, regions: Iterable[Rect],
                              check: Callable[[], None]) -> list[TextRegion]:
        """Recognize each candidate region independently and group lines
        within it. Candidates are bubble-sized areas, so lines from different
        balloons never get merged into each other's blocks."""
        output: list[TextRegion] = []
        for region in regions:
            check()
            crop = crop_image(image, region)
            if crop.size == 0:
                continue

            best_regions = self._recognize_with_retry(crop, region, image, check)
            lines = [
                item
                for item in best_regions
                if item.text.strip() and item.confidence >= self.config.min_confidence
            ]
            output.extend(group_text_regions(lines, image))
        return output

    def _recognize_with_retry(
        self,
        crop: np.ndarray,
        region: Rect,
        image: np.ndarray,
        check: Callable[[], None] = _no_cancel,
    ) -> list[TextRegion]:
        """First pass at native size; stylized comic fonts often read weak, so
        a low-confidence result triggers one retry on a 2x upscaled crop."""
        best_regions = self._recognize_best_for_crop(crop, region, image, check)
        check()
        retry_limit = int(
            getattr(getattr(self, "config", None), "upscale_retry_max_pixels", 250_000)
        )
        if (
            not _should_retry_upscale(best_regions)
            or crop.shape[0] * crop.shape[1] > retry_limit
            or max(crop.shape[:2]) > 1000
        ):
            return best_regions

        scaled = cv2.resize(
            crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
        )
        retry_regions = self._recognize_best_for_crop(
            scaled,
            Rect(x=0, y=0, width=scaled.shape[1], height=scaled.shape[0]),
            scaled,
            check,
        )
        remapped = _remap_scaled_regions(retry_regions, scale=2.0, offset=region, image=image)
        if _mean_confidence(remapped) > _mean_confidence(best_regions):
            return remapped
        return best_regions

    def _recognize_best_for_crop(
        self,
        crop: np.ndarray,
        region: Rect,
        full_image: np.ndarray,
        check: Callable[[], None] = _no_cancel,
    ) -> list[TextRegion]:
        primary_lang = self.languages[0]
        primary_engine = self._get_engine(primary_lang)
        primary_regions = self._run_engine(primary_engine, crop, region, primary_lang, full_image, check)
        primary_score = _score_regions(primary_regions)

        if len(self.languages) == 1:
            return primary_regions
        if primary_score >= 1.35 and not _looks_like_cjk_text(primary_regions):
            return primary_regions

        best_regions = primary_regions
        best_score = primary_score
        for lang in self.languages[1:]:
            engine = self._get_engine(lang)
            parsed = self._run_engine(engine, crop, region, lang, full_image, check)
            score = _score_regions(parsed)
            if score > best_score:
                best_regions = parsed
                best_score = score
        return best_regions

    def _run_engine(
        self,
        engine: object,
        crop: np.ndarray,
        offset: Rect,
        lang: str,
        full_image: np.ndarray,
        check: Callable[[], None] = _no_cancel,
    ) -> list[TextRegion]:
        # Cached Paddle predictors are shared with warmup and restarted readers.
        with _INFERENCE_LOCK:
            check()
            try:
                result = engine.ocr(crop, cls=True)  # type: ignore[attr-defined]
            except TypeError:
                result = engine.ocr(crop)  # type: ignore[attr-defined]
        check()

        return parse_paddle_result(result, offset, lang, full_image)

    def _build_engine(self, paddle_ocr_cls: object, lang: str) -> object:
        # Fastest path: disable the document preprocessor (UVDoc +
        # PP-LCNet orientation) that PaddleX 3.x enables by default. It adds
        # ~1.5s per crop and several model loads; the comic detector already
        # supplies tight line boxes, and horizontal manga text does not need
        # doc unwarping. Measured: 1643ms -> 59ms per crop.
        first_kwargs: dict[str, object] = {
            "lang": lang,
            "device": "cpu",
            "enable_mkldnn": False,
            "cpu_threads": 4,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if getattr(self.config, "det_model", "mobile") == "mobile":
            # PP-OCRv5 mobile detection is several times faster than the
            # server model, which matters for continuous scroll translation.
            first_kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
        rec_model = getattr(self.config, "rec_model", "mobile")
        if rec_model == "server":
            first_kwargs["text_recognition_model_name"] = f"{lang}_PP-OCRv5_server_rec"
        else:
            # Explicitly request the mobile recognizer. Without this, newer
            # PaddleX builds default to the slow server recognizer even when
            # rec_model = "mobile" (seen in reader_stderr.log).
            # NOTE: PaddleX 3.x has NO `japan_PP-OCRv5_mobile_rec` in its
            # registry (only japan_PP-OCRv3). Using an unregistered name makes
            # the constructor fall back to the slow default engine. Map the
            # known-good v5 names; the generic v5 covers everything else.
            v5_langs = {
                "en", "korean", "latin", "eslav", "th", "el", "arabic",
                "te", "ta", "devanagari", "cyrillic",
            }
            if lang in v5_langs:
                first_kwargs["text_recognition_model_name"] = (
                    f"{lang}_PP-OCRv5_mobile_rec"
                )
            else:
                first_kwargs["text_recognition_model_name"] = "PP-OCRv5_mobile_rec"

        constructors: list[dict[str, object]] = [first_kwargs]
        constructors.extend(
            [
                {"use_textline_orientation": True, "lang": lang},
                {"use_angle_cls": True, "lang": lang},
                {"use_textline_orientation": True, "lang": lang, "show_log": False},
                {"use_angle_cls": True, "lang": lang, "show_log": False},
                {"lang": lang},
            ]
        )

        last_error: Exception | None = None
        for kwargs in constructors:
            try:
                return paddle_ocr_cls(**kwargs)  # type: ignore[misc]
            except (TypeError, ValueError) as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Unable to initialize PaddleOCR for lang={lang!r}")

    def _get_engine(self, lang: str) -> object:
        key = (
            lang,
            str(getattr(self.config, "det_model", "mobile")),
            str(getattr(self.config, "rec_model", "mobile")),
        )
        with _ENGINE_LOCK:
            engine = _ENGINE_CACHE.get(key)
            if engine is None:
                engine = self._build_engine(self._paddle_ocr_cls, lang)
                _ENGINE_CACHE[key] = engine
            return engine


def lower_thread_priority() -> None:
    """Drop the calling thread below normal priority so OCR work yields CPU
    to the UI and to the browser renderer (which uses software OpenGL on
    many setups). OCR throughput barely changes; scroll jank goes away."""
    if platform.system().lower() != "windows":
        return
    try:
        import ctypes

        handle = ctypes.windll.kernel32.GetCurrentThread()
        ctypes.windll.kernel32.SetThreadPriority(handle, -1)  # BELOW_NORMAL
    except Exception:
        pass


class ComicTextOcr(OcrEngine):
    """Gaminik-style detection: the comic-text-detector ONNX model finds text
    line boxes on a 1024px blob via OpenCV DNN (trained specifically for
    manga/comic layouts), and PaddleOCR only reads the detected lines.

    Detection on the full region costs ~100-300ms (native inference) instead
    of running PaddleOCR detection over every candidate.
    """

    MODEL_URL = (
        "https://github.com/zyddnys/manga-image-translator/releases/"
        "download/beta-0.2.1/comictextdetector.pt.onnx"
    )
    MODEL_PATH = Path(".models") / "comictextdetector.pt.onnx"
    INPUT_SIZE = 1024
    # Calibrated from real reader captures: 0.35 dropped handwritten strokes;
    # 0.25 keeps more fragments, which _merge_close_boxes then recombines.
    LINE_THRESHOLD = 0.25
    # Boxes closer than this (along the reading axis) are fragments of the
    # same handwritten line and get merged back together.
    MERGE_GAP = 14
    detects_text = True

    def __init__(self, config: OcrConfig) -> None:
        self.config = config
        self._session: object | None = None
        self._input_name: str | None = None
        self._paddle: PaddleOcrEngine | None = None
        self._last_det_log = 0.0
        self.languages = list(dict.fromkeys([config.source_lang, *config.extra_langs]))

    def warmup(self) -> None:
        self._ensure_session()
        dummy = np.full((256, 256, 3), 240, dtype=np.uint8)
        self.detect_lines(dummy)
        # The comic detector hands line boxes to PaddleOCR for recognition;
        # pre-build those engines here too so the first frame does not pay a
        # 30-40s model-loading stall on the capture thread.
        paddle = self._paddle_engine()
        for lang in getattr(paddle, "languages", []):
            paddle._get_engine(lang)

    def _get_engine(self, lang: str) -> object:
        return self._paddle_engine()._get_engine(lang)

    def _ensure_session(self):
        if self._session is not None:
            return self._session
        path = self.MODEL_PATH
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("[ocr] downloading comic text detector (~90MB)...", flush=True)
            urllib.request.urlretrieve(self.MODEL_URL, path)
            print("[ocr] comic text detector downloaded", flush=True)
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            providers = [
                name
                for name in (
                    "DmlExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                )
                if name in available
            ]
            self._session = ort.InferenceSession(str(path), providers=providers)
            self._input_name = self._session.get_inputs()[0].name
            print(f"[ocr] comic detector providers: {providers}", flush=True)
        except ImportError:
            # Fallback: OpenCV DNN runs the same model on CPU (slower).
            self._session = cv2.dnn.readNetFromONNX(str(path))
            self._input_name = None
        return self._session

    def _paddle_engine(self) -> PaddleOcrEngine:
        if self._paddle is None:
            self._paddle = PaddleOcrEngine(self.config)
        return self._paddle

    def detect_lines(self, image: np.ndarray, *, merge_vertical: bool = True,
                     min_area: int = 600) -> list[Rect]:
        t0 = _time_module.perf_counter()
        session = self._ensure_session()
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image, scalefactor=1 / 255.0, size=(self.INPUT_SIZE, self.INPUT_SIZE)
        ).astype(np.float32)
        if self._input_name is not None:
            outs = session.run(None, {self._input_name: blob})
            name_map = {
                name: tensor
                for name, tensor in zip(
                    [o.name for o in session.get_outputs()], outs
                )
            }
            lines_map = name_map["det"]
        else:
            uoln = session.getUnconnectedOutLayersNames()
            session.setInput(blob)
            outs = session.forward(uoln)
            output_names = list(uoln)
            lines_map = outs[output_names.index("det")]
        line_heat = lines_map[0]
        # det has 2 channels: [0] = DB text-line probability, [1] = DB
        # threshold map (elevated almost everywhere — never max them).
        if line_heat.shape[0] > 1:
            line_heat = line_heat[0]
        heat = (line_heat > self.LINE_THRESHOLD).astype(np.uint8) * 255
        heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)
        contours, _ = cv2.findContours(heat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: list[Rect] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < 12 or bh < 6 or bw * bh < min_area:
                continue
            pad = 5
            boxes.append(
                Rect(
                    x=max(0, x - pad),
                    y=max(0, y - pad),
                    width=min(w - max(0, x - pad), bw + pad * 2),
                    height=min(h - max(0, y - pad), bh + pad * 2),
                )
            )
        boxes = _merge_close_boxes(boxes, gap=self.MERGE_GAP, max_width=w, max_height=h,
                                   merge_vertical=merge_vertical)
        elapsed = (_time_module.perf_counter() - t0) * 1000
        now = _time_module.time()
        if now - self._last_det_log > 10:
            self._last_det_log = now
            print(
                f"[ocr] comic det {elapsed:.0f}ms boxes={len(boxes)}",
                flush=True,
            )
        return boxes

    def recognize(self, image: np.ndarray, regions: Iterable[Rect]) -> list[TextRegion]:
        paddle = self._paddle_engine()
        output: list[TextRegion] = []
        for region in regions:
            crop = crop_image(image, region)
            if crop.size == 0:
                continue
            boxes = self.detect_lines(crop)
            if not boxes:
                # The comic detector is the authority on where text lines are.
                # Falling back to the whole frame as a single crop makes
                # PaddleOCR scan the entire page (seconds of freeze) and then
                # misreport non-text as balloons. Nothing to recognize here.
                continue
            for box in boxes:
                abs_box = Rect(
                    x=region.x + box.x,
                    y=region.y + box.y,
                    width=box.width,
                    height=box.height,
                )
                output.extend(paddle.recognize(image, [abs_box]))
        return group_text_regions(output, image)


def _merge_close_boxes(
    boxes: list[Rect],
    gap: int,
    max_width: int,
    max_height: int,
    merge_vertical: bool = True,
) -> list[Rect]:
    """Merge detector fragments back into whole text lines.

    The ONNX comic detector at a low threshold splits handwritten strokes
    into many small boxes. Fragments of the same line share a vertical band
    and are horizontally adjacent (or stacked a few px apart); merging them
    restores the line so PaddleOCR reads it as one unit.
    """
    if not boxes:
        return []

    merged = sorted(boxes, key=lambda b: (b.y, b.x))
    while True:
        changed = False
        output: list[Rect] = []
        for box in merged:
            if not output:
                output.append(box)
                continue
            # Merge into the best-matching existing box (nearest y band).
            best_index = -1
            best_band = 0.0
            for index, prev in enumerate(output):
                min_h = min(prev.height, box.height)
                y_overlap = min(prev.y2, box.y2) - max(prev.y, box.y)
                band = (y_overlap / min_h) if min_h > 0 else 0.0
                if band >= 0.5 and band > best_band:
                    best_band = band
                    best_index = index
            if best_index >= 0:
                prev = output[best_index]
                x_gap = box.x - prev.x2
                if -gap <= x_gap <= gap:
                    output[best_index] = prev.union(box)
                    changed = True
                    continue
            # Vertically stacked fragments of the same column (tall boxes
            # split by the threshold) — overlap horizontally and sit within
            # the gap along the vertical axis.
            best_index = -1
            best_band = 0.0
            for index, prev in enumerate(output):
                min_w = min(prev.width, box.width)
                x_overlap = min(prev.x2, box.x2) - max(prev.x, box.x)
                band = (x_overlap / min_w) if min_w > 0 else 0.0
                if band >= 0.5 and band > best_band:
                    best_band = band
                    best_index = index
            if best_index >= 0 and merge_vertical:
                prev = output[best_index]
                y_gap = box.y - prev.y2
                if -gap <= y_gap <= gap:
                    output[best_index] = prev.union(box)
                    changed = True
                    continue
            output.append(box)
        merged = output
        if not changed:
            break

    # Clamp back to the frame.
    return [
        Rect(
            x=max(0, b.x),
            y=max(0, b.y),
            width=min(max_width - max(0, b.x), b.width),
            height=min(max_height - max(0, b.y), b.height),
        )
        for b in merged
        if b.width > 0 and b.height > 0
    ]


def warmup_ocr(config: OcrConfig) -> None:
    """Pre-build the OCR engines (and download models, if needed) in a
    background thread so the first selected region translates right away."""

    def _build() -> None:
        try:
            lower_thread_priority()
            engine = create_ocr_engine(config)
            warm = getattr(engine, "warmup", None)
            if callable(warm):
                warm()
            for lang in getattr(engine, "languages", []):
                engine._get_engine(lang)
            print("[ocr] warm-up complete", flush=True)
        except Exception as exc:
            print(f"[ocr] warm-up skipped: {exc}", flush=True)

    threading.Thread(target=_build, daemon=True, name="ocr-warmup").start()


def create_ocr_engine(config: OcrConfig) -> OcrEngine:
    if not config.enabled:
        return DisabledOcrEngine()
    if config.detector == "hybrid":
        from manga_rt_translator.hybrid_ocr import HybridOcrEngine
        return HybridOcrEngine(config)
    if getattr(config, "detector", "paddle") == "comic":
        return ComicTextOcr(config)
    return PaddleOcrEngine(config)


def parse_paddle_result(
    result: object,
    offset: Rect,
    lang: str,
    full_image: np.ndarray,
) -> list[TextRegion]:
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return _parse_paddle_dict_result(result, offset, lang, full_image)

    entries = _flatten_paddle_entries(result)
    regions: list[TextRegion] = []

    for entry in entries:
        parsed = _parse_entry(entry)
        if parsed is None:
            continue
        points, text, confidence = parsed
        xs = [int(point[0]) for point in points]
        ys = [int(point[1]) for point in points]
        x = offset.x + max(0, min(xs))
        y = offset.y + max(0, min(ys))
        x2 = offset.x + max(xs)
        y2 = offset.y + max(ys)
        bbox = Rect(x=x, y=y, width=max(1, x2 - x), height=max(1, y2 - y))
        regions.append(
            TextRegion(
                bbox=bbox,
                text=text.strip(),
                confidence=float(confidence),
                source_lang=lang,
                visual_hash=perceptual_region_hash(full_image, bbox),
            )
        )

    return merge_text_regions(regions)


def _parse_paddle_dict_result(
    result: list[object],
    offset: Rect,
    lang: str,
    full_image: np.ndarray,
) -> list[TextRegion]:
    regions: list[TextRegion] = []

    for item in result:
        if not isinstance(item, dict):
            continue
        texts = item.get("rec_texts", [])
        scores = item.get("rec_scores", [])
        polys = item.get("rec_polys")
        if polys is None:
            polys = item.get("dt_polys", [])
        if texts is None or scores is None or polys is None:
            continue

        for text, score, poly in zip(texts, scores, polys):
            parsed_points = _poly_to_points(poly)
            if not parsed_points:
                continue
            try:
                confidence = float(score)
            except (TypeError, ValueError):
                confidence = 0.0
            text_value = str(text).strip()
            xs = [int(point[0]) for point in parsed_points]
            ys = [int(point[1]) for point in parsed_points]
            x = offset.x + max(0, min(xs))
            y = offset.y + max(0, min(ys))
            x2 = offset.x + max(xs)
            y2 = offset.y + max(ys)
            bbox = Rect(x=x, y=y, width=max(1, x2 - x), height=max(1, y2 - y))
            regions.append(
                TextRegion(
                    bbox=bbox,
                    text=text_value,
                    confidence=confidence,
                    source_lang=lang,
                    visual_hash=perceptual_region_hash(full_image, bbox),
                )
            )

    return merge_text_regions(regions)


def _iou(left: Rect, right: Rect) -> float:
    inter_w = min(left.x2, right.x2) - max(left.x, right.x)
    inter_h = min(left.y2, right.y2) - max(left.y, right.y)
    if inter_w <= 0 or inter_h <= 0:
        return 0.0
    inter = inter_w * inter_h
    return inter / max(1, left.area + right.area - inter)


def _dedup_overlapping_lines(regions: list[TextRegion]) -> list[TextRegion]:
    """Drop duplicate detections of the same line (the detector often fires
    twice at different scales), keeping the higher-confidence copy."""
    kept: list[TextRegion] = []
    for region in sorted(regions, key=lambda item: -item.confidence):
        duplicate = any(
            _iou(region.bbox, other.bbox) > 0.55 for other in kept
        )
        if not duplicate:
            kept.append(region)
    return kept


def _mean_confidence(regions: list[TextRegion]) -> float:
    if not regions:
        return 0.0
    weighted = sum(item.confidence * max(1, len(item.text)) for item in regions)
    total = sum(max(1, len(item.text)) for item in regions)
    return weighted / total


def _should_retry_upscale(regions: list[TextRegion]) -> bool:
    if not regions:
        return True
    return _mean_confidence(regions) < 0.85


def _remap_scaled_regions(
    regions: list[TextRegion],
    scale: float,
    offset: Rect,
    image: np.ndarray,
) -> list[TextRegion]:
    remapped: list[TextRegion] = []
    for region in regions:
        bbox = Rect(
            x=offset.x + int(region.bbox.x / scale),
            y=offset.y + int(region.bbox.y / scale),
            width=max(1, int(region.bbox.width / scale)),
            height=max(1, int(region.bbox.height / scale)),
        )
        remapped.append(
            TextRegion(
                bbox=bbox,
                text=region.text,
                confidence=region.confidence,
                source_lang=region.source_lang,
                visual_hash=perceptual_region_hash(image, bbox),
            )
        )
    return remapped


def merge_text_regions(regions: list[TextRegion]) -> list[TextRegion]:
    # Keep original line geometry until grouping can also inspect balloon edges.
    return sorted(_dedup_overlapping_lines(regions), key=lambda item: (item.bbox.y, item.bbox.x))


def _cjk_ratio(text: str) -> float:
    if not text:
        return 0.0
    cjk = sum(
        1
        for char in text
        if 0x3040 <= ord(char) <= 0x30FF
        or 0x3400 <= ord(char) <= 0x9FFF
        or 0xFF66 <= ord(char) <= 0xFF9F
    )
    return cjk / max(1, len(text))


def _is_vertical_box(bbox: Rect) -> bool:
    """Vertical Japanese (tategaki) columns are much taller than wide."""
    return bbox.height > bbox.width * 1.5


def _same_block(left: TextRegion, right: TextRegion) -> bool:
    if left.source_lang != right.source_lang:
        return False
    a, b = left.bbox, right.bbox
    height_ratio = max(a.height, b.height) / max(1, min(a.height, b.height))
    if height_ratio > 2.4:
        return False

    if _cjk_ratio(left.text + right.text) > 0.25 and _is_vertical_box(a) and _is_vertical_box(b):
        # Same column, stacked segments: strong x-overlap, small y gap.
        y_overlap = min(a.y2, b.y2) - max(a.y, b.y)
        x_gap = max(a.x, b.x) - min(a.x2, b.x2)
        if y_overlap >= 0.45 * min(a.height, b.height) and x_gap <= 1.6 * max(
            8, min(a.width, b.width)
        ):
            return True
        # Adjacent columns of the same block: strong y-overlap, small x gap.
        x_overlap = min(a.x2, b.x2) - max(a.x, b.x)
        y_gap = max(a.y, b.y) - min(a.y2, b.y2)
        return (
            x_overlap >= 0.45 * min(a.width, b.width)
            and -0.6 * max(8, min(a.width, b.width))
            <= y_gap
            <= 1.4 * max(8, min(a.width, b.width))
        )

    # Horizontal lines pair when they overlap horizontally and the vertical
    # gap between them matches a line spacing of the same text block.
    x_overlap = min(a.x2, b.x2) - max(a.x, b.x)
    y_gap = max(a.y, b.y) - min(a.y2, b.y2)
    y_overlap = min(a.y2, b.y2) - max(a.y, b.y)
    x_gap = max(a.x, b.x) - min(a.x2, b.x2)
    if y_overlap >= 0.65 * min(a.height, b.height) and 0 <= x_gap <= min(24, min(a.height, b.height)):
        return True
    return (
        x_overlap >= 0.45 * min(a.width, b.width)
        and -0.6 * min(a.height, b.height) <= y_gap <= 1.4 * min(a.height, b.height)
    )


def _has_separator(image: np.ndarray | None, a: Rect, b: Rect) -> bool:
    if image is None:
        return False
    # An ink edge crossing the gap is evidence of separate balloons/panels.
    if max(a.y, b.y) >= min(a.y2, b.y2) + 2:
        x, x2 = max(a.x, b.x), min(a.x2, b.x2)
        y, y2 = min(a.y2, b.y2), max(a.y, b.y)
        axis = 1
    elif max(a.x, b.x) >= min(a.x2, b.x2) + 2:
        x, x2 = min(a.x2, b.x2), max(a.x, b.x)
        y, y2 = max(a.y, b.y), min(a.y2, b.y2)
        axis = 0
    else:
        return False
    if x2 <= x or y2 <= y:
        return False
    crop = crop_image(image, Rect(x, y, x2-x, y2-y))
    if not crop.size:
        return False
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    return bool(np.any(np.mean(gray < 80, axis=axis) >= 0.7))


def _horizontal_reading_order(members: list[TextRegion]) -> list[TextRegion]:
    rows: list[list[TextRegion]] = []
    for member in sorted(members, key=lambda item: item.bbox.y):
        for row in rows:
            a, b = row[0].bbox, member.bbox
            if min(a.y2, b.y2)-max(a.y, b.y) >= 0.65*min(a.height, b.height):
                row.append(member)
                break
        else:
            rows.append([member])
    return [item for row in rows for item in sorted(row, key=lambda item: item.bbox.x)]


def group_text_regions(
    regions: list[TextRegion],
    image: np.ndarray | None = None,
) -> list[TextRegion]:
    """Group OCR line regions into text blocks (speech bubbles) and order the
    blocks for reading.

    Adapted from the approach used by open-source manga translators
    (manga-image-translator, BallonsTranslator): expand text evidence until
    connected blocks emerge, then sort. Here the "expansion" runs on OCR line
    boxes instead of pixel masks, so no extra model is needed.

    Reading order: CJK-dominant pages read right-to-left (manga), everything
    else reads top-to-bottom, left-to-right.
    """
    regions = _dedup_overlapping_lines(regions)

    parent = list(range(len(regions)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if (_same_block(regions[i], regions[j])
                    and not _has_separator(image, regions[i].bbox, regions[j].bbox)):
                root_j = find(j)
                if root_j != find(i):
                    parent[root_j] = find(i)

    groups: dict[int, list[TextRegion]] = {}
    for index, region in enumerate(regions):
        groups.setdefault(find(index), []).append(region)

    blocks: list[TextRegion] = []
    for members in groups.values():
        vertical_block = (_cjk_ratio(''.join(m.text for m in members)) > 0.25
                          and all(_is_vertical_box(m.bbox) for m in members))
        if vertical_block:
            # Right column first, top to bottom inside each column.
            members.sort(key=lambda m: (-m.bbox.x, m.bbox.y))
        else:
            # Top line first, left to right inside each line.
            members = _horizontal_reading_order(members)

        separator = "" if _cjk_ratio("".join(m.text for m in members)) > 0.25 else " "
        text = separator.join(m.text.strip() for m in members if m.text.strip())
        if not text:
            continue

        union_bbox = members[0].bbox
        for member in members[1:]:
            union_bbox = union_bbox.union(member.bbox)

        langs = [m.source_lang for m in members if m.source_lang]
        source_lang = max(set(langs), key=langs.count) if langs else None
        visual_hash = (
            perceptual_region_hash(image, union_bbox)
            if image is not None
            else members[0].visual_hash
        )
        blocks.append(
            TextRegion(
                bbox=union_bbox,
                text=text,
                confidence=min(m.confidence for m in members),
                source_lang=source_lang,
                visual_hash=visual_hash,
                metadata={"lines": len(members)},
            )
        )

    if blocks and _cjk_ratio(" ".join(b.text for b in blocks)) > 0.15:
        # Manga reading order: right to left, top to bottom.
        blocks.sort(key=lambda b: (-b.bbox.x, b.bbox.y))
    else:
        blocks.sort(key=lambda b: (b.bbox.y, b.bbox.x))
    return blocks


def _flatten_paddle_entries(result: object) -> list[object]:
    if not result:
        return []
    if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
        return list(result[0] or [])
    if isinstance(result, list):
        return result
    return []


def _poly_to_points(poly: object) -> list[list[float]]:
    if isinstance(poly, np.ndarray):
        return poly.tolist()
    if isinstance(poly, list):
        return poly
    if isinstance(poly, tuple):
        return list(poly)
    return []


def _parse_entry(entry: object) -> tuple[list[list[float]], str, float] | None:
    if not isinstance(entry, (list, tuple)) or len(entry) < 2:
        return None

    points = entry[0]
    text_info = entry[1]
    if (
        isinstance(points, (list, tuple))
        and isinstance(text_info, (list, tuple))
        and len(text_info) >= 2
    ):
        text = str(text_info[0])
        try:
            confidence = float(text_info[1])
        except (TypeError, ValueError):
            confidence = 0.0
        return list(points), text, confidence

    return None


def _looks_like_cjk_text(regions: list[TextRegion]) -> bool:
    if not regions:
        return False
    text = " ".join(region.text for region in regions)
    for char in text:
        code = ord(char)
        if (
            0x3040 <= code <= 0x30FF          # kana
            or 0x3400 <= code <= 0x9FFF        # CJK ideographs
            or 0xFF66 <= code <= 0xFF9F        # halfwidth katakana
            or 0x1100 <= code <= 0x11FF        # Hangul Jamo
            or 0x3130 <= code <= 0x318F        # Hangul compatibility Jamo
            or 0xAC00 <= code <= 0xD7A3        # Hangul syllables
        ):
            return True
    return False


def _score_regions(regions: list[TextRegion]) -> float:
    return sum(max(0.0, item.confidence) * max(1, len(item.text)) for item in regions)
