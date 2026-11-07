from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Iterable

import numpy as np

from manga_rt_translator.cache import TranslationMemory, normalize_text
from manga_rt_translator.capture.base import ScreenStream
from manga_rt_translator.config import AppConfig
from manga_rt_translator.diff import FrameDiffer
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.image_hash import crop_image, perceptual_region_hash
from manga_rt_translator.models import Frame, OverlayItem, Rect, TextRegion
from manga_rt_translator.overlay_background import (
    SpeechBubbleRegion,
    detect_enclosed_speech_bubbles,
    inpaint_bubble_glyphs,
    inpaint_text_background,
    sample_background_color,
)
from manga_rt_translator.ocr import OcrEngine
from manga_rt_translator.overlay import Overlay
from manga_rt_translator.text_regions import TextRegionDetector
from manga_rt_translator.translator import (
    EchoTranslator, TranslationResult, Translator, get_rate_tracker,
)
from manga_rt_translator.work_queue import ViewState, WorkCancelled, WorkToken


@dataclass
class PipelineStats:
    frames_seen: int = 0
    frames_with_rois: int = 0
    ocr_regions: int = 0
    cache_hits: int = 0
    translate_calls: int = 0
    last_error: str = ""
    stale_results: int = 0
    last_translation_ms: float = 0.0
    last_translated_regions: int = 0
    # Breakdown of the most recent analysed frame, logged by the reader so
    # slow frames show where the time went (base OCR vs refinement passes).
    last_ocr_ms: float = 0.0
    last_refine_ms: float = 0.0
    last_candidates: int = 0
    last_bubbles: int = 0
    last_rescued: int = 0


@dataclass
class MangaTranslationPipeline:
    config: AppConfig
    stream: ScreenStream
    differ: FrameDiffer
    detector: TextRegionDetector
    ocr: OcrEngine
    memory: TranslationMemory
    glossary: Glossary
    translator: Translator
    overlay: Overlay
    stats: PipelineStats = field(default_factory=PipelineStats)
    active_items: list[OverlayItem] = field(default_factory=list)
    context_lines: list[str] = field(default_factory=list)
    # Exact-content OCR cache: candidate rect hash -> OCR regions. Repeated
    # sweeps and re-scrolled panels skip the OCR entirely.
    roi_ocr_cache: "OrderedDict[str, list[TextRegion]]" = field(
        default_factory=lambda: OrderedDict()
    )
    roi_ocr_cache_size: int = 64
    # When True, translation runs on a background thread so the capture/OCR
    # loop never blocks on the network round trip. The reader enables this;
    # CLI/tests keep the synchronous path.
    async_translate: bool = False
    view: ViewState = field(default_factory=ViewState)
    _translate_executor: "ThreadPoolExecutor | None" = field(
        default=None, init=False, repr=False
    )
    _retired_translate_executors: list[ThreadPoolExecutor] = field(
        default_factory=list, init=False, repr=False
    )
    _items_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    # Visual hashes with a provider call already in flight. Async sweeps that
    # re-run before the translation lands skip re-dispatching the same region.
    _in_flight: set[tuple[int, str]] = field(default_factory=set, init=False, repr=False)
    _futures: dict[Future, WorkToken] = field(default_factory=dict, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _replacement_generation: int | None = field(default=None, init=False, repr=False)
    _staged_items: dict[int, list[OverlayItem]] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def has_pending_translations(self) -> bool:
        with self.view.lock:
            return any(not future.done() for future in self._futures)

    def run(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        self.stream.start()

        try:
            for frame in self.stream.frames():
                if stop_event.is_set():
                    break
                self.process_frame(frame)
        finally:
            self.stream.stop()
            self.close()

    def process_frame(
        self, frame: Frame, *, force: bool = False, token: WorkToken | None = None
    ) -> list[OverlayItem]:
        token = token or self.view.token()
        try:
            return self._process_frame(frame, force=force, token=token)
        except WorkCancelled:
            self.stats.stale_results += 1
            return []

    def _process_frame(self, frame: Frame, *, force: bool, token: WorkToken) -> list[OverlayItem]:
        token.check()
        set_viewport = getattr(self.overlay, "set_capture_viewport", None)
        if callable(set_viewport):
            set_viewport(frame.capture_viewport)
        if frame.capture_lost:
            self.discard_obsolete(clear_items=True)
            return []
        self.stats.frames_seen += 1
        if force:
            self._begin_replacement(token)
            self.differ = FrameDiffer(self.config.diff)
            self.differ.prime(frame)
            rois = [Rect(0, 0, frame.image.shape[1], frame.image.shape[0])]
        else:
            rois = self.differ.update(frame)

        # A large change needs a full replacement pass. Keep the previous
        # translations visible until OCR and translation finish, then replace
        # them atomically; expiring them on a short timer makes the overlay
        # blink while the slower full-frame OCR pass is still running.
        if not force and getattr(self.differ, "last_tick_had_large_change", False):
            force = True
            self._begin_replacement(token)
            self.differ.prime(frame)
            rois = [Rect(0, 0, frame.image.shape[1], frame.image.shape[0])]

        if not rois:
            return list(self.active_items)

        # A change that covers most of the frame (page turn, big jump) has the
        # same effect even when the tile ratio stays under the scroll guard.
        # Only stale items are faded — the very first full-frame ROI of a
        # session must pass through untouched.
        frame_area = max(1, frame.image.shape[0] * frame.image.shape[1])
        union = rois[0]
        for roi in rois[1:]:
            union = union.union(roi)
        if (
            not force
            and not frame.dirty_rects_stable
            and union.area > frame_area * 0.45
            and self.active_items
        ):
            force = True
            self._begin_replacement(token)
            self.differ.prime(frame)
            rois = [Rect(0, 0, frame.image.shape[1], frame.image.shape[0])]

        self.stats.frames_with_rois += 1
        token.commit(lambda: self._remove_items_overlapping(rois))

        candidates = rois if getattr(self.ocr, "detects_text", False) else self.detector.detect(frame.image, rois)
        if force and not candidates:
            candidates = rois
        if force and len(rois) == 1:
            full_frame = rois[0]
            if (
                full_frame.x == 0
                and full_frame.y == 0
                and full_frame.width == frame.image.shape[1]
                and full_frame.height == frame.image.shape[0]
            ):
                # A single full-page OCR crop gets downscaled to the model's
                # input limit, which loses small lettering in distant balloons.
                # Scan overlapping native-resolution tiles, then reconcile all
                # detections in page coordinates in the OCR engine.
                candidates = _tile_full_frame_for_ocr(
                    frame.image.shape[1],
                    frame.image.shape[0],
                    max(1, int(self.config.ocr.max_input_side)),
                )
        if not candidates:
            self._publish([], token)
            return list(self.active_items)

        ocr_started = time.perf_counter()
        regions = self._ocr_cached(frame.image, candidates, token)
        if force and not regions and candidates != rois:
            regions = self._ocr_cached(frame.image, rois, token)
        regions = self._recover_with_detector(frame.image, regions, rois, force, token)
        base_ocr_ms = (time.perf_counter() - ocr_started) * 1000
        token.check()
        refine_started = time.perf_counter()
        supports_text_detection = getattr(self.ocr, "detects_text", False)
        bubbles = (
            detect_enclosed_speech_bubbles(frame.image)
            if regions or supports_text_detection
            else []
        )
        if not force and bubbles:
            bubbles = [
                bubble
                for bubble in bubbles
                if any(bubble.bounds.intersects(roi) for roi in rois)
            ]
        rescued_count = 0
        if bubbles and supports_text_detection:
            # Re-read only changed balloons whose page-level OCR is absent,
            # weak, or suspiciously short. Crops are bounded to balloon
            # interiors, so the retry stays incremental in stream mode.
            rescued = self._rescue_unread_bubbles(regions, bubbles, frame.image, token)
            if rescued:
                regions.extend(rescued)
                rescued_count = len(rescued)
        regions = _merge_regions_in_bubbles(regions, bubbles, frame.image)
        regions = [
            region for region in regions
            if _region_inside_capture(
                region, frame.image.shape[1], frame.image.shape[0]
            )
        ]
        regions = self._refine_bubble_ocr(regions, frame.image, token)
        regions = self._refine_loose_text_ocr(regions, frame.image, token)
        refine_ms = (time.perf_counter() - refine_started) * 1000
        # Apply quality filtering to full-frame passes too: forced scans are
        # more likely to include isolated glyphs (page art, sound effects,
        # lettering fragments). Keep the full result set for those scans;
        # ordinary incremental scans retain the existing region cap.
        regions = _filter_text_regions(
            regions,
            frame.image.shape[1],
            frame.image.shape[0],
            # Dense pages have more than 6 balloons; a low cap silently left
            # the rest untranslated during scroll. Translation is batched, so
            # a higher cap costs almost nothing.
            max_regions=None if force else 16,
        )
        # A viewport edge can cut a line or the restored patch around it.
        # Wait until scrolling brings the whole text block into view instead
        # of translating a fragment and dragging a hard-edged patch along.
        regions = [
            region for region in regions
            if _region_inside_capture(
                region, frame.image.shape[1], frame.image.shape[0]
            )
        ]
        for region in regions:
            sampled_background = sample_background_color(frame.image, region.bbox)
            if sampled_background is not None:
                # Bubble grouping can supply a default white fill for layout,
                # but use the actual pixels around this lettering for erasing.
                region.metadata["background_color"] = sampled_background
            else:
                region.metadata.setdefault("background_color", None)
            background = region.metadata.get("background_color")
            background_luminance = (
                0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
                if background is not None else 0
            )
            # Erase only dark glyph pixels inside OCR polygons on a clearly
            # light local background. Requiring a detected bubble here left
            # source lettering visible under translations whenever bubble
            # detection missed a border or grouped text as a loose region.
            if (
                region.polygons
                and background_luminance >= 205
            ):
                bubble_bounds = region.metadata.get("display_bbox")
                # A bubble crop can expose source lines the OCR still missed.
                # Remove only small dark glyph components inside the detected
                # bright interior; do not cover the balloon with a solid box.
                restored = (
                    inpaint_bubble_glyphs(frame.image, bubble_bounds)
                    if isinstance(bubble_bounds, Rect)
                    else None
                )
                if restored is None:
                    restored = inpaint_text_background(
                        frame.image, region.bbox, padding=8, polygons=region.polygons
                    )
                if restored is not None:
                    region.metadata["background_rect"], region.metadata["background_patch"] = restored
        self.stats.ocr_regions += len(regions)
        self.stats.last_ocr_ms = base_ocr_ms
        self.stats.last_refine_ms = refine_ms
        self.stats.last_candidates = len(candidates)
        self.stats.last_bubbles = len(bubbles)
        self.stats.last_rescued = rescued_count
        if force:
            print(
                f"[ocr] frame={frame.index} base_ms={base_ocr_ms:.0f} "
                f"refine_ms={refine_ms:.0f} bubbles={len(bubbles)} "
                f"rescued={rescued_count} regions={len(regions)}",
                flush=True,
            )

        valid_regions = [region for region in regions if region.text.strip()]
        if valid_regions and isinstance(self.translator, EchoTranslator):
            # Source-language balloons belong only to the explicit OCR mode.
            now = time.time()
            provisional = [
                OverlayItem(
                    bbox=region.bbox,
                    text=region.text,
                    source_text=region.text,
                    confidence=region.confidence,
                    expires_at=now + min(20, self.config.overlay.ttl_seconds),
                    visual_hash=region.visual_hash,
                    background_color=region.metadata.get("background_color"),
                    background_patch=region.metadata.get("background_patch"),
                    background_rect=region.metadata.get("background_rect"),
                    display_bbox=region.metadata.get("display_bbox"),
                )
                for region in valid_regions
            ]
            self._publish(provisional, token)

        new_items = self._translate_regions(valid_regions, token)
        items_to_add = [item for item in new_items if item is not None]
        if items_to_add or (frame.dirty_rects_stable and valid_regions):
            self._publish(items_to_add, token)
        elif not valid_regions:
            # A forced viewport replacement that genuinely contains no text
            # must still finish, otherwise a suspended old generation would
            # remain hidden forever.
            self._publish([], token)
        return list(self.active_items)

    def _publish(self, items: list[OverlayItem], token: WorkToken) -> None:
        def publish() -> None:
            if self._replacement_generation == token.generation:
                staged = self._staged_items.pop(token.generation, [])
                self.active_items = self._dedupe_items([*staged, *items])
                self._replacement_generation = None
            else:
                self._upsert_overlay_items(items)
            # Canonical list stays as-is (identity/caching); the overlay gets a
            # collision-merged view so stacked captions never paint on top of
            # each other when bubble grouping missed.
            self.overlay.update(self._merge_overlapping_items(list(self.active_items)))
        token.commit(publish)

    def _begin_replacement(self, token: WorkToken) -> None:
        with self._items_lock:
            self._replacement_generation = token.generation
            self._staged_items = {token.generation: []}

    def _stage_replacement(self, items: list[OverlayItem], token: WorkToken) -> None:
        with self._items_lock:
            if self._replacement_generation == token.generation:
                self._staged_items[token.generation] = self._dedupe_items(items)

    def discard_obsolete(self, *, clear_items: bool = True) -> None:
        with self.view.lock:
            for future, token in list(self._futures.items()):
                if not token.current():
                    future.cancel()
            self._in_flight = {
                key for key in self._in_flight if key[0] == self.view.generation
            }
            self._replacement_generation = None
            self._staged_items.clear()
            if clear_items:
                with self._items_lock:
                    self.active_items = []
                self.overlay.update([])

    def shift_viewport_items(
        self, dy: int, width: int, height: int, dx: int = 0
    ) -> None:
        """Keep translated bubbles attached to their page content during scroll."""
        if (dx == 0 and dy == 0) or width <= 0 or height <= 0:
            return
        viewport = Rect(0, 0, width, height)

        def shifted(rect: Rect | None) -> Rect | None:
            if rect is None:
                return None
            return Rect(rect.x + dx, rect.y + dy, rect.width, rect.height)

        with self._items_lock:
            moved: list[OverlayItem] = []
            for item in self.active_items:
                bbox = shifted(item.bbox)
                if bbox is None or not bbox.intersects(viewport):
                    continue
                moved.append(
                    replace(
                        item,
                        bbox=bbox,
                        display_bbox=shifted(item.display_bbox),
                        background_rect=shifted(item.background_rect),
                    )
                )
            self.active_items = moved

    def _roi_key(self, image: object, rect: Rect) -> str:
        crop = crop_image(image, rect)
        digest = hashlib.blake2b(crop.tobytes(), digest_size=16).hexdigest()
        return f'{crop.shape}:{crop.dtype}:{digest}'

    def _ocr_cached(
        self,
        image: object,
        candidates: list[Rect],
        token: WorkToken | None = None,
    ) -> list[TextRegion]:
        """OCR with an exact-content cache: identical candidate content (same
        hash) returns the stored regions instead of re-running the model."""
        results: list[TextRegion] = []
        misses: list[Rect] = []
        # Hash each crop once; misses reuse the key when they are stored.
        miss_keys: list[str] = []
        for rect in candidates:
            if token:
                token.check()
            key = self._roi_key(image, rect)
            cached = self.roi_ocr_cache.get(key)
            if cached is None:
                misses.append(rect)
                miss_keys.append(key)
                continue
            self.roi_ocr_cache.move_to_end(key)
            results.extend(
                TextRegion(
                    bbox=Rect(rect.x + region.bbox.x, rect.y + region.bbox.y, region.bbox.width, region.bbox.height),
                    text=region.text,
                    confidence=region.confidence,
                    source_lang=region.source_lang,
                    visual_hash=region.visual_hash,
                    metadata=dict(region.metadata),
                    polygons=tuple(
                        tuple((rect.x + x, rect.y + y) for x, y in polygon)
                        for polygon in region.polygons
                    ),
                )
                for region in cached
            )

        if misses:
            if token and hasattr(self.ocr, "recognize_cancellable"):
                fresh = self.ocr.recognize_cancellable(image, misses, token.check)
            else:
                fresh = self.ocr.recognize(image, misses)
            if token:
                token.check()
            keys_by_rect = miss_keys
            partitions: list[list[TextRegion]] = [[] for _ in misses]
            for region in fresh:
                best_index = None
                best_overlap = -1
                for index, rect in enumerate(misses):
                    inter_w = min(region.bbox.x2, rect.x2) - max(region.bbox.x, rect.x)
                    inter_h = min(region.bbox.y2, rect.y2) - max(region.bbox.y, rect.y)
                    overlap = max(0, inter_w) * max(0, inter_h)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_index = index
                if best_index is not None:
                    partitions[best_index].append(region)
            for key, origin, parts in zip(keys_by_rect, misses, partitions):
                self.roi_ocr_cache[key] = [
                    TextRegion(
                        bbox=Rect(region.bbox.x - origin.x, region.bbox.y - origin.y, region.bbox.width, region.bbox.height),
                        text=region.text,
                        confidence=region.confidence,
                        source_lang=region.source_lang,
                        visual_hash=region.visual_hash,
                        metadata=dict(region.metadata),
                        polygons=tuple(
                            tuple((x - origin.x, y - origin.y) for x, y in polygon)
                            for polygon in region.polygons
                        ),
                    )
                    for region in parts
                ]
            while len(self.roi_ocr_cache) > self.roi_ocr_cache_size:
                self.roi_ocr_cache.popitem(last=False)
            results.extend(fresh)
        return results

    def _recover_with_detector(
        self,
        image: object,
        regions: list[TextRegion],
        rois: list[Rect],
        force: bool,
        token: WorkToken,
    ) -> list[TextRegion]:
        """Re-read text blocks the comic detector found but OCR missed.

        Page-level OCR misses lettering on dark screentone or reads only the
        first line of a block ("IF IT" of "IF IT WERE ME"). A tight crop of
        the detected block gives the recognizer a clean second chance; the
        longer reading replaces the partial one.
        """
        from manga_rt_translator.comic_detector import detect_cached

        detections = detect_cached(image)  # type: ignore[arg-type]
        if detections is None or not detections.texts:
            return regions
        height, width = image.shape[:2]  # type: ignore[attr-defined]
        crops = _detector_recovery_crops(
            regions, detections.texts, None if force else rois, width, height
        )
        if not crops:
            return regions
        token.check()
        result = list(regions)
        for crop in crops:
            fresh = [
                region for region in self._ocr_cached(image, [crop], token)
                if region.text.strip() and sum(c.isalpha() for c in region.text) >= 2
            ]
            if not fresh:
                continue
            members = [r for r in result if _center_inside(r.bbox, crop)]
            old_len = sum(len(r.text.strip()) for r in members)
            new_len = sum(len(r.text.strip()) for r in fresh)
            if new_len <= old_len:
                continue
            result = [r for r in result if r not in members] + fresh
        return result

    def _rescue_unread_bubbles(
        self,
        regions: list[TextRegion],
        bubbles: list[SpeechBubbleRegion],
        image: object,
        token: WorkToken,
    ) -> list[TextRegion]:
        """Run one small OCR pass for bright balloons with no OCR evidence."""
        image_height, image_width = image.shape[:2]
        missing: list[Rect] = []
        for bubble in sorted(bubbles, key=lambda item: item.bounds.area, reverse=True):
            bounds = bubble.bounds
            if (
                bounds.x < 8 or bounds.y < 8
                or bounds.x2 > image_width - 8 or bounds.y2 > image_height - 8
            ):
                continue
            if (
                bounds.width < 54 or bounds.height < 52
                or not 0.48 <= bounds.width / max(1, bounds.height) <= 2.1
            ):
                continue
            interior = image[bounds.y : bounds.y2, bounds.x : bounds.x2, :3]
            dark_ratio = float(np.mean(np.max(interior, axis=2) < 150))
            # A bubble is mostly light paper with small dark lettering. These
            # limits skip faces/clothing fragments that the bright-component
            # detector can mistake for a bubble, saving several OCR passes.
            if dark_ratio > 0.22:
                continue
            members: list[TextRegion] = []
            for region in regions:
                box = region.bbox
                center_x = box.x + box.width // 2
                center_y = box.y + box.height // 2
                if bounds.x <= center_x < bounds.x2 and bounds.y <= center_y < bounds.y2:
                    members.append(region)
            best = max(
                members,
                key=lambda item: (item.confidence, len(item.text.strip())),
                default=None,
            )
            word_count = len(best.text.split()) if best else 0
            # A single OCR word in a tall balloon is often just the first line.
            # Re-read it even when the recognizer assigned that fragment high
            # confidence; this is the failure mode seen in the user's screenshot.
            if (
                best is not None
                and best.confidence >= 0.86
                and (word_count >= 3 or bounds.height < 58)
            ):
                continue
            crop = bounds
            if crop.width < 30 or crop.height < 22 or crop.area > 90_000:
                continue
            missing.append(crop)
        rescued: list[TextRegion] = []
        for crop in missing:
            token.check()
            candidates = self._ocr_cached(image, [crop], token)
            for region in candidates:
                if (
                    region.text.strip()
                    and sum(char.isalpha() for char in region.text) >= 3
                    and region.bbox.area <= crop.area * 0.80
                ):
                    rescued.append(region)
        return rescued

    def _refine_bubble_ocr(
        self, regions: list[TextRegion], image: object, token: WorkToken
    ) -> list[TextRegion]:
        """Re-read balloon lettering in a tight crop before translation.

        Full-page OCR tiles can recognize the balloon but misread individual
        letters. A crop around its detected text removes surrounding artwork
        and gives the recognizer a much cleaner second pass.
        """
        if not getattr(self.ocr, "detects_text", False):
            return regions
        image_height, image_width = image.shape[:2]
        uncertain = sorted(
            (
                region for region in regions
                if isinstance(region.metadata.get("display_bbox"), Rect)
                and len(region.text.strip()) >= 4
                and region.confidence < 0.60
            ),
            key=lambda region: region.confidence,
        )[:1]
        for region in uncertain:
            bubble = region.metadata.get("display_bbox")
            assert isinstance(bubble, Rect)
            # Re-read just the detected lettering, never the whole balloon.
            # A single-line caption in a tall balloon is not a fragment.
            padded = region.bbox.expanded(5, image_width, image_height)
            x1, y1 = max(padded.x, bubble.x), max(padded.y, bubble.y)
            x2, y2 = min(padded.x2, bubble.x2), min(padded.y2, bubble.y2)
            crop = Rect(x1, y1, x2 - x1, y2 - y1)
            if crop.width < 24 or crop.height < 18 or crop.area > 250_000:
                continue
            retry = self._ocr_cached(image, [crop], token)
            retry = [
                item for item in retry
                if item.text.strip() and item.bbox.intersects(region.bbox)
            ]
            if retry:
                retry.sort(key=lambda item: (item.bbox.y, item.bbox.x))
                corrected = " ".join(item.text.strip() for item in retry)
                if len(corrected) >= max(2, int(len(region.text) * 0.55)):
                    confidence = min(item.confidence for item in retry)
                    if confidence > region.confidence or len(corrected) > len(region.text):
                        region.text = corrected
                        region.confidence = confidence
        return regions

    def _refine_loose_text_ocr(
        self, regions: list[TextRegion], image: object, token: WorkToken
    ) -> list[TextRegion]:
        """Recover lettering clipped by OCR tiles on dark manga artwork."""
        if not getattr(self.ocr, "detects_text", False):
            return regions
        image_height, image_width = image.shape[:2]
        uncertain = sorted(
            (
                region for region in regions
                if region.metadata.get("display_bbox") is None
                and region.polygons and region.confidence < 0.52
            ),
            key=lambda region: region.confidence,
        )[:1]
        for region in uncertain:
            background = sample_background_color(image, region.bbox)
            if (background is None or max(background) >= 100
                    or region.bbox.area > 30_000):
                continue
            box = region.bbox
            x1, y1 = max(0, box.x - 45), max(0, box.y - 18)
            x2, y2 = min(image_width, box.x2 + 45), min(image_height, box.y2 + 18)
            if (x2 - x1) * (y2 - y1) > 80_000:
                continue
            recovered = self._ocr_cached(image, [Rect(x1, y1, x2 - x1, y2 - y1)], token)
            recovered = [
                item for item in recovered
                if item.text.strip() and item.bbox.intersects(box)
            ]
            if not recovered:
                continue
            recovered.sort(key=lambda item: (item.bbox.y, item.bbox.x))
            corrected = " ".join(item.text.strip() for item in recovered)
            original_compact = "".join(
                char for char in region.text.casefold() if char.isalnum()
            )
            corrected_compact = "".join(
                char for char in corrected.casefold() if char.isalnum()
            )
            repaired_spacing = (
                original_compact == corrected_compact
                and len(corrected.split()) > len(region.text.split())
            )
            if not repaired_spacing and len(corrected) < len(region.text) + 4:
                continue
            recovered_box = recovered[0].bbox
            for item in recovered[1:]:
                recovered_box = recovered_box.union(item.bbox)
            if recovered_box.area > box.area * 2.5:
                continue
            region.bbox = recovered_box
            region.text = corrected
            region.confidence = min(item.confidence for item in recovered)
            region.polygons = tuple(poly for item in recovered for poly in item.polygons)
            region.visual_hash = perceptual_region_hash(image, recovered_box)
        return regions

    def _translate_regions(
        self,
        regions: list[TextRegion],
        token: WorkToken | None = None,
    ) -> list[OverlayItem | None]:
        """Cache-first, batch-second translation flow.

        Cache lookups run per region; the misses are grouped by source
        language and sent in as few provider calls as possible, so a page
        with five bubbles costs one request instead of five.

        In async mode the provider calls run on a background thread: the
        caller gets provisional (source-text) items immediately and the
        translation replaces them when it lands.
        """
        token = token or self.view.token()
        token.check()
        target_lang = self.config.translation.target_lang
        default_source = self.config.ocr.source_lang
        results: list[str | None] = [None] * len(regions)
        misses: list[int] = []
        cache_hits_before = self.stats.cache_hits

        for index, region in enumerate(regions):
            source_lang = region.source_lang or default_source
            pair = self._language_pair(source_lang, target_lang)
            visual_hit = self.memory.get_visual(region.visual_hash, pair)
            if (
                visual_hit
                and visual_hit.translated_text.strip()
                and visual_hit.normalized_text == normalize_text(region.text)
            ):
                self.stats.cache_hits += 1
                results[index] = visual_hit.translated_text
                continue
            text_hit = self.memory.lookup(region.text, pair)
            if (
                text_hit
                and text_hit.translated_text.strip()
                and text_hit.score == 100.0
            ):
                self.stats.cache_hits += 1
                results[index] = text_hit.translated_text
            else:
                misses.append(index)

        get_rate_tracker().record_cache_activity(
            self.stats.cache_hits - cache_hits_before, len(misses)
        )

        misses_by_lang: dict[str, list[int]] = {}
        for index in misses:
            source_lang = regions[index].source_lang or default_source
            misses_by_lang.setdefault(source_lang, []).append(index)

        if self.async_translate and misses_by_lang:
            pending: list[int] = []
            for index in misses:
                visual_hash = regions[index].visual_hash
                if visual_hash and (token.generation, visual_hash) in self._in_flight:
                    # A translation for this exact bubble is already on the
                    # wire; keep showing the provisional text until it lands.
                    continue
                pending.append(index)
            if pending:
                pending_by_lang: dict[str, list[int]] = {}
                for index in pending:
                    source_lang = regions[index].source_lang or default_source
                    pending_by_lang.setdefault(source_lang, []).append(index)
                for index in pending:
                    if regions[index].visual_hash:
                        self._in_flight.add((token.generation, regions[index].visual_hash))
                cached_items = [
                    item
                    for item in self._build_items(
                        regions, results, provisional_for_misses=False
                    )
                    if item is not None
                ]
                if self._replacement_generation == token.generation:
                    self._stage_replacement(cached_items, token)
                self._dispatch_translation(regions, pending_by_lang, target_lang, token)
                if self._replacement_generation == token.generation:
                    return []
            return self._build_items(regions, results, provisional_for_misses=False)

        if misses_by_lang:
            translated = self._translate_misses(
                regions, misses_by_lang, target_lang, token
            )
            for index, text in translated.items():
                results[index] = text

        return self._build_items(regions, results, provisional_for_misses=False)

    def _build_items(
        self,
        regions: list[TextRegion],
        results: list[str | None],
        provisional_for_misses: bool,
    ) -> list[OverlayItem | None]:
        now = time.time()
        items: list[OverlayItem | None] = []
        for index, region in enumerate(regions):
            self.context_lines.append(region.text)
            translated = results[index]
            if translated is None:
                if provisional_for_misses:
                    items.append(
                        OverlayItem(
                            bbox=region.bbox,
                            text=region.text,
                            source_text=region.text,
                            confidence=region.confidence,
                            expires_at=now + min(12, self.config.overlay.ttl_seconds),
                            visual_hash=region.visual_hash,
                            background_color=region.metadata.get("background_color"),
                            background_patch=region.metadata.get("background_patch"),
                            background_rect=region.metadata.get("background_rect"),
                            display_bbox=region.metadata.get("display_bbox"),
                        )
                    )
                else:
                    items.append(None)
                continue
            items.append(
                OverlayItem(
                    bbox=region.bbox,
                    text=translated,
                    source_text=region.text,
                    confidence=region.confidence,
                    expires_at=now + self.config.overlay.ttl_seconds,
                    visual_hash=region.visual_hash,
                    background_color=region.metadata.get("background_color"),
                    background_patch=region.metadata.get("background_patch"),
                    background_rect=region.metadata.get("background_rect"),
                    display_bbox=region.metadata.get("display_bbox"),
                )
            )
        if len(self.context_lines) > 20:
            self.context_lines = self.context_lines[-20:]
        return items

    def _translate_misses(
        self,
        regions: list[TextRegion],
        misses_by_lang: dict[str, list[int]],
        target_lang: str,
        token: WorkToken | None = None,
    ) -> dict[int, str]:
        """Run the provider chain on the cache misses and persist results.
        Called on the background thread in async mode."""
        translated: dict[int, str] = {}
        history_entries: list[tuple[str, str, str, str]] = []
        page_context = "\n".join(
            region.text.strip() for region in regions[:12] if region.text.strip()
        )[:1200]
        for source_lang, indices in misses_by_lang.items():
            if token:
                token.check()
            try:
                method = self.translator.translate_batch
                extra = {}
                if token and hasattr(self.translator, 'translate_batch_cancellable'):
                    method = self.translator.translate_batch_cancellable
                    extra['check'] = token.check
                batch = method(
                    texts=[regions[index].text for index in indices],
                    source_lang=source_lang,
                    target_lang=target_lang,
                    glossary=self.glossary,
                    context=page_context,
                    **extra,
                )
                self.stats.translate_calls += 1
            except WorkCancelled:
                raise
            except Exception:
                # No provider answered; keep the source text visible instead
                # of dropping the bubble.
                batch = [
                    TranslationResult(text=regions[index].text, provider="echo")
                    for index in indices
                ]
            if token:
                token.check()
            for index, result in zip(indices, batch):
                if not result.text.strip():
                    continue
                pair = self._language_pair(source_lang, target_lang)
                # Echo is a display fallback, not a translation. Persisting it
                # poisons the translation memory with source-language text and
                # prevents a later provider retry from becoming authoritative.
                selected_provider = str(self.config.translation.provider or "google").lower()
                # A temporary fallback may be less fluent than the selected
                # provider. Show it now, but let the preferred provider retry
                # on the next scan instead of keeping that fallback forever.
                cacheable = result.provider == selected_provider
                if target_lang.lower().startswith("pt"):
                    cacheable = cacheable and (
                        normalize_text(result.text)
                        != normalize_text(regions[index].text)
                    )
                if cacheable:
                    self.memory.save(
                        original_text=regions[index].text,
                        translated_text=result.text,
                        language_pair=pair,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                    self.memory.save_visual(
                        visual_hash=regions[index].visual_hash,
                        original_text=regions[index].text,
                        translated_text=result.text,
                        language_pair=pair,
                        bbox=regions[index].bbox.as_tuple(),
                    )
                    history_entries.append(
                        (regions[index].text, result.text, pair, result.provider)
                    )
                if result.provider == "echo" and not isinstance(
                    self.translator, EchoTranslator
                ):
                    self.stats.last_error = 'Tradutor indisponivel. Tente traduzir a regiao novamente.'
                    continue
                self.stats.last_error = ''
                translated[index] = result.text
        self.memory.record_history_batch(history_entries)
        return translated

    def _language_pair(self, source_lang: str, target_lang: str) -> str:
        pair = f"{source_lang}->{target_lang}"
        scope = self.glossary.cache_scope
        selected_provider = str(self.config.translation.provider or "google").lower()
        if selected_provider != "google":
            pair = f"provider:{selected_provider}|{pair}"
        return f"{scope}|{pair}" if scope else pair

    def _dispatch_translation(
        self,
        regions: list[TextRegion],
        misses_by_lang: dict[str, list[int]],
        target_lang: str,
        token: WorkToken,
    ) -> None:
        """Queue the provider call on a background thread. When it finishes
        the overlay is refreshed from this thread (the Qt bridge signal is
        thread-safe)."""
        with self.view.lock:
            if self._closed or not token.current():
                return
            if self._translate_executor is None:
                self._translate_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="translate"
                )
            for future, previous in list(self._futures.items()):
                if not previous.current():
                    future.cancel()
            future = self._translate_executor.submit(
                self._run_async_translation, regions, misses_by_lang, target_lang, token
            )
            self._futures[future] = token
            future.add_done_callback(self._translation_finished)

    def replace_translator(self, translator: Translator) -> None:
        with self.view.lock:
            if self._closed:
                raise RuntimeError("Translation pipeline is closed")
            executor = self._translate_executor
            if executor is not None:
                if any(not future.done() for future in self._futures):
                    self._retired_translate_executors.append(executor)
                executor.shutdown(wait=False, cancel_futures=True)
                self._translate_executor = None
            self.translator = translator

    def _translation_finished(self, future: Future) -> None:
        with self.view.lock:
            self._futures.pop(future, None)

    def _run_async_translation(
        self,
        regions: list[TextRegion],
        misses_by_lang: dict[str, list[int]],
        target_lang: str,
        token: WorkToken,
    ) -> None:
        try:
            t0 = time.time()
            token.check()
            translated = self._translate_misses(regions, misses_by_lang, target_lang, token)
            elapsed = (time.time() - t0) * 1000
            self.stats.last_translation_ms = elapsed
            self.stats.last_translated_regions = len(translated)
            results: list[str | None] = [None] * len(regions)
            for index, text in translated.items():
                results[index] = text
            token.check()
            items = self._build_items(regions, results, provisional_for_misses=False)
            self._publish([item for item in items if item is not None], token)
            print(
                f"[translate] async {len(translated)} regions in {elapsed:.0f}ms",
                flush=True,
            )
        except WorkCancelled:
            self.stats.stale_results += 1
        except Exception as exc:
            self.stats.last_error = f"async: {type(exc).__name__}: {exc}"
        finally:
            with self.view.lock:
                for indices in misses_by_lang.values():
                    for index in indices:
                        self._in_flight.discard((token.generation, regions[index].visual_hash))

    def close(self, wait: bool = False) -> None:
        with self.view.lock:
            self._closed = True
            self.view.close()
            executors = list(self._retired_translate_executors)
            self._retired_translate_executors.clear()
            if self._translate_executor is not None:
                executors.append(self._translate_executor)
            self._translate_executor = None
        for executor in executors:
            executor.shutdown(wait=wait, cancel_futures=True)

    def _remove_items_overlapping(self, rois: list[Rect]) -> None:
        with self._items_lock:
            # Small capture noise, lazy-load repainting, and sub-pixel browser
            # movement can mark a ROI dirty after its translation already
            # arrived. Keep the real translation on screen until the fresh
            # result replaces it; removing it here makes the overlay oscillate
            # between Portuguese and provisional OCR text. Large viewport
            # changes use a staged full-frame replacement instead.
            self.active_items = [
                item
                for item in self.active_items
                if (
                    not any(item.bbox.intersects(roi) for roi in rois)
                    or (
                        bool(item.source_text) and item.text != item.source_text
                    )
                )
            ]

    def _upsert_overlay_items(self, items: list[OverlayItem]) -> None:
        now = time.time()
        with self._items_lock:
            live = [
                item
                for item in self.active_items
                if item.expires_at is None or item.expires_at > now
            ]
            for new_item in items:
                # A later sweep can still be waiting on the provider and
                # return the source text as a provisional item. Never let that
                # provisional update overwrite a translation that already
                # arrived for the same bubble, or the overlay will flicker
                # between English and Portuguese.
                is_provisional = (
                    bool(new_item.source_text) and new_item.text == new_item.source_text
                )
                if is_provisional and any(
                    self._same_overlay_item(item, new_item)
                    and not (
                        bool(item.source_text) and item.text == item.source_text
                    )
                    for item in live
                ):
                    continue
                live = [
                    item for item in live
                    if not self._same_overlay_item(item, new_item)
                ]
                live.append(new_item)
            self.active_items = self._dedupe_items(live)

    @staticmethod
    def _same_overlay_item(left: OverlayItem, right: OverlayItem) -> bool:
        if left.visual_hash and right.visual_hash:
            return left.visual_hash == right.visual_hash
        if left.source_text and right.source_text:
            if normalize_text(left.source_text) == normalize_text(right.source_text):
                return left.bbox.intersects(right.bbox)
        return left.bbox.intersects(right.bbox)

    @classmethod
    def _dedupe_items(cls, items: list[OverlayItem]) -> list[OverlayItem]:
        result: list[OverlayItem] = []
        for item in items:
            result = [
                existing for existing in result
                if not cls._same_overlay_item(existing, item)
            ]
            result.append(item)
        return result

    @staticmethod
    def _paint_rect(bbox: Rect) -> Rect:
        """Approximate the on-screen caption rectangle the overlay paints for a
        bbox (see overlay._fit_rect): a small OCR box is grown to a minimum
        card size plus padding. Collision is judged on these painted rects, not
        the raw OCR boxes, because two tiny non-touching boxes still overlap
        once each is grown to caption size."""
        width = max(bbox.width, 110) + 16
        height = max(bbox.height, 52) + 20
        return Rect(x=bbox.x - 8, y=bbox.y - 10, width=width, height=height)

    @classmethod
    def _merge_overlapping_items(cls, items: list[OverlayItem]) -> list[OverlayItem]:
        """Display-time safety net against stacked captions.

        When bubble grouping misses (an open or large balloon whose lines were
        not merged into one block), each line becomes its own caption and the
        painted cards pile up on top of each other. This merges any group of
        captions whose painted rects overlap into a single caption: one box
        over the union, text joined in reading order (top-to-bottom,
        left-to-right — the English source path). Balloons that already render
        as one caption never overlap, so they pass through untouched. This only
        changes what is painted; the canonical item list (identity, caching) is
        left alone by the caller."""
        translations = [
            item for item in items
            if not (item.source_text and item.text == item.source_text)
        ]
        if len(translations) < 2:
            return items
        passthrough = [item for item in items if item not in translations]

        parent = list(range(len(translations)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        rects = [cls._paint_rect(item.bbox) for item in translations]
        for i in range(len(translations)):
            for j in range(i + 1, len(translations)):
                if cls._rects_collide(rects[i], rects[j]):
                    parent[find(j)] = find(i)

        groups: dict[int, list[int]] = {}
        for idx in range(len(translations)):
            groups.setdefault(find(idx), []).append(idx)

        merged: list[OverlayItem] = []
        for members in groups.values():
            if len(members) == 1:
                merged.append(translations[members[0]])
                continue
            group_items = cls._drop_superseded([translations[i] for i in members])
            if len(group_items) == 1:
                merged.append(group_items[0])
                continue
            merged.append(cls._merge_item_group(group_items))
        return passthrough + merged

    @staticmethod
    def _drop_superseded(group_items: list[OverlayItem]) -> list[OverlayItem]:
        """Drop captions that are older/partial readings of a bigger one.

        The same balloon read twice (a fragment first, the whole block later)
        leaves two captions on top of each other; joining them repeated the
        sentence ("VOCE E MEU AMIGO... VOCE E MEU AMIGO"). A caption mostly
        inside a larger one, or with the same text, is the same balloon.
        """
        ordered = sorted(group_items, key=lambda it: it.bbox.area, reverse=True)
        kept: list[OverlayItem] = []
        for item in ordered:
            box = item.bbox
            text = " ".join(item.text.casefold().split())
            superseded = False
            for big in kept:
                inter_w = min(box.x2, big.bbox.x2) - max(box.x, big.bbox.x)
                inter_h = min(box.y2, big.bbox.y2) - max(box.y, big.bbox.y)
                inside = (
                    max(0, inter_w) * max(0, inter_h) >= 0.65 * max(1, box.area)
                )
                same_text = text and text == " ".join(big.text.casefold().split())
                if inside or same_text:
                    superseded = True
                    break
            if not superseded:
                kept.append(item)
        return kept

    @staticmethod
    def _rects_collide(a: Rect, b: Rect) -> bool:
        inter_w = min(a.x2, b.x2) - max(a.x, b.x)
        inter_h = min(a.y2, b.y2) - max(a.y, b.y)
        if inter_w <= 0 or inter_h <= 0:
            return False
        inter = inter_w * inter_h
        return inter >= 0.30 * min(a.area, b.area)

    @staticmethod
    def _merge_item_group(group_items: list[OverlayItem]) -> OverlayItem:
        # Reading order: group into rough lines by vertical band, then order
        # each line left-to-right (matches the English source path).
        ordered = sorted(group_items, key=lambda it: (it.bbox.y, it.bbox.x))
        lines: list[list[OverlayItem]] = []
        for item in ordered:
            for line in lines:
                ref = line[0].bbox
                overlap = min(ref.y2, item.bbox.y2) - max(ref.y, item.bbox.y)
                if overlap >= 0.5 * min(ref.height, item.bbox.height):
                    line.append(item)
                    break
            else:
                lines.append([item])
        pieces: list[str] = []
        for line in lines:
            for item in sorted(line, key=lambda it: it.bbox.x):
                text = item.text.strip()
                if text:
                    pieces.append(text)
        union = group_items[0].bbox
        for item in group_items[1:]:
            union = union.union(item.bbox)
        background_color = next(
            (it.background_color for it in group_items if it.background_color), None
        )
        patch_rect, patch = _composite_background_patch(group_items, background_color)
        display_boxes = [it.display_bbox for it in group_items if it.display_bbox is not None]
        display_bbox = None
        if display_boxes:
            display_bbox = display_boxes[0]
            for box in display_boxes[1:]:
                display_bbox = display_bbox.union(box)
        return OverlayItem(
            bbox=union,
            text=" ".join(pieces),
            source_text="",
            confidence=min(it.confidence for it in group_items),
            expires_at=max((it.expires_at or 0.0) for it in group_items) or None,
            visual_hash="",
            background_color=background_color,
            # Keep the erased-lettering patches of every fragment; dropping
            # them left the English source visible under merged captions.
            background_patch=patch,
            background_rect=patch_rect,
            display_bbox=display_bbox,
        )


def _composite_background_patch(
    items: list[OverlayItem],
    background_color: tuple[int, int, int] | None,
) -> tuple[Rect | None, np.ndarray | None]:
    """Combine the fragments' erased-background patches into one patch.

    The canvas spans every patch and every fragment's text box; areas no patch
    covers get the local background colour, so no source glyph stays visible.
    """
    patches = [
        (it.background_rect, it.background_patch, it.bbox)
        for it in items
        if it.background_rect is not None and it.background_patch is not None
    ]
    if not patches:
        return None, None
    canvas_rect = patches[0][0]
    for rect, _, _ in patches[1:]:
        canvas_rect = canvas_rect.union(rect)
    for it in items:
        canvas_rect = canvas_rect.union(it.bbox)
    if canvas_rect.width <= 0 or canvas_rect.height <= 0:
        return None, None
    if background_color is None:
        background_color = tuple(
            int(v) for v in np.median(
                np.concatenate(
                    [np.asarray(p)[..., :3].reshape(-1, 3) for _, p, _ in patches]
                ),
                axis=0,
            )
        )
    # RGBA canvas, transparent by default: only real patches and the
    # fragments' own text boxes are painted. A flat-filled union rectangle
    # spilled white blocks over the artwork between and around the fragments.
    canvas = np.zeros((canvas_rect.height, canvas_rect.width, 4), dtype=np.uint8)
    fill = np.asarray(tuple(background_color) + (255,), dtype=np.uint8)
    for it in items:
        box = it.bbox
        x1, y1 = max(0, box.x - canvas_rect.x), max(0, box.y - canvas_rect.y)
        x2 = min(canvas_rect.width, box.x2 - canvas_rect.x)
        y2 = min(canvas_rect.height, box.y2 - canvas_rect.y)
        if x2 > x1 and y2 > y1:
            canvas[y1:y2, x1:x2] = fill
    for rect, patch, _ in patches:
        data = np.asarray(patch)
        x = rect.x - canvas_rect.x
        y = rect.y - canvas_rect.y
        h = min(data.shape[0], canvas_rect.height - y)
        w = min(data.shape[1], canvas_rect.width - x)
        if h > 0 and w > 0 and x >= 0 and y >= 0:
            canvas[y : y + h, x : x + w, :3] = data[:h, :w, :3]
            canvas[y : y + h, x : x + w, 3] = (
                data[:h, :w, 3] if data.shape[2] == 4 else 255
            )
    # A patch's padding can carry a neighbour's not-yet-erased glyphs. Re-apply
    # each fragment's own (erased) text box from its own patch on top.
    for rect, patch, box in patches:
        data = np.asarray(patch)
        x1 = max(box.x, rect.x)
        y1 = max(box.y, rect.y)
        x2 = min(box.x2, rect.x + data.shape[1])
        y2 = min(box.y2, rect.y + data.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue
        canvas[
            y1 - canvas_rect.y : y2 - canvas_rect.y,
            x1 - canvas_rect.x : x2 - canvas_rect.x,
            :3,
        ] = data[y1 - rect.y : y2 - rect.y, x1 - rect.x : x2 - rect.x, :3]
    return canvas_rect, np.ascontiguousarray(canvas)


def _merge_regions_in_bubbles(
    regions: list[TextRegion],
    bubbles: list[SpeechBubbleRegion],
    image: object,
) -> list[TextRegion]:
    """Translate all OCR lines inside one enclosed balloon as one passage."""
    grouped: dict[tuple[int, str | None], list[TextRegion]] = {}
    unmatched: list[TextRegion] = []
    for region in regions:
        box = region.bbox
        local_background = sample_background_color(image, box)
        local_luminance = (
            0.2126 * local_background[0]
            + 0.7152 * local_background[1]
            + 0.0722 * local_background[2]
            if local_background is not None else 0
        )
        # A bright connected component elsewhere in the panel is not enough
        # to call this a speech bubble. The OCR line itself must sit on a
        # light, mostly plain interior, otherwise keep its own text anchor.
        if local_luminance < 185:
            unmatched.append(region)
            continue
        center_x = box.x + box.width // 2
        center_y = box.y + box.height // 2
        choices: list[tuple[int, int]] = []
        for index, bubble in enumerate(bubbles):
            bounds = bubble.bounds
            if not (bounds.x <= center_x < bounds.x2 and bounds.y <= center_y < bounds.y2):
                continue
            overlap_width = max(0, min(box.x2, bounds.x2) - max(box.x, bounds.x))
            overlap_height = max(0, min(box.y2, bounds.y2) - max(box.y, bounds.y))
            if overlap_width * overlap_height < box.area * 0.55:
                continue
            choices.append((bounds.area, index))
        if not choices:
            unmatched.append(region)
            continue
        index = min(choices)[1]
        grouped.setdefault((index, region.source_lang), []).append(region)

    merged = list(unmatched)
    for (index, source_lang), members in grouped.items():
        bubble = bubbles[index]
        # Overlapping page tiles can report the same line twice with slightly
        # different OCR text. Keep the stronger/full passage before ordering.
        ranked = sorted(
            members,
            key=lambda item: (
                item.confidence
                + min(0.15, len(item.text.strip()) * 0.003)
                + min(0.10, len(item.text.split()) * 0.025),
                item.confidence,
            ),
            reverse=True,
        )
        unique: list[TextRegion] = []
        for candidate in ranked:
            duplicate = False
            for kept in unique:
                overlap_width = max(
                    0, min(candidate.bbox.x2, kept.bbox.x2)
                    - max(candidate.bbox.x, kept.bbox.x)
                )
                overlap_height = max(
                    0, min(candidate.bbox.y2, kept.bbox.y2)
                    - max(candidate.bbox.y, kept.bbox.y)
                )
                overlap = overlap_width * overlap_height
                if overlap >= min(candidate.bbox.area, kept.bbox.area) * 0.42:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(candidate)
        ordered = sorted(unique, key=lambda item: (item.bbox.y, item.bbox.x))
        bbox = ordered[0].bbox
        for member in ordered[1:]:
            bbox = bbox.union(member.bbox)
        background_color = sample_background_color(image, bbox)
        merged.append(
            TextRegion(
                bbox=bbox,
                text=" ".join(member.text.strip() for member in ordered if member.text.strip()),
                confidence=min(member.confidence for member in ordered),
                source_lang=source_lang,
                visual_hash=perceptual_region_hash(image, bbox),
                metadata={
                    "lines": sum(int(member.metadata.get("lines", 1)) for member in ordered),
                    "display_bbox": bubble.bounds,
                    "background_color": background_color,
                },
                polygons=tuple(
                    polygon for member in ordered for polygon in member.polygons
                ),
            )
        )
    return sorted(merged, key=lambda item: (item.bbox.y, item.bbox.x))


def _filter_text_regions(
    regions: Iterable[TextRegion],
    frame_width: int,
    frame_height: int,
    max_regions: int | None = 6,
) -> list[TextRegion]:
    # Keeps the incoming reading order produced by group_text_regions.
    filtered: list[TextRegion] = []
    seen: list[tuple[str, Rect]] = []
    frame_area = max(1, frame_width * frame_height)
    for region in regions:
        text = " ".join(region.text.split()).strip()
        if len(text) < 2:
            continue
        normalized = text.casefold()
        if any(
            normalized == previous_text and region.bbox.intersects(previous_bbox)
            for previous_text, previous_bbox in seen
        ):
            continue
        bbox_area_ratio = region.bbox.area / frame_area
        if bbox_area_ratio > 0.22:
            continue
        if region.bbox.width > int(frame_width * 0.78) and region.bbox.height < int(frame_height * 0.12):
            continue
        seen.append((normalized, region.bbox))
        region.text = text
        filtered.append(region)
    return filtered if max_regions is None else filtered[:max_regions]


def _center_inside(box: Rect, area: Rect) -> bool:
    cx = box.x + box.width / 2
    cy = box.y + box.height / 2
    return area.x <= cx < area.x2 and area.y <= cy < area.y2


def _detector_recovery_crops(
    regions: list[TextRegion],
    texts: list[Rect],
    rois: list[Rect] | None,
    width: int,
    height: int,
    max_crops: int = 6,
) -> list[Rect]:
    """Detected text blocks with no OCR reading, or less than half covered."""
    crops: list[Rect] = []
    for text in texts:
        if rois is not None and not any(text.intersects(roi) for roi in rois):
            continue
        covered = 0
        for region in regions:
            inter_w = min(region.bbox.x2, text.x2) - max(region.bbox.x, text.x)
            inter_h = min(region.bbox.y2, text.y2) - max(region.bbox.y, text.y)
            if inter_w > 0 and inter_h > 0:
                covered += inter_w * inter_h
        if covered >= 0.5 * max(1, text.area):
            continue
        crop = text.expanded(6, width, height)
        if crop.width < 12 or crop.height < 12:
            continue
        crops.append(crop)
    crops.sort(key=lambda r: r.area, reverse=True)
    return crops[:max_crops]


def _region_inside_capture(region: TextRegion, width: int, height: int) -> bool:
    """Defer OCR blocks clipped by the top/bottom of the capture.

    Only the vertical edges are checked: scrolling brings a cut-off balloon at
    the top or bottom fully into view later. A balloon touching the left or
    right edge never changes by scrolling, so deferring it meant it was never
    translated at all (balloons drawn against the panel border).
    """
    margin = 8
    display_rect = region.metadata.get("display_bbox")
    rect = display_rect if isinstance(display_rect, Rect) else region.bbox
    return rect.y >= margin and rect.y2 <= height - margin


def _tile_full_frame_for_ocr(
    width: int,
    height: int,
    tile_size: int,
    overlap: int = 96,
) -> list[Rect]:
    """Split a full reader viewport into overlapping native-resolution OCR tiles."""
    if width <= tile_size and height <= tile_size:
        return [Rect(0, 0, width, height)]
    stride = max(1, tile_size - min(overlap, tile_size // 3))
    xs = list(range(0, width, stride)) if width > tile_size else [0]
    ys = list(range(0, height, stride)) if height > tile_size else [0]
    return [
        Rect(x, y, min(tile_size, width - x), min(tile_size, height - y))
        for y in ys
        for x in xs
    ]
