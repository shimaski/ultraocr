from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Iterable

from manga_rt_translator.cache import TranslationMemory, normalize_text
from manga_rt_translator.capture.base import ScreenStream
from manga_rt_translator.config import AppConfig
from manga_rt_translator.diff import FrameDiffer
from manga_rt_translator.glossary import Glossary
from manga_rt_translator.image_hash import crop_image
from manga_rt_translator.models import Frame, OverlayItem, Rect, TextRegion
from manga_rt_translator.overlay_background import (
    detect_speech_bubble_bounds,
    inpaint_text_background,
    sample_background_color,
)
from manga_rt_translator.ocr import OcrEngine
from manga_rt_translator.overlay import Overlay
from manga_rt_translator.text_regions import TextRegionDetector
from manga_rt_translator.translator import EchoTranslator, TranslationResult, Translator
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

        # A scroll-like change moves every viewport-anchored bubble away from
        # its content. Instead of wiping them (which blanks the screen during
        # the next analysis), shorten their life so they fade out quickly and
        # get replaced by the fresh translation when it lands.
        if not force and getattr(self.differ, "last_tick_had_large_change", False):
            self._fade_active_items()
            self._publish([], token)
            return list(self.active_items)

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
            self._fade_active_items()
            self._publish([], token)
            return list(self.active_items)

        self.stats.frames_with_rois += 1
        token.commit(lambda: self._remove_items_overlapping(rois))

        candidates = rois if getattr(self.ocr, "detects_text", False) else self.detector.detect(frame.image, rois)
        if force and not candidates:
            candidates = rois
        if not candidates:
            self._publish([], token)
            return list(self.active_items)

        regions = self._ocr_cached(frame.image, candidates, token)
        if force and not regions and candidates != rois:
            regions = self._ocr_cached(frame.image, rois, token)
        token.check()
        if not force:
            regions = _filter_text_regions(regions, frame.image.shape[1], frame.image.shape[0])
        for region in regions:
            region.metadata["background_color"] = sample_background_color(
                frame.image, region.bbox
            )
            bubble_bounds = detect_speech_bubble_bounds(
                frame.image,
                region.bbox,
                region.metadata["background_color"],
            )
            if bubble_bounds is not None:
                region.metadata["display_bbox"] = bubble_bounds
            inpainted = inpaint_text_background(
                frame.image,
                region.bbox,
                polygons=region.polygons,
            )
            if inpainted is not None:
                region.metadata["background_rect"], region.metadata["background_patch"] = inpainted
        self.stats.ocr_regions += len(regions)

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
            self.overlay.update(list(self.active_items))
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

    def _fade_active_items(self, seconds: float = 1.2) -> None:
        now = time.time()
        fade_until = now + seconds
        with self._items_lock:
            for item in self.active_items:
                if item.expires_at is None or item.expires_at > fade_until:
                    item.expires_at = fade_until

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
        for rect in candidates:
            if token:
                token.check()
            key = self._roi_key(image, rect)
            cached = self.roi_ocr_cache.get(key)
            if cached is None:
                misses.append(rect)
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
            keys_by_rect = [self._roi_key(image, rect) for rect in misses]
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

        for index, region in enumerate(regions):
            source_lang = region.source_lang or default_source
            pair = self._language_pair(source_lang, target_lang)
            visual_hit = self.memory.get_visual(region.visual_hash, pair)
            if visual_hit:
                self.stats.cache_hits += 1
                results[index] = visual_hit.translated_text
                continue
            text_hit = self.memory.lookup(region.text, pair)
            if text_hit:
                self.stats.cache_hits += 1
                results[index] = text_hit.translated_text
            else:
                misses.append(index)

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
                pair = self._language_pair(source_lang, target_lang)
                # Echo is a display fallback, not a translation. Persisting it
                # poisons the translation memory with source-language text and
                # prevents a later provider retry from becoming authoritative.
                cacheable = result.provider != "echo"
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
            # between Portuguese and provisional OCR text. Scroll-like changes
            # are handled separately by _fade_active_items().
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


def _filter_text_regions(
    regions: Iterable[TextRegion],
    frame_width: int,
    frame_height: int,
    max_regions: int = 6,
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
    return filtered[:max_regions]
