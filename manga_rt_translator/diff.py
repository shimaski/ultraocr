from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # OpenCV is a project dependency; keep a numpy fallback regardless.
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from manga_rt_translator.config import DiffConfig
from manga_rt_translator.image_hash import region_hash
from manga_rt_translator.models import Frame, Rect


@dataclass
class PendingRegion:
    rect: Rect
    first_seen_at: float
    last_changed_at: float
    visual_hash: str


class FrameDiffer:
    """Tile-based frame differencing with stable-region debounce."""

    def __init__(self, config: DiffConfig) -> None:
        self.config = config
        self._previous: np.ndarray | None = None
        self._pending: list[PendingRegion] = []
        self._cooldown_until = 0.0
        # (rect, timestamp) of recent OCR fires, used to spot a location that
        # keeps re-firing (animation) and throttle it. Pruned to the window.
        self._fire_log: list[tuple[Rect, float]] = []
        # True on ticks where the change was scroll-like (most of the frame
        # moved). Consumers use it to drop viewport-anchored state such as
        # overlay bubbles that no longer sit on their content.
        self.last_tick_had_large_change = False

    @property
    def has_baseline(self) -> bool:
        return self._previous is not None

    def update(self, frame: Frame) -> list[Rect]:
        height, width = frame.image.shape[:2]
        self.last_tick_had_large_change = False
        if frame.dirty_rects_stable and frame.dirty_rects:
            self._pending.clear()
            self._cooldown_until = 0.0
            self._previous = frame.image.copy()
            rois = [
                rect.expanded(self.config.roi_padding, width, height)
                for rect in frame.dirty_rects
                if rect.area > 0
            ]
            return merge_rects(rois, gap=max(4, self.config.roi_padding))[
                : self.config.max_rois_per_tick
            ]
        if frame.timestamp < self._cooldown_until:
            self._previous = frame.image.copy()
            return []

        if self._previous is None:
            raw_changes = [Rect(0, 0, width, height)]
        elif frame.dirty_rects:
            raw_changes = list(frame.dirty_rects)
        else:
            raw_changes = self._diff_tiles(self._previous, frame.image, frame.timestamp)

        rois = [
            rect.expanded(self.config.roi_padding, width, height)
            for rect in raw_changes
            if rect.area > 0
        ]
        rois = merge_rects(rois, gap=max(4, self.config.roi_padding))

        for rect in rois:
            self._upsert_pending(frame, rect)

        ready = self._collect_ready(frame)
        self._previous = frame.image.copy()
        return ready[: self.config.max_rois_per_tick]

    def prime(self, frame: Frame) -> None:
        """Set a known full-frame baseline after an explicit viewport read."""
        self._previous = frame.image.copy()
        self._pending.clear()
        self._fire_log.clear()
        self._cooldown_until = 0.0
        self.last_tick_had_large_change = False

    def _diff_tiles(
        self,
        previous: np.ndarray,
        current: np.ndarray,
        timestamp: float,
    ) -> list[Rect]:
        if previous.shape != current.shape:
            height, width = current.shape[:2]
            return [Rect(0, 0, width, height)]

        mask = _changed_pixel_mask(previous, current, self.config.pixel_threshold)

        tile = self.config.tile_size
        height, width = mask.shape[:2]
        changed: list[Rect] = []

        # Vectorised per-tile changed-pixel ratio (was a Python loop over every
        # tile on each frame). Pad to a tile multiple, sum per tile, divide by
        # each tile's real area so edge tiles keep the exact same ratio.
        tiles_y = -(-height // tile)
        tiles_x = -(-width // tile)
        total_tiles = tiles_y * tiles_x
        padded = np.zeros((tiles_y * tile, tiles_x * tile), dtype=np.uint32)
        padded[:height, :width] = mask
        counts = padded.reshape(tiles_y, tile, tiles_x, tile).sum(axis=(1, 3))
        tile_h = np.minimum(tile, height - np.arange(tiles_y) * tile)
        tile_w = np.minimum(tile, width - np.arange(tiles_x) * tile)
        areas = tile_h[:, None] * tile_w[None, :]
        ratios = counts / areas
        for ty, tx in zip(*np.nonzero(ratios >= self.config.min_changed_ratio)):
            x = int(tx) * tile
            y = int(ty) * tile
            changed.append(
                Rect(
                    x=x,
                    y=y,
                    width=min(tile, width - x),
                    height=min(tile, height - y),
                )
            )

        changed_ratio = (len(changed) / total_tiles) if total_tiles else 0.0
        if changed_ratio >= self.config.large_change_tile_ratio:
            self._pending.clear()
            # A scroll/large change means fresh content; forget the old
            # per-location fire history so new text is never wrongly throttled.
            self._fire_log.clear()
            self._cooldown_until = timestamp + (self.config.scroll_cooldown_ms / 1000.0)
            self.last_tick_had_large_change = True
            return []

        return changed

    def _upsert_pending(self, frame: Frame, rect: Rect) -> None:
        visual_hash = region_hash(frame.image, rect)
        now = frame.timestamp

        for pending in self._pending:
            if pending.rect.intersects(rect) or pending.rect.contains(rect) or rect.contains(pending.rect):
                union = pending.rect.union(rect)
                if union != pending.rect or visual_hash != pending.visual_hash:
                    pending.rect = union
                    pending.last_changed_at = now
                    pending.visual_hash = region_hash(frame.image, union)
                return

        self._pending.append(
            PendingRegion(
                rect=rect,
                first_seen_at=now,
                last_changed_at=now,
                visual_hash=visual_hash,
            )
        )

    def _collect_ready(self, frame: Frame) -> list[Rect]:
        stable_seconds = self.config.stable_ms / 1000.0
        ready: list[Rect] = []
        keep: list[PendingRegion] = []

        for pending in self._pending:
            current_hash = region_hash(frame.image, pending.rect)
            if current_hash != pending.visual_hash:
                pending.visual_hash = current_hash
                pending.last_changed_at = frame.timestamp
                keep.append(pending)
                continue

            if frame.timestamp - pending.last_changed_at >= stable_seconds:
                if self._is_animation_spam(pending.rect, frame.timestamp):
                    # A spot re-firing this fast is animation (ad/carousel),
                    # not new reading content. Drop it this cycle instead of
                    # re-OCRing; it re-checks once the older fires age out.
                    continue
                ready.append(pending.rect)
                self._fire_log.append((pending.rect, frame.timestamp))
            else:
                keep.append(pending)

        self._pending = keep
        return merge_rects(ready, gap=self.config.roi_padding)

    def _is_animation_spam(self, rect: Rect, now: float) -> bool:
        if not getattr(self.config, "animation_suppression", True):
            return False
        window = self.config.animation_window_ms / 1000.0
        self._fire_log = [
            (fired_rect, fired_at)
            for fired_rect, fired_at in self._fire_log
            if now - fired_at <= window
        ]
        count = sum(
            1 for fired_rect, _ in self._fire_log if fired_rect.intersects(rect)
        )
        return count >= int(self.config.animation_max_fires)


_ONES_1X3 = np.ones((1, 3), np.float32)


def _changed_pixel_mask(previous: np.ndarray, current: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of pixels whose mean per-channel abs diff exceeds threshold.

    The fast path uses OpenCV (~4x quicker at 1080p). mean(|d|) > T is the same
    as sum(|d|) > 3T for 3 channels; the uint8 sum saturates at 255, so the fast
    path is only used while 3T < 255, where saturation cannot flip the result.
    """
    if (
        cv2 is not None
        and current.ndim == 3
        and current.shape[2] == 3
        and current.dtype == np.uint8
        and previous.dtype == np.uint8
        and 3 * threshold < 255
    ):
        summed = cv2.transform(cv2.absdiff(current, previous), _ONES_1X3)
        if summed.ndim == 3:
            summed = summed[..., 0]
        return summed > 3 * threshold
    diff = np.abs(current.astype(np.int16) - previous.astype(np.int16))
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    return diff > threshold


def merge_rects(rects: list[Rect], gap: int = 0) -> list[Rect]:
    if not rects:
        return []

    merged = list(rects)
    changed = True
    while changed:
        changed = False
        output: list[Rect] = []
        while merged:
            current = merged.pop()
            expanded = Rect(
                current.x - gap,
                current.y - gap,
                current.width + gap * 2,
                current.height + gap * 2,
            )
            matched = False
            for index, other in enumerate(output):
                if expanded.intersects(other):
                    output[index] = other.union(current)
                    matched = True
                    changed = True
                    break
            if not matched:
                output.append(current)
        merged = output

    return sorted(merged, key=lambda rect: rect.area, reverse=True)
