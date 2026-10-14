from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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

        diff = np.abs(current.astype(np.int16) - previous.astype(np.int16))
        if diff.ndim == 3:
            diff = diff.mean(axis=2)
        mask = diff > self.config.pixel_threshold

        tile = self.config.tile_size
        height, width = mask.shape[:2]
        changed: list[Rect] = []
        total_tiles = 0

        for y in range(0, height, tile):
            for x in range(0, width, tile):
                total_tiles += 1
                tile_mask = mask[y : min(y + tile, height), x : min(x + tile, width)]
                if tile_mask.size == 0:
                    continue
                ratio = float(tile_mask.mean())
                if ratio >= self.config.min_changed_ratio:
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
                ready.append(pending.rect)
            else:
                keep.append(pending)

        self._pending = keep
        return merge_rects(ready, gap=self.config.roi_padding)


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
