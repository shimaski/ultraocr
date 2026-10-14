from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return max(0, self.width) * max(0, self.height)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)

    def crop_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x2, self.y2)

    def expanded(self, padding: int, max_width: int, max_height: int) -> "Rect":
        x = max(0, self.x - padding)
        y = max(0, self.y - padding)
        x2 = min(max_width, self.x2 + padding)
        y2 = min(max_height, self.y2 + padding)
        return Rect(x=x, y=y, width=max(0, x2 - x), height=max(0, y2 - y))

    def intersects(self, other: "Rect") -> bool:
        return (
            self.x < other.x2
            and self.x2 > other.x
            and self.y < other.y2
            and self.y2 > other.y
        )

    def union(self, other: "Rect") -> "Rect":
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        x2 = max(self.x2, other.x2)
        y2 = max(self.y2, other.y2)
        return Rect(x=x, y=y, width=x2 - x, height=y2 - y)

    def contains(self, other: "Rect") -> bool:
        return (
            self.x <= other.x
            and self.y <= other.y
            and self.x2 >= other.x2
            and self.y2 >= other.y2
        )


@dataclass
class Frame:
    image: "np.ndarray"
    timestamp: float
    index: int
    dirty_rects: tuple[Rect, ...] = field(default_factory=tuple)
    dirty_rects_stable: bool = False
    capture_viewport: CaptureViewport | None = None
    capture_lost: bool = False


@dataclass(frozen=True)
class CaptureViewport:
    """Physical-pixel placement of a captured crop on a display."""

    capture_origin: tuple[int, int]
    capture_size: tuple[int, int]
    monitor_origin: tuple[int, int]
    monitor_device_name: str = ""
    monitor_index: int = 0


@dataclass
class TextRegion:
    bbox: Rect
    text: str = ""
    confidence: float = 0.0
    source_lang: str | None = None
    visual_hash: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    polygons: tuple[tuple[tuple[float, float], ...], ...] = ()


@dataclass
class OverlayItem:
    bbox: Rect
    text: str
    source_text: str = ""
    confidence: float = 0.0
    expires_at: float | None = None
    # Stable identity of the source pixels. Keeping it on the presentation
    # item lets later sweeps update the same balloon even when its OCR box
    # moves a few pixels or the translated text changes the painted size.
    visual_hash: str = ""
    background_color: tuple[int, int, int] | None = None
    background_patch: Any | None = None
    background_rect: Rect | None = None
    display_bbox: Rect | None = None
