from __future__ import annotations

import numpy as np

from manga_rt_translator.models import Rect


def sample_background_color(
    image: np.ndarray,
    rect: Rect,
    padding: int = 8,
) -> tuple[int, int, int] | None:
    """Estimate the bubble fill from a narrow ring outside the OCR text box."""
    if image.ndim < 2 or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1 = max(0, min(width, rect.x))
    y1 = max(0, min(height, rect.y))
    x2 = max(x1, min(width, rect.x2))
    y2 = max(y1, min(height, rect.y2))
    if x1 >= x2 or y1 >= y2:
        return None

    pad = max(1, int(padding))
    outer_x1 = max(0, x1 - pad)
    outer_y1 = max(0, y1 - pad)
    outer_x2 = min(width, x2 + pad)
    outer_y2 = min(height, y2 + pad)
    strips = (
        image[outer_y1:y1, outer_x1:outer_x2],
        image[y2:outer_y2, outer_x1:outer_x2],
        image[y1:y2, outer_x1:x1],
        image[y1:y2, x2:outer_x2],
    )
    channels = 1 if image.ndim == 2 else image.shape[2]
    samples = [strip.reshape(-1, channels) for strip in strips if strip.size]
    if not samples:
        return None

    pixels = np.concatenate(samples, axis=0)
    if pixels.shape[1] == 1:
        pixels = np.repeat(pixels, 3, axis=1)
    elif pixels.shape[1] >= 3:
        pixels = pixels[:, :3]
    else:
        return None
    medians = np.median(pixels, axis=0)
    return tuple(int(np.clip(round(float(channel)), 0, 255)) for channel in medians)


def foreground_color(background: tuple[int, int, int] | None) -> tuple[int, int, int]:
    if background is None:
        return (24, 20, 18)
    red, green, blue = background
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return (255, 255, 255) if luminance < 135 else (24, 20, 18)


def detect_speech_bubble_bounds(
    image: np.ndarray,
    text_rect: Rect,
    background: tuple[int, int, int] | None = None,
    *,
    tolerance: int = 30,
) -> Rect | None:
    """Find a nearby closed, mostly uniform fill around an OCR block.

    This is intentionally conservative: if a connected fill does not surround
    the text cleanly, the caller should keep using the OCR rectangle.
    """
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1, y1 = max(0, text_rect.x), max(0, text_rect.y)
    x2, y2 = min(width, text_rect.x2), min(height, text_rect.y2)
    if x1 >= x2 or y1 >= y2 or text_rect.area < 16:
        return None
    color = background or sample_background_color(
        image, Rect(x1, y1, x2 - x1, y2 - y1), padding=10
    )
    if color is None:
        return None

    radius = max(40, min(180, int(max(x2 - x1, y2 - y1) * 1.6)))
    rx1, ry1 = max(0, x1 - radius), max(0, y1 - radius)
    rx2, ry2 = min(width, x2 + radius), min(height, y2 + radius)
    crop = np.ascontiguousarray(image[ry1:ry2, rx1:rx2, :3])
    delta = np.abs(crop.astype(np.int16) - np.asarray(color, dtype=np.int16))
    color_mask = (np.max(delta, axis=2) <= max(8, min(64, tolerance))).astype(np.uint8)
    if not np.any(color_mask):
        return None
    try:
        import cv2
    except ImportError:
        return None
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        color_mask, connectivity=8
    )
    local_x1, local_y1 = x1 - rx1, y1 - ry1
    local_x2, local_y2 = x2 - rx1, y2 - ry1
    center_x = (local_x1 + local_x2) // 2
    center_y = (local_y1 + local_y2) // 2
    text_area = max(1, (x2 - x1) * (y2 - y1))
    best: tuple[int, int, int, int] | None = None
    best_score = 0
    crop_height, crop_width = color_mask.shape
    for label in range(1, count):
        left, top, comp_width, comp_height, area = (int(value) for value in stats[label])
        right, bottom = left + comp_width, top + comp_height
        if (
            left <= 0 or top <= 0 or right >= crop_width - 1 or bottom >= crop_height - 1
            or not (left <= center_x < right and top <= center_y < bottom)
            or area < text_area * 0.35
            or comp_width < (local_x2 - local_x1) * 1.15
            or comp_height < (local_y2 - local_y1) * 1.15
            or comp_width > max(220, (local_x2 - local_x1) * 6)
            or comp_height > max(220, (local_y2 - local_y1) * 7)
            or comp_width * comp_height > text_area * 36
        ):
            continue
        inside = labels[local_y1:local_y2, local_x1:local_x2] == label
        inside_count = int(np.count_nonzero(inside))
        if inside_count < max(8, int(text_area * 0.025)):
            continue
        score = inside_count * 4 + min(area, text_area * 12)
        if score > best_score:
            best_score = score
            best = (left, top, right, bottom)

    if best is None:
        return None
    left, top, right, bottom = best
    padding = 5
    bx1 = max(0, rx1 + left - padding)
    by1 = max(0, ry1 + top - padding)
    bx2 = min(width, rx1 + right + padding)
    by2 = min(height, ry1 + bottom + padding)
    bounds = Rect(bx1, by1, bx2 - bx1, by2 - by1)
    if not bounds.contains(Rect(x1, y1, x2 - x1, y2 - y1)):
        return None
    return bounds


def inpaint_text_background(
    image: np.ndarray,
    rect: Rect,
    padding: int = 12,
    polygons: tuple[tuple[tuple[float, float], ...], ...] = (),
) -> tuple[Rect, np.ndarray] | None:
    """Erase high-contrast glyph pixels in a small OCR crop, preserving fill texture."""
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1, y1 = max(0, rect.x), max(0, rect.y)
    x2, y2 = min(width, rect.x2), min(height, rect.y2)
    if x1 >= x2 or y1 >= y2 or rect.area <= 0 or rect.area > 300_000:
        return None

    pad = max(3, int(padding))
    patch_rect = Rect(
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(width, x2 + pad) - max(0, x1 - pad),
        min(height, y2 + pad) - max(0, y1 - pad),
    )
    source = np.ascontiguousarray(
        image[patch_rect.y : patch_rect.y2, patch_rect.x : patch_rect.x2, :3]
    )
    local_x1, local_y1 = x1 - patch_rect.x, y1 - patch_rect.y
    local_x2, local_y2 = local_x1 + (x2 - x1), local_y1 + (y2 - y1)
    text_crop = source[local_y1:local_y2, local_x1:local_x2]

    fill = sample_background_color(image, Rect(x1, y1, x2 - x1, y2 - y1), padding=pad)
    if fill is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    delta = np.abs(text_crop.astype(np.int16) - np.asarray(fill, dtype=np.int16))
    contrast_mask = (np.max(delta, axis=2) >= 26).astype(np.uint8) * 255
    if polygons:
        polygon_mask = np.zeros(text_crop.shape[:2], dtype=np.uint8)
        valid_polygon = False
        for polygon in polygons:
            points = np.rint(
                np.asarray(
                    [
                        (point[0] - x1, point[1] - y1)
                        for point in polygon
                    ],
                    dtype=np.float32,
                )
            ).astype(np.int32)
            if len(points) >= 3:
                cv2.fillPoly(polygon_mask, [points], 255)
                valid_polygon = True
        raw_mask = (
            cv2.bitwise_and(contrast_mask, polygon_mask)
            if valid_polygon
            else contrast_mask
        )
    else:
        raw_mask = contrast_mask
    if np.count_nonzero(raw_mask) < 3:
        return None
    if np.count_nonzero(raw_mask) > (x2 - x1) * (y2 - y1) * 0.55:
        return None

    mask = np.zeros(source.shape[:2], dtype=np.uint8)
    mask[local_y1:local_y2, local_x1:local_x2] = raw_mask
    mask = cv2.dilate(mask, np.ones((5, 5), dtype=np.uint8), iterations=1)
    restored = cv2.inpaint(source, mask, 4, cv2.INPAINT_TELEA)
    return patch_rect, np.ascontiguousarray(restored)
