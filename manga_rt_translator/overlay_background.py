from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from manga_rt_translator.models import Rect


@dataclass(frozen=True)
class SpeechBubbleRegion:
    bounds: Rect


def detect_enclosed_speech_bubbles(image: np.ndarray) -> list[SpeechBubbleRegion]:
    """Find bright balloon interiors enclosed by dark ink in a comic frame.

    Dark outlines are dilated just enough to close small gaps. This keeps a
    white balloon separate from page gutters; broad uniform backgrounds that
    touch the frame edge are rejected. This detection is used for text layout;
    clearing original lettering happens separately with OCR glyph masks.
    """
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return []
    try:
        import cv2
    except ImportError:
        return []

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    dark_ink = (gray < 110).astype(np.uint8) * 255
    barrier = cv2.dilate(dark_ink, np.ones((3, 3), np.uint8), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        255 - barrier, connectivity=8
    )
    frame_area = max(1, width * height)
    found: list[SpeechBubbleRegion] = []
    # Cheap geometric rejection for every component at once. Screentone and
    # lettering produce thousands of tiny components; walking them one by one
    # in Python dominated this function. Same rules as the per-label check.
    sx, sy, sw, sh, sa = (stats[:, i].astype(np.int64) for i in range(5))
    box_areas = sw * sh
    keep = (
        (sx > 1) & (sy > 1) & (sx + sw < width - 1) & (sy + sh < height - 1)
        & (sw >= 44) & (sh >= 32) & (sa >= 1100)
        & (box_areas <= frame_area * 0.20)
        & (sa / np.maximum(1, box_areas) >= 0.28)
    )
    keep[0] = False
    for label in np.flatnonzero(keep).tolist():
        x, y, box_width, box_height, area = (int(v) for v in stats[label])
        box_area = box_width * box_height
        if (
            x <= 1 or y <= 1 or x + box_width >= width - 1
            or y + box_height >= height - 1
            or box_width < 44 or box_height < 32
            or area < 1100 or box_area > frame_area * 0.20
            or area / max(1, box_area) < 0.28
        ):
            continue
        crop_labels = labels[y : y + box_height, x : x + box_width]
        component = (crop_labels == label).astype(np.uint8) * 255
        crop_gray = gray[y : y + box_height, x : x + box_width]
        if float((crop_gray[component > 0] >= 205).mean()) < 0.82:
            continue
        contours, _ = cv2.findContours(
            component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue
        found.append(SpeechBubbleRegion(bounds=Rect(x, y, box_width, box_height)))
    return found


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
    fill_luminance = 0.2126 * fill[0] + 0.7152 * fill[1] + 0.0722 * fill[2]
    gray = cv2.cvtColor(text_crop, cv2.COLOR_RGB2GRAY)
    if fill_luminance < 90 and polygons:
        # Light lettering over dark ink: select only the glyph pixels, not
        # surrounding dark illustration that happens to differ from the fill.
        contrast_mask = (gray >= 180).astype(np.uint8) * 255
    elif fill_luminance >= 185 and polygons:
        # Dark print in a white balloon. A simple color-distance mask also
        # picks up grey screentone and nearby line art, producing cutouts.
        contrast_mask = (gray <= 150).astype(np.uint8) * 255
    else:
        contrast_mask = (np.max(delta, axis=2) >= 48).astype(np.uint8) * 255
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
    kernel_size = 3 if polygons else 5
    mask = cv2.dilate(mask, np.ones((kernel_size, kernel_size), dtype=np.uint8), iterations=1)
    restored = cv2.inpaint(source, mask, 3, cv2.INPAINT_TELEA)
    # Paint only erased lettering. A full RGB crop carries stale manga pixels
    # and exposes a straight edge when the page moves or lazy-loads.
    alpha = cv2.GaussianBlur(mask, (3, 3), 0.6)
    patch = np.ascontiguousarray(np.dstack((restored, alpha)))
    return patch_rect, patch


def _balloon_interior_mask(gray: np.ndarray) -> np.ndarray | None:
    """Filled mask of the light region that fills the middle of the crop.

    The lettering forms holes in the light component; filling its outer
    contour gives the whole balloon interior, letters included.
    """
    import cv2

    h, w = gray.shape[:2]
    light = (gray > 150).astype(np.uint8)
    count, labels = cv2.connectedComponents(light, connectivity=4)
    if count <= 1:
        return None
    center = labels[h // 4 : h - h // 4, w // 4 : w - w // 4]
    values = center[center > 0]
    if values.size == 0:
        return None
    ids, counts = np.unique(values, return_counts=True)
    component = (labels == int(ids[counts.argmax()])).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled = np.zeros_like(component)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    # A balloon interior covers a good part of its bounding box; anything
    # smaller means the light region is not the balloon, so do not restrict.
    if np.count_nonzero(filled) < 0.35 * h * w:
        return None
    return filled


def inpaint_bubble_glyphs(
    image: np.ndarray,
    bounds: Rect,
) -> tuple[Rect, np.ndarray] | None:
    """Erase dark lettering throughout a detected white balloon interior.

    OCR polygons can cover only the first line of a stacked phrase. The
    detected bright-component bounds are a safer limit for a second, narrow
    dark-glyph mask: this removes missed source letters without painting a
    rectangular fill over the artwork around the balloon.
    """
    if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
        return None
    height, width = image.shape[:2]
    x1, y1 = max(0, bounds.x), max(0, bounds.y)
    x2, y2 = min(width, bounds.x2), min(height, bounds.y2)
    if x2 - x1 < 24 or y2 - y1 < 20 or (x2 - x1) * (y2 - y1) > 90_000:
        return None
    try:
        import cv2
    except ImportError:
        return None

    source = np.ascontiguousarray(image[y1:y2, x1:x2, :3])
    gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    raw = (gray <= 150).astype(np.uint8) * 255
    # Ignore any outline pixels touching the connected bright region's edge.
    if raw.shape[0] > 4 and raw.shape[1] > 4:
        raw[:2, :] = 0
        raw[-2:, :] = 0
        raw[:, :2] = 0
        raw[:, -2:] = 0
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    glyphs = np.zeros(raw.shape, dtype=np.uint8)
    max_component_area = max(16, int(raw.size * 0.08))
    for label in range(1, count):
        cx, cy, cw, ch, area = (int(v) for v in stats[label])
        if (
            area < 2 or area > max_component_area
            or cw > raw.shape[1] * 0.75 or ch > raw.shape[0] * 0.52
        ):
            continue
        glyphs[labels == label] = 255
    # Keep only glyphs inside the balloon itself. The bounds are a rectangle;
    # its corners hold artwork (hair, screentone) outside the oval outline,
    # and erasing those left smears around the balloon.
    interior = _balloon_interior_mask(gray)
    if interior is not None:
        glyphs[interior == 0] = 0
    if np.count_nonzero(glyphs) < 3:
        return None
    if np.count_nonzero(glyphs) > raw.size * 0.24:
        return None
    mask = cv2.dilate(glyphs, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if interior is not None:
        mask[interior == 0] = 0
    restored = cv2.inpaint(source, mask, 3, cv2.INPAINT_TELEA)
    alpha = cv2.GaussianBlur(mask, (3, 3), 0.6)
    patch_rect = Rect(x1, y1, x2 - x1, y2 - y1)
    patch = np.ascontiguousarray(np.dstack((restored, alpha)))
    return patch_rect, patch
