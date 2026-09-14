from __future__ import annotations

import numpy as np
import cv2

from manga_rt_translator.image_hash import crop_image
from manga_rt_translator.models import Rect


class TextRegionDetector:
    """Lightweight ROI filter before OCR.

    This is intentionally conservative. It keeps changed regions that have
    enough contrast and edge density to plausibly contain text. PaddleOCR still
    performs the real text detection inside those candidates.
    """

    def __init__(
        self,
        min_contrast: float = 18.0,
        min_edge_ratio: float = 0.006,
        ignore_top_pixels: int = 0,
        max_region_area_ratio: float = 0.35,
        scan_tile_size: int = 320,
    ) -> None:
        self.min_contrast = min_contrast
        self.min_edge_ratio = min_edge_ratio
        self.ignore_top_pixels = ignore_top_pixels
        self.max_region_area_ratio = max_region_area_ratio
        self.scan_tile_size = scan_tile_size

    def detect(self, image: np.ndarray, rois: list[Rect]) -> list[Rect]:
        candidates: list[Rect] = []
        frame_area = max(1, image.shape[0] * image.shape[1])
        for roi in rois:
            if roi.y2 <= self.ignore_top_pixels:
                continue
            if roi.y < self.ignore_top_pixels:
                clipped_y = self.ignore_top_pixels
                roi = Rect(
                    x=roi.x,
                    y=clipped_y,
                    width=roi.width,
                    height=max(1, roi.y2 - clipped_y),
                )
            bubble_candidates = self._detect_bubble_regions(image, roi)
            if self._trust_bubble_candidates(roi, bubble_candidates):
                candidates.extend(bubble_candidates)
                continue
            if (roi.area / frame_area) > self.max_region_area_ratio:
                candidates.extend(self._scan_large_roi(image, roi))
                continue
            crop = crop_image(image, roi)
            if crop.size == 0:
                continue
            if self._looks_textual(crop):
                candidates.append(roi)
        return _merge_text_candidate_rects(candidates)

    def _trust_bubble_candidates(self, roi: Rect, candidates: list[Rect]) -> bool:
        if not candidates:
            return False
        covered_area = sum(candidate.area for candidate in candidates)
        coverage_ratio = covered_area / max(1, roi.area)
        if len(candidates) >= 2:
            return True
        return coverage_ratio >= 0.12

    def _detect_bubble_regions(self, image: np.ndarray, roi: Rect) -> list[Rect]:
        crop = crop_image(image, roi)
        if crop.size == 0 or crop.shape[0] < 40 or crop.shape[1] < 40:
            return []
        gray = crop.mean(axis=2).astype(np.uint8) if crop.ndim == 3 else crop.astype(np.uint8)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 205, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        bubble_rects: list[Rect] = []
        crop_area = max(1, crop.shape[0] * crop.shape[1])
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if w < 44 or h < 24:
                continue
            if area < 1800 or area > int(crop_area * 0.45):
                continue
            rect_crop = Rect(x=x, y=y, width=w, height=h)
            rect_img = Rect(x=roi.x + x, y=roi.y + y, width=w, height=h)
            bubble = crop_image(crop, rect_crop)
            if bubble.size == 0:
                continue
            bubble_gray = bubble.mean(axis=2) if bubble.ndim == 3 else bubble
            bright_ratio = float((bubble_gray >= 200).mean())
            dark_ratio = float((bubble_gray <= 80).mean())
            if bright_ratio < 0.58:
                continue
            if dark_ratio > 0.28:
                continue
            bubble_rects.append(
                rect_img.expanded(10, image.shape[1], image.shape[0])
            )
        return _merge_text_candidate_rects(bubble_rects, gap=16)

    def _scan_large_roi(self, image: np.ndarray, roi: Rect) -> list[Rect]:
        candidates: list[Rect] = []
        tile = self.scan_tile_size
        x_end = roi.x + roi.width
        y_end = roi.y + roi.height
        for y in range(roi.y, y_end, tile):
            for x in range(roi.x, x_end, tile):
                width = min(tile, x_end - x)
                height = min(tile, y_end - y)
                if width < 32 or height < 32:
                    continue
                part = Rect(x=x, y=y, width=width, height=height)
                crop = crop_image(image, part)
                if crop.size == 0:
                    continue
                if self._looks_textual(crop):
                    candidates.append(part)
        return candidates

    def _looks_textual(self, crop: np.ndarray) -> bool:
        gray = crop.mean(axis=2) if crop.ndim == 3 else crop
        contrast = float(gray.std())
        if contrast < self.min_contrast:
            return False

        gy = np.abs(np.diff(gray.astype(np.float32), axis=0))
        gx = np.abs(np.diff(gray.astype(np.float32), axis=1))
        edge_ratio = (float((gx > 18).mean()) + float((gy > 18).mean())) / 2.0
        return edge_ratio >= self.min_edge_ratio


def _merge_text_candidate_rects(rects: list[Rect], gap: int = 12) -> list[Rect]:
    if not rects:
        return []
    ordered = sorted(rects, key=lambda item: (item.y, item.x))
    merged: list[Rect] = []
    for rect in ordered:
        if not merged:
            merged.append(rect)
            continue
        previous = merged[-1]
        horizontally_close = rect.x - previous.x2 <= gap
        vertically_close = abs(rect.y - previous.y) <= max(previous.height, rect.height, gap)
        if previous.intersects(rect) or (horizontally_close and vertically_close):
            merged[-1] = previous.union(rect)
        else:
            merged.append(rect)
    return merged
