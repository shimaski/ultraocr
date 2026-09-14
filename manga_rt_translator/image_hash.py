from __future__ import annotations

import hashlib

import numpy as np

from manga_rt_translator.models import Rect


def crop_image(image: np.ndarray, rect: Rect) -> np.ndarray:
    height, width = image.shape[:2]
    x = max(0, min(width, rect.x))
    y = max(0, min(height, rect.y))
    x2 = max(x, min(width, rect.x2))
    y2 = max(y, min(height, rect.y2))
    return image[y:y2, x:x2]


def region_hash(image: np.ndarray, rect: Rect, sample: int = 16) -> str:
    crop = crop_image(image, rect)
    if crop.size == 0:
        return ""
    sampled = _sample_grid(crop, sample)
    return hashlib.sha1(sampled.tobytes()).hexdigest()


def perceptual_region_hash(image: np.ndarray, rect: Rect, sample: int = 16) -> str:
    crop = crop_image(image, rect)
    if crop.size == 0:
        return ""
    sampled = _sample_grid(crop, sample).astype(np.float32)
    if sampled.ndim == 3:
        sampled = sampled.mean(axis=2)
    mean = float(sampled.mean())
    bits = (sampled >= mean).astype(np.uint8)
    return hashlib.sha1(bits.tobytes()).hexdigest()


def _sample_grid(image: np.ndarray, sample: int) -> np.ndarray:
    height, width = image.shape[:2]
    y_idx = np.linspace(0, max(0, height - 1), sample).astype(np.int32)
    x_idx = np.linspace(0, max(0, width - 1), sample).astype(np.int32)
    return image[y_idx][:, x_idx]
