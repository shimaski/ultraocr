from __future__ import annotations

import numpy as np

from manga_rt_translator.config import load_config
from manga_rt_translator.models import Rect, TextRegion
from manga_rt_translator.ocr import (
    _dedup_overlapping_lines,
    _iou,
    _mean_confidence,
    _remap_scaled_regions,
    _should_retry_upscale,
)


def _line(x, y, w, h, text, conf=0.9):
    return TextRegion(
        bbox=Rect(x=x, y=y, width=w, height=h),
        text=text,
        confidence=conf,
        source_lang="en",
        visual_hash="hash",
    )


def test_iou_same_box_is_one():
    box = Rect(x=10, y=10, width=100, height=30)
    assert _iou(box, box) == 1.0
    assert _iou(box, Rect(x=200, y=200, width=50, height=20)) == 0.0


def test_duplicate_detections_are_deduped_keeping_best_confidence():
    lines = [
        _line(100, 100, 200, 30, "Thank you for", conf=0.72),
        _line(102, 101, 198, 30, "Thank you for", conf=0.94),
        _line(100, 200, 200, 30, "confiding in me", conf=0.91),
    ]
    kept = _dedup_overlapping_lines(lines)
    assert len(kept) == 2
    assert all(item.confidence >= 0.9 for item in kept)


def test_side_by_side_lines_are_not_deduped():
    lines = [
        _line(100, 100, 100, 30, "left"),
        _line(220, 100, 100, 30, "right"),
    ]
    assert len(_dedup_overlapping_lines(lines)) == 2


def test_retry_triggers_only_on_weak_results():
    assert _should_retry_upscale([])
    assert _should_retry_upscale([_line(0, 0, 10, 10, "stylized", conf=0.70)])
    assert not _should_retry_upscale([_line(0, 0, 10, 10, "crisp", conf=0.97)])


def test_mean_confidence_is_length_weighted():
    lines = [_line(0, 0, 10, 10, "short", conf=0.5), _line(0, 0, 10, 10, "much longer", conf=0.9)]
    assert _mean_confidence(lines) > 0.7


def test_remap_scaled_regions_maps_back_to_original_coordinates():
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    offset = Rect(x=500, y=200, width=300, height=100)
    scaled = [_line(40, 60, 200, 50, "hello", conf=0.95)]
    remapped = _remap_scaled_regions(scaled, scale=2.0, offset=offset, image=image)
    assert len(remapped) == 1
    assert remapped[0].bbox.as_tuple() == (520, 230, 100, 25)
    assert remapped[0].visual_hash != "hash"


def test_config_reads_rec_model():
    config = load_config(None)
    assert config.ocr.rec_model == "mobile"
    assert config.ocr.det_model == "mobile"
    assert config.ocr.upscale_retry_max_pixels == 250_000


def test_large_viewport_does_not_trigger_expensive_upscale_retry(monkeypatch):
    from manga_rt_translator.ocr import PaddleOcrEngine

    engine = object.__new__(PaddleOcrEngine)
    engine.config = load_config(None).ocr
    calls = []

    def recognize(crop, rect, full, check):
        calls.append(crop.shape)
        return [_line(10, 10, 100, 20, "weak", conf=0.60)]

    monkeypatch.setattr(engine, "_recognize_best_for_crop", recognize)
    image = np.zeros((700, 700, 3), dtype=np.uint8)
    result = engine._recognize_with_retry(image, Rect(0, 0, 700, 700), image)

    assert result[0].text == "weak"
    assert calls == [(700, 700, 3)]
