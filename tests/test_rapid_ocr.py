from __future__ import annotations

import numpy as np
import pytest

from manga_rt_translator.config import OcrConfig
from manga_rt_translator.models import Rect
from manga_rt_translator.rapid_ocr import RapidOcrEngine
from manga_rt_translator.work_queue import ViewState, WorkCancelled


def box(x, y, w, h):
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class FakeRapid:
    """Stand-in for a RapidOCR instance: returns (rows, elapse) like the real
    engine, where each row is [polygon, text, score_string]."""

    def __init__(self, rows, on_call=None):
        self.rows = rows
        self.calls = []
        self.on_call = on_call

    def __call__(self, image):
        self.calls.append(image.shape)
        if self.on_call is not None:
            self.on_call()
        return self.rows, [0.0, 0.0, 0.0]


def make(rows, **cfg):
    engine = RapidOcrEngine(OcrConfig(detector="rapid", source_lang="en", **cfg))
    fake = FakeRapid(rows)
    engine._get_engine = lambda lang=None: fake  # type: ignore[assignment]
    return engine, fake


def test_rows_parse_and_group_into_one_phrase():
    rows = [
        [box(40, 50, 90, 24), "Thank", "0.97"],
        [box(140, 50, 70, 24), "you", "0.96"],
    ]
    engine, _ = make(rows)
    image = np.full((300, 400, 3), 255, np.uint8)
    result = engine.recognize(image, [Rect(0, 0, 400, 300)])
    assert [r.text for r in result] == ["Thank you"]


def test_low_confidence_rows_are_dropped():
    rows = [
        [box(40, 50, 90, 24), "Clear", "0.95"],
        [box(40, 120, 90, 24), "noise", "0.20"],
    ]
    engine, _ = make(rows, min_confidence=0.55)
    image = np.full((300, 400, 3), 255, np.uint8)
    result = engine.recognize(image, [Rect(0, 0, 400, 300)])
    assert [r.text for r in result] == ["Clear"]


def test_region_offset_is_applied_to_boxes():
    rows = [[box(10, 10, 50, 20), "Word", "0.99"]]
    engine, _ = make(rows)
    image = np.full((400, 400, 3), 255, np.uint8)
    # Crop starts at (100, 80); the box is local to the crop and must shift.
    result = engine.recognize(image, [Rect(100, 80, 200, 200)])
    assert result and result[0].bbox.x == 110 and result[0].bbox.y == 90


def test_cancellation_is_not_swallowed():
    rows = [[box(10, 10, 50, 20), "Word", "0.99"]]
    engine, fake = make(rows)
    view = ViewState()
    token = view.token()
    # Invalidate the view mid-inference; the check() after inference must raise.
    fake.on_call = view.advance
    with pytest.raises(WorkCancelled):
        engine.recognize_cancellable(
            np.zeros((200, 200, 3), np.uint8), [Rect(0, 0, 200, 200)], token.check
        )
