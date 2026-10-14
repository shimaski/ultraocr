from __future__ import annotations

import numpy as np
import pytest

from manga_rt_translator.config import OcrConfig
from manga_rt_translator.hybrid_ocr import HybridOcrEngine
from manga_rt_translator.models import Rect, TextRegion
from manga_rt_translator.ocr import PaddleOcrEngine, group_text_regions, parse_paddle_result
from manga_rt_translator.work_queue import ViewState, WorkCancelled


def line(x, y, text, confidence=.99, width=100):
    return TextRegion(Rect(x, y, width, 20), text, confidence, 'en', f'{x}-{y}')


@pytest.fixture
def hybrid(tmp_path, monkeypatch):
    model = tmp_path / 'detector.onnx'
    model.touch()
    engine = HybridOcrEngine(OcrConfig(detector='hybrid'))
    monkeypatch.setattr(engine.comic, 'MODEL_PATH', model)

    class Fallback:
        def __init__(self):
            self.calls = []

        def recognize_cancellable(self, image, regions, check):
            check()
            self.calls.extend(regions)
            return [TextRegion(regions[0], 'Repaired phrase', .99, 'en', 'fixed')]

    engine._fallback = Fallback()
    return engine


def test_high_confidence_lines_group_without_full_paddle(hybrid, monkeypatch):
    monkeypatch.setattr(hybrid, '_fast_lines', lambda *args: [
        line(100, 100, 'Thank you for'), line(100, 130, 'helping me.'),
    ])
    image = np.full((400, 400, 3), 255, np.uint8)
    result = hybrid.recognize(image, [Rect(0, 0, 400, 400)])
    assert [r.text for r in result] == ['Thank you for helping me.']
    assert hybrid._fallback.calls == []


def test_only_weak_block_gets_full_ocr_retry(hybrid, monkeypatch):
    monkeypatch.setattr(hybrid, '_fast_lines', lambda *args: [
        line(30, 40, 'Good phrase'), line(220, 220, 'Illegible', .65),
    ])
    image = np.full((400, 400, 3), 255, np.uint8)
    result = hybrid.recognize(image, [Rect(0, 0, 400, 400)])
    assert [r.text for r in result] == ['Good phrase', 'Repaired phrase']
    assert len(hybrid._fallback.calls) == 1
    assert hybrid._fallback.calls[0].area < 400*400/4


@pytest.mark.parametrize('failure', [False, True])
def test_empty_or_broken_fast_detector_falls_back(hybrid, monkeypatch, failure):
    def detect(*args):
        if failure:
            raise RuntimeError('GPU unavailable')
        return []
    monkeypatch.setattr(hybrid, '_fast_lines', detect)
    result = hybrid.recognize(np.zeros((400, 400, 3), np.uint8), [Rect(0, 0, 400, 400)])
    assert result[0].text == 'Repaired phrase'
    assert len(hybrid._fallback.calls) == 1


def test_cancellation_is_not_swallowed_by_ocr_fallback(hybrid, monkeypatch):
    view = ViewState()
    token = view.token()
    def detect(image, region, check):
        view.advance()
        check()
    monkeypatch.setattr(hybrid, '_fast_lines', detect)
    with pytest.raises(WorkCancelled):
        hybrid.recognize_cancellable(np.zeros((400, 400, 3), np.uint8),
                                     [Rect(0, 0, 400, 400)], token.check)
    assert hybrid._fallback.calls == []


def test_grouping_respects_balloon_boundary_and_original_geometry():
    image = np.full((200, 300, 3), 255, np.uint8)
    lines = [line(80, 50, 'First balloon'), line(80, 90, 'Second balloon')]
    image[80:82, 50:240] = 0
    result = group_text_regions(lines, image)
    assert len(result) == 2
    assert lines[0].bbox == Rect(80, 50, 100, 20)
    assert result[0].visual_hash != lines[0].visual_hash


def test_same_line_y_jitter_keeps_word_order():
    lines = [line(160, 48, 'BRAT', width=50), line(50, 50, 'IMPUDENT')]
    result = group_text_regions(lines)
    assert [r.text for r in result] == ['IMPUDENT BRAT']
    assert lines[0].text == 'BRAT'


def test_numpy_paddle_polygons_do_not_use_ambiguous_truth_value():
    raw = [{'rec_texts': ['Hello'], 'rec_scores': np.array([.98]),
            'rec_polys': np.array([[[10, 10], [110, 10], [110, 30], [10, 30]]])}]
    result = parse_paddle_result(raw, Rect(0, 0, 200, 200), 'en',
                                 np.zeros((200, 200, 3), np.uint8))
    assert result[0].text == 'Hello'
    assert result[0].polygons == (((10.0, 10.0), (110.0, 10.0), (110.0, 30.0), (10.0, 30.0)),)


def test_paddle_cancellation_checked_before_upscale_retry(monkeypatch):
    engine = object.__new__(PaddleOcrEngine)
    view = ViewState()
    token = view.token()
    def recognize(*args):
        view.advance()
        return []
    monkeypatch.setattr(engine, '_recognize_best_for_crop', recognize)
    image = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(WorkCancelled):
        engine._recognize_with_retry(image, Rect(0, 0, 100, 100), image, token.check)


def test_paddle_upscale_retry_maps_back_correctly(monkeypatch):
    engine = object.__new__(PaddleOcrEngine)
    calls = []
    def recognize(crop, rect, full, check):
        calls.append(crop.shape)
        check()
        if len(calls) == 1:
            return [line(10, 10, 'weak', .6)]
        return [line(20, 20, 'strong', .99)]
    monkeypatch.setattr(engine, '_recognize_best_for_crop', recognize)
    image = np.zeros((100, 100, 3), np.uint8)
    result = engine._recognize_with_retry(image, Rect(0, 0, 100, 100), image)
    assert result[0].text == 'strong'
    assert result[0].bbox.x == 10
    assert calls == [(100, 100, 3), (200, 200, 3)]
