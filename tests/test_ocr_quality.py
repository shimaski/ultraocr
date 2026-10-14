from __future__ import annotations

import numpy as np

from manga_rt_translator.config import OcrConfig, load_config
from manga_rt_translator.models import Rect, TextRegion
from manga_rt_translator.ocr import (
    _dedup_overlapping_lines,
    _iou,
    _mean_confidence,
    _remap_scaled_regions,
    _should_retry_upscale,
    _ENGINE_CACHE,
    _ENGINE_LOCK,
    PaddleOcrEngine,
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
    scaled = [TextRegion(
        bbox=Rect(40, 60, 200, 50),
        text="hello",
        confidence=0.95,
        source_lang="en",
        visual_hash="hash",
        polygons=(((40, 60), (240, 60), (240, 110), (40, 110)),),
    )]
    remapped = _remap_scaled_regions(scaled, scale=2.0, offset=offset, image=image)
    assert len(remapped) == 1
    assert remapped[0].bbox.as_tuple() == (520, 230, 100, 25)
    assert remapped[0].polygons == (((520.0, 230.0), (620.0, 230.0), (620.0, 255.0), (520.0, 255.0)),)
    assert remapped[0].visual_hash != "hash"


def test_config_reads_rec_model():
    config = load_config(None)
    assert config.ocr.rec_model == "mobile"
    assert config.ocr.det_model == "mobile"
    assert config.ocr.upscale_retry_max_pixels == 250_000
    assert config.ocr.max_input_side == 640


def test_large_ocr_crop_is_downscaled_and_geometry_mapped_to_capture(monkeypatch):
    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(max_input_side=768, upscale_retry_max_pixels=0)
    calls = []
    image = np.zeros((870, 650, 3), dtype=np.uint8)
    offset = Rect(40, 20, 650, 870)

    def recognize(crop, rect, full_image, check):
        calls.append((crop.shape, rect, full_image.shape))
        return [TextRegion(
            Rect(10, 20, 100, 40), "Balloon text", .97, "en", "scaled-hash",
            metadata={"lines": 1},
            polygons=(((10, 20), (110, 20), (110, 60), (10, 60)),),
        )]

    monkeypatch.setattr(engine, "_recognize_best_for_crop", recognize)
    result = engine._recognize_with_retry(image, offset, image)

    assert calls[0][0] == (768, 574, 3)
    assert result[0].text == "Balloon text"
    assert result[0].bbox.x == 51
    assert result[0].bbox.y == 42
    assert result[0].bbox.width > 100
    assert result[0].metadata == {"lines": 1}
    assert tuple(round(value) for value in result[0].polygons[0][0]) == (51, 43)
    assert result[0].visual_hash != "scaled-hash"


def test_small_ocr_crop_keeps_native_resolution(monkeypatch):
    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(max_input_side=768, upscale_retry_max_pixels=0)
    calls = []
    image = np.zeros((500, 640, 3), dtype=np.uint8)

    def recognize(crop, rect, full_image, check):
        calls.append(crop.shape)
        return [_line(10, 10, 50, 20, "Readable text", .96)]

    monkeypatch.setattr(engine, "_recognize_best_for_crop", recognize)
    engine._recognize_with_retry(image, Rect(0, 0, 640, 500), image)

    assert calls == [(500, 640, 3)]


def test_large_viewport_does_not_trigger_expensive_upscale_retry(monkeypatch):
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
    assert calls == [(640, 640, 3)]


def test_japanese_mobile_ocr_uses_dedicated_paddle_model():
    calls = []

    class PaddleFactory:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(rec_model="mobile")
    engine.languages = ["japan"]
    engine._build_engine(PaddleFactory, "japan")

    assert calls[0]["text_recognition_model_name"] == "japan_PP-OCRv3_mobile_rec"


def test_japanese_server_ocr_uses_supported_paddle_model():
    calls = []

    class PaddleFactory:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(rec_model="server")
    engine._build_engine(PaddleFactory, "japan")

    assert calls[0]["text_recognition_model_name"] == "PP-OCRv5_server_rec"


def test_vertical_japanese_mobile_result_retries_with_server_model():
    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(rec_model="mobile")
    engine.languages = ["japan"]
    crop = np.zeros((400, 200, 3), dtype=np.uint8)
    region = Rect(0, 0, 200, 400)
    mobile = [TextRegion(Rect(70, 10, 48, 340), "本当に大な夫でまかで", .97, "japan", "ja")]
    server = [TextRegion(Rect(70, 10, 48, 340), "本当に大丈夫ですか？", .91, "japan", "ja")]
    calls = []
    engine._get_engine = lambda lang: "mobile"
    engine._get_engine_variant = lambda lang, model: calls.append((lang, model)) or "server"
    engine._run_engine = lambda selected, *_args: server if selected == "server" else mobile

    result = engine._recognize_best_for_crop(crop, region, crop)

    assert calls == [("japan", "server")]
    assert result[0].text == "本当に大丈夫ですか？"


def test_horizontal_japanese_does_not_retry_server_model():
    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(rec_model="mobile")
    engine.languages = ["japan"]
    crop = np.zeros((120, 300, 3), dtype=np.uint8)
    region = Rect(0, 0, 300, 120)
    horizontal = [TextRegion(Rect(20, 30, 240, 38), "今日はいい天気ですね", .94, "japan", "ja")]
    engine._get_engine = lambda lang: "mobile"
    engine._get_engine_variant = lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected retry"))
    engine._run_engine = lambda *_args: horizontal

    result = engine._recognize_best_for_crop(crop, region, crop)

    assert result == horizontal


def test_cached_language_engine_is_available_during_other_model_warmup(monkeypatch):
    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(rec_model="mobile", det_model="mobile")
    cached = object()
    key = ("en", "mobile", "mobile")
    monkeypatch.setitem(_ENGINE_CACHE, key, cached)

    with _ENGINE_LOCK:
        assert engine._get_engine("en") is cached


def test_auto_ocr_uses_japanese_when_english_result_contains_cjk():
    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(rec_model="mobile")
    engine.languages = ["en", "japan"]
    calls = []
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    results = {
        "en": [TextRegion(Rect(0, 0, 80, 24), "正まれ", .99, "en", "en")],
        "japan": [TextRegion(Rect(0, 0, 80, 24), "止まれ", .82, "japan", "ja")],
    }
    engine._get_engine = lambda lang: lang

    def recognize(model, *_args):
        calls.append(model)
        return results[model]

    engine._run_engine = recognize
    selected = engine._recognize_best_for_crop(
        image, Rect(0, 0, 200, 100), image
    )

    assert calls == ["en", "japan"]
    assert selected[0].text == "止まれ"
    assert selected[0].source_lang == "japan"


def test_auto_ocr_skips_japanese_pass_for_clear_english():
    engine = object.__new__(PaddleOcrEngine)
    engine.config = OcrConfig(rec_model="mobile")
    engine.languages = ["en", "japan"]
    calls = []
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    english = [TextRegion(Rect(0, 0, 80, 24), "Thank you", .99, "en", "en")]
    engine._get_engine = lambda lang: lang

    def recognize(model, *_args):
        calls.append(model)
        return english

    engine._run_engine = recognize
    selected = engine._recognize_best_for_crop(
        image, Rect(0, 0, 200, 100), image
    )

    assert calls == ["en"]
    assert selected[0].source_lang == "en"
