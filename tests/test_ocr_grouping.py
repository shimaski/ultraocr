from __future__ import annotations

import numpy as np

from manga_rt_translator.models import Rect, TextRegion
from manga_rt_translator.ocr import _cjk_ratio, _looks_like_cjk_text, group_text_regions
from manga_rt_translator.pipeline import _filter_text_regions


def _line(x: int, y: int, width: int, height: int, text: str, lang: str = "en") -> TextRegion:
    return TextRegion(
        bbox=Rect(x=x, y=y, width=width, height=height),
        text=text,
        confidence=0.9,
        source_lang=lang,
        visual_hash=f"hash-{x}-{y}",
    )


def test_multi_line_bubble_is_grouped_into_one_block():
    lines = [
        _line(100, 100, 200, 30, "Because switching gears"),
        _line(100, 140, 200, 30, "is something I am good at."),
    ]
    blocks = group_text_regions(lines)
    assert len(blocks) == 1
    assert blocks[0].text == "Because switching gears is something I am good at."
    assert blocks[0].bbox.as_tuple() == (100, 100, 200, 70)
    assert blocks[0].metadata["lines"] == 2


def test_distant_boxes_stay_separate():
    lines = [
        _line(100, 100, 200, 30, "Hello there."),
        _line(600, 500, 200, 30, "Goodbye now."),
    ]
    blocks = group_text_regions(lines)
    assert len(blocks) == 2


def test_lines_with_too_much_vertical_gap_stay_separate():
    lines = [
        _line(100, 100, 200, 30, "First bubble."),
        _line(100, 240, 200, 30, "Second bubble."),  # gap of 110 px vs 30 px line height
    ]
    blocks = group_text_regions(lines)
    assert len(blocks) == 2


def test_short_second_line_still_groups():
    lines = [
        _line(140, 100, 120, 30, "Thanks!"),
        _line(100, 140, 200, 30, "I owe you one."),
    ]
    blocks = group_text_regions(lines)
    assert len(blocks) == 1
    assert blocks[0].text == "Thanks! I owe you one."


def test_cjk_vertical_columns_read_right_to_left():
    lines = [
        _line(90, 100, 40, 220, "左です", lang="japan"),
        _line(300, 100, 40, 220, "右です", lang="japan"),
    ]
    blocks = group_text_regions(lines)
    assert len(blocks) == 2
    assert blocks[0].text == "右です"
    assert blocks[1].text == "左です"


def test_cjk_members_join_without_spaces():
    lines = [
        _line(300, 80, 40, 90, "君の", lang="japan"),
        _line(300, 200, 40, 90, "せいだ", lang="japan"),
    ]
    blocks = group_text_regions(lines)
    assert len(blocks) == 1
    assert blocks[0].text == "君のせいだ"


def test_horizontal_text_reads_top_to_bottom_left_to_right():
    lines = [
        _line(400, 100, 200, 30, "Right bubble."),
        _line(100, 100, 200, 30, "Left bubble."),
        _line(100, 400, 200, 30, "Lower bubble."),
    ]
    blocks = group_text_regions(lines)
    assert [b.text for b in blocks] == ["Left bubble.", "Right bubble.", "Lower bubble."]


def test_visual_hash_is_recomputed_for_grouped_bbox():
    image = np.zeros((400, 400, 3), dtype=np.uint8)
    image[100:170, 100:300] = 255
    lines = [
        _line(100, 100, 200, 30, "Line one."),
        _line(100, 140, 200, 30, "Line two."),
    ]
    blocks = group_text_regions(lines, image)
    assert len(blocks) == 1
    assert blocks[0].visual_hash
    assert blocks[0].visual_hash != lines[0].visual_hash


def test_filter_keeps_same_text_in_disjoint_bubbles():
    frame_w, frame_h = 800, 600
    regions = [
        _line(100, 100, 120, 40, "Yeah."),
        _line(500, 400, 120, 40, "Yeah."),
    ]
    kept = _filter_text_regions(regions, frame_w, frame_h)
    assert len(kept) == 2


def test_filter_drops_double_detected_same_region():
    frame_w, frame_h = 800, 600
    regions = [
        _line(100, 100, 120, 40, "Yeah."),
        _line(104, 102, 118, 40, "Yeah."),
    ]
    kept = _filter_text_regions(regions, frame_w, frame_h)
    assert len(kept) == 1


def test_korean_reads_left_to_right_and_keeps_word_spaces():
    lines = [
        _line(100, 100, 200, 40, "왼쪽 말풍선", lang="korean"),
        _line(400, 100, 200, 40, "오른쪽 말풍선", lang="korean"),
    ]
    blocks = group_text_regions(lines)
    assert [b.text for b in blocks] == ["왼쪽 말풍선", "오른쪽 말풍선"]


def test_korean_stacked_lines_group_with_space():
    lines = [
        _line(100, 100, 220, 36, "그건 내 잘못이", lang="korean"),
        _line(100, 146, 220, 36, "아니야", lang="korean"),
    ]
    blocks = group_text_regions(lines)
    assert len(blocks) == 1
    assert blocks[0].text == "그건 내 잘못이 아니야"


def test_hangul_triggers_engine_fallback_but_not_rtl_order():
    assert _looks_like_cjk_text([_line(0, 0, 10, 10, "한국어")])
    # Hangul is not part of the right-to-left CJK reading heuristic.
    assert _cjk_ratio("한국어") == 0.0
