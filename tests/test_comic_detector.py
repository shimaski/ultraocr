from __future__ import annotations

from manga_rt_translator.comic_detector import _dedupe_boxes, block_index_for
from manga_rt_translator.models import Rect, TextRegion
from manga_rt_translator.pipeline import _detector_recovery_crops


def region(x, y, w, h, text):
    return TextRegion(bbox=Rect(x, y, w, h), text=text, confidence=0.9)


def test_line_belongs_to_block_holding_most_of_it():
    blocks = [Rect(0, 0, 100, 100), Rect(200, 0, 100, 100)]
    assert block_index_for(Rect(10, 10, 50, 20), blocks) == 0
    assert block_index_for(Rect(210, 40, 50, 20), blocks) == 1
    assert block_index_for(Rect(120, 40, 50, 20), blocks) is None


def test_unread_and_half_read_blocks_are_recovered():
    blocks = [
        Rect(370, 255, 90, 115),   # DON'T HURT HER! - OCR read nothing
        Rect(130, 480, 55, 80),    # IF IT WERE ME - OCR read only "IF IT"
        Rect(410, 680, 110, 120),  # fully read
    ]
    regions = [
        region(132, 482, 51, 33, "IF IT"),
        region(412, 680, 107, 119, "YOU HAVE NOTHING TO DO WITH ME"),
    ]
    crops = _detector_recovery_crops(regions, blocks, None, 600, 900)
    assert len(crops) == 2
    assert all(not crop.contains(Rect(412, 680, 107, 119)) for crop in crops)


def test_recovery_respects_changed_rois():
    blocks = [Rect(10, 10, 50, 50), Rect(300, 300, 50, 50)]
    crops = _detector_recovery_crops([], blocks, [Rect(0, 0, 100, 100)], 600, 900)
    assert len(crops) == 1 and crops[0].x <= 10


def test_slice_duplicates_are_merged():
    boxes = _dedupe_boxes([Rect(0, 0, 100, 100), Rect(5, 5, 90, 90), Rect(300, 0, 50, 50)])
    assert len(boxes) == 2
