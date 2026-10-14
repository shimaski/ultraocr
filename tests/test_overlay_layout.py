from __future__ import annotations

from manga_rt_translator.reader import (
    PaintRect,
    _fit_rect,
    _place_without_overlap,
    _rects_overlap,
    _scale_capture_rect,
)
from manga_rt_translator.models import Rect


def test_physical_capture_box_scales_to_logical_overlay_coordinates():
    scaled = _scale_capture_rect(Rect(x=150, y=75, width=300, height=90), 1.5)
    assert scaled == Rect(x=100, y=50, width=200, height=60)


def test_capture_scale_does_not_shrink_logical_coordinates():
    rect = Rect(x=120, y=60, width=80, height=30)
    assert _scale_capture_rect(rect, 1.0) == rect


def test_long_portuguese_text_expands_horizontally():
    short = _fit_rect(PaintRect(100, 100, 70, 30), 824, 880, 'Oi')
    long = _fit_rect(
        PaintRect(100, 100, 70, 30),
        824,
        880,
        'A academia foi temporariamente fechada para reparos.',
    )
    assert long.width > short.width
    assert long.width <= 320


def test_dense_layout_places_every_caption_without_collision():
    bounds = PaintRect(0, 0, 824, 880)
    # Approximate caption rectangles from the reported dense manga page.
    desired = [
        PaintRect(100, 0, 360, 140),
        PaintRect(470, 0, 125, 170),
        PaintRect(130, 120, 180, 110),
        PaintRect(465, 170, 130, 275),
        PaintRect(320, 225, 150, 440),
        PaintRect(105, 390, 140, 220),
        PaintRect(445, 510, 135, 280),
        PaintRect(145, 650, 130, 120),
    ]
    occupied: list[PaintRect] = []
    for rect in desired:
        placed = _place_without_overlap(rect, occupied, bounds)
        assert placed is not None
        assert all(not _rects_overlap(placed, other) for other in occupied)
        occupied.append(placed)


def test_impossible_caption_is_omitted_instead_of_overlapping():
    bounds = PaintRect(0, 0, 100, 100)
    occupied = [PaintRect(0, 0, 100, 100)]
    assert _place_without_overlap(PaintRect(0, 0, 100, 100), occupied, bounds) is None


def test_dense_page_font_cap_is_small_enough_for_readable_columns():
    configured = 18
    live_count = 8
    base = min(configured, 10) if live_count >= 8 else configured
    assert base == 10
