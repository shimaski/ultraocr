from __future__ import annotations

import numpy as np

from manga_rt_translator.models import Rect
from manga_rt_translator.overlay_background import (
    detect_speech_bubble_bounds,
    foreground_color,
    inpaint_text_background,
    sample_background_color,
)


def test_samples_bubble_fill_outside_dark_glyphs():
    image = np.full((60, 80, 3), (244, 241, 232), dtype=np.uint8)
    image[25:31, 28:54] = (10, 10, 10)

    assert sample_background_color(image, Rect(28, 25, 26, 6)) == (244, 241, 232)


def test_samples_dark_bubble_background_and_selects_light_text():
    image = np.full((50, 70, 3), (35, 39, 44), dtype=np.uint8)
    image[22:27, 24:48] = (245, 245, 245)
    color = sample_background_color(image, Rect(24, 22, 24, 5))

    assert color == (35, 39, 44)
    assert foreground_color(color) == (255, 255, 255)


def test_detects_closed_bubble_for_a_better_translation_anchor():
    image = np.full((260, 340, 3), (120, 120, 120), dtype=np.uint8)
    image[56:204, 76:244] = (249, 249, 249)
    image[56:60, 76:244] = (15, 15, 15)
    image[200:204, 76:244] = (15, 15, 15)
    image[56:204, 76:80] = (15, 15, 15)
    image[56:204, 240:244] = (15, 15, 15)
    image[103:118, 132:136] = (12, 12, 12)
    image[103:107, 132:150] = (12, 12, 12)
    text_rect = Rect(124, 99, 68, 24)

    bounds = detect_speech_bubble_bounds(
        image, text_rect, background=(249, 249, 249)
    )

    assert bounds is not None
    assert bounds.contains(text_rect)
    assert bounds.width >= 150
    assert bounds.height >= 130


def test_does_not_expand_into_unbounded_page_background():
    image = np.full((160, 220, 3), (249, 249, 249), dtype=np.uint8)
    image[70:85, 90:94] = (12, 12, 12)
    image[70:74, 90:108] = (12, 12, 12)

    assert detect_speech_bubble_bounds(
        image, Rect(82, 66, 50, 24), background=(249, 249, 249)
    ) is None


def test_clips_box_at_frame_edge_and_handles_grayscale():
    image = np.full((16, 20), 205, dtype=np.uint8)

    assert sample_background_color(image, Rect(0, 0, 8, 8)) == (205, 205, 205)
    assert sample_background_color(image, Rect(40, 40, 5, 5)) is None


def test_inpaints_glyph_strokes_inside_local_crop():
    image = np.full((60, 80, 3), (244, 241, 232), dtype=np.uint8)
    image[24:36, 30:32] = (12, 12, 12)
    image[24:26, 30:39] = (12, 12, 12)

    result = inpaint_text_background(image, Rect(28, 22, 20, 18))

    assert result is not None
    patch_rect, patch = result
    assert patch.shape[:2] == (patch_rect.height, patch_rect.width)
    assert patch_rect.contains(Rect(28, 22, 20, 18))
    assert np.mean(patch[24 - patch_rect.y : 36 - patch_rect.y, 30 - patch_rect.x : 32 - patch_rect.x]) > 180


def test_polygon_mask_preserves_contrast_marks_outside_text_shape():
    image = np.full((60, 80, 3), 245, dtype=np.uint8)
    image[24:36, 30:33] = 12
    image[25:28, 43:46] = 12
    polygon = ((29, 23), (35, 23), (35, 37), (29, 37))

    result = inpaint_text_background(
        image,
        Rect(28, 22, 24, 18),
        polygons=(polygon,),
    )

    assert result is not None
    patch_rect, patch = result
    preserved_mark = patch[25 - patch_rect.y : 28 - patch_rect.y, 43 - patch_rect.x : 46 - patch_rect.x]
    erased_glyph = patch[24 - patch_rect.y : 36 - patch_rect.y, 30 - patch_rect.x : 33 - patch_rect.x]
    assert np.min(preserved_mark) == 12
    assert np.mean(erased_glyph) > 180
    assert inpaint_text_background(
        image,
        Rect(28, 22, 24, 18),
        polygons=(((30, 30), (31, 31)),),
    ) is not None
