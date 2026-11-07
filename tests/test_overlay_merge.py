from __future__ import annotations

from manga_rt_translator.models import OverlayItem, Rect
from manga_rt_translator.pipeline import MangaTranslationPipeline as Pipeline


def item(x, y, w, h, text, source_text=""):
    return OverlayItem(
        bbox=Rect(x, y, w, h),
        text=text,
        source_text=source_text,
        confidence=0.9,
        expires_at=1e18,
        visual_hash=f"{x}-{y}",
    )


def test_stacked_fragments_in_one_balloon_merge_into_single_caption():
    # Bubble grouping missed: one balloon's sentence came back as several small
    # fragments whose painted cards pile up. They must collapse to one caption.
    frags = [
        item(400, 120, 70, 20, "TALVEZ FOI UM SUPER ERRO"),
        item(430, 150, 60, 20, "FILME DO"),
        item(410, 175, 90, 20, "CINEMA TAO CEDO"),
    ]
    out = Pipeline._merge_overlapping_items(list(frags))
    assert len(out) == 1
    assert "TALVEZ FOI UM SUPER ERRO" in out[0].text
    assert "CINEMA TAO CEDO" in out[0].text


def test_separate_balloons_are_not_merged():
    far = [
        item(50, 50, 120, 40, "Primeiro balao"),
        item(50, 400, 120, 40, "Segundo balao"),
    ]
    out = Pipeline._merge_overlapping_items(list(far))
    assert len(out) == 2


def test_single_item_passes_through():
    out = Pipeline._merge_overlapping_items([item(10, 10, 100, 30, "solo")])
    assert len(out) == 1 and out[0].text == "solo"


def test_provisional_source_item_is_not_merged_into_translations():
    # A provisional (source_text == text) overlapping a real translation must
    # not be folded into it; only finished translations merge.
    provisional = item(400, 120, 70, 20, "RAW TEXT", "RAW TEXT")
    translation = item(405, 125, 80, 20, "Traducao real")
    out = Pipeline._merge_overlapping_items([provisional, translation])
    assert len(out) == 2


def test_merged_caption_keeps_erased_background_patch():
    import numpy as np

    from manga_rt_translator.models import OverlayItem, Rect

    def with_patch(x, y, w, h, text):
        patch = np.full((h + 8, w + 8, 3), 250, np.uint8)
        return OverlayItem(
            bbox=Rect(x, y, w, h), text=text, confidence=0.9, expires_at=1e18,
            visual_hash=f"{x}", background_color=(250, 250, 250),
            background_patch=patch, background_rect=Rect(x - 4, y - 4, w + 8, h + 8),
        )

    out = Pipeline._merge_overlapping_items([
        with_patch(400, 120, 70, 20, "COMO ELES"),
        with_patch(410, 128, 80, 20, "ESTAO ALCANCANDO"),
    ])
    assert len(out) == 1
    merged = out[0]
    assert merged.background_patch is not None and merged.background_rect is not None
    rect = merged.background_rect
    assert merged.background_patch.shape[:2] == (rect.height, rect.width)
    # The patch covers every fragment's source text box.
    for x, y, w, h in ((400, 120, 70, 20), (410, 128, 80, 20)):
        assert rect.x <= x and rect.y <= y and rect.x2 >= x + w and rect.y2 >= y + h


def test_balloon_touching_side_edge_is_not_deferred():
    from manga_rt_translator.models import Rect, TextRegion
    from manga_rt_translator.pipeline import _region_inside_capture

    side = TextRegion(bbox=Rect(2, 300, 110, 100), text="I'D CHASE AFTER", confidence=0.9)
    assert _region_inside_capture(side, 600, 870)
    cut_bottom = TextRegion(bbox=Rect(200, 820, 110, 60), text="IN THAT CASE", confidence=0.9)
    assert not _region_inside_capture(cut_bottom, 600, 870)


def test_same_balloon_read_twice_is_not_repeated():
    whole = item(80, 700, 120, 90, "...SE VOCE E MEU AMIGO")
    fragment = item(100, 720, 70, 30, "VOCE E MEU AMIGO")
    out = Pipeline._merge_overlapping_items([fragment, whole])
    assert len(out) == 1
    assert out[0].text == "...SE VOCE E MEU AMIGO"
