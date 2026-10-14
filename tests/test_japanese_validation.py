from types import SimpleNamespace

from scripts.validate_japanese_reader import (
    _has_japanese,
    _japanese_char_count,
    _qualifying_translations,
)


def test_japanese_validation_rejects_translated_ocr_fragments():
    fragments = [
        SimpleNamespace(source_text="アよ", text="Ai"),
        SimpleNamespace(source_text="ル「", text="eu"),
    ]

    assert _has_japanese("アドビは")
    assert not _has_japanese("Adobe")
    assert _japanese_char_count("アドビは Adobe") == 4
    assert _qualifying_translations(fragments, 4, "") == []


def test_japanese_validation_can_require_an_expected_source_fragment():
    item = SimpleNamespace(
        source_text="アドビは世界トップレベル",
        text="A Adobe está no topo.",
    )

    assert _qualifying_translations([item], 4, "アドビは") == [item]
    assert _qualifying_translations([item], 4, "存在しない") == []
    assert _qualifying_translations(
        [SimpleNamespace(source_text=item.source_text, text=item.source_text)], 4, ""
    ) == []
