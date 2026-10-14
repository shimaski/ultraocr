from manga_rt_translator.reader import _reader_ocr_languages


def test_reader_auto_checks_english_then_japanese_only():
    assert _reader_ocr_languages("Auto") == ("en", ["japan"])


def test_reader_explicit_language_disables_fallback_models():
    assert _reader_ocr_languages("japan") == ("japan", [])
    assert _reader_ocr_languages("en") == ("en", [])
    assert _reader_ocr_languages("korean") == ("korean", [])
    assert _reader_ocr_languages("ch") == ("ch", [])
