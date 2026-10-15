from manga_rt_translator.reader import _reader_stylesheet


def test_reader_dark_theme_uses_dark_surfaces_and_readable_text():
    stylesheet = _reader_stylesheet(True)

    assert "background: #171b1e" in stylesheet
    assert "color: #e5ecee" in stylesheet
    assert "selection-background-color: #087f86" in stylesheet


def test_reader_light_theme_keeps_light_surfaces():
    stylesheet = _reader_stylesheet(False)

    assert "background: #f4f6f6" in stylesheet
    assert "background: #ffffff" in stylesheet
    assert "color: #263437" in stylesheet
