from pathlib import Path
from unittest.mock import patch

from PySide6.QtCore import QUrl

from manga_rt_translator.image_viewer import (
    build_image_gallery_html,
    choose_local_images,
)


class StubWebView:
    def __init__(self) -> None:
        self.url: QUrl | None = None
        self.html: str | None = None
        self.base_url: QUrl | None = None

    def setUrl(self, url: QUrl) -> None:  # noqa: N802
        self.url = url

    def setHtml(self, content: str, base_url: QUrl) -> None:  # noqa: N802
        self.html = content
        self.base_url = base_url


def test_choose_local_image_opens_supported_image_in_reader(tmp_path):
    image_path = tmp_path / "chapter-page.webp"
    image_path.write_bytes(b"test image payload")
    webview = StubWebView()
    statuses: list[str] = []

    with patch(
        "PySide6.QtWidgets.QFileDialog.getOpenFileNames",
        return_value=([str(image_path)], ""),
    ):
        assert choose_local_images(None, webview, statuses.append)

    assert webview.url == QUrl.fromLocalFile(str(image_path.resolve()))
    assert "Imagem aberta" in statuses[-1]


def test_choose_local_image_rejects_non_image_and_cancel(tmp_path):
    non_image = tmp_path / "notes.txt"
    non_image.write_text("not an image", encoding="utf-8")
    webview = StubWebView()
    statuses: list[str] = []

    with patch(
        "PySide6.QtWidgets.QFileDialog.getOpenFileNames",
        return_value=([str(non_image)], ""),
    ):
        assert not choose_local_images(None, webview, statuses.append)
    assert webview.url is None
    assert "compatível" in statuses[-1]

    with patch(
        "PySide6.QtWidgets.QFileDialog.getOpenFileNames",
        return_value=([], ""),
    ):
        assert not choose_local_images(None, webview, statuses.append)


def test_choose_multiple_images_builds_a_lazy_scrollable_gallery(tmp_path):
    paths = [tmp_path / "page 1.png", tmp_path / "page&2.webp"]
    for path in paths:
        path.write_bytes(b"test image payload")
    webview = StubWebView()
    statuses: list[str] = []

    with patch(
        "PySide6.QtWidgets.QFileDialog.getOpenFileNames",
        return_value=([str(path) for path in paths], ""),
    ):
        assert choose_local_images(None, webview, statuses.append)

    assert webview.url is None
    assert webview.html is not None
    assert webview.html.count("<img ") == 2
    assert "%20" in webview.html and "&amp;2" in webview.html
    assert webview.html.count('loading="lazy"') == 2
    assert webview.base_url == QUrl.fromLocalFile(str(tmp_path) + "/")
    assert "2 imagens abertas" in statuses[-1]


def test_image_gallery_contains_only_escaped_file_url_attributes(tmp_path):
    paths = [tmp_path / 'page"><script>.png']
    markup = build_image_gallery_html(paths)
    assert "<script>" not in markup
    assert "%22%3E%3Cscript%3E" in markup
    assert "file:///" in markup
