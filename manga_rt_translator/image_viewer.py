from __future__ import annotations

import html
from pathlib import Path
from typing import Callable


IMAGE_FILE_FILTER = "Imagens (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff)"


SUPPORTED_IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff",
}


def build_image_gallery_html(paths: list[Path]) -> str:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImageReader

    image_tags = []
    for path in paths:
        source = html.escape(
            QUrl.fromLocalFile(str(path)).toString(
                QUrl.ComponentFormattingOption.FullyEncoded
            ),
            quote=True,
        )
        size = QImageReader(str(path)).size()
        dimensions = (
            f' width="{size.width()}" height="{size.height()}"'
            if size.isValid()
            else ""
        )
        image_tags.append(
            f'<img src="{source}" alt="" loading="lazy"{dimensions}>'
        )
    images = "\n".join(image_tags)
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\"><style>"
        "html,body{margin:0;background:#202124}"
        "main{width:100%;padding:12px 0}"
        "img{display:block;width:auto;height:auto;max-width:100%;"
        "max-height:none;margin:0 auto 16px;object-fit:contain}"
        "</style></head><body><main>"
        f"{images}</main></body></html>"
    )


def choose_local_images(
    parent: object,
    webview: object,
    status_callback: Callable[[str], None],
) -> bool:
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QFileDialog

    filenames, _selected_filter = QFileDialog.getOpenFileNames(
        parent,
        "Abrir imagens para traduzir",
        str(Path.home()),
        IMAGE_FILE_FILTER,
    )
    if not filenames:
        return False
    paths = [Path(filename).expanduser().resolve() for filename in filenames]
    if any(
        not path.is_file() or path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
        for path in paths
    ):
        status_callback("Selecione um arquivo de imagem compatível.")
        return False

    if len(paths) == 1:
        webview.setUrl(QUrl.fromLocalFile(str(paths[0])))
        status_callback("Imagem aberta. Selecione a região e pressione Traduzir região.")
    else:
        webview.setHtml(
            build_image_gallery_html(paths),
            QUrl.fromLocalFile(str(paths[0].parent) + "/"),
        )
        status_callback(
            f"{len(paths)} imagens abertas. Use o modo Contínuo para traduzir ao rolar."
        )
    return True
