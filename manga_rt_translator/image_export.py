from __future__ import annotations


def composite_overlay_image(page_image: object, overlay_image: object):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage, QPainter

    if page_image.isNull():
        raise ValueError("A captura da página está vazia.")
    if overlay_image.isNull():
        raise ValueError("A captura do overlay está vazia.")

    result = page_image.convertToFormat(QImage.Format.Format_ARGB32_Premultiplied)
    overlay = overlay_image.convertToFormat(
        QImage.Format.Format_ARGB32_Premultiplied
    )
    result.setDevicePixelRatio(1.0)
    overlay.setDevicePixelRatio(1.0)
    if overlay.size() != result.size():
        overlay = overlay.scaled(
            result.size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    painter = QPainter(result)
    painter.drawImage(QPoint(0, 0), overlay)
    painter.end()
    return result
