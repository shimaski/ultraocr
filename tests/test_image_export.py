import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QImage, QPainter

from manga_rt_translator.image_export import composite_overlay_image


def _pixel(image: QImage, x: int = 0, y: int = 0) -> tuple[int, int, int, int]:
    color = image.pixelColor(x, y)
    return color.red(), color.green(), color.blue(), color.alpha()


def test_export_composites_transparent_overlay_over_page_without_mutating_it():
    page = QImage(8, 8, QImage.Format.Format_ARGB32)
    page.fill(QColor(20, 40, 60, 255))
    overlay = QImage(8, 8, QImage.Format.Format_ARGB32)
    overlay.fill(QColor(0, 0, 0, 0))
    painter = QPainter(overlay)
    painter.fillRect(0, 0, 4, 4, QColor(220, 30, 40, 255))
    painter.end()

    result = composite_overlay_image(page, overlay)

    assert _pixel(result, 1, 1) == (220, 30, 40, 255)
    assert _pixel(result, 6, 6) == (20, 40, 60, 255)
    assert _pixel(page, 1, 1) == (20, 40, 60, 255)


def test_export_resizes_overlay_to_the_page_pixel_dimensions():
    page = QImage(20, 12, QImage.Format.Format_ARGB32)
    page.fill(QColor("white"))
    overlay = QImage(10, 6, QImage.Format.Format_ARGB32)
    overlay.fill(QColor(0, 0, 0, 0))
    painter = QPainter(overlay)
    painter.fillRect(0, 0, 5, 3, QColor("red"))
    painter.end()

    result = composite_overlay_image(page, overlay)

    assert result.size() == page.size()
    assert _pixel(result, 2, 2) == (255, 0, 0, 255)
    assert _pixel(result, 18, 10) == (255, 255, 255, 255)
