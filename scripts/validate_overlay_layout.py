"""Render a dense caption fixture and fail if any translated boxes overlap."""
from __future__ import annotations

import argparse
from pathlib import Path

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QApplication

from manga_rt_translator.reader import (
    PaintRect,
    _fit_rect,
    _fit_text_rect,
    _place_without_overlap,
    _rects_overlap,
)


CAPTIONS = [
    (PaintRect(100, 0, 350, 90), 'A academia foi temporariamente fechada para reparos.'),
    (PaintRect(470, 0, 120, 130), 'Desde o dia após o ataque.'),
    (PaintRect(130, 220, 150, 55), 'Capítulo'),
    (PaintRect(465, 170, 125, 260), 'Eu estava planejando ajudar com os reparos também, mas...'),
    (PaintRect(320, 225, 145, 390), 'Está bem. Deixe isso para os professores que não puderam fazer nada.'),
    (PaintRect(105, 390, 135, 200), 'E deixe-nos ter nosso momento de brilhar.'),
    (PaintRect(445, 510, 130, 260), 'O herói da hora deve esticar as asas e se divertir.'),
    (PaintRect(145, 650, 125, 105), '...foi o que me disseram.'),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path,
                        default=Path('reader_debug_output/overlay-layout-validation.png'))
    args = parser.parse_args()
    QApplication.instance() or QApplication([])
    image = QImage(str(args.image)).convertToFormat(QImage.Format.Format_ARGB32)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    bounds = PaintRect(4, 4, image.width()-8, image.height()-8)
    occupied: list[PaintRect] = []
    layout = []
    flags = int(Qt.TextFlag.TextWordWrap | Qt.AlignmentFlag.AlignCenter)
    for source, text in sorted(CAPTIONS, key=lambda value: value[0].width*value[0].height,
                               reverse=True):
        chosen = None
        base_size = 10 if len(CAPTIONS) >= 8 else 12 if len(CAPTIONS) >= 5 else 18
        for size in range(base_size, 7, -1):
            font = QFont('Segoe UI', size)
            font.setBold(True)
            painter.setFont(font)
            rect = _fit_rect(source, image.width(), image.height(), text)
            rect = _fit_text_rect(rect, text, painter, bounds.y+bounds.height,
                                  flags, QRect)
            if rect.height > bounds.height:
                continue
            placed = _place_without_overlap(rect, occupied, bounds)
            if placed:
                chosen = placed, font
                break
        if chosen is None:
            raise SystemExit(f'Could not place: {text}')
        rect, font = chosen
        if any(_rects_overlap(rect, other) for other in occupied):
            raise SystemExit(f'Overlap detected: {text}')
        occupied.append(rect)
        layout.append((rect, font, text))
        print(f'placed size={font.pointSize()} rect={rect} text={text[:24]}')
    for rect, font, text in layout:
        painter.setFont(font)
        path = QPainterPath()
        path.addRoundedRect(rect.x, rect.y, rect.width, rect.height, 8, 8)
        painter.fillPath(path, QColor(255, 252, 245, 238))
        painter.setPen(QPen(QColor(36, 31, 28, 120), 1))
        painter.drawPath(path)
        painter.setPen(QColor(24, 20, 18))
        painter.drawText(QRect(rect.x+8, rect.y+6, rect.width-16, rect.height-12),
                         flags | int(Qt.AlignmentFlag.AlignVCenter), text)
    painter.end()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not image.save(str(args.output)):
        raise SystemExit('Could not save validation image')
    print(f'OK: {len(layout)} captions, 0 overlaps -> {args.output}')


if __name__ == '__main__':
    main()
