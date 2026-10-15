from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "docs" / "media"
BG = "#F4F7F7"
INK = "#183237"
MUTED = "#5D7276"
TEAL = "#087F86"
MINT = "#D9F0EE"
CORAL = "#E76F51"
WHITE = "#FFFFFF"
LINE = "#C9D7D8"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    path = Path("C:/Windows/Fonts") / name
    return ImageFont.truetype(str(path), size) if path.exists() else ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], value: str, size: int,
         fill: str = INK, bold: bool = False, anchor: str | None = None) -> None:
    draw.text(xy, value, font=font(size, bold), fill=fill, anchor=anchor)


def pipeline_image() -> None:
    image = Image.new("RGB", (1440, 440), BG)
    draw = ImageDraw.Draw(image)
    text(draw, (54, 38), "DO CAPÍTULO AO PT-BR", 30, INK, True)
    text(draw, (56, 82), "O trabalho pesado só acontece quando a imagem ou o texto muda.", 17, MUTED)
    steps = [
        ("01", "Captura", "janela / região", TEAL),
        ("02", "Mudanças", "diff por blocos", CORAL),
        ("03", "OCR local", "inglês / japonês", TEAL),
        ("04", "Memória", "SQLite antes da IA", CORAL),
        ("05", "Tradução", "provedor escolhido", TEAL),
        ("06", "Overlay", "texto em PT-BR", CORAL),
    ]
    left, top, width, height, gap = 52, 175, 202, 180, 32
    for index, (number, title, detail, accent) in enumerate(steps):
        x = left + index * (width + gap)
        draw.rounded_rectangle((x, top, x + width, top + height), radius=8,
                               fill=WHITE, outline=LINE, width=2)
        draw.rounded_rectangle((x, top, x + width, top + 9), radius=5, fill=accent)
        draw.ellipse((x + 18, top + 27, x + 57, top + 66), fill=MINT if accent == TEAL else "#FBE5DE")
        text(draw, (x + 37, top + 47), number, 14, accent, True, "mm")
        text(draw, (x + 18, top + 86), title, 21, INK, True)
        text(draw, (x + 18, top + 124), detail, 14, MUTED)
        if index < len(steps) - 1:
            start = x + width + 5
            end = start + gap - 10
            y = top + height // 2
            draw.line((start, y, end, y), fill=TEAL, width=3)
            draw.polygon([(end, y), (end - 8, y - 6), (end - 8, y + 6)], fill=TEAL)
    text(draw, (54, 390), "CACHE HIT → overlay imediato     •     CACHE MISS → traduz e salva para reutilizar", 15, TEAL, True)
    image.save(MEDIA / "pipeline.png", optimize=True)


def draw_comic(draw: ImageDraw.ImageDraw, top: int, mode: str) -> None:
    x0, x1 = 54, 504
    draw.rounded_rectangle((x0, top, x1, top + 360), radius=5, fill="#F8F8F6", outline=INK, width=3)
    panels = [(x0 + 14, top + 14, x1 - 14, top + 126),
              (x0 + 14, top + 136, x0 + 215, top + 345),
              (x0 + 229, top + 136, x1 - 14, top + 345)]
    for box in panels:
        draw.rectangle(box, fill="#ECECEA", outline=INK, width=2)
    # Original, deliberately simple line-art characters and speech balloons.
    draw.ellipse((115, top + 38, 174, top + 97), fill="#D2D2D0", outline=INK, width=3)
    draw.arc((99, top + 78, 192, top + 147), 180, 360, fill=INK, width=4)
    draw.ellipse((334, top + 32, 391, top + 89), fill="#C6C6C4", outline=INK, width=3)
    draw.arc((314, top + 75, 410, top + 151), 180, 360, fill=INK, width=4)
    for cx, cy in [(154, top + 187), (369, top + 193)]:
        draw.ellipse((cx - 26, cy - 29, cx + 26, cy + 23), fill="#D7D7D5", outline=INK, width=2)
        draw.line((cx, cy + 23, cx, cy + 94), fill=INK, width=4)
        draw.line((cx, cy + 48, cx - 27, cy + 73), fill=INK, width=3)
        draw.line((cx, cy + 48, cx + 28, cy + 70), fill=INK, width=3)
        draw.line((cx, cy + 94, cx - 22, cy + 124), fill=INK, width=3)
        draw.line((cx, cy + 94, cx + 24, cy + 124), fill=INK, width=3)
    draw.rounded_rectangle((66, top + 23, 199, top + 69), radius=9, fill=WHITE, outline=INK, width=2)
    draw.polygon([(105, top + 69), (119, top + 69), (107, top + 81)], fill=WHITE, outline=INK)
    text(draw, (132, top + 46), "We have to go!", 14, INK, True, "mm")
    draw.rounded_rectangle((325, top + 23, 489, top + 69), radius=9, fill=WHITE, outline=INK, width=2)
    text(draw, (407, top + 46), "Wait for me!", 14, INK, True, "mm")
    if mode in {"ocr", "translated"}:
        for box in [(66, top + 23, 199, top + 69), (325, top + 23, 489, top + 69)]:
            _dashed_rectangle(draw, box, CORAL if mode == "ocr" else TEAL, 3)
    if mode == "translated":
        draw.rounded_rectangle((59, top + 81, 257, top + 121), radius=8, fill="#DDF2EF", outline=TEAL, width=2)
        text(draw, (158, top + 101), "Temos que ir!", 15, TEAL, True, "mm")
        draw.rounded_rectangle((293, top + 81, 495, top + 121), radius=8, fill="#DDF2EF", outline=TEAL, width=2)
        text(draw, (394, top + 101), "Me espera!", 15, TEAL, True, "mm")


def _dashed_rectangle(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], color: str, width: int) -> None:
    x0, y0, x1, y1 = box
    for x in range(x0, x1, 12):
        draw.line((x, y0, min(x + 7, x1), y0), fill=color, width=width)
        draw.line((x, y1, min(x + 7, x1), y1), fill=color, width=width)
    for y in range(y0, y1, 12):
        draw.line((x0, y, x0, min(y + 7, y1)), fill=color, width=width)
        draw.line((x1, y, x1, min(y + 7, y1)), fill=color, width=width)


def workflow_frame(step: int) -> Image.Image:
    image = Image.new("RGB", (1120, 520), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1120, 84), fill=INK)
    text(draw, (42, 22), "ULTRAOCR  /  LEITOR EM TEMPO REAL", 24, WHITE, True)
    captions = ["1  Abra o capítulo", "2  Marque a área uma vez", "3  OCR lê somente a região", "4  Tradução alinhada ao balão"]
    text(draw, (43, 106), captions[step].upper(), 18, TEAL, True)
    draw_comic(draw, 139, ["ready", "selected", "ocr", "translated"][step])
    draw.rounded_rectangle((540, 139, 1073, 499), radius=8, fill=WHITE, outline=LINE, width=2)
    text(draw, (568, 164), "Tradução e leitura", 19, INK, True)
    controls = ["Cole o link do capítulo", "Selecione a coluna de leitura", "OCR local reconhece os balões", "Texto em português sobre a página"]
    for i, label in enumerate(controls):
        y = 211 + i * 60
        color = TEAL if i <= step else LINE
        draw.ellipse((568, y, 592, y + 24), fill=color)
        text(draw, (580, y + 12), str(i + 1), 12, WHITE, True, "mm")
        text(draw, (607, y + 2), label, 15, INK if i <= step else MUTED, i == step)
    draw.rounded_rectangle((568, 462, 1044, 488), radius=6, fill=MINT)
    status = ["Leitor pronto", "Área selecionada", "Texto detectado: inglês", "Tradução PT-BR pronta"][step]
    text(draw, (582, 475), status, 13, TEAL, True, "lm")
    return image


def scroll_frame(step: int) -> Image.Image:
    image = Image.new("RGB", (1120, 520), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1120, 84), fill=INK)
    text(draw, (42, 22), "ULTRAOCR  /  ROLAGEM INCREMENTAL", 24, WHITE, True)
    text(draw, (43, 106), "MUDOU A VIEWPORT? SÓ A ÁREA NOVA É REANALISADA.", 17, TEAL, True)
    viewport = (214, 142, 906, 456)
    draw.rounded_rectangle(viewport, radius=5, fill="#E8E8E6", outline=INK, width=3)
    y_positions = [156 - step * 36, 251 - step * 36, 346 - step * 36]
    for index, y in enumerate(y_positions):
        if 145 <= y <= 380:
            draw.rectangle((231, y, 889, y + 76), fill="#F8F8F6", outline=INK, width=2)
            draw.ellipse((258, y + 17, 303, y + 62), fill="#C9C9C7", outline=INK, width=2)
            draw.line((280, y + 62, 280, y + 70), fill=INK, width=3)
            words = ["I found the way!", "Keep moving!", "We're almost there!"]
            translations = ["Achei o caminho!", "Continue!", "Estamos quase lá!"]
            draw.rounded_rectangle((359, y + 14, 623, y + 59), radius=8, fill=WHITE, outline=INK, width=2)
            text(draw, (491, y + 36), words[index], 15, INK, True, "mm")
            if index < step + 1:
                draw.rounded_rectangle((634, y + 14, 871, y + 59), radius=8, fill="#DDF2EF", outline=TEAL, width=2)
                text(draw, (752, y + 36), translations[index], 14, TEAL, True, "mm")
    _dashed_rectangle(draw, viewport, CORAL, 3)
    draw.rounded_rectangle((48, 168, 190, 417), radius=8, fill=WHITE, outline=LINE, width=2)
    text(draw, (68, 190), "PROCESSO", 14, MUTED, True)
    labels = ["Frame", "Diff", "ROI", "OCR/cache", "Overlay"]
    for i, label in enumerate(labels):
        yy = 226 + i * 35
        text(draw, (68, yy), ("● " if i <= min(step + 1, 4) else "○ ") + label, 14,
             TEAL if i <= min(step + 1, 4) else MUTED, i == min(step + 1, 4))
    text(draw, (218, 478), f"Cena {step + 1} / 3   •   traduções preservadas enquanto a nova faixa é processada", 14, MUTED, True)
    return image


def save_gif(name: str, frames: list[Image.Image], duration: int) -> None:
    frames[0].save(MEDIA / name, save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True, disposal=2)


def main() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    pipeline_image()
    save_gif("reader-workflow.gif", [workflow_frame(i) for i in range(4)], 1100)
    scroll = [scroll_frame(i) for i in range(3)]
    save_gif("scroll-stream.gif", scroll + scroll[-1:], 1000)


if __name__ == "__main__":
    main()
