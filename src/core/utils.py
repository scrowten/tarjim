from PIL import ImageDraw, ImageFont
import os

def overlay_text(draw: ImageDraw.ImageDraw, text: str, pos: tuple, box: tuple, font_path: str):
    """
    Draws a white box to cover original text and overlays the new text,
    dynamically adjusting font size to fit.

    Args:
        draw (ImageDraw.ImageDraw): The PIL ImageDraw object.
        text (str): The translated text to overlay.
        pos (tuple): The (x, y) position to start drawing the text.
        box (tuple): The (width, height) of the original text's bounding box.
        font_path (str): Path to the .ttf font file.
    """
    x, y = pos
    w, h = box

    # 1. Cover original text with a white rectangle
    draw.rectangle([x, y, x + w, y + h], fill='white', outline='white')

    if not text:
        return

    # 2. Dynamically adjust font size to fit the box width
    font_size = h
    font = ImageFont.truetype(font_path, font_size)
    # Reduce font size until text fits within the original width
    while font.getbbox(text)[2] > w and font_size > 8:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    # 3. Draw the translated text
    draw.text((x, y), text, font=font, fill='black')
