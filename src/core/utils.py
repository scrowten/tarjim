"""
Overlay and drawing utilities for the tarjim PDF translation pipeline.

Provides text overlay, dynamic font sizing, word-wrapping, and
high-level image overlay functions for placing translated text
on PDF page images.
"""

import os
import logging
from typing import Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ===========================
# Font helpers
# ===========================

# Common font paths across platforms
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux alt
    "/System/Library/Fonts/Supplemental/Arial.ttf",          # macOS
    "C:/Windows/Fonts/arial.ttf",                            # Windows
]


def find_system_font() -> Optional[str]:
    """
    Find a suitable system font for English text overlay.

    Returns:
        Path to a TTF font file, or None if no common font is found.
    """
    for path in _FONT_SEARCH_PATHS:
        if os.path.exists(path):
            return path
    return None


def load_font(font_size: int = 16, font_path: Optional[str] = None) -> ImageFont.FreeTypeFont:
    """
    Load a TTF font at the given size.

    Args:
        font_size: Desired font size in points/pixels.
        font_path: Explicit path to a .ttf file. If None, searches system paths.

    Returns:
        A PIL FreeTypeFont object.
    """
    if font_path and os.path.exists(font_path):
        return ImageFont.truetype(font_path, font_size)

    system_font = find_system_font()
    if system_font:
        return ImageFont.truetype(system_font, font_size)

    # Fallback to PIL default
    return ImageFont.load_default()


def get_dynamic_font(
    bbox: Tuple[float, float, float, float],
    font_path: Optional[str] = None,
    scale: float = 0.8,
) -> ImageFont.FreeTypeFont:
    """
    Determine an appropriate font size based on the bounding box height.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2).
        font_path: Optional path to a .ttf font file.
        scale: Scale factor for font size relative to bbox height (default: 0.8).

    Returns:
        A PIL FreeTypeFont object sized to fit the bbox.
    """
    _x1, y1, _x2, y2 = bbox
    bbox_height = y2 - y1

    # Font size is often close to pixel height of the line
    font_size = max(1, int(bbox_height * scale))

    return load_font(font_size, font_path)


# ===========================
# Text drawing helpers
# ===========================

def draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: Tuple[float, float, float, float],
    font: ImageFont.FreeTypeFont,
    fill: str = "black",
) -> None:
    """
    Draw text with word-wrapping inside a bounding box.

    Args:
        draw: PIL ImageDraw object.
        text: Text to draw.
        bbox: Bounding box as (x1, y1, x2, y2).
        font: PIL font object.
        fill: Text color (default: 'black').
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    max_width = x2 - x1
    max_height = y2 - y1

    if max_width <= 0 or max_height <= 0 or not text.strip():
        return

    # Word-wrap text to fit within bbox width
    words = text.split()
    lines = []
    current = ""

    for w in words:
        tmp = (current + " " + w).strip()
        if draw.textlength(tmp, font=font) <= max_width:
            current = tmp
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    # Draw lines, respecting vertical bounds
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + 2

    y = y1
    for line in lines:
        if y + line_height > y2:
            break  # don't overflow the box
        draw.text((x1, y), line, font=font, fill=fill)
        y += line_height


def overlay_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    pos: Tuple[int, int],
    box: Tuple[int, int],
    font_path: str,
) -> None:
    """
    Legacy overlay function: draws a white box to cover original text,
    then overlays translated text with dynamic font sizing.

    Args:
        draw: PIL ImageDraw object.
        text: Translated text to overlay.
        pos: (x, y) position of the text box.
        box: (width, height) of the original text's bounding box.
        font_path: Path to the .ttf font file.
    """
    x, y = pos
    w, h = box

    # Cover original text with white rectangle
    draw.rectangle([x, y, x + w, y + h], fill='white', outline='white')

    if not text:
        return

    # Dynamically adjust font size to fit the box width
    font_size = h
    font = ImageFont.truetype(font_path, font_size)
    while font.getbbox(text)[2] > w and font_size > 8:
        font_size -= 1
        font = ImageFont.truetype(font_path, font_size)

    draw.text((x, y), text, font=font, fill='black')


# ===========================
# High-level overlay for Surya OCR output
# ===========================

def overlay_translations_on_image(
    image: Image.Image,
    page_prediction,
    translate_fn,
    from_code: str = "ar",
    to_code: str = "en",
    font_path: Optional[str] = None,
    mode: str = "replace",
) -> Image.Image:
    """
    Overlay translated text onto a page image using Surya OCR predictions.

    Args:
        image: Original page image (PIL Image).
        page_prediction: Surya OCR page prediction with .text_lines attribute.
            Each text_line has .text (str) and .bbox (x1, y1, x2, y2).
        translate_fn: Callable(text, from_code, to_code) -> translated_text.
        from_code: Source language code (default: 'ar').
        to_code: Target language code (default: 'en').
        font_path: Optional path to a .ttf font file.
        mode: Overlay mode:
            - 'replace': White-box over original, draw translated text (default).
            - 'clean': White background with only translated text.

    Returns:
        New PIL Image with translated text overlaid.
    """
    if font_path is None:
        font_path = find_system_font()

    if mode == "clean":
        # Start with a clean white image
        result_image = Image.new("RGB", image.size, "white")
    else:
        # Work on a copy of the original image
        result_image = image.copy()
        if result_image.mode == "RGBA":
            result_image = result_image.convert("RGB")

    draw = ImageDraw.Draw(result_image)

    text_lines = page_prediction.text_lines
    for line in text_lines:
        src_text = line.text
        bbox = line.bbox  # (x1, y1, x2, y2)

        if not src_text or not src_text.strip():
            continue

        # Translate the text
        try:
            translated = translate_fn(src_text, from_code=from_code, to_code=to_code)
        except Exception as e:
            logger.warning("Translation failed for line '%s...': %s", src_text[:30], e)
            # Try pivoting through English if target isn't English
            if to_code != "en":
                try:
                    en_text = translate_fn(src_text, from_code=from_code, to_code="en")
                    translated = translate_fn(en_text, from_code="en", to_code=to_code)
                except Exception:
                    translated = src_text  # Keep original as last resort
            else:
                translated = src_text

        # In replace mode, cover original text with white box first
        if mode == "replace":
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], fill="white", outline="white")

        # Get a font sized to fit the bounding box
        font = get_dynamic_font(bbox, font_path)

        # Draw translated text with word-wrapping
        draw_text_in_box(draw, translated, bbox, font, fill="black")

    return result_image
