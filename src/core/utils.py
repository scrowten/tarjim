"""
Overlay and drawing utilities for the tarjim PDF translation pipeline.

Provides text overlay, dynamic font sizing, word-wrapping, and
high-level image overlay functions for placing translated text
on PDF page images.

Tashkeel (Arabic diacritization) support:
    When tashkeel_fn is passed to overlay_translations_on_image(), each Arabic
    text line is diacritized before translation for improved accuracy.
    When show_tashkeel=True, the diacritized Arabic text is rendered in the
    top portion of each bounding box (in blue, RTL), with the translation
    rendered below it.
    When tashkeel_only=True, no translation is performed at all — the
    diacritized Arabic text is rendered in the full bounding box (in black,
    RTL). Use this to produce a fully-vowelized Arabic PDF without translation.

    Arabic RTL rendering uses arabic_reshaper + python-bidi to correctly
    shape and order Arabic characters for PIL rendering.
"""

import os
import logging
from typing import Callable, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)


# ===========================
# Font helpers
# ===========================

# Common font paths across platforms (for translated / Latin text)
_FONT_SEARCH_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      # Linux
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Linux alt
    "/System/Library/Fonts/Supplemental/Arial.ttf",          # macOS
    "C:/Windows/Fonts/arial.ttf",                            # Windows
]

# Arabic-capable font paths (supports harakat / combining diacritical marks)
# Amiri is ideal for classical Arabic (kitab). Download from:
#   https://github.com/aliftype/amiri/releases → amiri-regular.ttf
#   Place at: fonts/amiri-regular.ttf
#
# NOTE: fonts/times.ttf (bundled Latin font) is intentionally NOT listed here —
# Latin fonts have no Arabic glyph coverage and will render Arabic as invisible boxes.
_ARABIC_FONT_SEARCH_PATHS = [
    "fonts/amiri-regular.ttf",                               # Best: classical Arabic + harakat
    "/usr/share/fonts/truetype/amiri/Amiri-Regular.ttf",     # Linux (apt: fonts-amiri)
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",  # Linux (apt: fonts-noto-core)
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",   # Linux alt
    # macOS — Arabic-capable system fonts
    "/System/Library/Fonts/Supplemental/GeezaPro.ttc",       # macOS Geeza Pro (Arabic)
    "/Library/Fonts/Arial Unicode MS.ttf",                   # macOS (MS Office)
    # Windows — Arabic-capable system fonts (NOT times.ttf or arial.ttf — Latin only!)
    "C:/Windows/Fonts/arabtype.ttf",                         # Arabic Typesetting (Windows)
    "C:/Windows/Fonts/segoeui.ttf",                          # Segoe UI — good harakat support
    "C:/Windows/Fonts/tahoma.ttf",                           # Tahoma — solid Arabic support
    "C:/Windows/Fonts/calibri.ttf",                          # Calibri — Arabic support
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


def find_arabic_font() -> Optional[str]:
    """
    Find an Arabic-capable font file for RTL text rendering with harakat.

    Checks bundled and system font paths in priority order. Amiri Regular
    is the recommended choice for classical Arabic (kitab) text.

    IMPORTANT: Latin fonts (times.ttf, arial.ttf) are intentionally excluded —
    they have no Arabic glyph coverage and would render Arabic text as invisible
    empty boxes on the white background.

    To get the best results, download Amiri Regular and place it at
    fonts/amiri-regular.ttf (relative to the project root).

    On Windows without Amiri, the tool falls back to Segoe UI / Tahoma / Calibri /
    Arabic Typesetting (all support Arabic characters). Harakat display quality
    varies — Segoe UI gives the best harakat coverage among system fonts.

    Returns:
        Path to an Arabic-capable TTF/TTC font file, or None if none found.
        When None is returned, tashkeel text is rendered with the default
        system font (harakat likely invisible — download Amiri for best results).
    """
    # Resolve relative paths against the project root (two levels up from this file)
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    for path in _ARABIC_FONT_SEARCH_PATHS:
        # Try as absolute path first, then relative to project root
        if os.path.exists(path):
            return path
        abs_path = os.path.join(_project_root, path)
        if os.path.exists(abs_path):
            return abs_path

    logger.warning(
        "No Arabic-capable font found. Diacritized Arabic text will not be "
        "shown in the PDF. For best results, download Amiri Regular and place "
        "it at fonts/amiri-regular.ttf. "
        "See: https://github.com/aliftype/amiri/releases"
    )
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
# Arabic RTL text helpers
# ===========================

def _reshape_arabic(text: str) -> str:
    """
    Apply Arabic text shaping and bidi reordering for correct PIL rendering.

    PIL draws characters left-to-right without Arabic shaping. This helper:
      1. Reshapes Arabic characters so they connect properly (arabic_reshaper),
         with delete_harakat=False to preserve tashkeel (vowel marks).
      2. Applies the Unicode bidi algorithm to produce the visual display order
         (python-bidi), converting RTL logical order to LTR rendering order.

    Falls back to the original text if the libraries are not installed.

    Args:
        text: Logical-order Arabic string (as stored in Unicode).

    Returns:
        Visually-ordered, shaped string ready for PIL.ImageDraw.text().
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        # delete_harakat defaults to True in arabic_reshaper — must explicitly
        # disable it or all tashkeel (vowel marks) are stripped before rendering.
        reshaper = arabic_reshaper.ArabicReshaper(configuration={
            "delete_harakat": False,
        })
        reshaped = reshaper.reshape(text)
        return get_display(reshaped)
    except ImportError:
        logger.debug(
            "arabic_reshaper / python-bidi not installed. "
            "Install them for correct Arabic rendering: "
            "pip install arabic-reshaper python-bidi"
        )
        return text


def draw_arabic_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    bbox: Tuple[float, float, float, float],
    font: ImageFont.FreeTypeFont,
    fill: str = "darkblue",
) -> None:
    """
    Draw Arabic text right-aligned within a bounding box using RTL shaping.

    Applies arabic_reshaper + python-bidi for correct visual rendering in PIL,
    then right-aligns each word-wrapped line within the box.

    Args:
        draw: PIL ImageDraw object.
        text: Arabic text to render (diacritized / with harakat).
        bbox: Bounding box as (x1, y1, x2, y2).
        font: PIL font object (should be an Arabic-capable TTF font).
        fill: Text color (default: 'darkblue' to visually distinguish from translation).
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    max_width = x2 - x1
    max_height = y2 - y1

    if max_width <= 0 or max_height <= 0 or not text.strip():
        return

    # Shape and reorder for RTL rendering
    display_text = _reshape_arabic(text)

    # Arabic fonts (especially Amiri) have ascent+descent ≈ 1.75× the nominal font
    # size due to tall harakat marks.  Shrink the font until one line fits vertically.
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + 2
    if line_height > max_height:
        try:
            ratio = line_height / max(font.size, 1)        # e.g. 1.76 for Amiri
            new_size = max(6, int(max_height / ratio))
            font = load_font(new_size, font.path)
            ascent, descent = font.getmetrics()
            line_height = ascent + descent + 2
        except Exception:
            pass  # keep original font; first line may clip slightly

    # Word-wrap: split shaped text into lines that fit within max_width
    words = display_text.split()
    lines = []
    current = ""

    for w in words:
        candidate = (current + " " + w).strip()
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    if not lines:
        return

    # Draw lines right-aligned within the bbox.
    # Always render the first line even if it slightly clips the bottom —
    # skipping it entirely is worse than a minor visual overflow.
    y = y1
    for i, line in enumerate(lines):
        if i > 0 and y + line_height > y2:
            break  # subsequent lines that overflow are skipped
        line_width = draw.textlength(line, font=font)
        x_pos = x2 - int(line_width)  # right-align
        draw.text((x_pos, y), line, font=font, fill=fill)
        y += line_height


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
    translate_fn: Optional[Callable],
    from_code: str = "ar",
    to_code: str = "en",
    font_path: Optional[str] = None,
    mode: str = "replace",
    tashkeel_fn: Optional[Callable[[str], str]] = None,
    show_tashkeel: bool = False,
    tashkeel_only: bool = False,
) -> Image.Image:
    """
    Overlay translated (or diacritized) text onto a page image using Surya OCR predictions.

    When tashkeel_fn is provided, each Arabic text line is diacritized before
    translation, improving translation accuracy for undiacritized kitab text.

    When show_tashkeel=True (requires tashkeel_fn), the diacritized Arabic text
    is rendered in the top 40% of each bounding box (dark blue, RTL), with the
    translation in the lower 60%. A light separator line divides the two.

    When tashkeel_only=True (requires tashkeel_fn), no translation is performed.
    The diacritized Arabic text is rendered in the full bounding box (black, RTL).
    translate_fn is not called and may be None.

    Args:
        image: Original page image (PIL Image).
        page_prediction: Surya OCR page prediction with .text_lines attribute.
            Each text_line has .text (str) and .bbox (x1, y1, x2, y2).
        translate_fn: Callable(text, from_code, to_code) -> translated_text.
            May be None when tashkeel_only=True.
        from_code: Source language code (default: 'ar').
        to_code: Target language code (default: 'en').
        font_path: Optional path to a .ttf font file for translation text.
        mode: Overlay mode:
            - 'replace': White-box over original, draw translated text (default).
            - 'clean': White background with only translated text.
        tashkeel_fn: Optional callable(text: str) -> diacritized_text.
            When provided, Arabic text is diacritized before translation.
        show_tashkeel: If True, render the diacritized Arabic text above the
            translation within each bounding box. Requires tashkeel_fn.
        tashkeel_only: If True, skip translation entirely. Render the diacritized
            Arabic text in the full bounding box. Requires tashkeel_fn.
            translate_fn is unused when this is True.

    Returns:
        New PIL Image with translated (and optionally diacritized) text overlaid.
    """
    if font_path is None:
        font_path = find_system_font()

    # Resolve Arabic font once for the whole page (needed for show_tashkeel or tashkeel_only)
    arabic_font_path: Optional[str] = None
    if tashkeel_fn is not None and (show_tashkeel or tashkeel_only):
        arabic_font_path = find_arabic_font()
        if arabic_font_path is None:
            logger.warning(
                "%s but no Arabic font found. "
                "Tashkeel text will be rendered with a fallback font "
                "(harakat may not display correctly). "
                "Download Amiri Regular to fonts/amiri-regular.ttf for best results.",
                "tashkeel_only=True" if tashkeel_only else "show_tashkeel=True",
            )

    if mode == "clean":
        # Start with a clean white image
        result_image = Image.new("RGB", image.size, "white")
    else:
        # Work on a copy of the original image
        result_image = image.copy()
        if result_image.mode == "RGBA":
            result_image = result_image.convert("RGB")

    draw = ImageDraw.Draw(result_image)

    # Import tashkeel helper lazily (only when tashkeel is active)
    _is_arabic_text = None
    if tashkeel_fn is not None:
        try:
            from .tashkeel import is_arabic_text as _is_arabic_text
        except ImportError:
            pass

    text_lines = page_prediction.text_lines
    for line in text_lines:
        src_text = line.text
        bbox = line.bbox  # (x1, y1, x2, y2)

        if not src_text or not src_text.strip():
            continue

        # --- Step 1: Tashkeel (diacritize) the Arabic text ---
        diacritized = src_text
        if tashkeel_fn is not None:
            is_arabic = (_is_arabic_text(src_text) if _is_arabic_text else True)
            if is_arabic:
                try:
                    diacritized = tashkeel_fn(src_text)
                except Exception as exc:
                    logger.warning("Tashkeel skipped for line: %s", exc)

        # --- Step 2: Cover original text box with white ---
        if mode == "replace":
            x1, y1, x2, y2 = [int(v) for v in bbox]
            draw.rectangle([x1, y1, x2, y2], fill="white", outline="white")

        # --- Step 3: Tashkeel-only mode — render diacritized Arabic, skip translation ---
        if tashkeel_only:
            effective_font_path = arabic_font_path or font_path
            arabic_font = get_dynamic_font(bbox, effective_font_path)
            draw_arabic_text_in_box(draw, diacritized, bbox, arabic_font, fill="black")
            continue

        # --- Step 4: Translate (use diacritized text for better accuracy) ---
        try:
            translated = translate_fn(diacritized, from_code=from_code, to_code=to_code)
        except Exception as e:
            logger.warning("Translation failed for line '%s...': %s", src_text[:30], e)
            # Try pivoting through English if target isn't English
            if to_code != "en":
                try:
                    en_text = translate_fn(diacritized, from_code=from_code, to_code="en")
                    translated = translate_fn(en_text, from_code="en", to_code=to_code)
                except Exception:
                    translated = src_text  # Keep original as last resort
            else:
                translated = src_text

        # --- Step 5: Render text (translation ± diacritized Arabic header) ---
        has_diacritized = diacritized != src_text
        should_show_arabic = (
            show_tashkeel
            and tashkeel_fn is not None
            and has_diacritized
            and arabic_font_path is not None
        )

        if should_show_arabic:
            # Split bbox: top 40% → diacritized Arabic, bottom 60% → translation
            x1, y1, x2, y2 = [int(v) for v in bbox]
            height = y2 - y1
            split_y = y1 + int(height * 0.40)

            arabic_bbox = (x1, y1, x2, split_y)
            trans_bbox = (x1, split_y, x2, y2)

            # Draw a thin separator line between Arabic and translation
            draw.line([(x1, split_y), (x2, split_y)], fill="lightgray", width=1)

            # Render diacritized Arabic (RTL, dark blue) in top portion
            arabic_font = get_dynamic_font(arabic_bbox, arabic_font_path, scale=0.75)
            draw_arabic_text_in_box(draw, diacritized, arabic_bbox, arabic_font, fill="darkblue")

            # Render translation in bottom portion
            trans_font = get_dynamic_font(trans_bbox, font_path, scale=0.75)
            draw_text_in_box(draw, translated, trans_bbox, trans_font, fill="black")
        else:
            # Standard rendering: translation fills the whole bbox
            font = get_dynamic_font(bbox, font_path)
            draw_text_in_box(draw, translated, bbox, font, fill="black")

    return result_image
