"""
Tests for the tarjim utils module (overlay and drawing helpers).
"""

from PIL import Image, ImageDraw, ImageFont
from unittest.mock import MagicMock
import pytest

from src.core import utils


def test_find_system_font_returns_string_or_none():
    """find_system_font should return a string path or None."""
    result = utils.find_system_font()
    assert result is None or isinstance(result, str)


def test_load_font_returns_font():
    """load_font should return an ImageFont object."""
    font = utils.load_font(16)
    assert font is not None


def test_get_dynamic_font_scales_with_bbox():
    """get_dynamic_font should produce smaller font for smaller bbox."""
    small_bbox = (0, 0, 100, 20)
    large_bbox = (0, 0, 100, 60)

    font_small = utils.get_dynamic_font(small_bbox)
    font_large = utils.get_dynamic_font(large_bbox)

    # The large bbox font should have a larger or equal size
    assert font_large.size >= font_small.size


def test_draw_text_in_box_does_not_crash():
    """draw_text_in_box should draw text without errors."""
    img = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(img)
    font = utils.load_font(14)

    # Should not raise
    utils.draw_text_in_box(draw, "Hello world test text", (10, 10, 190, 50), font)


def test_draw_text_in_box_empty_text():
    """draw_text_in_box with empty text should be a no-op."""
    img = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(img)
    font = utils.load_font(14)

    # Should not raise
    utils.draw_text_in_box(draw, "", (10, 10, 190, 50), font)
    utils.draw_text_in_box(draw, "  ", (10, 10, 190, 50), font)


class FakeTextLine:
    def __init__(self, text, bbox):
        self.text = text
        self.bbox = bbox


class FakePagePrediction:
    def __init__(self, text_lines):
        self.text_lines = text_lines


def test_overlay_translations_on_image_replace_mode():
    """overlay_translations_on_image in 'replace' mode should return an image."""
    img = Image.new("RGB", (200, 100), "white")
    prediction = FakePagePrediction([
        FakeTextLine("مرحبا", (10, 10, 190, 40)),
    ])

    def mock_translate(text, from_code="ar", to_code="en"):
        return "Hello"

    result = utils.overlay_translations_on_image(
        image=img,
        page_prediction=prediction,
        translate_fn=mock_translate,
        mode="replace",
    )

    assert isinstance(result, Image.Image)
    assert result.size == img.size


def test_overlay_translations_on_image_clean_mode():
    """overlay_translations_on_image in 'clean' mode should return white background."""
    img = Image.new("RGB", (200, 100), "red")
    prediction = FakePagePrediction([
        FakeTextLine("test", (10, 10, 190, 40)),
    ])

    def mock_translate(text, from_code="ar", to_code="en"):
        return "translated"

    result = utils.overlay_translations_on_image(
        image=img,
        page_prediction=prediction,
        translate_fn=mock_translate,
        mode="clean",
    )

    assert isinstance(result, Image.Image)
    # In clean mode, the background should be white (not red)
    pixel = result.getpixel((0, 0))
    assert pixel == (255, 255, 255)
