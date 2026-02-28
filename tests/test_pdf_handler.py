"""
Tests for the tarjim PDF handler module.

Tests cover:
- PDF page to image conversion (with matrix/DPI)
- Full process_pdf pipeline (with mocked OCR/translation)
- Output path normalization (.pdf suffix, parent dir creation)
"""

import io
import os
from unittest.mock import MagicMock
from PIL import Image
import pytest

from src.core import pdf_handler


class FakeMatrix:
    def __init__(self, a, b):
        self.a = a
        self.b = b


class FakePixmap:
    def __init__(self, img_bytes):
        self._b = img_bytes

    def tobytes(self, fmt):
        return self._b


class FakePage:
    def __init__(self):
        self.called = {}

    def get_pixmap(self, matrix=None):
        self.called['matrix'] = matrix
        img = Image.new('RGBA', (10, 10), (255, 0, 0, 255))
        b = io.BytesIO()
        img.save(b, format='PNG')
        return FakePixmap(b.getvalue())


# ===========================
# Test: convert_page_to_image
# ===========================

def test_convert_page_to_image_uses_matrix(monkeypatch):
    """Verify that convert_page_to_image correctly applies DPI-based matrix."""
    fake_pymupdf = type('M', (), {'Matrix': FakeMatrix})
    monkeypatch.setattr(pdf_handler, 'pymupdf', fake_pymupdf)

    page = FakePage()
    img = pdf_handler.convert_page_to_image(page, dpi=144)

    assert isinstance(img, Image.Image)
    assert img.size == (10, 10)
    assert 'matrix' in page.called
    assert isinstance(page.called['matrix'], FakeMatrix)
    # zoom = dpi / 72 -> 144/72 = 2.0
    assert page.called['matrix'].a == pytest.approx(2.0)


# ===========================
# Test: process_pdf (full pipeline, mocked)
# ===========================

class FakeTextLine:
    """Mimics a Surya OCR text_line object."""
    def __init__(self, text, bbox):
        self.text = text
        self.bbox = bbox


class FakePagePrediction:
    """Mimics a Surya OCR page prediction."""
    def __init__(self, text_lines):
        self.text_lines = text_lines


def test_process_pdf_appends_pdf_and_creates_parent_dir(tmp_path, monkeypatch):
    """Test that process_pdf:
    - Appends .pdf to output path if missing
    - Creates parent directory if it doesn't exist
    - Processes the correct number of pages
    """
    input_pdf = tmp_path / 'input.pdf'
    input_pdf.write_bytes(b'%PDF-1.4\n%EOF')
    output_base = tmp_path / 'outdir' / 'outfile'

    # Mock pdf_to_images to return a simple image
    def fake_pdf_to_images(path, dpi=300):
        return [Image.new('RGB', (100, 100), 'white')]

    monkeypatch.setattr(pdf_handler, 'pdf_to_images', fake_pdf_to_images)

    # Mock Surya OCR initialization and execution
    mock_rec = MagicMock()
    mock_det = MagicMock()
    monkeypatch.setattr(pdf_handler, 'init_surya_ocr', lambda: (mock_rec, mock_det))

    # Mock run_ocr_on_page to return a fake prediction with one text line
    fake_prediction = FakePagePrediction([
        FakeTextLine("مرحبا", (10, 10, 90, 30)),
    ])
    monkeypatch.setattr(pdf_handler, 'run_ocr_on_page', lambda img, rec, det: fake_prediction)

    # Mock Argos translation setup and translate
    monkeypatch.setattr(pdf_handler, 'setup_argos_translation', lambda **kw: None)
    monkeypatch.setattr(pdf_handler, 'translate_text', lambda t, from_code='ar', to_code='en': 'hello')

    # Mock overlay to return the image unchanged
    monkeypatch.setattr(
        pdf_handler,
        'overlay_translations_on_image',
        lambda **kw: kw.get('image', Image.new('RGB', (100, 100), 'white')),
    )

    # Capture save call
    captured = {}

    def fake_save(images, out_path, resolution=300.0):
        captured['out_path'] = out_path
        captured['num_images'] = len(images)

    monkeypatch.setattr(pdf_handler, 'save_images_to_pdf', fake_save)

    # Run the pipeline
    pdf_handler.process_pdf(str(input_pdf), str(output_base), target_lang='en')

    assert 'out_path' in captured
    assert captured['out_path'].lower().endswith('.pdf')
    parent = os.path.dirname(captured['out_path'])
    assert os.path.exists(parent)
    assert captured['num_images'] == 1


def test_process_pdf_missing_input(tmp_path, monkeypatch, caplog):
    """Test that process_pdf handles missing input file gracefully."""
    fake_input = str(tmp_path / 'nonexistent.pdf')
    output = str(tmp_path / 'output.pdf')

    pdf_handler.process_pdf(fake_input, output)

    # Should log an error, not crash
    assert any("not found" in r.message.lower() for r in caplog.records)


# ===========================
# Test: save_images_to_pdf
# ===========================

def test_save_images_to_pdf(tmp_path):
    """Test that save_images_to_pdf creates a valid file."""
    images = [
        Image.new('RGB', (100, 100), 'red'),
        Image.new('RGB', (100, 100), 'blue'),
    ]
    output_path = str(tmp_path / 'test_output.pdf')

    pdf_handler.save_images_to_pdf(images, output_path)

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_save_images_to_pdf_empty(tmp_path):
    """Test that save_images_to_pdf handles empty list gracefully."""
    output_path = str(tmp_path / 'empty_output.pdf')
    pdf_handler.save_images_to_pdf([], output_path)
    assert not os.path.exists(output_path)
