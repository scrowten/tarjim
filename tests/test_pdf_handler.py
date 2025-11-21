import io
import os
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
        # record the matrix used
        self.called['matrix'] = matrix
        # produce a tiny PNG
        img = Image.new('RGBA', (10, 10), (255, 0, 0, 255))
        b = io.BytesIO()
        img.save(b, format='PNG')
        return FakePixmap(b.getvalue())


def test_convert_page_to_image_uses_matrix(monkeypatch):
    # Replace pymupdf.Matrix used inside the module with our FakeMatrix
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


def test_process_pdf_appends_pdf_and_creates_parent_dir(tmp_path, monkeypatch):
    # prepare a minimal input file
    input_pdf = tmp_path / 'input.pdf'
    input_pdf.write_bytes(b'%PDF-1.4\n%EOF')

    # choose an output base path without .pdf and inside a non-existing directory
    output_base = tmp_path / 'outdir' / 'outfile'

    # fake document returned by pymupdf.open
    class FakeDoc:
        page_count = 1

        def load_page(self, i):
            return FakePage()

        def close(self):
            pass

    fake_pymupdf = type('M', (), {'open': lambda p: FakeDoc(), 'Matrix': FakeMatrix})
    monkeypatch.setattr(pdf_handler, 'pymupdf', fake_pymupdf)

    # stub out image preprocessing / OCR / translation / overlay to keep test focused
    monkeypatch.setattr(pdf_handler, 'preprocess_image_for_ocr', lambda img: img)

    def fake_perform(img, lang='ara'):
        return {
            'level': [1],
            'conf': ['90'],
            'text': ['hello'],
            'left': [1],
            'top': [1],
            'width': [10],
            'height': [10],
        }

    monkeypatch.setattr(pdf_handler, 'perform_ocr_on_image', fake_perform)
    monkeypatch.setattr(pdf_handler, 'translate_text', lambda t, target_language=None: 'translated')
    monkeypatch.setattr(pdf_handler, 'overlay_text', lambda draw, txt, xy, wh, font: None)

    captured = {}

    def fake_save(images, out_path, resolution=300.0):
        captured['out_path'] = out_path
        captured['num_images'] = len(images)

    monkeypatch.setattr(pdf_handler, 'save_images_to_pdf', fake_save)

    # run
    pdf_handler.process_pdf(str(input_pdf), str(output_base), target_lang='en')

    assert 'out_path' in captured
    assert captured['out_path'].lower().endswith('.pdf')
    parent = os.path.dirname(captured['out_path'])
    assert os.path.exists(parent)
    assert captured['num_images'] == 1
