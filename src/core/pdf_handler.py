import pymupdf
from PIL import Image
import io
from typing import List, Generator

def read_pdf_pages(pdf_path: str) -> Generator[pymupdf.Page, None, None]:
    """
    Opens a PDF and yields its pages one by one.
    Uses a generator to be memory-efficient for large PDFs.
    """
    try:
        doc = pymupdf.open(pdf_path)
        yield from doc
        doc.close()
    except Exception as e:
        print(f"Error opening or reading PDF {pdf_path}: {e}")
        return

def convert_page_to_image(page: fitz.Page, dpi: int = 300) -> Image.Image:
    """Converts a PyMuPDF page object to a PIL Image."""
    pix = page.get_pixmap(dpi=dpi)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    return image

def save_images_to_pdf(images: List[Image.Image], output_path: str, resolution: float = 300.0):
    """Saves a list of PIL Images to a single PDF file."""
    if not images:
        print("Warning: No images to save to PDF.")
        return

    images[0].save(output_path, save_all=True, append_images=images[1:], resolution=resolution)
