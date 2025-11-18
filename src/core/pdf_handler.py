import os
import pymupdf
from PIL import Image, ImageDraw
import io
from typing import List, Generator

from .image_processor import preprocess_image_for_ocr
from .ocr import perform_ocr_on_image
from .translate import translate_text
from .utils import overlay_text

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

def convert_page_to_image(page: pymupdf.Page, dpi: int = 300) -> Image.Image:
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

def _find_system_font() -> str:
    """
    Finds a suitable default font on the system for text overlay.
    """
    # For Linux
    if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    # For Windows
    if os.path.exists("C:/Windows/Fonts/arial.ttf"):
        return "C:/Windows/Fonts/arial.ttf"
    # For macOS
    if os.path.exists("/System/Library/Fonts/Supplemental/Arial.ttf"):
        return "/System/Library/Fonts/Supplemental/Arial.ttf"
    # Fallback if no common font is found
    # Note: This will likely cause an error if no font is found.
    # Consider bundling a font with your application for maximum portability.
    return "arial.ttf"

def process_pdf(input_path: str, output_path: str, target_lang: str = "en"):
    """
    Orchestrates the full PDF translation pipeline.

    1. Reads a PDF and converts each page to an image.
    2. Preprocesses the image for better OCR.
    3. Performs OCR to extract Arabic text blocks.
    4. Translates each text block.
    5. Overlays the translated text onto the original image.
    6. Saves the modified images as a new PDF.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    font_path = _find_system_font()
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"Font file not found at '{font_path}'. "
            "Please install a common font like DejaVu Sans or Arial, or specify a font path."
        )

    print(f"Processing PDF: {input_path}")
    modified_images = []
    
    pdf_pages = list(read_pdf_pages(input_path))
    total_pages = len(pdf_pages)

    for i, page in enumerate(pdf_pages):
        print(f"--- Processing Page {i+1}/{total_pages} ---")

        # 1. Convert page to image
        image = convert_page_to_image(page)

        # 2. Preprocess for better OCR
        processed_image = preprocess_image_for_ocr(image)

        # 3. Perform OCR
        ocr_data = perform_ocr_on_image(processed_image, lang='ara')

        # 4. Translate and overlay text
        draw = ImageDraw.Draw(image)
        for j in range(len(ocr_data['level'])):
            # Filter for actual words with a decent confidence score
            if int(ocr_data['conf'][j]) > 40:
                original_text = ocr_data['text'][j].strip()
                if not original_text:
                    continue

                translated = translate_text(original_text, target_language=target_lang)
                x, y, w, h = (ocr_data['left'][j], ocr_data['top'][j], ocr_data['width'][j], ocr_data['height'][j])
                overlay_text(draw, translated, (x, y), (w, h), font_path)

        modified_images.append(image)

    # 5. Save the result
    save_images_to_pdf(modified_images, output_path)
    print(f"Successfully saved translated PDF to {output_path}")
