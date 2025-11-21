import os
import argparse
import logging
import pymupdf
from PIL import Image, ImageDraw
import io
from typing import List, Generator

from .image_processor import preprocess_image_for_ocr
from .ocr import perform_ocr_on_image
from .translate import translate_text
from .utils import overlay_text

logger = logging.getLogger(__name__)


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
        logger.error("Error opening or reading PDF %s: %s", pdf_path, e)
        return

def convert_page_to_image(page: pymupdf.Page, dpi: int = 300) -> Image.Image:
    """Converts a PyMuPDF page object to a PIL Image.

    Uses a transformation matrix based on DPI to render at the requested
    resolution. Calling `page.get_pixmap` with a matrix keeps the page
    rendering logic correct for the PyMuPDF API and avoids passing
    unsupported keyword arguments like `dpi`.
    """
    # create a scaling matrix: PDF units are 72 dpi baseline
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=matrix)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data)).convert("RGBA")
    return image

def save_images_to_pdf(images: List[Image.Image], output_path: str, resolution: float = 300.0):
    """Saves a list of PIL Images to a single PDF file."""
    if not images:
        logger.warning("No images to save to PDF.")
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
        logger.error("Input file not found at '%s'", input_path)
        return

    font_path = _find_system_font()
    if not os.path.exists(font_path):
        raise FileNotFoundError(
            f"Font file not found at '{font_path}'. "
            "Please install a common font like DejaVu Sans or Arial, or specify a font path."
        )

    logger.info("Processing PDF: %s", input_path)
    modified_images = []
    
    # Open the document once and iterate pages while the document is open.
    try:
        doc = pymupdf.open(input_path)
    except Exception as e:
        logger.error("Error opening PDF %s: %s", input_path, e)
        return

    total_pages = doc.page_count

    for i in range(total_pages):
        logger.info("Processing page %d/%d", i+1, total_pages)
        page = doc.load_page(i)

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

    # Close document handle
    try:
        doc.close()
    except Exception:
        logger.debug("Exception while closing document", exc_info=True)

    # 5. Save the result
    # Ensure output path ends with .pdf
    if not output_path.lower().endswith('.pdf'):
        output_path = output_path + '.pdf'

    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    save_images_to_pdf(modified_images, output_path)
    logger.info("Successfully saved translated PDF to %s", output_path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Translate PDF pages (OCR + translate + overlay)")
    parser.add_argument("input", help="Path to input PDF file")
    parser.add_argument("output", help="Path to output PDF file or base path ('.pdf' will be appended if missing)")
    parser.add_argument("--target-lang", default="en", help="Target language code for translation (default: en)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging (DEBUG)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.debug("Starting process with args: %s", args)

    process_pdf(args.input, args.output, target_lang=args.target_lang)


if __name__ == "__main__":
    raise SystemExit(main())
