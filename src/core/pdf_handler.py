"""
PDF handler for the tarjim translation pipeline.

Orchestrates the full pipeline:
  PDF → images (PyMuPDF) → OCR (Surya) → translate (Argos) → overlay → save PDF

Keeps PDF I/O utilities (read, convert, save) and the main process_pdf() orchestrator.
"""

import os
import argparse
import logging
import io
from typing import List, Generator, Optional

import pymupdf
from PIL import Image
from tqdm import tqdm

from .ocr_surya import init_surya_ocr, run_ocr_on_page
from .utils import overlay_translations_on_image, find_system_font

logger = logging.getLogger(__name__)


# ===========================
# PDF I/O utilities
# ===========================

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
    resolution.
    """
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    pix = page.get_pixmap(matrix=matrix)
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data)).convert("RGBA")
    return image


def pdf_to_images(pdf_path: str, dpi: int = 300) -> List[Image.Image]:
    """
    Convert all pages of a PDF to PIL Images.

    Args:
        pdf_path: Path to the input PDF file.
        dpi: Rendering resolution (default: 300).

    Returns:
        List of PIL Image objects, one per page.
    """
    doc = pymupdf.open(pdf_path)
    zoom = dpi / 72.0
    matrix = pymupdf.Matrix(zoom, zoom)

    images: List[Image.Image] = []
    for page_index in tqdm(range(len(doc)), desc="Rendering PDF pages"):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        images.append(img)

    doc.close()
    return images


def save_images_to_pdf(
    images: List[Image.Image],
    output_path: str,
    resolution: float = 300.0,
):
    """Saves a list of PIL Images to a single PDF file."""
    if not images:
        logger.warning("No images to save to PDF.")
        return

    # Convert all to RGB (PDF requirement)
    images_rgb = [img.convert("RGB") for img in images]

    first, *rest = images_rgb
    first.save(
        output_path,
        "PDF",
        resolution=resolution,
        save_all=True,
        append_images=rest,
    )


# ===========================
# Main pipeline orchestrator
# ===========================

def process_pdf(
    input_path: str,
    output_path: str,
    target_lang: str = "en",
    source_lang: str = "ar",
    dpi: int = 300,
    overlay_mode: str = "replace",
    font_path: Optional[str] = None,
    tashkeel: bool = False,
    show_tashkeel: bool = False,
    translator: str = "nllb",
    tashkeel_only: bool = False,
):
    """
    Orchestrate the full PDF translation pipeline.

    Pipeline steps:
        1. (Unless tashkeel_only) Set up translation models (NLLB or Argos).
        2. Convert PDF pages to images (PyMuPDF).
        3. Initialize Surya OCR models.
        4. (Optional) Initialize CATT tashkeel model if tashkeel=True or tashkeel_only=True.
        5. For each page: OCR → [tashkeel] → [translate] → overlay text.
        6. Save all modified images as a new PDF.

    Args:
        input_path: Path to the input Arabic PDF file.
        output_path: Path to save the translated output PDF.
        target_lang: Target language code (default: 'en' for English).
        source_lang: Source language code (default: 'ar' for Arabic).
        dpi: Rendering DPI for PDF → image conversion (default: 300).
        overlay_mode: How to overlay translations:
            - 'replace': White-box over original text, draw translation (default).
            - 'clean': White background with only translated text.
        font_path: Optional path to a .ttf font file for text rendering.
        tashkeel: If True, diacritize Arabic text before translation using CATT.
            Improves translation accuracy for undiacritized kitab text.
        show_tashkeel: If True (requires tashkeel=True), render the diacritized
            Arabic text in the output PDF above the translation.
        translator: Translation backend to use:
            - 'nllb' (default): NLLB-200 1.3B via CTranslate2 INT8. Better quality,
              direct ar→id, ~2.8 GB VRAM. Downloads ~3.2 GB on first use.
            - 'argos': Argos Translate (small, fast, lower quality, fully offline).
        tashkeel_only: If True, diacritize Arabic text and overlay it back onto the
            PDF with no translation at all. Implies tashkeel=True. The translator
            is not initialized. Useful for producing fully-vowelized Arabic PDFs.
    """
    if not os.path.exists(input_path):
        logger.error("Input file not found at '%s'", input_path)
        return

    # tashkeel_only implies tashkeel
    if tashkeel_only:
        tashkeel = True

    # Ensure output path ends with .pdf
    if not output_path.lower().endswith('.pdf'):
        output_path = output_path + '.pdf'

    # Ensure parent directory exists
    parent_dir = os.path.dirname(output_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Resolve font
    if font_path is None:
        font_path = find_system_font()

    logger.info("Processing PDF: %s", input_path)
    logger.info("Overlay mode: %s", overlay_mode)

    if tashkeel_only:
        logger.info("Mode: Tashkeel-only — diacritize Arabic and overlay (no translation)")
    else:
        logger.info("Target language: %s | Translator: %s", target_lang, translator)
        if tashkeel:
            logger.info(
                "Tashkeel: enabled (CATT EncoderDecoder) — diacritizing Arabic before translation"
            )
            if show_tashkeel:
                logger.info("Tashkeel: show mode — diacritized Arabic will appear in output PDF")
        else:
            logger.info("Tashkeel: disabled")

    # Step 1: Set up translation backend (skipped in tashkeel_only mode)
    translate_text = None
    if not tashkeel_only:
        if translator == "nllb":
            from .translator_nllb import translate_text, init_nllb_translator
            logger.info(
                "Initializing NLLB-200 translator (%s → %s, direct)...", source_lang, target_lang
            )
            init_nllb_translator()
            logger.info("Translation route: %s → %s (NLLB direct)", source_lang, target_lang)
        else:
            from .translator_argos import translate_text, setup_argos_translation
            logger.info("Setting up Argos translation (%s → %s)...", source_lang, target_lang)
            route = setup_argos_translation(from_code=source_lang, to_code=target_lang)
            if route == "pivot:en":
                logger.info(
                    "Translation route: %s → en → %s (Argos pivot through English)",
                    source_lang, target_lang,
                )
            else:
                logger.info(
                    "Translation route: %s → %s (Argos direct)", source_lang, target_lang
                )

    # Step 2: Convert PDF to images
    logger.info("Rendering PDF to images (DPI=%d)...", dpi)
    page_images = pdf_to_images(input_path, dpi=dpi)
    total_pages = len(page_images)
    logger.info("Rendered %d pages.", total_pages)

    # Step 3: Initialize Surya OCR
    logger.info("Initializing Surya OCR...")
    recognition_predictor, detection_predictor = init_surya_ocr()

    # Step 4 (optional): Initialize CATT tashkeel model
    tashkeel_fn = None
    if tashkeel:
        logger.info("Initializing CATT tashkeel model...")
        from .tashkeel import init_tashkeel_model, restore_harakat
        _tashkeel_model = init_tashkeel_model()
        tashkeel_fn = lambda text: restore_harakat(text, model=_tashkeel_model)

    # Step 5: OCR + [tashkeel] + [translate] + overlay for each page
    tqdm_label = "OCR + tashkeel pages" if tashkeel_only else "OCR + translate pages"
    translated_images: List[Image.Image] = []

    for idx, img in enumerate(tqdm(page_images, desc=tqdm_label)):
        logger.info("Processing page %d/%d", idx + 1, total_pages)

        # Run Surya OCR
        page_prediction = run_ocr_on_page(
            img, recognition_predictor, detection_predictor
        )

        # Log detected lines
        num_lines = len(page_prediction.text_lines) if page_prediction.text_lines else 0
        logger.info("Page %d: detected %d text lines", idx + 1, num_lines)

        # Overlay diacritized / translated text on the image
        translated_img = overlay_translations_on_image(
            image=img,
            page_prediction=page_prediction,
            translate_fn=translate_text,
            from_code=source_lang,
            to_code=target_lang,
            font_path=font_path,
            mode=overlay_mode,
            tashkeel_fn=tashkeel_fn,
            show_tashkeel=show_tashkeel,
            tashkeel_only=tashkeel_only,
        )

        translated_images.append(translated_img)

    # Step 6: Save the result
    if translated_images:
        logger.info("Saving output PDF to: %s", output_path)
        save_images_to_pdf(translated_images, output_path, resolution=float(dpi))
        logger.info(
            "%s complete! Saved %d pages.",
            "Tashkeel overlay" if tashkeel_only else "Translation",
            len(translated_images),
        )
    else:
        logger.warning("No pages were processed. Output PDF not created.")


# ===========================
# CLI entry point
# ===========================

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Tarjim: Translate Arabic PDF documents using OCR + translation + overlay."
    )
    parser.add_argument("input", help="Path to input PDF file")
    parser.add_argument("output", help="Path to output PDF file")
    parser.add_argument(
        "--target-lang", default="en",
        help="Target language code for translation (default: en)",
    )
    parser.add_argument(
        "--source-lang", default="ar",
        help="Source language code (default: ar)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Rendering DPI (default: 300)",
    )
    parser.add_argument(
        "--overlay-mode",
        choices=["replace", "clean"],
        default="replace",
        help="Overlay mode: 'replace' covers original text, 'clean' uses white background (default: replace)",
    )
    parser.add_argument(
        "--font", default=None,
        help="Path to a .ttf font file for text rendering",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable verbose logging (DEBUG)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.debug("Starting process with args: %s", args)

    process_pdf(
        input_path=args.input,
        output_path=args.output,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        dpi=args.dpi,
        overlay_mode=args.overlay_mode,
        font_path=args.font,
    )


if __name__ == "__main__":
    raise SystemExit(main())
