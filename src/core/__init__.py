"""
Tarjim core module.

Provides the main PDF translation pipeline using:
- Surya OCR for Arabic text recognition
- Argos Translate for offline translation
- PIL-based text overlay on PDF page images

Imports are lazy to allow importing individual submodules
even when not all dependencies are installed.
"""


def __getattr__(name):
    """Lazy imports for heavy dependencies."""
    if name == "process_pdf":
        from .pdf_handler import process_pdf
        return process_pdf
    if name == "init_surya_ocr":
        from .ocr_surya import init_surya_ocr
        return init_surya_ocr
    if name == "run_ocr_on_page":
        from .ocr_surya import run_ocr_on_page
        return run_ocr_on_page
    if name == "setup_argos_translation":
        from .translator_argos import setup_argos_translation
        return setup_argos_translation
    if name == "translate_text":
        from .translator_argos import translate_text
        return translate_text
    raise AttributeError(f"module 'src.core' has no attribute {name!r}")


__all__ = [
    "process_pdf",
    "init_surya_ocr",
    "run_ocr_on_page",
    "setup_argos_translation",
    "translate_text",
]
