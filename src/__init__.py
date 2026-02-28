"""
Tarjim source package.

Provides Arabic PDF OCR, translation, and overlay functionality.
"""


def __getattr__(name):
    """Lazy import for process_pdf."""
    if name == "process_pdf":
        from .core.pdf_handler import process_pdf
        return process_pdf
    raise AttributeError(f"module 'src' has no attribute {name!r}")


__all__ = ["process_pdf"]
