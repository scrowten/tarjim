"""
Tashkeel (Arabic diacritization) module for the tarjim pipeline.

Uses CATT (Character-based Arabic Tashkeel Transformer) to add harakat
(short vowel diacritical marks) to unvocalized Arabic text.

CATT was trained on the Tashkeela corpus — 75M words from 97 Shamela Digital
Library classical Islamic books (98.85% classical Arabic) — making it ideal
for kitab (Islamic scholarly texts) that lack harakat.

Two model variants:
  - CATTEncoderDecoder: better accuracy (default, recommended for kitab)
  - CATTEncoderOnly:    faster inference (use when speed is more important)

Usage:
    from .tashkeel import init_tashkeel_model, restore_harakat

    model = init_tashkeel_model()
    diacritized = restore_harakat("كتب الله", model=model)
    # → "كَتَبَ اللَّهُ"
"""

import logging
import unicodedata

logger = logging.getLogger(__name__)

# Module-level cached model — loaded once, reused across calls
_TASHKEEL_MODEL = None


def is_arabic_text(text: str) -> bool:
    """
    Check whether text contains Arabic characters.

    Used to skip tashkeel on non-Arabic lines (e.g. Latin page numbers,
    headers, or footnotes already in the target language).

    Args:
        text: Input string to check.

    Returns:
        True if any character in the string falls in the Arabic Unicode block.
    """
    return any("\u0600" <= ch <= "\u06FF" or "\u0750" <= ch <= "\u077F" for ch in text)


def init_tashkeel_model(use_encoder_decoder: bool = True):
    """
    Initialize and return the CATT tashkeel model.

    Uses lazy loading — the model is loaded once and cached for subsequent calls.
    The model weights are downloaded automatically on first use (~300–600 MB).

    Args:
        use_encoder_decoder: If True (default), loads CATTEncoderDecoder for
            higher accuracy. If False, loads CATTEncoderOnly for faster inference.

    Returns:
        Loaded CATT model instance.

    Raises:
        ImportError: If catt-tashkeel is not installed.
        RuntimeError: If the model fails to load.
    """
    global _TASHKEEL_MODEL

    if _TASHKEEL_MODEL is not None:
        return _TASHKEEL_MODEL

    try:
        from catt_tashkeel import CATTEncoderDecoder, CATTEncoderOnly
    except ImportError as exc:
        raise ImportError(
            "catt-tashkeel is not installed. "
            "Run: pip install catt-tashkeel"
        ) from exc

    model_name = "CATTEncoderDecoder" if use_encoder_decoder else "CATTEncoderOnly"
    logger.info("Initializing CATT tashkeel model (%s)...", model_name)
    logger.info(
        "Model weights will be downloaded on first use (~300–600 MB). "
        "Subsequent runs are fully offline."
    )

    try:
        if use_encoder_decoder:
            _TASHKEEL_MODEL = CATTEncoderDecoder()
        else:
            _TASHKEEL_MODEL = CATTEncoderOnly()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load CATT tashkeel model ({model_name}): {exc}"
        ) from exc

    logger.info("CATT tashkeel model loaded successfully.")
    return _TASHKEEL_MODEL


def restore_harakat(text: str, model=None) -> str:
    """
    Add harakat (diacritical marks) to Arabic text using CATT.

    Diacritizes the input text to restore short vowels (fatha, kasra, damma,
    sukun, shadda, tanwin) that are typically absent from Arabic kitab text.
    The diacritized output improves downstream translation accuracy and makes
    the Arabic text readable for pronunciation.

    Non-Arabic text (Latin, numerals, punctuation-only lines) is returned
    unchanged without calling the model.

    Args:
        text: Unvocalized Arabic text (as extracted by OCR).
        model: Pre-loaded CATT model instance. If None, initializes a new one
            via init_tashkeel_model() (uses the module-level cached model).

    Returns:
        Diacritized Arabic text string.
        Returns the original text unchanged if:
          - Input is empty or whitespace-only
          - Input contains no Arabic characters
          - The model call raises an exception (with a warning logged)
    """
    if not text or not text.strip():
        return text

    # Skip tashkeel for non-Arabic content (page numbers, Latin headers, etc.)
    if not is_arabic_text(text):
        return text

    if model is None:
        model = init_tashkeel_model()

    try:
        results = model.do_tashkeel_batch([text], verbose=False)
        diacritized = results[0]
        return diacritized
    except Exception as exc:
        logger.warning(
            "Tashkeel failed for text '%s...': %s. Using original text.",
            text[:40],
            exc,
        )
        return text
