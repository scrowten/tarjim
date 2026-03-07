"""
NLLB-200 translator using CTranslate2 INT8 quantized model.

Uses facebook/nllb-200-distilled-1.3B via CTranslate2 INT8 for efficient,
fully-offline translation across 200 languages.

Key advantages over Argos Translate:
- ~3-5× better translation quality (especially on classical Arabic)
- Direct Arabic → Indonesian without English pivot
- Single model for all language pairs

Model downloads on first use:
- OpenNMT/nllb-200-distilled-1.3B-ct2-int8  (~2.6 GB)
- facebook/nllb-200-distilled-1.3B tokenizer (~600 MB)

VRAM: ~2.8 GB in int8_float16 mode (fits RTX 3050 6GB alongside Surya OCR).
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level cached state (lazy loaded on first translate call)
_TRANSLATOR = None
_TOKENIZER = None

CT2_MODEL_ID = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"
TOKENIZER_MODEL_ID = "facebook/nllb-200-distilled-1.3B"

# ISO 639-1/639-3 → NLLB BCP-47 language code mapping
# Full list: https://github.com/facebookresearch/flores/blob/main/flores200/README.md
NLLB_LANG_CODES = {
    "ar": "ara_Arab",   # Arabic (Modern Standard / Classical)
    "en": "eng_Latn",   # English
    "id": "ind_Latn",   # Indonesian
    "ms": "zsm_Latn",   # Malay (Standard)
    "fr": "fra_Latn",   # French
    "de": "deu_Latn",   # German
    "es": "spa_Latn",   # Spanish
    "tr": "tur_Latn",   # Turkish
    "ur": "urd_Arab",   # Urdu
    "fa": "pes_Arab",   # Persian (Farsi)
    "hi": "hin_Deva",   # Hindi
    "zh": "zho_Hans",   # Chinese (Simplified)
    "ru": "rus_Cyrl",   # Russian
    "ja": "jpn_Jpan",   # Japanese
    "ko": "kor_Hang",   # Korean
    "pt": "por_Latn",   # Portuguese
    "it": "ita_Latn",   # Italian
    "nl": "nld_Latn",   # Dutch
    "bn": "ben_Beng",   # Bengali
    "sw": "swh_Latn",   # Swahili
}


def _get_nllb_code(iso_code: str) -> str:
    """Convert ISO 639-1 code to NLLB BCP-47 code."""
    if iso_code in NLLB_LANG_CODES:
        return NLLB_LANG_CODES[iso_code]
    # Pass through if already looks like an NLLB code (e.g. "ara_Arab")
    if "_" in iso_code and len(iso_code) > 4:
        return iso_code
    raise ValueError(
        f"Unknown language code: '{iso_code}'. "
        f"Supported: {sorted(NLLB_LANG_CODES.keys())}"
    )


def init_nllb_translator() -> Tuple:
    """
    Initialize and return (translator, tokenizer) for NLLB-200.

    Uses lazy loading — models are downloaded on first call (~3.2 GB total),
    then cached locally by HuggingFace Hub. Subsequent runs are fully offline.

    Returns:
        Tuple of (ctranslate2.Translator, AutoTokenizer)
    """
    global _TRANSLATOR, _TOKENIZER

    if _TRANSLATOR is not None and _TOKENIZER is not None:
        return _TRANSLATOR, _TOKENIZER

    import ctranslate2
    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download

    # CTranslate2 ships its own CUDA runtime — use its own device check,
    # not torch.cuda.is_available() which may be False on CPU-only PyTorch builds.
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    compute_type = "int8_float16" if device == "cuda" else "int8"

    logger.info(
        "Downloading/loading NLLB-200 CT2-INT8 model "
        "(first run: ~3.2 GB download, ~2.8 GB VRAM)..."
    )
    ct2_model_path = snapshot_download(CT2_MODEL_ID)

    logger.info(
        "Loading NLLB-200 translator (device=%s, compute_type=%s)...",
        device, compute_type,
    )
    _TRANSLATOR = ctranslate2.Translator(
        ct2_model_path,
        device=device,
        compute_type=compute_type,
        inter_threads=1,
    )

    logger.info("Loading NLLB-200 tokenizer...")
    _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_ID)

    logger.info("NLLB-200 ready — direct translation for 200 language pairs.")
    return _TRANSLATOR, _TOKENIZER


def translate_text(
    text: str,
    from_code: str = "ar",
    to_code: str = "en",
    beam_size: int = 4,
    max_length: int = 512,
) -> str:
    """
    Translate text using NLLB-200 distilled 1.3B (CTranslate2 INT8).

    All language pairs are supported directly — no English pivot needed.
    Significantly better quality than Argos Translate, especially for
    classical Arabic (kitab, hadith, fiqh) text.

    Args:
        text: Text to translate.
        from_code: Source language ISO code (e.g., 'ar'). Default: 'ar'.
        to_code: Target language ISO code (e.g., 'en', 'id'). Default: 'en'.
        beam_size: Beam search width. Default: 4 (quality/speed balance).
        max_length: Maximum output tokens. Default: 512.

    Returns:
        Translated text string. Returns original text on failure.
    """
    if not text or not text.strip():
        return ""

    try:
        src_nllb = _get_nllb_code(from_code)
        tgt_nllb = _get_nllb_code(to_code)
    except ValueError as e:
        logger.warning("NLLB language code error: %s — returning original.", e)
        return text

    translator, tokenizer = init_nllb_translator()

    try:
        tokenizer.src_lang = src_nllb
        encoded = tokenizer(text, add_special_tokens=True)
        source_tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])

        results = translator.translate_batch(
            [source_tokens],
            target_prefix=[[tgt_nllb]],
            beam_size=beam_size,
            max_decoding_length=max_length,
            max_batch_size=1,
        )

        output_tokens = results[0].hypotheses[0][1:]  # strip leading lang token
        output_ids = tokenizer.convert_tokens_to_ids(output_tokens)
        return tokenizer.decode(output_ids, skip_special_tokens=True)

    except Exception as exc:
        logger.warning("NLLB translation failed for '%.40s...': %s", text, exc)
        return text


def get_supported_languages() -> list:
    """Return supported languages as list of dicts with code and name."""
    names = {
        "ar": "Arabic", "en": "English", "id": "Indonesian",
        "ms": "Malay", "fr": "French", "de": "German",
        "es": "Spanish", "tr": "Turkish", "ur": "Urdu",
        "fa": "Persian", "hi": "Hindi", "zh": "Chinese (Simplified)",
        "ru": "Russian", "ja": "Japanese", "ko": "Korean",
        "pt": "Portuguese", "it": "Italian", "nl": "Dutch",
        "bn": "Bengali", "sw": "Swahili",
    }
    return [
        {"code": iso, "name": names.get(iso, iso), "nllb_code": nllb}
        for iso, nllb in NLLB_LANG_CODES.items()
    ]


def get_translation_route(from_code: str, to_code: str) -> str:
    """NLLB supports all pairs directly. Returns 'direct' or 'unavailable'."""
    if from_code == to_code:
        return "direct"
    try:
        _get_nllb_code(from_code)
        _get_nllb_code(to_code)
        return "direct"
    except ValueError:
        return "unavailable"
