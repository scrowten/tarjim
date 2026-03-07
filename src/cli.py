"""
Command-line interface for the Tarjim PDF translation tool.

Usage:
    python -m src.cli --input input.pdf --output translated.pdf --lang en
    python -m src.cli --input kitab.pdf --output kitab_id.pdf --lang id
"""

import argparse
import logging
import sys

from .core.pdf_handler import process_pdf


# Common language presets for user convenience
LANGUAGE_PRESETS = {
    "en": "English",
    "id": "Indonesian (Bahasa Indonesia)",
    "ms": "Malay",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "tr": "Turkish",
    "ur": "Urdu",
}


def main():
    """
    CLI entry point for PDF translation.
    """
    lang_help_lines = ", ".join(
        f"'{code}' ({name})" for code, name in LANGUAGE_PRESETS.items()
    )

    parser = argparse.ArgumentParser(
        description="Tarjim: Translate Arabic text in PDF documents (fully offline).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported target languages: {lang_help_lines}

Translation backends:
  --translator nllb   (default) NLLB-200 1.3B via CTranslate2 INT8.
                      Direct translation for all 200 languages — no pivot.
                      ~3.2 GB download on first use, ~2.8 GB VRAM.
  --translator argos  Argos Translate — small, lower quality, no GPU needed.

Examples:
  # Arabic to English (NLLB, default)
  python -m src.cli -i kitab.pdf -o kitab_en.pdf --lang en

  # Arabic to Indonesian (NLLB, direct — no English pivot)
  python -m src.cli -i kitab.pdf -o kitab_id.pdf --lang id

  # Use Argos Translate instead (lightweight fallback)
  python -m src.cli -i kitab.pdf -o kitab_en.pdf --lang en --translator argos

  # Best quality: NLLB + tashkeel (diacritize before translate)
  python -m src.cli -i kitab.pdf -o kitab_en.pdf --lang en --tashkeel
        """,
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the input Arabic PDF file.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to save the translated output PDF file.",
    )
    parser.add_argument(
        "--lang", "-l",
        default="en",
        help=(
            "Target language code (default: 'en'). "
            "Common codes: 'en' (English), 'id' (Indonesian), 'ms' (Malay), "
            "'fr' (French), 'de' (German), 'es' (Spanish), 'tr' (Turkish)."
        ),
    )
    parser.add_argument(
        "--source-lang",
        default="ar",
        help="Source language code (default: 'ar' for Arabic).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Rendering DPI for PDF pages (default: 300).",
    )
    parser.add_argument(
        "--overlay-mode",
        choices=["replace", "clean"],
        default="replace",
        help="Overlay mode: 'replace' covers original text, 'clean' uses white background (default: replace).",
    )
    parser.add_argument(
        "--font",
        default=None,
        help="Path to a .ttf font file for text rendering.",
    )
    parser.add_argument(
        "--tashkeel",
        action="store_true",
        help=(
            "Enable Arabic diacritization (tashkeel) before translation using CATT. "
            "Adds harakat to undiacritized kitab text, improving translation accuracy. "
            "Model weights (~300–600 MB) are downloaded on first use."
        ),
    )
    parser.add_argument(
        "--show-tashkeel",
        action="store_true",
        help=(
            "Show diacritized Arabic text in the output PDF, stacked above the translation "
            "within each text bounding box. Implies --tashkeel. "
            "Requires an Arabic font (see docs for Amiri font setup)."
        ),
    )
    parser.add_argument(
        "--translator",
        choices=["nllb", "argos"],
        default="nllb",
        help=(
            "Translation backend: 'nllb' (default, higher quality, ~3.2 GB download) "
            "or 'argos' (lightweight fallback, no GPU needed)."
        ),
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG).",
    )
    args = parser.parse_args()

    # show_tashkeel implies tashkeel
    if args.show_tashkeel and not args.tashkeel:
        import warnings
        warnings.warn(
            "--show-tashkeel requires --tashkeel. Enabling tashkeel automatically.",
            UserWarning,
            stacklevel=1,
        )
        args.tashkeel = True

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Display run info
    lang_name = LANGUAGE_PRESETS.get(args.lang, args.lang)
    if args.tashkeel and args.show_tashkeel:
        tashkeel_status = "enabled (show in PDF)"
    elif args.tashkeel:
        tashkeel_status = "enabled (silent — improves translation quality)"
    else:
        tashkeel_status = "disabled"

    translator_info = {
        "nllb": "NLLB-200 1.3B (CTranslate2 INT8) — high quality, direct",
        "argos": "Argos Translate — lightweight fallback",
    }.get(args.translator, args.translator)

    print(f"Tarjim: Translating {args.input}")
    print(f"  Source: {args.source_lang} | Target: {args.lang} ({lang_name})")
    print(f"  Translator: {translator_info}")
    print(f"  Overlay: {args.overlay_mode} | DPI: {args.dpi}")
    print(f"  Tashkeel: {tashkeel_status}")
    print()

    process_pdf(
        input_path=args.input,
        output_path=args.output,
        target_lang=args.lang,
        source_lang=args.source_lang,
        dpi=args.dpi,
        overlay_mode=args.overlay_mode,
        font_path=args.font,
        tashkeel=args.tashkeel,
        show_tashkeel=args.show_tashkeel,
        translator=args.translator,
    )
    print(f"\nTranslated PDF saved to {args.output}")


if __name__ == "__main__":
    main()
