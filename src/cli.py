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
from .core.translator_argos import get_translation_route


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
(Any Argos Translate language code is accepted.)

Translation routing:
  - Direct: used when a direct ar→<lang> package exists (e.g., ar→en)
  - Pivot:  ar→en→<lang> when no direct package exists (e.g., ar→en→id)
  The route is chosen automatically. Use --verbose to see which route is used.

Examples:
  # Arabic to English (direct)
  python -m src.cli -i kitab.pdf -o kitab_en.pdf --lang en

  # Arabic to Indonesian (auto-pivot via English)
  python -m src.cli -i kitab.pdf -o kitab_id.pdf --lang id

  # Clean mode (white background, no original text visible)
  python -m src.cli -i kitab.pdf -o kitab_en.pdf --lang en --overlay-mode clean

  # High quality with verbose logging
  python -m src.cli -i kitab.pdf -o kitab_id.pdf --lang id --dpi 400 --verbose
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

    print(f"Tarjim: Translating {args.input}")
    print(f"  Source: {args.source_lang} | Target: {args.lang} ({lang_name})")
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
    )
    print(f"\nTranslated PDF saved to {args.output}")


if __name__ == "__main__":
    main()
