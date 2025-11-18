import argparse
from core.pdf_handler import process_pdf

def main():
    """
    Command-line interface for the PDF translation tool.
    """
    parser = argparse.ArgumentParser(
        description="Extract, translate, and generate Arabic PDFs."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input Arabic PDF file."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the translated output PDF file."
    )
    parser.add_argument(
        "--lang",
        default="en",
        help="Target language code for translation (e.g., 'en' for English)."
    )
    args = parser.parse_args()

    print(f"Processing {args.input}...")
    process_pdf(args.input, args.output, args.lang)
    print(f"Translated PDF saved to {args.output}")

if __name__ == "__main__":
    main()
