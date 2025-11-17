import argparse
import os
from PIL import ImageDraw

from pdf_handler import read_pdf_pages, convert_page_to_image, save_images_to_pdf
from image_processor import preprocess_image_for_ocr
from ocr import perform_ocr_on_image
from translate import translate_text
from utils import overlay_text

def process_and_translate_pdf(input_path, output_path, target_lang, font_path):
    """
    Orchestrates the PDF translation process.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at '{input_path}'")
        return

    if not os.path.exists(font_path):
        print(f"Error: Font file not found at '{font_path}'. Please provide a valid path.")
        return

    print(f"Processing PDF: {input_path}")
    modified_images = []
    
    pdf_pages = list(read_pdf_pages(input_path))
    total_pages = len(pdf_pages)

    for i, page in enumerate(pdf_pages):
        print(f"--- Processing Page {i+1}/{total_pages} ---")

        # 1. Convert page to image
        image = convert_page_to_image(page)

        # 2. Preprocess the image for better OCR accuracy
        print("Preprocessing image for OCR...")
        processed_image = preprocess_image_for_ocr(image)

        # 3. Perform OCR on the enhanced image
        print("Performing OCR on enhanced image...")
        # Note: We pass the processed (OpenCV) image to pytesseract
        ocr_data = perform_ocr_on_image(processed_image, lang='ara')

        # 3. Process and overlay text
        num_boxes = len(ocr_data['level'])
        for j in range(num_boxes):
            # Filter for actual words with a decent confidence score
            if int(ocr_data['conf'][j]) > 40:
                original_text = ocr_data['text'][j].strip()
                if not original_text:
                    continue

                # Translate
                print(f"Translating '{original_text}'...")
                translated = translate_text(original_text, target_language=target_lang)
                print(f" -> '{translated}'")

                # Get coordinates for overlay
                x, y, w, h = (ocr_data['left'][j], ocr_data['top'][j], ocr_data['width'][j], ocr_data['height'][j])

                # We need to draw on the original color image, not the processed one
                draw = ImageDraw.Draw(image)
                # Overlay the translated text
                overlay_text(draw, translated, (x, y), (w, h), font_path)

        modified_images.append(image)
        print(f"--- Page {i+1} processed. ---")

    # 4. Save the result
    if modified_images:
        print(f"\nSaving translated PDF to: {output_path}")
        save_images_to_pdf(modified_images, output_path)
        print("Translation complete!")
    else:
        print("No pages were processed. Output PDF not created.")

def main():
    parser = argparse.ArgumentParser(description="Translate Arabic text in a PDF and overlay the translation.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input Arabic PDF file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the output translated PDF file.")
    parser.add_argument("-l", "--lang", default="id", help="Target language for translation (e.g., 'id' for Indonesian, 'en' for English).")
    parser.add_argument("-f", "--font", default="arial.ttf", help="Path to the .ttf font file for overlaying text.")
    
    args = parser.parse_args()
    process_and_translate_pdf(args.input, args.output, args.lang, args.font)

if __name__ == "__main__":
    main()