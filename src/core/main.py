import argparse
import os
from PIL import ImageDraw

from pdf_handler import read_pdf_pages, convert_page_to_image, save_images_to_pdf
from image_processor import preprocess_image_for_ocr
from ocr import perform_ocr_on_image
from translate import translate_text
from utils import overlay_text
from ocr_vlm import perform_vlm_ocr, perform_vlm_ocr_on_boxes
from layout_detector import detect_text_boxes

def process_and_translate_pdf(input_path, output_path, target_lang, font_path, use_vlm: bool = False, detector: str = 'contour'):
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

        # 3. Perform OCR on the enhanced image (or use VLM if requested)
        if use_vlm:
            print("Detecting text boxes for VLM recognition (method=%s)..." % detector)
            boxes = detect_text_boxes(processed_image, method=detector)
            if not boxes:
                # fallback to whole-page recognition
                print("No boxes detected; falling back to whole-page VLM recognition...")
                recognized = perform_vlm_ocr(image)
                if recognized:
                    translated = translate_text(recognized, target_language=target_lang)
                    draw = ImageDraw.Draw(image)
                    box_w = max(100, image.width - 20)
                    box_h = min(400, image.height - 20)
                    overlay_text(draw, translated, (10, 10), (box_w, box_h), font_path)
            else:
                print(f"Detected {len(boxes)} boxes; running VLM per-box...")
                box_texts = perform_vlm_ocr_on_boxes(image, boxes)
                for (box, recognized_text) in box_texts:
                    if not recognized_text:
                        continue
                    x, y, w, h = box
                    # Translate recognized text
                    translated = translate_text(recognized_text, target_language=target_lang)
                    # Draw translated text in the same box
                    draw = ImageDraw.Draw(image)
                    overlay_text(draw, translated, (x, y), (w, h), font_path)
        else:
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
    parser.add_argument("--use-vlm", action="store_true", help="Use VLM (Qwen2VL) for OCR instead of Tesseract.")
    parser.add_argument("--detector", choices=["contour", "east"], default="contour", help="Text detector to use when --use-vlm is enabled (default: contour).")
    
    args = parser.parse_args()
    process_and_translate_pdf(args.input, args.output, args.lang, args.font, use_vlm=args.use_vlm, detector=args.detector)

if __name__ == "__main__":
    main()