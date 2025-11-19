"""Proof-of-concept runner to test VLM + layout detector on a single image.

Usage: python -m src.core.poc_vlm_layout /path/to/image.jpg
"""
import sys
from PIL import Image
from .layout_detector import detect_text_boxes
from .ocr_vlm import perform_vlm_ocr_on_boxes


def main(image_path: str):
    img = Image.open(image_path).convert("RGB")
    boxes = detect_text_boxes(img)
    print(f"Detected {len(boxes)} boxes")
    for b in boxes:
        print(b)

    results = perform_vlm_ocr_on_boxes(img, boxes)
    for box, text in results:
        print(f"Box {box}: {text}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.core.poc_vlm_layout /path/to/image.jpg")
    else:
        main(sys.argv[1])
