import pytesseract
from PIL import Image
from typing import Dict, Any, Union
import numpy as np

def perform_ocr_on_image(image: Union[Image.Image, np.ndarray], lang: str = 'ara') -> Dict[str, Any]:
    """
    Performs OCR on a PIL image to get detailed data including text and bounding boxes.

    Args:
        image (Union[Image.Image, np.ndarray]): The image to process (PIL or OpenCV format).
        lang (str): The language for Tesseract to use. Defaults to 'ara' for Arabic.
    """
    # Use image_to_data to get bounding box info for each word
    return pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
