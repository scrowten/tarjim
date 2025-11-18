import cv2
import numpy as np
from PIL import Image

def preprocess_image_for_ocr(image: Image.Image) -> np.ndarray:
    """
    Enhances a PIL image for better OCR results using a multi-step OpenCV pipeline.

    Args:
        image (Image.Image): The input PIL image.

    Returns:
        np.ndarray: The processed image as an OpenCV-compatible NumPy array.
    """
    # 1. Convert PIL Image to OpenCV format (NumPy array)
    open_cv_image = np.array(image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # 2. Convert to Grayscale
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # # 3. Deskew the image
    # # Threshold to get a binary image, find contours, and determine the angle
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # coords = np.column_stack(np.where(thresh > 0))
    # angle = cv2.minAreaRect(coords)[-1]

    # # The `cv2.minAreaRect` angle can be in [-90, 0). We need to correct it.
    # if angle < -45:
    #     angle = -(90 + angle)
    # else:
    #     angle = -angle

    # # Rotate the image to deskew it
    # (h, w) = gray.shape[:2]
    # center = (w // 2, h // 2)
    # M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # print(f"Deskewing image by {angle:.2f} degrees.")
    rotated = gray

    # 4. Denoise the image
    # Using a median blur is effective against salt-and-pepper noise
    denoised = cv2.medianBlur(rotated, 3)

    # 5. Apply adaptive thresholding
    # This is often better than global thresholding for varying background illumination
    processed_image = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    return processed_image