import cv2
import numpy as np
from typing import List, Tuple, Union
from PIL import Image
from text_detector_east import detect_text_boxes_east


def _nms_boxes(boxes: List[Tuple[int, int, int, int]], overlapThresh: float = 0.3) -> List[Tuple[int, int, int, int]]:
    """Simple non-max suppression for boxes (x, y, w, h).
    Returns filtered boxes.
    """
    if not boxes:
        return []
    # convert to (x1,y1,x2,y2)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])
    x1 = rects[:,0]
    y1 = rects[:,1]
    x2 = rects[:,2]
    y2 = rects[:,3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    pick = []
    while len(idxs) > 0:
        last = idxs[-1]
        i = last
        pick.append(i)
        suppress = [len(idxs)-1]
        for pos in range(len(idxs)-1):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)
            overlap = (w * h) / areas[j]
            if overlap > overlapThresh:
                suppress.append(pos)
        idxs = np.delete(idxs, suppress)

    picked = rects[pick].astype(int).tolist()
    # convert back to x,y,w,h
    return [(int(x1), int(y1), int(x2 - x1), int(y2 - y1)) for (x1, y1, x2, y2) in picked]


def detect_text_boxes(image: Union[Image.Image, np.ndarray], min_area: int = 100, dilation_iter: int = 1, method: str = 'contour', **kwargs) -> List[Tuple[int, int, int, int]]:
    """
    Detect text regions using contour-based heuristic. Returns list of (x,y,w,h).

    Args:
        image: PIL Image or OpenCV numpy array. Prefer a preprocessed grayscale or binary image for best results.
    """
    # If the user requested EAST detector, call that implementation
    if method and method.lower() == 'east':
        model_path = kwargs.get('model_path')
        score_threshold = kwargs.get('score_threshold', 0.5)
        return detect_text_boxes_east(image, model_path=model_path, score_threshold=score_threshold)

    # Fallback to contour-based detection (original implementation)
    # Convert PIL to numpy BGR if needed
    if isinstance(image, Image.Image):
        img = np.array(image)
        # if RGBA or RGB, convert to BGR
        if img.ndim == 3:
            img = img[:, :, ::-1]
    else:
        img = image

    # If color, convert to gray
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Adaptive threshold to get binary
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 9)

    # Morphology to connect text regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    dilated = cv2.dilate(thr, kernel, iterations=dilation_iter)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < min_area:
            continue
        # Filter rectangular shapes that are too tall or too narrow
        if h < 8 or w < 8:
            continue
        boxes.append((x, y, w, h))

    # Apply a light NMS to merge overlapping boxes
    boxes = _nms_boxes(boxes, overlapThresh=0.35)

    # Sort boxes top-to-bottom, left-to-right
    boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
    return boxes
