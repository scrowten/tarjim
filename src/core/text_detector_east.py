import os
import cv2
import numpy as np
from typing import List, Tuple, Union
from PIL import Image
import requests

DEFAULT_EAST_URL = "https://github.com/argman/EAST/releases/download/v1.0/frozen_east_text_detection.pb"
DEFAULT_EAST_PATH = os.path.expanduser("~/.cache/tarjim/east/frozen_east_text_detection.pb")


def _ensure_east_model(model_path: str = None) -> str:
    """Ensure EAST model exists locally; download if missing. Returns path to model file."""
    model_path = model_path or DEFAULT_EAST_PATH
    model_dir = os.path.dirname(model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    if os.path.exists(model_path):
        return model_path

    # Try to download the default EAST model (small convenience helper)
    try:
        print(f"Downloading EAST model to {model_path} (this may take a moment)...")
        resp = requests.get(DEFAULT_EAST_URL, stream=True, timeout=60)
        resp.raise_for_status()
        with open(model_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("EAST model downloaded.")
        return model_path
    except Exception as e:
        # If download fails, raise a clear error instructing user how to obtain the model
        raise RuntimeError(
            f"Failed to download EAST model automatically: {e}\nPlease download `frozen_east_text_detection.pb` and place it at {model_path} or pass a model_path to the detector."
        )


def decode_predictions(scores, geometry, scoreThresh: float=0.5) -> List[Tuple[int,int,int,int]]:
    """Decode EAST output to bounding boxes (x,y,w,h).
    Implementation adapted from OpenCV EAST examples.
    """
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0,0,y]
        x0_data = geometry[0,0,y]
        x1_data = geometry[0,1,y]
        x2_data = geometry[0,2,y]
        x3_data = geometry[0,3,y]
        anglesData = geometry[0,4,y]

        for x in range(0, numCols):
            if scoresData[x] < scoreThresh:
                continue

            # compute the offset
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            endX = int(offsetX + (cos * x1_data[x]) + (sin * x2_data[x]))
            endY = int(offsetY - (sin * x1_data[x]) + (cos * x2_data[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, int(w), int(h)))
            confidences.append(float(scoresData[x]))

    # apply NMS
    boxes = []
    if len(rects) > 0:
        rects_np = np.array(rects)
        x = rects_np[:,0]
        y = rects_np[:,1]
        w = rects_np[:,2]
        h = rects_np[:,3]
        x2 = x + w
        y2 = y + h
        idxs = cv2.dnn.NMSBoxes(rects, confidences, scoreThresh, 0.4)
        if len(idxs) > 0:
            for i in idxs.flatten():
                boxes.append(rects[i])

    return boxes


def detect_text_boxes_east(image: Union[Image.Image, np.ndarray], model_path: str = None, score_threshold: float = 0.5) -> List[Tuple[int,int,int,int]]:
    """
    Detect text boxes using the EAST model. Returns list of (x,y,w,h).
    If `model_path` is None, attempts to auto-download the default EAST .pb into the cache.
    """
    model_path = _ensure_east_model(model_path)

    # Prepare image
    if isinstance(image, Image.Image):
        img = np.array(image)
        if img.ndim == 3:
            img = img[:, :, ::-1]
    else:
        img = image

    orig_h, orig_w = img.shape[:2]

    # EAST expects multiples of 32
    newW = (orig_w // 32) * 32
    newH = (orig_h // 32) * 32
    if newW == 0: newW = 32
    if newH == 0: newH = 32

    rW = orig_w / float(newW)
    rH = orig_h / float(newH)

    resized = cv2.resize(img, (newW, newH))
    blob = cv2.dnn.blobFromImage(resized, 1.0, (newW, newH), (123.68, 116.78, 103.94), swapRB=True, crop=False)

    net = cv2.dnn.readNet(model_path)
    net.setInput(blob)
    outputLayers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    (scores, geometry) = net.forward(outputLayers)

    boxes = decode_predictions(scores, geometry, score_threshold)

    # scale back to original image size
    scaled = []
    for (x, y, w, h) in boxes:
        sx = max(0, int(x * rW))
        sy = max(0, int(y * rH))
        sw = min(orig_w - sx, int(w * rW))
        sh = min(orig_h - sy, int(h * rH))
        scaled.append((sx, sy, sw, sh))

    # sort top-to-bottom left-to-right
    scaled = sorted(scaled, key=lambda b: (b[1], b[0]))
    return scaled
