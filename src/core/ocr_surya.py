"""
Surya OCR module for Arabic text recognition.

Uses the Surya OCR library for high-quality Arabic text detection and recognition.
This is the primary OCR engine for the tarjim pipeline.
"""

import logging
from typing import Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

# Lazy-loaded predictors (heavy models, loaded once)
_RECOGNITION_PREDICTOR = None
_DETECTION_PREDICTOR = None


def _patch_surya_transformers_compat() -> None:
    """
    Compatibility patch for surya-ocr 0.17.x with transformers 5.x.

    In transformers 5.x, `pad_token_id` was removed from `PretrainedConfig`
    (moved to GenerationConfig). However, transformers' own `modeling_utils.py`
    still accesses `self.config.pad_token_id` directly (line 4363), which raises
    `AttributeError: 'SuryaDecoderConfig' object has no attribute 'pad_token_id'`
    because SuryaDecoderConfig extends PretrainedConfig and never defined it.

    Fix: add `pad_token_id = None` to the SuryaDecoderConfig class. This tells
    transformers "this model has no pad token", causing the padding check to
    return early cleanly — which is the correct behavior for surya.

    Tracked upstream: surya-ocr issue with transformers>=5.0.0
    """
    try:
        from surya.common.surya.decoder.config import SuryaDecoderConfig
        if not hasattr(SuryaDecoderConfig, "pad_token_id"):
            SuryaDecoderConfig.pad_token_id = None
            logger.debug(
                "Applied surya-ocr/transformers-5.x compatibility patch "
                "(added pad_token_id=None to SuryaDecoderConfig)"
            )
    except Exception:
        pass  # Safe to ignore — patch is best-effort


def init_surya_ocr():
    """
    Initialize and return Surya OCR predictors.

    Uses lazy loading — models are loaded once and cached for subsequent calls.

    Returns:
        Tuple of (RecognitionPredictor, DetectionPredictor)
    """
    global _RECOGNITION_PREDICTOR, _DETECTION_PREDICTOR

    if _RECOGNITION_PREDICTOR is not None and _DETECTION_PREDICTOR is not None:
        return _RECOGNITION_PREDICTOR, _DETECTION_PREDICTOR

    # Apply compatibility patch before importing surya models
    _patch_surya_transformers_compat()

    logger.info("Initializing Surya OCR models (this may take a moment on first run)...")

    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    foundation_predictor = FoundationPredictor()
    _RECOGNITION_PREDICTOR = RecognitionPredictor(foundation_predictor)
    _DETECTION_PREDICTOR = DetectionPredictor()

    logger.info("Surya OCR models loaded successfully.")
    return _RECOGNITION_PREDICTOR, _DETECTION_PREDICTOR


def run_ocr_on_page(
    image: Image.Image,
    recognition_predictor=None,
    detection_predictor=None,
):
    """
    Run Surya OCR on a single PIL Image page.

    Args:
        image: PIL Image of the page to OCR.
        recognition_predictor: Surya RecognitionPredictor (if None, will be initialized).
        detection_predictor: Surya DetectionPredictor (if None, will be initialized).

    Returns:
        Page prediction object with .text_lines attribute.
        Each text_line has .text (str) and .bbox (x1, y1, x2, y2).
    """
    if recognition_predictor is None or detection_predictor is None:
        recognition_predictor, detection_predictor = init_surya_ocr()

    predictions = recognition_predictor(
        [image], det_predictor=detection_predictor
    )
    # predictions is a list (one per input image); we passed one image
    return predictions[0]
