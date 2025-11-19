import io
import torch
from typing import Union, List, Tuple
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

# Lazy-loaded model and processor
_MODEL = None
_PROCESSOR = None

def _load_vlm():
    global _MODEL, _PROCESSOR
    if _MODEL is not None and _PROCESSOR is not None:
        return _MODEL, _PROCESSOR

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        # Prefer auto device mapping on GPU, fall back to CPU when not available
        device_map = "auto" if device == "cuda" else "cpu"
        _MODEL = Qwen2VLForConditionalGeneration.from_pretrained(
            "MBZUAI/AIN", torch_dtype=torch_dtype, device_map=device_map
        )
        _PROCESSOR = AutoProcessor.from_pretrained("MBZUAI/AIN")
    except Exception:
        # Fallback to cpu-only load if auto mapping fails
        _MODEL = Qwen2VLForConditionalGeneration.from_pretrained("MBZUAI/AIN", device_map="cpu")
        _PROCESSOR = AutoProcessor.from_pretrained("MBZUAI/AIN")

    return _MODEL, _PROCESSOR


def perform_vlm_ocr(image: Union[Image.Image, str, bytes], prompt_text: str = None, max_new_tokens: int = 128) -> Union[str, List[str]]:
    """
    Use the Qwen2VL model to recognize (Arabic) text from an image.

    Args:
        image: PIL Image object, path to image, or raw bytes.
        prompt_text: Optional instruction to the model. Defaults to an Arabic OCR prompt.
        max_new_tokens: Maximum tokens to generate for the recognition output.

    Returns:
        Recognized text string (or list of strings for batch inputs).
    """
    model, processor = _load_vlm()

    # Load image if a path/bytes were provided
    if isinstance(image, (str, bytes)):
        image = Image.open(image if isinstance(image, str) else io.BytesIO(image)).convert("RGB")
    else:
        # Ensure PIL Image in RGB mode
        image = image.convert("RGB")

    if prompt_text is None:
        prompt_text = "Recognize the Arabic text on this image. Return only the recognized text."

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    inputs = processor(text=[text_prompt], images=[image], padding=True, return_tensors="pt")

    # Move tensors to CUDA if available and model uses CUDA
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items() if hasattr(v, "to")}

    # Generate
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Slice off the prompt tokens from the generated sequence for each batch item
    input_ids = inputs.get("input_ids")
    decoded_texts = []
    if input_ids is None:
        # Fallback: decode entire output
        decoded_texts = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    else:
        for i in range(output_ids.shape[0]):
            in_len = input_ids.shape[1]
            gen_part = output_ids[i, in_len:]
            # convert to list of ids for batch_decode
            gen_ids = gen_part.detach().cpu().tolist()
            decoded = processor.batch_decode([gen_ids], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
            decoded_texts.append(decoded)

    return decoded_texts[0] if len(decoded_texts) == 1 else decoded_texts


def perform_vlm_ocr_on_boxes(image: Union[Image.Image, str, bytes], boxes: List[Tuple[int, int, int, int]], prompt_text: str = None, max_new_tokens: int = 128) -> List[Tuple[Tuple[int, int, int, int], str]]:
    """
    Run VLM per detected box and return list of (box, recognized_text).

    Args:
        image: PIL Image, path, or bytes.
        boxes: list of (x, y, w, h) tuples.
    """
    # Ensure PIL image
    if isinstance(image, (str, bytes)):
        image = Image.open(image if isinstance(image, str) else io.BytesIO(image)).convert("RGB")
    else:
        image = image.convert("RGB")

    results = []
    for (x, y, w, h) in boxes:
        crop = image.crop((x, y, x + w, y + h))
        text = perform_vlm_ocr(crop, prompt_text=prompt_text, max_new_tokens=max_new_tokens)
        results.append(((x, y, w, h), text))

    return results
