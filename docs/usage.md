Tarjim OCR + VLM Usage
----------------------

This project can translate Arabic text in PDFs. By default it uses Tesseract for OCR. You can enable a Visual-Language Model (Qwen2VL) based OCR for better results on challenging images.

Install (recommended into a virtualenv):

```bash
pip install -r requirements.txt
```

Quick CLI usage:

```bash
python -m src.core.main -i input_ar.pdf -o output_translated.pdf -f /path/to/font.ttf
```

Use the VLM for OCR (may require GPU and more RAM):

```bash
python -m src.core.main -i input_ar.pdf -o output_translated.pdf -f /path/to/font.ttf --use-vlm
```

Notes:
- `--use-vlm` will load the Qwen2VL model from Hugging Face and ask it to recognize the Arabic text on each page as a block. The recognized text will be translated and overlaid as a single block at the top of each page.
- If you need bounding boxes per word/line for fine-grained replacement, keep using Tesseract (default) or extend `ocr_vlm.py` to parse layout from a detection model.
- Large models may require `accelerate` and a GPU. If loading fails with `device_map='auto'`, the loader falls back to CPU.
