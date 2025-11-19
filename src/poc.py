"""
pdf_ar_to_id_poc.py
PoC pipeline:
  PDF -> images (pymupdf)
  (optional) DiT layout -> get text regions
  PaddleOCR (Arabic) -> lines with boxes + ocr confidence
  ar->en translation (Helsinki OPUS-MT)
  en->id translation (Helsinki OPUS-MT)
  compute approximate translation score via model generate() token log-probs
  save CSV report

Caveat: this is a prototype. Optimize batching, GPU usage, error handling in production.
"""

import argparse
import math
import csv
from pathlib import Path
from tqdm import tqdm

import fitz  # pymupdf
from PIL import Image
import numpy as np
import torch
import pandas as pd

# PaddleOCR
from paddleocr import PaddleOCR

# Transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, logging as hf_logging
hf_logging.set_verbosity_error()

# ----------------------------
# Helpers: render PDF -> images
# ----------------------------
def pdf_to_images(pdf_path: str, dpi: int = 300):
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    images = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        mode = "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images

# ----------------------------
# Optional: DiT-based layout (placeholder)
# ----------------------------
def run_layout_dit(page_image: Image.Image):
    """
    Placeholder: if you have a DiT-based layoutparser setup,
    detect semantic blocks and return list of boxes [(x1,y1,x2,y2), ...]
    For this PoC we return None to indicate "not used".
    """
    # <- add DiT/layoutparser integration here if available
    return None

# ----------------------------
# PaddleOCR wrapper (detection+recognition)
# ----------------------------
def init_paddle_ocr(lang: str = "ar"):
    # PaddleOCR supports many languages; set use_angle_cls and det/rec model paths optionally
    # For Arabic, PaddleOCR provides pretrained Arabic recognition models (see PaddleOCR docs).
    ocr = PaddleOCR(lang="ar", use_angle_cls=False, use_gpu=torch.cuda.is_available())
    return ocr

def ocr_page_lines(paddle_ocr, page_image: Image.Image):
    """
    Run PaddleOCR on PIL image.
    Returns list of dicts: { 'text', 'box':(x1,y1,x2,y2), 'confidence' }
    PaddleOCR returns boxes as 4 corner points; we convert to bbox.
    """
    # PaddleOCR expects file path or numpy array
    np_img = np.array(page_image)
    results = paddle_ocr.ocr(np_img, cls=False)
    lines = []
    for line in results:
        # Each line is [box, (text, score)]
        box, (txt, score) = line
        # box is 4 points; convert to bbox
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        lines.append({
            "text": txt,
            "box": (float(x1), float(y1), float(x2), float(y2)),
            "confidence": float(score)
        })
    return lines

# ----------------------------
# Translation helpers (HF models) with generation score
# ----------------------------
def load_mt_model(model_name: str, device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    return tok, model

import torch.nn.functional as F

def translate_and_score(text: str, tokenizer, model, device="cpu", max_new_tokens=200):
    """
    Translate text and compute an approximate avg token log-prob (higher = more confident).
    Returns (translated_text, avg_logprob)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        # generate with scores
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            # do_sample=False ensures deterministic beam/greedy
        )
    # tokens produced
    sequences = outputs.sequences  # shape (batch, seq_len)
    seq = sequences[0].tolist()
    # outputs.scores is a list: each element is logits over vocab for a generated step
    scores = outputs.scores  # list of tensors len = generated_len
    if scores is None or len(scores) == 0:
        # fallback: no scores available (older HF versions)
        translated = tokenizer.decode(seq, skip_special_tokens=True)
        return translated, None

    # For seq2seq, the first generated token corresponds to scores[0], etc.
    # We need to align generated token ids to scores:
    # many models include BOS token at start of seq; to map safely, we compute generated tokens as:
    # find index where generated tokens start by comparing with input ids (crude but works for many)
    # Simpler heuristic: skip the initial token if it's BOS or equals input start.
    # We'll pick the last len(scores) tokens as the generated ones.
    gen_token_ids = seq[-len(scores):]  # take last N tokens corresponding to scores

    # compute avg logprob:
    logprob_sum = 0.0
    token_count = 0
    for t_idx, logits in enumerate(scores):
        # logits shape (batch, vocab_size)
        logits = logits[0]  # (vocab,)
        prob = F.log_softmax(logits, dim=0)
        tokid = gen_token_ids[t_idx]
        logp = prob[tokid].item()
        logprob_sum += logp
        token_count += 1
    avg_logprob = logprob_sum / token_count if token_count > 0 else None

    translated = tokenizer.decode(seq, skip_special_tokens=True)
    return translated, avg_logprob

# ----------------------------
# Top-level pipeline
# ----------------------------
def process_pdf(
    input_pdf: str,
    output_csv: str,
    device: str = "cpu",
    use_dit_layout: bool = False
):
    images = pdf_to_images(input_pdf, dpi=300)
    print(f"Rendered {len(images)} pages.")

    # init PaddleOCR Arabic
    paddle_ocr = init_paddle_ocr(lang="ar")

    # load MT models (ar->en and en->id)
    # If direct ar->id exists you'd replace this with one model; here we use ar->en then en->id
    ar_en_model_name = "Helsinki-NLP/opus-mt-tc-big-ar-en"   # ar->en
    en_id_model_name = "Helsinki-NLP/opus-mt-en-id"          # en->id

    print("Loading MT models (this may take a while)...")
    ar_en_tok, ar_en_model = load_mt_model(ar_en_model_name, device=device)
    en_id_tok, en_id_model = load_mt_model(en_id_model_name, device=device)

    rows = []
    for page_no, img in enumerate(tqdm(images, desc="pages")):
        # layout (optional)
        boxes = None
        if use_dit_layout:
            boxes = run_layout_dit(img)  # if returns list of boxes you'll need to crop; not implemented here

        # run OCR (we rely on PaddleOCR's detection+recog)
        lines = ocr_page_lines(paddle_ocr, img)
        print(f"Page {page_no+1}: {len(lines)} OCR lines found.")

        for li, ln in enumerate(lines):
            ar_text = ln["text"]
            ocr_conf = ln["confidence"]
            bbox = ln["box"]

            # 1) ar -> en
            en_text, en_score = translate_and_score(ar_text, ar_en_tok, ar_en_model, device=device)

            # 2) en -> id
            id_text, id_score = translate_and_score(en_text, en_id_tok, en_id_model, device=device)

            rows.append({
                "page": page_no+1,
                "line_index": li,
                "box": bbox,
                "ar_text": ar_text,
                "ocr_conf": ocr_conf,
                "en_text": en_text,
                "en_avg_logprob": en_score,
                "id_text": id_text,
                "id_avg_logprob": id_score
            })

    # save CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print("Saved CSV:", output_csv)
    return df

# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", "-i", required=True)
    ap.add_argument("--output", "-o", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--use-dit", action="store_true", help="Attempt DiT-based layout first (must implement run_layout_dit).")
    args = ap.parse_args()
    process_pdf(args.input, args.output, device=args.device, use_dit_layout=args.use_dit)

if __name__ == "__main__":
    main()
