<div align="center">
  <img src="docs/logo.png" alt="Tarjim Logo" width="150">
</div>

# 🕌 Tarjim: PDF Arabic OCR & Translator [WIP]
End-to-end pipeline to **extract Arabic text from PDF**, **translate it**, and **generate a new translated PDF** — all using open-source tools.

---

## 📘 Overview
This project automates the process of handling Arabic documents by integrating:
1. **PDF parsing** (via PyMuPDF / pdf2image)
2. **Arabic OCR** (via Tesseract OCR with enhanced Arabic model)
3. **Open-source translation** (via Argos Translate or LibreTranslate)
4. **PDF generation** (via PyMuPDF / ReportLab)

It’s designed for researchers, students, and automation developers who need an **offline, private, and flexible** document translation pipeline.

---

## 🧭 Features
✅ Extract text from scanned or non-searchable Arabic PDFs  
✅ Support for right-to-left (RTL) Arabic text  
✅ Translate text into English or any supported language  
✅ Save results as new searchable, translated PDF  
✅ Fully offline (if using Argos Translate)  
✅ Modular Python code (OCR / translation / PDF generation separated)

---

## 🏗️ Pipeline Architecture
```mermaid
graph TD
    A[Input PDF] --> B[Convert to Images]
    B --> C[Arabic OCR (Tesseract)]
    C --> D[Translation (Argos Translate / LibreTranslate)]
    D --> E[Generate Output PDF (PyMuPDF)]
    E --> F[Translated PDF Output]
```

## 🧰 Tech Stack

## ⚙️ Installation
```
# Clone the repo
git clone https://github.com/<your-username>/pdf-arabic-ocr-translate.git
cd pdf-arabic-ocr-translate

# Create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate (Windows)

# Install dependencies
pip install -r requirements.txt
```

## 📦 Dependencies (requirements.txt)

```
pytesseract
pdf2image
PyMuPDF
Pillow
argos-translate
```

## 🧩 Additional setup

## 🚀 Usage

```
python src/main.py --input input_arabic.pdf --output output_translated.pdf --lang en
```

Example Output:

## 🧠 How It Works

## 📊 Performance Notes

## 🛠️ Folder Structure
```
pdf-arabic-ocr-translate/
├── README.md
├── requirements.txt
├── notebooks                # Jupyter notebooks for exploration and analysis
├── src/
│   ├── main.py              # Orchestrator
│   ├── ocr.py               # OCR functions
│   ├── translate.py         # Translation functions
│   ├── pdf_handler.py       # PDF read/write utilities
│   └── utils.py             # Helper functions
├── tests/
│   ├── sample.pdf
│   └── expected_output.pdf
└── docs/
    ├── sample_input.png
    └── sample_output.png
```

## 🧪 Example Code Snippet

```import fitz
from pdf2image import convert_from_path
import pytesseract
from argostranslate import translate, package

# Load translator
installed_languages = translate.get_installed_languages()
from_lang = next(filter(lambda x: x.code == "ar", installed_languages))
to_lang = next(filter(lambda x: x.code == "en", installed_languages))
translator = from_lang.get_translation(to_lang)

def ocr_and_translate_page(image, translator):
    arabic_text = pytesseract.image_to_string(image, lang='ara')
    translated_text = translator.translate(arabic_text)
    return translated_text

def process_pdf(input_pdf, output_pdf):
    pages = convert_from_path(input_pdf, dpi=300)
    doc = fitz.open()
    for page_img in pages:
        translated = ocr_and_translate_page(page_img, translator)
        page = doc.new_page()
        page.insert_text((72, 72), translated, fontsize=12)
    doc.save(output_pdf)

if __name__ == "__main__":
    process_pdf("input_arabic.pdf", "output_translated.pdf")
```

## 🧩 Possible Extensions
✅ Add layout & formatting preservation (align translated text boxes)
✅ Add automatic language detection
✅ Add batch PDF support
✅ Support cloud translation APIs (Google, DeepL) optionally

## 🧑‍💻 Author
Risky Agung Dwi Putranto

## 🪪 License

MIT License — free to use, modify, and share.

## 🙌 Acknowledgements