"""
FastAPI web API and UI for the Tarjim PDF translation tool.

Run with:
    uvicorn src.api:app --host 0.0.0.0 --port 8000

Supports:
    - Arabic → English (direct)
    - Arabic → Indonesian (pivot via English)
    - Arabic → any Argos-supported language
"""

import shutil
import tempfile
import os
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .core.pdf_handler import process_pdf
from .core.translator_nllb import get_translation_route as nllb_route
from .core.translator_argos import get_translation_route as argos_route, get_supported_languages

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Tarjim PDF Translator",
    description=(
        "API and web UI to translate Arabic text in PDF documents "
        "using open-source OCR and translation. "
        "Supports direct and pivot translation routes."
    ),
    version="2.1.0",
)

# Mount the 'static' directory to serve files like index.html, css, js
_static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page for the GUI."""
    index_path = os.path.join(_static_dir, "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse(
            content="<h1>Tarjim PDF Translator</h1><p>Use POST /translate-pdf/ to translate a PDF.</p>",
            status_code=200,
        )
    with open(index_path) as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/api/languages")
async def list_languages(translator: str = Query(default="nllb")):
    """List available target languages and their translation routes from Arabic."""
    common_targets = [
        {"code": "en", "name": "English"},
        {"code": "id", "name": "Indonesian"},
        {"code": "ms", "name": "Malay"},
        {"code": "fr", "name": "French"},
        {"code": "de", "name": "German"},
        {"code": "es", "name": "Spanish"},
        {"code": "tr", "name": "Turkish"},
        {"code": "ur", "name": "Urdu"},
    ]
    route_fn = nllb_route if translator == "nllb" else argos_route
    for lang in common_targets:
        lang["route"] = route_fn("ar", lang["code"])
    return JSONResponse(content={"languages": common_targets, "translator": translator})


@app.get("/api/route")
async def check_route(
    source: str = Query(default="ar", description="Source language code"),
    target: str = Query(description="Target language code"),
    translator: str = Query(default="nllb", description="Translator backend"),
):
    """Check the translation route for a language pair."""
    route_fn = nllb_route if translator == "nllb" else argos_route
    route = route_fn(source, target)
    return JSONResponse(content={
        "source": source,
        "target": target,
        "translator": translator,
        "route": route,
        "description": {
            "direct": f"Direct translation: {source} → {target}",
            "pivot:en": f"Pivot translation: {source} → en → {target}",
            "unavailable": f"No translation path found for {source} → {target}",
        }.get(route, route),
    })


@app.post("/translate-pdf/", response_class=FileResponse)
async def translate_pdf_endpoint(
    file: UploadFile = File(..., description="The Arabic PDF file to translate."),
    target_lang: str = Form(default="en", description="Target language code (e.g., 'en', 'id')"),
    source_lang: str = Form(default="ar", description="Source language code (default: 'ar')"),
    overlay_mode: str = Form(default="replace", description="Overlay mode: 'replace' or 'clean'"),
    translator: str = Form(
        default="nllb",
        description="Translation backend: 'nllb' (default, higher quality) or 'argos' (lightweight fallback).",
    ),
    tashkeel: bool = Form(
        default=False,
        description=(
            "Enable Arabic diacritization (tashkeel) before translation. "
            "Improves translation accuracy for undiacritized kitab text. "
            "Uses CATT (trained on Shamela classical Arabic corpus)."
        ),
    ),
    show_tashkeel: bool = Form(
        default=False,
        description=(
            "Show diacritized Arabic text in the output PDF, stacked above the translation. "
            "Requires tashkeel=true."
        ),
    ),
):
    """
    Upload a PDF, translate its content, and return the translated PDF.

    Supports direct and pivot translation:
    - ar → en: direct translation
    - ar → id: automatically pivots through English (ar → en → id)

    Optional tashkeel modes:
    - tashkeel=true: diacritize Arabic before translation (better accuracy)
    - tashkeel=true&show_tashkeel=true: also render diacritized Arabic in output PDF
    """
    temp_input_path = None
    temp_output_path = None

    # show_tashkeel implies tashkeel
    if show_tashkeel and not tashkeel:
        tashkeel = True

    try:
        # Validate overlay mode
        if overlay_mode not in ("replace", "clean"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid overlay_mode: {overlay_mode}. Must be 'replace' or 'clean'.",
            )

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_input_file:
            shutil.copyfileobj(file.file, temp_input_file)
            temp_input_path = temp_input_file.name

        temp_output_path = tempfile.mktemp(suffix=".pdf")

        logger.info(
            "API request: translate %s → %s (translator: %s, overlay: %s, tashkeel: %s, show_tashkeel: %s)",
            source_lang, target_lang, translator, overlay_mode, tashkeel, show_tashkeel,
        )

        # Run the translation pipeline
        process_pdf(
            input_path=temp_input_path,
            output_path=temp_output_path,
            target_lang=target_lang,
            source_lang=source_lang,
            overlay_mode=overlay_mode,
            tashkeel=tashkeel,
            show_tashkeel=show_tashkeel,
            translator=translator,
        )

        # Build a descriptive filename
        lang_suffix = target_lang
        tashkeel_suffix = "_tashkeel" if tashkeel else ""
        translator_suffix = f"_{translator}" if translator != "nllb" else ""
        output_filename = f"translated_{lang_suffix}{tashkeel_suffix}{translator_suffix}_{file.filename}"

        return FileResponse(
            temp_output_path,
            media_type="application/pdf",
            filename=output_filename,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Translation failed")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up input temp file
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        # Note: FileResponse handles the output file delivery before cleanup
