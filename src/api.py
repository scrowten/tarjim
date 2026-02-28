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
from .core.translator_argos import get_translation_route, get_supported_languages

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
async def list_languages():
    """List available target languages and their translation routes from Arabic."""
    # Common targets with their route info
    common_targets = [
        {"code": "en", "name": "English", "route": "direct"},
        {"code": "id", "name": "Indonesian", "route": "pivot:en"},
        {"code": "ms", "name": "Malay", "route": "pivot:en"},
        {"code": "fr", "name": "French", "route": "pivot:en"},
        {"code": "de", "name": "German", "route": "pivot:en"},
        {"code": "es", "name": "Spanish", "route": "pivot:en"},
        {"code": "tr", "name": "Turkish", "route": "pivot:en"},
        {"code": "ur", "name": "Urdu", "route": "pivot:en"},
    ]
    return JSONResponse(content={"languages": common_targets})


@app.get("/api/route")
async def check_route(
    source: str = Query(default="ar", description="Source language code"),
    target: str = Query(description="Target language code"),
):
    """Check the translation route for a language pair."""
    route = get_translation_route(source, target)
    return JSONResponse(content={
        "source": source,
        "target": target,
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
):
    """
    Upload a PDF, translate its content, and return the translated PDF.

    Supports direct and pivot translation:
    - ar → en: direct translation
    - ar → id: automatically pivots through English (ar → en → id)
    """
    temp_input_path = None
    temp_output_path = None
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
            "API request: translate %s → %s (overlay: %s)",
            source_lang, target_lang, overlay_mode,
        )

        # Run the translation pipeline
        process_pdf(
            input_path=temp_input_path,
            output_path=temp_output_path,
            target_lang=target_lang,
            source_lang=source_lang,
            overlay_mode=overlay_mode,
        )

        # Build a descriptive filename
        lang_suffix = target_lang
        output_filename = f"translated_{lang_suffix}_{file.filename}"

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
