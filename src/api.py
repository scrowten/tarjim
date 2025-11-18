import shutil
import tempfile
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .core.pdf_handler import process_pdf

app = FastAPI(
    title="Tarjim PDF Translator",
    description="An API and web UI to translate Arabic text in PDF documents.",
    version="1.0.0"
)

# Mount the 'static' directory to serve files like index.html, css, js
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page for the GUI."""
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/translate-pdf/", response_class=FileResponse)
async def translate_pdf_endpoint(
    file: UploadFile = File(..., description="The Arabic PDF file to translate."),
    target_lang: str = "en"
):
    """
    Upload a PDF, translate its content, and return the translated PDF.
    """
    temp_input_path = None
    temp_output_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_input_file:
            shutil.copyfileobj(file.file, temp_input_file)
            temp_input_path = temp_input_file.name

        temp_output_path = tempfile.mktemp(suffix=".pdf")

        process_pdf(temp_input_path, temp_output_path, target_lang)

        return FileResponse(temp_output_path, media_type='application/pdf', filename=f"translated_{file.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    finally:
        # Clean up temporary files
        if temp_input_path and os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        # The FileResponse will handle the output file, but we can add cleanup for it too if needed.