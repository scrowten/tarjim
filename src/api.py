import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from .core.pdf_handler import process_pdf

app = FastAPI(
    title="Tarjim PDF Translator",
    description="An API to translate Arabic text in PDF documents.",
    version="1.0.0"
)

@app.post("/translate-pdf/", response_class=FileResponse)
async def translate_pdf_endpoint(
    file: UploadFile = File(..., description="The Arabic PDF file to translate."),
    target_lang: str = "en"
):
    """
    Upload a PDF, translate its content, and return the translated PDF.
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_input_file:
            shutil.copyfileobj(file.file, temp_input_file)
            temp_input_path = temp_input_file.name

        temp_output_path = tempfile.mktemp(suffix=".pdf")
        
        process_pdf(temp_input_path, temp_output_path, target_lang)

        return FileResponse(temp_output_path, media_type='application/pdf', filename=f"translated_{file.filename}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
