# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install Tesseract OCR and other system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-ara \
    libtesseract-dev \
    libleptonica-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download Argos Translate models
RUN python -c "from argostranslate import package; package.update_package_index(); available_packages = package.get_available_packages(); package.install_from_path(next(filter(lambda x: x.from_code == 'ar' and x.to_code == 'en', available_packages)).download())"

# Copy application code
COPY ./src /app/src

# Expose port and run the API
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
