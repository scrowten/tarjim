# ===========================
# Tarjim: Arabic PDF Translator
# ===========================
# Uses Surya OCR + Argos Translate (fully open-source, offline)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# - libgl1 / libglib2.0 for OpenCV
# - fonts for text overlay
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Argos Translate language packages
# ar -> en (direct), en -> id (for pivot ar -> en -> id)
RUN python -c "\
from argostranslate import package; \
package.update_package_index(); \
available = package.get_available_packages(); \
pairs = [('ar', 'en'), ('en', 'id')]; \
for src, tgt in pairs: \
    pkg = next((p for p in available if p.from_code == src and p.to_code == tgt), None); \
    if pkg: \
        package.install_from_path(pkg.download()); \
        print(f'Installed {src}->{tgt} package'); \
    else: \
        print(f'WARNING: {src}->{tgt} package not found')"

# Copy application code
COPY ./__init__.py /app/__init__.py
COPY ./src /app/src
COPY ./static /app/static
COPY ./fonts /app/fonts

# Expose port and run the API
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
