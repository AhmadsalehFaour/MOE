# ==========================================
# Base image: PyTorch + CUDA 11.8 + cuDNN 8
# ==========================================
FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# System deps: Tesseract + langs + core fonts + tools
# NOTE: we intentionally DO NOT install fonts-amiri (not present in jammy repo)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-ara \
    libtesseract-dev \
    libleptonica-dev \
    fonts-noto-core \
    fonts-noto-extra \
    fontconfig \
    curl ca-certificates \
    ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Add Arabic fonts directly (Amiri + Noto Naskh), then rebuild font cache
RUN set -eux; \
    mkdir -p /usr/local/share/fonts/custom; \
    curl -L -o /usr/local/share/fonts/custom/Amiri-Regular.ttf \
      https://github.com/aliftype/amiri/releases/download/0.116/Amiri-Regular.ttf; \
    curl -L -o /usr/local/share/fonts/custom/Amiri-Bold.ttf \
      https://github.com/aliftype/amiri/releases/download/0.116/Amiri-Bold.ttf; \
    curl -L -o /usr/local/share/fonts/custom/NotoNaskhArabic-Regular.ttf \
      https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoNaskhArabic/NotoNaskhArabic-Regular.ttf; \
    curl -L -o /usr/local/share/fonts/custom/NotoNaskhArabic-Bold.ttf \
      https://github.com/googlefonts/noto-fonts/raw/main/hinted/ttf/NotoNaskhArabic/NotoNaskhArabic-Bold.ttf; \
    fc-cache -f -v

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

EXPOSE 8501

# Optional healthcheck for Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=10 \
  CMD curl -fsS http://127.0.0.1:8501/_stcore/health || exit 1

CMD ["streamlit","run","app.py","--server.address=0.0.0.0","--server.port=8501","--browser.gatherUsageStats=false"]
