FROM python:3.12-slim

# ── Sistema: librerías que necesita OpenCV en entornos headless ────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencias Python ────────────────────────────────────────────────────
COPY requirements.txt .

# 1. Instalar opencv-python-headless PRIMERO (antes de que ultralytics
#    intente traer opencv-python con dependencias de GUI)
RUN pip install --no-cache-dir opencv-python-headless

# 2. Instalar el resto; si ultralytics intenta instalar opencv-python,
#    pip detecta el conflicto y conserva la versión headless ya instalada
RUN pip install --no-cache-dir -r requirements.txt

# 3. Por si acaso: forzar headless y eliminar la versión con GUI
RUN pip uninstall -y opencv-python 2>/dev/null || true \
    && pip install --no-cache-dir opencv-python-headless

# ── Código fuente ──────────────────────────────────────────────────────────
COPY src/ ./src/

# Copiar el modelo YOLO si existe localmente; si no, ultralytics lo descarga
COPY yolov8n.pt ./src/yolov8n.pt 2>/dev/null || true

# ── Arranque ───────────────────────────────────────────────────────────────
# DigitalOcean inyecta $PORT automáticamente (por defecto 8080)
ENV PORT=8080
# Reduce threads de PyTorch → elimina warnings de NNPACK en CPUs sin AVX2
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT} --app-dir src"]
