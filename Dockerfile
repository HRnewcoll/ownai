# ============================================================
# OwnAI – Dockerfile
# Build: docker build -t ownai .
# Run:   docker run -p 5000:5000 -v $(pwd)/models_dir:/app/models_dir ownai
# ============================================================

# ── Base image ───────────────────────────────────────────────
# CPU-only image; swap for pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime for GPU
FROM python:3.11-slim

LABEL org.opencontainers.image.title="OwnAI"
LABEL org.opencontainers.image.description="The AI Development Platform for Everyone"
LABEL org.opencontainers.image.source="https://github.com/HRnewcoll/ownai"

# ── System deps ──────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# ── App directory ────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ──────────────────────────────────────
COPY requirements.txt .
# Install CPU-only torch first (smaller image), full requirements second
RUN pip install --upgrade pip --no-cache-dir \
 && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu --no-cache-dir \
 && pip install -r requirements.txt --no-cache-dir

# ── Copy source ──────────────────────────────────────────────
COPY . .

# ── Persistent volumes ───────────────────────────────────────
RUN mkdir -p models_dir uploads
VOLUME ["/app/models_dir", "/app/uploads"]

# ── Environment ──────────────────────────────────────────────
ENV FLASK_ENV=production
ENV PORT=5000

# ── Expose & run ─────────────────────────────────────────────
EXPOSE 5000

CMD ["python", "app.py"]
