# ──────────────────────────────────────────────────────────────
# FastAPI backend — Hugging Face Spaces (Docker SDK)
# Build context = the contents of 03_web_app/ at the Space repo root.
# ──────────────────────────────────────────────────────────────

FROM python:3.12-slim

# Install build-essential as a safety net in case any pip package needs to compile from source
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set /app as the working directory
WORKDIR /app

# Install CPU-only torch, with version same as local conda env.
# (avoids default PyPI torch with CUDA ~2.5 GB, useless on a CPU Space and would blow the build)
RUN pip install --no-cache-dir torch==2.5.1 \
        --extra-index-url https://download.pytorch.org/whl/cpu

# Install the rest of the deps. Copy before code to cache, and avoid reinstalls on each deployment
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the backend code into /app
COPY . .

# Set dir where HF hub should cache downloaded artifacts (models, etc)
ENV HF_HOME=/app/.cache/huggingface

# Docker Space needs to listen on port 7860 by default
ENV PORT=7860
EXPOSE 7860

# Set the command that starts the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
