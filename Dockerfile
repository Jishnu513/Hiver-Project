FROM python:3.12-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch CPU-only first (saves ~1GB vs CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face Spaces requires port 7860
# AGENT_MODE=mock means demo works without any API keys
ENV PYTHONUNBUFFERED=1
ENV AGENT_MODE=mock
ENV PORT=7860

EXPOSE 7860

# Use 2 workers for better responsiveness on HF Spaces
CMD ["uvicorn", "src.ui_server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1", "--timeout-keep-alive", "75"]
