FROM python:3.12-slim

WORKDIR /app

# Install build dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set environment defaults
ENV PYTHONUNBUFFERED=1
ENV AGENT_MODE=mock
ENV PORT=8000

EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "src.ui_server:app", "--host", "0.0.0.0", "--port", "8000"]
