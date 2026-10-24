# BriefWise — Production Container Deployment
# Compatible with Docker, Hugging Face Spaces (Docker SDK), Render, Railway, and Cloud Run

FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application backend, frontend, and final model directory
COPY app.py index.html ./
COPY briefwise_final ./briefwise_final

# Environment variables for server binding
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

# Start FastAPI via uvicorn wrapper in app.py
CMD ["python", "app.py"]
