# Multi-Stage / Optimized Python 3.10 Production Image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=5000 \
    HOST=0.0.0.0 \
    FLASK_DEBUG=0

# Set working directory
WORKDIR /app

# Install system audio dependencies for librosa & soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy application source code
COPY . .

# Ensure upload/session directories exist
RUN mkdir -p static/spectrograms static/waveforms outputs/reports outputs/plots outputs/models

# Expose server port
EXPOSE 5000

# Run with multi-worker Gunicorn server for high concurrency
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--threads", "2", "--timeout", "120", "app:app"]
