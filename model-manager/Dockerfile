# Model Manager Service Dockerfile
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ARTIFACTS_DIR=/artifacts
ENV MODEL_CACHE_DIR=/model_cache
ENV HUGGINGFACE_CACHE_DIR=/model_cache/huggingface
ENV SERVER_URL=http://server:8000

# Create necessary directories
RUN mkdir -p /artifacts /model_cache /model_cache/huggingface

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Create a non-root user for security
RUN groupadd -r modelmanager && useradd -r -g modelmanager modelmanager
RUN chown -R modelmanager:modelmanager /app /artifacts /model_cache
USER modelmanager

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "-m", "uvicorn", "src.model_manager:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
