FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Update system packages and install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    httpx \
    huggingface_hub \
    psutil \
    transformers \
    torch \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy source code
COPY src/ /app/src/

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "src.model_manager:app", "--host", "0.0.0.0", "--port", "8001"]
