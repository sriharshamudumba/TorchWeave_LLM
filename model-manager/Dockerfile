FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn pydantic httpx huggingface_hub

# Copy source code
COPY src/ /app/src/

# Expose port
EXPOSE 8001

# Run the application
CMD ["python", "-m", "uvicorn", "src.model_manager:app", "--host", "0.0.0.0", "--port", "8001"]
