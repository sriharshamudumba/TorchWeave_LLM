FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Add /app to Python path (so "src.*" imports always resolve)
ENV PYTHONPATH=/app

# Copy the service source
COPY src /app/src

# Run FastAPI app in src/main.py (NOT src/server.py)
CMD ["uvicorn","src.main:app","--host","0.0.0.0","--port","8000","--no-reload"]
