FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy ONLY this directory contents into /app
COPY . /app

# Run main.py (not src/server.py); disable reload to avoid scanning stray paths
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","8000","--no-reload"]

