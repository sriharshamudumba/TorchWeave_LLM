FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# The COPY commands are now relative to the build context (./server)
COPY requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir -r /app/server/requirements.txt

# Set Python path to find modules
ENV PYTHONPATH=/app:/app/server/src

# Copy the source code
COPY src /app/server/src
