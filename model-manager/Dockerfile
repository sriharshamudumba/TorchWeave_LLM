FROM python:3.12-slim

# (optional) small build toolchain for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps from your requirements file
# (build context is ./model-manager)
COPY requirements.txt /app/model-manager/requirements.txt
RUN pip install --no-cache-dir -r /app/model-manager/requirements.txt

# Copy only the source
COPY src /app/model-manager/src

# Runtime env
ENV PYTHONPATH=/app:/app/model-manager/src
ENV TOKENIZERS_PARALLELISM=false

# Run from the source dir by default
WORKDIR /app/model-manager/src
