FROM python:3.12-slim


ENV PIP_NO_CACHE_DIR=1 \
PYTHONDONTWRITEBYTECODE=1 \
PYTHONUNBUFFERED=1


WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
git wget && \
rm -rf /var/lib/apt/lists/*


COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt


COPY src /app/src
EXPOSE 8000
CMD ["uvicorn", "src.server:app", "--host", "0.0.0.0", "--port", "8000"]