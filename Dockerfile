### Multi-stage build: build dependencies in a builder image, copy runtime files into smaller final image
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install build-time tools required by native wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements and install into a location we can copy to the final image
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefix=/install -r /app/requirements.txt

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from the builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app

# Expose default port
EXPOSE 8000

# Start the ASGI server; let the platform override PORT
CMD ["sh", "-c", "uvicorn app.main:api --host 0.0.0.0 --port ${PORT:-8000}"]
