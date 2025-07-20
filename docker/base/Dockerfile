FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Copy and install common Python packages
COPY docker/base/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 