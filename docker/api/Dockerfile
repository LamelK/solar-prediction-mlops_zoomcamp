# Dockerfile for API Service (no shared base)
FROM python:3.11-slim as production
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc curl procps && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY docker/api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ ./api/
COPY mlpipeline/ ./mlpipeline/
COPY config.py .
COPY .env .

EXPOSE 8000
ENTRYPOINT ["python", "api/wait_for_mlflow_model.py"] 