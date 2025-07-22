# Multi-stage build for API Service
FROM solar-base:latest as base

# Build stage
FROM base as builder
WORKDIR /app

# Copy service-specific requirements
COPY docker/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production
WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Only copy what you need
COPY api/ ./api/
COPY mlpipeline/ ./mlpipeline/
COPY config.py .
COPY .env .

EXPOSE 8000
ENTRYPOINT ["python", "api/wait_for_mlflow_model.py"] 