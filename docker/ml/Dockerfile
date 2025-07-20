# Multi-stage build for ML Pipeline
FROM solar-base:latest as base

# Build stage
FROM base as builder
WORKDIR /app

# Copy service-specific requirements
COPY docker/ml/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY mlpipeline/ ./mlpipeline/
COPY pipeline.py .
COPY retrain.py .
COPY config.py .

# Keep container running - Prefect will execute flows when needed
CMD ["tail", "-f", "/dev/null"] 