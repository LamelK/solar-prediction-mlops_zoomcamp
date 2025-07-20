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

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY api/ ./api/
COPY mlpipeline/preprocessing_utils.py ./mlpipeline/
COPY config.py .

EXPOSE 8000
ENTRYPOINT ["uvicorn", "api.serve_model:app", "--host", "0.0.0.0", "--port", "8000"] 