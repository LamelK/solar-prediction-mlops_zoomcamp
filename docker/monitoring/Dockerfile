# Multi-stage build for Monitoring
FROM solar-base:latest as base

# Build stage
FROM base as builder
WORKDIR /app

# Copy service-specific requirements
COPY docker/monitoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM base as production
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY monitoring/ ./monitoring/
COPY config.py .

# Copy startup script
COPY docker/start_monitoring_container.sh /app/start_monitoring_container.sh
RUN chmod +x /app/start_monitoring_container.sh

EXPOSE 8080
ENTRYPOINT ["/app/start_monitoring_container.sh"] 