# Dockerfile for Monitoring Service (no shared base)
FROM python:3.11-slim as production
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc curl procps && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY docker/monitoring/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY monitoring/ ./monitoring/
COPY mlpipeline/ ./mlpipeline/
COPY config.py .

# Set PYTHONPATH so config.py is importable
ENV PYTHONPATH=/app

# Copy startup script
COPY docker/start_monitoring_container.sh /app/start_monitoring_container.sh
RUN chmod +x /app/start_monitoring_container.sh

EXPOSE 8080
ENTRYPOINT ["/app/start_monitoring_container.sh"] 