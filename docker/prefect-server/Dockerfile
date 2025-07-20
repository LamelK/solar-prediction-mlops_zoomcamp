FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install only Prefect
COPY docker/prefect-server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy startup script
COPY docker/start_prefect_server.sh /app/start_prefect_server.sh
RUN chmod +x /app/start_prefect_server.sh

EXPOSE 4200
ENTRYPOINT ["/app/start_prefect_server.sh"] 