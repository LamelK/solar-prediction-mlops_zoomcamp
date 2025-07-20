# Use Python 3.11 as base image
# This is a good balance between stability and features
FROM python:3.11-slim

# Set working directory in the container
# This is where our code will live inside the container
WORKDIR /app

# Install system dependencies
# These are needed for some Python packages to compile properly
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
# Docker caches layers, so if requirements don't change, this layer is reused
COPY requirements.txt .

# Install Python dependencies
# This creates a layer with all our Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
# This includes all your ML pipeline code, config files, etc.
COPY . .

# Create a startup script that will orchestrate everything
# This script will handle the Prefect server → worker → deployment sequence
COPY docker/start_ml_container.sh /app/start_ml_container.sh
RUN chmod +x /app/start_ml_container.sh

# Expose Prefect server port
# This allows you to access the Prefect UI from your browser
EXPOSE 4200

# Set the entry point to our startup script
# This is what runs when the container starts
ENTRYPOINT ["/app/start_ml_container.sh"] 