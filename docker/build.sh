#!/bin/bash

# Build script for optimized Docker setup
# Builds base image first, then all services

set -e

echo "🏗️  Building Solar Prediction MLOps Docker Images..."

# Step 1: Build base image first
echo "📦 Building base image..."
docker-compose --profile build-only build base

# Step 2: Build all services (they depend on base image)
echo "🔨 Building all services..."
docker-compose build

echo "✅ All images built successfully!"
echo "📊 Check image sizes:"
docker images | grep solar 