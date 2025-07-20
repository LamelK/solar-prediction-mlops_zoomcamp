#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting Prefect Server..."

# Start Prefect server
echo "📊 Starting Prefect server on port 4200..."
prefect server start --host 0.0.0.0 --port 4200 