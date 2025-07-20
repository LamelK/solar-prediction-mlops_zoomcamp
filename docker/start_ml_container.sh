#!/bin/bash

# Exit on any error - this ensures the container stops if something goes wrong
set -e

echo "🚀 Starting ML Pipeline Container..."

# Function to check if Prefect server container is accessible
check_prefect_server() {
    echo "🔍 Checking if Prefect server container is accessible..."
    # Try to connect to the Prefect server container
    if curl -f http://prefect-server:4200/api/health > /dev/null 2>&1; then
        echo "✅ Prefect server container is accessible!"
        return 0
    else
        echo "⏳ Prefect server container not accessible yet..."
        return 1
    fi
}

# Function to wait for Prefect server container to be accessible
wait_for_prefect_server() {
    echo "🔄 Waiting for Prefect server container to be accessible..."
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if check_prefect_server; then
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts - waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "❌ Prefect server container not accessible within expected time"
    echo "⚠️  Make sure the prefect-server container is running"
    exit 1
}

# Function to check if worker is running
check_worker() {
    echo "🔍 Checking if worker is running..."
    # Check if worker process is running
    if pgrep -f "prefect worker" > /dev/null; then
        echo "✅ Worker is running!"
        return 0
    else
        echo "⏳ Worker not running yet..."
        return 1
    fi
}

# Function to wait for worker to be ready
wait_for_worker() {
    echo "🔄 Waiting for worker to be ready..."
    local max_attempts=10
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if check_worker; then
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts - waiting 5 seconds..."
        sleep 5
        attempt=$((attempt + 1))
    done
    
    echo "❌ Worker failed to start within expected time"
    exit 1
}

# Function to check if model is registered in MLflow
check_model_registered() {
    echo "🔍 Checking if model is registered in MLflow..."
    # Try to check if model exists in MLflow
    python3 -c "
import mlflow
from mlflow.tracking import MlflowClient
from config import get_mlflow_config
import sys

try:
    mlflow_config = get_mlflow_config()
    mlflow.set_tracking_uri(mlflow_config['tracking_uri'])
    
    client = MlflowClient()
    model_name = mlflow_config['model_name']
    
    # Check if model exists and has production version
    versions = client.search_model_versions(f\"name='{model_name}'\")
    production_versions = [v for v in versions if v.tags.get('status') == 'production']
    
    if production_versions:
        print(f'✅ Found production model: {model_name}')
        sys.exit(0)
    else:
        print(f'❌ No production model found for: {model_name}')
        sys.exit(1)
        
except Exception as e:
    print(f'❌ Error checking model: {e}')
    sys.exit(1)
"
    
    return $?
}

# Step 1: Wait for Prefect server container to be accessible
wait_for_prefect_server

# Step 2: Start Prefect worker (connects to Prefect server container)
echo "👷 Starting Prefect worker (ml-pipeline-worker)..."
echo "🔗 Connecting to Prefect server container at prefect-server:4200"
prefect worker start -p ml-pipeline-worker -n ml-pipeline-worker &
WORKER_PID=$!

# Step 3: Wait for worker to be ready
wait_for_worker

# Step 4: Run the ML pipeline deployment
echo "🔧 Running ML pipeline deployment..."
python pipeline.py

# Step 5: Wait for pipeline completion and model registration
echo "⏳ Waiting for pipeline completion and model registration..."
# Add a small delay to ensure everything is processed
sleep 30

# Step 6: Verify model is registered
if check_model_registered; then
    echo "✅ Model successfully registered in MLflow!"
else
    echo "⚠️  Warning: Could not verify model registration"
fi

echo "🎉 ML Pipeline Container is ready!"
echo "👷 Worker name: ml-pipeline-worker"
echo "📊 Prefect UI available at: http://localhost:4200"

# Keep the container running
# This is important - without this, the container would exit
echo "🔄 Container is running. Press Ctrl+C to stop."
wait $WORKER_PID 