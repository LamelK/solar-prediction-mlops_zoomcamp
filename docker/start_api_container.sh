#!/bin/bash

# Exit on any error
set -e

echo "🚀 Starting API Container..."

# Function to check if model is available in MLflow
check_model_availability() {
    echo "🔍 Checking if model is available in MLflow..."
    
    # Try to load the model using Python
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

# Function to wait for model to be available
wait_for_model() {
    echo "🔄 Waiting for model to be available in MLflow..."
    local max_attempts=60  # 10 minutes with 10-second intervals
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if check_model_availability; then
            echo "✅ Model is available!"
            return 0
        fi
        
        echo "Attempt $attempt/$max_attempts - waiting 10 seconds..."
        sleep 10
        attempt=$((attempt + 1))
    done
    
    echo "❌ Model not available within expected time"
    echo "⚠️  Make sure the ML container has completed training and model registration"
    exit 1
}

# Step 1: Wait for model to be available
wait_for_model

# Step 2: Start FastAPI service
echo "🌐 Starting FastAPI service..."
echo "📊 API will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"

# Start uvicorn server
# --host 0.0.0.0 allows external connections
# --port 8000 matches the exposed port
uvicorn api.serve_model:app --host 0.0.0.0 --port 8000 