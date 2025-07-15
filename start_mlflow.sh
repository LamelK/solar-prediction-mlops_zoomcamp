#!/bin/bash

# Start MLflow server with backend and artifact store outside the repo
# mkdir -p /tmp/mlflow_data/mlruns
mlflow server \
  --backend-store-uri "sqlite:////tmp/mlflow_data/mlflow.db" \
  --default-artifact-root "file:///tmp/mlflow_data/mlruns" \
  --host 0.0.0.0 \
  --port 5000

