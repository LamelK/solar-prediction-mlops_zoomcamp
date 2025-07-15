#!/bin/bash

mlflow server \
  --backend-store-uri "file://$(pwd)/mlflow_server/mlruns" \
  --default-artifact-root "file://$(pwd)/mlflow_server/mlruns" \
  --host 0.0.0.0 \
  --port 5000
