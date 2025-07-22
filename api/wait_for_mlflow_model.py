import os
import time
from mlflow.tracking import MlflowClient
import subprocess
import sys
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    print("MLFLOW_TRACKING_URI environment variable must be set", file=sys.stderr)
    sys.exit(1)

MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "MyTopModel")
MLFLOW_MODEL_STAGE = os.getenv(
    "MLFLOW_MODEL_STAGE", "production"
)  # now matches tag value
API_CMD = os.getenv("API_START_CMD", "uvicorn main:app --host 0.0.0.0 --port 8000")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

print(
    f"Waiting for model '{MLFLOW_MODEL_NAME}' with tag status='{MLFLOW_MODEL_STAGE}'"
    f"to be available in MLflow at {MLFLOW_TRACKING_URI}..."
)

while True:
    try:
        versions = client.search_model_versions(f"name='{MLFLOW_MODEL_NAME}'")
        found = False
        for v in versions:
            if v.tags.get("status", "").lower() == MLFLOW_MODEL_STAGE.lower():
                found = True
                break
        if found:
            print(
                f"Model with tag status='{MLFLOW_MODEL_STAGE}' found in MLflow! "
                f"Starting API server..."
            )
            break
        else:
            print(
                f"Model with tag status='{MLFLOW_MODEL_STAGE}' not found. "
                f"Waiting 60 seconds..."
            )
    except Exception as e:
        print(f"Error querying MLflow: {e}. Retrying in 60 seconds...")
    time.sleep(60)

# Start the API server
print("Starting API with command:", API_CMD)
subprocess.run(API_CMD.split())
