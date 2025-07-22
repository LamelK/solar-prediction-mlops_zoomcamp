"""
Configuration file for MLflow and other environment settings.
Update these values according to your deployment environment.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
if not MLFLOW_TRACKING_URI:
    # Ensure tracking URI is set for MLflow
    raise ValueError("MLFLOW_TRACKING_URI environment variable must be set")

# Name of the model to use in MLflow
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "MyTopModel")
# Name of the experiment in MLflow
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "My_Model_Experiment")

# S3 configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
if not S3_BUCKET_NAME:
    # Ensure S3 bucket name is set
    raise ValueError("S3_BUCKET_NAME environment variable must be set")

# S3 key for the raw baseline data
S3_RAW_BASELINE_KEY = os.getenv("S3_RAW_BASELINE_KEY", "data/raw_baseline.csv")
# S3 key for the new data
S3_NEW_DATA_KEY = os.getenv("S3_NEW_DATA_KEY", "data/new_data.csv")
# S3 key for the processed data
S3_PROCESSED_DATA_KEY = os.getenv("S3_PROCESSED_DATA_KEY", "data/processed_data.csv")

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
if not SUPABASE_URL:
    # Ensure Supabase URL is set
    raise ValueError("SUPABASE_URL environment variable must be set")

SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_KEY:
    # Ensure Supabase key is set
    raise ValueError("SUPABASE_KEY environment variable must be set")

# AWS configuration for S3 access
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

# Monitoring configuration
MONITORING_PORT = int(os.getenv("MONITORING_PORT", "8080"))
MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "300"))  # 5 minutes


def get_mlflow_config():
    """
    Return a dictionary with MLflow configuration values.
    """
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "model_name": MLFLOW_MODEL_NAME,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
    }


def get_s3_config():
    """
    Return a dictionary with S3 configuration values for data storage and access.
    """
    return {
        "bucket_name": S3_BUCKET_NAME,
        "raw_baseline_key": S3_RAW_BASELINE_KEY,
        "processed_data_key": S3_PROCESSED_DATA_KEY,
        "new_data_key": S3_NEW_DATA_KEY,
        "access_key_id": AWS_ACCESS_KEY_ID,
        "secret_access_key": AWS_SECRET_ACCESS_KEY,
        "region": AWS_DEFAULT_REGION,
    }


def get_supabase_config():
    """
    Return a dictionary with Supabase URL and key for database access.
    """
    return {"url": SUPABASE_URL, "key": SUPABASE_KEY}


def get_monitoring_config():
    """
    Return a dictionary with monitoring port and interval settings.
    """
    return {"port": MONITORING_PORT, "interval": MONITORING_INTERVAL}
