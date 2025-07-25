"""
Configuration file for MLflow and other environment settings.
Update these values according to your deployment environment.
"""

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


# ----------------- MLflow Config -----------------


def get_mlflow_config():
    """
    Return a dictionary with MLflow configuration values.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable must be set")

    return {
        "tracking_uri": tracking_uri,
        "model_name": os.getenv("MLFLOW_MODEL_NAME", "MyTopModel"),
        "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "My_Model_Experiment"),
    }


# ----------------- S3 Config -----------------


def get_s3_config():
    """
    Return a dictionary with S3 configuration values for data storage and access.
    """
    bucket = os.getenv("S3_BUCKET_NAME")
    if not bucket:
        raise ValueError("S3_BUCKET_NAME environment variable must be set")

    return {
        "bucket_name": bucket,
        "raw_baseline_key": os.getenv("S3_RAW_BASELINE_KEY"),
        "new_data_key": os.getenv("S3_NEW_DATA_KEY"),
        "processed_data_key": os.getenv("S3_PROCESSED_DATA_KEY"),
        "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
        "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    }


# ----------------- Supabase Config -----------------


def get_supabase_config():
    """
    Return a dictionary with Supabase URL and key for database access.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

    return {
        "url": url,
        "key": key,
    }


# ----------------- Monitoring Config -----------------


def get_monitoring_config():
    """
    Return a dictionary with monitoring port and interval settings.
    """
    return {
        "port": int(os.getenv("MONITORING_PORT", "8080")),
        "interval": int(os.getenv("MONITORING_INTERVAL", "300")),
        "distance_feature_threshold": float(
            os.getenv("DISTANCE_FEATURE_THRESHOLD", "0.3")
        ),
    }
