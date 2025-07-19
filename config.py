"""
Configuration file for MLflow and other environment settings.
Update these values according to your deployment environment.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "My_Model_Experiment")

# S3 Configuration
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "model-data-bucket-4821")
S3_ARTIFACT_PREFIX = os.getenv("S3_ARTIFACT_PREFIX", "mlflow-artifacts/")

# Model Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "MyTopModel")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://ccfmfqtlizzbaxlshzbu.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Data Configuration
RAW_BASELINE_KEY = os.getenv("RAW_BASELINE_KEY", "raw-data/training_data.csv")
NEW_DATA_KEY = os.getenv("NEW_DATA_KEY", "raw-data/new_data/new_data.csv")
PROCESSED_DATA_KEY = os.getenv("PROCESSED_DATA_KEY", "processed-data/training_data.csv")

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-west-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-1")

# Monitoring Configuration
DISTANCE_FEATURE_THRESHOLD = float(os.getenv("DISTANCE_FEATURE_THRESHOLD", "0.1"))
MONITORING_PORT = int(os.getenv("MONITORING_PORT", "8080"))
MONITORING_INTERVAL = int(os.getenv("MONITORING_INTERVAL", "3600"))  # seconds (1 hour)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("API_RELOAD", "True").lower() == "true"

# Development/Environment Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")  # development, staging, production
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

def get_mlflow_config():
    """Get MLflow configuration dictionary."""
    return {
        "tracking_uri": MLFLOW_TRACKING_URI,
        "experiment_name": MLFLOW_EXPERIMENT_NAME,
        "model_name": MODEL_NAME
    }

def get_s3_config():
    """Get S3 configuration dictionary."""
    return {
        "bucket_name": S3_BUCKET_NAME,
        "artifact_prefix": S3_ARTIFACT_PREFIX,
        "raw_baseline_key": RAW_BASELINE_KEY,
        "new_data_key": NEW_DATA_KEY,
        "processed_data_key": PROCESSED_DATA_KEY
    }

def get_aws_config():
    """Get AWS configuration dictionary."""
    return {
        "region": AWS_REGION,
        "access_key_id": AWS_ACCESS_KEY_ID,
        "secret_access_key": AWS_SECRET_ACCESS_KEY,
        "default_region": AWS_DEFAULT_REGION
    }

def get_supabase_config():
    """Get Supabase configuration dictionary."""
    return {
        "url": SUPABASE_URL,
        "key": SUPABASE_KEY
    }

def get_monitoring_config():
    """Get monitoring configuration dictionary."""
    return {
        "distance_feature_threshold": DISTANCE_FEATURE_THRESHOLD,
        "port": MONITORING_PORT,
        "interval": MONITORING_INTERVAL
    }

def get_api_config():
    """Get API configuration dictionary."""
    return {
        "host": API_HOST,
        "port": API_PORT,
        "reload": API_RELOAD
    }

def get_environment_config():
    """Get environment configuration dictionary."""
    return {
        "environment": ENVIRONMENT,
        "debug": DEBUG,
        "log_level": LOG_LEVEL
    }

def get_all_config():
    """Get all configuration as a single dictionary."""
    return {
        "mlflow": get_mlflow_config(),
        "s3": get_s3_config(),
        "aws": get_aws_config(),
        "supabase": get_supabase_config(),
        "monitoring": get_monitoring_config(),
        "api": get_api_config(),
        "environment": get_environment_config()
    } 