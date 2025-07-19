# MLflow EC2 Setup Guide

## Overview
Your MLflow server has been moved to an EC2 instance with SQLite backend and S3 artifacts storage. This document outlines the changes needed in your scripts.

## Required Environment Variables

Create a `.env` file in your project root with the following variables:

```bash
# MLflow Configuration
MLFLOW_TRACKING_URI=http://YOUR_EC2_PUBLIC_IP:5000
MLFLOW_EXPERIMENT_NAME=My_Model_Experiment

# S3 Configuration
S3_BUCKET_NAME=model-data-bucket-4821
S3_ARTIFACT_PREFIX=mlflow-artifacts/

# Model Configuration
MODEL_NAME=MyTopModel

# Supabase Configuration
SUPABASE_URL=https://ccfmfqtlizzbaxlshzbu.supabase.co
SUPABASE_KEY=your_supabase_key_here

# Data Configuration
RAW_BASELINE_KEY=raw-data/training_data.csv
NEW_DATA_KEY=raw-data/new_data/new_data.csv
PROCESSED_DATA_KEY=processed-data/training_data.csv

# AWS Configuration (if needed for local development)
AWS_REGION=us-west-1
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_DEFAULT_REGION=us-west-1

# Monitoring Configuration
DISTANCE_FEATURE_THRESHOLD=0.1
MONITORING_PORT=8080
MONITORING_INTERVAL=60

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Development/Environment Configuration
ENVIRONMENT=development
DEBUG=False
LOG_LEVEL=INFO
```

## Key Changes Made

### 1. Centralized Configuration (`config.py`)
- Created a comprehensive configuration file that loads all environment variables
- Provides helper functions for different configuration groups:
  - `get_mlflow_config()` - MLflow settings
  - `get_s3_config()` - S3 bucket and file paths
  - `get_aws_config()` - AWS credentials and region
  - `get_supabase_config()` - Supabase connection
  - `get_monitoring_config()` - Monitoring settings
  - `get_api_config()` - API server settings
  - `get_environment_config()` - Environment and debug settings
  - `get_all_config()` - All configurations combined
- Makes it easier to manage settings across different environments

### 2. Updated Files

#### `mlpipeline/model_logging.py`
- Added comment explaining the EC2 setup
- Uses environment variable `MLFLOW_TRACKING_URI` with localhost fallback

#### `api/serve_model.py`
- Updated to use centralized configuration
- Points to EC2 MLflow server instead of localhost
- Uses configuration for model name and Supabase settings

#### `pipeline.py`
- Updated to use centralized configuration
- Explicitly logs the MLflow tracking URI being used
- Passes configuration to setup_mlflow function

#### `retrain.py`
- Updated to use centralized configuration
- Removed direct environment variable access

#### `monitoring/monitor_drift.py`
- Updated to use centralized configuration
- Uses configuration for Supabase, S3, and monitoring settings

## How to Use the Configuration System

### Option 1: Import specific config functions
```python
from config import get_mlflow_config, get_s3_config

mlflow_config = get_mlflow_config()
s3_config = get_s3_config()

tracking_uri = mlflow_config["tracking_uri"]
bucket_name = s3_config["bucket_name"]
```

### Option 2: Import all configuration
```python
from config import get_all_config

config = get_all_config()
tracking_uri = config["mlflow"]["tracking_uri"]
bucket_name = config["s3"]["bucket_name"]
```

### Option 3: Import individual variables
```python
from config import MLFLOW_TRACKING_URI, S3_BUCKET_NAME

tracking_uri = MLFLOW_TRACKING_URI
bucket_name = S3_BUCKET_NAME
```

## How to Get Your EC2 Public IP

After running your Terraform deployment, you can get the EC2 public IP in several ways:

1. **From Terraform output:**
   ```bash
   cd terraform
   terraform output mlflow_server_public_ip
   ```

2. **From AWS Console:**
   - Go to EC2 Dashboard
   - Find your MLflow instance
   - Copy the Public IPv4 address

3. **From AWS CLI:**
   ```bash
   aws ec2 describe-instances --filters "Name=tag:Name,Values=mlflow-server" --query "Reservations[].Instances[].PublicIpAddress" --output text
   ```

## Testing the Setup

1. **Test MLflow Connection:**
   ```bash
   export MLFLOW_TRACKING_URI=http://YOUR_EC2_PUBLIC_IP:5000
   python -c "import mlflow; mlflow.set_tracking_uri('$MLFLOW_TRACKING_URI'); print('Connected successfully')"
   ```

2. **Test Your Pipeline:**
   ```bash
   python pipeline.py
   ```

3. **Test Your API:**
   ```bash
   python -m uvicorn api.serve_model:app --reload
   ```

4. **Test Monitoring:**
   ```bash
   python monitoring/monitor_drift.py
   ```

## Security Considerations

1. **EC2 Security Group:** Ensure your EC2 security group allows inbound traffic on port 5000 from your IP
2. **Network Access:** The MLflow UI will be accessible at `http://YOUR_EC2_PUBLIC_IP:5000`
3. **IAM Permissions:** The EC2 instance has IAM role permissions to access S3 for artifacts
4. **Environment Variables:** Keep your `.env` file secure and never commit it to version control

## Troubleshooting

### Common Issues:

1. **Connection Refused:**
   - Check if MLflow service is running on EC2: `sudo systemctl status mlflow`
   - Verify security group allows port 5000
   - Check if the EC2 instance is running

2. **S3 Access Issues:**
   - Verify IAM role permissions
   - Check S3 bucket name and prefix configuration

3. **Model Loading Issues:**
   - Ensure models are properly registered in MLflow
   - Check if the model name matches your configuration

4. **Configuration Issues:**
   - Verify all required environment variables are set in `.env`
   - Check that `config.py` is importing correctly
   - Ensure `python-dotenv` is installed: `pip install python-dotenv`

### Useful Commands:

```bash
# Check MLflow service status on EC2
ssh -i terraform/mlflow-key.pem ubuntu@YOUR_EC2_PUBLIC_IP "sudo systemctl status mlflow"

# View MLflow logs
ssh -i terraform/mlflow-key.pem ubuntu@YOUR_EC2_PUBLIC_IP "sudo journalctl -u mlflow -f"

# Restart MLflow service
ssh -i terraform/mlflow-key.pem ubuntu@YOUR_EC2_PUBLIC_IP "sudo systemctl restart mlflow"

# Test configuration loading
python -c "from config import get_all_config; print(get_all_config())"
```

## Migration Checklist

- [ ] Update your `.env` file with the correct EC2 public IP
- [ ] Add any missing environment variables from `env_template.txt`
- [ ] Test MLflow connection
- [ ] Run your pipeline to ensure models are logged correctly
- [ ] Test your API endpoints
- [ ] Verify models are accessible in MLflow UI
- [ ] Test retraining functionality
- [ ] Test monitoring functionality
- [ ] Verify all configuration is loading correctly 