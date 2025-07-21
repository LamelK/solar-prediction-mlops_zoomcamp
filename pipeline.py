from mlpipeline.data_preparation import load_and_prepare_data, load_data_s3
from mlpipeline.model_training import train_tune_models
from mlpipeline.model_logging import log_models_to_mlflow, setup_mlflow
from mlpipeline.evaluate_and_register import evaluate_and_register
from prefect import flow, get_run_logger
import os
from config import get_s3_config, get_mlflow_config

@flow(name="ML Pipeline")
def main(bucket_name=None, raw_key=None, processed_key=None):
    """
    Main pipeline flow for data preparation, model training, logging, and evaluation.
    Accepts optional S3 bucket and key overrides.
    """
    logger = get_run_logger()
    
    # Retrieve configuration for S3 and MLflow
    s3_config = get_s3_config()
    mlflow_config = get_mlflow_config()
    
    # Use provided parameters or fall back to configuration
    bucket = bucket_name or s3_config["bucket_name"]
    raw_key = raw_key or s3_config["raw_baseline_key"]
    processed_key = processed_key or s3_config["processed_data_key"]
    
    if not bucket:
        raise ValueError("S3 bucket name must be provided as an argument or in the S3_BUCKET_NAME environment variable.")

    # Step 1: Preprocess raw data and save processed data to S3
    logger.info("Running data preparation...")
    logger.info(f"Using raw data from: s3://{bucket}/{raw_key}")
    load_and_prepare_data(file_key=raw_key, bucket_name=bucket)

    # Step 2: Load processed data from S3 for model training
    logger.info("Loading processed data from S3 for model training...")
    logger.info(f"Loading from: s3://{bucket}/{processed_key}")
    df = load_data_s3(bucket, processed_key)

    # Step 3: Model training and subsequent steps
    logger.info(f"Data prepared: {df.shape[0]} rows, {df.shape[1]} columns")

    logger.info("Training and tuning models...")
    all_runs, X_val, X_test, y_test = train_tune_models(df)
    logger.info(f"Model tuning completed. Total runs: {len(all_runs)}")

    # Set up MLflow tracking and experiment
    logger.info("Setting up MLflow...")
    logger.info(f"MLflow tracking URI: {mlflow_config['tracking_uri']}")
    setup_mlflow(tracking_uri=mlflow_config['tracking_uri'], experiment_name=mlflow_config['experiment_name'])

    # Log models to MLflow
    logger.info("Logging models to MLflow...")
    logged_runs = log_models_to_mlflow(all_runs, X_val)
    logger.info(f"Logged {len(logged_runs)} runs to MLflow.")

    # Evaluate and register the best model
    logger.info("Evaluating and registering best model...")
    best_run, test_results = evaluate_and_register(logged_runs, X_test, y_test)
    logger.info(f"Best model registered: {best_run}")
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    # Entry point for running the pipeline directly
    main()
