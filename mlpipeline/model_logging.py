import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from prefect import task, get_run_logger
import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


def validate_mlflow_connection(tracking_uri, timeout=30):
    """
    Validate MLflow connection with timeout to prevent hanging.
    """
    try:
        # Test connection to MLflow server
        response = requests.get(f"{tracking_uri}/health", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@task(name="Setup MLflow", retries=1, retry_delay_seconds=10)
def setup_mlflow(tracking_uri=None, experiment_name=None, timeout=30):
    """
    Configures MLflow tracking URI and experiment name.
    Uses environment variable if tracking_uri is not provided.
    Includes timeout validation to prevent hanging on invalid URLs.
    """
    logger = get_run_logger()
    if tracking_uri is None:
        # Use tracking URI from environment if not provided
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if not tracking_uri:
            raise ValueError("MLFLOW_TRACKING_URI environment variable must be set")

    logger.info(
        f"Setting MLflow tracking URI: {tracking_uri} and experiment: {experiment_name}"
    )

    # Validate MLflow connection before proceeding
    logger.info("Validating MLflow connection...")
    if not validate_mlflow_connection(tracking_uri, timeout):
        raise ConnectionError(
            f"Failed to connect to MLflow server at {tracking_uri} "
            f"within {timeout} seconds. "
            f"Please check the URL and ensure the MLflow server is running."
        )

    # Set MLflow tracking URI and experiment with timeout protection
    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow configured successfully")
    except Exception as e:
        raise ConnectionError(
            f"Failed to configure MLflow: {str(e)}. "
            f"Please check the tracking URI and experiment name."
        )


@task(name="Log Models to MLflow", retries=1, retry_delay_seconds=10)
def log_models_to_mlflow(all_runs, X_val):
    """
    Logs a list of trained models and their parameters/metrics to MLflow.
    Returns a list of run metadata for each logged model.
    """
    logger = get_run_logger()
    logger.info(f"Logging {len(all_runs)} models to MLflow")

    logged_runs = []  # Store metadata for each logged run

    for idx, run in enumerate(all_runs, 1):
        logger.info(
            f"Logging model {idx}: {run['model_name']} with params {run['params']}"
        )
        # Start a new MLflow run for each model
        with mlflow.start_run(run_name=f"{run['model_name']}_run_{idx}") as mlflow_run:
            # Log model parameters and validation metrics
            mlflow.log_params(run["params"])
            mlflow.log_metrics({"val_rmse": run["val_rmse"], "val_r2": run["val_r2"]})

            # Prepare input data and signature for model logging
            input_data = X_val[run["features"]]
            predictions = run["model"].predict(input_data)
            signature = infer_signature(input_data, predictions)

            # Log the model to MLflow with input example and signature
            mlflow.sklearn.log_model(
                sk_model=run["model"],
                name="model",
                signature=signature,
                input_example=input_data.iloc[:5],
            )

            run_id = mlflow_run.info.run_id
            logger.info(f"Model logged with MLflow run ID: {run_id}")

            # Store run metadata for later use (e.g., evaluation/registration)
            logged_runs.append({"run_id": run_id, **run})

    logger.info("All models logged successfully")
    return logged_runs
