import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from prefect import task, get_run_logger
import os
from dotenv import load_dotenv

load_dotenv()


@task(name="Setup MLflow")
def setup_mlflow(tracking_uri=None, experiment_name="My_Model_Experiment"):
    logger = get_run_logger()
    if tracking_uri is None:
        # Use EC2 MLflow server if available, fallback to localhost for development
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.info(f"Setting MLflow tracking URI: {tracking_uri} and experiment: {experiment_name}")
    
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    logger.info("MLflow configured successfully")


@task(name="Log Models to MLflow")
def log_models_to_mlflow(all_runs, X_val):
    logger = get_run_logger()
    logger.info(f"Logging {len(all_runs)} models to MLflow")

    logged_runs = []

    for idx, run in enumerate(all_runs, 1):
        logger.info(f"Logging model {idx}: {run['model_name']} with params {run['params']}")
        with mlflow.start_run(run_name=f"{run['model_name']}_run_{idx}") as mlflow_run:
            mlflow.log_params(run['params'])
            mlflow.log_metrics({'val_rmse': run['val_rmse'], 'val_r2': run['val_r2']})

            input_data = X_val[run['features']]
            predictions = run['model'].predict(input_data)
            signature = infer_signature(input_data, predictions)

            mlflow.sklearn.log_model(
                sk_model=run['model'],
                name="model",
                signature=signature,
                input_example=input_data.iloc[:5]
            )

            run_id = mlflow_run.info.run_id
            logger.info(f"Model logged with MLflow run ID: {run_id}")

            logged_runs.append({
                'run_id': run_id,
                **run
            })

    logger.info("All models logged to MLflow")
    return logged_runs
