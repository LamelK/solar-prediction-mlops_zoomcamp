# import mlflow
# import mlflow.sklearn
# from mlflow.models.signature import infer_signature

# """
# Logs trained models to MLflow with hyperparameters, metrics, and signatures.
# """

# def setup_mlflow(tracking_uri="http://localhost:5000", experiment_name="My_Model_Experiment"):
#     """
#     Configure MLflow tracking URI and experiment.

#     Args:
#         tracking_uri (str): URI of MLflow server.
#         experiment_name (str): Name of the experiment.
#     """
#     mlflow.set_tracking_uri(tracking_uri)
#     mlflow.set_experiment(experiment_name)


# def log_models_to_mlflow(all_runs, X_val):
#     """
#     Logs models, parameters, metrics, and signatures to MLflow.

#     Args:
#         all_runs (list): List of trained model runs with metadata.
#         X_val (pd.DataFrame): Validation data.

#     Returns:
#         list: Runs with MLflow run IDs appended.
#     """
#     logged_runs = []

#     for idx, run in enumerate(all_runs, 1):
#         with mlflow.start_run(run_name=f"{run['model_name']}_run_{idx}") as mlflow_run:
#             # Log parameters and metrics
#             mlflow.log_params(run['params'])
#             mlflow.log_metrics({'val_rmse': run['val_rmse'], 'val_r2': run['val_r2']})

#             # Infer model signature
#             input_data = X_val[run['features']]
#             predictions = run['model'].predict(input_data)
#             signature = infer_signature(input_data, predictions)

#             # Log model
#             mlflow.sklearn.log_model(
#                 sk_model=run['model'],
#                 name="model",
#                 signature=signature,
#                 input_example=input_data.iloc[:5]
#             )

#             logged_runs.append({
#                 'run_id': mlflow_run.info.run_id,
#                 **run
#             })

#     return logged_runs
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from prefect import task, get_run_logger



def setup_mlflow(tracking_uri="http://localhost:5000", experiment_name="My_Model_Experiment"):
    logger = get_run_logger()
    logger.info(f"Setting MLflow tracking URI to {tracking_uri} and experiment to '{experiment_name}'")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


@task(task_run_name="log_validation_metrics_to_mlflow")
def log_models_to_mlflow(all_runs, X_val):
    logger = get_run_logger()
    logger.info("Starting to log models to MLflow")

    logged_runs = []

    for idx, run in enumerate(all_runs, 1):
        logger.info(f"Logging {run['model_name']} run {idx} to MLflow")
        
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

            logger.info(f"Logged {run['model_name']} with run ID {mlflow_run.info.run_id}")

            logged_runs.append({
                'run_id': mlflow_run.info.run_id,
                **run
            })

    logger.info(f"Completed logging {len(logged_runs)} models to MLflow")
    return logged_runs
