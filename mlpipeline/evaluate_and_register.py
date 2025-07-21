import json
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from prefect import task, flow, get_run_logger

@task(name="Evaluate Model on Test Set")
def evaluate_model_on_test(model, X_test, y_test):
    """
    Evaluate a trained model on the test set and return RMSE and R2 metrics.
    """
    logger = get_run_logger()
    logger.info("Evaluating model on test set")

    # Generate predictions for the test set
    predictions = model.predict(X_test)
    # Calculate RMSE and R2 metrics
    test_rmse = root_mean_squared_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)

    logger.info(f"Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")
    return {'test_rmse': test_rmse, 'test_r2': test_r2}

@task(name="Log Test Metrics to MLflow")
def log_test_metrics_to_mlflow(run_id, test_metrics):
    """
    Log test set metrics to MLflow for a given run ID.
    """
    logger = get_run_logger()
    logger.info(f"Logging test metrics to MLflow for run_id: {run_id}")

    # Log metrics to the specified MLflow run
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(test_metrics)
        # Tag the run to indicate it was evaluated with this script
        mlflow.set_tag("evaluation_status", "tested_with_separate_script")

@task(name="Register Best Model")
def register_best_model(best_run, registry_model_name="MyTopModel"):
    """
    Register the best model in MLflow Model Registry and set tags for production status.
    Archives previous production versions.
    """
    logger = get_run_logger()
    logger.info(f"Registering best model: Run ID {best_run['run_id']} as {registry_model_name}")

    client = MlflowClient()
    # Build the model URI for MLflow
    model_uri = f"runs:/{best_run['run_id']}/model"
    # Register the model in the MLflow Model Registry
    registered_model = mlflow.register_model(model_uri, registry_model_name)
    version = registered_model.version

    # Set tags for the registered model and version
    tags = {
        "model_type": best_run['model_name'],
        "test_rmse": str(best_run['test_rmse']),
        "model_framework": best_run['model_name'],
        "status": "production"  # mark this version as production
    }

    # Set tags for both the registered model and the specific version
    for k, v in tags.items():
        client.set_registered_model_tag(registry_model_name, k, v)
        client.set_model_version_tag(registry_model_name, version, k, v)

    # Set alias for the current production version
    client.set_registered_model_alias(registry_model_name, "champion", version)
    
    # Archive old production versions by changing their status tag
    for mv in client.search_model_versions(f"name='{registry_model_name}'"):
        if mv.version != version and mv.tags.get("status") == "production":
            client.set_model_version_tag(registry_model_name, mv.version, "status", "archived")

    logger.info(f"Model registered as version {version} and marked as production")
    return version

@flow(name="Evaluate and Register Best Model")
def evaluate_and_register(logged_runs, X_test, y_test):
    """
    Evaluate the models on the test set, log their metrics, and register the best one in the model registry.
    Returns the best run and all test results.
    """
    logger = get_run_logger()
    logger.info("Evaluating top 3 models on the test set")

    client = MlflowClient()
    # Select top 3 runs based on validation RMSE
    top_3_runs = sorted(logged_runs, key=lambda x: x['val_rmse'])[:3]
    test_results = []

    # Evaluate each of the models on the test set
    for run in top_3_runs:
        logger.info(f"Loading model for run_id: {run['run_id']}")
        # Load the model from MLflow using the run ID
        model_uri = f"runs:/{run['run_id']}/model"
        model = mlflow.sklearn.load_model(model_uri)
        # Use only the features used during training
        X_test_subset = X_test[run['features']]

        # Evaluate and log test metrics
        test_metrics = evaluate_model_on_test(model, X_test_subset, y_test)
        test_results.append({
            'run_id': run['run_id'],
            'model_name': run['model_name'],
            **test_metrics
        })

        # Log test metrics to MLflow for this run
        log_test_metrics_to_mlflow(run['run_id'], test_metrics)

    # Select the best run based on test RMSE
    best_run = min(test_results, key=lambda x: x['test_rmse'])
    # Register the best model in the registry
    version = register_best_model(best_run)

    logger.info(f"Best model: {best_run['model_name']} | Run ID: {best_run['run_id']} | Version: {version}")
    return best_run, test_results
