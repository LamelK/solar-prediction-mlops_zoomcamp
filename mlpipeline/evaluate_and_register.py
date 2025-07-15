import json
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score
from prefect import task, flow, get_run_logger


@task(name="Evaluate Model on Test Set")
def evaluate_model_on_test(model, X_test, y_test):
    logger = get_run_logger()
    logger.info("Evaluating model on test set")

    predictions = model.predict(X_test)
    test_rmse = root_mean_squared_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)

    logger.info(f"Test RMSE: {test_rmse:.4f}, Test R2: {test_r2:.4f}")
    return {'test_rmse': test_rmse, 'test_r2': test_r2}


@task(name="Log Test Metrics to MLflow")
def log_test_metrics_to_mlflow(run_id, test_metrics):
    logger = get_run_logger()
    logger.info(f"Logging test metrics to MLflow for run_id: {run_id}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(test_metrics)
        mlflow.set_tag("evaluation_status", "tested_with_separate_script")


@task(name="Register Best Model")
def register_best_model(best_run, registry_model_name="MyTopModel"):
    logger = get_run_logger()
    logger.info(f"Registering best model: Run ID {best_run['run_id']} as {registry_model_name}")

    client = MlflowClient()
    model_uri = f"runs:/{best_run['run_id']}/model"
    registered_model = mlflow.register_model(model_uri, registry_model_name)
    version = registered_model.version

    tags = {
        "model_type": best_run['model_name'],
        "test_rmse": str(best_run['test_rmse']),
        "test_r2": str(best_run['test_r2']),
        "model_framework": best_run['model_name']
    }

    for k, v in tags.items():
        client.set_registered_model_tag(registry_model_name, k, v)
        client.set_model_version_tag(registry_model_name, version, k, v)

    client.set_registered_model_alias(registry_model_name, "champion", version)
    logger.info(f"Model registered as version {version} with alias 'champion'")
    return version


@flow(name="Evaluate and Register Best Model")
def evaluate_and_register(logged_runs, X_test, y_test):
    logger = get_run_logger()
    logger.info("Evaluating top 3 models on the test set")

    client = MlflowClient()
    top_3_runs = sorted(logged_runs, key=lambda x: x['val_rmse'])[:3]
    test_results = []

    for run in top_3_runs:
        logger.info(f"Loading model for run_id: {run['run_id']}")
        model_uri = f"runs:/{run['run_id']}/model"
        model = mlflow.sklearn.load_model(model_uri)
        X_test_subset = X_test[run['features']]

        test_metrics = evaluate_model_on_test(model, X_test_subset, y_test)
        test_results.append({
            'run_id': run['run_id'],
            'model_name': run['model_name'],
            **test_metrics
        })

        log_test_metrics_to_mlflow(run['run_id'], test_metrics)

    best_run = min(test_results, key=lambda x: x['test_rmse'])
    version = register_best_model(best_run)

    logger.info(f"Best model: {best_run['model_name']} | Run ID: {best_run['run_id']} | Version: {version}")
    return best_run, test_results
