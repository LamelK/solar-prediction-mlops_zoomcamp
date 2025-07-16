from mlpipeline.data_preparation import load_and_prepare_data
from mlpipeline.model_training import train_tune_models
from mlpipeline.model_logging import log_models_to_mlflow, setup_mlflow
from mlpipeline.evaluate_and_register import evaluate_and_register
from prefect import flow, get_run_logger


@flow(name="ML Pipeline")
def main():
    logger = get_run_logger()
    DATA_PATH = './data/training_data.csv'

    logger.info("Starting data preparation...")
    df = load_and_prepare_data(DATA_PATH)
    logger.info(f"Data prepared: {df.shape[0]} rows, {df.shape[1]} columns")

    logger.info("Training and tuning models...")
    all_runs, X_val, X_test, y_test = train_tune_models(df)
    logger.info(f"Model tuning completed. Total runs: {len(all_runs)}")

    logger.info("Setting up MLflow...")
    setup_mlflow()

    logger.info("Logging models to MLflow...")
    logged_runs = log_models_to_mlflow(all_runs, X_val)
    logger.info(f"Logged {len(logged_runs)} runs to MLflow.")

    logger.info("Evaluating and register best model...")
    best_run, test_results = evaluate_and_register(logged_runs, X_test, y_test)
    logger.info(f"Best model registered: {best_run}")
    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()
