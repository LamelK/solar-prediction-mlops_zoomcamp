# import logging
# import sys
# from mlpipeline.data_preparation import load_and_prepare_data
# from mlpipeline.feature_selection import build_baseline_models
# from mlpipeline.model_training import train_tune_models
# from mlpipeline.model_logging import log_models_to_mlflow, setup_mlflow
# from mlpipeline.evaluate_and_register import evaluate_and_register

# def setup_logging():
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[logging.StreamHandler(sys.stdout)]
#     )

# def main():
#     setup_logging()
#     setup_mlflow()
#     logger = logging.getLogger(__name__)

#     DATA_PATH = './data/data.csv'  # could add CLI override if desired

#     try:
#         logger.info("Starting data preparation...")
#         df = load_and_prepare_data(DATA_PATH)
#         logger.info(f"Data prepared: {df.shape[0]} rows, {df.shape[1]} columns")

#         logger.info("Starting baseline model training...")
#         selected_features_per_model = build_baseline_models(df)
#         logger.info(f"Baseline models trained. Selected features: {selected_features_per_model}")

#         logger.info("Starting model tuning...")
#         all_runs, X_val, X_test, y_test = train_tune_models(df, selected_features_per_model)
#         logger.info(f"Model tuning completed. Total runs: {len(all_runs)}")

#         # Extract validation set from tune_models or return it from there for correct signature logging
#         # For now, passing full df; ideally modify tune_models to return X_val subset for logging
#         logger.info("Logging models to MLflow...")
#         logged_runs = log_models_to_mlflow(all_runs, X_val)  
#         logger.info(f"Logged {len(logged_runs)} runs to MLflow.")

#         logger.info("Evaluating and registering best model...")
#         best_run, test_results = evaluate_and_register(logged_runs, X_test, y_test)
#         logger.info(f"Best model registered: {best_run}")

#         logger.info("Pipeline completed successfully.")

#     except Exception as e:
#         logger.exception(f"Pipeline failed: {e}")
#         sys.exit(1)

# if __name__ == '__main__':
#     main()

from prefect import flow, get_run_logger
from mlpipeline.data_preparation import load_and_prepare_data
from mlpipeline.feature_selection import build_baseline_models
from mlpipeline.model_training import train_tune_models
from mlpipeline.model_logging import log_models_to_mlflow, setup_mlflow
from mlpipeline.evaluate_and_register import evaluate_and_register

@flow(name="ML Pipeline Main Flow")
def main_flow():
    logger = get_run_logger()
    setup_mlflow()

    DATA_PATH = './data/data.csv'  # could add CLI override if desired

    logger.info("Starting data preparation...")
    df = load_and_prepare_data(DATA_PATH)

    logger.info("Starting baseline model training...")
    selected_features_per_model = build_baseline_models(df)
    logger.info(f"Baseline models trained. Selected features: {selected_features_per_model}")

    logger.info("Starting model tuning...")
    all_runs, X_val, X_test, y_test = train_tune_models(df, selected_features_per_model)
    logger.info(f"Model tuning completed. Total runs: {len(all_runs)}")

    logger.info("Logging models to MLflow...")
    logged_runs = log_models_to_mlflow(all_runs, X_val)
    logger.info(f"Logged {len(logged_runs)} runs to MLflow.")

    logger.info("Evaluating and registering best model...")
    best_run, test_results = evaluate_and_register(logged_runs, X_test, y_test)
    logger.info(f"Best model registered: {best_run}")

    logger.info("Pipeline completed successfully.")

if __name__ == '__main__':
    main_flow()
