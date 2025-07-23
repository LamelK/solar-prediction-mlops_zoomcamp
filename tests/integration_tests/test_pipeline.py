from unittest.mock import patch, MagicMock
from pipeline import main
import pytest


@patch("pipeline.get_s3_config")
@patch("pipeline.get_mlflow_config")
@patch("pipeline.load_and_prepare_data")
@patch("pipeline.load_data_s3")
@patch("pipeline.train_tune_models")
@patch("pipeline.setup_mlflow")
@patch("pipeline.log_models_to_mlflow")
@patch("pipeline.evaluate_and_register")
@pytest.mark.integration
def test_main_flow(
    mock_evaluate_register,
    mock_log_models,
    mock_setup_mlflow,
    mock_train_tune,
    mock_load_data_s3,
    mock_load_prepare,
    mock_mlflow_config,
    mock_s3_config,
):
    # Setup mocks
    mock_s3_config.return_value = {
        "bucket_name": "test-bucket",
        "raw_baseline_key": "raw-data.csv",
        "processed_data_key": "processed-data.csv",
    }
    mock_mlflow_config.return_value = {
        "tracking_uri": "http://localhost:5000",
        "experiment_name": "TestExperiment",
    }
    mock_load_data_s3.return_value = MagicMock(shape=(100, 10))
    mock_train_tune.return_value = (["run1", "run2"], "X_val", "X_test", "y_test")
    mock_log_models.return_value = ["logged_run1", "logged_run2"]
    mock_evaluate_register.return_value = ("best_run", {"accuracy": 0.9})

    # Run main flow
    main()

    # Assertions
    mock_load_prepare.assert_called_once()
    mock_load_data_s3.assert_called_once()
    mock_train_tune.assert_called_once()
    mock_setup_mlflow.assert_called_once()
    mock_log_models.assert_called_once()
    mock_evaluate_register.assert_called_once()
