import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from mlpipeline.evaluate_and_register import evaluate_and_register  # replace your_module with actual filename

@pytest.fixture
def dummy_logged_runs():
    return [
        {'run_id': 'run_1', 'model_name': 'RandomForest', 'val_rmse': 0.5, 'features': ['f1', 'f2']},
        {'run_id': 'run_2', 'model_name': 'GradientBoosting', 'val_rmse': 0.4, 'features': ['f1', 'f2']},
        {'run_id': 'run_3', 'model_name': 'KNN', 'val_rmse': 0.6, 'features': ['f1', 'f2']}
    ]

@pytest.fixture
def dummy_test_data():
    X_test = pd.DataFrame(np.random.rand(10, 2), columns=['f1', 'f2'])
    y_test = np.random.rand(10)
    return X_test, y_test

@patch("mlpipeline.evaluate_and_register.get_run_logger")
@patch("mlpipeline.evaluate_and_register.register_best_model")
@patch("mlpipeline.evaluate_and_register.log_test_metrics_to_mlflow")
@patch("mlpipeline.evaluate_and_register.evaluate_model_on_test")
@patch("mlpipeline.evaluate_and_register.mlflow")
@patch("mlpipeline.evaluate_and_register.MlflowClient")
def test_evaluate_and_register(
    mock_client, mock_mlflow, mock_eval, mock_log_metrics, mock_register, mock_logger,
    dummy_logged_runs, dummy_test_data
):
    mock_logger.return_value = MagicMock()

    # Mock loaded model from mlflow
    mock_model = MagicMock()
    mock_model.predict.return_value = np.random.rand(10)
    mock_mlflow.sklearn.load_model.return_value = mock_model

    # Mock test metrics evaluation
    mock_eval.return_value = {'test_rmse': 0.3, 'test_r2': 0.9}
    mock_register.return_value = 2  # pretend registered model version is 2

    X_test, y_test = dummy_test_data
    best_run, test_results = evaluate_and_register.fn(dummy_logged_runs, X_test, y_test)

    # Assertions
    assert best_run['test_rmse'] == 0.3
    assert best_run['test_r2'] == 0.9
    assert len(test_results) == 3  # since top_3_runs is sliced
    mock_mlflow.sklearn.load_model.assert_called()

    # Ensure register_best_model is called with best_run
    mock_register.assert_called_once()
    # Ensure metrics logged for all 3 models
    assert mock_log_metrics.call_count == 3
    # Ensure evaluate_model_on_test is called for each model
    assert mock_eval.call_count == 3
