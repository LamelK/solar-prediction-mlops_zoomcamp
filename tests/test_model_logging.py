import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor

from mlpipeline.model_logging import log_models_to_mlflow  # replace 'your_module'

@pytest.fixture
def dummy_all_runs():
    model = RandomForestRegressor()
    X_dummy = pd.DataFrame(np.random.rand(10, 3), columns=['A', 'B', 'C'])
    y_dummy = np.random.rand(10)
    model.fit(X_dummy, y_dummy)

    return [{
        'model_name': 'RandomForest',
        'params': {'n_estimators': 10},
        'features': ['A', 'B', 'C'],
        'val_rmse': 0.5,
        'val_r2': 0.8,
        'model': model
    }], X_dummy

@patch("mlpipeline.model_logging.get_run_logger")
@patch("mlpipeline.model_logging.mlflow")
def test_log_models_to_mlflow(mock_mlflow, mock_logger, dummy_all_runs):
    mock_logger.return_value = MagicMock()
    mock_run = MagicMock()
    mock_run.info.run_id = "test-run-id"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    all_runs, X_val = dummy_all_runs
    logged_runs = log_models_to_mlflow.fn(all_runs, X_val)

    # Assert mlflow log_params and log_metrics called
    mock_mlflow.log_params.assert_called_with(all_runs[0]['params'])
    mock_mlflow.log_metrics.assert_called()

    # Assert model logged
    mock_mlflow.sklearn.log_model.assert_called()

    # Check run ID injected into result
    assert logged_runs[0]['run_id'] == "test-run-id"
