import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from mlpipeline.model_training import train_tune_models  # Replace 'your_module' with actual module name

@pytest.fixture
def dummy_df():
    np.random.seed(42)
    data = {
        'Temperature': np.random.uniform(20, 30, 100),
        'Pressure': np.random.uniform(1000, 1020, 100),
        'Humidity': np.random.uniform(30, 70, 100),
        'WindDirection_Degrees': np.random.uniform(0, 360, 100),
        'Speed': np.random.uniform(0, 10, 100),
        'Hour_sin': np.random.rand(100),
        'Hour_cos': np.random.rand(100),
        'Minute_sin': np.random.rand(100),
        'Minute_cos': np.random.rand(100),
        'Day_sin': np.random.rand(100),
        'Day_cos': np.random.rand(100),
        'Month_sin': np.random.rand(100),
        'Month_cos': np.random.rand(100),
        'Weekday_sin': np.random.rand(100),
        'Weekday_cos': np.random.rand(100),
        'MinutesSinceSunrise': np.random.uniform(0, 720, 100),
        'MinutesUntilSunset': np.random.uniform(0, 720, 100),
        'Radiation': np.random.uniform(0, 1000, 100)
    }
    return pd.DataFrame(data)

@patch("mlpipeline.model_training.get_run_logger")
def test_train_tune_models(mock_logger, dummy_df):
    mock_logger.return_value = MagicMock()

    results, X_val, X_test, y_test = train_tune_models.fn(dummy_df)

    # Assert outputs are not empty
    assert len(results) > 0
    assert not X_val.empty
    assert not X_test.empty
    assert len(y_test) == len(X_test)

    # Check keys in each result
    expected_keys = {'model_name', 'params', 'features', 'val_rmse', 'val_r2', 'model'}
    for run in results:
        assert expected_keys.issubset(run.keys())
        assert isinstance(run['val_rmse'], float)
        assert isinstance(run['val_r2'], float)
        assert hasattr(run['model'], 'predict')
