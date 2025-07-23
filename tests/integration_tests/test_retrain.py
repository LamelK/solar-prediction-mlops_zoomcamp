import pytest
from unittest.mock import patch, MagicMock
import importlib


@pytest.fixture
def retrain_flow_with_mocked_config():
    with patch("retrain.get_s3_config") as mock_s3_config:
        mock_s3_config.return_value = {
            "bucket_name": "test-bucket",
            "raw_baseline_key": "baseline.csv",
            "new_data_key": "new_data.csv",
        }
        import retrain

        importlib.reload(retrain)
        return retrain.retrain_on_drift_distance_rmse


@patch("retrain.get_run_logger")
@patch("retrain.main")
@patch("retrain.save_df_to_s3")
@patch("retrain.load_data_s3")
@patch("retrain.archive_new_data_s3")
@pytest.mark.integration
def test_retrain_flow(
    mock_archive,
    mock_load_data,
    mock_save_df,
    mock_main,
    mock_logger,
    retrain_flow_with_mocked_config,
):
    baseline_df = MagicMock(empty=False, shape=(100, 10))
    # Make new_data_df empty to simulate no new data key
    # -> flow calls load_data_s3 only once
    new_data_df = MagicMock(empty=True, shape=(0, 0))

    mock_load_data.side_effect = [baseline_df, new_data_df]

    retrain_flow_with_mocked_config()

    assert mock_load_data.call_count == 2  # baseline and new data loaded
    mock_save_df.assert_not_called()  # no merge, so no save
    mock_main.assert_called_once()  # retrain called once
    mock_archive.assert_not_called()  # no archive since no new data
