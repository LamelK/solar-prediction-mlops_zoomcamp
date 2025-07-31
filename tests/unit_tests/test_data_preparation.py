import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from mlpipeline.data_preparation import (
    load_and_prepare_data,
)  # adjust import to your actual module name


SAMPLE_CSV = (
    "UNIXTime,Data,Time,Radiation,Temperature,Pressure,Humidity,"
    "WindDirection_Degrees,Speed,TimeSunRise,TimeSunSet\n"
    "1472793006,9/1/2016 12:00:00 AM,19:10:06,2.53,55,30.45,65,"
    "155.71,3.37,06:07:00,18:38:00\n"
    "1472781308,9/1/2016 12:00:00 AM,15:55:08,628.8,63,30.42,58,"
    "1.55,6.75,06:07:00,18:38:00\n"
)


@pytest.fixture
def mock_boto3_client():
    with patch("mlpipeline.data_preparation.boto3.Session") as mock_session_cls, patch(
        "mlpipeline.data_preparation.boto3.client"
    ) as mock_client_func:

        # Mock S3 client for get_object
        mock_s3_client = MagicMock()

        # For load_data_s3: mock get_object returns obj with Body.read() returning bytes
        from io import BytesIO

        mock_s3_client.get_object.return_value = {
            "Body": BytesIO(SAMPLE_CSV.encode("utf-8"))
        }

        # boto3.Session().client() returns the mocked s3 client
        mock_session = MagicMock()
        mock_session.client.return_value = mock_s3_client
        mock_session_cls.return_value = mock_session

        # For upload_df_to_s3: boto3.client('s3') returns mock_s3_client also
        mock_client_func.return_value = mock_s3_client

        yield mock_s3_client


@patch("mlpipeline.data_preparation.get_run_logger")
@patch("mlpipeline.data_preparation.upload_df_to_s3")
@patch("mlpipeline.data_preparation.feature_engineer")
@patch("mlpipeline.data_preparation.clean_data")
@patch("mlpipeline.data_preparation.load_data_s3")
def test_load_and_prepare_data_flow(
    mock_load_data, mock_clean_data, mock_feature_engineer, mock_upload,
    mock_logger, mock_boto3_client
):
    mock_logger.return_value = MagicMock()

    # Create a real DataFrame from the sample CSV
    from io import StringIO
    sample_df = pd.read_csv(StringIO(SAMPLE_CSV))

    # Set up the mock chain
    mock_load_data.return_value = sample_df
    mock_clean_data.return_value = sample_df
    mock_feature_engineer.return_value = sample_df
    mock_upload.return_value = None

    # Run the flow with mock bucket and key
    bucket_name = "test-bucket"
    file_key = "raw-data/test.csv"

    df = load_and_prepare_data.fn(file_key=file_key, bucket_name=bucket_name)

    # Check that the flow called the expected tasks in the right order
    mock_load_data.assert_called_once_with(bucket_name, file_key)
    mock_clean_data.assert_called_once_with(sample_df)
    mock_feature_engineer.assert_called_once_with(sample_df)

    # Check that upload was called with the right processed key
    mock_upload.assert_called_once()
    args, kwargs = mock_upload.call_args
    assert args[0] is sample_df  # First arg should be the DataFrame
    assert args[1] == bucket_name  # Second arg should be bucket name
    assert args[2].startswith("processed-data/")  # Third arg should be processed key

    # Check that the flow returns the processed DataFrame
    assert df is sample_df
