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


def test_load_and_prepare_data_flow(mock_boto3_client):
    # Run the flow with mock bucket and key
    bucket_name = "test-bucket"
    file_key = "raw-data/test.csv"

    df = load_and_prepare_data(file_key=file_key, bucket_name=bucket_name)

    # Check that boto3 get_object was called once
    mock_boto3_client.get_object.assert_called_once_with(
        Bucket=bucket_name, Key=file_key
    )

    # Check that upload put_object was called once
    mock_boto3_client.put_object.assert_called_once()
    args, kwargs = mock_boto3_client.put_object.call_args
    assert kwargs["Bucket"] == bucket_name
    assert kwargs["Key"].startswith("processed-data/")
    assert "Body" in kwargs

    # Check dataframe shape and columns after feature engineering and cleaning
    # Initial data has 2 rows, no duplicates or all-NA rows => expect 2 rows
    assert df.shape[0] == 2

    # Check some of the new engineered columns exist
    expected_columns = {
        "Radiation",
        "Temperature",
        "Pressure",
        "Humidity",
        "WindDirection_Degrees",
        "Speed",
        "Hour_sin",
        "Hour_cos",
        "Minute_sin",
        "Minute_cos",
        "Day_sin",
        "Day_cos",
        "Month_sin",
        "Month_cos",
        "Weekday_sin",
        "Weekday_cos",
        "MinutesSinceSunrise",
        "MinutesUntilSunset",
    }
    assert expected_columns.issubset(set(df.columns))

    # Check dropped columns are gone
    dropped_columns = [
        "UNIXTime",
        "Data",
        "Time",
        "TimeSunRise",
        "TimeSunSet",
        "TimeSunRise_obj",
        "TimeSunSet_obj",
        "SunriseDateTime",
        "SunsetDateTime",
        "Hour",
        "Minute",
        "Day",
        "DateTime",
        "Month",
        "Weekday",
    ]
    for col in dropped_columns:
        assert col not in df.columns
