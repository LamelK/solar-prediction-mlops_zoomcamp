import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import boto3  # type: ignore
from io import StringIO
from prefect import task, flow, get_run_logger  # type: ignore
from typing import Optional
import os
from dotenv import load_dotenv  # type: ignore

# Load environment variables from .env if present
load_dotenv()


@task(name="Load Data from S3", retries=1, retry_delay_seconds=10)
def load_data_s3(bucket_name, file_key, aws_profile=None):
    """
    Loads a CSV file from S3 into a pandas DataFrame.
    Optionally uses a specific AWS profile.
    """
    logger = get_run_logger()
    logger.info(f"Loading data from S3 bucket: {bucket_name}, key: {file_key}")

    # Create a boto3 session, optionally with a specific AWS profile
    session = (
        boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    )
    s3 = session.client("s3")
    # Download the object and read it into a DataFrame
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    logger.info(f"Data loaded from S3: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


@task(name="Upload processed data to S3", retries=1, retry_delay_seconds=10)
def upload_df_to_s3(df: pd.DataFrame, bucket: str, key: str):
    """
    Uploads a DataFrame as a CSV to the specified S3 bucket and key.
    """
    # Convert DataFrame to CSV in memory
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    # Upload to S3
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())


@task(name="Clean Data")
def clean_data(df):
    """
    Removes duplicate rows and drops rows where all values are NA.
    """
    logger = get_run_logger()
    logger.info("Cleaning data: removing duplicates and dropping all-NA rows")
    initial_shape = df.shape
    # Remove duplicate rows
    df = df.drop_duplicates()
    # Remove rows where all values are NA
    df = df.dropna(how="all")
    logger.info(f"Data cleaned: {initial_shape} -> {df.shape}")
    return df


@task(name="Feature Engineering")
def feature_engineer(df):
    """
    Adds time-based and cyclical features to the DataFrame for modeling.
    Drops columns that are no longer needed after feature creation.
    """
    logger = get_run_logger()
    logger.info("Starting feature engineering")

    # Convert UNIX timestamp to datetime
    df["DateTime"] = pd.to_datetime(df["UNIXTime"], unit="s")

    # Extract time features from DateTime
    df["Hour"] = df["DateTime"].dt.hour
    df["Minute"] = df["DateTime"].dt.minute
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["Weekday"] = df["DateTime"].dt.weekday

    # Add cyclical encodings for time features (useful for ML models)
    df["Hour_sin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["Minute_sin"] = np.sin(2 * np.pi * df["Minute"] / 60)
    df["Minute_cos"] = np.cos(2 * np.pi * df["Minute"] / 60)
    df["Day_sin"] = np.sin(2 * np.pi * df["Day"] / 31)
    df["Day_cos"] = np.cos(2 * np.pi * df["Day"] / 31)
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
    df["Weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

    # Calculate sunrise and sunset related features
    df["TimeSunRise_obj"] = pd.to_timedelta(df["TimeSunRise"])
    df["TimeSunSet_obj"] = pd.to_timedelta(df["TimeSunSet"])
    df["SunriseDateTime"] = df["DateTime"].dt.normalize() + df["TimeSunRise_obj"]
    df["SunsetDateTime"] = df["DateTime"].dt.normalize() + df["TimeSunSet_obj"]
    df["MinutesSinceSunrise"] = (
        df["DateTime"] - df["SunriseDateTime"]
    ).dt.total_seconds() / 60
    df["MinutesUntilSunset"] = (
        df["SunsetDateTime"] - df["DateTime"]
    ).dt.total_seconds() / 60

    # Drop columns that are no longer needed after feature engineering
    df.drop(
        columns=[
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
        ],
        inplace=True,
    )

    logger.info(
        f"Feature engineering completed. Data now has columns: {list(df.columns)}"
    )
    return df


@flow(name="Load and Preprocess Data")
def load_and_prepare_data(file_key: str, bucket_name: Optional[str] = None):
    """
    Loads data from S3, cleans it, performs feature engineering, and uploads
    the processed data back to S3.
    Returns the processed DataFrame.
    """
    logger = get_run_logger()
    logger.info("Starting the data load and preprocessing flow")

    # Use provided bucket or fall back to environment variable
    bucket = bucket_name or os.getenv("S3_BUCKET_NAME")
    key = file_key
    if not bucket or not key:
        raise ValueError(
            """Both S3 bucket name and file key must be provided
            (either as argument or env var for bucket, and argument for key).
             Only S3 loading is supported."""
        )
    logger.info(f"Loading data from S3: bucket={bucket}, key={key}")
    df = load_data_s3(bucket, key)

    # Clean and engineer features
    df = clean_data(df)
    df = feature_engineer(df)

    # Save processed data to S3
    if key.startswith("raw-data/"):
        # Replace 'raw-data/' with 'processed-data/' in the key
        processed_key = key.replace("raw-data/", "processed-data/")
    else:
        # Otherwise, just prepend 'processed-data/'
        processed_key = "processed-data/" + os.path.basename(key)
    logger.info(f"Uploading processed data to S3: bucket={bucket}, key={processed_key}")
    upload_df_to_s3(df, bucket, processed_key)
    logger.info("Processed data uploaded to S3.")

    logger.info("Preprocessing flow completed")
    return df
