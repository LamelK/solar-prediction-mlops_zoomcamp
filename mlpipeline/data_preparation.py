import pandas as pd
import numpy as np
import boto3
from io import StringIO
from prefect import task, flow, get_run_logger


@task(name="Load Data from Local")
def load_data_local(path='../data/data.csv'):
    logger = get_run_logger()
    logger.info(f"Loading data from local path: {path}")
    return pd.read_csv(path)


@task(name="Load Data from S3")
def load_data_s3(bucket_name, file_key, aws_profile=None):
    logger = get_run_logger()
    logger.info(f"Loading data from S3 bucket: {bucket_name}, key: {file_key}")

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

    logger.info(f"Data loaded from S3: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


@task(name="Clean Data")
def clean_data(df):
    logger = get_run_logger()
    logger.info("Cleaning data: removing duplicates and dropping all-NA rows")
    initial_shape = df.shape
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    logger.info(f"Data cleaned: {initial_shape} -> {df.shape}")
    return df


@task(name="Feature Engineering")
def feature_engineer(df):
    logger = get_run_logger()
    logger.info("Starting feature engineering")

    df['DateTime'] = pd.to_datetime(df['UNIXTime'], unit='s')

    df['Hour'] = df['DateTime'].dt.hour
    df['Minute'] = df['DateTime'].dt.minute
    df['Day'] = df['DateTime'].dt.day
    df['Month'] = df['DateTime'].dt.month
    df['Weekday'] = df['DateTime'].dt.weekday

    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
    df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
    df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

    df['TimeSunRise_obj'] = pd.to_timedelta(df['TimeSunRise'])
    df['TimeSunSet_obj'] = pd.to_timedelta(df['TimeSunSet'])
    df['SunriseDateTime'] = df['DateTime'].dt.normalize() + df['TimeSunRise_obj']
    df['SunsetDateTime'] = df['DateTime'].dt.normalize() + df['TimeSunSet_obj']
    df['MinutesSinceSunrise'] = (df['DateTime'] - df['SunriseDateTime']).dt.total_seconds() / 60
    df['MinutesUntilSunset'] = (df['SunsetDateTime'] - df['DateTime']).dt.total_seconds() / 60

    df.drop(columns=[
        'UNIXTime', 'Data', 'Time',
        'TimeSunRise', 'TimeSunSet',
        'TimeSunRise_obj', 'TimeSunSet_obj',
        'SunriseDateTime', 'SunsetDateTime',  
        'Hour', 'Minute', 'Day', 'datetime', 'DateTime',
        'Month', 'Weekday', 
    ], inplace=True)

    logger.info(f"Feature engineering completed. Data now has columns: {list(df.columns)}")
    return df


@flow(name="Load and Prepare Data Pipeline")
def load_and_prepare_data(path='../data/data.csv'):
    logger = get_run_logger()
    logger.info("Starting the full data load and preparation pipeline")

    df = load_data_local(path)
    df = clean_data(df)
    df = feature_engineer(df)

    logger.info("Pipeline completed")
    return df


if __name__ == "__main__":
    df = load_and_prepare_data()
    print(df.columns)
