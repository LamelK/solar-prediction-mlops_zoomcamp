# import pandas as pd
# import numpy as np
# import boto3
# from io import StringIO

# def load_data_local(path='../data/data.csv'):
#     """
#     Load data from a local CSV file.

#     Args:
#         path (str): Path to the CSV file.

#     Returns:
#         pd.DataFrame: Loaded data.
#     """
#     return pd.read_csv(path)


# def load_data_s3(bucket_name, file_key, aws_profile=None):
#     """
#     Load data from an S3 bucket.

#     Args:
#         bucket_name (str): Name of the S3 bucket.
#         file_key (str): Key/path to the CSV file in the bucket.
#         aws_profile (str, optional): AWS CLI profile name.

#     Returns:
#         pd.DataFrame: Loaded data from S3.
#     """
#     session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
#     s3 = session.client('s3')
#     obj = s3.get_object(Bucket=bucket_name, Key=file_key)
#     return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))


# def clean_data(df):
#     """
#     Clean the input DataFrame.

#     - Removes duplicates.
#     - Drops rows that are entirely NA.

#     Args:
#         df (pd.DataFrame): Raw data.

#     Returns:
#         pd.DataFrame: Cleaned data.
#     """
#     df = df.drop_duplicates()
#     df = df.dropna(how='all')
#     return df


# def feature_engineer(df):
#     """
#     Perform feature engineering on the DataFrame.

#     - Converts UNIXTime to datetime.
#     - Extracts time-based features.
#     - Generates cyclical features.
#     - Calculates minutes since sunrise and until sunset.
#     - Drops unnecessary columns.

#     Args:
#         df (pd.DataFrame): Cleaned data.

#     Returns:
#         pd.DataFrame: Data with engineered features.
#     """
#     # Convert UNIXTime to datetime
#     df['DateTime'] = pd.to_datetime(df['UNIXTime'], unit='s')

#     # Extract components
#     df['Hour'] = df['DateTime'].dt.hour
#     df['Minute'] = df['DateTime'].dt.minute
#     df['Day'] = df['DateTime'].dt.day
#     df['Month'] = df['DateTime'].dt.month
#     df['Weekday'] = df['DateTime'].dt.weekday

#     # Time-based cyclical features
#     df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
#     df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
#     df['Minute_sin'] = np.sin(2 * np.pi * df['Minute'] / 60)
#     df['Minute_cos'] = np.cos(2 * np.pi * df['Minute'] / 60)
#     df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
#     df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
#     df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
#     df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
#     df['Weekday_sin'] = np.sin(2 * np.pi * df['Weekday'] / 7)
#     df['Weekday_cos'] = np.cos(2 * np.pi * df['Weekday'] / 7)

#     # Sunrise and Sunset features
#     df['TimeSunRise_obj'] = pd.to_timedelta(df['TimeSunRise'])
#     df['TimeSunSet_obj'] = pd.to_timedelta(df['TimeSunSet'])
#     df['SunriseDateTime'] = df['DateTime'].dt.normalize() + df['TimeSunRise_obj']
#     df['SunsetDateTime'] = df['DateTime'].dt.normalize() + df['TimeSunSet_obj']
#     df['MinutesSinceSunrise'] = (df['DateTime'] - df['SunriseDateTime']).dt.total_seconds() / 60
#     df['MinutesUntilSunset'] = (df['SunsetDateTime'] - df['DateTime']).dt.total_seconds() / 60

#     # Drop intermediate and unneeded columns
#     df.drop(columns=[
#         'UNIXTime', 'Data', 'Time',
#         'TimeSunRise', 'TimeSunSet',
#         'TimeSunRise_obj', 'TimeSunSet_obj',
#         'Hour', 'Minute', 'Day',
#         'Month', 'Weekday', 'DateTime'
#     ], inplace=True)

#     return df


# def load_and_prepare_data(path='../data/data.csv'):
#     """
#     Full pipeline to load, clean, and feature engineer data from local storage.

#     Args:
#         path (str): Path to local CSV.

#     Returns:
#         pd.DataFrame: Prepared dataset.
#     """
#     df = load_data_local(path)
#     df = clean_data(df)
#     df = feature_engineer(df)
#     return df

import pandas as pd
import numpy as np
import boto3
from io import StringIO
from prefect import flow, task, get_run_logger



def load_data_local(path='../data/data.csv'):
    logger = get_run_logger()
    logger.info(f"Loading data from local path: {path}")
    return pd.read_csv(path)


def load_data_s3(bucket_name, file_key, aws_profile=None):
    logger = get_run_logger()
    logger.info(f"Loading data from S3 bucket: {bucket_name}, file: {file_key}")
    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))


def clean_data(df):
    logger = get_run_logger()
    logger.info("Cleaning data: removing duplicates and empty rows")
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    logger.info(f"Data shape after cleaning: {df.shape}")
    return df


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
        'Hour', 'Minute', 'Day',
        'Month', 'Weekday', 'DateTime'
    ], inplace=True)

    logger.info(f"Feature engineering completed. Final data shape: {df.shape}")
    return df


@task(task_run_name="Data Loading and Prepo")
def load_and_prepare_data(path='../data/data.csv'):
    logger = get_run_logger()
    logger.info("Starting data load and preparation pipeline")

    df = load_data_local(path) 
    df = clean_data(df)
    df = feature_engineer(df)

    logger.info(f"Data prepared with shape: {df.shape}")
    return df
