import pandas as pd
import numpy as np
import boto3
from io import StringIO


def load_data_local(path='../data/inference_data.csv'):
    return pd.read_csv(path)

def load_data_s3(bucket_name, file_key, aws_profile=None):

    session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    s3 = session.client('s3')
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

    return df



def clean_data(df):
    initial_shape = df.shape
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    return df


def feature_engineer(df):


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

    cols_to_drop = [
        'UNIXTime', 'Data', 'Time',
        'TimeSunRise', 'TimeSunSet',
        'TimeSunRise_obj', 'TimeSunSet_obj',
        'SunriseDateTime', 'SunsetDateTime',  
        'Hour', 'Minute', 'Day', 'datetime', 'DateTime',
        'Month', 'Weekday','Radiation' 
    ]

    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)

    return df



def load_and_prepare_data(df=None, path=None):
    if df is None and path is None:
        raise ValueError("Must provide either df or path")

    if df is None:
        df = load_data_local(path)

    df = clean_data(df)
    df = feature_engineer(df)
    return df


# if __name__ == "__main__":
#     df = load_and_prepare_data()
#     print(df.columns)
