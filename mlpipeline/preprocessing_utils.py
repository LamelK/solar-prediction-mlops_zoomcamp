import pandas as pd
import numpy as np


def clean_data(df):
    """
    Remove duplicate rows and drop rows where all values are NA.
    """
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df


def feature_engineer(df):
    """
    Add time-based and cyclical features to the DataFrame for inference.
    Drops columns that are no longer needed after feature creation.
    """

    df["DateTime"] = pd.to_datetime(df["UNIXTime"], unit="s")

    # Extract time features
    df["Hour"] = df["DateTime"].dt.hour
    df["Minute"] = df["DateTime"].dt.minute
    df["Day"] = df["DateTime"].dt.day
    df["Month"] = df["DateTime"].dt.month
    df["Weekday"] = df["DateTime"].dt.weekday

    # Add cyclical encodings for time features
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

    # Columns to drop - same as training preprocessing
    cols_to_drop = [
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
    # For inference data, 'Radiation' column won't exist
    # since that's what is being predicted
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    df.drop(columns=existing_cols_to_drop, inplace=True)

    return df


def load_and_prepare_data(df):
    """
    Prepare data for model inference by cleaning and feature engineering.
    Args:
        df: DataFrame with raw input data (no Radiation column)
    Returns:
        DataFrame with processed features ready for model prediction
    """
    if df is None:
        raise ValueError("DataFrame must be provided")

    df = clean_data(df)
    df = feature_engineer(df)
    return df


# if __name__ == "__main__":
#     df = load_and_prepare_data()
#     print(df.columns)
