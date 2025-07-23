import pandas as pd
from pipeline import main
from mlpipeline.data_preparation import load_data_s3
from prefect import flow, get_run_logger
import boto3
from datetime import datetime
import io
from config import get_s3_config
import requests


def trigger_model_reload(api_url: str):
    try:
        response = requests.post(api_url)
        if response.status_code == 200:
            print("Model reload triggered successfully.")
        else:
            print(f"Failed to reload model: {response.status_code} {response.text}")
    except Exception as e:
        print(f"Error calling reload endpoint: {e}")


def get_config():
    """
    Retrieve S3 configuration values for bucket and data keys.
    """
    s3_config = get_s3_config()
    bucket = s3_config["bucket_name"]
    baseline_key = s3_config["raw_baseline_key"]
    new_data_key = s3_config["new_data_key"]
    return bucket, baseline_key, new_data_key


def save_df_to_s3(df, bucket, key):
    """
    Save a DataFrame as a CSV file to the specified S3 bucket and key.
    """
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())


def archive_new_data_s3(bucket, new_data_key):
    """
    Archive the new data file in S3 by copying
    it to an archive location with a timestamp,
    then delete the original.
    """
    s3 = boto3.client("s3")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_key = f"raw-data/archived/new_data_{timestamp}.csv"
    try:
        # Copy the new data file to the archive location
        s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": new_data_key},
            Key=archive_key,
        )
        # Delete the original new data file
        s3.delete_object(Bucket=bucket, Key=new_data_key)
        print(f"Archived new data to s3://{bucket}/{archive_key} and deleted original.")
    except Exception as e:
        print(f"Failed to archive new data in S3: {e}")


@flow(name="Retrain on Drift, Distance, RMSE")
def retrain_on_drift_distance_rmse():
    """
    Main retraining flow. If new data is available,
    merge it with the baseline, retrain, and archive
    the new data. Otherwise, retrain on the baseline only.
    """
    logger = get_run_logger()

    # Get S3 configuration for bucket and data keys
    bucket, baseline_key, new_data_key = get_config()
    if not bucket:
        raise ValueError("S3_BUCKET_NAME must be set in the environment.")

    # Load baseline data from S3
    logger.info(f"Fetching baseline data from S3: {baseline_key}")
    baseline = load_data_s3(bucket, baseline_key)

    # Try to load new data from S3 if a key is provided
    if new_data_key:
        logger.info(f"Fetching new data from S3: {new_data_key}")
        try:
            new_data = load_data_s3(bucket, new_data_key)
        except Exception as e:
            logger.warning(f"Could not fetch new data from S3: {e}")
            new_data = pd.DataFrame()  # Use empty DataFrame if loading fails
    else:
        new_data = pd.DataFrame()  # No new data key provided

    # If new data exists, merge with baseline and retrain
    if not new_data.empty:
        logger.info("Merging baseline and new data...")
        # Concatenate and deduplicate the combined dataset
        combined = (
            pd.concat([baseline, new_data], ignore_index=True)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        logger.info(f"Combined dataset shape after merging: {combined.shape}")

        # Save the merged data back to S3 (overwriting the baseline)
        save_df_to_s3(combined, bucket, baseline_key)
        logger.info(f"Saved merged data to s3://{bucket}/{baseline_key}")

        # Run the main pipeline using the updated baseline
        main(bucket_name=bucket, raw_key=baseline_key)
        logger.info("Retraining completed with new data.")

        # Archive the new data file in S3
        archive_new_data_s3(bucket, new_data_key)
    else:
        # If no new data, retrain on the baseline only
        logger.info("No new labeled data found. Retraining with baseline only.")
        main(bucket_name=bucket, raw_key=baseline_key)
        logger.info("Retraining completed with baseline only.")


if __name__ == "__main__":
    # Entry point for running the retraining flow directly
    retrain_on_drift_distance_rmse()
