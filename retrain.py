import pandas as pd
from pipeline import main
from mlpipeline.data_preparation import load_data_s3
from prefect import flow, get_run_logger
import os
import boto3
from datetime import datetime
import io
from config import get_s3_config

s3_config = get_s3_config()
S3_BUCKET_NAME = s3_config["bucket_name"]
BASELINE_KEY = s3_config["raw_baseline_key"]
NEW_DATA_KEY = s3_config["new_data_key"]

# Helper function to save a DataFrame to S3 as CSV
def save_df_to_s3(df, bucket, key):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3 = boto3.client('s3')
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

# Helper function to archive the new data file in S3 after retraining
def archive_new_data_s3():
    s3 = boto3.client('s3')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_key = f"raw-data/archived/new_data_{timestamp}.csv"
    try:
        # Copy the new data file to the archive location
        s3.copy_object(Bucket=S3_BUCKET_NAME, CopySource={'Bucket': S3_BUCKET_NAME, 'Key': NEW_DATA_KEY}, Key=archive_key)
        # Delete the original new data file
        s3.delete_object(Bucket=S3_BUCKET_NAME, Key=NEW_DATA_KEY)
        print(f"Archived new data to s3://{S3_BUCKET_NAME}/{archive_key} and deleted original.")
    except Exception as e:
        print(f"Failed to archive new data in S3: {e}")

@flow(name="Retrain on Drift, Distance, RMSE")
def retrain_on_drift_distance_rmse():
    logger = get_run_logger()
    if not S3_BUCKET_NAME:
        raise ValueError("S3_BUCKET_NAME must be set in the environment.")
    # Step 1: Fetch the baseline data from S3
    logger.info(f"Fetching baseline data from S3: {BASELINE_KEY}")
    baseline = load_data_s3(S3_BUCKET_NAME, BASELINE_KEY)
    # Step 2: Fetch the new data from S3 (if it exists)
    logger.info(f"Fetching new data from S3: {NEW_DATA_KEY}")
    try:
        new_data = load_data_s3(S3_BUCKET_NAME, NEW_DATA_KEY)
    except Exception as e:
        logger.warning(f"Could not fetch new data from S3: {e}")
        new_data = pd.DataFrame()
    # Step 3: If new data exists, merge it with the baseline, deduplicate, and save back to S3
    if not new_data.empty:
        logger.info(f"Merging baseline and new data...")
        combined = pd.concat([baseline, new_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
        logger.info(f"Combined dataset shape after merging: {combined.shape}")
        # Save the merged DataFrame back to S3 (overwriting the baseline)
        save_df_to_s3(combined, S3_BUCKET_NAME, BASELINE_KEY)
        logger.info(f"Saved merged data to s3://{S3_BUCKET_NAME}/{BASELINE_KEY}")
        # Step 4: Call the main pipeline, which will now use the updated baseline from S3
        main(bucket_name=S3_BUCKET_NAME, raw_key=BASELINE_KEY)
        logger.info("Retraining completed with new data.")
        # Step 5: Archive the new data file in S3
        archive_new_data_s3()
    else:
        # If no new data, just retrain on the baseline
        logger.info("No new labeled data found. Retraining with baseline only.")
        main(bucket_name=S3_BUCKET_NAME, raw_key=BASELINE_KEY)
        logger.info("Retraining completed with baseline only.")

if __name__ == "__main__":
    retrain_on_drift_distance_rmse() 