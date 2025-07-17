# import os
# import shutil
# import pandas as pd
# from pipeline import main  
# from monitoring.monitor_drift import update_metrics, clear_model_logs

# DRIFT_THRESHOLD = 0.4
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASELINE_PATH = os.path.join(BASE_DIR, '..', 'data', 'training_data.csv')
# NEW_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'new_data')
# ARCHIVE_DIR = os.path.join(BASE_DIR, '..', 'data', 'archived')

# def retrain_if_drift():
#     print("Running drift detection...")
#     drift_share = update_metrics()  
#     print(f"Drift share detected: {drift_share}")

#     if drift_share > DRIFT_THRESHOLD:
#         print("Drift threshold exceeded. Starting retraining process...")
#         baseline = pd.read_csv(BASELINE_PATH)

#         new_files = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith('.csv')]
#         new_data_frames = []

#         for filename in new_files:
#             filepath = os.path.join(NEW_DATA_DIR, filename)
#             print(f"Loading new labeled data from {filepath}")
#             df_new = pd.read_csv(filepath)
#             new_data_frames.append(df_new)

#         if new_data_frames:
#             new_data = pd.concat(new_data_frames, ignore_index=True)
#             print(f"Combined new data shape: {new_data.shape}")

#             combined = pd.concat([baseline, new_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
#             print(f"Combined dataset shape after merging: {combined.shape}")

#             combined.to_csv(BASELINE_PATH, index=False)
#             print(f"Updated baseline saved to {BASELINE_PATH}")

#             main(df=combined)  # call full pipeline 
#             print("Retraining completed.")

#             if not os.path.exists(ARCHIVE_DIR):
#                 os.makedirs(ARCHIVE_DIR)

#             for filename in new_files:
#                 shutil.move(os.path.join(NEW_DATA_DIR, filename), os.path.join(ARCHIVE_DIR, filename))
#                 print(f"Moved processed file {filename} to archive.")
#         else:
#             print("No new labeled data files found. Retraining with baseline only.")
#             main(df=baseline)  # call full pipeline 
#             print("Retraining completed with baseline only.")
#     else:
#         print("Drift below threshold. No retraining needed.")

# if __name__ == "__main__":
#     retrain_if_drift()
#     clear_model_logs()

import os
import shutil
import pandas as pd
from pipeline import main  
from monitoring.monitor_drift import update_metrics, clear_model_logs
from prefect import task, flow, get_run_logger

DRIFT_THRESHOLD = 0.4
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_PATH = os.path.join(BASE_DIR, '..', 'data', 'training_data.csv')
NEW_DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'new_data')
ARCHIVE_DIR = os.path.join(BASE_DIR, '..', 'data', 'archived')


@task
def detect_drift():
    logger = get_run_logger()
    logger.info("Running drift detection...")
    drift_share = update_metrics()
    logger.info(f"Drift share detected: {drift_share}")
    return drift_share

@task
def load_baseline():
    return pd.read_csv(BASELINE_PATH)

@task
def load_new_data():
    new_files = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith('.csv')]
    new_data_frames = []
    for filename in new_files:
        filepath = os.path.join(NEW_DATA_DIR, filename)
        df_new = pd.read_csv(filepath)
        new_data_frames.append(df_new)
    if new_data_frames:
        return pd.concat(new_data_frames, ignore_index=True)
    else:
        return pd.DataFrame()

@task
def save_baseline(df):
    df.to_csv(BASELINE_PATH, index=False)

@task
def archive_files(filenames):
    if not os.path.exists(ARCHIVE_DIR):
        os.makedirs(ARCHIVE_DIR)
    for filename in filenames:
        shutil.move(os.path.join(NEW_DATA_DIR, filename), os.path.join(ARCHIVE_DIR, filename))

@flow(name="Retrain if Drift Flow")
def retrain_if_drift():
    drift_share = detect_drift()
    if drift_share > DRIFT_THRESHOLD:
        baseline = load_baseline()
        new_data = load_new_data()
        if not new_data.empty:
            combined = pd.concat([baseline, new_data], ignore_index=True).drop_duplicates().reset_index(drop=True)
            save_baseline(combined)
            main(df=combined)  # retrain full pipeline
            filenames = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith('.csv')]
            archive_files(filenames)
        else:
            main(df=baseline)
    clear_model_logs()

if __name__ == "__main__":
    retrain_if_drift()
