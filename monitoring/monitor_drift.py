import json
import time
import os
import random
import numpy as np
import pandas as pd
from prometheus_client import start_http_server, Gauge
from supabase import create_client, Client
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metrics import ColumnDriftMetric

np.random.seed(42)
random.seed(42)

SUPABASE_URL = "https://ccfmfqtlizzbaxlshzbu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNjZm1mcXRsaXp6YmF4bHNoemJ1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjY2NzQwMiwiZXhwIjoyMDY4MjQzNDAyfQ.oa_2mmgazvIaiDk8BnymXiXZACb0iLRGmnnlGS0xhhE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Resolve absolute path to baseline data relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
baseline_path = os.path.join(BASE_DIR, '..', 'data', 'training_data.csv')

baseline = pd.read_csv(baseline_path)

# Overall drift gauge
data_drift_gauge = Gauge('data_drift_share', 'Share of features with detected drift')

# Create one gauge per column (skip 'id' and 'datetime')
column_drift_gauges = {
    col: Gauge(f"data_drift_{col.lower()}", f"Drift detected for column {col}")
    for col in baseline.columns if col not in ['id', 'datetime']
}

def fetch_recent_data():
    response = supabase.table("model_logs").select("*").limit(1000).execute()
    data = response.data
    return pd.DataFrame(data)

def update_metrics():
    recent = fetch_recent_data()
    if recent.empty:
        print("No recent data fetched.")
        return 0.0

    # Align columns (exclude id and datetime)
    common_cols = [col for col in baseline.columns if col in recent.columns and col not in ['id', 'datetime']]
    baseline_aligned = baseline[common_cols]
    recent_aligned = recent[common_cols]

    # Filter numeric columns only for drift check to avoid errors
    numeric_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(baseline[col])]
    baseline_numeric = baseline[numeric_cols]
    recent_numeric = recent[numeric_cols]

    # Cast recent columns to baseline dtype exactly
    for col in numeric_cols:
        recent_numeric[col] = recent_numeric[col].astype(baseline_numeric[col].dtype)

    print("Dtypes equal:", baseline_numeric.dtypes.equals(recent_numeric.dtypes))

    # Calculate overall data drift on numeric columns
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline_numeric, current_data=recent_numeric)
    result = report.as_dict()
    drift_share = result['metrics'][0]['result'].get('share_of_drifted_columns', 0.0)
    data_drift_gauge.set(drift_share)
    print(f"Updated overall data drift share: {drift_share}")

    # Calculate per-column drift and update gauges
    for col in column_drift_gauges.keys():
        if not pd.api.types.is_numeric_dtype(baseline[col]):
            print(f"Skipping non-numeric column '{col}' for drift calculation.")
            column_drift_gauges[col].set(0)
            continue

        metric = ColumnDriftMetric(column_name=col)
        report_col = Report(metrics=[metric])
        report_col.run(reference_data=baseline[[col]], current_data=recent[[col]])
        metric_result = report_col.as_dict()['metrics'][0]['result']
        drifted = metric_result.get('drift_detected', False)
        print(f"Column '{col}' drifted: {drifted}")
        column_drift_gauges[col].set(int(drifted))

    # Save HTML report for overall drift (last report)
    html_path = os.path.join(BASE_DIR, "data_drift_report.html")
    report.save_html(html_path)
    print(f"Drift report saved to {html_path}")

    # Optional: print full JSON result for debugging
    print(json.dumps(result['metrics'][0]['result'], indent=2))

    return drift_share

def clear_model_logs():
    try:
        supabase.table("model_logs").delete().neq("id", 0).execute()
        print("Supabase model_logs cleared.")
    except Exception as e:
        print("Failed to clear model_logs:", e)


if __name__ == "__main__":
    start_http_server(8080)
    print("Prometheus metrics available on port 8080")
    while True:
        update_metrics()
        time.sleep(60)




# import json
# from prometheus_client import start_http_server, Gauge
# from evidently.report import Report
# from evidently.metric_preset import DataDriftPreset
# import pandas as pd
# import time
# from supabase import create_client, Client
# import os
# import numpy as np
# import random

# np.random.seed(42)
# random.seed(42)

# SUPABASE_URL = "https://ccfmfqtlizzbaxlshzbu.supabase.co"
# SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNjZm1mcXRsaXp6YmF4bHNoemJ1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjY2NzQwMiwiZXhwIjoyMDY4MjQzNDAyfQ.oa_2mmgazvIaiDk8BnymXiXZACb0iLRGmnnlGS0xhhE"
# supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# baseline = pd.read_csv('../data/df_short.csv')

# data_drift_gauge = Gauge('data_drift_share', 'Share of features with detected drift')

# def fetch_recent_data():
#     response = supabase.table("model_logs").select("*").limit(1000).execute()
#     data = response.data
#     return pd.DataFrame(data)


# def update_metrics():
#     recent = fetch_recent_data()
#     if recent.empty:
#         print("No recent data fetched.")
#         return

#     common_cols = [col for col in baseline.columns if col in recent.columns and col not in ['id', 'datetime']]
#     baseline_aligned = baseline[common_cols]
#     recent_aligned = recent[common_cols]

#     variable_cols = [col for col in common_cols 
#                      if baseline[col].nunique() > 1 and recent[col].nunique() > 1]

#     baseline_aligned = baseline[variable_cols]
#     recent_aligned = recent[variable_cols]

#     for col in baseline_aligned.columns:
#         recent_aligned[col] = recent_aligned[col].astype(baseline_aligned[col].dtype)

#     print("Dtypes equal:", baseline_aligned.dtypes.equals(recent_aligned.dtypes))

#     report = Report(metrics=[DataDriftPreset()])
#     report.run(reference_data=baseline_aligned, current_data=recent_aligned)

#     result = report.as_dict()
#     drift_share = result['metrics'][0]['result'].get('share_of_drifted_columns', 0.0)
#     data_drift_gauge.set(drift_share)

#     print(f"Updated data drift share: {drift_share}")

#     html_path = "data_drift_report.html"
#     report.save_html(html_path)
#     print(f"Drift report saved to {os.path.abspath(html_path)}")
#     print(json.dumps(result['metrics'][0]['result'], indent=2))

# if __name__ == "__main__":
#     start_http_server(8080)
#     print("Prometheus metrics available on port 8080")
#     while True:
#         update_metrics()
#         time.sleep(60)


