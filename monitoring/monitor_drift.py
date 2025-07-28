import os
import sys
from dotenv import load_dotenv
from datetime import datetime
import json
import time
import random
import io
import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from prometheus_client import start_http_server, Gauge
from supabase import create_client, Client
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from scipy.stats import ks_2samp, anderson_ksamp
from io import StringIO

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # noqa: E402

from config import (  # noqa: E402
    get_supabase_config,
    get_s3_config,
    get_monitoring_config,
)  # noqa: E402


load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

np.random.seed(42)
random.seed(42)


# load baseline data from s3 function
def load_data_s3(bucket_name, file_key, aws_profile=None):
    """
    Loads a CSV file from S3 into a pandas DataFrame.
    Optionally uses a specific AWS profile.
    """
    session = (
        boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
    )
    s3 = session.client("s3")

    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    df = pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))

    print(
        f"Loaded data from S3 bucket: {bucket_name}, "
        f"key: {file_key} => {df.shape[0]} rows, {df.shape[1]} columns"
    )
    return df


# Get configuration
supabase_config = get_supabase_config()
s3_config = get_s3_config()
monitoring_config = get_monitoring_config()

supabase: Client = create_client(supabase_config["url"], supabase_config["key"])

# Load baseline from S3
S3_BUCKET_NAME = s3_config["bucket_name"]
RAW_KEY = s3_config["raw_baseline_key"]
if not S3_BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME must be set in the environment.")
baseline = load_data_s3(S3_BUCKET_NAME, RAW_KEY)


# Helper to upload file to S3
def upload_file_to_s3(local_path, bucket, s3_key):

    s3 = boto3.client("s3")
    with open(local_path, "rb") as f:
        s3.upload_fileobj(f, bucket, s3_key)


# Prometheus metrics
data_drift_gauge = Gauge(
    "data_drift_share", "Share of features with detected drift (Evidently)"
)
distance_drift_gauge = Gauge(
    "distance_drift_share", "Share of features exceeding distance threshold (Custom)"
)
statistical_drift_gauge = Gauge(
    "statistical_drift_share",
    "Share of features with statistically significant drift (p<0.05)",
)

# Enhanced drift metrics
enhanced_drift_gauge = Gauge(
    "enhanced_drift_share",
    "Share of features with enhanced drift detection (CRITICAL + WARNING)",
)
critical_drift_count = Gauge(
    "critical_drift_count", "Number of features with CRITICAL drift status"
)
warning_drift_count = Gauge(
    "warning_drift_count", "Number of features with WARNING drift status"
)
ok_drift_count = Gauge("ok_drift_count", "Number of features with OK drift status")

# Level-specific drift metrics
distribution_drift_share = Gauge(
    "distribution_drift_share", "Share of features with distribution family issues"
)
parameter_drift_share = Gauge(
    "parameter_drift_share", "Share of features with parameter drift"
)
statistical_significance_share = Gauge(
    "statistical_significance_share",
    "Share of features with statistically significant drift",
)

# Individual feature drift status (0=OK, 1=WARNING, 2=CRITICAL, 3=ERROR)
feature_drift_status = Gauge(
    "feature_drift_status", "Drift status for individual features", ["feature_name"]
)

# Feature-specific detailed metrics
feature_mean_change = Gauge(
    "feature_mean_change", "Mean change percentage for features", ["feature_name"]
)
feature_std_change = Gauge(
    "feature_std_change",
    "Standard deviation change percentage for features",
    ["feature_name"],
)
feature_p_value = Gauge(
    "feature_p_value", "P-value from statistical test for features", ["feature_name"]
)

# Create one gauge per column (skip 'id' and 'datetime')
column_drift_gauges = {
    col: Gauge(f"data_drift_{col.lower()}", f"Drift detected for column {col}")
    for col in baseline.columns
    if col not in ["id", "datetime"]
}

# Add RMSE Prometheus metrics
rmse_gauge = Gauge("model_rmse", "Root Mean Squared Error of model predictions")
rmse_pct_gauge = Gauge("model_rmse_pct", "RMSE as percent of radiation range")

# Get distance threshold from configuration
DISTANCE_FEATURE_THRESHOLD = monitoring_config["distance_feature_threshold"]
MONITORING_INTERVAL = 60  # 1 minute in seconds
CONFIDENCE_LEVEL = 0.05  # 95% confidence level (standard in statistics)

print(f"Using distance feature threshold: {DISTANCE_FEATURE_THRESHOLD}")
print(
    f"Monitoring interval: {MONITORING_INTERVAL} seconds "
    f"({MONITORING_INTERVAL/3600:.1f} hours)"
)
print(
    f"Statistical confidence level: {CONFIDENCE_LEVEL} "
    f"(p < {CONFIDENCE_LEVEL} = significant)"
)


def check_statistical_significance(baseline_series, recent_series, feature_name):
    """
    Check statistical significance using multiple tests.
    Returns dict with test results and overall significance.
    """
    # Remove NaN values
    baseline_clean = baseline_series.dropna()
    recent_clean = recent_series.dropna()

    if len(baseline_clean) < 10 or len(recent_clean) < 10:
        return {
            "significant": False,
            "reason": "Insufficient data (< 10 samples)",
            "tests": {},
        }

    results = {}

    # Kolmogorov-Smirnov test (most common for distribution comparison)
    try:
        ks_stat, ks_p_value = ks_2samp(baseline_clean, recent_clean)
        if ks_p_value < CONFIDENCE_LEVEL:
            interpretation = f"p={ks_p_value:.4f} < 0.05(SIGNIFICANT)"
        else:
            interpretation = f"p={ks_p_value:.4f} > 0.05(NOT SIGNIFICANT)"

        results["ks_test"] = {
            "statistic": ks_stat,
            "p_value": ks_p_value,
            "significant": ks_p_value < CONFIDENCE_LEVEL,
            "interpretation": interpretation,
        }
    except Exception as e:
        results["ks_test"] = {"significant": False, "error": str(e)}

    # Anderson-Darling test (for normality and distribution comparison)
    try:
        if len(baseline_clean) > 20 and len(recent_clean) > 20:
            ad_stat, ad_critical, ad_significance = anderson_ksamp(
                [baseline_clean, recent_clean]
            )
            if ad_significance < CONFIDENCE_LEVEL:
                interpretation = f"p={ad_significance:.4f} < 0.05 (SIGNIFICANT)"
            else:
                interpretation = f"p={ad_significance:.4f} > 0.05 (NOT SIGNIFICANT)"

            results["anderson_test"] = {
                "statistic": ad_stat,
                "significance": ad_significance,
                "significant": ad_significance < CONFIDENCE_LEVEL,
                "interpretation": interpretation,
            }
            results["anderson_test"] = {
                "significant": False,
                "reason": "Insufficient data for Anderson test (< 20 samples)",
            }
    except Exception as e:
        results["anderson_test"] = {"significant": False, "error": str(e)}

    # Count significant tests
    significant_tests = sum(
        [
            results.get("ks_test", {}).get("significant", False),
            results.get("anderson_test", {}).get("significant", False),
        ]
    )

    # Overall significance (at least one test significant)
    overall_significant = significant_tests >= 1

    return {
        "significant": overall_significant,
        "significant_tests": significant_tests,
        "total_tests": len(results),
        "tests": results,
    }


def enhanced_drift_analysis(baseline_series, recent_series, feature_name):
    """
    Enhanced drift detection using scaled parameter comparison and statistical test.
    """
    from scipy import stats
    from sklearn.preprocessing import StandardScaler

    baseline_clean = baseline_series.dropna()
    recent_clean = recent_series.dropna()

    if len(baseline_clean) < 10 or len(recent_clean) < 10:
        return {
            "overall_status": "ERROR",
            "message": "Insufficient data (< 10 samples)",
            "details": {},
        }

    results = {
        "distribution_family": {},
        "parameter_drift": {},
        "statistical_test": {},
        "overall_status": "OK",
    }

    # Distribution check removed; focus on parameters and significance
    results["distribution_family"] = {
        "status": "OK",
        "message": (
            "Distribution check removed - " "focusing on parameters and significance"
        ),
    }

    try:
        scaler = StandardScaler()
        baseline_scaled = scaler.fit_transform(
            baseline_clean.values.reshape(-1, 1)
        ).flatten()
        recent_scaled = scaler.transform(recent_clean.values.reshape(-1, 1)).flatten()

        baseline_mean_scaled = np.mean(baseline_scaled)
        baseline_std_scaled = np.std(baseline_scaled)
        recent_mean_scaled = np.mean(recent_scaled)
        recent_std_scaled = np.std(recent_scaled)

        mean_change = abs(recent_mean_scaled - baseline_mean_scaled)
        std_change = abs(recent_std_scaled - baseline_std_scaled)

        baseline_mean_orig = np.mean(baseline_clean)
        baseline_std_orig = np.std(baseline_clean)
        recent_mean_orig = np.mean(recent_clean)
        recent_std_orig = np.std(recent_clean)

        combined_distance = max(mean_change, std_change)
        SCALED_DISTANCE_THRESHOLD = DISTANCE_FEATURE_THRESHOLD

        if combined_distance > SCALED_DISTANCE_THRESHOLD:
            param_status = "WARNING"
            param_message = (
                f"Scaled distance: {combined_distance:.2f} > "
                f"{SCALED_DISTANCE_THRESHOLD} (mean:{mean_change:.2f}, "
                f"std:{std_change:.2f})"
            )
        else:
            param_status = "OK"
            param_message = (
                f"Scaled distance: {combined_distance:.2f} <= "
                f"{SCALED_DISTANCE_THRESHOLD} (mean:{mean_change:.2f}, "
                f"std:{std_change:.2f})"
            )

        results["parameter_drift"] = {
            "status": param_status,
            "message": param_message,
            "mean_change": mean_change,
            "std_change": std_change,
            "combined_distance": combined_distance,
            "threshold": SCALED_DISTANCE_THRESHOLD,
            "baseline_mean_scaled": baseline_mean_scaled,
            "baseline_std_scaled": baseline_std_scaled,
            "recent_mean_scaled": recent_mean_scaled,
            "recent_std_scaled": recent_std_scaled,
            "baseline_mean_orig": baseline_mean_orig,
            "baseline_std_orig": baseline_std_orig,
            "recent_mean_orig": recent_mean_orig,
            "recent_std_orig": recent_std_orig,
            "scaling_applied": True,
        }
    except Exception as e:
        results["parameter_drift"] = {
            "status": "ERROR",
            "message": f"Error analyzing parameters: {e}",
            "mean_change": 0,
            "std_change": 0,
            "combined_distance": 0,
        }

    try:
        statistic, p_value = stats.ks_2samp(baseline_clean, recent_clean)
        significant = p_value < CONFIDENCE_LEVEL

        if p_value < 0.0001:
            p_display = "<0.0001"
        else:
            p_display = f"{p_value:.6f}".rstrip("0").rstrip(".")

        results["statistical_test"] = {
            "status": "WARNING" if significant else "OK",
            "message": (
                "p="
                + str(p_display)
                + " "
                + (
                    "< 0.05 (SIGNIFICANT)"
                    if significant
                    else "> 0.05 (NOT SIGNIFICANT)"
                )
            ),
            "p_value": p_value,
            "significant": significant,
        }
    except Exception as e:
        results["statistical_test"] = {
            "status": "ERROR",
            "message": f"Error in statistical test: {e}",
            "p_value": 1.0,
            "significant": False,
        }

    p_value = results["statistical_test"].get("p_value", 1.0)
    combined_distance = results["parameter_drift"].get("combined_distance", 0.0)

    if p_value < CONFIDENCE_LEVEL and combined_distance > SCALED_DISTANCE_THRESHOLD:
        if combined_distance > 1.0:
            results["overall_status"] = "CRITICAL"
        else:
            results["overall_status"] = "WARNING"
    else:
        results["overall_status"] = "OK"

    results["combined_logic"] = {
        "p_value": p_value,
        "combined_distance": combined_distance,
        "threshold": SCALED_DISTANCE_THRESHOLD,
        "statistical_significant": p_value < CONFIDENCE_LEVEL,
        "distance_above_threshold": combined_distance > SCALED_DISTANCE_THRESHOLD,
        "both_conditions_met": p_value < CONFIDENCE_LEVEL
        and combined_distance > SCALED_DISTANCE_THRESHOLD,
    }

    return results


def fetch_recent_data():
    response = supabase.table("model_logs").select("*").limit(1200).execute()
    data = response.data
    return pd.DataFrame(data)


def update_metrics():
    recent = fetch_recent_data()
    if recent.empty:
        print("No recent data fetched.")
        return 0.0, 0.0, recent

    # DRIFT/DISTANCE SECTION: Exclude UNIXTime only here
    exclude_cols = ["id", "datetime", "UNIXTime"]
    common_cols = [
        col
        for col in baseline.columns
        if col in recent.columns and col not in exclude_cols
    ]
    baseline_drift = baseline[common_cols]
    recent_drift = recent[common_cols]

    # Filter numeric columns only for drift check to avoid errors
    numeric_cols = [
        col for col in common_cols if pd.api.types.is_numeric_dtype(baseline[col])
    ]
    baseline_numeric = baseline_drift[numeric_cols]
    recent_numeric = recent_drift[numeric_cols]

    # Cast recent columns to baseline dtype exactly
    for col in numeric_cols:
        recent_numeric[col] = recent_numeric[col].astype(baseline_numeric[col].dtype)

    print("Dtypes equal:", baseline_numeric.dtypes.equals(recent_numeric.dtypes))

    # Enhanced Multi-Level Drift Analysis
    print("🔍 Enhanced Multi-Level Drift Analysis:")
    print("=" * 50)

    enhanced_results = []
    critical_count = 0
    warning_count = 0
    ok_count = 0

    for col in numeric_cols:
        # Run enhanced analysis
        enhanced_result = enhanced_drift_analysis(
            baseline_numeric[col], recent_numeric[col], col
        )
        enhanced_results.append(enhanced_result)

        # Count statuses
        if enhanced_result["overall_status"] == "CRITICAL":
            critical_count += 1
        elif enhanced_result["overall_status"] == "WARNING":
            warning_count += 1
        else:
            ok_count += 1

        # Print detailed results
        status_emoji = {"CRITICAL": "🚨", "WARNING": "⚠️", "OK": "✅", "ERROR": "❌"}

        print(f"\n{status_emoji.get(enhanced_result['overall_status'], '❓')} {col}:")
        print(
            f"  📈 Parameters: {enhanced_result['parameter_drift']['status']} "
            f"- {enhanced_result['parameter_drift']['message']}"
        )
        print(
            f"  🔬 Statistical: {enhanced_result['statistical_test']['status']} "
            f"- {enhanced_result['statistical_test']['message']}"
        )
        print(f"  🎯 Overall: {enhanced_result['overall_status']}")

    # Calculate enhanced drift metrics
    total_features = len(enhanced_results)
    enhanced_share = (
        (critical_count + warning_count) / total_features if total_features > 0 else 0.0
    )

    # Calculate level-specific drift shares
    distribution_issues = sum(
        1
        for r in enhanced_results
        if r["distribution_family"]["status"] in ["CRITICAL", "WARNING"]
    )
    parameter_issues = sum(
        1
        for r in enhanced_results
        if r["parameter_drift"]["status"] in ["CRITICAL", "WARNING"]
    )
    statistical_issues = sum(
        1
        for r in enhanced_results
        if r["statistical_test"]["status"] in ["CRITICAL", "WARNING"]
    )

    distribution_share = (
        distribution_issues / total_features if total_features > 0 else 0.0
    )
    parameter_share = parameter_issues / total_features if total_features > 0 else 0.0
    statistical_share = (
        statistical_issues / total_features if total_features > 0 else 0.0
    )

    # Update Prometheus metrics
    enhanced_drift_gauge.set(enhanced_share)
    critical_drift_count.set(critical_count)
    warning_drift_count.set(warning_count)
    ok_drift_count.set(ok_count)

    distribution_drift_share.set(distribution_share)
    parameter_drift_share.set(parameter_share)
    statistical_significance_share.set(statistical_share)

    # Update individual feature metrics
    for i, result in enumerate(enhanced_results):
        feature_name = numeric_cols[i]

        # Status mapping: 0=OK, 1=WARNING, 2=CRITICAL, 3=ERROR
        status_mapping = {"OK": 0, "WARNING": 1, "CRITICAL": 2, "ERROR": 3}
        status_value = status_mapping.get(result["overall_status"], 3)

        feature_drift_status.labels(feature_name=feature_name).set(status_value)

        # Detailed metrics
        if "mean_change" in result["parameter_drift"]:
            feature_mean_change.labels(feature_name=feature_name).set(
                result["parameter_drift"]["mean_change"] * 100
            )
            feature_std_change.labels(feature_name=feature_name).set(
                result["parameter_drift"]["std_change"] * 100
            )

        if "p_value" in result["statistical_test"]:
            feature_p_value.labels(feature_name=feature_name).set(
                result["statistical_test"]["p_value"]
            )

    # Keep the old metric for backward compatibility
    statistical_drift_gauge.set(enhanced_share)

    print("\n📊 Enhanced Drift Summary:")
    print(f"  🚨 Critical: {critical_count}/{total_features}")
    print(f"  ⚠️ Warning: {warning_count}/{total_features}")
    print(f"  ✅ OK: {ok_count}/{total_features}")
    print(f"  📈 Drift Share: {enhanced_share:.2f}")

    # Determine overall drift status
    if enhanced_share > 0.7:
        drift_status = "⚠️ WARNING: High drift detected (>70% features)"
    elif enhanced_share > 0.5:
        drift_status = "⚠️ WARNING: Majority of features show drift"
    elif enhanced_share > 0.2:
        drift_status = "⚠️ WARNING: Some features show drift"
    else:
        drift_status = "✅ OK: No significant drift detected"

    print(f"\n🎯 Overall Status: {drift_status}")

    # Calculate overall data drift on numeric columns
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=baseline_numeric, current_data=recent_numeric)
    result = report.as_dict()
    drift_share = result["metrics"][0]["result"].get("share_of_drifted_columns", 0.0)
    data_drift_gauge.set(drift_share)
    print(f"Evidently drift share: {drift_share}")

    # Generate and save HTML report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_filename = f"data_drift_report_{timestamp}.html"
    html_path = os.path.join(BASE_DIR, html_filename)

    try:
        # Save HTML report locally
        report.save_html(html_path)
        print(f"HTML report saved locally: {html_path}")

        # Upload HTML report to S3
        s3_key = f"monitoring-reports/{html_filename}"
        upload_file_to_s3(html_path, S3_BUCKET_NAME, s3_key)
        print(f"HTML report uploaded to S3: s3://{S3_BUCKET_NAME}/{s3_key}")

        # Clean up local file (optional)
        # os.remove(html_path)

    except Exception as e:
        print(f"Failed to save/upload HTML report: {e}")

    # Print full JSON drift result for debugging
    drift_json = result["metrics"][0]["result"].copy()
    drift_json.pop("drift_share", None)  # Remove legacy/unused field if present
    print(json.dumps(drift_json, indent=2))

    # For RMSE: use the original 'recent' DataFrame (with UNIXTime) and ground truth
    # Do NOT drop/exclude UNIXTime from these DataFrames

    return drift_share, enhanced_share, recent


def clear_model_logs():
    try:
        supabase.table("model_logs").delete().neq("id", 0).execute()
        print("Supabase model_logs cleared.")
    except Exception as e:
        print("Failed to clear model_logs:", e)


def compute_rmse_with_ground_truth(recent):

    # Fetch ground truth from S3 (raw-data/new_data/new_data.csv)
    s3 = boto3.client("s3")
    key = "raw-data/new_data/new_data.csv"
    try:
        csv_obj = s3.get_object(Bucket=S3_BUCKET_NAME, Key=key)
        ground_truth_df = pd.read_csv(io.BytesIO(csv_obj["Body"].read()))
    except Exception as e:
        print(f"Could not fetch ground truth from S3: {e}")
        return
    if ground_truth_df.empty:
        print("No ground truth data found in S3.")
        return
    if recent.empty:
        print("No recent predictions found in Supabase.")
        return
    # Ensure UNIXTime is the same type
    recent["UNIXTime"] = recent["UNIXTime"].astype(int)
    ground_truth_df["UNIXTime"] = ground_truth_df["UNIXTime"].astype(int)
    merged = pd.merge(
        recent, ground_truth_df[["UNIXTime", "Radiation"]], on="UNIXTime", how="inner"
    )
    if merged.empty:
        print("No matching UNIXTime values between predictions and ground truth.")
        return
    if "Radiation_x" not in merged.columns or "Radiation_y" not in merged.columns:
        print("Merged DataFrame does not contain required columns for RMSE.")
        print("Merged columns:", merged.columns.tolist())
        print("Merged DataFrame head:\n", merged.head())
        return
    rmse = np.sqrt(mean_squared_error(merged["Radiation_y"], merged["Radiation_x"]))

    # Provide context for RMSE interpretation
    radiation_range = (
        ground_truth_df["Radiation"].max() - ground_truth_df["Radiation"].min()
    )
    rmse_percentage = (rmse / radiation_range) * 100 if radiation_range > 0 else 0

    # Set Prometheus metrics
    rmse_gauge.set(rmse)
    rmse_pct_gauge.set(rmse_percentage)

    print("📊 MODEL PERFORMANCE:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  Radiation range: {radiation_range:.2f}")
    print(f"  RMSE as % of range: {rmse_percentage:.1f}%")

    # Provide interpretation
    if rmse_percentage < 5:
        performance_status = "🟢 EXCELLENT"
        interpretation = "Model performance is very good"
    elif rmse_percentage < 10:
        performance_status = "🟡 GOOD"
        interpretation = "Model performance is acceptable"
    elif rmse_percentage < 20:
        performance_status = "🟠 FAIR"
        interpretation = "Model performance needs improvement"
    else:
        performance_status = "🔴 POOR"
        interpretation = "Model performance is concerning"

    print(f"  Performance: {performance_status}")
    print(f"  Interpretation: {interpretation}")

    # Check if RMSE is trending (if we had historical data)
    print("  Note: Consider tracking RMSE trends over time for better context")

    return rmse


if __name__ == "__main__":
    start_http_server(8080)
    print("Prometheus metrics available on port 8080")
    print("Starting drift monitoring...")
    while True:
        drift_share, enhanced_share, recent = update_metrics()
        print(f"Drift Share: {drift_share:.2f}, Enhanced Share: {enhanced_share:.2f}")
        compute_rmse_with_ground_truth(recent)
        print(f"Next check in {MONITORING_INTERVAL/60:.1f} minutes...")
        time.sleep(MONITORING_INTERVAL)
