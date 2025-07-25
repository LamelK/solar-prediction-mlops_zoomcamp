import requests
from retrain import retrain_on_drift_distance_rmse
from prefect import flow, task

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"


@task(task_run_name="fetch_metrics from prometheus")
def fetch_metric(metric_name: str) -> float | None:
    """
    Fetch a single Prometheus metric value by name.
    """
    try:
        response = requests.get(PROMETHEUS_URL, params={"query": metric_name})
        data = response.json()
        if data["status"] == "success" and data["data"]["result"]:
            return float(data["data"]["result"][0]["value"][1])
    except Exception as e:
        print(f"Error fetching {metric_name}: {e}")
    return None


@flow(name="auto_retrain_monitor")
def check_metrics_and_retrain_flow():
    """
    Flow that checks Prometheus metrics and triggers retrain if thresholds are exceeded.
    """
    model_rmse = fetch_metric.submit("model_rmse").result()
    drift_share = fetch_metric.submit("enhanced_drift_share").result()

    print(f"Fetched model_rmse: {model_rmse}")
    print(f"Fetched enhanced_drift_share: {drift_share}")

    if model_rmse is not None and drift_share is not None:
        if model_rmse >= 170 and drift_share > 0.6:
            print("🚨 Conditions met: triggering retraining flow.")
            retrain_on_drift_distance_rmse()
        else:
            print("✅ Metrics below threshold. No retraining.")
    else:
        print("⚠️ One or both metrics are missing — skipping retraining.")


if __name__ == "__main__":
    check_metrics_and_retrain_flow()
