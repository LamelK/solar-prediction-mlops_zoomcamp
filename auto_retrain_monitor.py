import requests
from retrain import retrain_on_drift_distance_rmse
from prefect import flow, task, get_run_logger

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

@task(name="Fetch Metric from Prometheus")
def fetch_metric(metric_name: str) -> float | None:
    logger = get_run_logger()
    try:
        logger.info(f"Fetching metric: {metric_name}")
        response = requests.get(PROMETHEUS_URL, params={"query": metric_name})
        data = response.json()
        if data["status"] == "success" and data["data"]["result"]:
            value = float(data["data"]["result"][0]["value"][1])
            logger.info(f"Metric '{metric_name}' fetched with value: {value}")
            return value
        else:
            logger.warning(f"No result returned for metric: {metric_name}")
    except Exception as e:
        logger.error(f"Error fetching {metric_name}: {e}")
    return None


@flow(name="Auto Retrain Monitor")
def check_metrics_and_retrain_flow():
    logger = get_run_logger()
    logger.info("🚀 Starting metric check and retrain flow")

    model_rmse = fetch_metric.submit("model_rmse").result()
    drift_share = fetch_metric.submit("enhanced_drift_share").result()

    logger.info(f"Fetched model_rmse: {model_rmse}")
    logger.info(f"Fetched enhanced_drift_share: {drift_share}")

    if model_rmse is not None and drift_share is not None:
        if model_rmse >= 170 and drift_share > 0.6:
            logger.warning("🚨 Thresholds exceeded — triggering retraining")
            retrain_on_drift_distance_rmse()
        else:
            logger.info("✅ Metrics below threshold — no retraining triggered")
    else:
        logger.warning("⚠️ Missing one or more metrics — retraining skipped")


if __name__ == "__main__":
    check_metrics_and_retrain_flow()
