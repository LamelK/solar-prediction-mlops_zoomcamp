import requests
from retrain import retrain_on_drift_distance_rmse
from prefect import flow, task, get_run_logger

# 👇 Try Docker internal first, fallback to localhost
POSSIBLE_URLS = [
    "http://host.docker.internal:9090/api/v1/query",
    "http://localhost:9090/api/v1/query",
]


def get_prometheus_url() -> str | None:
    for url in POSSIBLE_URLS:
        try:
            response = requests.get(url, timeout=2)
            if response.status_code == 200:
                return url
        except Exception:
            continue
    return None


@task(name="Fetch Metric from Prometheus", retries=1, retry_delay_seconds=10)
def fetch_metric(metric_name: str) -> float | None:
    logger = get_run_logger()
    prometheus_url = get_prometheus_url()
    if not prometheus_url:
        logger.error("❌ Prometheus is unreachable on all fallback URLs.")
        return None

    try:
        logger.info(f"Fetching metric: {metric_name} from {prometheus_url}")
        response = requests.get(prometheus_url, params={"query": metric_name})
        data = response.json()
        if data["status"] == "success" and data["data"]["result"]:
            value = float(data["data"]["result"][0]["value"][1])
            logger.info(f"✅ Metric '{metric_name}' = {value}")
            return value
        else:
            logger.warning(f"No result returned for metric: {metric_name}")
    except Exception as e:
        logger.error(f"Error fetching {metric_name}: {e}")
    return None


@task(name="Check if Thresholds Are Exceeded")
def metrics_exceeded(model_rmse: float, drift_share: float) -> bool:
    logger = get_run_logger()
    if model_rmse >= 170 and drift_share > 0.6:
        logger.warning("🚨 Thresholds exceeded")
        return True
    logger.info("✅ Metrics below threshold")
    return False


@flow(name="Auto Retrain Monitor")
def check_metrics_and_retrain_flow():
    logger = get_run_logger()
    logger.info("🚀 Starting metric check and retrain flow")

    model_rmse = fetch_metric.submit("model_rmse").result()
    drift_share = fetch_metric.submit("enhanced_drift_share").result()

    if model_rmse is None or drift_share is None:
        logger.warning("⚠️ Missing one or more metrics — retraining skipped")
        return

    should_retrain = metrics_exceeded.submit(model_rmse, drift_share).result()

    if should_retrain:
        retrain_on_drift_distance_rmse()


if __name__ == "__main__":
    check_metrics_and_retrain_flow()
