import os
from dotenv import load_dotenv, dotenv_values
from prefect import flow
from prefect.runner.storage import GitRepository

load_dotenv()
env_vars = dotenv_values(".env")


REPO_URL = os.getenv("SOURCE_REPO")

COMMIT_HASH = os.getenv("GIT_COMMIT_HASH")


@flow(log_prints=True)
def main():
    print("Running main pipeline")


@flow(log_prints=True)
def retrain_on_drift_distance_rmse():
    print("Running retraining pipeline")


if __name__ == "__main__":

    source = GitRepository(url=REPO_URL, commit_sha=COMMIT_HASH)

    main.from_source(
        source=source,
        entrypoint="pipeline.py:main",
    ).deploy(
        name="ml-pipeline",
        work_pool_name="ml-pool",
        job_variables={
            "pip_packages": [
                "pandas",
                "numpy",
                "prefect-aws",
                "supabase",
                "mlflow",
                "scikit-learn",
                "scipy",
                "boto3",
                "requests",
            ],
            "env": env_vars,
        },
    )

    retrain_on_drift_distance_rmse.from_source(
        source=source,
        entrypoint="retrain.py:retrain_on_drift_distance_rmse",
    ).deploy(
        name="retrain-pipeline",
        work_pool_name="ml-pool",
        job_variables={
            "pip_packages": [
                "pandas",
                "numpy",
                "prefect-aws",
                "mlflow",
                "scikit-learn",
                "boto3",
                "requests",
            ],
            "env": env_vars,
        },
    )
