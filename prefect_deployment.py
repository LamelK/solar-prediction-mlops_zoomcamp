import os
from dotenv import load_dotenv, dotenv_values
from prefect import flow

load_dotenv()


# Load .env as a dict for Prefect job variables
env_vars = dotenv_values(".env")
source_repo = os.getenv("SOURCE_REPO")

if __name__ == "__main__":
    flow.from_source(source=source_repo, entrypoint="pipeline.py:main",).deploy(
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
    # Second deployment
    flow.from_source(
        source=source_repo, entrypoint="retrain.py:retrain_on_drift_distance_rmse",
    ).deploy(
        name="retrain-deployment",
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
