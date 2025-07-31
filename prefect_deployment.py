# import os
# from dotenv import load_dotenv, dotenv_values
# from prefect import flow
# from prefect.runner.storage import GitRepository

# load_dotenv()
# env_vars = dotenv_values(".env")

# # REPO_URL = os.getenv("SOURCE_REPO")
# # if REPO_URL is None:
# #     REPO_URL = env_vars.get("SOURCE_REPO")
# # if isinstance(REPO_URL, bytes):
# #     REPO_URL = REPO_URL.decode("utf-8")
# # elif not isinstance(REPO_URL, str):
# #     REPO_URL = str(REPO_URL)
# # print("REPO_URL:", REPO_URL, type(REPO_URL))

# REPO_URL = "https://github.com/LamelK/solar-prediction-mlops_zoomcamp.git"

# COMMIT_HASH = os.getenv("GIT_COMMIT_HASH")


# @flow(log_prints=True)
# def main():
#     print("Running main pipeline")


# @flow(log_prints=True)
# def retrain_on_drift_distance_rmse():
#     print("Running retraining pipeline")


# if __name__ == "__main__":

#     source = GitRepository(url=REPO_URL, commit_sha=COMMIT_HASH)

#     main.from_source(
#         source=source,
#         entrypoint="pipeline.py:main",
#     ).deploy(
#         name="ml-pipeline",
#         work_pool_name="ml-pool",
#         job_variables={
#             "pip_packages": [
#                 "pandas",
#                 "numpy",
#                 "prefect-aws",
#                 "supabase",
#                 "mlflow",
#                 "scikit-learn",
#                 "scipy",
#                 "boto3",
#                 "requests",
#             ],
#             "env": env_vars,
#         },
#     )

#     retrain_on_drift_distance_rmse.from_source(
#         source=source,
#         entrypoint="retrain.py:retrain_on_drift_distance_rmse",
#     ).deploy(
#         name="retrain-pipeline",
#         work_pool_name="ml-pool",
#         job_variables={
#             "pip_packages": [
#                 "pandas",
#                 "numpy",
#                 "prefect-aws",
#                 "mlflow",
#                 "scikit-learn",
#                 "boto3",
#                 "requests",
#             ],
#             "env": env_vars,
#         },
#     )

import os
from dotenv import load_dotenv, dotenv_values
from prefect import flow
from prefect.runner.storage import GitRepository
from prefect.filesystems import LocalFileSystem

load_dotenv()
env_vars = dotenv_values(".env")

REPO_URL = "https://github.com/LamelK/solar-prediction-mlops_zoomcamp.git"
COMMIT_HASH = os.getenv("GIT_COMMIT_HASH")

USE_CLOUD = os.getenv("USE_CLOUD", "true").lower() == "true"


@flow(log_prints=True)
def main():
    print("Running main pipeline")


@flow(log_prints=True)
def retrain_on_drift_distance_rmse():
    print("Running retraining pipeline")


def deploy_cloud():
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


def deploy_local():
    storage = LocalFileSystem(basepath=os.getcwd())
    storage.save("local-storage", overwrite=True)

    main.deploy(
        name="ml-pipeline-local",
        work_pool_name="default",
        storage=storage,
        path=".",
        entrypoint="pipeline.py:main",
        job_variables={"env": env_vars},
    )

    retrain_on_drift_distance_rmse.deploy(
        name="retrain-pipeline-local",
        work_pool_name="default",
        storage=storage,
        path=".",
        entrypoint="retrain.py:retrain_on_drift_distance_rmse",
        job_variables={"env": env_vars},
    )


if __name__ == "__main__":
    if USE_CLOUD:
        print("Deploying to Prefect Cloud with GitHub source...")
        deploy_cloud()
    else:
        print("Deploying locally to Prefect Server from local files...")
        deploy_local()
