from dotenv import load_dotenv, dotenv_values
load_dotenv()  # loads .env into os.environ

from prefect import flow

# Load .env as a dict for Prefect job variables
env_vars = dotenv_values(".env")

if __name__ == "__main__":
    flow.from_source(
        source="https://github.com/LamelK/solar-prediction-mlops_zoomcamp.git",
        entrypoint="pipeline.py:main",
    ).deploy(
        name="ml-pipeline",
        work_pool_name="ml-pool",
        job_variables={
            "pip_packages": [
                "pandas", "prefect-aws", "supabase", 
                "mlflow", "scikit-learn", "scipy", 
                "boto3", "requests"
            ],
            "env": env_vars
        }
    )
