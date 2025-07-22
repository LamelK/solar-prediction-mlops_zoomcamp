from fastapi import FastAPI, UploadFile, File, HTTPException, Body
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from typing import List, Union
from supabase import create_client, Client
from mlpipeline.preprocessing_utils import load_and_prepare_data
from api.schemas import RawInputData
from dotenv import load_dotenv
from config import get_mlflow_config, get_s3_config, get_supabase_config
from pydantic import ValidationError

load_dotenv()

# Config
mlflow_config = get_mlflow_config()
s3_config = get_s3_config()
supabase_config = get_supabase_config()

mlflow.set_tracking_uri(mlflow_config["tracking_uri"])

app = FastAPI()

MODEL_NAME = mlflow_config["model_name"]


def get_production_model_version(model_name):
    """
    Retrieve the version number of the current production model
    from MLflow Model Registry.Raises an error if no production
    version is found.
    """
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    for v in versions:
        if v.tags.get("status") == "production":
            return v.version
    raise RuntimeError(f"No production model version found for '{model_name}'")


try:
    prod_version = get_production_model_version(MODEL_NAME)
    model_uri = f"models:/{MODEL_NAME}/{prod_version}"
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    raise RuntimeError(f"Failed to load production model: {e}")

# Supabase setup
supabase: Client = create_client(supabase_config["url"], supabase_config["key"])


def log_to_supabase(data: dict, prediction: float):
    """
    Log input data and prediction to the Supabase table 'model_logs'.
    """
    record = {
        "UNIXTime": data.get("UNIXTime"),
        "Data": data.get("Data"),
        "Time": data.get("Time"),
        "Temperature": data.get("Temperature"),
        "Pressure": data.get("Pressure"),
        "Humidity": data.get("Humidity"),
        "WindDirection_Degrees": data.get("WindDirection_Degrees"),
        "Speed": data.get("Speed"),
        "TimeSunRise": data.get("TimeSunRise"),
        "TimeSunSet": data.get("TimeSunSet"),
        "datetime": data.get("datetime"),
        "Radiation": prediction,
    }
    response = supabase.table("model_logs").insert(record).execute()
    if not response.data:
        print("Insert failed or returned no data")
    else:
        print("Insert succeeded:", response.data)


@app.get("/")
async def root():
    """
    Health check endpoint for the API.
    """
    return {"message": "ML Model API is running"}


@app.post("/predict")
async def predict_json(data: Union[RawInputData, List[RawInputData]] = Body(...)):
    """
    Predict endpoint for JSON input. Accepts a single or list of RawInputData objects.
    Returns predictions for each input.
    """
    try:
        if isinstance(data, RawInputData):
            data_list = [data.model_dump()]  # Pydantic v2
        else:
            data_list = [item.model_dump() for item in data]

        df_raw = pd.DataFrame(data_list)
        df_preprocessed = load_and_prepare_data(df=df_raw)
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON data or preprocessing failed: {e}"
        )

    preds = model.predict(df_preprocessed)

    # Log each prediction to Supabase
    for input_dict, pred in zip(data_list, preds):
        log_to_supabase(input_dict, float(pred))

    return {"predictions": preds.tolist()}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """
    Predict endpoint for CSV file upload. Validates each row and returns predictions.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        df_raw = pd.read_csv(file.file)

        # Validate each CSV row with Pydantic model
        validated_rows = []
        for _, row in df_raw.iterrows():
            try:
                validated = RawInputData.model_validate(
                    row.to_dict()
                )  # Pydantic v2 validation
                validated_rows.append(validated.model_dump())
            except ValidationError as ve:
                raise HTTPException(status_code=400, detail=f"Invalid row in CSV: {ve}")

        df_validated = pd.DataFrame(validated_rows)
        df_preprocessed = load_and_prepare_data(df=df_validated)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {e}")

    preds = model.predict(df_preprocessed)

    # Log each prediction to Supabase
    for idx, row in enumerate(validated_rows):
        log_to_supabase(row, float(preds[idx]))

    return {"predictions": preds.tolist()}
