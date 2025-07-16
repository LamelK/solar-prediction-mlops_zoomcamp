# from fastapi import FastAPI, UploadFile, File, HTTPException, Body
# import pandas as pd
# import mlflow.pyfunc
# from mlflow.tracking import MlflowClient
# from typing import List, Union
# from mlpipeline.preprocessing_utils import load_and_prepare_data
# from api.schemas import RawInputData  

# app = FastAPI()

# mlflow.set_tracking_uri("http://localhost:5000")
# MODEL_NAME = "MyTopModel"

# def get_production_model_version(model_name):
#     client = MlflowClient()
#     versions = client.search_model_versions(f"name='{model_name}'")
#     for v in versions:
#         if v.tags.get("status") == "production":
#             return v.version
#     raise RuntimeError(f"No production model version found for '{model_name}'")

# try:
#     prod_version = get_production_model_version(MODEL_NAME)
#     model_uri = f"models:/{MODEL_NAME}/{prod_version}"
#     model = mlflow.pyfunc.load_model(model_uri)
# except Exception as e:
#     raise RuntimeError(f"Failed to load production model: {e}")

# @app.get("/")
# async def root():
#     return {"message": "ML Model API is running"}

# @app.post("/predict")
# async def predict_json(
#     data: Union[RawInputData, List[RawInputData]] = Body(...)
# ):
#     """
#     Accept either a single JSON object or a list of JSON objects validated by Pydantic.
#     """
#     try:
#         if isinstance(data, RawInputData):
#             # Single object -> wrap in list
#             data_list = [data.dict()]
#         else:
#             # List of objects
#             data_list = [item.dict() for item in data]

#         df_raw = pd.DataFrame(data_list)
#         df_preprocessed = load_and_prepare_data(df=df_raw)

#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Invalid JSON data or preprocessing failed: {e}")

#     preds = model.predict(df_preprocessed)
#     return {"predictions": preds.tolist()}

# @app.post("/predict_csv")
# async def predict_csv(file: UploadFile = File(...)):
#     if not file.filename.endswith(".csv"):
#         raise HTTPException(status_code=400, detail="Only CSV files are supported.")
#     try:
#         df_raw = pd.read_csv(file.file)
#         df_preprocessed = load_and_prepare_data(df=df_raw)
#     except Exception as e:
#         raise HTTPException(status_code=400, detail=f"Error processing CSV: {e}")

#     preds = model.predict(df_preprocessed)
#     return {"predictions": preds.tolist()}

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
import pandas as pd
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
from typing import List, Union
from datetime import datetime
from supabase import create_client, Client
from mlpipeline.preprocessing_utils import load_and_prepare_data
from api.schemas import RawInputData  

app = FastAPI()

# MLflow setup
mlflow.set_tracking_uri("http://localhost:5000")
MODEL_NAME = "MyTopModel"

def get_production_model_version(model_name):
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
SUPABASE_URL = "https://ccfmfqtlizzbaxlshzbu.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNjZm1mcXRsaXp6YmF4bHNoemJ1Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1MjY2NzQwMiwiZXhwIjoyMDY4MjQzNDAyfQ.oa_2mmgazvIaiDk8BnymXiXZACb0iLRGmnnlGS0xhhE"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def log_to_supabase(data: dict, prediction: float):
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
        "Prediction": prediction
    }
    response = supabase.table("model_logs").insert(record).execute()

    # Print the whole response object and its attributes to explore structure
    # print("Response object:", response)
    # print("Response __dict__:", response.__dict__)
    # print("Response dir():", dir(response))
    if not response.data:
        print("Insert failed or returned no data")
    else:
        print("Insert succeeded:", response.data)

@app.get("/")
async def root():
    return {"message": "ML Model API is running"} 

@app.post("/predict")
async def predict_json(
    data: Union[RawInputData, List[RawInputData]] = Body(...)
):
    try:
        if isinstance(data, RawInputData):
            data_list = [data.dict()]
        else:
            data_list = [item.dict() for item in data]

        df_raw = pd.DataFrame(data_list)
        df_preprocessed = load_and_prepare_data(df=df_raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON data or preprocessing failed: {e}")

    preds = model.predict(df_preprocessed)

    for input_dict, pred in zip(data_list, preds):
        log_to_supabase(input_dict, float(pred))

    return {"predictions": preds.tolist()}

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
    try:
        df_raw = pd.read_csv(file.file)
        df_preprocessed = load_and_prepare_data(df=df_raw)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {e}")

    preds = model.predict(df_preprocessed)

    for idx, row in df_raw.iterrows():
        input_dict = row.to_dict()
        log_to_supabase(input_dict, float(preds[idx]))

    return {"predictions": preds.tolist()}
