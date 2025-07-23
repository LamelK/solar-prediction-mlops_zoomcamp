import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from api.serve_model import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def patch_mlflow_and_client(monkeypatch):
    with patch("api.serve_model.mlflow.pyfunc.load_model", return_value=MagicMock()), \
         patch("api.serve_model.create_client", return_value=MagicMock()):
        yield


pytestmark = pytest.mark.integration


@pytest.mark.integration
@pytest.fixture
def sample_json_input():
    return {
        "UNIXTime": 1472793006,
        "Data": "9/1/2016 12:00:00 AM",
        "Time": "19:10:06",
        "Temperature": 25,
        "Pressure": 1015.0,
        "Humidity": 50,
        "WindDirection_Degrees": 180.0,
        "Speed": 5.0,
        "TimeSunRise": "06:07:00",
        "TimeSunSet": "18:38:00",
        "datetime": "2016-09-01T19:10:06",
    }


@patch("api.serve_model.model")
@patch("api.serve_model.load_and_prepare_data")
@patch("api.serve_model.log_to_supabase")
@pytest.mark.integration
def test_predict_json(
    mock_log_supabase, mock_preprocess, mock_model, sample_json_input, client
):
    mock_preprocess.return_value = pd.DataFrame(np.random.rand(1, 3))
    mock_model.predict.return_value = np.array([123.45])

    response = client.post("/predict", json=sample_json_input)
    assert response.status_code == 200
    json_resp = response.json()
    assert "predictions" in json_resp
    assert isinstance(json_resp["predictions"], list)
    assert len(json_resp["predictions"]) == 1
    assert json_resp["predictions"][0] == 123.45

    mock_preprocess.assert_called_once()
    mock_model.predict.assert_called_once()
    mock_log_supabase.assert_called_once()


@patch("api.serve_model.model")
@patch("api.serve_model.load_and_prepare_data")
@patch("api.serve_model.log_to_supabase")
@pytest.mark.integration
def test_predict_csv(
    mock_log_supabase, mock_preprocess, mock_model, tmp_path, sample_json_input, client
):
    # Create a dummy CSV file
    df = pd.DataFrame([sample_json_input])
    csv_path = tmp_path / "test.csv"
    df.to_csv(csv_path, index=False)

    mock_preprocess.return_value = pd.DataFrame(np.random.rand(1, 3))
    mock_model.predict.return_value = np.array([456.78])

    with open(csv_path, "rb") as f:
        response = client.post(
            "/predict_csv", files={"file": ("test.csv", f, "text/csv")}
        )

    assert response.status_code == 200
    json_resp = response.json()
    assert "predictions" in json_resp
    assert isinstance(json_resp["predictions"], list)
    assert len(json_resp["predictions"]) == 1
    assert json_resp["predictions"][0] == 456.78

    mock_preprocess.assert_called_once()
    mock_model.predict.assert_called_once()
    mock_log_supabase.assert_called_once()


def test_predict_csv_invalid_file_type(client):
    response = client.post(
        "/predict_csv", files={"file": ("test.txt", b"not,a,csv", "text/plain")}
    )
    assert response.status_code == 400
    assert "Only CSV files are supported." in response.json()["detail"]


def test_predict_json_invalid_data(client):
    invalid_data = {"invalid": "data"}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
    assert "detail" in response.text
