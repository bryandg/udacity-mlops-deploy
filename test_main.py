import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture
def n_data():
    return {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
        "salary": "<=50K",
    }


@pytest.fixture
def p_data():
    return {
        "age": 44,
        "workclass": "Private",
        "fnlgt": 198282,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 15024,
        "capital-loss": 0,
        "hours-per-week": 60,
        "native-country": "United-States",
        "salary": ">50K",
    }


def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Welcome": "This API has an endpoint for model inference."}


def test_inference_positive_class(p_data):
    r = client.post("/inference/", json=p_data)
    assert r.status_code == 200
    assert r.json() == {"prediction": "1"}


def test_inference_negative_class(n_data):
    r = client.post("/inference/", json=n_data)
    assert r.status_code == 200
    assert r.json() == {"prediction": "0"}
