import os
from typing import Union

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


def to_hyphens(string: str) -> str:
    return string.replace("_", "-")


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str
    # potentially make salary an optional field
    salary: str

    class Config:

        alias_generator = to_hyphens

        schema_extra = {
            "example": {
                "age": 49,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "Asian-Pac-Islander",
                "sex": "Female",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United States",
                "salary": "<=50K",
            }
        }


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


app = FastAPI()


@app.get("/")
async def welcome():
    return {"Welcome": "This API has an endpoint for model inference."}


@app.post("/inference/")
async def predict(data: InputData):
    data = pd.DataFrame(data.dict(by_alias=True), index=[0])

    lb = load("model/lb.joblib")
    encoder = load("model/encoder.joblib")
    X, y, encoder_ret, lb_ret = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = load("model/model.joblib")
    return {"prediction": str(inference(model, X)[0])}
