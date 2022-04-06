from typing import Union

import pandas as pd
from fastapi import FastAPI
from joblib import load
from pydantic import BaseModel, Field

from starter.ml.data import process_data
from starter.ml.model import inference


class InputData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")
    # potentially make salary an optional field
    salary: str


CAT_FEATURES = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


app = FastAPI()


@app.get("/")
async def welcome():
    return {"Welcome": "This API has an endpoint for model inference."}


@app.post("/inference/")
async def predict(data: InputData):
    data = pd.DataFrame(data.dict(), index=[0])

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
