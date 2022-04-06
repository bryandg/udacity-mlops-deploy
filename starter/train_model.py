# Script to train machine learning model.

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import ml.model as mo
from ml.data import process_data

data = pd.read_csv("../data/census.csv")

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20, random_state=0)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder_ret, lb_ret = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train and save a model.
model = mo.train_model(X_train, y_train)
dump(encoder, "../model/encoder.joblib")
dump(lb, "../model/lb.joblib")
dump(model, "../model/model.joblib")

# make predictions
preds = mo.inference(model, X_test)
# TODO: compute and store metrics

slice_df = mo.compute_all_slice_metrics(
    test[cat_features].reset_index(drop=True),
    pd.Series(y_test, name="y").reset_index(drop=True),
    pd.Series(preds, name="pred").reset_index(drop=True)
)
slice_df.to_csv("../data/slice_metrics.csv", index=False)
