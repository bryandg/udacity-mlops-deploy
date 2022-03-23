import numpy as np
import pandas as pd
import pytest
from .ml.model import _compute_slice_metrics, compute_all_slice_metrics, compute_model_metrics


@pytest.fixture
def data():
    """ Simple function to generate some fake Pandas data."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "numeric_feat": [3.14, 2.72, 1.62],
            "cat_feat_1": ["dog", "dog", "cat"],
            "cat_feat_2": ["gray", "green", "green"],
            "y": [0, 0, 1],
            "pred": [0, 1, 0]
        }
    )
    return df


def test__compute_slice_metrics(data):
    """
    test to see that dataframe has the expected columns and length
    """

    slice_df = _compute_slice_metrics(data["y"], data["pred"], data["cat_feat_1"])

    expected_columns = ["class", "sample-size", "proportion-positive-class", "precision", "recall", "fbeta"]
    assert set(slice_df.columns) == set(expected_columns)

    expected_length = len(data["cat_feat_1"].unique())
    assert len(slice_df) == expected_length


def test_compute_all_slice_metrics(data):
    """
    test that return type is pd.DataFrame
    test that each feature is represented
    """

    feature_cols = ["numeric_feat", "cat_feat_1", "cat_feat_2"]
    df = compute_all_slice_metrics(data[feature_cols], data["y"], data["pred"])
    assert type(df) == pd.DataFrame

    for col in feature_cols:
        assert df["class"].str.contains(col).any()


def test_compute_model_metrics(data):
    """
    test that output is expected type
    test that metrics are not null
    """
    output = compute_model_metrics(data["y"], data["pred"])
    assert type(output) == tuple
    for e in output:
        assert e is not np.nan
