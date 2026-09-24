import numpy as np
import pandas as pd
import pytest

from prodml.features import prepare_features
from prodml.predict import DurationPredictor


def test_prepare_feature_creates_duration_and_pu_do() -> None:
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": [pd.Timestamp("2026-01-01 10:00:00")],
            "lpep_dropoff_datetime": [pd.Timestamp("2026-01-01 10:10:00")],
            "PULocationID": [74],
            "DOLocationID": [75],
            "trip_distance": [3.5],
        }
    )

    result = prepare_features(df)

    assert len(result) == 1
    assert result.iloc[0]["duration"] == pytest.approx(10.0)
    assert result.iloc[0]["PU_DO"] == "74_75"


@pytest.mark.parametrize(
    (
        "pickup_id",
        "dropoff_id",
        "trip_distance",
        "expected_rows",
        "expected_pu_do",
    ),
    [
        (np.nan, 75, 3.5, 1, "nan_75"),
        (74, 75, 0.0, 0, None),
    ],
)
def test_feature_edge_cases(
    pickup_id: float,
    dropoff_id: int,
    trip_distance: float,
    expected_rows: int,
    expected_pu_do: str | None,
) -> None:
    df = pd.DataFrame(
        {
            "lpep_pickup_datetime": [pd.Timestamp("2026-01-01 10:00:00")],
            "lpep_dropoff_datetime": [pd.Timestamp("2026-01-01 10:10:00")],
            "PULocationID": [pickup_id],
            "DOLocationID": [dropoff_id],
            "trip_distance": [trip_distance],
        }
    )

    result = prepare_features(df)

    assert len(result) == expected_rows

    if expected_pu_do is not None:
        assert result.iloc[0]["PU_DO"] == expected_pu_do


def test_unseen_pu_do_pair_does_not_break_prediction(
    trained_model: DurationPredictor,
) -> None:
    features = {
        "PU_DO": "999_998",
        "trip_distance": 3.5,
    }

    prediction = trained_model.predict_one(features)

    assert isinstance(prediction, float)
