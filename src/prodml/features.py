import pandas as pd

from prodml.config import settings


def add_duration(
    df: pd.DataFrame,
) -> pd.DataFrame:  # function expects dataframe and returns dataframe
    result = df.copy()

    result["duration"] = (
        result["lpep_dropoff_datetime"] - result["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    return result


def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    duration_filter = (df["duration"] >= settings.min_duration) & (
        df["duration"] <= settings.max_duration
    )

    distance_filter = (df["trip_distance"] > settings.min_trip_distance) & (
        df["trip_distance"] <= settings.max_trip_distance
    )

    return df[duration_filter & distance_filter].copy()


def pu_do(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["PU_DO"] = (
        result["PULocationID"].astype(str) + "_" + result["DOLocationID"].astype(str)
    )

    return result


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    result = add_duration(df)
    result = clean_rows(result)
    result = pu_do(result)
    return result
