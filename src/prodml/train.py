import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import prepare_features

# X
CATEGORICAL = ["PU_DO"]
NUMERICAL = ["trip_distance"]
# Y
TARGET = "duration"


def feature_to_dict(df: pd.DataFrame) -> list[dict[str, object]]:
    return df[CATEGORICAL + NUMERICAL].to_dict(orient="records")


def train_model(train_df: pd.DataFrame) -> tuple[DictVectorizer, LinearRegression]:
    train_dicts = feature_to_dict(train_df)

    vectorizer = DictVectorizer(sparse=True)

    X_train = vectorizer.fit_transform(train_dicts)
    y_train = train_df[TARGET]

    model = LinearRegression()
    model.fit(X_train, y_train)

    return vectorizer, model


def evaluate_model(
    val_df: pd.DataFrame,
    vectorizer: DictVectorizer,
    model: LinearRegression,
) -> tuple[float, float]:

    val_dicts = feature_to_dict(val_df)
    X_val = vectorizer.transform(val_dicts)
    y_val = val_df[TARGET]

    y_pred = model.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    mae = mean_absolute_error(y_val, y_pred)

    return mae, rmse


def save_model(vectorizer: DictVectorizer, model: LinearRegression) -> None:
    settings.model_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "vectorizer": vectorizer,
        "model": model,
    }

    with open(settings.model_path, "wb") as file:
        pickle.dump(artifact, file)


def main() -> None:

    df = load_data()

    df = prepare_features(df)

    train_df, val_df = split_data(df)

    vectorizer, model = train_model(train_df)

    mae, rmse = evaluate_model(val_df, vectorizer, model)

    save_model(vectorizer, model)

    print(f"Validation MAE: {mae:.2f} minutes")
    print(f"Validation RMSE: {rmse:.2f} minutes")
    print(f"Model saved to: {settings.model_path}")


if __name__ == "__main__":
    main()
