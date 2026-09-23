import pickle
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.config import settings

FeatureDict = dict[str, str | float]


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)

        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.6f} seconds")

        return result

    return wrapper


# groups everything prediction-related together
class DurationPredictor:  # To pass both objects to every function
    def __init__(
        self,
        vectorizer: DictVectorizer,
        model: LinearRegression,
    ) -> None:
        self.vectorizer = vectorizer
        self.model = model

    @classmethod
    def load(cls) -> "DurationPredictor":
        with settings.model_path.open("rb") as file:
            artifact = pickle.load(file)

        return cls(
            vectorizer=artifact["vectorizer"],
            model=artifact["model"],
        )

    # clean interface for API single prediction
    @timed
    def predict_one(
        self,
        features: FeatureDict,
    ) -> float:
        X = self.vectorizer.transform([features])
        prediction = self.model.predict(X)[0]

        return float(prediction)

    # clean interface for API batch prediction
    def predict_batch(
        self,
        features: list[FeatureDict],
    ) -> list[float]:
        X = self.vectorizer.transform(features)
        prediction = self.model.predict(X)

        return [float(value) for value in prediction]
