import logging
import pickle
import time
from collections.abc import Callable
from functools import wraps
from typing import Any

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.config import settings

logger = logging.getLogger(__name__)

FeatureDict = dict[str, str | float]


def timed(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)

        elapsed_ms = (time.perf_counter() - start) * 100

        logger.info("%s served in %.3f ms", func.__name__, elapsed_ms)

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
        try:
            with settings.model_path.open("rb") as file:
                artifact = pickle.load(file)

        except Exception:
            logger.exception("model load failed")
            raise

        return cls(
            vectorizer=artifact["vectorizer"],
            model=artifact["model"],
        )

    # clean interface for API single prediction
    @timed
    def predict_one(self, features: FeatureDict) -> float:
        logger.debug("feature vector: %s", features)

        trip_distance = features.get("trip_distance")
        if isinstance(trip_distance, (int, float)) and trip_distance > 100:
            logger.warning("trip distance outside training range: %.2f", trip_distance)

        X = self.vectorizer.transform([features])
        prediction = self.model.predict(X)[0]

        return float(prediction)

    # clean interface for API batch prediction
    def predict_batch(self, features: list[FeatureDict]) -> list[float]:
        logger.debug("batch feature vextors: %s", features)

        X = self.vectorizer.transform(features)
        prediction = self.model.predict(X)

        return [float(value) for value in prediction]
