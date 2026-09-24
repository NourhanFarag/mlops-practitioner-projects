import pytest
from fastapi.testclient import TestClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from prodml.api.main import app
from prodml.predict import DurationPredictor


@pytest.fixture
def sample_features() -> dict[str, str | float]:
    return {
        "PU_DO": "74_75",
        "trip_distance": 3.5,
    }


@pytest.fixture(scope="session")
def trained_model() -> DurationPredictor:
    training_features = [
        {
            "PU_DO": "74_75",
            "trip_distance": 1.0,
        },
        {
            "PU_DO": "74_75",
            "trip_distance": 2.0,
        },
        {
            "PU_DO": "41_42",
            "trip_distance": 1.5,
        },
        {
            "PU_DO": "41_42",
            "trip_distance": 3.0,
        },
    ]

    targets = [
        10.0,
        14.0,
        12.0,
        20.0,
    ]

    vectorizer = DictVectorizer()
    X = vectorizer.fit_transform(training_features)

    model = LinearRegression()
    model.fit(X, targets)

    return DurationPredictor(
        vectorizer=vectorizer,
        model=model,
        metadata={
            "model_version": "test-1.0",
            "training_date": "2026-01-01T00:00:00+00:00",
            "framework": "scikit-learn",
        },
    )


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch, trained_model: DurationPredictor
) -> TestClient:
    monkeypatch.setattr("prodml.api.main.DurationPredictor.load", lambda: trained_model)

    monkeypatch.setattr(
        "prodml.api.main.calculate_sha256", lambda path: "test-artifact-hash"
    )

    with TestClient(app) as test_client:
        yield test_client
