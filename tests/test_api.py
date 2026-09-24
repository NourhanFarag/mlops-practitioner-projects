from fastapi.testclient import TestClient

from prodml.api.schemas import PredictionResponse


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["model_loaded"] is True


def test_predict_happy_path(
    client: TestClient, sample_features: dict[str, str | float]
) -> None:
    response = client.post("/predict", json=sample_features)
    assert response.status_code == 200

    body = response.json()

    assert isinstance(body["prediction"], float)
    assert body["model_version"] == "test-1.0"
    assert body["latency_ms"] >= 0
    assert body["correlation_id"]

    assert response.headers["X-Request-ID"] == body["correlation_id"]


def test_predict_response_matches_schema(
    client: TestClient, sample_features: dict[str, str | float]
) -> None:
    response = client.post("/predict", json=sample_features)

    assert response.status_code == 200

    parsed_response = PredictionResponse.model_validate(response.json())

    assert isinstance(parsed_response.prediction, float)


def test_predict_invalid_payload_returns_422(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict",
        json={
            "PU_DO": "74_75",
            "trip_distance": -5,
        },
    )

    assert response.status_code == 422

    body = response.json()

    assert body["detail"] == "Request validation failed."
    assert body["errors"]
    assert body["correlation_id"]

    assert response.headers["X-Request-ID"] == body["correlation_id"]


def test_metadata_returns_model_information(
    client: TestClient,
) -> None:
    response = client.get("/metadata")

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "test-1.0"
    assert body["framework"] == "scikit-learn"
    assert body["artifact_hash"] == "test-artifact-hash"
    assert isinstance(body["feature_names"], list)
    assert body["feature_names"]


def test_batch_prediction_happy_path(
    client: TestClient,
) -> None:
    response = client.post(
        "/predict/batch",
        json={
            "items": [
                {
                    "PU_DO": "74_75",
                    "trip_distance": 3.5,
                },
                {
                    "PU_DO": "41_42",
                    "trip_distance": 2.0,
                },
            ]
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["predictions"]) == 2
    assert all(isinstance(prediction, float) for prediction in body["predictions"])
    assert body["model_version"] == "test-1.0"
    assert body["latency_ms"] >= 0
    assert response.headers["X-Request-ID"] == body["correlation_id"]
