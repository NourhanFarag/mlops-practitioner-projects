# Production ML API — NYC Taxi Trip Duration

This project turns a simple NYC green-taxi trip-duration regression model into a production-style ML service. It includes reproducible feature engineering and training, structured JSON logging with correlation IDs, Pickle and ONNX serialization checks, a FastAPI prediction API, a coverage-gated test suite, and a non-root multi-stage Docker image published to Docker Hub. The model predicts trip duration in minutes from a pickup/dropoff location pair (`PU_DO`) and trip distance.

## Quickstart

A prediction can be served without cloning this repository or installing Python.

### 1. Pull the image

```bash
docker pull nourhan17/prodml-api:0.1.0
```

### 2. Start the API

```bash
docker run --rm -d --name prodml-api -p 8000:8000 nourhan17/prodml-api:0.1.0
```

### 3. Request a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "PU_DO": "74_75",
    "trip_distance": 3.5
  }'
```

Example response:

```json
{
  "prediction": 17.43947267130485,
  "model_version": "0.1.0",
  "correlation_id": "6a5c31b8-ee17-4087-ae03-c1b24ed3673c",
  "latency_ms": 2.0912959989800584
}
```

The exact `correlation_id` and latency change for every request.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Check whether the model is loaded |
| `GET` | `/metadata` | Return model version, training metadata, features, framework, and artifact hash |
| `POST` | `/predict` | Make a single prediction |
| `POST` | `/predict/batch` | Make predictions for a batch of inputs |
| `GET` | `/docs` | Open the FastAPI Swagger interface |

## Example Input

The prediction API accepts:

```json
{
  "PU_DO": "74_75",
  "trip_distance": 3.5
}
```

`PU_DO` combines pickup and dropoff location IDs:

```text
PULocationID + "_" + DOLocationID
```

For example:

```text
74 + "_" + 75 -> 74_75
```

The API validates that:

```text
0 < trip_distance < 200
```

## Repository Structure

```text
mlops-practitioner-projects/
├── .dockerignore
├── .gitignore
├── pyproject.toml
├── README.md
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── models/
│   └── model.pkl
│
├── notebooks/
│   └── 00-baseline.ipynb
│
├── reports/
│   └── module-1.md
│
├── src/
│   └── prodml/
│       ├── __init__.py
│       ├── benchmark.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       ├── train.py
│       └── api/
│           ├── __init__.py
│           ├── main.py
│           └── schemas.py
│
└── tests/
    ├── conftest.py
    ├── test_api.py
    ├── test_features.py
    ├── test_predict.py
    └── test_serialization.py
```

The optional Litestar `/predict` implementation is maintained on the
`litestar-predict` branch.

## Testing

The test suite covers feature engineering, prediction behavior, API behavior,
and Pickle-to-ONNX prediction parity.

Run:

```bash
pytest
```

The project enforces a minimum test coverage gate of:

```text
70%
```

The current suite contains 14 tests and reaches approximately 72.7% coverage.

## Docker Compose

For local development with the model directory mounted read-only:

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

Check the service:

```bash
curl http://localhost:8000/health
```

Stop it with:

```bash
docker compose -f docker/docker-compose.yml down
```

## Model

The current baseline uses:

```text
DictVectorizer
+
LinearRegression
```

with:

```text
PU_DO
trip_distance
```

as model features.

Baseline validation performance:

```text
MAE:  4.22 minutes
RMSE: 6.51 minutes
```

More detailed implementation decisions, serialization benchmarks, Docker image
measurements, and limitations are documented in:

```text
reports/module-1.md
```
