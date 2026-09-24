from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    PU_DO: str = Field(
        min_length=3,
        description="Combined pickup and dropoff location IDs, for example 74_75.",
    )

    trip_distance: float = Field(
        gt=0,
        lt=200,
        description="Trip distance in miles.",
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PU_DO": "74_75",
                "trip_distance": 3.5,
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionRequest(BaseModel):
    items: list[PredictionRequest]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {
                        "PU_DO": "74_75",
                        "trip_distance": 3.5,
                    },
                    {
                        "PU_DO": "41_42",
                        "trip_distance": 7.2,
                    },
                ]
            }
        }
    )


class BatchPredictionResponse(BaseModel):
    predictions: list[float]
    model_version: str
    correlation_id: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class MetadataResponse(BaseModel):
    model_version: str
    training_date: str
    feature_names: list[str]
    framework: str
    artifact_hash: str
