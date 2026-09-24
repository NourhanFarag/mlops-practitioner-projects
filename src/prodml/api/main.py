import hashlib
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.responses import Response

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    MetadataResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import settings
from prodml.logging_conf import (
    configure_logging,
    get_correlation_id,
    reset_correlation_id,
    set_correlation_id,
)
from prodml.predict import DurationPredictor

configure_logging()

logger = logging.getLogger("prodml.api")


def calculate_sha256(path: Path) -> str:
    sha256 = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.predictor = None
    app.state.model_metadata = None

    try:
        predictor = DurationPredictor.load()

        artifact_hash = calculate_sha256(settings.model_path)

        app.state.predictor = predictor

        app.state.model_metadata = {
            "model_version": predictor.metadata["model_version"],
            "training_date": predictor.metadata["training_date"],
            "feature_names": list(predictor.vectorizer.feature_names_),
            "framework": predictor.metadata["framework"],
            "artifact_hash": artifact_hash,
        }

        logger.info(
            "model loaded at startup: version=%s features=%d",
            predictor.metadata["model_version"],
            len(predictor.vectorizer.feature_names_),
        )

    except Exception:
        logger.exception("application startup model initialization failed")

    yield

    logger.info("application shutdown")


app = FastAPI(
    title="ProdML Ride Duration API",
    lifespan=lifespan,
)


@app.middleware("http")
async def correlation_id_middleware(
    request: Request,
    call_next,
) -> Response:
    correlation_id = str(uuid4())

    # Store the ID on the request itself so exception handlers
    # can still retrieve it.
    request.state.correlation_id = correlation_id

    # Also store it in ContextVar so all logs during this request
    # automatically receive the same correlation ID.
    token = set_correlation_id(correlation_id)

    try:
        logger.info(
            "request received: method=%s path=%s",
            request.method,
            request.url.path,
        )

        response = await call_next(request)

        response.headers["X-Request-ID"] = correlation_id

        logger.info(
            "request completed: status_code=%d",
            response.status_code,
        )

        return response

    finally:
        reset_correlation_id(token)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    correlation_id = getattr(
        request.state,
        "correlation_id",
        "-",
    )

    errors = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]

    logger.error(
        "validation rejected: %s",
        errors,
    )

    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed.",
            "errors": errors,
            "correlation_id": correlation_id,
        },
        headers={
            "X-Request-ID": correlation_id,
        },
    )


@app.exception_handler(Exception)
async def unexpected_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    correlation_id = getattr(
        request.state,
        "correlation_id",
        "-",
    )

    # The general exception handler can run after the middleware
    # ContextVar has already been reset, so temporarily restore
    # the request's correlation ID while logging the traceback.
    token = set_correlation_id(correlation_id)

    try:
        logger.exception(
            "unexpected server error: %s",
            exc,
        )
    finally:
        reset_correlation_id(token)

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "correlation_id": correlation_id,
        },
        headers={
            "X-Request-ID": correlation_id,
        },
    )


@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health(
    request: Request,
) -> HealthResponse | JSONResponse:
    predictor = request.app.state.predictor

    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "model_loaded": False,
            },
        )

    return HealthResponse(
        status="healthy",
        model_loaded=True,
    )


@app.get(
    "/metadata",
    response_model=MetadataResponse,
)
async def metadata(
    request: Request,
) -> MetadataResponse | JSONResponse:
    metadata_value = request.app.state.model_metadata

    if metadata_value is None:
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Model metadata is unavailable because " "the model is not loaded."
                )
            },
        )

    return MetadataResponse(**metadata_value)


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    predictor = request.app.state.predictor

    if predictor is None:
        raise RuntimeError("Model is not loaded.")

    start = time.perf_counter()

    prediction = predictor.predict_one(
        {
            "PU_DO": payload.PU_DO,
            "trip_distance": payload.trip_distance,
        }
    )

    latency_ms = (time.perf_counter() - start) * 1000

    correlation_id = get_correlation_id()

    logger.info(
        "prediction served: prediction=%.3f latency_ms=%.3f",
        prediction,
        latency_ms,
    )

    return PredictionResponse(
        prediction=prediction,
        model_version=predictor.metadata["model_version"],
        correlation_id=correlation_id,
        latency_ms=latency_ms,
    )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
) -> BatchPredictionResponse:
    predictor = request.app.state.predictor

    if predictor is None:
        raise RuntimeError("Model is not loaded.")

    features = [
        {
            "PU_DO": item.PU_DO,
            "trip_distance": item.trip_distance,
        }
        for item in payload.items
    ]

    start = time.perf_counter()

    predictions = predictor.predict_batch(features)

    latency_ms = (time.perf_counter() - start) * 1000

    correlation_id = get_correlation_id()

    logger.info(
        "batch prediction served: count=%d latency_ms=%.3f",
        len(predictions),
        latency_ms,
    )

    return BatchPredictionResponse(
        predictions=predictions,
        model_version=predictor.metadata["model_version"],
        correlation_id=correlation_id,
        latency_ms=latency_ms,
    )
