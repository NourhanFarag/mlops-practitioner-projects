import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from litestar import Litestar, Request, Response, post

from prodml.api.schemas import PredictionRequest, PredictionResponse
from prodml.logging_conf import (
    configure_logging,
    reset_correlation_id,
    set_correlation_id,
)
from prodml.predict import DurationPredictor

configure_logging()

logger = logging.getLogger("prodml.litestar")


@asynccontextmanager
async def lifespan(app: Litestar) -> AsyncIterator[None]:
    app.state.predictor = None

    try:
        app.state.predictor = DurationPredictor.load()

        logger.info(
            "model loaded at startup: version=%s",
            app.state.predictor.metadata["model_version"],
        )

    except Exception:
        logger.exception("application startup model initialization failed")

    yield

    logger.info("application shutdown")


@post(
    path="/predict",
    status_code=200,
)
async def predict(
    data: PredictionRequest,
    request: Request,
) -> Response[PredictionResponse]:
    predictor = request.app.state.predictor

    if predictor is None:
        raise RuntimeError("Model is not loaded.")

    correlation_id = str(uuid4())
    token = set_correlation_id(correlation_id)

    try:
        start = time.perf_counter()

        prediction = predictor.predict_one(
            {
                "PU_DO": data.PU_DO,
                "trip_distance": data.trip_distance,
            }
        )

        latency_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "prediction served: prediction=%.3f latency_ms=%.3f",
            prediction,
            latency_ms,
        )

        response_body = PredictionResponse(
            prediction=prediction,
            model_version=predictor.metadata["model_version"],
            correlation_id=correlation_id,
            latency_ms=latency_ms,
        )

        return Response(
            content=response_body,
            status_code=200,
            headers={
                "X-Request-ID": correlation_id,
            },
        )

    finally:
        reset_correlation_id(token)


app = Litestar(
    route_handlers=[predict],
    lifespan=[lifespan],
)
