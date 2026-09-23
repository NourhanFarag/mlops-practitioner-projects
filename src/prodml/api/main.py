import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.responses import Response

from prodml.logging_conf import (
    configure_logging,
    reset_correlation_id,
    set_correlation_id,
)
from prodml.predict import DurationPredictor

configure_logging()

logger = logging.getLogger("prodml.api")


class PredictionRequest(BaseModel):
    PU_DO: str
    trip_distance: float


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.predictor = DurationPredictor.load()
    yield


app = FastAPI(title="ProdML Ride Duration API", lifespan=lifespan)


@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next) -> Response:
    correlation_id = str(uuid4())
    token = set_correlation_id(correlation_id)

    try:
        logger.info(
            "request received: method=%s path=%s", request.method, request.url.path
        )

        response = await call_next(request)
        response.headers["X-Request-ID"] = correlation_id

        logger.info("request completed: status_code=%d", response.status_code)

        return response

    finally:
        reset_correlation_id(token)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.error("validation rejected:%s", exc.errors())

    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.post("/predict")
async def predict(payload: PredictionRequest, request: Request) -> dict[str, float]:
    predictor: DurationPredictor = request.app.state.predictor

    prediction = predictor.predict_one(
        {
            "PU_DO": payload.PU_DO,
            "trip_distance": payload.trip_distance,
        }
    )

    return {"duration": prediction}
