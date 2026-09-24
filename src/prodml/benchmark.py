import logging
import time

import numpy as np
import onnxruntime as ort

from prodml.config import settings
from prodml.data import load_data, split_data
from prodml.features import prepare_features
from prodml.logging_conf import configure_logging
from prodml.predict import DurationPredictor
from prodml.train import feature_to_dict

logger = logging.getLogger("prodml.benchmark")


def measure_latency(function, runs: int = 100) -> tuple[float, float]:
    latencies_ms = []

    for _ in range(runs):
        start = time.perf_counter()

        function()

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    mean_latency = float(np.mean(latencies_ms))
    p95_latency = float(np.percentile(latencies_ms, 95))

    return mean_latency, p95_latency


def benchmark_models(runs: int = 100) -> dict[str, dict[str, float]]:
    predictor = DurationPredictor.load()

    raw_df = load_data()
    prepared_df = prepare_features(raw_df)

    _, validation_df = split_data(prepared_df)

    validation_sample = validation_df.iloc[:500]

    feature_dicts = feature_to_dict(validation_sample)

    X = predictor.vectorizer.transform(feature_dicts)
    X_onnx = X.toarray().astype(np.float32)

    session = ort.InferenceSession(
        str(settings.onnx_model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # Warm-up runs so initialization does not distort the benchmark.
    for _ in range(5):
        predictor.model.predict(X)
        session.run(
            None,
            {input_name: X_onnx},
        )

    pickle_mean, pickle_p95 = measure_latency(
        lambda: predictor.model.predict(X),
        runs=runs,
    )

    onnx_mean, onnx_p95 = measure_latency(
        lambda: session.run(
            None,
            {input_name: X_onnx},
        ),
        runs=runs,
    )

    return {
        "pickle": {
            "mean_ms": pickle_mean,
            "p95_ms": pickle_p95,
        },
        "onnx": {
            "mean_ms": onnx_mean,
            "p95_ms": onnx_p95,
        },
    }


def main() -> None:
    configure_logging()

    results = benchmark_models()

    logger.info(
        "pickle benchmark: mean_ms=%.3f p95_ms=%.3f",
        results["pickle"]["mean_ms"],
        results["pickle"]["p95_ms"],
    )

    logger.info(
        "onnx benchmark: mean_ms=%.3f p95_ms=%.3f",
        results["onnx"]["mean_ms"],
        results["onnx"]["p95_ms"],
    )


if __name__ == "__main__":
    main()
