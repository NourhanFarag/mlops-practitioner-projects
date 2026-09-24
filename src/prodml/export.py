import logging
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import settings
from prodml.logging_conf import configure_logging
from prodml.predict import DurationPredictor

logger = logging.getLogger("prodml.export")


def export_to_onnx(output_path: Path | None = None) -> Path:
    predictor = DurationPredictor.load()

    n_features = len(predictor.vectorizer.feature_names_)

    initial_types = [
        (
            "features",
            FloatTensorType([None, n_features]),
        )
    ]

    onnx_model = convert_sklearn(
        predictor.model,
        initial_types=initial_types,
    )

    target_path = output_path or settings.onnx_model_path

    target_path.parent.mkdir(parents=True, exist_ok=True)

    target_path.write_bytes(onnx_model.SerializeToString())

    logger.info("ONNX model exported: path=%s features=%d", target_path, n_features)

    return target_path


def main() -> None:
    configure_logging()
    export_to_onnx()


if __name__ == "__main__":
    main()
