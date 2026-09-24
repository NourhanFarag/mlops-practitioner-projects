import numpy as np
import onnxruntime as ort

from prodml.data import load_data, split_data
from prodml.export import export_to_onnx
from prodml.features import prepare_features
from prodml.predict import DurationPredictor
from prodml.train import feature_to_dict


def test_pickle_onnx_parity() -> None:
    onnx_path = export_to_onnx()

    predictor = DurationPredictor.load()

    raw_df = load_data()
    prepared_df = prepare_features(raw_df)

    _, validation_df = split_data(prepared_df)
    validation_sample = validation_df.iloc[:500]

    feature_dicts = feature_to_dict(validation_sample)

    X = predictor.vectorizer.transform(feature_dicts)

    pred_pkl = predictor.model.predict(X)

    X_onnx = X.toarray().astype(np.float32)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name

    pred_onnx = session.run(None, {input_name: X_onnx})[0].ravel()

    max_absolute_difference = np.max(np.abs(pred_pkl - pred_onnx))

    assert np.allclose(pred_pkl, pred_onnx, atol=1e-4), (
        "Pickle and ONNX predictions differ. "
        f"Max absolute difference: {max_absolute_difference}"
    )
