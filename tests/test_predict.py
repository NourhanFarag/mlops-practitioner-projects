from prodml.predict import DurationPredictor


def test_prediction_returns_float(
    trained_model: DurationPredictor,
    sample_features: dict[str, str | float],
) -> None:
    prediction = trained_model.predict_one(sample_features)
    assert isinstance(prediction, float)


def test_prediction_is_in_sane_range(
    trained_model: DurationPredictor,
    sample_features: dict[str, str | float],
) -> None:
    prediction = trained_model.predict_one(sample_features)

    assert 0.0 < prediction < 120.0


def test_prediction_is_deterministic(
    trained_model: DurationPredictor,
    sample_features: dict[str, str | float],
) -> None:
    first_prediction = trained_model.predict_one(sample_features)
    second_prediction = trained_model.predict_one(sample_features)

    assert first_prediction == second_prediction
