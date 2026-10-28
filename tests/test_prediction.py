"""
Unit tests for single-image and batch prediction inference pipeline.
"""

import pytest
import numpy as np

from src.model import build_neural_network
from src.prediction import predict_digit, predict_batch


@pytest.fixture(scope="module")
def trained_test_model():
    """Provides a small compiled model for prediction testing."""
    return build_neural_network(
        input_shape=(28, 28),
        num_classes=10,
        hidden_units_1=32,
        hidden_units_2=16,
    )


def test_predict_digit_structure(trained_test_model):
    """Verify return format, types, and probability constraints for single prediction."""
    dummy_digit = np.random.uniform(0.0, 1.0, size=(28, 28)).astype(np.float32)

    result = predict_digit(trained_test_model, dummy_digit)

    assert "predicted_digit" in result
    assert isinstance(result["predicted_digit"], int)
    assert 0 <= result["predicted_digit"] <= 9

    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0

    assert "probabilities" in result
    assert len(result["probabilities"]) == 10
    total_prob = sum(result["probabilities"].values())
    assert total_prob == pytest.approx(1.0, abs=1e-4)

    assert "top_3" in result
    assert len(result["top_3"]) == 3
    assert result["top_3"][0]["digit"] == result["predicted_digit"]


def test_predict_digit_invalid_shape(trained_test_model):
    """Verify ValueError is raised if input shape is not 28x28."""
    bad_img = np.zeros((20, 20), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected image shape"):
        predict_digit(trained_test_model, bad_img)


def test_predict_batch(trained_test_model):
    """Verify batch inference returns correct predictions and probability array shapes."""
    batch_images = np.random.uniform(0.0, 1.0, size=(8, 28, 28)).astype(np.float32)

    preds, probs = predict_batch(trained_test_model, batch_images, batch_size=4)

    assert preds.shape == (8,)
    assert probs.shape == (8, 10)
    assert np.all(preds >= 0) and np.all(preds <= 9)
    for p in probs:
        assert np.sum(p) == pytest.approx(1.0, abs=1e-4)
