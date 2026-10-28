"""
Unit tests for neural network architecture, forward pass, and persistence.
"""

import os
import pytest
import numpy as np

from src.model import (
    build_neural_network,
    save_trained_model,
    load_model_from_disk,
    get_architecture_details,
)


@pytest.fixture(scope="module")
def small_model():
    """Fixture providing a lightweight compiled neural network."""
    return build_neural_network(
        input_shape=(28, 28),
        num_classes=10,
        hidden_units_1=32,
        hidden_units_2=16,
        dropout_rate_1=0.1,
        dropout_rate_2=0.0,
    )


def test_model_layers_and_output_shape(small_model):
    """Verify layer configuration and output tensor shape."""
    assert small_model.input_shape == (None, 28, 28)
    assert small_model.output_shape == (None, 10)


def test_model_forward_pass_smoke(small_model):
    """Verify forward inference on a batch of dummy inputs."""
    batch_size = 4
    dummy_input = np.random.uniform(0.0, 1.0, size=(batch_size, 28, 28)).astype(np.float32)

    predictions = small_model.predict(dummy_input, verbose=0)
    assert predictions.shape == (batch_size, 10)

    # Probabilities across classes must sum to 1.0 (softmax property)
    for sample_probs in predictions:
        assert np.sum(sample_probs) == pytest.approx(1.0, abs=1e-4)
        assert np.all(sample_probs >= 0.0)
        assert np.all(sample_probs <= 1.0)


def test_model_save_and_reload(small_model, tmp_path):
    """Verify model can be saved to disk and reloaded with identical predictions."""
    save_path = str(tmp_path / "test_model.keras")
    save_trained_model(small_model, save_path)
    assert os.path.exists(save_path)

    reloaded_model = load_model_from_disk(save_path)
    assert reloaded_model.output_shape == (None, 10)

    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 28, 28)).astype(np.float32)
    orig_preds = small_model.predict(dummy_input, verbose=0)
    reloaded_preds = reloaded_model.predict(dummy_input, verbose=0)

    np.testing.assert_allclose(orig_preds, reloaded_preds, rtol=1e-5, atol=1e-5)


def test_architecture_details():
    """Verify viva educational details dictionary contains all required concepts."""
    details = get_architecture_details()
    required_keys = [
        "input_features",
        "weights_and_biases",
        "flatten_layer",
        "hidden_layers",
        "relu_activation",
        "dropout_regularization",
        "softmax_output",
        "loss_function",
        "backpropagation",
        "optimizer",
        "epochs_and_batch_size",
    ]
    for key in required_keys:
        assert key in details
        assert len(details[key]) > 20
