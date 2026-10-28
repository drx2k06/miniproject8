"""
Unit tests for evaluation metrics computation and plot generation.
"""

import os
import json
import pytest
import numpy as np
import matplotlib
matplotlib.use("Agg")

from src.evaluation import (
    evaluate_test_set,
    plot_training_history,
    plot_confusion_matrix,
    plot_sample_predictions,
)
from src.model import build_neural_network


def test_evaluate_test_set_mock(tmp_path):
    """Test full evaluation pipeline on a synthetic test set using a dummy model."""
    output_dir = str(tmp_path / "eval_outputs")
    model = build_neural_network(
        input_shape=(28, 28),
        num_classes=10,
        hidden_units_1=16,
        hidden_units_2=8,
    )

    # 30 synthetic samples
    X_test = np.random.uniform(0.0, 1.0, size=(30, 28, 28)).astype(np.float32)
    y_test = np.array([i % 10 for i in range(30)])

    metrics = evaluate_test_set(model, X_test, y_test, output_dir=output_dir)

    assert "test_accuracy" in metrics
    assert "test_loss" in metrics
    assert "macro_avg" in metrics
    assert "confusion_matrix" in metrics
    assert len(metrics["confusion_matrix"]) == 10
    assert len(metrics["confusion_matrix"][0]) == 10

    # Ensure artifacts exist
    assert os.path.exists(os.path.join(output_dir, "metrics.json"))
    assert os.path.exists(os.path.join(output_dir, "confusion_matrix.png"))
    assert os.path.exists(os.path.join(output_dir, "example_predictions.png"))


def test_plot_training_history(tmp_path):
    """Test generating training vs validation history plot."""
    history = {
        "accuracy": [0.8, 0.9, 0.95],
        "val_accuracy": [0.78, 0.88, 0.93],
        "loss": [0.6, 0.3, 0.15],
        "val_loss": [0.65, 0.35, 0.20],
    }
    plot_path = str(tmp_path / "test_history.png")
    plot_training_history(history, save_path=plot_path)
    assert os.path.exists(plot_path)
    assert os.path.getsize(plot_path) > 1000
