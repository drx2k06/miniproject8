"""
Unit tests for data validation, label checking, and dataset summaries.
"""

import pytest
import numpy as np

from src.data_loader import (
    validate_dataset_shapes,
    validate_labels,
    get_dataset_summary,
    get_sample_digits_per_class,
)


def test_validate_dataset_shapes_valid():
    """Verify validation passes for properly shaped datasets."""
    imgs = np.zeros((10, 28, 28), dtype=np.uint8)
    lbls = np.zeros(10, dtype=np.uint8)
    validate_dataset_shapes(imgs, lbls, expected_samples=10)


def test_validate_dataset_shapes_invalid():
    """Verify ValueError is raised on mismatched counts or wrong dimensions."""
    imgs = np.zeros((10, 32, 32), dtype=np.uint8)
    lbls = np.zeros(10, dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid image array shape"):
        validate_dataset_shapes(imgs, lbls)

    imgs_correct = np.zeros((10, 28, 28), dtype=np.uint8)
    lbls_mismatch = np.zeros(8, dtype=np.uint8)
    with pytest.raises(ValueError, match="Mismatched number"):
        validate_dataset_shapes(imgs_correct, lbls_mismatch)


def test_validate_labels():
    """Verify label validator accepts [0-9] and rejects out-of-range labels."""
    valid_labels = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    validate_labels(valid_labels)

    invalid_labels = np.array([0, 1, 10, -1])
    with pytest.raises(ValueError, match="Labels contain invalid digit classes"):
        validate_labels(invalid_labels)


def test_get_dataset_summary():
    """Verify dataset summary metadata dictionary structure."""
    train = (np.zeros((20, 28, 28)), np.array([i % 10 for i in range(20)]))
    val = (np.zeros((5, 28, 28)), np.array([i % 10 for i in range(5)]))
    test = (np.zeros((5, 28, 28)), np.array([i % 10 for i in range(5)]))

    summary = get_dataset_summary(train, val, test)
    assert summary["dataset_name"] == "MNIST Handwritten Digits"
    assert summary["splits"]["total_samples"] == 30
    assert summary["image_dimensions"] == [28, 28]


def test_get_sample_digits_per_class():
    """Verify sample extractor pulls representatives for all 10 classes."""
    images = np.arange(20 * 28 * 28).reshape(20, 28, 28)
    labels = np.array([i % 10 for i in range(20)])

    samples = get_sample_digits_per_class(images, labels, samples_per_class=1)
    assert len(samples) == 10
    for digit in range(10):
        assert len(samples[digit]) == 1
        assert samples[digit][0].shape == (28, 28)
