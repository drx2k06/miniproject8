"""
Unit tests for image preprocessing and normalization pipeline.
"""

import pytest
import numpy as np
from PIL import Image

from src.preprocessing import (
    normalize_pixels,
    validate_input_tensor,
    center_digit_on_canvas,
    preprocess_user_image,
)


def test_normalize_pixels_range_255():
    """Test normalization of uint8 0-255 images to float32 0.0-1.0."""
    raw = np.array([[0, 127.5, 255]], dtype=np.float32)
    norm = normalize_pixels(raw)
    assert norm.dtype == np.float32
    assert norm[0, 0] == pytest.approx(0.0)
    assert norm[0, 1] == pytest.approx(0.5, abs=0.01)
    assert norm[0, 2] == pytest.approx(1.0)


def test_normalize_pixels_already_normalized():
    """Verify idempotency if pixels are already in range [0.0, 1.0]."""
    raw = np.array([[0.0, 0.45, 1.0]], dtype=np.float32)
    norm = normalize_pixels(raw)
    assert norm[0, 1] == pytest.approx(0.45)
    assert norm[0, 2] == pytest.approx(1.0)


def test_validate_input_tensor_valid():
    """Ensure valid 28x28 normalized tensors pass validation."""
    valid_2d = np.zeros((28, 28), dtype=np.float32)
    valid_3d = np.ones((1, 28, 28), dtype=np.float32)
    valid_batch = np.random.uniform(0.0, 1.0, size=(5, 28, 28)).astype(np.float32)

    validate_input_tensor(valid_2d)
    validate_input_tensor(valid_3d)
    validate_input_tensor(valid_batch)


def test_validate_input_tensor_invalid_shape():
    """Ensure invalid shapes raise ValueError."""
    bad_shape = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected shape"):
        validate_input_tensor(bad_shape)


def test_validate_input_tensor_invalid_values():
    """Ensure NaN, Inf, and out-of-bound values raise ValueError."""
    with pytest.raises(ValueError, match="NaN or Inf"):
        validate_input_tensor(np.array([[np.nan] * 28] * 28))

    with pytest.raises(ValueError, match="Pixel values must be in"):
        validate_input_tensor(np.full((28, 28), 2.5, dtype=np.float32))


def test_center_digit_on_canvas():
    """Verify aspect-ratio preserving bounding-box centering produces 28x28 canvas."""
    # Create small rectangular dummy digit (10x5)
    digit_pil = Image.new("L", (10, 5), color=255)
    centered = center_digit_on_canvas(digit_pil, canvas_size=28, inner_box=20)
    assert centered.shape == (28, 28)
    assert np.max(centered) > 0.0


def test_preprocess_user_image_canvas_polarity():
    """Verify that light backgrounds with dark strokes are properly inverted to MNIST format."""
    # White canvas (255) with a black stroke (0) in center
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[40:60, 45:55] = 0  # vertical black bar

    processed = preprocess_user_image(img)
    assert processed.shape == (28, 28)
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0
    # Background should now be dark (near 0) and stroke should be bright (near 1.0)
    assert processed[0, 0] == pytest.approx(0.0, abs=0.05)


def test_preprocess_user_image_blank():
    """Verify that blank image returns all zeros without throwing unexpected errors."""
    blank = np.zeros((100, 100), dtype=np.uint8)
    processed = preprocess_user_image(blank)
    assert processed.shape == (28, 28)
    assert np.all(processed == 0.0)
