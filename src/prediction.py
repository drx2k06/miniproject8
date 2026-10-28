"""
Prediction and inference engine for handwritten digit recognition.
Handles single-sample and batch predictions, confidence calculation, and top-k rankings.
"""

from typing import Dict, Any, List, Union, Tuple
import numpy as np


def predict_digit(model, image: np.ndarray) -> Dict[str, Any]:
    """
    Executes a model forward pass on a single preprocessed 28x28 normalized image.

    Parameters:
        model: Trained tf.keras.Model.
        image: 2D numpy array of shape (28, 28) with values in [0.0, 1.0].

    Returns:
        Dict containing:
            - predicted_digit (int): Class with highest probability.
            - confidence (float): Highest probability in [0.0, 1.0].
            - probabilities (Dict[int, float]): Map from digit 0-9 to probability.
            - top_3 (List[Dict]): Top 3 likely digits with probabilities.
    """
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.float32)

    if image.ndim == 2:
        if image.shape != (28, 28):
            raise ValueError(f"Expected image shape (28, 28), got {image.shape}")
        input_tensor = np.expand_dims(image, axis=0)  # Shape: (1, 28, 28)
    elif image.ndim == 3 and image.shape == (1, 28, 28):
        input_tensor = image
    else:
        raise ValueError(f"Unsupported input dimension for single prediction: {image.shape}")

    # Validate range
    if np.min(input_tensor) < -1e-4 or np.max(input_tensor) > 1.0001:
        raise ValueError("Input image pixels must be normalized to [0.0, 1.0].")

    # Run inference
    probs = model.predict(input_tensor, verbose=0)[0]
    probs = np.array(probs, dtype=np.float64)

    # Normalize if needed to avoid precision drift
    sum_probs = np.sum(probs)
    if sum_probs > 0:
        probs = probs / sum_probs

    predicted_digit = int(np.argmax(probs))
    confidence = float(probs[predicted_digit])

    probabilities_dict = {int(d): float(probs[d]) for d in range(10)}

    # Top 3 predictions
    sorted_indices = np.argsort(probs)[::-1]
    top_3: List[Dict[str, Any]] = [
        {"digit": int(idx), "probability": float(probs[idx]), "percentage": f"{probs[idx] * 100:.2f}%"}
        for idx in sorted_indices[:3]
    ]

    return {
        "predicted_digit": predicted_digit,
        "confidence": confidence,
        "confidence_percentage": f"{confidence * 100:.2f}%",
        "probabilities": probabilities_dict,
        "probabilities_array": probs.tolist(),
        "top_3": top_3,
    }


def predict_batch(model, images: np.ndarray, batch_size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs batch inference on a dataset of images.

    Parameters:
        model: Trained tf.keras.Model.
        images: Array of shape (N, 28, 28) normalized to [0.0, 1.0].
        batch_size: Batch size for prediction.

    Returns:
        (predicted_classes, probability_matrix)
    """
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Expected batch shape (N, 28, 28), got {images.shape}")

    probs = model.predict(images, batch_size=batch_size, verbose=0)
    preds = np.argmax(probs, axis=1)
    return preds, probs
