"""
Data loading and dataset validation for MNIST handwritten digits.
Ensures separation of training, validation, and untouched testing sets.
"""

from typing import Dict, Tuple
import numpy as np
from sklearn.model_selection import train_test_split


def load_mnist_data(
    val_split: float = 0.166667,  # Default: 10,000 validation samples from 60,000 train
    random_state: int = 42,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """
    Loads MNIST dataset from Keras and partitions into train, validation, and test sets.
    
    The official MNIST test set (10,000 samples) remains completely untouched for final evaluation.
    The validation set is extracted strictly from the training portion.

    Parameters:
        val_split: Fraction of the training set used for validation.
        random_state: Random seed for reproducible splitting.

    Returns:
        (X_train, y_train), (X_val, y_val), (X_test, y_test)
        Pixel values are raw uint8 in range [0, 255].
    """
    try:
        import tensorflow as tf
        mnist = tf.keras.datasets.mnist
        (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load MNIST dataset from TensorFlow: {exc}.\n"
            "Ensure an active internet connection to download the dataset on first run, "
            "or verify TensorFlow installation."
        ) from exc

    # Validate dataset dimensions
    validate_dataset_shapes(X_train_full, y_train_full, expected_samples=60000)
    validate_dataset_shapes(X_test, y_test, expected_samples=10000)
    validate_labels(y_train_full)
    validate_labels(y_test)

    # Stratified validation split strictly from training data
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_split,
        random_state=random_state,
        stratify=y_train_full,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def validate_dataset_shapes(images: np.ndarray, labels: np.ndarray, expected_samples: int = None) -> None:
    """
    Validates that images have shape (N, 28, 28) and labels have shape (N,).
    """
    if images.ndim != 3 or images.shape[1:] != (28, 28):
        raise ValueError(f"Invalid image array shape {images.shape}. Expected (N, 28, 28).")
    if labels.ndim != 1:
        raise ValueError(f"Invalid labels array shape {labels.shape}. Expected 1D array of shape (N,).")
    if len(images) != len(labels):
        raise ValueError(
            f"Mismatched number of images ({len(images)}) and labels ({len(labels)})."
        )
    if expected_samples is not None and len(images) != expected_samples:
        # Warning/informative check
        pass


def validate_labels(labels: np.ndarray) -> None:
    """
    Validates that labels are integers in the range [0, 9].
    """
    unique_labels = np.unique(labels)
    if not np.all(np.isin(unique_labels, np.arange(10))):
        invalid = [int(lbl) for lbl in unique_labels if lbl not in range(10)]
        raise ValueError(f"Labels contain invalid digit classes: {invalid}. Expected 0-9.")


def get_dataset_summary(
    train_set: Tuple[np.ndarray, np.ndarray],
    val_set: Tuple[np.ndarray, np.ndarray],
    test_set: Tuple[np.ndarray, np.ndarray],
) -> Dict:
    """
    Returns structured metadata and distribution summary of the loaded dataset.
    """
    X_train, y_train = train_set
    X_val, y_val = val_set
    X_test, y_test = test_set

    train_dist = {int(d): int(np.sum(y_train == d)) for d in range(10)}
    val_dist = {int(d): int(np.sum(y_val == d)) for d in range(10)}
    test_dist = {int(d): int(np.sum(y_test == d)) for d in range(10)}

    return {
        "dataset_name": "MNIST Handwritten Digits",
        "image_dimensions": [28, 28],
        "color_channels": 1,
        "classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "splits": {
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "test_samples": len(X_test),
            "total_samples": len(X_train) + len(X_val) + len(X_test),
        },
        "class_distributions": {
            "train": train_dist,
            "val": val_dist,
            "test": test_dist,
        },
    }


def get_sample_digits_per_class(images: np.ndarray, labels: np.ndarray, samples_per_class: int = 1) -> Dict[int, list]:
    """
    Extracts representative sample images for each digit class (0-9).
    """
    samples = {}
    for digit in range(10):
        indices = np.where(labels == digit)[0][:samples_per_class]
        samples[digit] = [images[idx] for idx in indices]
    return samples
