"""
Image preprocessing and normalization pipeline.
Handles MNIST pixel normalization and user-submitted digit images (canvas/upload)
with bounding-box extraction, aspect-ratio preservation, and center-of-mass alignment.
"""

from typing import Union
import numpy as np
from PIL import Image
import io


def normalize_pixels(images: np.ndarray) -> np.ndarray:
    """
    Normalizes pixel values from [0, 255] to [0.0, 1.0] as float32.
    
    If inputs are already in range [0.0, 1.0], ensures float32 precision.
    """
    if not isinstance(images, np.ndarray):
        images = np.array(images)

    images = images.astype(np.float32)

    # Check max value to avoid double normalization
    max_val = np.max(images) if images.size > 0 else 0.0
    if max_val > 1.0:
        images = images / 255.0

    return np.clip(images, 0.0, 1.0)


def validate_input_tensor(tensor: np.ndarray, expected_shape: tuple = (28, 28)) -> None:
    """
    Validates input image tensor before feeding into neural network.
    """
    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"Expected numpy.ndarray, got {type(tensor)}")

    if np.isnan(tensor).any() or np.isinf(tensor).any():
        raise ValueError("Input image tensor contains NaN or Inf values.")

    # Accept either (28, 28), (1, 28, 28), or (N, 28, 28)
    if tensor.ndim == 2:
        if tensor.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {tensor.shape}")
    elif tensor.ndim == 3:
        if tensor.shape[1:] != expected_shape:
            raise ValueError(f"Expected image dimensions (*, {expected_shape[0]}, {expected_shape[1]}), got {tensor.shape}")
    elif tensor.ndim == 4:
        if tensor.shape[1:3] != expected_shape:
            raise ValueError(f"Expected shape (*, {expected_shape[0]}, {expected_shape[1]}, *), got {tensor.shape}")
    else:
        raise ValueError(f"Unexpected tensor dimensionality: {tensor.ndim}D")

    # Range check
    min_val, max_val = float(np.min(tensor)), float(np.max(tensor))
    if min_val < -1e-5 or max_val > 1.0001:
        raise ValueError(
            f"Pixel values must be in [0.0, 1.0]. Found min={min_val:.3f}, max={max_val:.3f}."
        )


def center_digit_on_canvas(digit_crop: Image.Image, canvas_size: int = 28, inner_box: int = 20) -> np.ndarray:
    """
    Scales the cropped digit to fit inside an inner bounding box (default 20x20)
    preserving aspect ratio, and centers it on a 28x28 black canvas.
    This aligns with the official MNIST dataset preprocessing methodology.
    """
    orig_w, orig_h = digit_crop.size
    if orig_w == 0 or orig_h == 0:
        return np.zeros((canvas_size, canvas_size), dtype=np.float32)

    # Determine scaling factor to fit inside inner_box
    scale = inner_box / max(orig_w, orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    # High quality resampling
    resized = digit_crop.resize((new_w, new_h), resample=Image.Resampling.BILINEAR)

    # Create 28x28 blank black canvas (0s)
    canvas = Image.new("L", (canvas_size, canvas_size), 0)

    # Compute center position
    paste_x = (canvas_size - new_w) // 2
    paste_y = (canvas_size - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))

    # Center-of-mass fine adjustment
    arr = np.array(canvas, dtype=np.float32)
    total_mass = np.sum(arr)
    if total_mass > 0:
        y_indices, x_indices = np.indices(arr.shape)
        cy = np.sum(y_indices * arr) / total_mass
        cx = np.sum(x_indices * arr) / total_mass

        # Desired center is (13.5, 13.5)
        shift_y = int(round(13.5 - cy))
        shift_x = int(round(13.5 - cx))

        if abs(shift_x) <= 3 and abs(shift_y) <= 3 and (shift_x != 0 or shift_y != 0):
            shifted = np.zeros_like(arr)
            src_y_start = max(0, -shift_y)
            src_y_end = min(canvas_size, canvas_size - shift_y)
            dst_y_start = max(0, shift_y)
            dst_y_end = min(canvas_size, canvas_size + shift_y)

            src_x_start = max(0, -shift_x)
            src_x_end = min(canvas_size, canvas_size - shift_x)
            dst_x_start = max(0, shift_x)
            dst_x_end = min(canvas_size, canvas_size + shift_x)

            shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = arr[
                src_y_start:src_y_end, src_x_start:src_x_end
            ]
            arr = shifted

    return arr


def preprocess_user_image(
    image_input: Union[np.ndarray, Image.Image, bytes, str],
    threshold: float = 30.0,
) -> np.ndarray:
    """
    Preprocesses user-supplied images (canvas drawings or uploads) for MNIST model inference.

    Handles:
    1. Conversion of RGBA/RGB inputs to grayscale.
    2. Polarity detection: ensures background is dark (0) and digit stroke is bright (up to 255).
    3. Noise suppression and bounding box detection.
    4. Aspect-ratio preserving scaling into a 20x20 box inside a 28x28 canvas.
    5. Normalization to [0.0, 1.0] float32.

    Parameters:
        image_input: Numpy array, PIL Image, raw image bytes, or filepath.
        threshold: Intensity threshold for detecting stroke pixels.

    Returns:
        Numpy array of shape (28, 28) with values in [0.0, 1.0].
    """
    # 1. Convert input to PIL Image in grayscale
    if isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
    elif isinstance(image_input, str):
        pil_img = Image.open(image_input)
    elif isinstance(image_input, np.ndarray):
        # Handle RGBA from canvas or RGB from camera
        if image_input.ndim == 3:
            if image_input.shape[2] == 4:
                # Streamlit canvas RGBA
                # If drawing has alpha channel, stroke is where alpha > 0
                alpha = image_input[:, :, 3]
                rgb = image_input[:, :, :3]
                if np.max(alpha) > 0 and np.mean(rgb) == 0:
                    # Canvas drew black on transparent, or white on dark
                    pil_img = Image.fromarray(alpha, mode="L")
                else:
                    pil_img = Image.fromarray(image_input, mode="RGBA").convert("L")
            else:
                pil_img = Image.fromarray(image_input.astype(np.uint8)).convert("L")
        else:
            pil_img = Image.fromarray(image_input.astype(np.uint8), mode="L")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("L")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    img_arr = np.array(pil_img, dtype=np.float32)

    # 2. Check polarity: MNIST is bright digit on black background
    # Check corner and border pixels to determine background color
    h, w = img_arr.shape
    border_pixels = np.concatenate([
        img_arr[0, :], img_arr[-1, :], img_arr[:, 0], img_arr[:, -1]
    ])
    border_mean = np.mean(border_pixels)

    if border_mean > 127.0:
        # User uploaded/drew dark digit on light background -> invert
        img_arr = 255.0 - img_arr

    # Threshold small noise
    img_arr[img_arr < threshold] = 0.0

    # 3. Find bounding box of the digit stroke
    stroke_coords = np.argwhere(img_arr > threshold)
    if stroke_coords.size == 0:
        # Blank canvas / no stroke detected
        return np.zeros((28, 28), dtype=np.float32)

    ymin, xmin = stroke_coords.min(axis=0)
    ymax, xmax = stroke_coords.max(axis=0)

    # Crop to bounding box
    cropped_arr = img_arr[ymin : ymax + 1, xmin : xmax + 1]
    cropped_pil = Image.fromarray(cropped_arr.astype(np.uint8), mode="L")

    # 4. Center on 28x28 canvas using MNIST standard procedure
    centered_arr = center_digit_on_canvas(cropped_pil, canvas_size=28, inner_box=20)

    # 5. Normalize to [0.0, 1.0]
    normalized = normalize_pixels(centered_arr)
    validate_input_tensor(normalized, expected_shape=(28, 28))

    return normalized
