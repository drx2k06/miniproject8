"""
Neural Network model definition, compilation, and architectural inspection.
Implements an explainable Multilayer Perceptron (Feedforward Neural Network) for MNIST.
"""

from typing import Dict, Any, Tuple
import os
import numpy as np


def build_neural_network(
    input_shape: Tuple[int, int] = (28, 28),
    num_classes: int = 10,
    hidden_units_1: int = 128,
    hidden_units_2: int = 64,
    dropout_rate_1: float = 0.2,
    dropout_rate_2: float = 0.1,
    learning_rate: float = 0.001,
):
    """
    Constructs and compiles an explainable, CPU-friendly Feedforward Neural Network.

    Architecture:
        1. Flatten: Reshapes 28x28 2D input matrix to 784-element 1D vector.
        2. Dense 1: 128 neurons, ReLU activation (extracts stroke primitives and edges).
        3. Dropout 1: Rate 0.2 (regularization to combat overfitting).
        4. Dense 2: 64 neurons, ReLU activation (extracts higher-level digit features/loops).
        5. Dropout 2: Rate 0.1 (secondary regularization).
        6. Dense Output: 10 neurons, Softmax activation (outputs class probabilities for digits 0-9).

    Parameters:
        input_shape: Dimensions of each input image (default: 28x28).
        num_classes: Number of target digit classes (default: 10).
        hidden_units_1: Number of neurons in primary hidden layer.
        hidden_units_2: Number of neurons in secondary hidden layer.
        dropout_rate_1: Dropout fraction after primary hidden layer.
        dropout_rate_2: Dropout fraction after secondary hidden layer.
        learning_rate: Initial learning rate for Adam optimizer.

    Returns:
        Compiled tf.keras.Model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses

    model = models.Sequential(
        [
            layers.Input(shape=input_shape, name="input_image"),
            layers.Flatten(name="flatten_vector"),
            layers.Dense(
                hidden_units_1,
                activation="relu",
                name="dense_hidden_1",
            ),
            layers.Dropout(dropout_rate_1, name="dropout_1"),
            layers.Dense(
                hidden_units_2,
                activation="relu",
                name="dense_hidden_2",
            ),
            layers.Dropout(dropout_rate_2, name="dropout_2"),
            layers.Dense(
                num_classes,
                activation="softmax",
                name="output_probabilities",
            ),
        ],
        name="NeuralVision_MLP",
    )

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    loss = losses.SparseCategoricalCrossentropy()

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"],
    )

    return model


def save_trained_model(model, filepath: str) -> str:
    """
    Saves the trained model to disk using standard Keras .keras format.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    model.save(filepath)
    return filepath


def load_model_from_disk(filepath: str):
    """
    Loads a saved model from disk.
    """
    import tensorflow as tf

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found at: {filepath}")

    model = tf.keras.models.load_model(filepath)
    return model


def get_architecture_details() -> Dict[str, Any]:
    """
    Returns educational documentation of each neural network component for student viva.
    """
    return {
        "model_name": "NeuralVision Feedforward Neural Network (MLP)",
        "input_features": (
            "28x28 = 784 individual pixel intensity values. Each value is normalized to the "
            "continuous range [0.0, 1.0], representing grayscale brightness."
        ),
        "weights_and_biases": (
            "Weights (W) scale the input signals, representing the synaptic strength between neurons. "
            "Biases (b) provide an offset allowing the activation function to shift horizontally. "
            "Each neuron computes z = (W · x) + b."
        ),
        "flatten_layer": (
            "Converts the 2D spatial grid (28 rows × 28 columns) into a flat 1D vector of 784 numbers. "
            "Contains 0 trainable parameters."
        ),
        "hidden_layers": (
            "Layers situated between the input and output. Layer 1 (128 neurons) detects stroke segments, "
            "slopes, and curvatures. Layer 2 (64 neurons) synthesizes these parts into digit concepts "
            "(e.g., loops in 8 and 0, horizontal bars in 7 and 5)."
        ),
        "relu_activation": (
            "Rectified Linear Unit: f(x) = max(0, x). Introduces non-linearity so the network can solve "
            "non-linear boundaries. Prevents vanishing gradients for positive inputs and enables fast computation."
        ),
        "dropout_regularization": (
            "Randomly zeros a fraction of neuron outputs during training passes (p=0.2 and p=0.1). "
            "Prevents co-adaptation of neurons and forces the network to learn robust, generalized features."
        ),
        "softmax_output": (
            "Softmax function: σ(z)_i = exp(z_i) / Σ exp(z_j). Converts 10 raw logits into a normalized "
            "probability distribution over digits 0 through 9, such that all 10 probabilities sum to exactly 1.0."
        ),
        "loss_function": (
            "Sparse Categorical Cross-Entropy: L = -log(p_true). Quantifies the penalty when predicted "
            "probabilities diverge from the actual ground truth integer class label."
        ),
        "backpropagation": (
            "The backward pass applying the calculus Chain Rule to compute the partial derivative of the loss "
            "with respect to every weight and bias (∂L/∂W, ∂L/∂b)."
        ),
        "optimizer": (
            "Adam (Adaptive Moment Estimation): Updates weights using running estimates of the first moment (mean) "
            "and second moment (uncentered variance) of gradients with individual adaptive learning rates."
        ),
        "epochs_and_batch_size": (
            "Epoch: One complete traversal of all training examples. "
            "Batch Size: Number of samples processed in a single forward/backward pass before updating weights. "
            "Mini-batch size 64 balances gradient stability with CPU training speed."
        ),
    }


def get_layer_parameter_table(model) -> list:
    """
    Extracts structured layer parameter breakdown.
    """
    breakdown = []
    for layer in model.layers:
        trainable_params = sum(
            int(np.prod(w.shape)) for w in layer.trainable_weights
        )
        out_shape = getattr(layer, "output_shape", None)
        if out_shape is None:
            try:
                out_shape = getattr(layer, "_output_shape", "dynamic")
            except Exception:
                out_shape = "dynamic"
        breakdown.append({
            "layer_name": layer.name,
            "layer_type": layer.__class__.__name__,
            "output_shape": str(out_shape),
            "trainable_parameters": trainable_params,
        })
    return breakdown
