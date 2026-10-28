"""
Training pipeline for NeuralVision AI.
Trains an explainable Feedforward Neural Network on MNIST with validation monitoring,
evaluates on the untouched test set, generates evaluation artifacts, and exports sample digits.
"""

import argparse
import os
import random
import time
import numpy as np
from PIL import Image

# Ensure reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from src.data_loader import load_mnist_data, get_dataset_summary, get_sample_digits_per_class
from src.preprocessing import normalize_pixels
from src.model import build_neural_network, save_trained_model, get_layer_parameter_table
from src.evaluation import evaluate_test_set, plot_training_history


def export_sample_digits(X_test: np.ndarray, y_test: np.ndarray, sample_dir: str = "sample_digits") -> None:
    """
    Saves clean sample images for digits 0-9 into sample_digits/ directory for quick testing and UI upload.
    """
    os.makedirs(sample_dir, exist_ok=True)
    samples = get_sample_digits_per_class(X_test, y_test, samples_per_class=1)
    for digit, imgs in samples.items():
        if imgs:
            img = imgs[0]
            # Convert to PIL and save
            pil_img = Image.fromarray((img * 255.0).astype(np.uint8), mode="L")
            pil_img.save(os.path.join(sample_dir, f"sample_digit_{digit}.png"))
    print(f"[Info] Exported representative sample digits (0-9) to '{sample_dir}/'.")


def train_neural_network(
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    val_split: float = 0.166667,  # 10,000 validation samples from 60,000
    model_path: str = "models/neuralvision_model.keras",
    output_dir: str = "outputs",
    seed: int = 42,
):
    """
    Main training execution function.
    """
    import tensorflow as tf
    tf.random.set_seed(seed)

    print("=" * 70)
    print("  NeuralVision AI - Handwritten Digit Recognition Training Pipeline")
    print("  Developer: Daniel Richard Xavier")
    print("=" * 70)

    # 1. Load dataset
    print(f"\n[1/6] Loading MNIST dataset (Validation split: {val_split:.2%})...")
    (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test) = load_mnist_data(
        val_split=val_split, random_state=seed
    )

    summary = get_dataset_summary(
        (X_train_raw, y_train), (X_val_raw, y_val), (X_test_raw, y_test)
    )
    print(f"      Training samples:   {summary['splits']['train_samples']:,}")
    print(f"      Validation samples: {summary['splits']['val_samples']:,}")
    print(f"      Test samples:       {summary['splits']['test_samples']:,} (Untouched)")

    # 2. Preprocess & Normalize
    print("\n[2/6] Preprocessing and normalizing pixel intensities to [0.0, 1.0]...")
    X_train = normalize_pixels(X_train_raw)
    X_val = normalize_pixels(X_val_raw)
    X_test = normalize_pixels(X_test_raw)
    print(f"      X_train shape: {X_train.shape}, dtype: {X_train.dtype}, range: [{X_train.min():.1f}, {X_train.max():.1f}]")
    print(f"      X_val shape:   {X_val.shape}, dtype: {X_val.dtype}, range: [{X_val.min():.1f}, {X_val.max():.1f}]")
    print(f"      X_test shape:  {X_test.shape}, dtype: {X_test.dtype}, range: [{X_test.min():.1f}, {X_test.max():.1f}]")

    # 3. Build Neural Network Architecture
    print("\n[3/6] Building Multilayer Perceptron (Feedforward Neural Network)...")
    model = build_neural_network(
        input_shape=(28, 28),
        num_classes=10,
        hidden_units_1=128,
        hidden_units_2=64,
        dropout_rate_1=0.2,
        dropout_rate_2=0.1,
        learning_rate=learning_rate,
    )

    param_table = get_layer_parameter_table(model)
    total_params = sum(p["trainable_parameters"] for p in param_table)
    print(f"      Architecture: 784 -> Dense(128, ReLU) -> Dropout(0.2) -> Dense(64, ReLU) -> Dropout(0.1) -> Dense(10, Softmax)")
    print(f"      Total Trainable Parameters: {total_params:,}")

    # 4. Train Model
    print(f"\n[4/6] Training model for {epochs} epochs (Batch size: {batch_size})...")
    start_time = time.time()

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1,
    )

    training_time = time.time() - start_time
    print(f"      Training completed in {training_time:.2f} seconds.")

    # 5. Save model and history plot
    print(f"\n[5/6] Saving trained model and training history...")
    save_trained_model(model, model_path)
    print(f"      Saved model to: {model_path}")

    history_dict = {
        "accuracy": [float(v) for v in history.history["accuracy"]],
        "val_accuracy": [float(v) for v in history.history["val_accuracy"]],
        "loss": [float(v) for v in history.history["loss"]],
        "val_loss": [float(v) for v in history.history["val_loss"]],
    }
    plot_training_history(history_dict, os.path.join(output_dir, "training_history.png"))
    print(f"      Saved training curve plot to: {os.path.join(output_dir, 'training_history.png')}")

    # 6. Evaluate on Untouched Test Set
    print("\n[6/6] Evaluating on UNTOUCHED test set...")
    metrics = evaluate_test_set(model, X_test, y_test, output_dir=output_dir)

    # Attach training metadata
    metrics["training_metadata"] = {
        "epochs_trained": len(history_dict["accuracy"]),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_duration_seconds": round(training_time, 2),
        "total_trainable_parameters": total_params,
        "early_stopping_triggered": len(history_dict["accuracy"]) < epochs,
    }

    # Re-save with training metadata included
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(metrics, f, indent=4)

    # Export sample digits for testing
    export_sample_digits(X_test, y_test, sample_dir="sample_digits")

    print("\n" + "=" * 70)
    print("  EVALUATION SUMMARY ON UNTOUCHED TEST SET")
    print("=" * 70)
    print(f"  Test Accuracy:    {metrics['test_accuracy_percentage']}")
    print(f"  Test Loss:        {metrics['test_loss']:.4f}")
    print(f"  Macro Precision:  {metrics['macro_avg']['precision']:.4f}")
    print(f"  Macro Recall:     {metrics['macro_avg']['recall']:.4f}")
    print(f"  Macro F1-Score:   {metrics['macro_avg']['f1_score']:.4f}")
    print("=" * 70)
    print(f"  Model saved to:   {model_path}")
    print(f"  Metrics saved to: {os.path.join(output_dir, 'metrics.json')}")
    print("=" * 70)

    return model, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuralVision AI digit recognition model.")
    parser.add_argument("--epochs", type=int, default=12, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.166667, help="Validation split from training set")
    parser.add_argument("--model-path", type=str, default="models/neuralvision_model.keras", help="Path to save model")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for plots and metrics")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    train_neural_network(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        model_path=args.model_path,
        output_dir=args.output_dir,
        seed=args.seed,
    )
