"""
Evaluation script for NeuralVision AI.
Loads the trained model from disk, evaluates performance on the untouched MNIST test set,
prints the classification report, and updates metrics and visualization outputs.
"""

import argparse
import os
import json
import numpy as np

from src.data_loader import load_mnist_data
from src.preprocessing import normalize_pixels
from src.model import load_model_from_disk
from src.evaluation import evaluate_test_set


def run_evaluation(
    model_path: str = "models/neuralvision_model.keras",
    output_dir: str = "outputs",
):
    print("=" * 70)
    print("  NeuralVision AI - Handwritten Digit Recognition Evaluation")
    print("  Developer: Daniel Richard Xavier")
    print("=" * 70)

    # 1. Load model
    print(f"\n[1/3] Loading saved model from: {model_path}...")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Please train the model first by running:\n"
            f"  python train.py"
        )
    model = load_model_from_disk(model_path)
    print("      Model loaded successfully.")

    # 2. Load test data
    print("\n[2/3] Loading untouched MNIST test set...")
    _, _, (X_test_raw, y_test) = load_mnist_data()
    X_test = normalize_pixels(X_test_raw)
    print(f"      Loaded {len(X_test):,} test images.")

    # 3. Evaluate
    print(f"\n[3/3] Evaluating model performance and computing metrics...")
    metrics = evaluate_test_set(model, X_test, y_test, output_dir=output_dir)

    print("\n" + "=" * 70)
    print("  CLASSIFICATION PERFORMANCE ON MNIST TEST SET (10,000 samples)")
    print("=" * 70)
    print(f"  Overall Test Accuracy: {metrics['test_accuracy_percentage']}")
    print(f"  Overall Test Loss:     {metrics['test_loss']:.4f}")
    print(f"  Macro Precision:       {metrics['macro_avg']['precision']:.4f}")
    print(f"  Macro Recall:          {metrics['macro_avg']['recall']:.4f}")
    print(f"  Macro F1-Score:        {metrics['macro_avg']['f1_score']:.4f}")
    print("-" * 70)
    print("  PER-DIGIT METRICS:")
    print(f"  {'Digit':<8} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
    print("  " + "-" * 52)
    for digit in range(10):
        d_metrics = metrics["per_digit_metrics"][str(digit)]
        print(
            f"  {digit:<8} "
            f"{d_metrics['precision']:<12.4f} "
            f"{d_metrics['recall']:<12.4f} "
            f"{d_metrics['f1_score']:<12.4f} "
            f"{d_metrics['support']:<8}"
        )
    print("-" * 70)
    print("  TOP CONFUSED DIGIT PAIRS:")
    for pair in metrics["top_confused_pairs"]:
        print(f"    - {pair['description']}")
    print("=" * 70)
    print(f"  Metrics file: {os.path.join(output_dir, 'metrics.json')}")
    print(f"  Visualizations saved to: {output_dir}/")
    print("=" * 70)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NeuralVision AI model.")
    parser.add_argument("--model-path", type=str, default="models/neuralvision_model.keras", help="Path to model")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    args = parser.parse_args()

    run_evaluation(model_path=args.model_path, output_dir=args.output_dir)
