"""
Evaluation, metrics computation, and visualization generation for NeuralVision AI.
Calculates real test accuracy, loss, confusion matrix, and generates publication plots.
"""

from typing import Dict, Any, List, Tuple
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from src.prediction import predict_batch


def evaluate_test_set(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """
    Evaluates the trained model on the untouched test set and computes all metrics.

    Parameters:
        model: Trained tf.keras.Model.
        X_test: Test images of shape (N, 28, 28) normalized to [0.0, 1.0].
        y_test: Test integer labels of shape (N,).
        output_dir: Directory where plots and metrics.json will be saved.

    Returns:
        Structured dictionary containing all evaluation metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Evaluate loss and accuracy directly with Keras
    eval_results = model.evaluate(X_test, y_test, batch_size=128, verbose=0)
    test_loss = float(eval_results[0])
    test_accuracy = float(eval_results[1])

    # 2. Predict on test set
    y_pred, y_probs = predict_batch(model, X_test, batch_size=128)

    # 3. Overall Precision, Recall, F1
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    # 4. Detailed per-digit classification report
    report_dict = classification_report(
        y_test, y_pred, digits=4, output_dict=True, zero_division=0
    )

    per_digit_metrics = {}
    for digit in range(10):
        digit_str = str(digit)
        if digit_str in report_dict:
            per_digit_metrics[digit_str] = {
                "precision": float(report_dict[digit_str]["precision"]),
                "recall": float(report_dict[digit_str]["recall"]),
                "f1_score": float(report_dict[digit_str]["f1-score"]),
                "support": int(report_dict[digit_str]["support"]),
            }

    # 5. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # 6. Top confused pairs (off-diagonal elements)
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    confused_indices = np.argsort(cm_copy.ravel())[::-1]
    top_confused = []
    for idx in confused_indices[:6]:
        true_class = int(idx // 10)
        pred_class = int(idx % 10)
        count = int(cm[true_class, pred_class])
        if count > 0:
            top_confused.append({
                "true_digit": true_class,
                "predicted_digit": pred_class,
                "misclassified_count": count,
                "description": f"Digit '{true_class}' misclassified as '{pred_class}' ({count} times)",
            })

    metrics = {
        "model_name": "NeuralVision Feedforward Neural Network (MLP)",
        "dataset": "MNIST Test Set (untouched)",
        "test_samples": int(len(y_test)),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_accuracy, 4),
        "test_accuracy_percentage": f"{test_accuracy * 100:.2f}%",
        "macro_avg": {
            "precision": round(float(precision_macro), 4),
            "recall": round(float(recall_macro), 4),
            "f1_score": round(float(f1_macro), 4),
        },
        "weighted_avg": {
            "precision": round(float(precision_weighted), 4),
            "recall": round(float(recall_weighted), 4),
            "f1_score": round(float(f1_weighted), 4),
        },
        "per_digit_metrics": per_digit_metrics,
        "confusion_matrix": cm.tolist(),
        "top_confused_pairs": top_confused,
    }

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    # Generate and save plots
    plot_confusion_matrix(cm, os.path.join(output_dir, "confusion_matrix.png"))
    plot_sample_predictions(
        X_test, y_test, y_pred, y_probs, os.path.join(output_dir, "example_predictions.png")
    )

    return metrics


def plot_training_history(history_dict: Dict[str, List[float]], save_path: str = "outputs/training_history.png") -> None:
    """
    Plots training vs validation accuracy and loss over epochs.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    epochs = range(1, len(history_dict["accuracy"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Accuracy subplot
    ax1.plot(epochs, history_dict["accuracy"], "o-", color="#2563eb", linewidth=2, label="Training Accuracy")
    ax1.plot(epochs, history_dict["val_accuracy"], "s--", color="#10b981", linewidth=2, label="Validation Accuracy")
    ax1.set_title("Training vs Validation Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_ylim([min(history_dict["accuracy"]) * 0.95, 1.01])
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right", frameon=True)

    # Loss subplot
    ax2.plot(epochs, history_dict["loss"], "o-", color="#dc2626", linewidth=2, label="Training Loss")
    ax2.plot(epochs, history_dict["val_loss"], "s--", color="#f59e0b", linewidth=2, label="Validation Loss")
    ax2.set_title("Training vs Validation Loss", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Cross-Entropy Loss", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, save_path: str = "outputs/confusion_matrix.png") -> None:
    """
    Renders an annotated heatmap of the 10x10 confusion matrix.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
        square=True,
        linewidths=0.5,
        linecolor="#e2e8f0",
        ax=ax,
        annot_kws={"size": 10, "weight": "bold"},
    )

    ax.set_title("MNIST Test Set Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Digit Class", fontsize=12, labelpad=8)
    ax.set_ylabel("True Digit Class", fontsize=12, labelpad=8)
    ax.set_xticklabels([str(i) for i in range(10)], fontsize=10)
    ax.set_yticklabels([str(i) for i in range(10)], fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_sample_predictions(
    X_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    save_path: str = "outputs/example_predictions.png",
    num_samples: int = 15,
) -> None:
    """
    Renders a grid of sample test predictions showing image, ground truth, prediction, and confidence.
    Includes both correctly classified samples and misclassified cases for educational analysis.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    # Select both correct and incorrect cases
    correct_idx = np.where(y_test == y_pred)[0]
    incorrect_idx = np.where(y_test != y_pred)[0]

    # Pick 10 correct and 5 incorrect (or as available)
    np.random.seed(42)
    chosen_correct = np.random.choice(correct_idx, size=min(10, len(correct_idx)), replace=False)
    chosen_incorrect = (
        np.random.choice(incorrect_idx, size=min(5, len(incorrect_idx)), replace=False)
        if len(incorrect_idx) > 0
        else np.array([], dtype=int)
    )
    selected_indices = np.concatenate([chosen_correct, chosen_incorrect])

    rows = 3
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(13, 8), dpi=150)
    axes = axes.flatten()

    for idx, sample_idx in enumerate(selected_indices[:rows * cols]):
        ax = axes[idx]
        img = X_test[sample_idx]
        true_lbl = int(y_test[sample_idx])
        pred_lbl = int(y_pred[sample_idx])
        conf = float(y_probs[sample_idx][pred_lbl])

        is_correct = true_lbl == pred_lbl
        color = "#16a34a" if is_correct else "#dc2626"
        status = "CORRECT" if is_correct else "MISCLASSIFIED"

        ax.imshow(img, cmap="gray", interpolation="nearest")
        ax.set_title(
            f"True: {true_lbl} | Pred: {pred_lbl}\nConf: {conf * 100:.1f}%\n[{status}]",
            fontsize=9,
            fontweight="bold",
            color=color,
            pad=4,
        )
        ax.axis("off")

    plt.suptitle("NeuralVision AI - Sample Test Predictions (Correct & Incorrect)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
