# 🧠 NeuralVision AI — Handwritten Digit Recognition Using Neural Networks

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![TensorFlow 2.16+](https://img.shields.io/badge/TensorFlow-2.21.0-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.66.1-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Pytest-22%20Passed-brightgreen.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**NeuralVision AI** is an end-to-end, CPU-optimized deep learning system designed to classify handwritten digits (0–9) using an explainable Multilayer Perceptron (MLP) trained on the benchmark MNIST dataset. Developed for **Mini Project 8 (Neural Network Use Case)**, it combines strict experimental discipline, robust image preprocessing, full test coverage, and an interactive Streamlit web application.

- **Developer:** Daniel Richard Xavier
- **GitHub Repository:** [https://github.com/drx2k06/miniproject8.git](https://github.com/drx2k06/miniproject8.git)
- **Local Directory:** `C:\Users\drx20\Projects\miniproject8`

---

## 📌 Table of Contents
1. [Abstract](#-abstract)
2. [Introduction & Problem Statement](#-introduction--problem-statement)
3. [Project Objectives](#-project-objectives)
4. [Dataset Specifications](#-dataset-specifications)
5. [Neural Network Architecture](#-neural-network-architecture)
6. [Methodology & Preprocessing](#-methodology--preprocessing)
7. [Evaluation Results](#-evaluation-results)
8. [Interactive Streamlit Application](#-interactive-streamlit-application)
9. [Installation & Setup](#-installation--setup)
10. [How to Train the Model](#-how-to-train-the-model)
11. [How to Run Evaluations](#-how-to-run-evaluations)
12. [How to Launch Streamlit](#-how-to-launch-streamlit)
13. [Automated Testing](#-automated-testing)
14. [Limitations & Future Enhancements](#-limitations--future-enhancements)
15. [Documentation Index](#-documentation-index)
16. [Developer Contributions & License](#-developer-contributions--license)

---

## 📄 Abstract

Handwritten digit recognition is a foundational computer vision problem that bridges biological visual perception and gradient-based mathematical optimization. This project implements a fully explainable, CPU-friendly Feedforward Neural Network (Multilayer Perceptron) trained on the benchmark MNIST dataset.

The system features:
- A multi-stage image preprocessing pipeline that performs automatic polarity detection, noise thresholding, aspect-ratio preserving bounding-box scaling, and center-of-mass alignment.
- A 109,386-parameter feedforward neural network comprising a 784-dimensional input layer, two dense hidden layers (128 and 64 units) with ReLU activation, two intermediate Dropout regularization layers (rates 0.2 and 0.1), and a 10-unit Softmax output layer.
- Optimization using Adam and sparse categorical cross-entropy loss with validation-based early stopping.
- Evaluated on the strictly isolated 10,000-sample test set, the model achieves a verified **97.65% test accuracy**, **0.0801 test loss**, and a macro F1-score of **0.9764**, training in under **30 seconds on CPU**.
- A feature-rich Streamlit web dashboard allowing live drawing on an HTML5 canvas, image uploads, step-by-step preprocessing inspection, and real-time probability bar charts.

---

## 🎯 Introduction & Problem Statement

Optical character recognition (OCR) for numerical digits is widely applied in postal sorting, automated bank check clearing, and administrative document digitization. However, handwriting exhibits substantial intra-class variance in stroke thickness, curvature, and orientation, alongside subtle inter-class similarities (e.g., an unclosed 4 resembling a 9, or a hooked 7 resembling a 2).

Given a grayscale handwritten digit image $\mathbf{X} \in \mathbb{R}^{28 \times 28}$, the objective is to learn a mapping function $f: \mathbb{R}^{28 \times 28} \rightarrow [0, 1]^{10}$ that outputs normalized posterior class probabilities $\mathbf{p} = [p_0, p_1, \dots, p_9]$ such that $\sum_{i=0}^9 p_i = 1.0$, with the predicted class given by $\hat{y} = \arg\max_i p_i$.

---

## 🎯 Project Objectives

1. **Strict Data Partitioning:** Maintain strict isolation between training ($49,999$ samples), validation ($10,001$ samples), and untouched test ($10,000$ samples) partitions to ensure zero data leakage.
2. **Transparent Architecture:** Build an explainable Multilayer Perceptron emphasizing fundamental mathematical concepts (weights, biases, activations, backpropagation, and cross-entropy).
3. **CPU Optimization:** Deliver sub-minute training and $<15$ ms per-sample inference on standard laptop CPUs.
4. **Empirical Rigor:** Record real measured metrics in structured JSON without simulated or fabricated performance.
5. **Interactive Interface:** Deliver a classroom-ready Streamlit web application supporting live drawing, file uploads, and probability distributions.
6. **Viva Readiness:** Provide student-friendly documentation and question-and-answer guides for technical examination.

---

## 📊 Dataset Specifications

The model is trained and benchmarked on the official **MNIST Handwritten Digit Dataset**:
- **Source:** Scanned binary images from the National Institute of Standards and Technology.
- **Total Population:** 70,000 grayscale images.
- **Resolution:** 28 × 28 pixels (784 continuous input features).
- **Pixel Values:** Grayscale $[0, 255]$ ($0 = \text{black background}$, $255 = \text{white stroke}$).
- **Splits:**
  - Training Partition: 49,999 samples.
  - Validation Partition: 10,001 samples (stratified split from training set).
  - Untouched Test Set: 10,000 samples (held out in total isolation).

---

## 📐 Neural Network Architecture

```
[Input: 28×28] ──> [Flatten: 784] ──> [Dense: 128 (ReLU)] ──> [Dropout: 0.2]
       ──> [Dense: 64 (ReLU)] ──> [Dropout: 0.1] ──> [Dense: 10 (Softmax)]
```

### Layer Configuration and Parameters

| Layer Name | Layer Type | Output Shape | Trainable Parameters | Mathematical Formula |
| :--- | :--- | :--- | :--- | :--- |
| `input_image` | InputLayer | `(None, 28, 28)` | 0 | Raw input tensor |
| `flatten_vector` | Flatten | `(None, 784)` | 0 | $\mathbf{a}^{[0]} = \text{vec}(\mathbf{X})$ |
| `dense_hidden_1` | Dense | `(None, 128)` | $784 \times 128 + 128 = \mathbf{100,480}$ | $\mathbf{a}^{[1]} = \max(0, \mathbf{a}^{[0]}\mathbf{W}^{[1]} + \mathbf{b}^{[1]})$ |
| `dropout_1` | Dropout | `(None, 128)` | 0 | Mask rate = 0.2 (training only) |
| `dense_hidden_2` | Dense | `(None, 64)` | $128 \times 64 + 64 = \mathbf{8,256}$ | $\mathbf{a}^{[2]} = \max(0, \mathbf{a}^{[1]}\mathbf{W}^{[2]} + \mathbf{b}^{[2]})$ |
| `dropout_2` | Dropout | `(None, 64)` | 0 | Mask rate = 0.1 (training only) |
| `output_probabilities` | Dense | `(None, 10)` | $64 \times 10 + 10 = \mathbf{650}$ | $p_i = \frac{e^{z_i}}{\sum_{j=0}^9 e^{z_j}}$ |
| **Total** | | | **109,386** | All parameters fully trainable |

---

## ⚙️ Methodology & Preprocessing

Real-world user inputs require specialized preprocessing to match the distribution of the MNIST training set:
1. **Grayscale Conversion:** Strips alpha and color channels to single-channel intensity.
2. **Polarity Detection & Inversion:** Samples border pixels. If mean border brightness $> 127$, the image is automatically inverted so the background is dark and the digit stroke is bright.
3. **Thresholding & Bounding-Box Detection:** Isolates stroke pixels $> 30$ and crops tightly around the digit boundaries.
4. **Aspect-Ratio Preserving Scaling:** Resizes the cropped digit to fit within a $20 \times 20$ box while maintaining its original aspect ratio.
5. **Center-of-Mass Alignment:** Pastes the $20 \times 20$ digit onto a $28 \times 28$ black canvas and shifts it slightly to align the center of mass at $(13.5, 13.5)$.
6. **Pixel Normalization:** Divides pixel intensities by $255.0$ to produce continuous values in $[0.0, 1.0]$ float32.

---

## 📈 Evaluation Results

The model was evaluated on the complete, untouched 10,000-sample MNIST test set:

### Overall Test Set Metrics
- **Overall Test Accuracy:** **97.65%**
- **Overall Test Loss:** **0.0801**
- **Macro Precision:** **0.9763**
- **Macro Recall:** **0.9764**
- **Macro F1-Score:** **0.9764**
- **Weighted F1-Score:** **0.9765**
- **CPU Training Time:** **28.88 seconds** (12 epochs)

### Per-Digit Performance

| Digit | Precision | Recall | F1-Score | Support |
| :---: | :---: | :---: | :---: | :---: |
| **0** | 0.9818 | 0.9908 | 0.9863 | 980 |
| **1** | **0.9842** | **0.9903** | **0.9873** | 1,135 |
| **2** | 0.9749 | 0.9797 | 0.9773 | 1,032 |
| **3** | 0.9792 | 0.9782 | 0.9787 | 1,010 |
| **4** | 0.9650 | 0.9827 | 0.9738 | 982 |
| **5** | 0.9723 | 0.9832 | 0.9777 | 892 |
| **6** | 0.9781 | 0.9781 | 0.9781 | 958 |
| **7** | 0.9821 | 0.9630 | 0.9725 | 1,028 |
| **8** | 0.9700 | 0.9620 | 0.9660 | 974 |
| **9** | 0.9757 | 0.9564 | 0.9660 | 1,009 |

### Generated Visualizations (in `outputs/`)
- `outputs/training_history.png`: Side-by-side plots of training vs. validation accuracy and loss over 12 epochs.
- `outputs/confusion_matrix.png`: Heatmap of the 10×10 confusion matrix.
- `outputs/example_predictions.png`: Gallery of test predictions displaying true labels, predicted labels, confidence, and status.

---

## 💻 Interactive Streamlit Application

The application is organized into 5 intuitive pages:
1. **🏠 Home & Overview:** High-level project KPIs, architecture overview, dataset specifications, and execution instructions.
2. **✏️ Live Digit Recognition:** Interactive drawing canvas (`st_canvas`) with stroke thickness control, image file upload fallback, 4-stage preprocessing visualizer, predicted digit badge, confidence score, and Softmax probability distribution bar chart.
3. **📊 Model Performance & Metrics:** Actual test accuracy, loss, precision/recall/F1 metrics, training curves, confusion matrix heatmap, and per-digit table.
4. **🔍 Sample Digit Explorer:** Real MNIST test sample inspector with correct and misclassified case studies.
5. **🎓 Viva & Theory Guide:** Theoretical deep dive with mathematical formulas for student viva defense.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.11 or 3.12 (64-bit)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/drx2k06/miniproject8.git
cd miniproject8
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Train the Model

Train the model from scratch on CPU:
```bash
python train.py --epochs 12 --batch-size 64
```

**CLI Arguments:**
- `--epochs`: Number of training epochs (default: `12`).
- `--batch-size`: Mini-batch size (default: `64`).
- `--lr`: Adam learning rate (default: `0.001`).
- `--val-split`: Validation split fraction (default: `0.166667`, 10,000 samples).
- `--model-path`: Save path for model bundle (default: `models/neuralvision_model.keras`).
- `--output-dir`: Output directory for metrics and plots (default: `outputs`).

---

## 📊 How to Run Evaluations

Evaluate the saved model on the untouched MNIST test set:
```bash
python evaluate.py --model-path models/neuralvision_model.keras --output-dir outputs
```

---

## 🌐 How to Launch Streamlit

Run the interactive web application:
```bash
streamlit run app.py
```
Then open your browser at `http://localhost:8501`.

---

## 🧪 Automated Testing

Execute the automated test suite with Pytest:
```bash
pytest tests/
```

**Test Coverage Summary:**
- `tests/test_data_loader.py`: 5 tests (shape validation, label bounds, dataset summary).
- `tests/test_preprocessing.py`: 8 tests (pixel normalization, dimension check, polarity inversion, canvas centering).
- `tests/test_model.py`: 4 tests (architecture dimensions, forward pass smoke test, model save/reload roundtrip).
- `tests/test_prediction.py`: 3 tests (single prediction format, probability distribution sums, batch inference).
- `tests/test_evaluation.py`: 2 tests (metrics computation, training plot generation).

**Result:** `22 passed in ~10.4s` (100% pass rate).

---

## ⚠️ Limitations & Future Enhancements

### Current Limitations
- **Translational Invariance:** As an MLP, the model treats input pixels as a flat 1D vector; non-centered digits rely heavily on the preprocessing pipeline.
- **Single-Digit Scope:** Does not perform multi-digit segmentation.

### Future Enhancements
- **Convolutional Neural Network (CNN):** Add an optional CNN comparative mode to showcase 2D convolutional filter invariance and boost accuracy to $>99.2\%$.
- **Edge Deployment:** Export the model to TensorFlow Lite (`.tflite`) or ONNX for real-time mobile and web deployment.
- **Multi-Digit OCR:** Implement bounding box contour extraction for reading continuous number sequences.

---

## 📚 Documentation Index

| Document | Path | Description |
| :--- | :--- | :--- |
| **Assignment Report** | [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) | Full 16-section college assignment report |
| **Methodology** | [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | Experimental protocol, data splits, and preprocessing design |
| **Architecture** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Mathematical layer equations, dimensions, and parameter formulas |
| **Evaluation Report** | [`docs/EVALUATION_REPORT.md`](docs/EVALUATION_REPORT.md) | In-depth classification report, confusion matrix, error analysis |
| **Final Summary & Viva** | [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) | Project checklist and viva question-and-answer defense guide |

---

## 👨‍💻 Developer Contributions & License

- **Developer:** Daniel Richard Xavier
- **Role:** Deep Learning Developer & ML Engineer
- **License:** MIT License — free for educational and research usage.
