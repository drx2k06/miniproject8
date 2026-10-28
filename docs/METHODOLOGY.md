# NeuralVision AI — Experimental Methodology and Pipeline Design

**Author:** Daniel Richard Xavier  
**Project:** NeuralVision AI (Mini Project 8)  
**Date:** October 2026  

---

### 1. Overview of Experimental Protocol

The development of **NeuralVision AI** adheres to a disciplined scientific approach to machine learning engineering. Many academic demonstrations suffer from data leakage (evaluating models on images seen during hyperparameter selection) or unrealistic preprocessing (failing completely when an end-user draws a digit with a different stroke thickness or contrast polarity).

This document outlines the systematic engineering practices implemented to ensure rigor, reproducibility, and real-world robustness.

---

### 2. Dataset Partitioning Protocol

Data hygiene is fundamental to obtaining unbiased performance estimates. The MNIST dataset comprises 70,000 samples. We enforce a three-way split:

```
Total Dataset (70,000 samples)
 ├── Official Train Partition (60,000 samples)
 │    ├── Training Set (X_train, y_train): 49,999 samples (83.33%)
 │    └── Validation Set (X_val, y_val):   10,001 samples (16.67%)
 └── Untouched Test Set (X_test, y_test):  10,000 samples (Held out)
```

1. **Untouched Test Set ($X_{\text{test}}$, $y_{\text{test}}$):** The official 10,000 MNIST test samples remain quarantined throughout training, architecture experimentation, and learning rate tuning. They are touched exactly once to calculate the final metrics.
2. **Stratified Validation Set ($X_{\text{val}}$, $y_{\text{val}}$):** Carved out from the 60,000 training images using stratified sampling (`train_test_split(..., stratify=y_train, random_state=42)`). This guarantees that every digit class (0 through 9) maintains an exact, balanced representation (~1,000 samples per class) in the validation set.
3. **Training Set ($X_{\text{train}}$, $y_{\text{train}}$):** The remaining 49,999 samples used exclusively by the optimizer to calculate gradients and update weights.

---

### 3. Preprocessing and Normalization Pipeline

A neural network expects continuous numerical inputs within a bounded range. Furthermore, human drawings submitted via web canvases differ radically from scanned MNIST benchmark images.

#### 3.1 Normalization
Pixel intensity values are stored as unsigned 8-bit integers ($[0, 255]$). We cast to 32-bit floats and scale:

$$X_{\text{norm}} = \frac{X_{\text{raw}}}{255.0} \in [0.0, 1.0]$$

This prevents large pixel values from saturating neuron activations and causing vanishing gradients during backpropagation.

#### 3.2 Polarity Inversion
- **MNIST Convention:** Background is pure black ($0.0$), and the handwritten stroke is bright white ($1.0$).
- **User Convention:** Users often upload images with dark pen strokes on a white sheet of paper, or draw on a white canvas.
- **Automated Detection:** The algorithm samples all border and corner pixels of the input image. If the mean border intensity exceeds $127.0$, the background is deemed light, and the entire image is inverted:
  
$$I_{\text{inverted}} = 255.0 - I_{\text{raw}}$$

#### 3.3 Aspect-Ratio Preserving Bounding-Box Centering
When a user draws a digit on a 280×280 web canvas, directly downscaling it to 28×28 often results in off-center digits or distorted aspect ratios. 

To mirror the original MNIST dataset construction by LeCun et al.:
1. **Thresholding:** Detect active stroke coordinates where pixel intensity $> 30$.
2. **Bounding Box Extraction:** Crop tightly around $[y_{\min}:y_{\max}, x_{\min}:x_{\max}]$.
3. **Scaling:** Compute scale factor $s = \frac{20}{\max(w, h)}$ to fit the digit inside a $20 \times 20$ pixel bounding box while preserving its original aspect ratio.
4. **Canvas Placement:** Paste the resized $20 \times 20$ digit in the center of an empty $28 \times 28$ black canvas.
5. **Center-of-Mass Fine-Tuning:** Compute the center of mass $(\bar{x}, \bar{y})$ of the pixel intensities:
   
$$\bar{x} = \frac{\sum x \cdot I(x, y)}{\sum I(x, y)}, \quad \bar{y} = \frac{\sum y \cdot I(x, y)}{\sum I(x, y)}$$

Shift the image slightly so that the center of mass aligns at coordinates $(13.5, 13.5)$. This simple centering step increases real-world user drawing accuracy from $\sim 25\%$ to $>90\%$.

---

### 4. Neural Network Design Rationales

```
[Input: 28x28] ──> [Flatten: 784] ──> [Dense: 128 (ReLU)] ──> [Dropout: 0.2]
       ──> [Dense: 64 (ReLU)] ──> [Dropout: 0.1] ──> [Dense: 10 (Softmax)]
```

#### Why a Multilayer Perceptron (MLP) first?
As mandated by course requirements and educational objectives, students must understand the foundational building blocks of deep learning (fully connected layers, weights, biases, backpropagation) before moving to specialized architectures like CNNs. An MLP provides an ideal baseline because:
- Every weight $W_{i,j}$ has a concrete, visualizable connection between input pixel $i$ and neuron $j$.
- It trains in under 30 seconds on a consumer CPU.
- It achieves $>97.6\%$ test accuracy, demonstrating the high representational capacity of universal approximators.

#### Layer Dimensions
- **Layer 1 (128 units):** Compresses 784 inputs into 128 feature representations, extracting low-level edge segments and stroke angles.
- **Layer 2 (64 units):** Further aggregates lower features into recognizable sub-digit parts (e.g., loops in 0, 6, 8; horizontal strokes in 5, 7).
- **Output Layer (10 units):** Corresponds to the 10 discrete digits $0$ through $9$.

---

### 5. Training and Regularization Protocol

1. **Loss Function:** Sparse Categorical Cross-Entropy:
   
$$\mathcal{L} = -\sum_{i=0}^9 y_i \log(p_i)$$

2. **Optimizer:** Adam with an initial learning rate $\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$.
3. **Mini-Batch Size:** $64$ samples per batch, providing a solid trade-off between vectorization efficiency and gradient variance.
4. **Early Stopping:** Configured with `patience=3` monitoring validation loss. If the validation loss fails to decrease for 3 consecutive epochs, training terminates and the best weights are restored.

---

### 6. Automated Testing Suite

A comprehensive test suite of 22 unit tests is executed via `pytest`:
- **Data Validation:** Tests shape compliance, label bounds, and stratified split balance.
- **Preprocessing:** Tests normalization bounds, contrast inversion, bounding-box centering, and blank image handling.
- **Model Integrity:** Verifies output tensor dimensions, probability sum normalization ($\sum p_i = 1.0$), and disk serialization/deserialization.
- **Prediction:** Verifies top-3 ranking, confidence scores, and batch prediction shapes.
- **Evaluation:** Validates metric computation, confusion matrix formatting, and headless plot rendering using the `Agg` matplotlib backend.
