# NeuralVision AI — Deep Learning Architecture Specification

**Project:** NeuralVision AI (Mini Project 8)  
**Author:** Daniel Richard Xavier  
**Date:** October 2026  

---

### 1. Mathematical Architecture Overview

NeuralVision AI employs a Feedforward Artificial Neural Network (Multilayer Perceptron) designed for high classification accuracy, numerical stability, and interpretability.

The network operates as a sequential composite function:

$$\mathbf{y} = f(\mathbf{X}; \mathbf{\Theta}) = f^{[3]} \circ f^{[2]} \circ f^{[1]} \circ \text{flatten}(\mathbf{X})$$

Where $\mathbf{\Theta} = \{\mathbf{W}^{[1]}, \mathbf{b}^{[1]}, \mathbf{W}^{[2]}, \mathbf{b}^{[2]}, \mathbf{W}^{[3]}, \mathbf{b}^{[3]}\}$ represents the complete set of learnable parameters ($109,386$ scalars).

---

### 2. Layer-by-Layer Specifications and Parameter Counts

```
Input (28, 28)
  │
  ▼
[ Flatten ] ─────────── Dimension: 784
  │
  ▼
[ Dense 1 + ReLU ] ──── Dimension: 128  | Weights: (784, 128) + Biases: (128) = 100,480 params
  │
  ▼
[ Dropout (0.2) ] ───── Dimension: 128  | Active during training only
  │
  ▼
[ Dense 2 + ReLU ] ──── Dimension: 64   | Weights: (128, 64)  + Biases: (64)  = 8,256 params
  │
  ▼
[ Dropout (0.1) ] ───── Dimension: 64   | Active during training only
  │
  ▼
[ Dense 3 + Softmax ] ─ Dimension: 10   | Weights: (64, 10)   + Biases: (10)  = 650 params
  │
  ▼
Output Probabilities [p_0, p_1, ..., p_9]
```

#### Detailed Layer Computations

#### 1. Input Layer
- **Input Tensor:** $\mathbf{X} \in \mathbb{R}^{B \times 28 \times 28}$, where $B$ is the mini-batch size.
- **Parameters:** 0.

#### 2. Flatten Layer
- **Operation:** Unrolls 2D spatial dimensions into a 1D feature vector:
  
$$\mathbf{a}^{[0]} = \text{vec}(\mathbf{X}) \in \mathbb{R}^{B \times 784}$$

- **Parameters:** 0.

#### 3. Hidden Layer 1 (Dense 128 + ReLU)
- **Linear Combination:**
  
$$\mathbf{z}^{[1]} = \mathbf{a}^{[0]} \mathbf{W}^{[1]} + \mathbf{b}^{[1]} \in \mathbb{R}^{B \times 128}$$

  - Weight Matrix: $\mathbf{W}^{[1]} \in \mathbb{R}^{784 \times 128}$ ($100,352$ parameters).
  - Bias Vector: $\mathbf{b}^{[1]} \in \mathbb{R}^{128}$ ($128$ parameters).
- **Activation Function (ReLU):**
  
$$\mathbf{a}^{[1]} = \max(\mathbf{0}, \mathbf{z}^{[1]})$$

- **Trainable Parameters:** $784 \times 128 + 128 = \mathbf{100,480}$.

#### 4. Dropout Layer 1
- **Operation (Training mode):**
  
$$\tilde{\mathbf{a}}^{[1]} = \mathbf{a}^{[1]} \odot \mathbf{m}^{[1]}, \quad m_j \sim \text{Bernoulli}(1 - 0.2)$$

- **Trainable Parameters:** 0.

#### 5. Hidden Layer 2 (Dense 64 + ReLU)
- **Linear Combination:**
  
$$\mathbf{z}^{[2]} = \tilde{\mathbf{a}}^{[1]} \mathbf{W}^{[2]} + \mathbf{b}^{[2]} \in \mathbb{R}^{B \times 64}$$

  - Weight Matrix: $\mathbf{W}^{[2]} \in \mathbb{R}^{128 \times 64}$ ($8,192$ parameters).
  - Bias Vector: $\mathbf{b}^{[2]} \in \mathbb{R}^{64}$ ($64$ parameters).
- **Activation Function (ReLU):**
  
$$\mathbf{a}^{[2]} = \max(\mathbf{0}, \mathbf{z}^{[2]})$$

- **Trainable Parameters:** $128 \times 64 + 64 = \mathbf{8,256}$.

#### 6. Dropout Layer 2
- **Operation (Training mode):**
  
$$\tilde{\mathbf{a}}^{[2]} = \mathbf{a}^{[2]} \odot \mathbf{m}^{[2]}, \quad m_j \sim \text{Bernoulli}(1 - 0.1)$$

- **Trainable Parameters:** 0.

#### 7. Output Layer (Dense 10 + Softmax)
- **Linear Logits:**
  
$$\mathbf{z}^{[3]} = \tilde{\mathbf{a}}^{[2]} \mathbf{W}^{[3]} + \mathbf{b}^{[3]} \in \mathbb{R}^{B \times 10}$$

  - Weight Matrix: $\mathbf{W}^{[3]} \in \mathbb{R}^{64 \times 10}$ ($640$ parameters).
  - Bias Vector: $\mathbf{b}^{[3]} \in \mathbb{R}^{10}$ ($10$ parameters).
- **Activation Function (Softmax):**
  
$$p_i = \frac{e^{z_i^{[3]}}}{\sum_{k=0}^{9} e^{z_k^{[3]}}} \quad \text{for } i \in \{0, \dots, 9\}$$

- **Trainable Parameters:** $64 \times 10 + 10 = \mathbf{650}$.

---

### 3. Summary of Parameters and Memory Footprint

| Category | Count | Storage (float32) |
| :--- | :--- | :--- |
| **Layer 1 Weights + Biases** | 100,480 | 401.92 KB |
| **Layer 2 Weights + Biases** | 8,256 | 33.02 KB |
| **Layer 3 Weights + Biases** | 650 | 2.60 KB |
| **Total Trainable Parameters** | **109,386** | **437.54 KB** |
| **Adam Optimizer States ($m_t, v_t$)** | 218,772 | 875.09 KB |
| **Saved Keras Bundle on Disk** | — | **~1.34 MB** |

The ultra-compact memory footprint ($<1.5$ MB) guarantees that the entire model can be loaded into RAM instantly and evaluated in single-digit milliseconds on any standard computer.
