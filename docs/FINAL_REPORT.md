# NeuralVision AI — Final Project Report and Viva Defense Guide

**Project Title:** NeuralVision AI — Handwritten Digit Recognition Using Neural Networks  
**Developer:** Daniel Richard Xavier  
**Academic Requirement:** Mini Project 8  
**Repository:** [https://github.com/drx2k06/miniproject8.git](https://github.com/drx2k06/miniproject8.git)  
**Submission Date:** October 2026  

---

### 1. Executive Summary

**NeuralVision AI** is a complete, production-grade deep learning solution for handwritten digit recognition trained on the MNIST dataset. The project demonstrates the practical application of artificial neural networks, from raw mathematical foundations (weights, biases, activations, backpropagation) to an interactive, user-facing Streamlit application.

The model was implemented using TensorFlow/Keras with CPU training constraints, completing 12 epochs in **28.88 seconds**. Evaluated on the strictly isolated 10,000-sample test set, it achieved **97.65% test accuracy**, **0.0801 test loss**, and a macro F1-score of **0.9764**.

---

### 2. Verified Deliverables Checklist

- [x] **Data Pipeline:** Strict three-way partition ($X_{\text{train}}$: 49,999, $X_{\text{val}}$: 10,001, $X_{\text{test}}$: 10,000 untouched).
- [x] **Image Preprocessing:** Grayscale conversion, contrast polarity detection, bounding-box cropping, aspect-ratio scaling to 20x20, center-of-mass canvas alignment, and $[0.0, 1.0]$ normalization.
- [x] **Neural Network:** Multilayer Perceptron (784 $\rightarrow$ 128 ReLU $\rightarrow$ Dropout 0.2 $\rightarrow$ 64 ReLU $\rightarrow$ Dropout 0.1 $\rightarrow$ 10 Softmax), 109,386 trainable parameters.
- [x] **Model Persistence:** Compact $\sim 1.3$ MB `.keras` bundle saved at `models/neuralvision_model.keras`.
- [x] **Evaluation Artifacts:** `outputs/metrics.json`, `outputs/training_history.png`, `outputs/confusion_matrix.png`, `outputs/example_predictions.png`.
- [x] **Streamlit Web Application:** Multi-page dashboard with interactive drawing canvas, image upload, live preprocessing visualizer, probability bar charts, and test sample explorer.
- [x] **Automated Testing:** 22 unit tests passing cleanly via `pytest` (0 errors, 0 failures).
- [x] **Complete Academic Documentation:** `README.md`, `PROJECT_REPORT.md`, `METHODOLOGY.md`, `ARCHITECTURE.md`, `EVALUATION_REPORT.md`, `FINAL_REPORT.md`.

---

### 3. Student Viva Examination Preparation Guide

#### Question 1: Why did you choose an MLP instead of a Convolutional Neural Network (CNN)?
> **Answer:** For educational purposes, an MLP directly demonstrates the fundamental mechanics of artificial neural networks—such as weight matrices, bias vectors, dense vector products, activation functions, and calculus-based backpropagation. An MLP achieves $>97.6\%$ test accuracy in under 30 seconds on CPU without needing dedicated GPU hardware. In the future scope, CNNs can be added to demonstrate translation invariance.

#### Question 2: Why do you normalize pixels from [0, 255] to [0.0, 1.0]?
> **Answer:** If raw pixel inputs $[0, 255]$ are fed directly into neurons, the dot product $z = \mathbf{w}^T \mathbf{x} + b$ results in large activation values. This causes vanishing gradients in activation functions, slows down gradient descent convergence, and creates numerical instability. Scaling inputs to $[0.0, 1.0]$ keeps gradients bounded and ensures balanced weight updates.

#### Question 3: Why is ReLU preferred over the Sigmoid activation function?
> **Answer:** Sigmoid saturates at $0$ and $1$, where its derivative $\sigma'(z) \approx 0$. In multi-layer networks, this leads to the severe **vanishing gradient problem**, where gradients shrink exponentially during backpropagation. ReLU ($f(z) = \max(0, z)$) has a constant gradient of $1$ for all positive inputs, enabling fast, stable learning. It also produces sparse representations and is computationally cheap.

#### Question 4: How does Dropout combat overfitting?
> **Answer:** Dropout randomly deactivates a fraction of neurons (e.g., $20\%$ in layer 1, $10\%$ in layer 2) during each training forward-backward pass. This prevents neurons from developing co-dependencies where one neuron compensates for errors made by another. Instead, every neuron learns robust, generalized features. During inference, Dropout is turned off and full weights are used.

#### Question 5: What is the purpose of the Softmax function?
> **Answer:** Softmax converts 10 arbitrary raw logits $[z_0, \dots, z_9]$ into a valid probability distribution:
> $$\sigma(z)_i = \frac{e^{z_i}}{\sum_{j=0}^9 e^{z_j}}$$
> It guarantees that each output $p_i \in [0, 1]$ and the sum of all 10 probabilities equals exactly $1.0$.

#### Question 6: What is Sparse Categorical Cross-Entropy Loss?
> **Answer:** It measures the divergence between the predicted probability distribution and the actual true class label. When true labels are represented as discrete integers ($y \in \{0, \dots, 9\}$) rather than one-hot encoded vectors, sparse categorical cross-entropy computes the negative log-likelihood: $\mathcal{L} = -\ln(p_{\text{true}})$. If the model assigns $100\%$ probability to the true class, the loss is $0$; if it assigns $0\%$, the loss approaches infinity.

#### Question 7: Why is your test set kept separate from the validation set?
> **Answer:** If we tune hyperparameters (such as epochs, learning rates, or dropout rates) using the test set, information from the test set leaks into our modeling choices. The validation set is used for epoch-by-epoch monitoring and early stopping. The test set is evaluated exactly once at the end to provide an unbiased estimate of real-world generalization.

---

### 4. How to Run and Verify the System

```bash
# 1. Clone repository
git clone https://github.com/drx2k06/miniproject8.git
cd miniproject8

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run automated tests
pytest tests/

# 4. Train model (optional, pre-trained model included)
python train.py --epochs 12 --batch-size 64

# 5. Evaluate model on test set
python evaluate.py

# 6. Launch Streamlit web application
streamlit run app.py
```
