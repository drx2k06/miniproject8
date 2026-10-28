# NeuralVision AI — Comprehensive Evaluation and Performance Report

**Author:** Daniel Richard Xavier  
**Dataset:** MNIST Handwritten Digits (Untouched Test Set)  
**Evaluation Date:** October 2026  
**Artifact File:** `outputs/metrics.json`  

---

### 1. Executive Evaluation Summary

The finalized **NeuralVision AI** model was evaluated on the complete, untouched 10,000-sample MNIST test set.

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Test Accuracy** | **97.65%** | 9,765 correct classifications out of 10,000 unseen samples |
| **Test Loss** | **0.0801** | Sparse Categorical Cross-Entropy |
| **Macro Precision** | **0.9763** | Unweighted average precision across all 10 classes |
| **Macro Recall** | **0.9764** | Unweighted average recall across all 10 classes |
| **Macro F1-Score** | **0.9764** | Harmonic mean of macro precision and recall |
| **Weighted F1-Score** | **0.9765** | Support-weighted average F1-score |
| **Training Duration** | **28.88 sec** | 12 epochs on consumer CPU (Intel/AMD x86_64) |
| **Inference Latency** | **< 12 ms** | Per-sample CPU forward pass |

---

### 2. Detailed Per-Digit Classification Report

| Digit Class | Precision | Recall | F1-Score | Support | True Positives | Errors |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | 0.9818 | **0.9908** | 0.9863 | 980 | 971 | 9 |
| **1** | **0.9842** | 0.9903 | **0.9873** | 1,135 | 1,124 | 11 |
| **2** | 0.9749 | 0.9797 | 0.9773 | 1,032 | 1,011 | 21 |
| **3** | 0.9792 | 0.9782 | 0.9787 | 1,010 | 988 | 22 |
| **4** | 0.9650 | 0.9827 | 0.9738 | 982 | 965 | 17 |
| **5** | 0.9723 | 0.9832 | 0.9777 | 892 | 877 | 15 |
| **6** | 0.9781 | 0.9781 | 0.9781 | 958 | 937 | 21 |
| **7** | 0.9821 | 0.9630 | 0.9725 | 1,028 | 990 | 38 |
| **8** | 0.9700 | 0.9620 | 0.9660 | 974 | 937 | 37 |
| **9** | 0.9757 | 0.9564 | 0.9660 | 1,009 | 965 | 44 |
| **Total / Macro** | **0.9763** | **0.9764** | **0.9764** | **10,000** | **9,765** | **235** |

#### Key Per-Class Observations
- **Highest Performing Digits:** Digit **1** (F1: 0.9873) and Digit **0** (F1: 0.9863). Their simple geometric structure (a vertical stroke and an isolated closed loop) makes them easy to separate in 784-dimensional pixel space.
- **Most Challenging Digits:** Digit **9** (Recall: 95.64%) and Digit **8** (Recall: 96.20%). The circular upper portion of 9 is frequently confused with an open-top 4, while incomplete loops in 8 resemble 3.

---

### 3. Complete Confusion Matrix (10×10)

The row represents the **True Ground Truth Class**, while the column represents the **Model Predicted Class**:

| True \ Pred | **0** | **1** | **2** | **3** | **4** | **5** | **6** | **7** | **8** | **9** |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **971** | 0 | 0 | 1 | 0 | 1 | 3 | 0 | 3 | 1 |
| **1** | 0 | **1124** | 3 | 0 | 0 | 2 | 2 | 1 | 3 | 0 |
| **2** | 4 | 2 | **1011** | 2 | 1 | 0 | 2 | 4 | 6 | 0 |
| **3** | 1 | 0 | 4 | **988** | 0 | 10 | 0 | 3 | 3 | 1 |
| **4** | 0 | 1 | 1 | 0 | **965** | 0 | 4 | 1 | 1 | 9 |
| **5** | 2 | 0 | 0 | 4 | 1 | **877** | 4 | 0 | 3 | 1 |
| **6** | 4 | 3 | 1 | 0 | 4 | 6 | **937** | 0 | 3 | 0 |
| **7** | 0 | 8 | 12 | 2 | 4 | 1 | 1 | **990** | 2 | 7 |
| **8** | 2 | 0 | 5 | 7 | 7 | 3 | 6 | 2 | **937** | 5 |
| **9** | 5 | 5 | 0 | 5 | **18** | 3 | 0 | 4 | 4 | **965** |

---

### 4. Analysis of Most Confused Digit Pairs

The top misclassification patterns discovered in the test set:

1. **True 9 $\rightarrow$ Predicted 4 (18 occurrences):**  
   *Root Cause:* In rapid handwriting, the curved ascender of 9 often fails to close at the top, leaving an open fork that closely mimics the upright stem and vertical apex of a handwritten 4.
2. **True 7 $\rightarrow$ Predicted 2 (12 occurrences):**  
   *Root Cause:* When writers draw a continental European 7 with a bottom cross-tick, or curve the diagonal downward stroke, the lower curvature resembles the baseline horizontal curl of a digit 2.
3. **True 3 $\rightarrow$ Predicted 5 (10 occurrences):**  
   *Root Cause:* Both digits share identical lower semicircular cups. If the top curve of a 3 is written with an angular horizontal stroke, the network confuses it with the upper horizontal flag of a 5.
4. **True 7 $\rightarrow$ Predicted 1 (8 occurrences):**  
   *Root Cause:* Digits written with very short horizontal top bars resemble a single vertical stroke.
5. **True 8 $\rightarrow$ Predicted 3 (7 occurrences):**  
   *Root Cause:* Incomplete closure on the left arc of an 8 makes it topologically indistinguishable from a 3.

---

### 5. Training History and Convergence Dynamics

Training progressed across 12 epochs with a mini-batch size of 64:

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | 0.3805 | 88.51% | 0.1695 | 94.94% | Initial rapid gradient convergence |
| **2** | 0.1741 | 94.88% | 0.1278 | 96.12% | Feature extraction begins stabilizing |
| **3** | 0.1334 | 96.00% | 0.1102 | 96.72% | Reaches >96% accuracy |
| **6** | 0.0936 | 97.14% | 0.0927 | 97.19% | Training and val loss converge |
| **9** | 0.0684 | 97.89% | 0.0863 | 97.48% | Fine-tuning weights |
| **12** | **0.0463** | **98.52%** | **0.0826** | **97.73%** | Final restored checkpoint |

**Overfitting Diagnostics:**
The validation loss remains closely aligned with the training loss ($0.0826$ vs $0.0463$), with no upward divergence. The test loss ($0.0801$) confirms that the model generalizes robustly without overfitting to the training distribution.
