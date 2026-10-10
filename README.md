# Military vs Civilian Vehicle Intelligence

> **Explainable, robust military-vs-civilian vehicle image classification using transfer learning, uncertainty analysis, robustness testing, and visual explainability.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-Transfer%20Learning-orange.svg)](https://pytorch.org/vision/stable/index.html)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF.svg)](https://www.kaggle.com/datasets/theibrarbashir/military-vs-civilian-vehicle-image-classification)
[![Status](https://img.shields.io/badge/Status-Research%20Complete-success.svg)]()

## Overview

This project investigates whether modern transfer learning can reliably distinguish **six vehicle categories** spanning civilian and military domains.

Rather than stopping at a single accuracy number, the project builds a small end-to-end computer-vision research pipeline covering:

- A custom CNN baseline
- EfficientNet-B0 transfer learning
- Controlled fine-tuning experiments
- Per-class evaluation and confusion analysis
- Derived military-vs-civilian decision layer
- Robustness testing under realistic image corruptions
- Grad-CAM visual explainability
- Top-k predictions and uncertainty analysis
- High-confidence error analysis
- UMAP embedding visualization
- Parameter, memory, latency, and throughput benchmarking
- Dataset/class-balance forensics
- Reproducible training configuration
- Champion-model checkpointing and backup

The goal is to demonstrate **model quality, failure analysis, robustness, interpretability, and engineering discipline**, rather than treating image classification as a single-metric exercise.

## 🚀 Try the Model

**Champion model:** EfficientNet-B0

Download the trained checkpoint from the
[GitHub v1.0.0 Release](https://github.com/Devang-555/military-vs-civilian-vehicle-intelligence/releases/tag/v1.0.0).

- **Test Accuracy:** 94.11%
- **Test Macro-F1:** 93.76%
- **Classes:** 6
- **Input Size:** 224 × 224

See the notebook for the evaluation methodology, robustness tests,
explainability analysis, and limitations.

---

## Key Results

The final EfficientNet-B0 model substantially outperformed the custom CNN baseline.

| Model | Parameters | Test Accuracy | Test Macro-F1 | Best Val Macro-F1 |
|---|---:|---:|---:|---:|
| Custom CNN | 389,958 | **64.22%** | **63.43%** | ~64% |
| EfficientNet-B0 | 4,015,234 | **94.11%** | **93.76%** | **95.49%** |

### Transfer-learning improvement

- **+29.89 percentage points** test accuracy
- **+30.33 percentage points** test Macro-F1
- EfficientNet-B0 uses **10.30×** more parameters than the baseline
- On the measured T4 benchmark, EfficientNet was approximately **1.85× slower per image**
- Throughput was approximately **0.54×** the baseline

This makes the project a useful example of the trade-off between **model capacity, accuracy, interpretability, robustness, and inference cost**.

---

## Problem Definition

The primary task is six-class image classification:

1. Civilian aircraft
2. Civilian car
3. Military aircraft
4. Military helicopter
5. Military tank
6. Military truck

A secondary decision layer aggregates the six-class probabilities into:

- **Civilian**
- **Military**

The binary layer is **derived from the six-class model's softmax probabilities**. It is not a separately trained binary classifier.

---

## Dataset

Dataset source:

**Military vs Civilian Vehicle Image Classification — Kaggle**

Dataset identifier:

`theibrarbashir/military-vs-civilian-vehicle-image-classification`

The dataset contains **8,213 images** divided into training, validation, and testing sets.

| Split | Images |
|---|---:|
| Training | 5,696 |
| Validation | 1,005 |
| Testing | 1,512 |
| **Total** | **8,213** |

### Class distribution

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Civilian aircraft | 812 | 143 | 220 | 1,175 |
| Civilian car | 871 | 153 | 229 | 1,253 |
| Military aircraft | 829 | 147 | 228 | 1,204 |
| Military helicopter | 993 | 175 | 264 | 1,432 |
| Military tank | 1,373 | 242 | 358 | 1,973 |
| Military truck | 818 | 145 | 213 | 1,176 |

### Dataset forensics

- Largest class: **military tank — 1,973 images**
- Smallest class: **civilian aircraft — 1,175 images**
- Overall largest-to-smallest class ratio: **1.68×**
- Split composition is relatively consistent, with observed class-share variation across train/validation/test remaining below approximately **0.53 percentage points**

The project does **not** claim that split consistency proves the absence of data leakage. It is treated as a useful dataset-quality diagnostic rather than a leakage guarantee.

---

## Project Pipeline

```text
Kaggle Dataset
      │
      ▼
Dataset Inspection & Forensics
      │
      ▼
Image Preprocessing + Augmentation
      │
      ├──────────────► Custom CNN Baseline
      │
      ▼
EfficientNet-B0 Transfer Learning
      │
      ├──────────────► Six-Class Prediction
      │                       │
      │                       ├── Confusion Matrix
      │                       ├── Per-Class Metrics
      │                       ├── Top-k Predictions
      │                       └── Uncertainty
      │
      ├──────────────► Military/Civilian Decision Layer
      │
      ├──────────────► Robustness Lab
      │
      ├──────────────► Grad-CAM Explainability
      │
      ├──────────────► High-Confidence Error Analysis
      │
      ├──────────────► UMAP Embedding Explorer
      │
      └──────────────► Efficiency Benchmark
                              │
                              ▼
                   Accuracy / F1 / Cost Trade-off
```

---

## Preprocessing

All images are resized to **224 × 224**.

### Training augmentation

- Random horizontal flip
- Random rotation up to 10°
- Color jitter:
  - Brightness ±20%
  - Contrast ±20%
  - Saturation ±20%

Images are normalized using ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

Evaluation uses deterministic resizing and ImageNet normalization without training-time augmentation.

Batch size:

```text
32
```

---

# Model 1 — Custom CNN Baseline

A lightweight CNN was implemented from scratch to establish a controlled baseline.

### Architecture

```text
Input
  ↓
Conv 3 → 32 + ReLU + MaxPool
  ↓
Conv 32 → 64 + ReLU + MaxPool
  ↓
Conv 64 → 128 + ReLU + MaxPool
  ↓
Conv 128 → 256 + ReLU
  ↓
Adaptive Average Pooling
  ↓
Dropout(0.3)
  ↓
Linear(256 → 6)
```

### Training

- Optimizer: AdamW
- Initial learning rate: `1e-3`
- Weight decay: `1e-4`
- Loss: Cross-Entropy
- Epochs: 15
- LR scheduler: ReduceLROnPlateau
- Model selection: best validation Macro-F1
- Random seed: 42

### Test performance

**Accuracy: 64.22%**

**Macro-F1: 63.43%**

The baseline provides an important reference point for measuring the actual benefit of transfer learning.

---

# Model 2 — EfficientNet-B0 Transfer Learning

The main model uses **EfficientNet-B0 pretrained on ImageNet** with a new six-class classification head.

```text
ImageNet-pretrained EfficientNet-B0
              ↓
       Replace classifier
              ↓
         6 output classes
```

### Training configuration

- Pretrained ImageNet weights
- End-to-end transfer learning
- Epochs: 10
- Learning rate: `3e-4`
- Optimizer: AdamW
- Weight decay: `1e-4`
- Loss: Cross-Entropy
- Scheduler: ReduceLROnPlateau
- Best checkpoint selected by validation Macro-F1
- Random seed: 42

Best checkpoint:

- Best validation Macro-F1: **95.49%**
- Best checkpoint: **Epoch 9**

Training time on the measured T4 runtime was approximately **10.8 minutes**.

---

## EfficientNet-B0 Test Results

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Civilian aircraft | 90.78% | 89.55% | **90.16%** | 220 |
| Civilian car | 95.67% | 96.51% | **96.09%** | 229 |
| Military aircraft | 89.33% | 88.16% | **88.74%** | 228 |
| Military helicopter | 96.27% | 97.73% | **96.99%** | 264 |
| Military tank | 96.91% | 96.37% | **96.64%** | 358 |
| Military truck | 93.49% | 94.37% | **93.93%** | 213 |

Overall:

- **Accuracy: 94.11%**
- **Macro-F1: 93.76%**
- Weighted F1: **94.10%**

The weakest class by F1 is **military aircraft (88.74%)**, indicating that aircraft appearance remains a harder visual distinction than several other categories.

---

## Confusion Analysis

EfficientNet-B0 confusion matrix:

```text
[[197   0  20   1   2   0]
 [  1 221   0   1   0   6]
 [ 16   2 201   6   3   0]
 [  1   1   3 258   1   0]
 [  1   2   1   1 345   8]
 [  1   5   0   1   5 201]]
```

Important remaining error patterns include:

- Civilian aircraft ↔ military aircraft confusion
- Some military aircraft ↔ helicopter confusion
- Military tank ↔ military truck confusion
- A small number of civilian car / military truck confusions

The baseline model exhibited much larger confusion, especially for military aircraft and the tank/truck pair.

---

# Controlled Fine-Tuning Experiment

A second EfficientNet experiment was performed to test a common transfer-learning strategy:

### Stage 1
- Backbone frozen
- Classifier trained
- 3 epochs
- Learning rate: `1e-3`
- Best validation Macro-F1: **84.41%**

### Stage 2
- Full backbone unfrozen
- 7 additional epochs
- Learning rate: `1e-5`
- Best validation Macro-F1: **90.37%**

This experiment **underperformed the direct end-to-end transfer-learning configuration**, whose best validation Macro-F1 reached **95.49%**.

This negative result is intentionally retained as part of the research record: a more elaborate two-stage fine-tuning recipe was not automatically better for this dataset and training setup.

---

# Robustness Lab

The final EfficientNet model was evaluated under controlled image corruptions.

| Condition | Accuracy | Macro-F1 | Accuracy Drop |
|---|---:|---:|---:|
| **Clean** | **94.11%** | **93.76%** | — |
| Brightness | 94.18% | 93.83% | -0.07 pp |
| Contrast | 93.92% | 93.55% | -0.19 pp |
| Blur | 87.24% | 86.27% | -6.87 pp |
| Low resolution | 71.36% | 68.23% | -22.75 pp |
| Gaussian noise | 55.62% | 54.03% | -38.49 pp |

### Main finding

The model is highly tolerant of moderate brightness and contrast changes, but substantially less robust to **noise and severe resolution degradation**.

This is an important result because a model with 94% clean accuracy can still fail dramatically under distribution shifts and image-quality degradation.

### Per-class robustness highlights

The most pronounced observed drops included:

- **Military truck:** clean F1 93.93% → Gaussian-noise F1 **37.72%**
- **Civilian aircraft:** clean F1 90.16% → low-resolution F1 **50.49%**

---

# Explainability — Grad-CAM

Grad-CAM was used to visualize the spatial regions contributing to EfficientNet-B0 predictions.

Target layer:

```python
model_tl.features[-1]
```

The explainability analysis includes:

- Correct predictions
- Six-class visual comparison
- High-confidence errors
- High-confidence errors with Grad-CAM overlays

This provides a qualitative inspection mechanism for asking whether the model is focusing on meaningful vehicle/image regions rather than relying on obviously irrelevant visual cues.

Grad-CAM is treated as an **interpretability aid**, not as proof of causal reasoning.

---

# Confidence, Top-k & Uncertainty

The project also examines model confidence instead of reporting only the top-1 label.

On the full 1,512-image test set:

- Mean top-1 confidence: **96.38%**
- Median top-1 confidence: **99.86%**
- Mean normalized entropy-based uncertainty: **0.0632**

### High-confidence errors

Using:

```text
confidence ≥ 90%
AND
prediction ≠ ground truth
```

the model produced:

- **43 high-confidence errors**
- **2.84% of the test set**

Examples include extremely confident mistakes such as:

```text
Military truck → Civilian car       99.9999%
Military aircraft → Civilian aircraft 99.9990%
Military helicopter → Civilian car   99.9965%
Military tank → Civilian car         99.9954%
Military truck → Military tank       99.9948%
Civilian aircraft → Military aircraft 99.9854%
```

These examples demonstrate an important limitation of softmax confidence: **high confidence does not guarantee correctness**.

---

# Derived Military vs Civilian Decision Layer

The six-class probability distribution is aggregated into two higher-level groups.

### Civilian

- Civilian aircraft
- Civilian car

### Military

- Military aircraft
- Military helicopter
- Military tank
- Military truck

The resulting derived binary classification achieved:

| Metric | Result |
|---|---:|
| Accuracy | **96.03%** |
| Precision | **97.00%** |
| Recall | **97.37%** |
| Macro-F1 | **95.24%** |

Confusion matrix:

```text
              Predicted
              Civilian  Military

Civilian          417       32
Military           28     1035
```

Again, this is a **derived decision layer**, not an independently trained binary model.

---

# Embedding Explorer

Intermediate EfficientNet representations were extracted from the validation set.

- Samples: **1,005**
- Embedding dimension: **1,280**
- Feature normalization: L2 normalization
- Dimensionality reduction: UMAP
- Output dimension: 2
- `n_neighbors = 15`
- `min_dist = 0.1`
- Metric: cosine
- Random state: 42

The resulting two-dimensional embedding visualization provides a way to inspect how the learned representation organizes the six vehicle categories.

UMAP is used for **exploration and visualization**, not as a quantitative replacement for classification metrics.

---

# Model Efficiency

The project benchmarks both models on the measured NVIDIA T4 runtime.

| Metric | Baseline CNN | EfficientNet-B0 |
|---|---:|---:|
| Parameters | 389,958 | 4,015,234 |
| Parameter memory (FP32) | 1.49 MB | 15.32 MB |
| Latency | 0.783 ms/image | 1.45 ms/image |
| Throughput | ~1,277 img/s | ~669 img/s |

Measured comparison:

- Parameter multiplier: **10.30×**
- Latency multiplier: **1.85×**
- Throughput ratio: **0.54×**
- Accuracy gain: **+29.89 pp**
- Macro-F1 gain: **+30.33 pp**

Timing is hardware/runtime dependent and should be treated as a benchmark for the recorded environment, not a universal deployment specification.

---

# Training & Reproducibility

Random seed:

```python
42
```

Main training settings are documented in the notebook.

The notebook includes the full experimental workflow from:

```text
Dataset download
→ dataset inspection
→ preprocessing
→ model training
→ validation
→ testing
→ robustness
→ explainability
→ uncertainty
→ embeddings
→ benchmarking
→ checkpointing
```

The dataset is downloaded programmatically with `kagglehub`.

---

# Champion Checkpoint

The final EfficientNet-B0 state was saved as:

```text
efficientnet_b0_vehicle_champion.pth
```

Checkpoint metadata includes:

- Model state dictionary
- Class names
- Image size
- Number of classes
- Architecture name
- Test accuracy
- Test Macro-F1
- Validation Macro-F1
- Random seed

The checkpoint was also backed up to Google Drive.

The model checkpoint is intentionally kept separate from the main source repository until a suitable release/storage strategy is finalized.

---

# Repository Structure

The repository is being organized toward:

```text
military-vs-civilian-vehicle-intelligence/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── Military_Vehicle_Intelligence.ipynb
│
├── src/
│   ├── data.py
│   ├── model.py
│   ├── inference.py
│   ├── robustness.py
│   └── explainability.py
│
├── results/
│   ├── model_comparison/
│   ├── robustness/
│   ├── explainability/
│   └── embeddings/
│
├── models/
│   └── README.md
│
└── docs/
    └── methodology.md
```

The current repository is being built incrementally from the completed Colab research notebook.

---

# Limitations

Despite the strong clean-test performance, this project has important limitations:

1. **Dataset scope** — performance is specific to the source dataset and its visual distribution.
2. **Domain shift** — real-world imagery can differ substantially from curated benchmark images.
3. **Image quality sensitivity** — Gaussian noise and low resolution caused major performance degradation.
4. **High-confidence mistakes** — softmax confidence can remain extremely high on incorrect predictions.
5. **Military aircraft difficulty** — this class has the lowest final per-class F1.
6. **Dataset imbalance** — military tank is substantially more represented than civilian aircraft.
7. **No claim of leakage-free data** — split consistency is a diagnostic, not proof that duplicate/near-duplicate images are absent.
8. **No operational validation** — the project is an academic/portfolio computer-vision study, not a validated real-world intelligence system.
9. **Benchmark dependence** — latency and throughput depend on hardware, runtime, batch configuration, and software versions.
10. **Explainability limitations** — Grad-CAM highlights influential regions but does not establish causal reasoning.

---

# Future Work

Potential extensions include:

- Confidence calibration and Expected Calibration Error (ECE)
- Reliability diagrams
- Temperature scaling
- More systematic corruption benchmarks
- Test-time augmentation
- Stronger data augmentation
- Class-balanced training experiments
- ONNX export and deployment benchmarking
- Lightweight inference optimization
- Interactive image-upload demo
- AI vehicle classification dashboard
- Embedding explorer UI
- Automated model evaluation report
- Model versioning and release management
- More diverse external evaluation data

---

# Responsible Use

This project is intended for **education, computer-vision research, model evaluation, and portfolio demonstration**.

It should not be interpreted as a validated operational military-intelligence, surveillance, targeting, or identification system.

Predictions can be wrong, confidence can be misleading, and robustness can degrade substantially outside the training distribution.

---

# Technology Stack

Core:

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

Additional project tooling:

- KaggleHub
- UMAP
- pytorch-grad-cam
- Google Colab
- Google Drive
- GitHub

Each library is used for a specific part of the experimental pipeline rather than being included only for technology-count purposes.

---

# How to Reproduce

The primary experiment can be reproduced from the notebook:

```text
Military-Civilian-Vehicle-Classification.ipynb
```

### Basic workflow

```bash
pip install -r requirements.txt
```

Then open the notebook in Google Colab or another compatible Jupyter environment.

The notebook downloads the dataset through KaggleHub, constructs the data loaders, trains the models, evaluates the test set, runs robustness and explainability experiments, and generates the analysis artifacts.

For the pretrained EfficientNet-B0 weights, Torchvision downloads the required ImageNet weights automatically.

---

# Project Outcome

This project goes beyond a basic image classifier.

The final system demonstrates:

**Classification**
→ six-class vehicle recognition

**Transfer Learning**
→ EfficientNet-B0 vs custom CNN

**Evaluation**
→ accuracy, precision, recall, Macro-F1, confusion matrices

**Robustness**
→ noise, blur, brightness, contrast, low-resolution testing

**Explainability**
→ Grad-CAM

**Uncertainty**
→ confidence, entropy-style uncertainty, high-confidence errors

**Hierarchical Decision**
→ derived military vs civilian classification

**Representation Analysis**
→ 1,280-D embeddings + UMAP

**Efficiency**
→ parameters, memory, latency, throughput

**Research Discipline**
→ controlled negative experiment, reproducibility seed, checkpointing, limitations, and failure analysis

The result is therefore not just a classifier, but a compact **computer-vision model evaluation and intelligence-style analysis pipeline** built around a reproducible six-class image-classification problem.

---

## Author

**Devang-555**

Computer Science & Engineering (AI/ML)

GitHub:
`https://github.com/Devang-555`

---

## Dataset Attribution

Dataset:

**Military vs Civilian Vehicle Image Classification**

Kaggle dataset by **Ibrar Bashir**:

`https://www.kaggle.com/datasets/theibrarbashir/military-vs-civilian-vehicle-image-classification`

Please consult the original dataset page for its current licensing and usage terms. The dataset is **not redistributed in this repository**.

---

## License

The source code in this repository is intended to be released under the **MIT License**.

The dataset remains subject to the terms and license of its original source.

See `LICENSE` for the repository's code license.
