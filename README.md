# EEG-Based Alzheimer's Disease Detection

**Robust classification of Alzheimer's disease from resting-state EEG, evaluated under strict subject-level protocols.**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Research conducted under the supervision of **Prof. Kang-Ming Chang**, Department of Computer and Communication Engineering, National Kaohsiung University of Science and Technology (NKUST), Taiwan.

---

## Overview

This repository provides a complete pipeline for classifying Alzheimer's disease (AD) versus healthy controls (HC) using 19-channel resting-state EEG.

Most published EEG-AD results are evaluated at the **epoch level** — individual short EEG segments are split across training and test sets without regard to which subject they came from. Because segments from the same subject share subject-specific noise, this lets a model partly "recognize" a subject rather than learn disease-relevant patterns, inflating reported accuracy. We measure this effect directly and find it inflates accuracy by roughly **9.4 percentage points** on our data.

To avoid this, every model here is evaluated with **subject-level Leave-One-Subject-Out (LOSO) cross-validation**: no subject's data ever appears in both the training and test sets for the same fold. This is a stricter, more clinically realistic standard than most prior work in this area.

![Model Comparison](docs/images/model_comparison.png)
*Figure 1 — Baseline models compared against the proposed Hybrid SIR-EEGNet, all under subject-level LOSO.*

---

## Model: Hybrid SIR-EEGNet

The Hybrid SIR-EEGNet combines three components:

| Component | Role |
|---|---|
| **EEGNet backbone** | Learns spatial-temporal features directly from raw EEG signals. |
| **Relative Band Power (RBP) branch** | Explicitly encodes known clinical EEG biomarkers for AD (band-power ratios), rather than relying on the network to rediscover them from raw signal alone. |
| **Gradient Reversal Layer (GRL)** | Adds a subject-adversarial training objective, following Ganin et al. (2016), so the shared feature extractor is pushed toward disease-relevant patterns and away from subject-specific artifacts. |

![SIR-EEGNet Architecture](docs/images/sir_eegnet_architecture.png)
*Figure 2 — The EEGNet stream and RBP branch are concatenated into a shared representation, which feeds two heads: disease classification, and a subject-adversarial GRL head.*

![Confusion Matrices](docs/images/confusion_matrices.png)
*Figure 3 — Confusion matrices showing prediction performance on AD subjects.*

---

## Results

Binary classification on the OpenNeuro **ds004504** dataset (N = 65 subjects), all under subject-level LOSO:

| Model | Accuracy | F1 (weighted) | AD Recall |
|---|---:|---:|---:|
| SVM + RBP | 83.1% | 83.0% | 88.9% |
| EEGNet | 78.5% | — | — |
| **Hybrid SIR-EEGNet** | 81.5% | 81.3% | 88.9% |

![Leakage Demonstration](docs/images/leakage_demonstration.png)
*Figure 4 — Accuracy inflation under epoch-level cross-validation compared to subject-level LOSO, on the same data and models.*

The SVM + RBP baseline currently edges out the Hybrid SIR-EEGNet on raw accuracy, while both share the same AD recall. The value of the hybrid model is explored further — including cross-dataset generalization, where subject-invariant features matter more — in the [Results documentation](docs/results.md).

For full metrics, statistical significance tests, and cross-dataset results (FSU, ADSZ), see the [**Results**](docs/results.md) page.

---

## Documentation

| Page | Contents |
|---|---|
| [Methodology](docs/methodology.md) | LOSO evaluation protocol, EEG preprocessing pipeline, and feature extraction details. |
| [Results](docs/results.md) | Full metrics, statistical tests, and cross-dataset evaluation (FSU, ADSZ). |
| [Datasets](docs/datasets.md) | Description of the primary dataset (ds004504) and external datasets used for cross-dataset testing. |

---

## Notebooks

The pipeline runs as a sequence of notebooks, each building on the previous:

| # | Notebook | Description |
|---|---|---|
| 1 | `01_data_exploration` | Dataset loading, exploratory data analysis, and DTABR biomarker analysis. |
| 2 | `02_svm_baseline_loso` | SVM + RBP feature baseline, evaluated under strict LOSO. |
| 3 | `03_eegnet_loso` | EEGNet baseline, evaluated under LOSO. |
| 4 | `04_sir_eegnet_loso` | Hybrid SIR-EEGNet training and evaluation on ds004504. |
| 5 | `05_SIREEGNet_CrossDataset_ADSZ` | Hybrid SIR-EEGNet cross-dataset testing on the ADSZ dataset. |
| 6 | `06_SIREEGNet_CrossDataset_FSU` | Hybrid SIR-EEGNet cross-dataset testing on the FSU dataset. |

---

## Requirements

- Python 3.9+
- MNE-Python, PyTorch, scikit-learn, SciPy, NumPy, pandas, matplotlib, seaborn

```bash
pip install -r requirements.txt
```

---

## Citation

If you use this code, please cite the primary dataset:

> Miltiadous, A., et al. (2023). *A Dataset of Scalp EEG Recordings of Alzheimer's Disease, Frontotemporal Dementia and Healthy Subjects.* Data, 8(6), 95.

---

## Acknowledgements

This research was conducted at the Department of Computer and Communication Engineering, National Kaohsiung University of Science and Technology (NKUST), Taiwan.

## License

MIT License


