<p align="center">
  <img src="figures/banner.png" alt="Federated Learning for Heart Disease Diagnosis" width="700"/>
</p>

<h1 align="center">🏥 Federated Learning for Heart Disease Diagnosis</h1>

<p align="center">
  <em>Comparing Centralized MLP, FedAvg (IID & non-IID), FedProx, Privacy, and Fairness on UCI Heart Disease</em>
</p>

<p align="center">
  <strong>Author:</strong> KHOUAJA Ahmed
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=flat&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-MLP-EE4C2C?style=flat&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/FL-FedAvg%20%7C%20FedProx-00897B?style=flat" alt="FL"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License"/>
</p>

---

## 📋 Overview

This project implements a complete **Federated Learning pipeline for medical diagnosis** (heart disease classification) and compares:

1. **Centralized MLP baseline** — Standard training on all data
2. **FedAvg (IID)** — Federated averaging with uniform data distribution
3. **FedAvg (non-IID)** — Federated averaging with Dirichlet-skewed distributions
4. **FedProx** — Proximal regularization for handling data heterogeneity
5. **Privacy analysis** — Empirical noise-utility trade-off study
6. **Fairness audit** — SPD, DI, EOD, FPR gap with reweighting mitigation

All experiments use **5 seeds** (mean ± std) for stability, **validation-based model selection**, and **medical diagnostic metrics** (ROC-AUC, PR-AUC, Sensitivity, Specificity, Brier score).

### Key Findings

| Method | Accuracy | ROC-AUC |
|:---|:---:|:---:|
| **Baseline (centralized)** | **0.783 ± 0.056** | **0.890 ± 0.034** |
| **FedAvg (IID)** | **0.793 ± 0.042** | ~0.896 |
| **FedAvg (non-IID)** | **0.757 ± 0.077** | ~0.884 |
| FedProx (μ=0.01) | ≈ FedAvg | — |
| FedProx (μ=0.1) | ≈ FedAvg | — |

### Key Takeaways

- 📊 **non-IID hurts performance** — Lower accuracy and higher variability under client heterogeneity (Dirichlet β=0.5)
- 🔬 **FedProx shows no gain** — Proximal term does not improve on this small dataset (honest negative result)
- 🔒 **Noise-utility stable** — Gradient noise (σ ≤ 0.1) does not significantly degrade accuracy
- ⚖️ **Fairness reweighting helps** — SPD drops from +0.34 to +0.21, DI improves toward 1.0
- ⚠️ **Small test set caveat** — Only 60 test samples; accuracy changes by steps of 1/60 ≈ 0.017

---

## 📊 Results

### Baseline Centralized (5 seeds)

| Seed | Accuracy | ROC-AUC | Sensitivity | Specificity |
|:---:|:---:|:---:|:---:|:---:|
| 42 | 0.850 | 0.951 | 0.821 | 0.875 |
| 123 | 0.850 | 0.892 | 0.893 | 0.813 |
| 456 | 0.717 | 0.861 | 0.643 | 0.781 |
| 789 | 0.750 | 0.854 | 0.750 | 0.750 |
| 2024 | 0.750 | 0.892 | 0.857 | 0.656 |

### FedAvg Multi-Seeds

| Seed | IID Accuracy | non-IID Accuracy |
|:---:|:---:|:---:|
| 42 | 0.850 | 0.833 |
| 123 | 0.833 | 0.817 |
| 456 | 0.750 | 0.700 |
| 789 | 0.783 | 0.800 |
| 2024 | 0.750 | 0.633 |

### Privacy Analysis (σ ∈ {0.0, 0.01, 0.05, 0.1})

| Noise Level | Accuracy |
|:---:|:---:|
| σ=0.0 | 0.793 ± 0.042 |
| σ=0.01 | 0.780 ± 0.053 |
| σ=0.05 | 0.793 ± 0.034 |
| σ=0.1 | 0.800 ± 0.049 |

> **Note**: These noise levels are NOT formal (ε,δ)-DP guarantees. This is an empirical robustness study.

### Fairness Metrics (mean over 3 seeds)

| Metric | Baseline | Reweighting |
|:---:|:---:|:---:|
| SPD | +0.345 | +0.210 |
| DI | 0.417 | 0.632 |
| EOD | +0.127 | −0.057 |
| FPR Gap | +0.161 | +0.021 |

---

## 🗂️ Repository Structure

```
FL-Heart-Disease/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── .gitignore
├── requirements.txt                   # Python dependencies
│
├── src/                               # Modular source code
│   ├── __init__.py
│   ├── config.py                      # All hyperparameters & constants
│   ├── data.py                        # Heart Disease UCI loading & preprocessing
│   ├── models.py                      # HeartDiseaseMLP architecture
│   ├── training.py                    # Centralized training & evaluation
│   ├── federated.py                   # FedAvg, FedProx, IID/non-IID splits
│   ├── privacy.py                     # Noise-utility analysis
│   ├── fairness.py                    # SPD, DI, EOD, FPR gap + reweighting
│   └── visualization.py              # Publication-quality plotting
│
├── notebooks/
│   └── FL_Heart_Disease_KHOUAJA_Ahmed.ipynb  # Full experiment notebook
│
├── docs/
│   └── Presentation_FL_KHOUAJA_Ahmed.pdf     # Presentation slides
│
└── figures/
    └── banner.png                     # README banner
```

---

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/kwija/FL-Heart-Disease.git
cd FL-Heart-Disease

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### Run the Full Notebook

```bash
jupyter notebook notebooks/FL_Heart_Disease_KHOUAJA_Ahmed.ipynb
```

### Using the Modules Directly

```python
from src.data import load_and_prepare_data
from src.models import HeartDiseaseMLP
from src.training import set_seed, train_baseline_full, evaluate_medical
from src.federated import split_iid, train_fedavg_full

# Load data
set_seed(42)
data = load_and_prepare_data(42)

# Train centralized baseline
metrics, model = train_baseline_full(data)
print(f"Baseline: acc={metrics['accuracy']:.3f}, ROC-AUC={metrics['roc_auc']:.3f}")

# Run FedAvg (IID)
client_data, _ = split_iid(data["X_train"], data["y_train"], n_clients=5, seed=42)
fl_result = train_fedavg_full(client_data, data)
print(f"FedAvg IID: acc={fl_result['test_at_best_val']:.3f}")
```

---

## 🧪 Methodology

### Data

- **Dataset**: Heart Disease UCI (303 samples, 13 features, binary classification)
- **Split**: Train 177 / Val 60 / Test 60 (stratified)
- **Preprocessing**: StandardScaler (fit on train, transform on val/test)

### Model

| Component | Details |
|:---|:---|
| Architecture | MLP: 13 → 64 → 32 → 2 |
| Normalization | BatchNorm1d |
| Regularization | Dropout (0.3) |
| Optimizer | SGD (momentum=0.9) |
| Learning Rate | 0.01 |
| Gradient Clipping | 1.0 |
| Parameters | 3,234 |

### Federated Learning

| Parameter | Value |
|:---|:---|
| Clients | 5 |
| Rounds | 20 |
| Local Epochs | 5 |
| non-IID | Dirichlet (β=0.5) |
| Model Selection | Best validation accuracy round |

### Evaluation

- **5 seeds** for stability (mean ± std)
- **Medical metrics**: ROC-AUC, PR-AUC, Sensitivity, Specificity, Brier score
- **Fairness**: SPD, Disparate Impact, Equal Opportunity Difference, FPR Gap
- **Privacy**: Empirical noise-utility trade-off (no formal DP)

---

## ⚠️ Limitations

- Test set very small (**60 examples**) — accuracy changes by steps of 1/60 ≈ 0.017
- Small differences between methods cannot be interpreted as "significant"
- FedProx negative result may be due to insufficient heterogeneity or small scale
- Privacy analysis is empirical only (no ε/δ accounting)

---

## 📄 Presentation

The presentation slides are available in [`docs/Presentation_FL_KHOUAJA_Ahmed.pdf`](docs/Presentation_FL_KHOUAJA_Ahmed.pdf).

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <em>Built with ❤️ by KHOUAJA Ahmed — Master 2 MIASHS — February 2026</em>
</p>
