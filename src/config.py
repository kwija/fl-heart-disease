"""
Configuration: All hyperparameters and constants for the FL Heart Disease project.
"""
from pathlib import Path

# ─── Seeds for reproducibility ──────────────────────────────────────────────
SEEDS = [42, 123, 456, 789, 2024]

# ─── Model architecture ─────────────────────────────────────────────────────
H1 = 64          # First hidden layer size
H2 = 32          # Second hidden layer size
OUT = 2          # Number of output classes (binary)
DROPOUT = 0.3    # Dropout rate

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LR = 0.01
GRAD_CLIP = 1.0  # Gradient clipping norm
EPOCHS = 100     # Max epochs for baseline training
PATIENCE = 15    # Early stopping patience

# ─── Federated Learning ─────────────────────────────────────────────────────
N_CLIENTS = 5
N_ROUNDS = 20
LOCAL_EPOCHS = 5
BETA_NONIID = 0.5  # Dirichlet distribution parameter for non-IID

# ─── FedProx ─────────────────────────────────────────────────────────────────
MU_VALUES = [0.0, 0.01, 0.1]

# ─── Privacy (gradient noise) ───────────────────────────────────────────────
SIGMA_VALUES = [0.0, 0.01, 0.05, 0.1]

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT = Path.cwd()
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# ─── Visualization ───────────────────────────────────────────────────────────
COLORS = {
    "iid": "#3498db",
    "noniid": "#e74c3c",
    "baseline": "#2ecc71",
    "fedprox": "#9b59b6",
    "privacy": "#f39c12",
    "fair": "#1abc9c",
}
