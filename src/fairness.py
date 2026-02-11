"""
Fairness analysis: SPD, DI, EOD, FPR gap, and reweighting mitigation.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .config import BATCH_SIZE, GRAD_CLIP, LR, SEEDS
from .data import load_and_prepare_data
from .models import HeartDiseaseMLP
from .training import set_seed, train_baseline_full


def compute_fairness_full(y_true, y_pred, protected):
    """
    Compute full fairness metrics with group sizes and confusion matrices.

    Metrics: SPD, DI, EOD, FPR gap.
    Protected attribute: 0=female, 1=male.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    protected = np.array(protected)
    female, male = protected == 0, protected == 1

    n_f, n_m = int(female.sum()), int(male.sum())

    pr_f = y_pred[female].mean() if n_f > 0 else 0
    pr_m = y_pred[male].mean() if n_m > 0 else 0
    spd = pr_m - pr_f
    di = pr_f / pr_m if pr_m > 0 else 0

    def rates(mask):
        tp = ((y_pred == 1) & (y_true == 1) & mask).sum()
        fp = ((y_pred == 1) & (y_true == 0) & mask).sum()
        tn = ((y_pred == 0) & (y_true == 0) & mask).sum()
        fn = ((y_pred == 0) & (y_true == 1) & mask).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        return {
            "TP": int(tp), "FP": int(fp),
            "TN": int(tn), "FN": int(fn),
            "TPR": tpr, "FPR": fpr,
        }

    rates_f = rates(female)
    rates_m = rates(male)

    eod = rates_m["TPR"] - rates_f["TPR"]
    fpr_gap = rates_m["FPR"] - rates_f["FPR"]

    return {
        "spd": spd, "di": di, "eod": eod, "fpr_gap": fpr_gap,
        "n_female": n_f, "n_male": n_m,
        "female": rates_f, "male": rates_m,
    }


def train_reweighted(data, epochs=100, lr=LR):
    """Train model with fairness reweighting (inverse-frequency sampling)."""
    n = len(data["y_train"])
    weights = torch.ones(n)
    sex_t = torch.LongTensor(data["sex_train"])
    for a in [0, 1]:
        for y in [0, 1]:
            mask = (sex_t == a) & (data["y_train"] == y)
            count = mask.sum().item()
            if count > 0:
                weights[mask] = n / (4 * count)

    model = HeartDiseaseMLP(data["input_dim"])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    loader = DataLoader(
        TensorDataset(data["X_train"], data["y_train"]),
        batch_size=BATCH_SIZE,
        sampler=sampler,
    )

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(data["X_val"]), data["y_val"]).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model


def run_all_fairness(seeds=None):
    """Run fairness analysis (baseline + reweighting) across seeds."""
    if seeds is None:
        seeds = SEEDS[:3]  # 3 seeds for fairness analysis

    results = {"baseline": [], "reweighting": []}

    for seed in seeds:
        data = load_and_prepare_data(seed)
        set_seed(seed)

        # Baseline
        _, model_base = train_baseline_full(data)
        model_base.eval()
        with torch.no_grad():
            pred_base = torch.max(model_base(data["X_test"]), 1)[1].numpy()
        fair_base = compute_fairness_full(
            data["y_test"].numpy(), pred_base, data["sex_test"]
        )
        results["baseline"].append(fair_base)

        # Reweighting
        set_seed(seed)
        model_rew = train_reweighted(data)
        model_rew.eval()
        with torch.no_grad():
            pred_rew = torch.max(model_rew(data["X_test"]), 1)[1].numpy()
        fair_rew = compute_fairness_full(
            data["y_test"].numpy(), pred_rew, data["sex_test"]
        )
        results["reweighting"].append(fair_rew)

    return results
