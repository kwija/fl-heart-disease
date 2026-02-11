"""
Training utilities: centralized baseline training, evaluation, and reproducibility.
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .config import BATCH_SIZE, EPOCHS, GRAD_CLIP, LR, PATIENCE, SEEDS
from .data import load_and_prepare_data
from .models import HeartDiseaseMLP


def set_seed(seed):
    """Set seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_medical(model, X, y, return_preds=False):
    """
    Full medical evaluation with diagnostic metrics.

    Returns dict with: accuracy, f1, roc_auc, pr_auc, sensitivity,
    specificity, brier, confusion_matrix.
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)[:, 1].numpy()
        preds = torch.max(logits, 1)[1].numpy()

    y_np = y.numpy()

    acc = accuracy_score(y_np, preds)
    f1 = f1_score(y_np, preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_np, probs)
    except Exception:
        roc_auc = 0.5

    try:
        precision_curve, recall_curve, _ = precision_recall_curve(y_np, probs)
        pr_auc = auc(recall_curve, precision_curve)
    except Exception:
        pr_auc = 0.5

    cm = confusion_matrix(y_np, preds)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        sensitivity, specificity = 0, 0

    brier = brier_score_loss(y_np, probs)

    result = {
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "brier": brier,
        "confusion_matrix": cm.tolist(),
    }

    if return_preds:
        result["preds"] = preds
        result["probs"] = probs

    return result


def train_baseline_full(data, epochs=EPOCHS, patience=PATIENCE, lr=LR):
    """
    Train centralized baseline with SGD + gradient clipping + medical metrics.

    Uses early stopping based on validation loss.
    Returns (metrics_dict, trained_model).
    """
    model = HeartDiseaseMLP(data["input_dim"])
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(data["X_train"], data["y_train"]),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    best_val_loss, best_state, patience_cnt = float("inf"), None, 0

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

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_state)
    metrics = evaluate_medical(model, data["X_test"], data["y_test"])
    return metrics, model


def run_all_baselines(seeds=SEEDS):
    """Run baseline training across all seeds and return aggregated results."""
    results = []
    for s in seeds:
        data = load_and_prepare_data(s)
        set_seed(s)
        metrics, _ = train_baseline_full(data)
        results.append(metrics)

    agg = {}
    for key in ["accuracy", "roc_auc", "pr_auc", "sensitivity", "specificity", "f1", "brier"]:
        vals = [r[key] for r in results]
        agg[key] = {"mean": np.mean(vals), "std": np.std(vals)}

    return results, agg
