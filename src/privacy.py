"""
Privacy analysis: gradient noise injection for noise-utility trade-off study.

Note: This is an empirical robustness study, NOT formal (ε,δ)-DP guarantees.
"""
import numpy as np
import torch
import torch.nn as nn

from .config import (
    BATCH_SIZE,
    GRAD_CLIP,
    LOCAL_EPOCHS,
    LR,
    N_CLIENTS,
    N_ROUNDS,
    SEEDS,
    SIGMA_VALUES,
)
from .data import load_and_prepare_data, make_train_loader
from .federated import split_iid
from .models import HeartDiseaseMLP
from .training import set_seed


def train_fl_noise(client_data, data, sigma, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS):
    """
    FedAvg with gradient noise injection for privacy analysis.

    After gradient clipping, Gaussian noise (σ) is added to gradients.
    """
    n_clients = len(client_data)
    global_model = HeartDiseaseMLP(data["input_dim"])
    val_accs, test_accs = [], []

    for r in range(n_rounds):
        client_states, client_sizes = [], []
        for X_c, y_c in client_data:
            local = HeartDiseaseMLP(data["input_dim"])
            local.load_state_dict(global_model.state_dict())
            opt = torch.optim.SGD(local.parameters(), lr=LR, momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            loader, eff_n = make_train_loader(X_c, y_c, BATCH_SIZE)
            if loader is None:
                client_states.append(
                    {k: v.clone() for k, v in global_model.state_dict().items()}
                )
                client_sizes.append(0)
                continue

            local.train()
            for _ in range(local_epochs):
                for xb, yb in loader:
                    opt.zero_grad()
                    criterion(local(xb), yb).backward()
                    torch.nn.utils.clip_grad_norm_(local.parameters(), GRAD_CLIP)
                    if sigma > 0:
                        for p in local.parameters():
                            if p.grad is not None:
                                p.grad += torch.randn_like(p.grad) * sigma
                    opt.step()
            client_states.append(local.state_dict())
            client_sizes.append(eff_n)

        total = sum(client_sizes)
        if total == 0:
            raise RuntimeError("All clients have <2 samples: BatchNorm impossible.")

        new_state = {
            k: sum(
                client_sizes[i] / total * client_states[i][k]
                for i in range(n_clients)
            )
            for k in global_model.state_dict()
        }
        global_model.load_state_dict(new_state)

        global_model.eval()
        with torch.no_grad():
            val_accs.append(
                (torch.max(global_model(data["X_val"]), 1)[1] == data["y_val"])
                .float()
                .mean()
                .item()
            )
            test_accs.append(
                (torch.max(global_model(data["X_test"]), 1)[1] == data["y_test"])
                .float()
                .mean()
                .item()
            )

    best_r = np.argmax(val_accs)
    return {"test_at_best_val": test_accs[best_r], "sigma": sigma}


def run_all_privacy(seeds=SEEDS, sigma_values=SIGMA_VALUES):
    """Run noise-utility experiments across seeds and sigma values."""
    results = {}
    for sigma in sigma_values:
        results_sigma = []
        for seed in seeds:
            data = load_and_prepare_data(seed)
            set_seed(seed)
            cd, _ = split_iid(data["X_train"], data["y_train"], N_CLIENTS, seed)
            r = train_fl_noise(cd, data, sigma)
            results_sigma.append(r)

        accs = [r["test_at_best_val"] for r in results_sigma]
        results[f"sigma={sigma}"] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "sigma": sigma,
            "per_seed": accs,
        }

    return results
