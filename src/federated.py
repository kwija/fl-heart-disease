"""
Federated Learning: FedAvg, FedProx, IID/non-IID partitioning.
"""
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .config import (
    BATCH_SIZE,
    BETA_NONIID,
    GRAD_CLIP,
    LOCAL_EPOCHS,
    LR,
    MU_VALUES,
    N_CLIENTS,
    N_ROUNDS,
    SEEDS,
)
from .data import load_and_prepare_data, make_train_loader
from .models import HeartDiseaseMLP
from .training import evaluate_medical, set_seed


def split_iid(X, y, n_clients, seed):
    """Split data IID across clients."""
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    splits = np.array_split(idx, n_clients)
    client_data = [(X[s], y[s]) for s in splits]
    proof = [
        {
            "client": i,
            "n_samples": len(yc),
            "n_pos": int((yc == 1).sum()),
            "n_neg": int((yc == 0).sum()),
            "pos_rate": yc.float().mean().item(),
        }
        for i, (xc, yc) in enumerate(client_data)
    ]
    return client_data, proof


def split_noniid_fixed(X, y, n_clients, beta, seed, max_attempts=100):
    """Split data non-IID via Dirichlet with guarantee of no empty clients."""
    for attempt in range(max_attempts):
        np.random.seed(seed + attempt)
        n_classes = len(torch.unique(y))
        label_idx = [torch.where(y == c)[0].numpy() for c in range(n_classes)]
        client_idx = [[] for _ in range(n_clients)]

        for c in range(n_classes):
            idx_c = label_idx[c].copy()
            np.random.shuffle(idx_c)
            props = np.random.dirichlet([beta] * n_clients)
            splits = (np.cumsum(props) * len(idx_c)).astype(int)
            for i, part in enumerate(np.split(idx_c, splits[:-1])):
                client_idx[i].extend(part.tolist())

        if all(len(idx) > 0 for idx in client_idx):
            break

    assert len(client_idx) == n_clients, f"Expected {n_clients} clients"
    assert all(len(idx) > 0 for idx in client_idx), "Empty clients!"

    client_data, proof = [], []
    for i, idx in enumerate(client_idx):
        arr = np.array(idx)
        xc, yc = X[arr], y[arr]
        client_data.append((xc, yc))
        proof.append(
            {
                "client": i,
                "n_samples": len(yc),
                "n_pos": int((yc == 1).sum()),
                "n_neg": int((yc == 0).sum()),
                "pos_rate": yc.float().mean().item(),
            }
        )
    return client_data, proof


def train_fedavg_full(client_data, data, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS):
    """
    FedAvg with gradient clipping + medical metrics.

    Uses validation-based best round selection.
    """
    n_clients = len(client_data)
    global_model = HeartDiseaseMLP(data["input_dim"])
    n_params = sum(p.numel() for p in global_model.parameters())
    val_accs, test_accs = [], []
    round_times = []

    for r in range(n_rounds):
        t_start = time.time()
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
        round_times.append(time.time() - t_start)

        global_model.eval()
        with torch.no_grad():
            val_pred = torch.max(global_model(data["X_val"]), 1)[1]
            val_acc = (val_pred == data["y_val"]).float().mean().item()
            test_pred = torch.max(global_model(data["X_test"]), 1)[1]
            test_acc = (test_pred == data["y_test"]).float().mean().item()
        val_accs.append(val_acc)
        test_accs.append(test_acc)

    best_round = np.argmax(val_accs)
    metrics = evaluate_medical(global_model, data["X_test"], data["y_test"])

    return {
        "val_accs": val_accs,
        "test_accs": test_accs,
        "best_round": int(best_round),
        "test_at_best_val": test_accs[best_round],
        "metrics": metrics,
        "time_total": sum(round_times),
        "comm_cost_mb": n_params * 4 * n_clients * n_rounds * 2 / 1e6,
    }


def train_fedprox_full(client_data, data, mu, n_rounds=N_ROUNDS, local_epochs=LOCAL_EPOCHS):
    """FedProx with proximal term (mu) + gradient clipping."""
    n_clients = len(client_data)
    global_model = HeartDiseaseMLP(data["input_dim"])
    val_accs, test_accs = [], []

    for r in range(n_rounds):
        global_w = {k: v.clone() for k, v in global_model.state_dict().items()}
        client_states, client_sizes = [], []

        for X_c, y_c in client_data:
            local = HeartDiseaseMLP(data["input_dim"])
            local.load_state_dict(global_w)
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
                    loss = criterion(local(xb), yb)
                    if mu > 0:
                        prox = sum(
                            ((p - global_w[n]) ** 2).sum()
                            for n, p in local.named_parameters()
                        )
                        loss += (mu / 2) * prox
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(local.parameters(), GRAD_CLIP)
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
    metrics = evaluate_medical(global_model, data["X_test"], data["y_test"])
    return {"test_at_best_val": test_accs[best_r], "metrics": metrics, "mu": mu}


def aggregate_fl(results_list, seeds=SEEDS):
    """Aggregate FL results across seeds."""
    assert len(results_list) == len(seeds), "Missing seeds!"
    accs = [r["test_at_best_val"] for r in results_list]
    roc_aucs = [r["metrics"]["roc_auc"] for r in results_list]
    return {
        "acc_mean": np.mean(accs),
        "acc_std": np.std(accs),
        "roc_auc_mean": np.mean(roc_aucs),
        "roc_auc_std": np.std(roc_aucs),
        "time": np.mean([r["time_total"] for r in results_list]),
        "comm_mb": results_list[0]["comm_cost_mb"],
    }


def run_all_fl_experiments(seeds=SEEDS):
    """Run full FedAvg experiments (IID + non-IID) across all seeds."""
    fl_results = {"iid": [], "noniid": []}
    client_distributions = {"iid": [], "noniid": []}

    for seed in seeds:
        data = load_and_prepare_data(seed)
        set_seed(seed)

        # IID
        cd_iid, _ = split_iid(data["X_train"], data["y_train"], N_CLIENTS, seed)
        r_iid = train_fedavg_full(cd_iid, data)
        fl_results["iid"].append(r_iid)
        if seed == seeds[0]:
            client_distributions["iid"] = [
                y_c.detach().cpu().numpy() for (_, y_c) in cd_iid
            ]

        # non-IID
        set_seed(seed)
        cd_noniid, _ = split_noniid_fixed(
            data["X_train"], data["y_train"], N_CLIENTS, BETA_NONIID, seed
        )
        r_noniid = train_fedavg_full(cd_noniid, data)
        fl_results["noniid"].append(r_noniid)
        if seed == seeds[0]:
            client_distributions["noniid"] = [
                y_c.detach().cpu().numpy() for (_, y_c) in cd_noniid
            ]

    agg_iid = aggregate_fl(fl_results["iid"], seeds)
    agg_noniid = aggregate_fl(fl_results["noniid"], seeds)

    return fl_results, client_distributions, agg_iid, agg_noniid


def run_all_fedprox(seeds=SEEDS, mu_values=MU_VALUES):
    """Run FedProx experiments across seeds and mu values."""
    results = {}
    for mu in mu_values:
        results_mu = []
        for seed in seeds:
            data = load_and_prepare_data(seed)
            set_seed(seed)
            cd, _ = split_noniid_fixed(
                data["X_train"], data["y_train"], N_CLIENTS, BETA_NONIID, seed
            )
            r = train_fedprox_full(cd, data, mu)
            results_mu.append(r)

        accs = [r["test_at_best_val"] for r in results_mu]
        name = "FedAVG" if mu == 0 else f"mu={mu}"
        results[name] = {
            "mean": np.mean(accs),
            "std": np.std(accs),
            "per_seed": accs,
        }

    return results
