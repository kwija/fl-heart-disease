"""
Visualization utilities for publication-quality plotting.
"""
import matplotlib.pyplot as plt
import numpy as np

from .config import COLORS


def setup_style():
    """Configure matplotlib for publication-quality plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.size": 11,
        "figure.figsize": (10, 6),
        "figure.dpi": 100,
    })


def plot_convergence_curves(fl_results, save_path=None):
    """Plot accuracy vs rounds (convergence) for IID and non-IID."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (mode, label, color) in zip(
        axes,
        [("iid", "IID", COLORS["iid"]), ("noniid", "non-IID", COLORS["noniid"])],
    ):
        for i, result in enumerate(fl_results[mode]):
            alpha = 0.3 if i > 0 else 1.0
            ax.plot(result["val_accs"], alpha=alpha, color=color, linewidth=1.5)
        ax.set_title(f"FedAvg — {label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Validation Accuracy")
        ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_client_distributions(client_distributions, save_path=None):
    """Plot IID vs non-IID client data distributions as bar charts."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (mode, label) in zip(axes, [("iid", "IID"), ("noniid", "non-IID")]):
        dist = client_distributions[mode]
        n_clients = len(dist)
        pos_counts = [int((d == 1).sum()) for d in dist]
        neg_counts = [int((d == 0).sum()) for d in dist]
        x = np.arange(n_clients)
        width = 0.35

        ax.bar(x - width / 2, pos_counts, width, label="Positive", color="#e74c3c")
        ax.bar(x + width / 2, neg_counts, width, label="Negative", color="#3498db")
        ax.set_title(f"Client Distribution — {label}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Client")
        ax.set_ylabel("Number of Samples")
        ax.set_xticks(x)
        ax.set_xticklabels([f"C{i}" for i in range(n_clients)])
        ax.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_results_comparison(baseline_agg, agg_iid, agg_noniid, save_path=None):
    """Plot synthesis bar chart comparing all methods."""
    setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    methods = ["Baseline", "FedAvg (IID)", "FedAvg (non-IID)"]
    means = [
        baseline_agg["accuracy"]["mean"],
        agg_iid["acc_mean"],
        agg_noniid["acc_mean"],
    ]
    stds = [
        baseline_agg["accuracy"]["std"],
        agg_iid["acc_std"],
        agg_noniid["acc_std"],
    ]
    colors = [COLORS["baseline"], COLORS["iid"], COLORS["noniid"]]

    bars = ax.bar(methods, means, yerr=stds, capsize=8, color=colors, edgecolor="white",
                  linewidth=2, alpha=0.9)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.01,
            f"{mean:.3f}±{std:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Performance Comparison (5 seeds)", fontsize=15, fontweight="bold")
    ax.set_ylim(0.5, 1.0)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cost_analysis(agg_iid, agg_noniid, baseline_time=None, save_path=None):
    """Plot time and communication cost comparison."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Time comparison
    times = [agg_iid["time"], agg_noniid["time"]]
    if baseline_time is not None:
        labels = ["Baseline", "FedAvg IID", "FedAvg non-IID"]
        times = [baseline_time] + times
        colors = [COLORS["baseline"], COLORS["iid"], COLORS["noniid"]]
    else:
        labels = ["FedAvg IID", "FedAvg non-IID"]
        colors = [COLORS["iid"], COLORS["noniid"]]

    axes[0].bar(labels, times, color=colors, edgecolor="white", linewidth=2)
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Training Time", fontsize=14, fontweight="bold")

    # Communication cost
    comm = [agg_iid["comm_mb"], agg_noniid["comm_mb"]]
    axes[1].bar(["IID", "non-IID"], comm,
                color=[COLORS["iid"], COLORS["noniid"]],
                edgecolor="white", linewidth=2)
    axes[1].set_ylabel("Communication (MB)")
    axes[1].set_title("Communication Cost", fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
