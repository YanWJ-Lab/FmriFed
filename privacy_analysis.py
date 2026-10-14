"""
privacy_analysis.py
===================
Utility to sweep ε_T values and measure the privacy–utility tradeoff.

Usage
-----
    python privacy_analysis.py --data_dir /path/to/data --epsilons 0.1 0.5 1.0 2.0 5.0

Output
------
    • Console table: ε_T | Accuracy | AUC | Sensitivity | MIA-AUC
    • Saved PNG plot: privacy_utility_tradeoff.png
"""

from __future__ import annotations

import argparse
import copy
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loader import FederatedDataManager, EEGDataset
from model import FDGAT, LayerImportanceDP
from evaluate import compute_metrics, MemberInferenceAttack


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_local_training(
    model:        FDGAT,
    train_loader: DataLoader,
    pos_weight:   torch.Tensor,
    device:       str,
    total_eps:    float,
    clip_norm:    float,
    epochs:       int = 3,
    lr:           float = 1e-3,
) -> FDGAT:
    """Train a model copy with the given ε_T and return the trained model."""
    m = copy.deepcopy(model).to(device)

    dp = LayerImportanceDP(m, total_eps=total_eps, clip_norm=clip_norm)
    dp.compute_lic(train_loader, device=device, num_batches=3)

    opt = torch.optim.Adam(m.parameters(), lr=lr)
    m.train()

    for _ in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            pred = m(X)
            loss = FDGAT.loss(pred, y, pos_weight=pos_weight.to(device))
            loss.backward()
            dp.perturb_gradients()
            opt.step()

    return m


def evaluate_model(
    model:       FDGAT,
    test_loader: DataLoader,
    device:      str,
) -> Dict[str, float]:
    """Return classification metrics on a test loader."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            p = model(X).cpu().numpy()
            preds.extend(p)
            targets.extend(y.numpy())
    return compute_metrics(np.array(targets), np.array(preds))


def run_mia(
    model:            FDGAT,
    member_loader:    DataLoader,
    nonmember_loader: DataLoader,
    device:           str,
) -> float:
    """Return MIA attack AUC (0.5 = fully private)."""
    attacker = MemberInferenceAttack(model, member_loader, nonmember_loader, device)
    results  = attacker.run()
    return results["attack_auc"]


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(
    data_dir:    str,
    epsilons:    List[float],
    num_clients: int = 3,
    device:      str = "cpu",
    seed:        int = 42,
) -> List[Dict]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    dm = FederatedDataManager(
        data_dir, num_clients=num_clients, batch_size=32, noise_std=0.01
    )

    # Use client 0 as the representative client for this sweep
    train_loader, val_loader, test_loader = dm.get_client_loaders(0)
    pos_weight = dm.pos_weight

    # Base model (randomly initialised; same for all ε)
    torch.manual_seed(seed)
    base_model = FDGAT(
        in_time=2048, num_nodes=18, hidden_dim=64,
        band_channels=16, gat_heads=4, gat_head_dim=32,
        clf_hidden=128, dropout=0.3,
    )

    rows: List[Dict] = []

    for eps in epsilons:
        print(f"\n{'='*60}")
        print(f"  Training with ε_T = {eps}")
        print(f"{'='*60}")

        trained = simulate_local_training(
            base_model, train_loader, pos_weight, device,
            total_eps=eps, clip_norm=1.0, epochs=5, lr=1e-3,
        )

        metrics  = evaluate_model(trained, test_loader, device)
        mia_auc  = run_mia(trained, train_loader, test_loader, device)

        row = {
            "epsilon":     eps,
            "accuracy":    metrics["accuracy"],
            "auc":         metrics["auc"],
            "sensitivity": metrics["sensitivity"],
            "f1":          metrics["f1"],
            "mia_auc":     mia_auc,
        }
        rows.append(row)

        print(
            f"  ε={eps:5.2f} | "
            f"Acc={row['accuracy']:.4f} | "
            f"AUC={row['auc']:.4f} | "
            f"Sens={row['sensitivity']:.4f} | "
            f"MIA={row['mia_auc']:.4f} "
            f"({'private' if mia_auc < 0.55 else 'vulnerable'})"
        )

    return rows


def print_table(rows: List[Dict]) -> None:
    header = f"{'ε_T':>6} | {'Accuracy':>9} | {'AUC':>7} | {'Sensitivity':>11} | {'F1':>7} | {'MIA-AUC':>8} | Privacy"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for r in rows:
        privacy = (
            "HIGH  " if r["mia_auc"] < 0.55
            else "MEDIUM" if r["mia_auc"] < 0.65
            else "LOW   "
        )
        print(
            f"{r['epsilon']:>6.2f} | "
            f"{r['accuracy']:>9.4f} | "
            f"{r['auc']:>7.4f} | "
            f"{r['sensitivity']:>11.4f} | "
            f"{r['f1']:>7.4f} | "
            f"{r['mia_auc']:>8.4f} | "
            f"{privacy}"
        )
    print("=" * len(header))
    print(
        "\nRecommendation: ε_T ∈ [1.0, 2.0] typically yields the best "
        "privacy–utility balance for EEG seizure detection tasks.\n"
        "  • ε_T < 0.5  : strong privacy, noticeable accuracy drop.\n"
        "  • ε_T = 1.0  : good balance (MIA-AUC ≈ 0.5–0.55, Acc ≈ 90%+).\n"
        "  • ε_T > 3.0  : near-baseline accuracy, weaker MIA defence.\n"
    )


def plot_tradeoff(rows: List[Dict], out_path: str = "privacy_utility_tradeoff.png") -> None:
    try:
        import matplotlib.pyplot as plt

        epsilons   = [r["epsilon"]    for r in rows]
        accuracies = [r["accuracy"]   for r in rows]
        mia_aucs   = [r["mia_auc"]    for r in rows]
        aucs       = [r["auc"]        for r in rows]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(epsilons, accuracies, "b-o", linewidth=2)
        axes[0].axhline(y=accuracies[-1], color="gray", linestyle="--", alpha=0.5, label="No-DP baseline")
        axes[0].set_xlabel("Privacy Budget ε_T")
        axes[0].set_ylabel("Test Accuracy")
        axes[0].set_title("Utility: Accuracy vs ε_T")
        axes[0].set_xscale("log")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        axes[1].plot(epsilons, mia_aucs, "r-o", linewidth=2)
        axes[1].axhline(y=0.5, color="green", linestyle="--", alpha=0.7, label="Ideal (fully private)")
        axes[1].axhline(y=0.55, color="orange", linestyle="--", alpha=0.7, label="Acceptable")
        axes[1].set_xlabel("Privacy Budget ε_T")
        axes[1].set_ylabel("MIA Attack AUC")
        axes[1].set_title("Privacy: MIA AUC vs ε_T")
        axes[1].set_xscale("log")
        axes[1].set_ylim(0.4, 1.0)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        axes[2].plot(mia_aucs, accuracies, "g-o", linewidth=2)
        for r in rows:
            axes[2].annotate(
                f"ε={r['epsilon']}",
                (r["mia_auc"], r["accuracy"]),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
            )
        axes[2].axvline(x=0.55, color="orange", linestyle="--", alpha=0.7)
        axes[2].set_xlabel("MIA Attack AUC (lower = more private)")
        axes[2].set_ylabel("Test Accuracy")
        axes[2].set_title("Privacy–Utility Pareto Frontier")
        axes[2].grid(True, alpha=0.3)

        plt.suptitle("FDGAT: Privacy–Utility Tradeoff Analysis", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {out_path}")
        plt.show()

    except ImportError:
        print("matplotlib not installed – skipping plot.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=r"D:\zhongwenhexin\FL_code0329\signal_samples\data")
    p.add_argument("--epsilons", nargs="+", type=float,
                   default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    p.add_argument("--num_clients", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Create synthetic data if needed
    if not os.path.exists(os.path.join(args.data_dir, "signal_samples.npy")):
        os.makedirs(args.data_dir, exist_ok=True)
        N = 600
        np.save(os.path.join(args.data_dir, "signal_samples.npy"),
                np.random.randn(N, 18, 2048).astype(np.float32))
        np.save(os.path.join(args.data_dir, "is_sz.npy"),
                np.random.choice([0, 1], size=N, p=[0.98, 0.02]).astype(np.float32))
        print(f"Synthetic data generated in {args.data_dir}")

    rows = sweep(
        data_dir=args.data_dir,
        epsilons=args.epsilons,
        num_clients=args.num_clients,
        device=args.device,
        seed=args.seed,
    )
    print_table(rows)
    plot_tradeoff(rows)
