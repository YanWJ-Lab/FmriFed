"""
evaluate.py
===========
Evaluation utilities for FDGAT:
  • compute_metrics  – Accuracy, Sensitivity (Recall), F1-score
  • MemberInferenceAttack – basic shadow-model MIA simulation
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Standard classification metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Parameters
    ----------
    y_true      : binary ground-truth labels (0 / 1)
    y_pred_prob : predicted seizure probability in [0, 1]
    threshold   : decision boundary

    Returns
    -------
    dict with keys: accuracy, sensitivity, f1, auc
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        auc = float("nan")

    return {"accuracy": acc, "sensitivity": sens, "f1": f1, "auc": auc}


def print_metrics(metrics: Dict[str, float], prefix: str = "") -> None:
    prefix_str = f"[{prefix}] " if prefix else ""
    print(
        f"{prefix_str}"
        f"Accuracy={metrics['accuracy']:.4f}  "
        f"Sensitivity={metrics['sensitivity']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"AUC={metrics['auc']:.4f}"
    )


# ---------------------------------------------------------------------------
# Member Inference Attack (MIA) simulation
# ---------------------------------------------------------------------------

class MemberInferenceAttack:
    """
    Basic threshold-based MIA:
    The adversary assumes that *members* (training samples) will have
    higher model confidence (lower loss) than *non-members*.

    Attack procedure
    ----------------
    1. Collect model predictions for training samples (members) and
       held-out test samples (non-members).
    2. Train a simple threshold classifier on loss values.
    3. Measure attack success (AUC).  A perfectly private model gives AUC ≈ 0.5.

    Parameters
    ----------
    model          : FDGAT instance (eval mode)
    member_loader  : DataLoader for training set (members)
    nonmember_loader : DataLoader for test set (non-members)
    device         : torch device
    """

    def __init__(
        self,
        model,
        member_loader,
        nonmember_loader,
        device,
    ) -> None:
        import torch
        self.model            = model
        self.member_loader    = member_loader
        self.nonmember_loader = nonmember_loader
        self.device           = device

    def _get_losses(self, loader) -> np.ndarray:
        import torch
        from model import FDGAT

        self.model.eval()
        losses = []
        with torch.no_grad():
            for X, y in loader:
                X = X.to(self.device)
                y = y.to(self.device)
                pred = self.model(X)
                loss_per_sample = -(
                    y * torch.log(pred + 1e-8)
                    + (1 - y) * torch.log(1 - pred + 1e-8)
                )
                losses.extend(loss_per_sample.cpu().numpy())
        return np.array(losses)

    def run(self) -> Dict[str, float]:
        """
        Execute the MIA and return a results dict.

        Returns
        -------
        {
          "attack_auc"   : ROC-AUC of the attack (0.5 = random = private),
          "attack_acc"   : accuracy at optimal threshold,
          "member_loss_mean"    : avg loss on training samples,
          "nonmember_loss_mean" : avg loss on test samples,
        }
        """
        member_losses    = self._get_losses(self.member_loader)
        nonmember_losses = self._get_losses(self.nonmember_loader)

        # Labels: 1 = member, 0 = non-member
        all_losses = np.concatenate([member_losses, nonmember_losses])
        all_labels = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(nonmember_losses)),
        ])

        # Attack score: lower loss → more likely a member
        attack_scores = -all_losses   # negate so higher = more likely member

        try:
            attack_auc = roc_auc_score(all_labels, attack_scores)
        except ValueError:
            attack_auc = float("nan")

        # Threshold: median loss across all samples
        threshold    = np.median(all_losses)
        pred_members = (all_losses <= threshold).astype(int)
        attack_acc   = accuracy_score(all_labels, pred_members)

        return {
            "attack_auc":           attack_auc,
            "attack_acc":           attack_acc,
            "member_loss_mean":     float(member_losses.mean()),
            "nonmember_loss_mean":  float(nonmember_losses.mean()),
            "loss_gap":             float(member_losses.mean() - nonmember_losses.mean()),
        }

    def print_report(self, results: Dict[str, float]) -> None:
        print("\n" + "=" * 55)
        print("  Member Inference Attack (MIA) Report")
        print("=" * 55)
        print(f"  Attack AUC   : {results['attack_auc']:.4f}  (0.5 = fully private)")
        print(f"  Attack Acc   : {results['attack_acc']:.4f}  (0.5 = fully private)")
        print(f"  Member loss  : {results['member_loss_mean']:.4f}")
        print(f"  Non-member   : {results['nonmember_loss_mean']:.4f}")
        print(f"  Loss gap     : {results['loss_gap']:.4f}  (0 = no leakage)")
        privacy_rating = (
            "HIGH" if results["attack_auc"] < 0.55
            else "MEDIUM" if results["attack_auc"] < 0.65
            else "LOW"
        )
        print(f"  Privacy      : {privacy_rating}")
        print("=" * 55 + "\n")
