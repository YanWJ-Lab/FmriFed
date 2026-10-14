"""
client.py  (updated)
====================
Flower FL client for FDGAT with **full adaptive local-DP integration**.

Privacy mechanisms (layered)
-----------------------------
1. **Data Perturbation (DP-Data)**
   Gaussian noise added to raw EEG at dataset load time (data_loader.py).

2. **Adaptive Gradient Perturbation (LDP-Grad)** ← NEW
   • LIC (Layer Importance Coefficient) computed on a warm-up pass.
   • Per-layer privacy budget ε_m allocated inversely to importance.
   • Gradients clipped (DP-SGD norm clip) then perturbed via PM-ISUB.

3. **Weight Perturbation (MIA defence)**
   Additional Gaussian noise added to weights before upload to server.
   Now bypassed in favour of the more principled LDP gradient perturbation,
   but kept as a configurable fallback via `use_weight_perturbation`.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import NDArrays, Scalar

from data_loader import FederatedDataManager
from model import FDGAT, LayerImportanceDP
from evaluate import compute_metrics


# ---------------------------------------------------------------------------
# Default hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Scalar] = {
    "local_epochs":          3,
    "lr":                    1e-3,
    "weight_decay":          1e-4,
    # --- Adaptive LDP (gradient perturbation) ---
    "total_eps":             1.0,    # total privacy budget ε_T
    "clip_norm":             1.0,    # gradient clip norm C
    "lic_warmup_batches":    5,      # mini-batches used to compute LIC
    # --- Legacy weight-noise fallback ---
    "use_weight_perturbation": False,
    "weight_noise_std":      1e-4,
}


# ---------------------------------------------------------------------------
# FDGATClient
# ---------------------------------------------------------------------------

class FDGATClient(fl.client.NumPyClient):
    """
    Flower NumPy client for FDGAT with adaptive local-DP.

    Parameters
    ----------
    client_id    : integer ID (0, 1, 2, …)
    data_manager : FederatedDataManager
    model        : shared FDGAT architecture template
    device       : torch.device
    config       : optional hyper-parameter overrides
    """

    def __init__(
        self,
        client_id:    int,
        data_manager: FederatedDataManager,
        model:        FDGAT,
        device:       torch.device,
        config:       Optional[Dict[str, Scalar]] = None,
    ) -> None:
        self.client_id = client_id
        self.device    = device
        self.cfg       = {**DEFAULT_CONFIG, **(config or {})}

        # Each client owns its own model copy
        self.model      = copy.deepcopy(model).to(device)
        self.pos_weight = data_manager.pos_weight.to(device)

        self.train_loader, self.val_loader, self.test_loader = (
            data_manager.get_client_loaders(client_id)
        )

        # Adaptive LDP module (LIC not yet computed)
        self.dp_module: Optional[LayerImportanceDP] = None
        self._lic_computed = False

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> NDArrays:
        return _model_to_ndarrays(self.model)

    def set_parameters(self, parameters: NDArrays) -> None:
        _ndarrays_to_model(self.model, parameters)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """
        1. Receive global weights.
        2. (First round only) Compute LIC and allocate per-layer ε_m.
        3. Train locally with PM-ISUB gradient perturbation each step.
        4. Return perturbed weights + metadata.
        """
        cfg = {**self.cfg, **config}
        self.set_parameters(parameters)

        # --- Initialise / refresh DP module each round ---
        total_eps  = float(cfg["total_eps"])
        clip_norm  = float(cfg["clip_norm"])
        self.dp_module = LayerImportanceDP(
            self.model, total_eps=total_eps, clip_norm=clip_norm
        )

        # Compute LIC on local training data (warm-up pass)
        if not self._lic_computed:
            self.dp_module.compute_lic(
                self.train_loader,
                device=str(self.device),
                num_batches=int(cfg["lic_warmup_batches"]),
            )
            self._lic_computed = True
            print(
                f"[Client {self.client_id}] Privacy budget allocation:\n"
                + self.dp_module.summary()
            )
        else:
            # Reuse previous LIC but re-allocate budgets for new total_eps
            self.dp_module._allocate_budgets()

        # --- Local training with gradient-level LDP ---
        train_loss = self._train(
            epochs=int(cfg["local_epochs"]),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )

        # --- Optional weight perturbation fallback (MIA defence) ---
        if cfg.get("use_weight_perturbation", False):
            final_params = _add_weight_noise(
                self.model, float(cfg["weight_noise_std"])
            )
        else:
            final_params = _model_to_ndarrays(self.model)

        metrics: Dict[str, Scalar] = {
            "train_loss": float(train_loss),
            "client_id":  self.client_id,
            "total_eps":  total_eps,
        }
        return final_params, len(self.train_loader.dataset), metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        self.set_parameters(parameters)
        loss, metrics = self._evaluate(self.test_loader)
        return float(loss), len(self.test_loader.dataset), metrics

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train(
        self,
        epochs:       int,
        lr:           float,
        weight_decay: float,
    ) -> float:
        self.model.train()
        optimizer  = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        total_loss = 0.0
        n_batches  = 0

        for epoch in range(epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self.model(X_batch)
                loss = FDGAT.loss(pred, y_batch, pos_weight=self.pos_weight)
                loss.backward()

                # ---- Adaptive LDP: clip + PM-ISUB gradient perturbation ----
                if self.dp_module is not None:
                    self.dp_module.perturb_gradients()
                else:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg["clip_norm"]
                    )

                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        print(
            f"[Client {self.client_id}] "
            f"epochs={epochs}  avg_train_loss={avg:.4f}"
        )
        return avg

    def _evaluate(
        self, loader: DataLoader
    ) -> Tuple[float, Dict[str, Scalar]]:
        self.model.eval()
        all_preds, all_targets = [], []
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred    = self.model(X_batch)
                loss    = FDGAT.loss(pred, y_batch)
                total_loss += loss.item() * len(y_batch)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / max(len(loader.dataset), 1)
        metrics  = compute_metrics(
            np.array(all_targets), np.array(all_preds)
        )
        return avg_loss, {k: float(v) for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_to_ndarrays(model: nn.Module) -> NDArrays:
    return [v.cpu().numpy() for v in model.state_dict().values()]


def _ndarrays_to_model(model: nn.Module, arrays: NDArrays) -> None:
    state = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), arrays)}
    )
    model.load_state_dict(state, strict=True)


def _add_weight_noise(model: nn.Module, noise_std: float) -> NDArrays:
    """Legacy Gaussian weight perturbation (MIA defence fallback)."""
    noisy: NDArrays = []
    for param in model.parameters():
        w     = param.detach().cpu().numpy().copy()
        scale = noise_std * (np.linalg.norm(w) + 1e-8)
        w    += np.random.normal(0.0, scale, size=w.shape).astype(w.dtype)
        noisy.append(w)
    return noisy
