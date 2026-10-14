"""
server.py
=========
Flower FL server for FDGAT.

Strategy: FedAvg with server-side weight perturbation.

After aggregating client updates the server adds a small amount of Gaussian
noise to the aggregated weights before broadcasting them back.  This provides
an additional layer of MIA defence at the server level and can also be viewed
as a form of server-side differential privacy.
"""

from __future__ import annotations

from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import flwr as fl
from flwr.common import (
    FitRes,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate


# ---------------------------------------------------------------------------
# Custom FedAvg strategy with server-side weight perturbation
# ---------------------------------------------------------------------------

class FedAvgWithWeightPerturbation(fl.server.strategy.FedAvg):
    """
    Extends standard FedAvg with:
      1. Server-side Gaussian noise added to the aggregated global model
         (additional MIA / reconstruction-attack defence).
      2. Logging of per-round aggregated metrics (train_loss, accuracy, …).

    Parameters
    ----------
    server_noise_std : float
        Standard deviation of the Gaussian noise added to the aggregated
        weights before broadcasting.  Set to 0.0 to disable.
    **kwargs         : passed through to FedAvg
    """

    def __init__(
        self,
        server_noise_std: float = 1e-5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.server_noise_std = server_noise_std
        self.round_metrics:   List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    # Override aggregate_fit to inject noise into the global model
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, FitRes]],
        failures:     List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # 1. Weighted average (standard FedAvg)
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results)

        # 2. Server-side weight perturbation
        if self.server_noise_std > 0.0:
            aggregated_ndarrays = _perturb_weights(
                aggregated_ndarrays, self.server_noise_std
            )

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # 3. Collect and log per-client metrics
        metrics_aggregated = _aggregate_metrics(results)
        metrics_aggregated["round"] = server_round
        self.round_metrics.append(metrics_aggregated)

        if failures:
            print(f"[Server] Round {server_round}: {len(failures)} client(s) failed.")

        print(
            f"[Server] Round {server_round} | "
            f"clients={len(results)} | "
            + " | ".join(f"{k}={v:.4f}" for k, v in metrics_aggregated.items()
                         if k != "round")
        )

        return parameters_aggregated, metrics_aggregated

    # ------------------------------------------------------------------
    # Override aggregate_evaluate (optional – log evaluation metrics)
    # ------------------------------------------------------------------

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      List,
        failures:     List,
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        print(
            f"[Server] Round {server_round} EVAL | "
            f"loss={loss_aggregated:.4f} | "
            + " | ".join(
                f"{k}={v:.4f}" for k, v in metrics_aggregated.items()
            )
        )
        return loss_aggregated, metrics_aggregated

    def get_round_history(self) -> List[Dict[str, float]]:
        """Return the list of per-round aggregated metrics."""
        return self.round_metrics


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _perturb_weights(ndarrays: NDArrays, noise_std: float) -> NDArrays:
    """Add scaled Gaussian noise to each weight array."""
    perturbed = []
    for w in ndarrays:
        scale = noise_std * (np.linalg.norm(w) + 1e-8)
        w_noisy = w + np.random.normal(0.0, scale, size=w.shape).astype(w.dtype)
        perturbed.append(w_noisy)
    return perturbed


def _aggregate_metrics(
    results: List[Tuple[ClientProxy, FitRes]]
) -> Dict[str, float]:
    """
    Compute weighted average of scalar metrics returned by clients.
    Only numeric (float/int) values are averaged; others are ignored.
    """
    total_examples = sum(fit_res.num_examples for _, fit_res in results)
    if total_examples == 0:
        return {}

    aggregated: Dict[str, float] = {}
    for _, fit_res in results:
        weight = fit_res.num_examples / total_examples
        for k, v in fit_res.metrics.items():
            if isinstance(v, (int, float)):
                aggregated[k] = aggregated.get(k, 0.0) + weight * float(v)

    return aggregated


# ---------------------------------------------------------------------------
# Factory function (used by main.py)
# ---------------------------------------------------------------------------

def build_strategy(
    initial_parameters: Optional[Parameters] = None,
    num_rounds:         int   = 10,
    min_clients:        int   = 3,
    server_noise_std:   float = 1e-5,
    fraction_fit:       float = 1.0,
    fraction_evaluate:  float = 1.0,
) -> FedAvgWithWeightPerturbation:
    """Return a configured FedAvgWithWeightPerturbation strategy."""
    return FedAvgWithWeightPerturbation(
        server_noise_std    = server_noise_std,
        fraction_fit        = fraction_fit,
        fraction_evaluate   = fraction_evaluate,
        min_fit_clients     = min_clients,
        min_evaluate_clients= min_clients,
        min_available_clients= min_clients,
        initial_parameters  = initial_parameters,
        fit_metrics_aggregation_fn   = None,   # handled in aggregate_fit
        evaluate_metrics_aggregation_fn = None,
    )
