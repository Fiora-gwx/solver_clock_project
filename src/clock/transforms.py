from __future__ import annotations

import numpy as np


def build_lambda_table(scheduler) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return training (timesteps, sigmas, lambdas) arrays from a VP scheduler."""
    if not hasattr(scheduler, "alphas_cumprod"):
        raise RuntimeError(f"Scheduler {scheduler.__class__.__name__} does not expose alphas_cumprod.")
    alphas_cumprod = scheduler.alphas_cumprod.detach().float().cpu().numpy()
    alpha = np.sqrt(np.clip(alphas_cumprod, 1.0e-12, 1.0))
    sigma = np.sqrt(np.clip(1.0 - alphas_cumprod, 0.0, 1.0) / np.clip(alphas_cumprod, 1.0e-12, None))
    lamb = np.log(np.clip(alpha, 1.0e-12, None)) - np.log(np.clip(sigma, 1.0e-12, None))
    timesteps = np.arange(len(alphas_cumprod), dtype=np.float64)
    return timesteps, sigma.astype(np.float64), lamb.astype(np.float64)


def lambda_to_sigma(lamb_nodes: np.ndarray, train_lambdas: np.ndarray, train_sigmas: np.ndarray) -> np.ndarray:
    order = np.argsort(train_lambdas)
    return np.interp(
        np.asarray(lamb_nodes, dtype=np.float64),
        np.asarray(train_lambdas, dtype=np.float64)[order],
        np.asarray(train_sigmas, dtype=np.float64)[order],
    )


def lambda_to_timestep(
    lamb_nodes: np.ndarray,
    train_lambdas: np.ndarray,
    train_timesteps: np.ndarray,
) -> np.ndarray:
    order = np.argsort(train_lambdas)
    return np.interp(
        np.asarray(lamb_nodes, dtype=np.float64),
        np.asarray(train_lambdas, dtype=np.float64)[order],
        np.asarray(train_timesteps, dtype=np.float64)[order],
    )


def sigma_to_lambda(sigmas: np.ndarray, train_sigmas: np.ndarray, train_lambdas: np.ndarray) -> np.ndarray:
    sigma_values = np.asarray(sigmas, dtype=np.float64)
    train_sigma_values = np.asarray(train_sigmas, dtype=np.float64)
    train_lambda_values = np.asarray(train_lambdas, dtype=np.float64)
    order = np.argsort(train_sigma_values)
    sigma_sorted = train_sigma_values[order]
    lambda_sorted = train_lambda_values[order]
    clipped = np.clip(sigma_values, float(sigma_sorted[0]), float(sigma_sorted[-1]))
    lambdas = np.interp(clipped, sigma_sorted, lambda_sorted)
    lambdas = np.asarray(lambdas, dtype=np.float64)
    lambdas[sigma_values <= 0.0] = float(lambda_sorted[0])
    return lambdas


def lambda_to_sigma_derivative(
    lamb_nodes: np.ndarray,
    train_lambdas: np.ndarray,
    train_sigmas: np.ndarray,
) -> np.ndarray:
    order = np.argsort(train_lambdas)
    lamb_sorted = np.asarray(train_lambdas, dtype=np.float64)[order]
    sigma_sorted = np.asarray(train_sigmas, dtype=np.float64)[order]
    derivative = np.gradient(sigma_sorted, lamb_sorted, edge_order=1)
    return np.interp(np.asarray(lamb_nodes, dtype=np.float64), lamb_sorted, derivative)
