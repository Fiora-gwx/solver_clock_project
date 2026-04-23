from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch


ProgressCallback = Callable[[str, dict[str, object]], None]


@dataclass(frozen=True)
class AysEarlyStopConfig:
    metric: str = "fid"
    proxy_num_samples: int = 2048
    batch_size: int | None = None
    patience: int = 3
    min_iterations: int = 3
    min_delta: float = 0.0
    seed_offset: int = 100000

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "AysEarlyStopConfig":
        payload = payload or {}
        batch_size = payload.get("batch_size")
        return cls(
            metric=str(payload.get("metric", cls.metric)),
            proxy_num_samples=int(payload.get("proxy_num_samples", cls.proxy_num_samples)),
            batch_size=None if batch_size is None else int(batch_size),
            patience=int(payload.get("patience", cls.patience)),
            min_iterations=int(payload.get("min_iterations", cls.min_iterations)),
            min_delta=float(payload.get("min_delta", cls.min_delta)),
            seed_offset=int(payload.get("seed_offset", cls.seed_offset)),
        )

    @property
    def enabled(self) -> bool:
        return self.metric.lower() != "none"


@dataclass(frozen=True)
class AysConfig:
    candidate_count: int = 11
    data_samples: int = 8192
    batch_size: int = 128
    sigma_data: float = 0.5
    seed: int = 0
    initial_steps: int = 10
    subdivision_rounds: int = 2
    max_iterations_initial: int = 300
    max_iterations_subdivision: int = 64
    early_stop: AysEarlyStopConfig = field(default_factory=AysEarlyStopConfig)

    @classmethod
    def from_dict(cls, payload: dict[str, object] | None) -> "AysConfig":
        payload = payload or {}
        legacy_max_iterations = int(payload.get("max_iterations", cls.max_iterations_initial))
        return cls(
            candidate_count=int(payload.get("candidate_count", cls.candidate_count)),
            data_samples=int(payload.get("data_samples", cls.data_samples)),
            batch_size=int(payload.get("batch_size", cls.batch_size)),
            sigma_data=float(payload.get("sigma_data", cls.sigma_data)),
            seed=int(payload.get("seed", cls.seed)),
            initial_steps=int(payload.get("initial_steps", cls.initial_steps)),
            subdivision_rounds=int(payload.get("subdivision_rounds", cls.subdivision_rounds)),
            max_iterations_initial=int(payload.get("max_iterations_initial", legacy_max_iterations)),
            max_iterations_subdivision=int(payload.get("max_iterations_subdivision", legacy_max_iterations)),
            early_stop=AysEarlyStopConfig.from_dict(payload.get("early_stop")),
        )

    @property
    def num_batches(self) -> int:
        return max(1, math.ceil(self.data_samples / self.batch_size))

    @property
    def reference_steps(self) -> int:
        return int(self.initial_steps * (2**self.subdivision_rounds))


@dataclass(frozen=True)
class AysOptimizationResult:
    schedule: np.ndarray
    iterations: int
    converged: bool
    stopped_early: bool = False
    best_proxy_value: float | None = None


@dataclass(frozen=True)
class AysHierarchicalResult:
    reference_schedule: np.ndarray
    stage_results: dict[int, AysOptimizationResult]


def time_uniform_schedule(num_train_timesteps: int, num_steps: int) -> np.ndarray:
    if num_steps < 1:
        raise ValueError("AYS requires at least one optimization step.")
    max_timestep = int(num_train_timesteps) - 1
    schedule = np.linspace(0, max_timestep, num_steps + 1, dtype=np.float64)
    schedule = np.round(schedule).astype(np.int64)
    schedule[0] = 0
    schedule[-1] = max_timestep
    return schedule


def build_candidate_grid(left: int, current: int, right: int, count: int) -> np.ndarray:
    lower = int(left) + 1
    upper = int(right) - 1
    if lower > upper:
        return np.asarray([int(current)], dtype=np.int64)
    candidates = np.linspace(lower, upper, max(2, count), dtype=np.float64)
    candidates = np.round(candidates).astype(np.int64)
    candidates = np.unique(np.concatenate([candidates, np.asarray([int(current)], dtype=np.int64)]))
    candidates = candidates[(candidates > left) & (candidates < right)]
    if len(candidates) == 0:
        return np.asarray([int(current)], dtype=np.int64)
    return candidates


def vp_sigma_from_alpha_bar(alpha_bar: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    alpha_bar = np.clip(alpha_bar, eps, 1.0)
    return np.sqrt(np.maximum(1.0 - alpha_bar, 0.0) / alpha_bar)


def build_sigma_lookup(alphas_cumprod: torch.Tensor) -> np.ndarray:
    alpha_bar = alphas_cumprod.detach().cpu().numpy().astype(np.float64)
    return vp_sigma_from_alpha_bar(alpha_bar)


def importance_weights(sigmas: np.ndarray, sigma_upper: np.ndarray, sigma_data: float, eps: float = 1.0e-12) -> np.ndarray:
    sigmas = np.maximum(sigmas, eps)
    sigma_upper = np.maximum(sigma_upper, eps)
    delta = (1.0 / (sigmas**2 + sigma_data**2)) - (1.0 / (sigma_upper**2 + sigma_data**2))
    weights = np.abs(delta) / (sigmas**3)
    return np.maximum(weights, eps)


def schedule_sigmas(schedule: np.ndarray, sigma_lookup: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    sigmas = np.asarray(sigma_lookup[np.asarray(schedule, dtype=np.int64)], dtype=np.float64)
    return np.maximum(sigmas, eps)


def snap_sigmas_to_timesteps(target_sigmas: np.ndarray, sigma_lookup: np.ndarray) -> np.ndarray:
    sigma_lookup = np.asarray(sigma_lookup, dtype=np.float64)
    target_sigmas = np.asarray(target_sigmas, dtype=np.float64)
    last_index = len(sigma_lookup) - 1

    right = np.searchsorted(sigma_lookup, target_sigmas, side="left")
    right = np.clip(right, 0, last_index)
    left = np.clip(right - 1, 0, last_index)
    choose_left = np.abs(sigma_lookup[left] - target_sigmas) <= np.abs(sigma_lookup[right] - target_sigmas)
    snapped = np.where(choose_left, left, right).astype(np.int64)

    snapped[0] = 0
    snapped[-1] = last_index

    for index in range(1, len(snapped) - 1):
        lower = snapped[index - 1] + 1
        upper = last_index - (len(snapped) - index - 1)
        snapped[index] = int(np.clip(snapped[index], lower, upper))

    for index in range(len(snapped) - 2, 0, -1):
        lower = index
        upper = snapped[index + 1] - 1
        snapped[index] = int(np.clip(snapped[index], lower, upper))

    return snapped


def subdivide_schedule(schedule: np.ndarray, sigma_lookup: np.ndarray) -> np.ndarray:
    current_sigmas = schedule_sigmas(schedule, sigma_lookup)
    midpoint_sigmas = np.exp(0.5 * (np.log(current_sigmas[:-1]) + np.log(current_sigmas[1:])))
    refined_sigmas = np.empty(len(schedule) * 2 - 1, dtype=np.float64)
    refined_sigmas[0::2] = current_sigmas
    refined_sigmas[1::2] = midpoint_sigmas
    return snap_sigmas_to_timesteps(refined_sigmas, sigma_lookup)


def interpolate_reference_schedule(reference_schedule: np.ndarray, target_nfe: int, sigma_lookup: np.ndarray) -> np.ndarray:
    if target_nfe < 1:
        raise ValueError("AYS interpolation requires at least one target step.")
    if target_nfe == len(reference_schedule) - 1:
        return np.asarray(reference_schedule, dtype=np.int64).copy()

    reference_sigmas = schedule_sigmas(reference_schedule, sigma_lookup)
    reference_x = np.linspace(0.0, 1.0, len(reference_schedule), dtype=np.float64)
    target_x = np.linspace(0.0, 1.0, target_nfe + 1, dtype=np.float64)
    target_log_sigmas = np.interp(target_x, reference_x, np.log(reference_sigmas))
    target_sigmas = np.exp(target_log_sigmas)
    target_sigmas[0] = reference_sigmas[0]
    target_sigmas[-1] = reference_sigmas[-1]
    return snap_sigmas_to_timesteps(target_sigmas, sigma_lookup)


def forward_interval_samples(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    upper_timesteps: torch.Tensor,
    alphas_cumprod: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise_t = torch.randn_like(x0)

    alpha_bar_t = alphas_cumprod.index_select(0, timesteps)
    alpha_bar_upper = alphas_cumprod.index_select(0, upper_timesteps)

    alpha_t = alpha_bar_t.sqrt().view(-1, 1, 1, 1)
    sigma_t = (1.0 - alpha_bar_t).clamp(min=0.0).sqrt().view(-1, 1, 1, 1)
    x_t = alpha_t * x0 + sigma_t * noise_t

    alpha_ratio = (alpha_bar_upper / alpha_bar_t.clamp(min=1.0e-12)).clamp(min=0.0, max=1.0)
    transition_alpha = alpha_ratio.sqrt().view(-1, 1, 1, 1)
    transition_sigma = (1.0 - alpha_ratio).clamp(min=0.0).sqrt().view(-1, 1, 1, 1)
    noise_interval = torch.randn_like(x0)
    x_upper = transition_alpha * x_t + transition_sigma * noise_interval
    return x_t, x_upper


@torch.no_grad()
def estimate_local_klub(
    *,
    model: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    sigma_lookup: np.ndarray,
    left: int,
    middle: int,
    right: int,
    batch_provider: Callable[[int], torch.Tensor],
    config: AysConfig,
    rng: np.random.Generator,
    device: torch.device,
) -> float:
    if not (left < middle < right):
        return float("inf")

    time_values = np.arange(int(left), int(right) + 1, dtype=np.int64)
    upper_values = np.where(time_values <= int(middle), int(middle), int(right))

    raw_weights = importance_weights(
        sigma_lookup[time_values],
        sigma_lookup[upper_values],
        sigma_data=config.sigma_data,
    )
    total_weight = float(raw_weights.sum())
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        return float("inf")

    probabilities = raw_weights / total_weight
    samples_remaining = int(config.data_samples)
    mse_total = 0.0

    for _ in range(config.num_batches):
        requested_batch = min(config.batch_size, samples_remaining)
        x0 = batch_provider(requested_batch).to(device=device, dtype=torch.float32)
        current_batch = int(x0.shape[0])

        chosen = rng.choice(len(time_values), size=current_batch, replace=True, p=probabilities)
        timestep_batch = torch.from_numpy(time_values[chosen]).to(device=device, dtype=torch.long)
        upper_batch = torch.from_numpy(upper_values[chosen]).to(device=device, dtype=torch.long)

        x_t, x_upper = forward_interval_samples(x0, timestep_batch, upper_batch, alphas_cumprod)
        prediction_t = model(x_t, timestep_batch.float())
        prediction_upper = model(x_upper, upper_batch.float())
        mse = (prediction_t - prediction_upper).reshape(current_batch, -1).pow(2).mean(dim=1)
        mse_total += float(mse.mean().item())
        samples_remaining -= current_batch

    return total_weight * mse_total / float(config.num_batches)


def optimize_schedule(
    *,
    model: torch.nn.Module,
    initial_schedule: np.ndarray,
    active_indices: Sequence[int],
    max_iterations: int,
    alphas_cumprod: torch.Tensor,
    sigma_lookup: np.ndarray,
    batch_provider: Callable[[int], torch.Tensor],
    config: AysConfig,
    device: torch.device,
    proxy_evaluator: Callable[[np.ndarray, int], float] | None = None,
    progress_callback: ProgressCallback | None = None,
    stage_steps: int | None = None,
) -> AysOptimizationResult:
    schedule = np.asarray(initial_schedule, dtype=np.int64).copy()
    if len(active_indices) == 0:
        return AysOptimizationResult(schedule=schedule, iterations=0, converged=True)

    rng = np.random.default_rng(config.seed)
    best_schedule = schedule.copy()
    best_proxy_value: float | None = None
    patience_counter = 0

    for iteration in range(1, max_iterations + 1):
        changed = False
        for index in active_indices:
            candidates = build_candidate_grid(
                int(schedule[index - 1]),
                int(schedule[index]),
                int(schedule[index + 1]),
                config.candidate_count,
            )
            if len(candidates) == 1:
                continue

            scores = [
                estimate_local_klub(
                    model=model,
                    alphas_cumprod=alphas_cumprod,
                    sigma_lookup=sigma_lookup,
                    left=int(schedule[index - 1]),
                    middle=int(candidate),
                    right=int(schedule[index + 1]),
                    batch_provider=batch_provider,
                    config=config,
                    rng=rng,
                    device=device,
                )
                for candidate in candidates
            ]
            best_index = int(np.argmin(scores))
            best_candidate = int(candidates[best_index])
            if best_candidate != int(schedule[index]):
                schedule[index] = best_candidate
                changed = True

        if proxy_evaluator is not None and config.early_stop.enabled:
            if progress_callback is not None:
                progress_callback(
                    "proxy_start",
                    {
                        "stage_steps": stage_steps or (len(schedule) - 1),
                        "iteration": iteration,
                        "active_points": len(active_indices),
                    },
                )
            proxy_value = float(proxy_evaluator(schedule.copy(), iteration))
            if best_proxy_value is None or proxy_value < best_proxy_value - config.early_stop.min_delta:
                best_proxy_value = proxy_value
                best_schedule = schedule.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            if progress_callback is not None:
                progress_callback(
                    "proxy_end",
                    {
                        "stage_steps": stage_steps or (len(schedule) - 1),
                        "iteration": iteration,
                        "proxy_value": proxy_value,
                        "best_proxy_value": best_proxy_value,
                        "patience_counter": patience_counter,
                        "patience": config.early_stop.patience,
                    },
                )
            if iteration >= config.early_stop.min_iterations and patience_counter >= config.early_stop.patience:
                if progress_callback is not None:
                    progress_callback(
                        "stage_stop",
                        {
                            "stage_steps": stage_steps or (len(schedule) - 1),
                            "iteration": iteration,
                            "reason": "early_stop",
                            "best_proxy_value": best_proxy_value,
                        },
                    )
                return AysOptimizationResult(
                    schedule=best_schedule,
                    iterations=iteration,
                    converged=False,
                    stopped_early=True,
                    best_proxy_value=best_proxy_value,
                )

        if progress_callback is not None:
            progress_callback(
                "iteration_end",
                {
                    "stage_steps": stage_steps or (len(schedule) - 1),
                    "iteration": iteration,
                    "max_iterations": max_iterations,
                    "changed": changed,
                    "best_proxy_value": best_proxy_value,
                    "active_points": len(active_indices),
                },
            )

        if not changed:
            if progress_callback is not None:
                progress_callback(
                    "stage_stop",
                    {
                        "stage_steps": stage_steps or (len(schedule) - 1),
                        "iteration": iteration,
                        "reason": "converged_no_change",
                        "best_proxy_value": best_proxy_value,
                    },
                )
            return AysOptimizationResult(
                schedule=best_schedule if best_proxy_value is not None else schedule,
                iterations=iteration,
                converged=True,
                best_proxy_value=best_proxy_value,
            )

    if progress_callback is not None:
        progress_callback(
            "stage_stop",
            {
                "stage_steps": stage_steps or (len(schedule) - 1),
                "iteration": max_iterations,
                "reason": "max_iterations",
                "best_proxy_value": best_proxy_value,
            },
        )
    return AysOptimizationResult(
        schedule=best_schedule if best_proxy_value is not None else schedule,
        iterations=max_iterations,
        converged=False,
        best_proxy_value=best_proxy_value,
    )


def hierarchical_optimize_schedule(
    *,
    model: torch.nn.Module,
    num_train_timesteps: int,
    alphas_cumprod: torch.Tensor,
    sigma_lookup: np.ndarray,
    batch_provider: Callable[[int], torch.Tensor],
    config: AysConfig,
    device: torch.device,
    proxy_evaluator: Callable[[np.ndarray, int], float] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AysHierarchicalResult:
    if config.initial_steps < 2:
        raise ValueError("AYS initial_steps must be at least 2.")

    initial_schedule = time_uniform_schedule(num_train_timesteps, config.initial_steps)
    stage_results: dict[int, AysOptimizationResult] = {}
    if progress_callback is not None:
        progress_callback(
            "stage_start",
            {
                "stage_steps": len(initial_schedule) - 1,
                "max_iterations": config.max_iterations_initial,
                "active_points": len(initial_schedule) - 2,
                "uses_proxy_metric": proxy_evaluator is not None and config.early_stop.enabled,
            },
        )

    current_result = optimize_schedule(
        model=model,
        initial_schedule=initial_schedule,
        active_indices=tuple(range(1, len(initial_schedule) - 1)),
        max_iterations=config.max_iterations_initial,
        alphas_cumprod=alphas_cumprod,
        sigma_lookup=sigma_lookup,
        batch_provider=batch_provider,
        config=config,
        device=device,
        proxy_evaluator=proxy_evaluator,
        progress_callback=progress_callback,
        stage_steps=len(initial_schedule) - 1,
    )
    stage_results[len(current_result.schedule) - 1] = current_result
    current_schedule = current_result.schedule
    if progress_callback is not None:
        progress_callback(
            "stage_end",
            {
                "stage_steps": len(current_schedule) - 1,
                "iterations_ran": current_result.iterations,
                "converged": current_result.converged,
                "stopped_early": current_result.stopped_early,
                "best_proxy_value": current_result.best_proxy_value,
            },
        )

    for _ in range(config.subdivision_rounds):
        current_schedule = subdivide_schedule(current_schedule, sigma_lookup)
        active_indices = tuple(index for index in range(1, len(current_schedule) - 1) if index % 2 == 1)
        if progress_callback is not None:
            progress_callback(
                "stage_start",
                {
                    "stage_steps": len(current_schedule) - 1,
                    "max_iterations": config.max_iterations_subdivision,
                    "active_points": len(active_indices),
                    "uses_proxy_metric": False,
                },
            )
        current_result = optimize_schedule(
            model=model,
            initial_schedule=current_schedule,
            active_indices=active_indices,
            max_iterations=config.max_iterations_subdivision,
            alphas_cumprod=alphas_cumprod,
            sigma_lookup=sigma_lookup,
            batch_provider=batch_provider,
            config=config,
            device=device,
            progress_callback=progress_callback,
            stage_steps=len(current_schedule) - 1,
        )
        stage_results[len(current_result.schedule) - 1] = current_result
        current_schedule = current_result.schedule
        if progress_callback is not None:
            progress_callback(
                "stage_end",
                {
                    "stage_steps": len(current_schedule) - 1,
                    "iterations_ran": current_result.iterations,
                    "converged": current_result.converged,
                    "stopped_early": current_result.stopped_early,
                    "best_proxy_value": current_result.best_proxy_value,
                },
            )

    return AysHierarchicalResult(reference_schedule=current_schedule, stage_results=stage_results)


def schedule_for_nfe(
    *,
    target_nfe: int,
    hierarchical_result: AysHierarchicalResult,
    sigma_lookup: np.ndarray,
) -> tuple[np.ndarray, str]:
    if target_nfe in hierarchical_result.stage_results:
        return hierarchical_result.stage_results[target_nfe].schedule.copy(), "optimized_stage"

    reference_schedule = hierarchical_result.reference_schedule
    if target_nfe > len(reference_schedule) - 1:
        raise ValueError(
            f"AYS reference schedule has {len(reference_schedule) - 1} steps, cannot derive target NFE {target_nfe}."
        )
    return interpolate_reference_schedule(reference_schedule, target_nfe, sigma_lookup), "log_linear_interpolation"
