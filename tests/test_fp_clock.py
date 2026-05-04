import numpy as np
import torch

from src.clock.fp_clock import (
    FPTrajectoryStats,
    build_fp_clock_profile,
    collect_anchored_replay_stats,
    project_residual_to_frenet_normal,
)
from src.clock.profile import build_reparameterized_bundle


def test_frenet_projection_removes_tangent_component() -> None:
    residual = torch.tensor([[3.0, 4.0], [2.0, 0.0]], dtype=torch.float64)
    tangent = torch.tensor([[1.0, 0.0], [2.0, 0.0]], dtype=torch.float64)

    residual_perp, residual_parallel = project_residual_to_frenet_normal(residual, tangent)

    assert torch.allclose(residual_parallel, torch.tensor([[3.0, 0.0], [2.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(residual_perp, torch.tensor([[0.0, 4.0], [0.0, 0.0]], dtype=torch.float64), atol=1.0e-6)


def _stats_for_reparameterization(grid: np.ndarray) -> FPTrajectoryStats:
    del grid
    delta_s = np.asarray([[0.08, 0.12, 0.2, 0.24, 0.18, 0.1, 0.08]], dtype=np.float64)
    q = np.full_like(delta_s, 3.0)
    arc_defect = np.asarray([[0.4, 0.8, 1.2, 0.6, 1.0, 0.7, 0.5]], dtype=np.float64)
    rho = np.abs(1.0 - np.power(2.0, 1.0 - q))
    residual_perp = arc_defect * np.power(delta_s, q) * rho
    return FPTrajectoryStats(
        full_step_error=residual_perp.copy(),
        half_step_error=0.25 * residual_perp,
        effective_order=q,
        delta_s=delta_s,
        residual_perp_norm=residual_perp,
    )


def _interval_masses(profile) -> np.ndarray:
    return np.diff(profile.tau_profile)


def test_arc_length_pullback_is_invariant_to_physical_reparameterization() -> None:
    arc_nodes = np.asarray([0.0, 0.08, 0.20, 0.40, 0.64, 0.82, 0.92, 1.0], dtype=np.float64)
    linear_grid = arc_nodes.copy()
    quadratic_grid = np.square(arc_nodes)

    linear = build_fp_clock_profile(linear_grid, _stats_for_reparameterization(linear_grid), smoothing_window=1)
    quadratic = build_fp_clock_profile(quadratic_grid, _stats_for_reparameterization(quadratic_grid), smoothing_window=1)

    assert np.allclose(_interval_masses(linear.profile), _interval_masses(quadratic.profile), atol=1.0e-6)


def test_fp_clock_profile_materializes_valid_schedule() -> None:
    grid = np.linspace(1.0, 0.0, 9, dtype=np.float64)
    delta_s = np.tile(np.linspace(0.1, 0.2, 8, dtype=np.float64), (2, 1))
    q = np.full((2, 8), 3.0, dtype=np.float64)
    residual = np.tile(np.asarray([0.02, 0.04, 0.03, 0.05, 0.01, 0.02, 0.03, 0.04]), (2, 1))
    stats = FPTrajectoryStats(
        full_step_error=residual,
        half_step_error=0.25 * residual,
        effective_order=q,
        delta_s=delta_s,
        residual_perp_norm=residual,
    )

    artifacts = build_fp_clock_profile(grid, stats, target_steps=10, smoothing_window=1)
    profile = artifacts.profile
    assert np.all(np.isfinite(profile.alpha_profile))
    assert np.all(profile.alpha_profile > 0.0)
    assert np.isclose(profile.tau_profile[0], 0.0)
    assert np.isclose(profile.tau_profile[-1], 1.0)
    assert np.all(np.diff(profile.tau_profile) >= -1.0e-12)

    bundle = build_reparameterized_bundle(
        profile,
        effective_nfe=6,
        solver_name="euler",
        representation="timesteps",
        schedule_family="FP_CLOCK",
    )
    assert bundle.timesteps is not None
    assert bundle.time_grid is not None
    assert np.all(np.isfinite(bundle.time_grid))
    assert np.all(np.diff(bundle.time_grid) <= 1.0e-12)


def test_anchored_replay_uses_same_anchor_replay_residual() -> None:
    grid = np.asarray([2.0, 1.0, 0.0], dtype=np.float64)
    reference = torch.tensor(
        [
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[2.0, 0.0]],
        ],
        dtype=torch.float64,
    )
    replay_1x = torch.tensor(
        [
            [[1.0, 0.30]],
            [[2.0, 0.20]],
        ],
        dtype=torch.float64,
    )
    replay_2x = torch.tensor(
        [
            [[1.0, 0.10]],
            [[2.0, 0.05]],
        ],
        dtype=torch.float64,
    )
    replay_4x = torch.tensor(
        [
            [[1.0, 0.04]],
            [[2.0, 0.02]],
        ],
        dtype=torch.float64,
    )

    stats, details = collect_anchored_replay_stats(
        physical_grid=grid,
        reference_states=reference,
        replay_1x_endpoints=replay_1x,
        replay_2x_endpoints=replay_2x,
        replay_4x_endpoints=replay_4x,
        window_size=1,
    )

    assert stats.residual_perp_norm.shape == (1, 2)
    assert np.all(stats.residual_perp_norm > 0.0)
    assert details.window_size == 1


def test_anchored_replay_distributes_multistep_windows_to_intervals() -> None:
    grid = np.asarray([4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64)

    def states_for(nodes: np.ndarray) -> torch.Tensor:
        values = []
        for value in nodes:
            x = 4.0 - float(value)
            values.append([[x, 0.1 * x * (4.0 - x)]])
        return torch.tensor(values, dtype=torch.float64)

    reference = states_for(grid)
    replay_1x = reference[1:].clone()
    replay_2x = reference[1:].clone()
    replay_4x = reference[1:].clone()
    replay_1x[..., 1] += torch.tensor([0.20, 0.16, 0.12, 0.08], dtype=torch.float64).reshape(4, 1)
    replay_2x[..., 1] += torch.tensor([0.08, 0.06, 0.04, 0.03], dtype=torch.float64).reshape(4, 1)
    replay_4x[..., 1] += torch.tensor([0.03, 0.02, 0.015, 0.01], dtype=torch.float64).reshape(4, 1)

    stats, details = collect_anchored_replay_stats(
        physical_grid=grid,
        reference_states=reference,
        replay_1x_endpoints=replay_1x,
        replay_2x_endpoints=replay_2x,
        replay_4x_endpoints=replay_4x,
        window_size=2,
        q_min=1.05,
        q_max=6.0,
    )

    assert details.window_size == 2
    assert stats.residual_perp_norm.shape == (1, 4)
    assert stats.delta_s.shape == (1, 4)
    assert np.all(details.coverage > 0.0)
    assert np.all(np.isfinite(stats.residual_perp_norm))
    assert np.all(np.isfinite(stats.effective_order))
