import numpy as np
import torch

from src.clock.defect_balanced import StepRefinementStats, build_defect_balanced_profile
from src.clock.ri_sadb import TrajectoryGeometryStats, build_ri_sadb_profile, collect_ri_sadb_stats


def test_original_sadb_interval_alpha_formula_is_unchanged() -> None:
    grid = np.asarray([1.0, 0.5, 0.0], dtype=np.float64)
    stats = StepRefinementStats(
        full_step_error=np.ones((1, 2), dtype=np.float64),
        half_step_error=np.ones((1, 2), dtype=np.float64),
        effective_order=np.asarray([[2.0, 3.0]], dtype=np.float64),
        defect_strength=np.asarray([[4.0, 8.0]], dtype=np.float64),
    )

    artifacts = build_defect_balanced_profile(grid, stats, smoothing_window=1, eps=1.0e-12)

    expected = np.asarray(
        [
            ((2.0 - 1.0) * 4.0) ** (1.0 / 2.0),
            ((3.0 - 1.0) * 8.0) ** (1.0 / 3.0),
        ],
        dtype=np.float64,
    )
    assert np.allclose(artifacts.interval_alpha_profile, expected)


def test_ri_sadb_profile_is_finite_positive_and_monotone() -> None:
    grid = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    stats = TrajectoryGeometryStats(
        full_step_error=np.ones((2, 4), dtype=np.float64),
        half_step_error=0.25 * np.ones((2, 4), dtype=np.float64),
        effective_order=np.full((2, 4), 3.0, dtype=np.float64),
        delta_s=np.tile(np.linspace(0.1, 0.2, 4, dtype=np.float64), (2, 1)),
        curvature=np.tile(np.asarray([0.0, 0.5, 1.0, 0.2], dtype=np.float64), (2, 1)),
        residual_perp_norm=np.tile(np.asarray([0.2, 0.4, 0.1, 0.3], dtype=np.float64), (2, 1)),
        residual_parallel_norm=np.tile(np.asarray([1.0, 0.5, 0.2, 0.1], dtype=np.float64), (2, 1)),
    )

    artifacts = build_ri_sadb_profile(
        grid,
        stats,
        target_steps=10,
        eta=0.25,
        beta=0.0,
        smoothing_window=1,
    )

    assert np.all(np.isfinite(artifacts.profile.alpha_profile))
    assert np.all(artifacts.profile.alpha_profile > 0.0)
    assert np.isclose(artifacts.profile.tau_profile[0], 0.0)
    assert np.isclose(artifacts.profile.tau_profile[-1], 1.0)
    assert np.all(np.diff(artifacts.profile.tau_profile) >= -1.0e-12)


def test_ri_sadb_straight_line_reparameterization_has_zero_normal_curvature() -> None:
    grid = np.linspace(0.2, 1.0, 9, dtype=np.float64)

    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        del sample
        value = float(coordinate.detach().cpu().item())
        return torch.tensor([[3.0 * value * value, 0.0]], dtype=torch.float64)

    def exact_step(sample: torch.Tensor, start: float, end: float) -> torch.Tensor:
        del sample, start
        return torch.tensor([[float(end) ** 3.0, 0.0]], dtype=torch.float64)

    stats = collect_ri_sadb_stats(
        initial_sample=torch.tensor([[grid[0] ** 3.0, 0.0]], dtype=torch.float64),
        physical_grid=grid,
        velocity_fn=velocity_fn,
        step_fn=exact_step,
        eps=1.0e-12,
    )

    assert np.max(stats.curvature) < 1.0e-5
    artifacts = build_ri_sadb_profile(grid, stats, target_steps=8, eta=1.0, beta=0.0)
    assert np.all(np.isfinite(artifacts.interval_alpha_profile))


def test_ri_sadb_circle_arc_has_unit_curvature_and_uniform_geometry_density() -> None:
    grid = np.linspace(0.0, 0.5 * np.pi, 33, dtype=np.float64)

    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        del sample
        value = float(coordinate.detach().cpu().item())
        return torch.tensor([[-np.sin(value), np.cos(value)]], dtype=torch.float64)

    def exact_step(sample: torch.Tensor, start: float, end: float) -> torch.Tensor:
        del sample, start
        return torch.tensor([[np.cos(float(end)), np.sin(float(end))]], dtype=torch.float64)

    stats = collect_ri_sadb_stats(
        initial_sample=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
        physical_grid=grid,
        velocity_fn=velocity_fn,
        step_fn=exact_step,
        eps=1.0e-12,
    )

    assert np.allclose(stats.curvature, 1.0, atol=0.04, rtol=0.04)
    artifacts = build_ri_sadb_profile(grid, stats, target_steps=16, eta=1.0, beta=0.0)
    interval_alpha = artifacts.interval_alpha_profile
    assert float(np.std(interval_alpha) / np.mean(interval_alpha)) < 0.05
