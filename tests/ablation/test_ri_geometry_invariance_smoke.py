import numpy as np
import torch

from src.clock.archive.ri_sadb import build_ri_sadb_profile, collect_ri_sadb_stats


def test_straight_line_nonuniform_parameterization_has_no_artificial_curvature() -> None:
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


def test_circle_arc_has_unit_curvature_and_uniform_pure_geometry_clock() -> None:
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
    assert float(np.std(artifacts.interval_alpha_profile) / np.mean(artifacts.interval_alpha_profile)) < 0.05
