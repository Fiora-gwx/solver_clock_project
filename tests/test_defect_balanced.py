import numpy as np
import torch

from src.clock.defect_balanced import (
    StepRefinementStats,
    build_defect_balanced_profile,
    build_velocity_stepper,
    collect_step_refinement_stats,
    collect_velocity_curvature_stats,
    estimate_refinement_order_and_defect,
    nonuniform_second_derivative,
)
from src.clock.profile import build_reparameterized_bundle
from src.utils.schedule_bundle import ScheduleBundle


def test_refinement_ratio_estimates_effective_order_and_defect() -> None:
    full_error = np.asarray([[0.03, 0.006]], dtype=np.float64)
    half_error = full_error / 4.0
    step_sizes = np.asarray([0.5, 0.25], dtype=np.float64)

    q_eff, defect = estimate_refinement_order_and_defect(
        full_step_error=full_error,
        half_step_error=half_error,
        step_sizes=step_sizes,
        q_min=1.05,
        q_max=6.0,
        eps=1.0e-12,
    )

    expected_defect = full_error / (np.abs(step_sizes)[None, :] ** 3.0 * 0.75)
    assert np.allclose(q_eff, 3.0)
    assert np.allclose(defect, expected_defect)


def test_collect_step_refinement_stats_uses_solver_step_behavior() -> None:
    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        del coordinate
        return sample.square()

    grid = np.linspace(0.0, 0.5, 4, dtype=np.float64)
    initial_sample = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    stats = collect_step_refinement_stats(
        initial_sample=initial_sample,
        physical_grid=grid,
        step_fn=build_velocity_stepper(velocity_fn, "euler"),
        observation_microbatch=1,
        q_min=1.05,
        q_max=6.0,
        eps=1.0e-12,
    )

    assert stats.full_step_error.shape == (2, 3)
    assert stats.half_step_error.shape == (2, 3)
    assert stats.effective_order.shape == (2, 3)
    assert stats.defect_strength.shape == (2, 3)
    assert np.all(np.isfinite(stats.effective_order))
    assert np.all(stats.defect_strength > 0.0)


def test_nonuniform_second_derivative_is_exact_for_quadratic_descending_grid() -> None:
    t_k = 0.55
    t_km1 = 0.7
    t_km2 = 1.0
    h1 = t_km1 - t_k
    h2 = t_km2 - t_km1

    second = nonuniform_second_derivative(
        torch.tensor(t_k * t_k, dtype=torch.float64),
        torch.tensor(t_km1 * t_km1, dtype=torch.float64),
        torch.tensor(t_km2 * t_km2, dtype=torch.float64),
        h1,
        h2,
    )

    assert torch.allclose(second, torch.tensor(2.0, dtype=torch.float64), atol=1.0e-12, rtol=1.0e-12)


def test_velocity_curvature_stats_are_floored_for_linear_velocity() -> None:
    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(sample) * coordinate

    def pilot_step(sample: torch.Tensor, start: float, end: float) -> torch.Tensor:
        return sample + (float(end) - float(start))

    stats = collect_velocity_curvature_stats(
        initial_sample=torch.zeros(2, 1, dtype=torch.float64),
        physical_grid=np.asarray([1.0, 0.7, 0.55, 0.2, 0.0], dtype=np.float64),
        velocity_fn=velocity_fn,
        pilot_step_fn=pilot_step,
        q_const=3.0,
        eps=1.0e-9,
    )

    assert stats.defect_strength.shape == (2, 4)
    assert np.allclose(stats.effective_order, 3.0)
    assert np.allclose(stats.defect_strength, 1.0e-9)


def test_velocity_curvature_profile_densifies_localized_curvature_and_saves_bundle(tmp_path) -> None:
    grid = np.linspace(1.0, 0.0, 65, dtype=np.float64)

    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        bump = torch.exp(-100.0 * (coordinate - 0.5).square())
        return torch.ones_like(sample) * bump

    def pilot_step(sample: torch.Tensor, start: float, end: float) -> torch.Tensor:
        del start, end
        return sample

    stats = collect_velocity_curvature_stats(
        initial_sample=torch.zeros(2, 1),
        physical_grid=grid,
        velocity_fn=velocity_fn,
        pilot_step_fn=pilot_step,
        q_const=3.0,
        eps=1.0e-12,
        defect_clip_quantile=0.95,
    )
    artifacts = build_defect_balanced_profile(grid, stats, smoothing_window=3, eps=1.0e-12)
    center = np.argmin(np.abs(grid[:-1] - 0.5))
    edge = np.argmin(np.abs(grid[:-1] - 0.9))
    assert artifacts.interval_alpha_profile[center] > artifacts.interval_alpha_profile[edge]

    bundle = build_reparameterized_bundle(
        artifacts.profile,
        effective_nfe=7,
        solver_name="euler",
        representation="timesteps",
        schedule_family="SADB",
        meta={"estimator": "velocity_curvature"},
    )
    bundle.save(tmp_path)
    loaded = ScheduleBundle.load(tmp_path)

    assert loaded.time_grid is not None
    assert loaded.timesteps is not None
    assert np.all(np.diff(loaded.time_grid) < 0.0)
    assert np.all(np.isfinite(loaded.time_grid))
    assert loaded.meta["estimator"] == "velocity_curvature"


def test_defect_balanced_profile_is_normalized_and_monotone() -> None:
    grid = np.linspace(1.0, 0.0, 5, dtype=np.float64)
    stats = StepRefinementStats(
        full_step_error=np.ones((2, 4), dtype=np.float64),
        half_step_error=0.5 * np.ones((2, 4), dtype=np.float64),
        effective_order=np.tile(np.linspace(1.05, 2.0, 4, dtype=np.float64), (2, 1)),
        defect_strength=np.tile(np.linspace(4.0, 1.0, 4, dtype=np.float64), (2, 1)),
    )

    artifacts = build_defect_balanced_profile(grid, stats, smoothing_window=3, eps=1.0e-12)

    assert artifacts.profile.alpha_profile.shape == (5,)
    assert artifacts.defect_profile.shape == (4,)
    assert artifacts.effective_order_profile.shape == (4,)
    assert np.all(artifacts.profile.density >= 0.0)
    assert np.isclose(artifacts.profile.tau_profile[0], 0.0)
    assert np.isclose(artifacts.profile.tau_profile[-1], 1.0)
    assert np.all(np.diff(artifacts.profile.tau_profile) > 0.0)


def test_defect_balanced_profile_uses_local_order_density_weight() -> None:
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
