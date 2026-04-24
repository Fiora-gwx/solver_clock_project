import numpy as np
import torch

from src.clock.defect_balanced import (
    StepRefinementStats,
    build_defect_balanced_profile,
    build_velocity_stepper,
    collect_step_refinement_stats,
    estimate_refinement_order_and_defect,
)


def test_refinement_ratio_estimates_effective_order_and_defect() -> None:
    full_error = np.asarray([[0.03, 0.006]], dtype=np.float64)
    half_error = full_error / 4.0
    step_sizes = np.asarray([0.5, 0.25], dtype=np.float64)

    q_eff, defect = estimate_refinement_order_and_defect(
        full_step_error=full_error,
        half_step_error=half_error,
        step_sizes=step_sizes,
        q_min=0.25,
        q_max=6.0,
        eps=1.0e-12,
    )

    expected_defect = full_error / (np.abs(step_sizes)[None, :] ** 3.0 * 0.75)
    assert np.allclose(q_eff, 2.0)
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
        q_min=0.25,
        q_max=6.0,
        eps=1.0e-12,
    )

    assert stats.full_step_error.shape == (2, 3)
    assert stats.half_step_error.shape == (2, 3)
    assert stats.effective_order.shape == (2, 3)
    assert stats.defect_strength.shape == (2, 3)
    assert np.all(np.isfinite(stats.effective_order))
    assert np.all(stats.defect_strength > 0.0)


def test_defect_balanced_profile_is_normalized_and_monotone() -> None:
    grid = np.linspace(1.0, 0.0, 5, dtype=np.float64)
    stats = StepRefinementStats(
        full_step_error=np.ones((2, 4), dtype=np.float64),
        half_step_error=0.5 * np.ones((2, 4), dtype=np.float64),
        effective_order=np.tile(np.linspace(1.0, 2.0, 4, dtype=np.float64), (2, 1)),
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
