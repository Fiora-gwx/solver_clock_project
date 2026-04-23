import numpy as np
import torch

from src.clock.lcs import (
    build_lcs_profile,
    collect_lcs_norms,
    heun2_step,
    material_derivative_jvp,
)


def test_material_derivative_matches_affine_vector_field() -> None:
    a = 2.0
    b = 3.0

    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        return a * sample + b * coordinate

    sample = torch.tensor([[1.5], [-0.5]], dtype=torch.float32)
    coordinate = 0.25
    derivative = material_derivative_jvp(velocity_fn, sample, coordinate)
    expected = b + a * (a * sample + b * coordinate)
    assert torch.allclose(derivative, expected, atol=1.0e-5)


def test_heun2_step_is_exact_for_constant_velocity_field() -> None:
    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        del coordinate
        return torch.ones_like(sample) * 2.0

    sample = torch.tensor([[0.0]], dtype=torch.float32)
    big = heun2_step(velocity_fn, sample, 0.0, 0.5)
    small = heun2_step(velocity_fn, heun2_step(velocity_fn, sample, 0.0, 0.25), 0.25, 0.5)
    assert torch.allclose(big, small, atol=1.0e-6)


def test_collect_lcs_norms_and_profile_build_for_order_one() -> None:
    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        return sample.square() + coordinate

    grid = np.linspace(1.0, 0.0, 5, dtype=np.float64)
    initial_sample = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    norms = collect_lcs_norms(
        initial_sample=initial_sample,
        physical_grid=grid,
        velocity_fn=velocity_fn,
        pilot_solver="euler",
        order=1,
        observation_microbatch=1,
    )
    artifacts = build_lcs_profile(grid, norms, order=1, smoothing_window=3, eps=1.0e-6)
    assert norms.shape == (2, 5)
    assert artifacts.profile.alpha_profile.shape == (5,)
    assert np.all(artifacts.profile.density >= 0.0)
    assert np.isclose(artifacts.profile.tau_profile[0], 0.0)
    assert np.isclose(artifacts.profile.tau_profile[-1], 1.0)


def test_collect_lcs_norms_rejects_deleted_order_two() -> None:
    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        return sample.square() + 3.0 * coordinate

    grid = np.linspace(1.0, 0.0, 5, dtype=np.float64)
    initial_sample = torch.tensor([[0.15]], dtype=torch.float32)
    try:
        collect_lcs_norms(
            initial_sample=initial_sample,
            physical_grid=grid,
            velocity_fn=velocity_fn,
            pilot_solver="heun2",
            order=2,
        )
    except ValueError as error:
        assert "LCS-2" in str(error)
    else:
        raise AssertionError("LCS-2 collection should be disabled.")


def test_generic_lcs_stepper_rejects_stork_alias() -> None:
    def velocity_fn(sample: torch.Tensor, coordinate: torch.Tensor) -> torch.Tensor:
        return sample.square() + coordinate

    grid = np.linspace(1.0, 0.0, 5, dtype=np.float64)
    initial_sample = torch.tensor([[0.1], [0.2]], dtype=torch.float32)
    try:
        collect_lcs_norms(
            initial_sample=initial_sample,
            physical_grid=grid,
            velocity_fn=velocity_fn,
            pilot_solver="stork4_2nd",
            order=1,
            observation_microbatch=1,
        )
    except ValueError as error:
        assert "backend adapter" in str(error)
    else:
        raise AssertionError("Generic LCS stepper should not alias STORK to Heun2.")
