import numpy as np

from src.clock.va import (
    build_reparameterized_bundle,
    build_shared_clock_profile,
    slice_profile_interval,
)
from src.clock.lcs import LCS_SCHEDULE_IMPLEMENTATION_VERSION
from src.utils.nfe_budget import resolve_effective_nfe_plan


def test_shared_clock_profile_is_normalized_and_monotone() -> None:
    physical_grid = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    material_derivative_norms = np.asarray(
        [
            [4.0, 3.0, 2.0, 1.0, 0.5],
            [3.5, 2.5, 2.0, 1.5, 0.5],
        ],
        dtype=np.float64,
    )
    profile = build_shared_clock_profile(physical_grid, material_derivative_norms, eps=1.0e-6)
    assert np.all(profile.density >= 0.0)
    assert np.isclose(np.trapz(profile.density, profile.physical_grid), 1.0)
    assert np.isclose(profile.tau_profile[0], 0.0)
    assert np.isclose(profile.tau_profile[-1], 1.0)
    assert np.all(np.diff(profile.tau_profile) > 0.0)


def test_shared_clock_profile_uses_first_order_material_derivative_power() -> None:
    physical_grid = np.asarray([0.0, 1.0], dtype=np.float64)
    material_derivative_norms = np.asarray([[4.0, 1.0]], dtype=np.float64)
    profile = build_shared_clock_profile(physical_grid, material_derivative_norms, eps=1.0e-6)

    assert np.allclose(profile.alpha_profile, np.asarray([2.0, 1.0], dtype=np.float64))


def test_materialized_bundle_matches_solver_steps() -> None:
    physical_grid = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    material_derivative_norms = np.tile(np.linspace(4.0, 1.0, 9, dtype=np.float64), (3, 1))
    profile = build_shared_clock_profile(physical_grid, material_derivative_norms, eps=1.0e-6)
    bundle = build_reparameterized_bundle(
        profile,
        effective_nfe=5,
        solver_name="heun2",
        representation="timesteps",
        schedule_family="LCS-1",
        meta={"schedule_implementation_version": LCS_SCHEDULE_IMPLEMENTATION_VERSION},
    )
    assert bundle.timesteps is not None
    assert bundle.time_grid is not None
    assert bundle.tau_grid is not None
    assert bundle.g_grid is not None
    assert len(bundle.timesteps) == 3
    assert len(bundle.time_grid) == 4
    assert len(bundle.tau_grid) == 4
    assert len(bundle.g_grid) == 4
    assert bundle.meta["schedule_implementation_version"] == LCS_SCHEDULE_IMPLEMENTATION_VERSION


def test_sigma_bundle_can_materialize_from_sigma_nodes_with_separate_time_grid() -> None:
    physical_grid = np.linspace(2.0, 0.0, 9, dtype=np.float64)
    material_derivative_norms = np.tile(np.linspace(3.0, 1.0, 9, dtype=np.float64), (2, 1))
    profile = build_shared_clock_profile(physical_grid, material_derivative_norms, eps=1.0e-6)
    plan = resolve_effective_nfe_plan("heun2", 5)
    bundle = build_reparameterized_bundle(
        profile,
        effective_nfe=5,
        solver_name="heun2",
        representation="sigmas",
        schedule_family="LCS-1",
        time_transform=lambda values: values * 100.0,
    )

    assert bundle.sigmas is not None
    assert bundle.sigma_grid is not None
    assert bundle.timesteps is not None
    assert bundle.time_grid is not None
    assert len(bundle.sigmas) == plan.solver_steps
    assert len(bundle.sigma_grid) == plan.solver_steps + 1
    assert len(bundle.timesteps) == plan.solver_steps
    assert len(bundle.time_grid) == plan.solver_steps + 1
    assert np.all(np.diff(bundle.sigma_grid) < 0.0)
    assert np.all(np.diff(bundle.time_grid) < 0.0)


def test_slice_profile_interval_restricts_support_and_renormalizes() -> None:
    physical_grid = np.linspace(10.0, 0.0, 11, dtype=np.float64)
    material_derivative_norms = np.tile(np.linspace(5.0, 1.0, 11, dtype=np.float64), (2, 1))
    profile = build_shared_clock_profile(physical_grid, material_derivative_norms, eps=1.0e-6)
    sliced = slice_profile_interval(profile, 8.0, 2.0)

    assert np.isclose(sliced.physical_grid[0], 8.0)
    assert np.isclose(sliced.physical_grid[-1], 2.0)
    assert np.all(np.diff(sliced.physical_grid) < 0.0)
    assert np.all(sliced.density >= 0.0)
    assert np.all(np.diff(sliced.tau_profile) > 0.0)
    assert np.isclose(sliced.tau_profile[0], 0.0)
    assert np.isclose(sliced.tau_profile[-1], 1.0)
