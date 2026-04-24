import numpy as np

from src.clock.profile import (
    build_clock_profile_from_alpha,
    build_reparameterized_bundle,
    slice_profile_interval,
)
from src.utils.nfe_budget import resolve_effective_nfe_plan


def test_clock_profile_is_normalized_and_monotone() -> None:
    physical_grid = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    alpha_profile = np.asarray([2.0, 1.5, 1.0, 0.75, 0.5], dtype=np.float64)
    profile = build_clock_profile_from_alpha(physical_grid, alpha_profile)

    assert np.all(profile.density >= 0.0)
    assert np.isclose(np.trapz(profile.density, np.linspace(0.0, 1.0, 5)), 1.0)
    assert np.isclose(profile.tau_profile[0], 0.0)
    assert np.isclose(profile.tau_profile[-1], 1.0)
    assert np.all(np.diff(profile.tau_profile) > 0.0)


def test_materialized_bundle_matches_solver_steps() -> None:
    physical_grid = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    profile = build_clock_profile_from_alpha(physical_grid, np.linspace(4.0, 1.0, 9, dtype=np.float64))
    bundle = build_reparameterized_bundle(
        profile,
        effective_nfe=5,
        solver_name="heun2",
        representation="timesteps",
        schedule_family="SADB",
    )
    assert bundle.timesteps is not None
    assert bundle.time_grid is not None
    assert bundle.tau_grid is not None
    assert bundle.g_grid is not None
    assert len(bundle.timesteps) == 3
    assert len(bundle.time_grid) == 4
    assert len(bundle.tau_grid) == 4
    assert len(bundle.g_grid) == 4


def test_sigma_bundle_can_materialize_from_sigma_nodes_with_separate_time_grid() -> None:
    physical_grid = np.linspace(2.0, 0.0, 9, dtype=np.float64)
    profile = build_clock_profile_from_alpha(physical_grid, np.linspace(3.0, 1.0, 9, dtype=np.float64))
    plan = resolve_effective_nfe_plan("heun2", 5)
    bundle = build_reparameterized_bundle(
        profile,
        effective_nfe=5,
        solver_name="heun2",
        representation="sigmas",
        schedule_family="SADB",
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
    profile = build_clock_profile_from_alpha(physical_grid, np.linspace(5.0, 1.0, 11, dtype=np.float64))
    sliced = slice_profile_interval(profile, 8.0, 2.0)

    assert np.isclose(sliced.physical_grid[0], 8.0)
    assert np.isclose(sliced.physical_grid[-1], 2.0)
    assert np.all(np.diff(sliced.physical_grid) < 0.0)
    assert np.all(sliced.density >= 0.0)
    assert np.all(np.diff(sliced.tau_profile) > 0.0)
    assert np.isclose(sliced.tau_profile[0], 0.0)
    assert np.isclose(sliced.tau_profile[-1], 1.0)
