import numpy as np
import torch

from src.adapters.pndm import (
    _configure_scheduler_timesteps,
    _resolve_custom_heun_grid,
    build_pndm_native_coordinate_grid,
    build_scheduler,
    collect_solver_refinement_stats,
    preferred_calibration_domain,
    preferred_schedule_representation,
)
from src.utils.schedule_bundle import ScheduleBundle


def test_noise_stork_accepts_custom_schedule_bundle() -> None:
    scheduler = build_scheduler("stork4_1st")
    bundle = ScheduleBundle(timesteps=np.asarray([999.0, 500.0, 10.0], dtype=np.float64))
    _configure_scheduler_timesteps(
        scheduler,
        num_inference_steps=3,
        device=torch.device("cpu"),
        schedule_bundle=bundle,
    )

    assert np.allclose(scheduler.timesteps.detach().cpu().numpy(), np.asarray([999.0, 500.0, 10.0], dtype=np.float32))
    assert np.isclose(float(scheduler.sigmas[-1].item()), 0.0)
    assert np.allclose(
        scheduler.dt_list.detach().cpu().numpy(),
        np.asarray([0.499, 0.49], dtype=np.float32),
        atol=1.0e-6,
    )


def test_noise_stork_schedule_bundle_preserves_fractional_timesteps() -> None:
    scheduler = build_scheduler("stork4_2nd")
    timesteps = np.asarray([933.3333, 466.6667, 10.0], dtype=np.float64)
    sigmas = np.asarray([82.85, 2.88, 0.046], dtype=np.float64)
    bundle = ScheduleBundle(timesteps=timesteps, sigmas=sigmas)
    _configure_scheduler_timesteps(
        scheduler,
        num_inference_steps=3,
        device=torch.device("cpu"),
        schedule_bundle=bundle,
    )

    assert np.allclose(scheduler.timesteps.detach().cpu().numpy(), timesteps.astype(np.float32))
    assert np.allclose(scheduler.sigmas[:-1].detach().cpu().numpy(), sigmas.astype(np.float32))
    assert np.allclose(
        scheduler.dt_list.detach().cpu().numpy(),
        np.asarray([0.4666666, 0.4566667], dtype=np.float32),
        atol=1.0e-6,
    )


def test_noise_stork_set_timesteps_keeps_custom_sigmas_and_dt_list() -> None:
    scheduler = build_scheduler("stork4_2nd")
    scheduler.set_timesteps(
        num_inference_steps=3,
        device=torch.device("cpu"),
        timesteps=[999.0, 500.0, 10.0],
        sigmas=[4.0, 1.5, 0.1],
    )

    assert np.allclose(scheduler.timesteps.detach().cpu().numpy(), np.asarray([999.0, 500.0, 10.0], dtype=np.float32))
    assert np.allclose(scheduler.sigmas[:-1].detach().cpu().numpy(), np.asarray([4.0, 1.5, 0.1], dtype=np.float32))
    assert np.isclose(float(scheduler.sigmas[-1].item()), 0.0)
    assert np.allclose(
        scheduler.dt_list.detach().cpu().numpy(),
        np.asarray([0.499, 0.49], dtype=np.float32),
        atol=1.0e-6,
    )


def test_dpm_solver_variants_reject_custom_offline_schedules() -> None:
    for solver_name in ("dpm_solver_lu", "dpm_solver_default"):
        scheduler = build_scheduler(solver_name)
        scheduler.set_timesteps(4, device=torch.device("cpu"))

        bundle = ScheduleBundle(timesteps=np.asarray([999.0, 700.0, 200.0], dtype=np.float64))
        try:
            _configure_scheduler_timesteps(
                scheduler,
                num_inference_steps=3,
                device=torch.device("cpu"),
                schedule_bundle=bundle,
            )
        except ValueError as error:
            assert "DPMSolver" in str(error)
        else:
            raise AssertionError("DPMSolver custom offline schedules should be disabled.")


def test_sigma_native_solvers_choose_expected_domains() -> None:
    assert preferred_schedule_representation("heun2") == "sigmas"
    assert preferred_schedule_representation("stork4_1st") == "sigmas"
    assert preferred_schedule_representation("dpm_solver_lu") == "timesteps"
    assert preferred_schedule_representation("euler") == "timesteps"
    assert preferred_calibration_domain("heun2") == "sigmas"
    assert preferred_calibration_domain("dpm_solver_lu") == "timesteps"
    assert preferred_calibration_domain("dpm_solver_default") == "timesteps"
    assert preferred_calibration_domain("euler") == "timesteps"


def test_native_sigma_coordinate_grid_matches_vendor_stork_range() -> None:
    scheduler = build_scheduler("stork4_2nd")
    sigma_grid = build_pndm_native_coordinate_grid(
        scheduler,
        solver_name="stork4_2nd",
        effective_nfe=15,
        coordinate_domain="sigmas",
    )

    assert len(sigma_grid) == 16
    assert np.isclose(float(sigma_grid[0]), 82.85083770751953)
    assert np.isclose(float(sigma_grid[-1]), 0.0)
    assert np.all(np.diff(sigma_grid) <= 0.0)


def test_native_lambda_coordinate_grid_is_disabled() -> None:
    scheduler = build_scheduler("dpm_solver_lu")
    try:
        build_pndm_native_coordinate_grid(
            scheduler,
            solver_name="dpm_solver_lu",
            effective_nfe=15,
            coordinate_domain="lambda",
        )
    except ValueError as error:
        assert "Unsupported PNDM coordinate domain" in str(error)
    else:
        raise AssertionError("lambda-domain coordinate grids should be disabled.")


def test_dpm_solver_custom_sigma_grid_is_rejected() -> None:
    scheduler = build_scheduler("dpm_solver_lu")
    scheduler.set_timesteps(3, device=torch.device("cpu"))
    default_sigmas = scheduler.sigmas.detach().cpu().numpy().astype(np.float64)
    custom_sigma_grid = default_sigmas.copy()
    custom_sigma_grid[1] = 0.5 * (default_sigmas[0] + default_sigmas[1])
    custom_sigma_grid[2] = 0.5 * (default_sigmas[1] + default_sigmas[2])

    bundle = ScheduleBundle(
        sigmas=custom_sigma_grid[:-1],
        sigma_grid=custom_sigma_grid,
    )
    try:
        _configure_scheduler_timesteps(
            scheduler,
            num_inference_steps=3,
            device=torch.device("cpu"),
            schedule_bundle=bundle,
        )
    except ValueError as error:
        assert "DPMSolver" in str(error)
    else:
        raise AssertionError("DPMSolver custom sigma-grid injection should be disabled.")


def test_heun_resolves_custom_sigma_grid_directly() -> None:
    scheduler = build_scheduler("heun2")
    bundle = ScheduleBundle(
        sigmas=np.asarray([5.0, 2.0, 0.5], dtype=np.float64),
        sigma_grid=np.asarray([5.0, 2.0, 0.5, 0.0], dtype=np.float64),
    )
    anchor_timesteps, time_grid, sigma_grid, step_methods = _resolve_custom_heun_grid(
        scheduler,
        effective_nfe=5,
        schedule_bundle=bundle,
        device=torch.device("cpu"),
    )

    assert np.allclose(sigma_grid, np.asarray([5.0, 2.0, 0.5, 0.0], dtype=np.float64))
    assert len(anchor_timesteps) == 3
    assert len(time_grid) == 4
    assert np.all(np.diff(time_grid) <= 0.0)
    assert step_methods == ("heun2", "heun2", "euler_tail")


def test_heun_prefers_explicit_time_grid_when_bundle_contains_both_time_and_sigma() -> None:
    scheduler = build_scheduler("heun2")
    bundle = ScheduleBundle(
        sigmas=np.asarray([50.0, 25.0, 10.0], dtype=np.float64),
        sigma_grid=np.asarray([50.0, 25.0, 10.0, 0.0], dtype=np.float64),
        timesteps=np.asarray([999.0, 500.0, 100.0], dtype=np.float64),
        time_grid=np.asarray([999.0, 500.0, 100.0, 0.0], dtype=np.float64),
    )
    anchor_timesteps, time_grid, sigma_grid, _ = _resolve_custom_heun_grid(
        scheduler,
        effective_nfe=5,
        schedule_bundle=bundle,
        device=torch.device("cpu"),
    )

    expected_sigma_grid = np.asarray(scheduler.alphas_cumprod.detach().cpu().float().numpy(), dtype=np.float64)
    expected_sigma_grid = np.sqrt(np.maximum(1.0 - expected_sigma_grid, 0.0) / np.maximum(expected_sigma_grid, 1.0e-12))
    expected_sigma_grid = np.interp(
        np.asarray([999.0, 500.0, 100.0, 0.0], dtype=np.float64),
        np.arange(len(expected_sigma_grid), dtype=np.float64),
        expected_sigma_grid,
    )
    expected_sigma_grid[-1] = 0.0

    assert np.allclose(anchor_timesteps, np.asarray([999.0, 500.0, 100.0], dtype=np.float64))
    assert np.allclose(time_grid, np.asarray([999.0, 500.0, 100.0, 0.0], dtype=np.float64))
    assert np.allclose(sigma_grid, expected_sigma_grid)


def test_schedule_bundle_save_clears_stale_arrays(tmp_path) -> None:
    ScheduleBundle(
        timesteps=np.asarray([999.0, 500.0, 100.0], dtype=np.float64),
        time_grid=np.asarray([999.0, 500.0, 100.0, 0.0], dtype=np.float64),
    ).save(tmp_path)

    ScheduleBundle(
        sigmas=np.asarray([5.0, 2.0, 0.5], dtype=np.float64),
        sigma_grid=np.asarray([5.0, 2.0, 0.5, 0.0], dtype=np.float64),
    ).save(tmp_path)

    assert not (tmp_path / "timesteps.npy").exists()
    assert not (tmp_path / "time_grid.npy").exists()
    loaded = ScheduleBundle.load(tmp_path)
    assert loaded.timesteps is None
    assert loaded.time_grid is None
    assert loaded.sigmas is not None
    assert loaded.sigma_grid is not None


def test_legacy_dpm_solver_name_raises_migration_error() -> None:
    try:
        build_scheduler("dpm_solver")
    except ValueError as error:
        assert "dpm_solver_lu" in str(error)
        assert "dpm_solver_default" in str(error)
    else:
        raise AssertionError("Legacy dpm_solver alias should require explicit migration.")


def test_collect_solver_refinement_stats_accepts_stork_sigma_domain() -> None:
    class DummyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_channels = 1
            self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))

        def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(x)

    model = DummyModel().to("cpu")
    scheduler = build_scheduler("stork4_2nd")
    stats = collect_solver_refinement_stats(
        model=model,
        scheduler=scheduler,
        physical_grid=np.asarray([10.0, 5.0, 0.0], dtype=np.float64),
        solver="stork4_2nd",
        image_size=4,
        batch_size=2,
        num_batches=1,
        seed=0,
        coordinate_domain="sigmas",
    )

    assert stats.full_step_error.shape == (2, 2)
    assert stats.half_step_error.shape == (2, 2)
    assert stats.effective_order.shape == (2, 2)
    assert stats.defect_strength.shape == (2, 2)
    assert np.all(np.isfinite(stats.defect_strength))
