import numpy as np

from goes.gpde import estimate_global_q, materialize_gpde_schedule, snap_schedule_to_grid


def test_gpde_global_q_estimation_recovers_power_law() -> None:
    probe_steps = np.asarray([0.01, 0.02, 0.04], dtype=np.float64)
    q_true = 4.0
    coefficients = np.asarray([2.0, 3.0, 5.0], dtype=np.float64)
    defects = np.empty((5, probe_steps.size, coefficients.size), dtype=np.float64)
    for step_index, eta in enumerate(probe_steps):
        defects[:, step_index, :] = coefficients * eta**q_true

    q_estimate, source = estimate_global_q(
        defects,
        probe_steps,
        aggregation={"name": "mean"},
        default_q=2.0,
    )

    assert source == "global_loglog_fit"
    assert np.isclose(q_estimate, q_true)


def test_gpde_monitor_inverse_cdf_equalizes_linear_density_mass() -> None:
    grid = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    density = 1.0 + grid

    schedule = materialize_gpde_schedule(grid, density, 4)

    assert schedule.u_schedule.size == 5
    assert np.all(np.diff(schedule.u_schedule) > 0.0)
    assert np.max(schedule.interval_monitor_masses) - np.min(schedule.interval_monitor_masses) < 1.0e-3


def test_gpde_snapping_preserves_strict_monotonicity() -> None:
    schedule = np.asarray([0.0, 0.26, 0.51, 0.76, 1.0], dtype=np.float64)
    admissible = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

    snapped, indices, errors = snap_schedule_to_grid(schedule, admissible)

    assert indices == [0, 1, 2, 3, 4]
    assert np.all(np.diff(snapped) > 0.0)
    assert np.max(np.abs(errors)) <= 0.01 + 1.0e-12
