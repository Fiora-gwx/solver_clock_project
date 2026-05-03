import csv

import numpy as np

from src.clock.archive.ri_sadb import TrajectoryGeometryStats, build_ri_sadb_profile


def _synthetic_stats() -> tuple[np.ndarray, TrajectoryGeometryStats]:
    grid = np.linspace(0.0, 1.0, 9, dtype=np.float64)
    base = np.asarray([0.2, 0.3, 0.5, 0.4, 0.25, 0.35, 0.45, 0.3], dtype=np.float64)
    return grid, TrajectoryGeometryStats(
        full_step_error=np.tile(base, (3, 1)),
        half_step_error=np.tile(0.25 * base, (3, 1)),
        effective_order=np.full((3, 8), 3.0, dtype=np.float64),
        delta_s=np.tile(np.linspace(0.1, 0.25, 8, dtype=np.float64), (3, 1)),
        curvature=np.tile(np.asarray([0.0, 0.1, 0.4, 1.0, 0.6, 0.2, 0.1, 0.0]), (3, 1)),
        residual_perp_norm=np.tile(base, (3, 1)),
        residual_parallel_norm=np.tile(0.5 * base, (3, 1)),
    )


def test_eta_sweep_schedule_only_smoke(tmp_path) -> None:
    grid, stats = _synthetic_stats()
    rows = []
    for eta in [0.0, 0.3, 0.5, 0.7, 1.0]:
        artifacts = build_ri_sadb_profile(
            grid,
            stats,
            target_steps=10,
            eta=eta,
            beta=0.0,
            smoothing_window=1,
        )
        profile = artifacts.profile
        assert np.all(np.isfinite(profile.alpha_profile))
        assert np.all(profile.alpha_profile > 0.0)
        assert np.all(np.isfinite(profile.tau_profile))
        assert np.isclose(profile.tau_profile[0], 0.0)
        assert np.isclose(profile.tau_profile[-1], 1.0)
        assert np.all(np.diff(profile.tau_profile) >= -1.0e-12)
        tau_steps = np.diff(profile.tau_profile)
        alpha = profile.alpha_profile
        density = alpha / np.sum(alpha)
        rows.append(
            {
                "eta": eta,
                "alpha_min": float(np.min(alpha)),
                "alpha_max": float(np.max(alpha)),
                "alpha_mean": float(np.mean(alpha)),
                "density_entropy": float(-np.sum(density * np.log(np.maximum(density, 1.0e-300)))),
                "tau_min_step": float(np.min(tau_steps)),
                "tau_max_step": float(np.max(tau_steps)),
                "valid": True,
            }
        )

    diagnostics = tmp_path / "eta_diagnostics.csv"
    with diagnostics.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert diagnostics.exists()
