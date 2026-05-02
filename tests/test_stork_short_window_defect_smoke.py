import numpy as np
import pytest

from src.adapters.pndm import build_scheduler
from src.clock.ri_sadb import distribute_short_window_arc_defect


def test_short_window_stork_defect_distribution_smoke() -> None:
    try:
        build_scheduler("stork4_1st")
    except Exception as exc:  # pragma: no cover - depends on optional STORK import path
        pytest.skip(f"STORK solver not available: {exc}")

    delta_s = np.asarray(
        [
            [0.10, 0.20, 0.15, 0.18, 0.12],
            [0.08, 0.16, 0.22, 0.14, 0.10],
        ],
        dtype=np.float64,
    )
    residual = np.asarray(
        [
            [0.01, 0.02, 0.03, 0.04, 0.05],
            [0.02, 0.03, 0.01, 0.05, 0.04],
        ],
        dtype=np.float64,
    )

    stats = distribute_short_window_arc_defect(
        delta_s=delta_s,
        window_residual=residual,
        window_len=4,
        refine_factor=2,
        q_prior=3.0,
        defect_source="target_stork_short_window",
    )
    metadata = {
        "defect_source": stats.defect_source,
        "window_len": stats.window_len,
        "history_len": 4,
    }

    assert stats.interval_arc_defect.shape == delta_s.shape
    assert np.all(np.isfinite(stats.interval_arc_defect))
    assert np.all(stats.interval_arc_defect > 0.0)
    assert metadata["defect_source"] == "target_stork_short_window"
    assert metadata["window_len"] >= metadata["history_len"] or metadata["window_len"] == 4


def test_fixed_euler_proxy_is_not_labeled_as_target_stork() -> None:
    stats = distribute_short_window_arc_defect(
        delta_s=np.ones((1, 4), dtype=np.float64),
        window_residual=np.full((1, 4), 0.01, dtype=np.float64),
        defect_source="fixed_euler_proxy",
    )
    assert stats.defect_source != "target_stork_short_window"
