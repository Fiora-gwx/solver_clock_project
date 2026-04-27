from __future__ import annotations

import sys

import torch

from src.utils.config import repo_root


ROOT = repo_root()
for path in (ROOT / "third_party" / "diffusers" / "src", ROOT / "third_party" / "STORK"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from STORKScheduler import STORKScheduler  # type: ignore  # noqa: E402


def _nonuniform_derivatives(f_k, f_km1, f_km2, h1, h2):
    first = (
        -f_k * (2 * h1 + h2) / (h1 * (h1 + h2))
        + f_km1 * (h1 + h2) / (h1 * h2)
        - f_km2 * h1 / (h2 * (h1 + h2))
    )
    second = 2 / (h1 * h2 * (h1 + h2)) * (f_km2 * h1 - f_km1 * (h1 + h2) + f_k * h2)
    return first, second


def test_nonuniform_first_derivative_reduces_to_uniform_stork_formula():
    h = torch.tensor(0.3)
    f_k = torch.randn(4, 3)
    f_km1 = torch.randn(4, 3)
    f_km2 = torch.randn(4, 3)

    derivative, _ = _nonuniform_derivatives(f_k, f_km1, f_km2, h, h)
    old_uniform = (-f_km2 + 4 * f_km1 - 3 * f_k) / (2 * h)

    assert torch.allclose(derivative, old_uniform, atol=1e-6, rtol=1e-5)


def test_nonuniform_derivatives_are_exact_for_quadratic():
    for t_k, t_km1, t_km2 in ((0.4, 0.7, 1.0), (0.4, 0.65, 1.0)):
        h1 = torch.tensor(t_km1 - t_k, dtype=torch.float64)
        h2 = torch.tensor(t_km2 - t_km1, dtype=torch.float64)
        f_k = torch.tensor(t_k * t_k, dtype=torch.float64)
        f_km1 = torch.tensor(t_km1 * t_km1, dtype=torch.float64)
        f_km2 = torch.tensor(t_km2 * t_km2, dtype=torch.float64)

        first, second = _nonuniform_derivatives(f_k, f_km1, f_km2, h1, h2)

        assert torch.allclose(first, torch.tensor(0.8, dtype=torch.float64), atol=1e-12, rtol=1e-12)
        assert torch.allclose(second, torch.tensor(2.0, dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_stork_flow_custom_sigmas_keep_nonuniform_dt_without_duplicate_zero():
    scheduler = STORKScheduler(
        prediction_type="flow_prediction",
        solver_order=4,
        derivative_order=2,
        shift=7.0,
        use_karras_sigmas=True,
    )

    scheduler.set_timesteps(num_inference_steps=5, device="cpu", sigmas=[1.0, 0.7, 0.55, 0.2, 0.0])

    assert torch.allclose(scheduler.sigmas, torch.tensor([1.0, 0.7, 0.55, 0.2, 0.0]), atol=1e-6)
    assert torch.allclose(scheduler.dt_list, torch.tensor([0.3, 0.15, 0.35, 0.2]), atol=1e-6)
    assert scheduler.num_inference_steps == 4
