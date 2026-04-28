from __future__ import annotations

import sys

import torch

from src.utils.config import repo_root


ROOT = repo_root()
for path in (ROOT / "third_party" / "diffusers" / "src", ROOT / "third_party" / "STORK"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from STORKScheduler import STORKScheduler  # type: ignore  # noqa: E402


def _scheduler() -> STORKScheduler:
    return STORKScheduler(prediction_type="flow_prediction", solver_order=4, derivative_order=2)


def test_nonuniform_first_derivative_reduces_to_uniform_stork_formula():
    h = torch.tensor(0.3)
    f_k = torch.randn(4, 3)
    f_km1 = torch.randn(4, 3)
    f_km2 = torch.randn(4, 3)

    derivative, _ = _scheduler()._nonuniform_first_second_derivatives(f_k, f_km1, f_km2, h, h)
    old_uniform = (-f_km2 + 4 * f_km1 - 3 * f_k) / (2 * h)

    assert torch.allclose(derivative, old_uniform, atol=1e-6, rtol=1e-5)


def test_scheduler_nonuniform_derivatives_are_exact_for_polynomials():
    scheduler = _scheduler()
    for power, expected_first, expected_second in ((0, 0.0, 0.0), (1, 1.0, 0.0), (2, 0.8, 2.0)):
        for t_k, t_km1, t_km2 in ((0.4, 0.7, 1.0), (0.4, 0.65, 1.0)):
            h1 = torch.tensor(t_km1 - t_k, dtype=torch.float64)
            h2 = torch.tensor(t_km2 - t_km1, dtype=torch.float64)
            f_k = torch.tensor(t_k**power, dtype=torch.float64)
            f_km1 = torch.tensor(t_km1**power, dtype=torch.float64)
            f_km2 = torch.tensor(t_km2**power, dtype=torch.float64)

            first, second = scheduler._nonuniform_first_second_derivatives(f_k, f_km1, f_km2, h1, h2)

            assert torch.allclose(first, torch.tensor(expected_first, dtype=torch.float64), atol=1e-12, rtol=1e-12)
            assert torch.allclose(second, torch.tensor(expected_second, dtype=torch.float64), atol=1e-12, rtol=1e-12)


def test_scheduler_variable_step_ab2_integrates_linear_function_exactly():
    scheduler = _scheduler()
    t_current = 0.4
    h_previous = 0.2
    h_current = 0.5
    a = 1.25
    b = -0.75
    f_current = torch.tensor(a + b * t_current, dtype=torch.float64)
    f_previous = torch.tensor(a + b * (t_current + h_previous), dtype=torch.float64)
    sample = torch.tensor(3.0, dtype=torch.float64)

    actual = scheduler._variable_step_ab2_update(sample, f_current, f_previous, h_current, h_previous)
    expected_integral = a * h_current + 0.5 * b * (t_current**2 - (t_current - h_current) ** 2)
    expected = sample - expected_integral

    assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_scheduler_variable_step_ab2_reduces_to_uniform_coefficients():
    scheduler = _scheduler()
    h = torch.tensor(0.3)
    sample = torch.randn(2, 3)
    f_current = torch.randn(2, 3)
    f_previous = torch.randn(2, 3)

    actual = scheduler._variable_step_ab2_update(sample, f_current, f_previous, h, h)
    expected = sample - 1.5 * h * f_current + 0.5 * h * f_previous

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_noise_stork4_2nd_calls_nonuniform_derivative_helper():
    scheduler = STORKScheduler(prediction_type="epsilon", solver_order=4, derivative_order=2)
    scheduler.set_timesteps(
        num_inference_steps=4,
        device="cpu",
        timesteps=[900.0, 400.0, 100.0, 10.0],
        sigmas=[80.0, 10.0, 1.0, 0.1],
    )

    captured = {}
    original = scheduler._nonuniform_first_second_derivatives

    def recording_derivatives(f_k, f_km1, f_km2, h1, h2):
        first, second = original(f_k, f_km1, f_km2, h1, h2)
        captured["h1"] = float(h1)
        captured["h2"] = float(h2)
        captured["first"] = first.detach().clone()
        captured["second"] = second.detach().clone()
        return first, second

    scheduler._nonuniform_first_second_derivatives = recording_derivatives
    sample = torch.zeros(1, 1, 1, 1)
    for timestep in scheduler.timesteps[:3]:
        normalized_t = (timestep / scheduler.config.num_train_timesteps).reshape(1, 1, 1, 1)
        model_output = normalized_t.square()
        sample = scheduler.step(model_output, timestep, sample).prev_sample

    assert abs(captured["h1"] - 0.3) < 1.0e-6
    assert abs(captured["h2"] - 0.5) < 1.0e-6
    assert torch.allclose(captured["first"], torch.full((1, 1, 1, 1), 0.2), atol=1e-5, rtol=1e-5)
    assert torch.allclose(captured["second"], torch.full((1, 1, 1, 1), 2.0), atol=1e-5, rtol=1e-5)


def test_noise_stork4_1st_uses_previous_timestep_gap_for_derivative():
    scheduler = STORKScheduler(prediction_type="epsilon", solver_order=4, derivative_order=1)
    scheduler.set_timesteps(
        num_inference_steps=3,
        device="cpu",
        timesteps=[900.0, 400.0, 100.0],
        sigmas=[80.0, 10.0, 1.0],
    )

    captured = {}
    original = scheduler._nonuniform_first_derivative

    def recording_derivative(f_k, f_km1, h1):
        derivative = original(f_k, f_km1, h1)
        captured["h1"] = float(h1)
        captured["derivative"] = derivative.detach().clone()
        return derivative

    scheduler._nonuniform_first_derivative = recording_derivative
    sample = torch.zeros(1, 1, 1, 1)
    for timestep in scheduler.timesteps[:2]:
        normalized_t = (timestep / scheduler.config.num_train_timesteps).reshape(1, 1, 1, 1)
        model_output = normalized_t
        sample = scheduler.step(model_output, timestep, sample).prev_sample

    assert abs(captured["h1"] - 0.5) < 1.0e-6
    assert torch.allclose(captured["derivative"], torch.ones(1, 1, 1, 1), atol=1e-6, rtol=1e-6)


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
