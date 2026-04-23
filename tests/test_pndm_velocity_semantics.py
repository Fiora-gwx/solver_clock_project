import numpy as np
import torch

from src.adapters.pndm import _beta_at_timestep_torch, _evaluate_velocity, build_scheduler


class ConstantOutputModel(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = float(value)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return torch.full_like(x, self.value)


def _beta_at(scheduler, timestep_value: float) -> float:
    betas = scheduler.betas.detach().cpu().numpy().astype(np.float64)
    return float(np.interp(timestep_value, np.arange(len(betas), dtype=np.float64), betas))


def test_epsilon_model_output_converts_to_pf_ode_velocity() -> None:
    scheduler = build_scheduler("euler", diffusion_step=10, beta_start=0.1, beta_end=0.2)
    sample = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    velocity = _evaluate_velocity(
        ConstantOutputModel(1.5),
        scheduler,
        sample,
        timestep_value=3.25,
        sigma_value=0.5,
        model_output_type="epsilon",
        sigma_floor=1.0e-6,
    )
    expected = 0.5 * _beta_at(scheduler, 3.25) * (1.5 / 0.5 - 2.0)
    assert torch.allclose(velocity, torch.full_like(sample, expected), atol=1.0e-6)


def test_v_prediction_model_output_converts_via_epsilon() -> None:
    scheduler = build_scheduler("euler", diffusion_step=10, beta_start=0.1, beta_end=0.2)
    sample = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    sigma = 0.5
    desired_epsilon = 1.2
    alpha_t = 1.0 / np.sqrt(1.0 + sigma**2)
    sigma_t = sigma * alpha_t
    x_t = sample.item() / np.sqrt(1.0 + sigma**2)
    raw_v = (desired_epsilon - sigma_t * x_t) / alpha_t
    velocity = _evaluate_velocity(
        ConstantOutputModel(raw_v),
        scheduler,
        sample,
        timestep_value=4.5,
        sigma_value=sigma,
        model_output_type="v_prediction",
        sigma_floor=1.0e-6,
    )
    expected = 0.5 * _beta_at(scheduler, 4.5) * (desired_epsilon / sigma - sample.item())
    assert torch.allclose(velocity, torch.full_like(sample, expected), atol=1.0e-6)


def test_sigma_floor_prevents_division_by_zero_at_terminal_sigma() -> None:
    scheduler = build_scheduler("euler", diffusion_step=10, beta_start=0.1, beta_end=0.2)
    sample = torch.tensor([[[[0.25]]]], dtype=torch.float32)
    velocity = _evaluate_velocity(
        ConstantOutputModel(0.5),
        scheduler,
        sample,
        timestep_value=0.0,
        sigma_value=0.0,
        model_output_type="epsilon",
        sigma_floor=0.2,
    )
    expected = 0.5 * _beta_at(scheduler, 0.0) * (0.5 / 0.2 - sample.item())
    assert torch.isfinite(velocity).all()
    assert torch.allclose(velocity, torch.full_like(sample, expected), atol=1.0e-6)


def test_flow_model_output_is_used_directly() -> None:
    scheduler = build_scheduler("euler", diffusion_step=10, beta_start=0.1, beta_end=0.2)
    sample = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    velocity = _evaluate_velocity(
        ConstantOutputModel(3.0),
        scheduler,
        sample,
        timestep_value=2.0,
        sigma_value=0.75,
        model_output_type="flow",
        sigma_floor=1.0e-6,
    )
    assert torch.allclose(velocity, torch.full_like(sample, 3.0), atol=1.0e-6)


def test_beta_interpolation_is_torch_native_and_differentiable() -> None:
    scheduler = build_scheduler("euler", diffusion_step=10, beta_start=0.1, beta_end=0.2)
    timestep = torch.tensor(3.25, dtype=torch.float32, requires_grad=True)
    beta_value = _beta_at_timestep_torch(
        scheduler,
        timestep,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    beta_value.backward()
    assert timestep.grad is not None
    assert torch.isfinite(timestep.grad)
