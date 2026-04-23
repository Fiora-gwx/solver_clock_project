import torch

from src.adapters.pndm import _evaluate_sigma_derivative, build_scheduler


class _ConstantOutputModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.in_channels = 3
        self.register_parameter("_dummy", torch.nn.Parameter(torch.zeros(())))
        self._value = float(value)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, self._value)


def test_heun_sigma_derivative_matches_epsilon_output() -> None:
    model = _ConstantOutputModel(0.0)
    sample = torch.randn(2, 3, 4, 4)
    derivative = _evaluate_sigma_derivative(
        model,
        sample,
        timestep_value=500.0,
        sigma_value=3.0,
        model_output_type="epsilon",
    )
    assert torch.allclose(derivative, torch.zeros_like(sample))

    model = _ConstantOutputModel(1.25)
    derivative = _evaluate_sigma_derivative(
        model,
        sample,
        timestep_value=500.0,
        sigma_value=3.0,
        model_output_type="epsilon",
    )
    assert torch.allclose(derivative, torch.full_like(sample, 1.25))


def test_heun_scheduler_raw_grid_has_one_terminal_sigma_beyond_unique_timesteps() -> None:
    scheduler = build_scheduler("heun2")
    scheduler.set_timesteps(3, device=torch.device("cpu"))

    unique_timesteps = []
    for value in scheduler.timesteps.detach().cpu().tolist():
        if not unique_timesteps or unique_timesteps[-1] != value:
            unique_timesteps.append(value)

    unique_sigmas = []
    for value in scheduler.sigmas.detach().cpu().tolist():
        if not unique_sigmas or unique_sigmas[-1] != value:
            unique_sigmas.append(value)

    assert len(unique_timesteps) == 3
    assert len(unique_sigmas) == 4
    assert unique_timesteps[-1] == 0.0
    assert unique_sigmas[-1] == 0.0
