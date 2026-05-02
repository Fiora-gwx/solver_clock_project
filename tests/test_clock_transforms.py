import numpy as np
import torch

from src.clock.transforms import build_lambda_table, lambda_to_sigma, lambda_to_timestep, sigma_to_lambda


class DummyScheduler:
    def __init__(self) -> None:
        self.alphas_cumprod = torch.linspace(0.999, 0.001, 1000)


def test_build_lambda_table_is_strictly_monotone_for_vp_scheduler() -> None:
    timesteps, sigmas, lambdas = build_lambda_table(DummyScheduler())

    assert timesteps.shape == sigmas.shape == lambdas.shape
    assert np.all(np.diff(sigmas) > 0.0)
    assert np.all(np.diff(lambdas) < 0.0)


def test_lambda_sigma_timestep_roundtrip_interpolation() -> None:
    timesteps, sigmas, lambdas = build_lambda_table(DummyScheduler())
    selected_sigmas = np.asarray([sigmas[999], sigmas[500], sigmas[0], 0.0], dtype=np.float64)
    selected_lambdas = sigma_to_lambda(selected_sigmas, sigmas, lambdas)
    recovered_sigmas = lambda_to_sigma(selected_lambdas, lambdas, sigmas)
    recovered_timesteps = lambda_to_timestep(selected_lambdas, lambdas, timesteps)

    assert np.all(np.isfinite(selected_lambdas))
    assert np.allclose(recovered_sigmas[:-1], selected_sigmas[:-1])
    assert recovered_sigmas[-1] == sigmas[0]
    assert np.all(np.diff(recovered_timesteps[:-1]) < 0.0)
    assert recovered_timesteps[-1] == 0.0
