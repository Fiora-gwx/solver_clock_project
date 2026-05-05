import numpy as np
import torch

from src.clock.ays import AysConfig, AysEarlyStopConfig, hierarchical_optimize_schedule


class ZeroModel(torch.nn.Module):
    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return torch.zeros_like(x)


def test_hierarchical_optimizer_reports_completed_stage_results() -> None:
    stages: list[tuple[int, np.ndarray]] = []

    def batch_provider(batch_size: int) -> torch.Tensor:
        return torch.zeros((batch_size, 1, 2, 2), dtype=torch.float32)

    result = hierarchical_optimize_schedule(
        model=ZeroModel(),
        num_train_timesteps=8,
        alphas_cumprod=torch.linspace(0.95, 0.05, 8),
        sigma_lookup=np.linspace(0.1, 2.0, 8),
        batch_provider=batch_provider,
        config=AysConfig(
            candidate_count=3,
            data_samples=2,
            batch_size=2,
            initial_steps=2,
            subdivision_rounds=1,
            max_iterations_initial=1,
            max_iterations_subdivision=1,
            early_stop=AysEarlyStopConfig(metric="none"),
        ),
        device=torch.device("cpu"),
        stage_result_callback=lambda stage_nfe, stage_result: stages.append(
            (stage_nfe, stage_result.schedule.copy())
        ),
    )

    assert [stage_nfe for stage_nfe, _ in stages] == [2, 4]
    assert set(result.stage_results) == {2, 4}
    assert np.array_equal(stages[0][1], result.stage_results[2].schedule)
    assert np.array_equal(stages[1][1], result.stage_results[4].schedule)
