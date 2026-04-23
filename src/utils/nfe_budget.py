from __future__ import annotations

from dataclasses import dataclass


ONE_EVAL_SOLVERS = {
    "euler",
    "deis",
    "dpm_solver_lu",
    "dpm_solver_default",
    "dpm_solver_pp",
    "dpm_solverpp",
    "unipc",
    "stork4_1st",
    "stork4_2nd",
    "stork4_3rd",
    "stork_4_1st",
    "stork_4_2nd",
    "stork_4_3rd",
    "flow_euler",
    "flow_dpm_solver",
    "flow_unipc",
    "flow_stork4_1st",
    "flow_stork4_2nd",
    "flow_stork4_3rd",
    "flow_stork_4_1st",
    "flow_stork_4_2nd",
    "flow_stork_4_3rd",
    "ddim",
    "pndm",
}

HEUN_SOLVERS = {
    "heun2",
    "flow_heun",
}


def normalize_solver_name(name: str) -> str:
    return name.lower().replace("-", "_").replace("+", "p")


@dataclass(frozen=True)
class EffectiveNfePlan:
    solver_name: str
    effective_nfe: int
    solver_steps: int
    step_methods: tuple[str, ...]
    execution_backend: str

    def to_meta(self) -> dict[str, object]:
        return {
            "effective_nfe": self.effective_nfe,
            "solver_steps": self.solver_steps,
            "step_methods": list(self.step_methods),
            "execution_backend": self.execution_backend,
        }


def resolve_effective_nfe_plan(solver_name: str, effective_nfe: int) -> EffectiveNfePlan:
    normalized = normalize_solver_name(solver_name)
    if effective_nfe <= 0:
        raise ValueError("effective_nfe must be positive.")
    if normalized == "dpm_solver":
        raise ValueError(
            "Legacy solver `dpm_solver` has been removed. Use `dpm_solver_lu` or `dpm_solver_default` explicitly."
        )

    if normalized in HEUN_SOLVERS:
        if effective_nfe % 2 == 0:
            raise ValueError(
                f"Vendor {normalized} execution requires an odd effective_nfe, got {effective_nfe}."
            )
        solver_steps = (effective_nfe + 1) // 2
        step_methods = ("heun2",) * max(solver_steps - 1, 0) + ("euler_tail",)
        execution_backend = "native_scheduler"
        return EffectiveNfePlan(
            solver_name=normalized,
            effective_nfe=effective_nfe,
            solver_steps=solver_steps,
            step_methods=step_methods,
            execution_backend=execution_backend,
        )

    if normalized in ONE_EVAL_SOLVERS:
        return EffectiveNfePlan(
            solver_name=normalized,
            effective_nfe=effective_nfe,
            solver_steps=effective_nfe,
            step_methods=(normalized,) * effective_nfe,
            execution_backend="native_scheduler",
        )

    raise ValueError(f"Unsupported solver for effective NFE resolution: {solver_name}")
