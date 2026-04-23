from src.utils.nfe_budget import resolve_effective_nfe_plan


def test_one_eval_solver_uses_identity_budget() -> None:
    plan = resolve_effective_nfe_plan("euler", 12)
    assert plan.effective_nfe == 12
    assert plan.solver_steps == 12
    assert len(plan.step_methods) == 12
    assert all(method == "euler" for method in plan.step_methods)


def test_heun_odd_budget_matches_vendor_forward_count() -> None:
    plan = resolve_effective_nfe_plan("heun2", 11)
    assert plan.solver_steps == 6
    assert len(plan.step_methods) == 6
    assert plan.step_methods[:-1] == ("heun2",) * 5
    assert plan.step_methods[-1] == "euler_tail"
    assert plan.execution_backend == "native_scheduler"


def test_heun_even_budget_is_rejected() -> None:
    try:
        resolve_effective_nfe_plan("heun2", 12)
    except ValueError as error:
        assert "odd effective_nfe" in str(error)
    else:
        raise AssertionError("Even vendor heun budget should be rejected.")
