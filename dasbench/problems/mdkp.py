from __future__ import annotations

from dasbench.problems.base import ProblemDefinition, ScoreResult
from dasbench.problems.packing_utils import (
    FEASIBILITY_TOLERANCE,
    canonicalize_binary_selection,
    greedy_binary_solution,
    redundancy_improved_binary_solution,
    selected_resource_usage,
    selection_objective_value,
    solve_mdkp_cpsat,
    summarize_packing_training_data,
    validate_binary_solution,
    validate_packing_instance,
)


def validate_instance(instance: dict[str, object]) -> None:
    validate_packing_instance(instance, require_integral=True)


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[int]:
    return canonicalize_binary_selection(raw_solution, instance)


def validate_solution(solution: list[int], instance: dict[str, object]) -> tuple[bool, str | None]:
    return validate_binary_solution(solution, instance, tolerance=FEASIBILITY_TOLERANCE)


def score_solution(instance: dict[str, object], solution: list[int]) -> ScoreResult:
    valid, error = validate_solution(solution, instance)
    if not valid:
        return ScoreResult(
            is_valid=False,
            is_feasible=False,
            objective_value=0.0,
            normalized_quality=0.0,
            is_optimal=False,
            error=error,
        )
    objective = selection_objective_value(instance, solution)
    optimum = float(instance["optimum_objective"])
    normalized = 0.0 if optimum <= 0 else objective / optimum
    return ScoreResult(
        is_valid=True,
        is_feasible=True,
        objective_value=objective,
        normalized_quality=min(1.0, normalized),
        is_optimal=abs(objective - optimum) < 1e-9,
    )


def failure_case(
    instance: dict[str, object],
    solution: list[int],
    score: ScoreResult,
    runtime_seconds: float,
) -> dict[str, object]:
    usage = selected_resource_usage(instance, solution) if isinstance(solution, list) else []
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "selected_items": solution[:16],
        "resource_usage": [round(value, 4) for value in usage],
        "capacities": instance.get("capacities", []),
        "error": score.error,
    }


def _lp_relax_rounding(instance: dict[str, object]) -> list[int]:
    from dasbench.problems.packing_utils import solve_packing_lp_glop

    relaxation = solve_packing_lp_glop(instance).solution
    order = sorted(
        range(int(instance["num_items"])),
        key=lambda item: (float(relaxation[item]), float(instance["values"][item])),
        reverse=True,
    )
    selected: list[int] = []
    capacities = [float(value) for value in instance["capacities"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    usage = [0.0] * int(instance["num_resources"])
    for item in order:
        if all(
            usage[resource] + weights[item][resource] <= capacities[resource] + FEASIBILITY_TOLERANCE
            for resource in range(int(instance["num_resources"]))
        ):
            selected.append(item)
            for resource in range(int(instance["num_resources"])):
                usage[resource] += weights[item][resource]
    return sorted(selected)


def baseline_registry() -> dict[str, object]:
    return {
        "value_density_greedy": greedy_binary_solution,
        "redundancy_improved_greedy": redundancy_improved_binary_solution,
        "lp_relax_rounding": _lp_relax_rounding,
        "cpsat_exact": lambda instance: list(solve_mdkp_cpsat(instance).solution),
        "exact": lambda instance: list(solve_mdkp_cpsat(instance).solution),
    }


PROBLEM = ProblemDefinition(
    name="mdkp",
    description="Distribution-aware synthesis benchmark for binary multidimensional knapsack.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is returned_value / optimum_value for feasible binary selections",
    },
    instance_schema_version="mdkp.v1",
    default_instance_params={"num_items": 28, "num_resources": 4},
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_packing_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_mdkp_cpsat,
)
