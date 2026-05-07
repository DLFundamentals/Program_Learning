from __future__ import annotations

import math

from dasbench.problems.base import ProblemDefinition, ScoreResult
from dasbench.problems.packing_utils import (
    FEASIBILITY_TOLERANCE,
    OPTIMALITY_TOLERANCE,
    canonicalize_fractional_solution,
    greedy_fractional_solution,
    objective_value,
    resource_usage,
    solve_packing_lp_glop,
    summarize_packing_training_data,
    validate_fractional_solution,
    validate_packing_instance,
)


def validate_instance(instance: dict[str, object]) -> None:
    validate_packing_instance(instance, require_integral=False)


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[float]:
    return canonicalize_fractional_solution(raw_solution, instance)


def validate_solution(solution: list[float], instance: dict[str, object]) -> tuple[bool, str | None]:
    return validate_fractional_solution(solution, instance, tolerance=FEASIBILITY_TOLERANCE)


def score_solution(instance: dict[str, object], solution: list[float]) -> ScoreResult:
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
    objective = objective_value(instance, solution)
    optimum = float(instance["optimum_objective"])
    normalized = 0.0 if optimum <= 0 else objective / optimum
    gap = abs(objective - optimum)
    tolerance = max(OPTIMALITY_TOLERANCE, OPTIMALITY_TOLERANCE * abs(optimum))
    return ScoreResult(
        is_valid=True,
        is_feasible=True,
        objective_value=objective,
        normalized_quality=min(1.0, normalized),
        is_optimal=gap <= tolerance,
    )


def failure_case(
    instance: dict[str, object],
    solution: list[float],
    score: ScoreResult,
    runtime_seconds: float,
) -> dict[str, object]:
    usage = resource_usage(instance, solution) if len(solution) == int(instance["num_items"]) else []
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "solution_prefix": [round(value, 4) for value in solution[:10]],
        "resource_usage": [round(value, 4) for value in usage],
        "capacities": instance.get("capacities", []),
        "error": score.error,
    }


def _uniform_fraction_solution(instance: dict[str, object]) -> list[float]:
    capacities = [float(value) for value in instance["capacities"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    fractions = []
    for resource, capacity in enumerate(capacities):
        total = sum(row[resource] for row in weights)
        fractions.append(1.0 if math.isclose(total, 0.0) else capacity / total)
    fraction = max(0.0, min(1.0, min(fractions)))
    return [fraction for _ in range(int(instance["num_items"]))]


def baseline_registry() -> dict[str, object]:
    return {
        "uniform_fraction": _uniform_fraction_solution,
        "density_fractional": greedy_fractional_solution,
        "glop_simplex_exact": lambda instance: list(solve_packing_lp_glop(instance).solution),
        "exact": lambda instance: list(solve_packing_lp_glop(instance).solution),
    }


PROBLEM = ProblemDefinition(
    name="packing_lp",
    description="Distribution-aware synthesis benchmark for continuous bounded multidimensional packing LPs.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is returned_objective / optimum_objective under capacity feasibility tolerance 1e-6",
    },
    instance_schema_version="packing_lp.v1",
    default_instance_params={"num_items": 28, "num_resources": 4},
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_packing_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_packing_lp_glop,
)
