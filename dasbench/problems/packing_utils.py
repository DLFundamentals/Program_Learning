from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from typing import Any

from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from dasbench.problems.base import ExactSolveResult

FEASIBILITY_TOLERANCE = 1e-6
OPTIMALITY_TOLERANCE = 1e-6


def validate_packing_instance(instance: dict[str, object], *, require_integral: bool = False) -> None:
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    values = instance["values"]
    weights = instance["weights"]
    capacities = instance["capacities"]
    if num_items <= 0:
        raise ValueError("Packing instances require at least one item.")
    if num_resources <= 0:
        raise ValueError("Packing instances require at least one resource.")
    if not isinstance(values, list) or len(values) != num_items:
        raise ValueError("Packing instances require values with length num_items.")
    if not isinstance(weights, list) or len(weights) != num_items:
        raise ValueError("Packing instances require item-major weights with length num_items.")
    if not isinstance(capacities, list) or len(capacities) != num_resources:
        raise ValueError("Packing instances require capacities with length num_resources.")
    _validate_numeric_vector(values, "values", require_integral=require_integral, strictly_positive=True)
    _validate_numeric_vector(capacities, "capacities", require_integral=require_integral, strictly_positive=True)
    for item_index, row in enumerate(weights):
        if not isinstance(row, list) or len(row) != num_resources:
            raise ValueError(f"Item {item_index} weight vector does not match num_resources.")
        _validate_numeric_vector(row, f"weights[{item_index}]", require_integral=require_integral)


def _validate_numeric_vector(
    values: Sequence[object],
    name: str,
    *,
    require_integral: bool,
    strictly_positive: bool = False,
) -> None:
    for index, raw_value in enumerate(values):
        value = float(raw_value)
        if strictly_positive and value <= 0:
            raise ValueError(f"{name}[{index}] must be positive.")
        if not strictly_positive and value < 0:
            raise ValueError(f"{name}[{index}] must be nonnegative.")
        if require_integral and abs(value - round(value)) > 1e-9:
            raise ValueError(f"{name}[{index}] must be integral for MDKP.")


def canonicalize_fractional_solution(raw_solution: Any, instance: dict[str, object]) -> list[float]:
    if not isinstance(raw_solution, Sequence) or isinstance(raw_solution, (str, bytes)):
        raise TypeError("Packing LP solver output must be a sequence of floats.")
    return [float(value) for value in raw_solution]


def canonicalize_binary_selection(raw_solution: Any, instance: dict[str, object]) -> list[int]:
    num_items = int(instance["num_items"])
    if not isinstance(raw_solution, Sequence) or isinstance(raw_solution, (str, bytes)):
        raise TypeError("MDKP solver output must be item indices or a binary vector.")
    if len(raw_solution) == num_items and all(isinstance(value, bool) for value in raw_solution):
        return [index for index, value in enumerate(raw_solution) if value]
    if len(raw_solution) == num_items and all(_is_binary_number(value) for value in raw_solution):
        return [index for index, value in enumerate(raw_solution) if int(round(float(value))) == 1]
    return sorted({int(value) for value in raw_solution})


def _is_binary_number(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return abs(float(value)) < 1e-9 or abs(float(value) - 1.0) < 1e-9
    return False


def validate_fractional_solution(
    solution: list[float],
    instance: dict[str, object],
    *,
    tolerance: float = FEASIBILITY_TOLERANCE,
) -> tuple[bool, str | None]:
    num_items = int(instance["num_items"])
    if len(solution) != num_items:
        return False, f"Returned {len(solution)} item values for {num_items} items."
    for item_index, value in enumerate(solution):
        if value < -tolerance or value > 1.0 + tolerance:
            return False, f"x[{item_index}]={value} is outside [0, 1]."
    return validate_capacity_usage(instance, solution, tolerance=tolerance)


def validate_binary_solution(
    solution: list[int],
    instance: dict[str, object],
    *,
    tolerance: float = FEASIBILITY_TOLERANCE,
) -> tuple[bool, str | None]:
    num_items = int(instance["num_items"])
    if len(solution) != len(set(solution)):
        return False, "Selected item list contains duplicates."
    for item in solution:
        if item < 0 or item >= num_items:
            return False, f"Selected item {item} is outside 0..{num_items - 1}."
    vector = [1.0 if item in set(solution) else 0.0 for item in range(num_items)]
    return validate_capacity_usage(instance, vector, tolerance=tolerance)


def validate_capacity_usage(
    instance: dict[str, object],
    vector: Sequence[float],
    *,
    tolerance: float,
) -> tuple[bool, str | None]:
    usage = resource_usage(instance, vector)
    capacities = [float(value) for value in instance["capacities"]]
    for resource, (used, capacity) in enumerate(zip(usage, capacities, strict=True)):
        if used > capacity + tolerance:
            return False, f"Resource {resource} usage {used:.6g} exceeds capacity {capacity:.6g}."
    return True, None


def objective_value(instance: dict[str, object], vector: Sequence[float]) -> float:
    values = [float(value) for value in instance["values"]]
    return sum(value * float(x) for value, x in zip(values, vector, strict=True))


def selection_objective_value(instance: dict[str, object], selected_items: Sequence[int]) -> float:
    values = [float(value) for value in instance["values"]]
    return sum(values[item] for item in selected_items)


def resource_usage(instance: dict[str, object], vector: Sequence[float]) -> list[float]:
    num_resources = int(instance["num_resources"])
    weights = [[float(value) for value in row] for row in instance["weights"]]
    usage = [0.0] * num_resources
    for amount, row in zip(vector, weights, strict=True):
        for resource, weight in enumerate(row):
            usage[resource] += float(amount) * weight
    return usage


def selected_resource_usage(instance: dict[str, object], selected_items: Sequence[int]) -> list[float]:
    num_items = int(instance["num_items"])
    vector = [0.0] * num_items
    for item in selected_items:
        vector[item] = 1.0
    return resource_usage(instance, vector)


def density_score(instance: dict[str, object], item: int, *, resource_prices: Sequence[float] | None = None) -> float:
    values = [float(value) for value in instance["values"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    if resource_prices is None:
        resource_prices = [1.0 / max(capacity, FEASIBILITY_TOLERANCE) for capacity in capacities]
    pressure = sum(price * weights[item][resource] for resource, price in enumerate(resource_prices))
    return values[item] / max(pressure, FEASIBILITY_TOLERANCE)


def greedy_fractional_solution(
    instance: dict[str, object],
    *,
    resource_prices: Sequence[float] | None = None,
) -> list[float]:
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    capacities = [float(value) for value in instance["capacities"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    solution = [0.0] * num_items
    remaining = list(capacities)
    order = sorted(
        range(num_items),
        key=lambda item: (density_score(instance, item, resource_prices=resource_prices), float(instance["values"][item])),
        reverse=True,
    )
    for item in order:
        max_fraction = 1.0
        for resource in range(num_resources):
            weight = weights[item][resource]
            if weight > FEASIBILITY_TOLERANCE:
                max_fraction = min(max_fraction, remaining[resource] / weight)
        fraction = max(0.0, min(1.0, max_fraction))
        if fraction <= FEASIBILITY_TOLERANCE:
            continue
        solution[item] = fraction
        for resource in range(num_resources):
            remaining[resource] -= weights[item][resource] * fraction
            if abs(remaining[resource]) < FEASIBILITY_TOLERANCE:
                remaining[resource] = 0.0
    return solution


def greedy_binary_solution(
    instance: dict[str, object],
    *,
    resource_prices: Sequence[float] | None = None,
) -> list[int]:
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    capacities = [float(value) for value in instance["capacities"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    remaining = list(capacities)
    selected: list[int] = []
    for item in sorted(
        range(num_items),
        key=lambda candidate: (
            density_score(instance, candidate, resource_prices=resource_prices),
            float(instance["values"][candidate]),
        ),
        reverse=True,
    ):
        if all(weights[item][resource] <= remaining[resource] + FEASIBILITY_TOLERANCE for resource in range(num_resources)):
            selected.append(item)
            for resource in range(num_resources):
                remaining[resource] -= weights[item][resource]
    return sorted(selected)


def redundancy_improved_binary_solution(instance: dict[str, object]) -> list[int]:
    selected = set(greedy_binary_solution(instance))
    values = [float(value) for value in instance["values"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    improved = True
    while improved:
        improved = False
        usage = selected_resource_usage(instance, sorted(selected))
        for add_item in range(int(instance["num_items"])):
            if add_item in selected:
                continue
            if all(usage[resource] + weights[add_item][resource] <= capacities[resource] + FEASIBILITY_TOLERANCE for resource in range(int(instance["num_resources"]))):
                selected.add(add_item)
                improved = True
                break
        if improved:
            continue
        for remove_item in list(selected):
            usage_without = [used - weights[remove_item][resource] for resource, used in enumerate(usage)]
            for add_item in range(int(instance["num_items"])):
                if add_item in selected:
                    continue
                if values[add_item] <= values[remove_item]:
                    continue
                if all(
                    usage_without[resource] + weights[add_item][resource] <= capacities[resource] + FEASIBILITY_TOLERANCE
                    for resource in range(int(instance["num_resources"]))
                ):
                    selected.remove(remove_item)
                    selected.add(add_item)
                    improved = True
                    break
            if improved:
                break
    return sorted(selected)


def summarize_packing_training_data(
    train_instances: list[dict[str, object]],
    manifest: dict[str, object],
) -> dict[str, object]:
    value_means = []
    tightness_by_resource: list[list[float]] = []
    density_examples = []
    for instance in train_instances:
        values = [float(value) for value in instance["values"]]
        capacities = [float(value) for value in instance["capacities"]]
        weights = [[float(value) for value in row] for row in instance["weights"]]
        value_means.append(statistics.mean(values))
        totals = [
            sum(weights[item][resource] for item in range(int(instance["num_items"])))
            for resource in range(int(instance["num_resources"]))
        ]
        tightness_by_resource.append([
            capacities[resource] / max(totals[resource], FEASIBILITY_TOLERANCE)
            for resource in range(int(instance["num_resources"]))
        ])
        density_examples.append(
            sorted(
                [
                    round(density_score(instance, item), 4)
                    for item in range(min(8, int(instance["num_items"])))
                ],
                reverse=True,
            )
        )
    transposed = list(zip(*tightness_by_resource, strict=True))
    return {
        "problem": manifest["problem"],
        "family": manifest["family"],
        "num_instances": len(train_instances),
        "num_items": int(train_instances[0]["num_items"]),
        "num_resources": int(train_instances[0]["num_resources"]),
        "value_mean": round(statistics.mean(value_means), 4),
        "capacity_tightness_mean_by_resource": [round(statistics.mean(values), 4) for values in transposed],
        "capacity_tightness_std_by_resource": [
            round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0
            for values in transposed
        ],
        "density_prefix_examples": density_examples[:4],
        "sample_instances": [
            {
                "id": instance["id"],
                "value_prefix": instance["values"][:8],
                "capacity_vector": instance["capacities"],
                "weight_prefix": instance["weights"][:5],
            }
            for instance in train_instances[:3]
        ],
    }


def solve_packing_lp_glop(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    solver = pywraplp.Solver.CreateSolver("GLOP")
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    variables = [solver.NumVar(0.0, 1.0, f"x_{item}") for item in range(num_items)]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    for resource in range(num_resources):
        solver.Add(sum(weights[item][resource] * variables[item] for item in range(num_items)) <= capacities[resource])
    solver.Maximize(sum(float(instance["values"][item]) * variables[item] for item in range(num_items)))
    status = solver.Solve()
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        raise RuntimeError(f"GLOP failed to solve packing LP, status={status}.")
    solution = [max(0.0, min(1.0, variables[item].solution_value())) for item in range(num_items)]
    return ExactSolveResult(
        solution=solution,
        objective_value=float(solver.Objective().Value()),
        runtime_ms=(time.perf_counter() - start) * 1000.0,
        source="ortools-glop",
    )


def solve_mdkp_cpsat(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    model = cp_model.CpModel()
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    values = [int(round(float(value))) for value in instance["values"]]
    weights = [[int(round(float(value))) for value in row] for row in instance["weights"]]
    capacities = [int(round(float(value))) for value in instance["capacities"]]
    variables = [model.NewBoolVar(f"x_{item}") for item in range(num_items)]
    for resource in range(num_resources):
        model.Add(sum(weights[item][resource] * variables[item] for item in range(num_items)) <= capacities[resource])
    model.Maximize(sum(values[item] * variables[item] for item in range(num_items)))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise RuntimeError(f"CP-SAT failed to solve MDKP, status={status}.")
    selected = [item for item in range(num_items) if solver.BooleanValue(variables[item])]
    return ExactSolveResult(
        solution=selected,
        objective_value=float(sum(values[item] for item in selected)),
        runtime_ms=(time.perf_counter() - start) * 1000.0,
        source="ortools-cpsat",
    )
