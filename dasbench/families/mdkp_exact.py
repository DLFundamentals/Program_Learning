from __future__ import annotations

import os
import time
from contextlib import contextmanager

import gurobipy as gp
from gurobipy import GRB

from dasbench.problems.base import ExactSolveResult
from dasbench.problems.packing_utils import (
    FEASIBILITY_TOLERANCE,
    greedy_binary_solution,
    selection_objective_value,
    selected_resource_usage,
    validate_binary_solution,
)


def _dataset_exact_threads() -> int:
    try:
        return max(1, int(os.environ.get("DASBENCH_DATASET_EXACT_THREADS", "1")))
    except ValueError:
        return 1


@contextmanager
def _quiet_gurobi_model(name: str):
    env = gp.Env(empty=True)
    model = None
    try:
        env.setParam("OutputFlag", 0)
        env.setParam("LogToConsole", 0)
        env.start()
        model = gp.Model(name, env=env)
        yield model
    finally:
        if model is not None:
            model.dispose()
        env.dispose()


def _resource_index(instance: dict[str, object], private_key: str) -> int:
    if private_key in instance:
        return int(instance[private_key])
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    totals = [sum(row[resource] for row in weights) for resource in range(int(instance["num_resources"]))]
    return min(
        range(int(instance["num_resources"])),
        key=lambda resource: capacities[resource] / max(totals[resource], FEASIBILITY_TOLERANCE),
    )


def _primary_resource_dp(instance: dict[str, object], primary_resource: int) -> tuple[list[int], float]:
    capacity = int(round(float(instance["capacities"][primary_resource])))
    values = [int(round(float(value))) for value in instance["values"]]
    primary_weights = [
        int(round(float(instance["weights"][item][primary_resource])))
        for item in range(int(instance["num_items"]))
    ]

    dp = [0] * (capacity + 1)
    take_rows: list[bytearray] = []
    for item, (weight, value) in enumerate(zip(primary_weights, values, strict=True)):
        take = bytearray(capacity + 1)
        for used in range(capacity, weight - 1, -1):
            candidate = dp[used - weight] + value
            if candidate > dp[used]:
                dp[used] = candidate
                take[used] = 1
        take_rows.append(take)

    used = max(range(capacity + 1), key=lambda candidate: (dp[candidate], -candidate))
    selected: list[int] = []
    for item in range(len(primary_weights) - 1, -1, -1):
        if take_rows[item][used]:
            selected.append(item)
            used -= primary_weights[item]
    selected.sort()
    return selected, float(dp[max(range(capacity + 1), key=lambda candidate: (dp[candidate], -candidate))])


def _repaired_selection(instance: dict[str, object], selection: list[int]) -> list[int]:
    if not selection:
        return []
    values = [float(value) for value in instance["values"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    chosen = set(selection)
    while True:
        ordered = sorted(chosen)
        valid, _ = validate_binary_solution(ordered, instance)
        if valid:
            return ordered
        usage = selected_resource_usage(instance, ordered)
        violated = [resource for resource, used in enumerate(usage) if used > capacities[resource] + FEASIBILITY_TOLERANCE]
        if not violated:
            return ordered
        worst_item = max(
            ordered,
            key=lambda item: (
                sum(weights[item][resource] / max(usage[resource] - capacities[resource], 1.0) for resource in violated),
                -values[item],
                item,
            ),
        )
        chosen.remove(worst_item)


def _best_feasible_warm_start(
    instance: dict[str, object],
    *,
    primary_resource: int,
    primary_selection: list[int],
) -> list[int]:
    candidates: list[list[int]] = []

    repaired = _repaired_selection(instance, primary_selection)
    if repaired:
        candidates.append(repaired)

    capacities = [float(value) for value in instance["capacities"]]
    prices = [1.0 / max(capacity, FEASIBILITY_TOLERANCE) for capacity in capacities]
    prices[primary_resource] *= 3.0
    greedy = greedy_binary_solution(instance, resource_prices=prices)
    if greedy:
        candidates.append(greedy)

    if not candidates:
        return []
    return max(
        candidates,
        key=lambda candidate: (selection_objective_value(instance, candidate), -len(candidate)),
    )


def _gurobi_exact_mdkp(instance: dict[str, object], *, warm_start: list[int]) -> tuple[list[int], float]:
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    values = [float(value) for value in instance["values"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]

    with _quiet_gurobi_model("dasbench_dataset_mdkp") as model:
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = 0
        model.Params.Threads = _dataset_exact_threads()
        model.Params.MIPGap = 0.0
        variables = model.addVars(num_items, vtype=GRB.BINARY, name="x")
        for resource in range(num_resources):
            model.addConstr(
                gp.quicksum(weights[item][resource] * variables[item] for item in range(num_items)) <= capacities[resource],
                name=f"capacity_{resource}",
            )
        model.setObjective(gp.quicksum(values[item] * variables[item] for item in range(num_items)), GRB.MAXIMIZE)
        if warm_start:
            selected = set(warm_start)
            for item in range(num_items):
                variables[item].Start = 1.0 if item in selected else 0.0
        model.optimize()
        if int(model.Status) != GRB.OPTIMAL or int(model.SolCount) <= 0:
            raise RuntimeError(f"Gurobi failed to prove optimality for MDKP dataset labeling, status={int(model.Status)}.")
        solution = [item for item in range(num_items) if float(variables[item].X) > 0.5]
        return solution, float(model.ObjVal)


def _solve_family_mdkp_exact(
    instance: dict[str, object],
    *,
    primary_resource: int,
    source_prefix: str,
) -> ExactSolveResult:
    start = time.perf_counter()
    primary_selection, primary_value = _primary_resource_dp(instance, primary_resource)
    valid, _ = validate_binary_solution(primary_selection, instance)
    if valid:
        return ExactSolveResult(
            solution=primary_selection,
            objective_value=primary_value,
            runtime_ms=(time.perf_counter() - start) * 1000.0,
            source=f"{source_prefix}:primary_resource_dp",
        )

    warm_start = _best_feasible_warm_start(
        instance,
        primary_resource=primary_resource,
        primary_selection=primary_selection,
    )
    solution, objective = _gurobi_exact_mdkp(instance, warm_start=warm_start)
    return ExactSolveResult(
        solution=solution,
        objective_value=objective,
        runtime_ms=(time.perf_counter() - start) * 1000.0,
        source=f"{source_prefix}:gurobi_exact",
    )


def solve_single_resource_density_exact(
    instance: dict[str, object],
    context: dict[str, object],
    state: object,
) -> ExactSolveResult:
    del context, state
    return _solve_family_mdkp_exact(
        instance,
        primary_resource=_resource_index(instance, "_bottleneck_resource"),
        source_prefix="family-single_resource_density_v1",
    )


def solve_latent_class_knapsack_exact(
    instance: dict[str, object],
    context: dict[str, object],
    state: object,
) -> ExactSolveResult:
    del context, state
    return _solve_family_mdkp_exact(
        instance,
        primary_resource=_resource_index(instance, "_regime_resource"),
        source_prefix="family-latent_class_knapsack_v1",
    )


def solve_decoy_complement_mixture_exact(
    instance: dict[str, object],
    context: dict[str, object],
    state: object,
) -> ExactSolveResult:
    del context, state
    return _solve_family_mdkp_exact(
        instance,
        primary_resource=_resource_index(instance, "_scarce_resource"),
        source_prefix="family-decoy_complement_mixture_v1",
    )
