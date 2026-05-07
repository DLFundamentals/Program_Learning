from __future__ import annotations

import itertools
import math
import os
import statistics
import time
from contextlib import contextmanager

import gurobipy as gp
from gurobipy import GRB

from dasbench.problems.base import ExactSolveResult
from dasbench.problems.tsp_utils import (
    canonicalize_tour,
    distance_matrix,
    farthest_insertion_tour,
    nearest_insertion_tour,
    tour_length_from_matrix,
    two_opt_improve,
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


def _tour_edges(tour: list[int]) -> list[tuple[int, int]]:
    edges = []
    for left, right in zip(tour, tour[1:], strict=False):
        edges.append((min(left, right), max(left, right)))
    edges.append((min(tour[-1], tour[0]), max(tour[-1], tour[0])))
    return edges


def _shortest_subtour(selected_edges: list[tuple[int, int]], num_cities: int) -> list[int]:
    neighbors = {city: [] for city in range(num_cities)}
    for left, right in selected_edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    remaining = set(range(num_cities))
    best_cycle = list(range(num_cities))
    while remaining:
        start = next(iter(remaining))
        cycle = []
        stack = [start]
        while stack:
            city = stack.pop()
            if city not in remaining:
                continue
            remaining.remove(city)
            cycle.append(city)
            for neighbor in neighbors[city]:
                if neighbor in remaining:
                    stack.append(neighbor)
        if len(cycle) < len(best_cycle):
            best_cycle = cycle
    return best_cycle


def _extract_tour(selected_edges: list[tuple[int, int]], num_cities: int) -> list[int]:
    neighbors = {city: [] for city in range(num_cities)}
    for left, right in selected_edges:
        neighbors[left].append(right)
        neighbors[right].append(left)
    if any(len(items) != 2 for items in neighbors.values()):
        raise RuntimeError("Selected edges do not form a Hamiltonian cycle.")
    tour = [0]
    previous = None
    current = 0
    while len(tour) < num_cities:
        options = neighbors[current]
        next_city = options[0] if options[0] != previous else options[1]
        tour.append(next_city)
        previous, current = current, next_city
    return canonicalize_tour(tour, num_cities)


def _matrix_and_points(instance: dict[str, object]) -> tuple[list[list[float]], list[tuple[float, float]]]:
    points = [(float(x), float(y)) for x, y in instance["points"]]
    return distance_matrix(points), points


def _best_candidate_tour(instance: dict[str, object], candidates: list[list[int]]) -> list[int]:
    num_cities = int(instance["num_cities"])
    matrix, _ = _matrix_and_points(instance)
    valid: list[list[int]] = []
    for candidate in candidates:
        if not candidate:
            continue
        if len(candidate) != num_cities or len(set(candidate)) != num_cities:
            continue
        valid.append(canonicalize_tour(candidate, num_cities))
    if not valid:
        fallback = farthest_insertion_tour(instance)
        return two_opt_improve(instance, fallback, max_rounds=6)
    best = min(valid, key=lambda tour: tour_length_from_matrix(matrix, tour))
    return two_opt_improve(instance, best, max_rounds=6)


def _global_angle_tour(instance: dict[str, object]) -> list[int]:
    _, points = _matrix_and_points(instance)
    centroid = (
        statistics.mean(point[0] for point in points),
        statistics.mean(point[1] for point in points),
    )
    return [
        city
        for city in sorted(
            range(len(points)),
            key=lambda city: (
                math.atan2(points[city][1] - centroid[1], points[city][0] - centroid[0]),
                city,
            ),
        )
    ]


def _ring_cluster_warm_start(instance: dict[str, object]) -> list[int]:
    cluster_ids = instance.get("_cluster_ids")
    centers = instance.get("_cluster_centers")
    if not isinstance(cluster_ids, list) or not isinstance(centers, list) or not cluster_ids or not centers:
        return _best_candidate_tour(instance, [_global_angle_tour(instance), nearest_insertion_tour(instance)])

    _, points = _matrix_and_points(instance)
    normalized_centers = [(float(x), float(y)) for x, y in centers]
    grouped: dict[int, list[int]] = {cluster: [] for cluster in range(len(normalized_centers))}
    for city, raw_cluster in enumerate(cluster_ids):
        cluster = int(raw_cluster)
        grouped.setdefault(cluster, []).append(city)

    center_centroid = (
        statistics.mean(center[0] for center in normalized_centers),
        statistics.mean(center[1] for center in normalized_centers),
    )
    base_order = sorted(
        range(len(normalized_centers)),
        key=lambda cluster: (
            math.atan2(
                normalized_centers[cluster][1] - center_centroid[1],
                normalized_centers[cluster][0] - center_centroid[0],
            ),
            cluster,
        ),
    )
    local_orders = {
        cluster: sorted(
            grouped.get(cluster, []),
            key=lambda city: (
                math.atan2(
                    points[city][1] - normalized_centers[cluster][1],
                    points[city][0] - normalized_centers[cluster][0],
                ),
                city,
            ),
        )
        for cluster in base_order
    }

    candidates: list[list[int]] = [_global_angle_tour(instance), nearest_insertion_tour(instance)]
    for cluster_order in (base_order, list(reversed(base_order))):
        cluster_count = len(cluster_order)
        for orientation_mask in range(1 << cluster_count):
            tour: list[int] = []
            for position, cluster in enumerate(cluster_order):
                local = local_orders.get(cluster, [])
                if (orientation_mask >> position) & 1:
                    tour.extend(reversed(local))
                else:
                    tour.extend(local)
            candidates.append(tour)
    return _best_candidate_tour(instance, candidates)


def _ribbon_warm_start(instance: dict[str, object]) -> list[int]:
    sides = instance.get("_ribbon_sides")
    major_coordinates = instance.get("_major_coordinates")
    if not isinstance(sides, list) or not isinstance(major_coordinates, list) or len(sides) != len(major_coordinates):
        points = [(float(x), float(y)) for x, y in instance["points"]]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        major_axis = 0 if (max(xs) - min(xs)) >= (max(ys) - min(ys)) else 1
        minor_axis = 1 - major_axis
        median_minor = statistics.median(point[minor_axis] for point in points)
        sides = [1 if point[minor_axis] >= median_minor else 0 for point in points]
        major_coordinates = [point[major_axis] for point in points]

    groups = sorted({int(side) for side in sides})
    if len(groups) != 2:
        return _best_candidate_tour(instance, [_global_angle_tour(instance), nearest_insertion_tour(instance)])
    first = sorted(
        [city for city, side in enumerate(sides) if int(side) == groups[0]],
        key=lambda city: (float(major_coordinates[city]), city),
    )
    second = sorted(
        [city for city, side in enumerate(sides) if int(side) == groups[1]],
        key=lambda city: (float(major_coordinates[city]), city),
    )
    candidates = [
        first + list(reversed(second)),
        second + list(reversed(first)),
        list(reversed(first)) + second,
        list(reversed(second)) + first,
        nearest_insertion_tour(instance),
    ]
    return _best_candidate_tour(instance, candidates)


def _barrier_warm_start(instance: dict[str, object]) -> list[int]:
    groups = instance.get("_barrier_groups")
    if not isinstance(groups, list):
        return _best_candidate_tour(instance, [_global_angle_tour(instance), nearest_insertion_tour(instance)])

    _, points = _matrix_and_points(instance)
    left = sorted(
        [city for city, group in enumerate(groups) if int(group) == 0],
        key=lambda city: (points[city][1], points[city][0], city),
    )
    bridge = sorted(
        [city for city, group in enumerate(groups) if int(group) == 1],
        key=lambda city: (points[city][0], points[city][1], city),
    )
    right = sorted(
        [city for city, group in enumerate(groups) if int(group) == 2],
        key=lambda city: (points[city][1], points[city][0], city),
    )

    candidates = [
        left + bridge + list(reversed(right)),
        right + list(reversed(bridge)) + list(reversed(left)),
        list(reversed(left)) + bridge + right,
        list(reversed(right)) + list(reversed(bridge)) + left,
        nearest_insertion_tour(instance),
    ]
    return _best_candidate_tour(instance, candidates)


def _family_tsp_warm_start(instance: dict[str, object]) -> list[int]:
    regime_name = str(instance.get("_regime_name", ""))
    if regime_name == "ring_clusters" or "_cluster_ids" in instance:
        return _ring_cluster_warm_start(instance)
    if regime_name == "ribbons" or "_ribbon_sides" in instance:
        return _ribbon_warm_start(instance)
    if regime_name == "barrier_bridge" or "_barrier_groups" in instance:
        return _barrier_warm_start(instance)
    return _best_candidate_tour(
        instance,
        [
            nearest_insertion_tour(instance),
            farthest_insertion_tour(instance),
            _global_angle_tour(instance),
        ],
    )


def _gurobi_exact_tsp(instance: dict[str, object], *, warm_start: list[int]) -> tuple[list[int], float]:
    num_cities = int(instance["num_cities"])
    matrix, _ = _matrix_and_points(instance)
    warm_start_edges = set(_tour_edges(warm_start)) if warm_start else set()

    with _quiet_gurobi_model("dasbench_dataset_tsp") as model:
        model.Params.OutputFlag = 0
        model.Params.LogToConsole = 0
        model.Params.Threads = _dataset_exact_threads()
        model.Params.MIPGap = 0.0
        model.Params.LazyConstraints = 1
        edge_vars = {
            (left, right): model.addVar(vtype=GRB.BINARY, obj=matrix[left][right], name=f"e_{left}_{right}")
            for left in range(num_cities)
            for right in range(left + 1, num_cities)
        }
        for city in range(num_cities):
            incident = [
                edge_vars[(min(city, other), max(city, other))]
                for other in range(num_cities)
                if other != city
            ]
            model.addConstr(gp.quicksum(incident) == 2, name=f"degree_{city}")
        model.ModelSense = GRB.MINIMIZE
        for edge, variable in edge_vars.items():
            if edge in warm_start_edges:
                variable.Start = 1.0

        model._edge_vars = edge_vars  # type: ignore[attr-defined]
        model._num_cities = num_cities  # type: ignore[attr-defined]

        def callback(callback_model, where):  # type: ignore[no-untyped-def]
            if where != GRB.Callback.MIPSOL:
                return
            values = callback_model.cbGetSolution(list(callback_model._edge_vars.values()))
            selected = [
                edge
                for edge, value in zip(callback_model._edge_vars.keys(), values, strict=True)
                if value > 0.5
            ]
            cycle = _shortest_subtour(selected, int(callback_model._num_cities))
            if len(cycle) == int(callback_model._num_cities):
                return
            callback_model.cbLazy(
                gp.quicksum(
                    callback_model._edge_vars[(min(left, right), max(left, right))]
                    for left, right in itertools.combinations(cycle, 2)
                )
                <= len(cycle) - 1
            )

        model.optimize(callback)
        if int(model.Status) != GRB.OPTIMAL or int(model.SolCount) <= 0:
            raise RuntimeError(f"Gurobi failed to prove optimality for TSP dataset labeling, status={int(model.Status)}.")
        selected_edges = [edge for edge, variable in edge_vars.items() if float(variable.X) > 0.5]
        solution = _extract_tour(selected_edges, num_cities)
        return solution, float(model.ObjVal)


def _solve_family_tsp_exact(instance: dict[str, object], *, source_prefix: str) -> ExactSolveResult:
    start = time.perf_counter()
    warm_start = _family_tsp_warm_start(instance)
    solution, objective = _gurobi_exact_tsp(instance, warm_start=warm_start)
    return ExactSolveResult(
        solution=solution,
        objective_value=objective,
        runtime_ms=(time.perf_counter() - start) * 1000.0,
        source=f"{source_prefix}:gurobi_exact",
    )


def solve_clustered_euclidean_exact(
    instance: dict[str, object],
    context: dict[str, object],
    state: object,
) -> ExactSolveResult:
    del context, state
    return _solve_family_tsp_exact(instance, source_prefix="family-clustered_euclidean_v1")


def solve_paired_ribbon_zigzag_exact(
    instance: dict[str, object],
    context: dict[str, object],
    state: object,
) -> ExactSolveResult:
    del context, state
    return _solve_family_tsp_exact(instance, source_prefix="family-paired_ribbon_zigzag_v1")


def solve_latent_metric_mixture_exact(
    instance: dict[str, object],
    context: dict[str, object],
    state: object,
) -> ExactSolveResult:
    del context, state
    return _solve_family_tsp_exact(instance, source_prefix="family-latent_metric_mixture_v1")
