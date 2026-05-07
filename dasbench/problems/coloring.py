from __future__ import annotations

import statistics

from dasbench.problems.base import ProblemDefinition, ScoreResult
from dasbench.problems.graph_utils import (
    adjacency_sets,
    canonicalize_coloring,
    connected_component_sizes,
    degree_histogram,
    degree_sequence,
    dsatur_coloring,
    edge_density,
    greedy_coloring_largest_degree,
    greedy_coloring_random_order,
    is_proper_coloring,
    solve_coloring_dsatur_exact,
    smallest_last_coloring,
    solve_coloring_exact,
    validate_edges,
)


def validate_instance(instance: dict[str, object]) -> None:
    num_vertices = int(instance["num_vertices"])
    if num_vertices <= 0:
        raise ValueError("Coloring instances require at least one vertex.")
    edges = instance["edges"]
    if not isinstance(edges, list):
        raise ValueError("Coloring instances require an edge list.")
    validate_edges(num_vertices, edges)


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[int]:
    return canonicalize_coloring(raw_solution, int(instance["num_vertices"]))


def validate_solution(solution: list[int], instance: dict[str, object]) -> tuple[bool, str | None]:
    return is_proper_coloring(int(instance["num_vertices"]), instance["edges"], solution)


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
    objective = float(len(set(solution)))
    optimum = float(instance["optimum_objective"])
    normalized = 0.0 if objective <= 0 else optimum / objective
    return ScoreResult(
        is_valid=True,
        is_feasible=True,
        objective_value=objective,
        normalized_quality=min(1.0, normalized),
        is_optimal=abs(objective - optimum) < 1e-9,
    )


def summarize_training_data(
    train_instances: list[dict[str, object]],
    manifest: dict[str, object],
) -> dict[str, object]:
    densities: list[float] = []
    average_degrees: list[float] = []
    component_signatures: list[list[int]] = []
    degree_histograms: list[dict[str, int]] = []
    heuristic_color_counts: list[dict[str, int]] = []
    for instance in train_instances:
        adjacency = adjacency_sets(int(instance["num_vertices"]), instance["edges"])
        degrees = degree_sequence(adjacency)
        densities.append(edge_density(int(instance["num_vertices"]), instance["edges"]))
        average_degrees.append(statistics.mean(degrees))
        component_signatures.append(connected_component_sizes(int(instance["num_vertices"]), adjacency)[:6])
        degree_histograms.append(degree_histogram(adjacency))
        heuristic_color_counts.append(
            {
                "largest_degree": len(set(greedy_coloring_largest_degree(instance))),
                "smallest_last": len(set(smallest_last_coloring(instance))),
                "dsatur": len(set(dsatur_coloring(instance))),
            }
        )
    return {
        "problem": manifest["problem"],
        "family": manifest["family"],
        "num_instances": len(train_instances),
        "num_vertices": int(train_instances[0]["num_vertices"]),
        "density_mean": round(statistics.mean(densities), 4),
        "density_std": round(statistics.pstdev(densities), 4) if len(densities) > 1 else 0.0,
        "average_degree_mean": round(statistics.mean(average_degrees), 4),
        "component_size_examples": component_signatures[:4],
        "degree_histogram_examples": degree_histograms[:3],
        "heuristic_color_count_examples": heuristic_color_counts[:4],
        "sample_instances": [
            {
                "id": instance["id"],
                "num_edges": len(instance["edges"]),
                "edge_prefix": instance["edges"][:10],
            }
            for instance in train_instances[:3]
        ],
    }


def failure_case(
    instance: dict[str, object],
    solution: list[int],
    score: ScoreResult,
    runtime_seconds: float,
) -> dict[str, object]:
    conflicts = []
    if len(solution) == int(instance["num_vertices"]):
        for u, v in instance["edges"]:
            if solution[int(u)] == solution[int(v)]:
                conflicts.append([int(u), int(v), int(solution[int(u)])])
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "used_colors": sorted(set(solution))[:8],
        "conflict_examples": conflicts[:4],
        "error": score.error,
    }


def baseline_registry() -> dict[str, object]:
    return {
        "random_greedy": lambda instance: greedy_coloring_random_order(instance, seed_label="coloring"),
        "largest_degree": greedy_coloring_largest_degree,
        "smallest_last": smallest_last_coloring,
        "dsatur": dsatur_coloring,
        "exact": lambda instance: list(solve_coloring_exact(instance).solution),
        "cpsat_exact": lambda instance: list(solve_coloring_exact(instance).solution),
        "dsatur_branch_bound_exact": lambda instance: list(solve_coloring_dsatur_exact(instance).solution),
    }


PROBLEM = ProblemDefinition(
    name="coloring",
    description="Distribution-aware synthesis benchmark for graph coloring.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is optimum_number_of_colors / returned_number_of_colors",
    },
    instance_schema_version="coloring.v1",
    default_instance_params={"num_vertices": 16},
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_coloring_exact,
)
