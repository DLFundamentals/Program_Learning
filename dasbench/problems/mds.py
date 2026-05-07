from __future__ import annotations

import statistics

from dasbench.problems.base import ProblemDefinition, ScoreResult
from dasbench.problems.graph_utils import (
    adjacency_sets,
    connected_component_sizes,
    degree_histogram,
    degree_sequence,
    edge_density,
    greedy_dominating_set_max_gain,
    is_dominating_set,
    prune_redundant_dominating_vertices_fast,
    solve_mds_set_cover_branch_and_bound,
    solve_mds_exact,
    validate_edges,
)


def validate_instance(instance: dict[str, object]) -> None:
    num_vertices = int(instance["num_vertices"])
    if num_vertices <= 0:
        raise ValueError("MDS instances require at least one vertex.")
    edges = instance["edges"]
    if not isinstance(edges, list):
        raise ValueError("MDS instances require an edge list.")
    validate_edges(num_vertices, edges)


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[int]:
    if not isinstance(raw_solution, (list, tuple, set)):
        raise TypeError("MDS solver output must be a sequence of vertex ids.")
    return sorted(int(value) for value in raw_solution)


def validate_solution(solution: list[int], instance: dict[str, object]) -> tuple[bool, str | None]:
    adjacency = adjacency_sets(int(instance["num_vertices"]), instance["edges"])
    return is_dominating_set(int(instance["num_vertices"]), adjacency, solution)


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
    objective = float(len(solution))
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
    overlap_examples: list[list[int]] = []
    component_signatures: list[list[int]] = []
    degree_histograms: list[dict[str, int]] = []
    for instance in train_instances:
        adjacency = adjacency_sets(int(instance["num_vertices"]), instance["edges"])
        degrees = degree_sequence(adjacency)
        densities.append(edge_density(int(instance["num_vertices"]), instance["edges"]))
        average_degrees.append(statistics.mean(degrees))
        component_signatures.append(connected_component_sizes(int(instance["num_vertices"]), adjacency)[:6])
        degree_histograms.append(degree_histogram(adjacency))
        if int(instance["num_vertices"]) >= 2:
            overlap_examples.append(
                sorted(
                    [
                        len((adjacency[left] | {left}) & (adjacency[right] | {right}))
                        for left in range(min(4, int(instance["num_vertices"])))
                        for right in range(left + 1, min(6, int(instance["num_vertices"])))
                    ],
                    reverse=True,
                )[:6]
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
        "closed_neighborhood_overlap_examples": overlap_examples[:4],
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
    adjacency = adjacency_sets(int(instance["num_vertices"]), instance["edges"])
    valid, error = is_dominating_set(int(instance["num_vertices"]), adjacency, solution)
    dominated = set(solution)
    for vertex in solution:
        dominated.update(adjacency[vertex])
    missing = sorted(set(range(int(instance["num_vertices"]))) - dominated)
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "selected_vertices": solution[:12],
        "undominated_vertices": missing[:6],
        "error": score.error or (None if valid else error),
    }


def baseline_registry() -> dict[str, object]:
    def closed_sets(instance: dict[str, object]) -> list[set[int]]:
        adjacency = adjacency_sets(int(instance["num_vertices"]), instance["edges"])
        return [neighbors | {vertex} for vertex, neighbors in enumerate(adjacency)]

    return {
        "high_degree_greedy": lambda instance: prune_redundant_dominating_vertices_fast(
            instance,
            greedy_dominating_set_max_gain(instance),
        ),
        "fast_marginal_gain_greedy": lambda instance: prune_redundant_dominating_vertices_fast(
            instance,
            greedy_dominating_set_max_gain(instance),
        ),
        "fast_redundancy_aware": lambda instance: prune_redundant_dominating_vertices_fast(
            instance,
            greedy_dominating_set_max_gain(instance, prefer_low_overlap=True),
        ),
        "marginal_gain_greedy": lambda instance: prune_redundant_dominating_vertices_fast(
            instance,
            greedy_dominating_set_max_gain(instance),
        ),
        "redundancy_aware": lambda instance: prune_redundant_dominating_vertices_fast(
            instance,
            greedy_dominating_set_max_gain(instance, prefer_low_overlap=True),
        ),
        "exact": lambda instance: list(solve_mds_exact(instance).solution),
        "cpsat_exact": lambda instance: list(solve_mds_exact(instance).solution),
        "set_cover_branch_bound_exact": lambda instance: list(solve_mds_set_cover_branch_and_bound(instance).solution),
    }


PROBLEM = ProblemDefinition(
    name="mds",
    description="Distribution-aware synthesis benchmark for minimum dominating set.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is optimum_dominating_set_size / returned_dominating_set_size",
    },
    instance_schema_version="mds.v1",
    default_instance_params={"num_vertices": 24},
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_mds_exact,
)
