from __future__ import annotations

import statistics

from dasbench.problems.base import ProblemDefinition, ScoreResult
from dasbench.problems.graph_utils import (
    adjacency_sets,
    connected_component_sizes,
    degree_histogram,
    degree_sequence,
    edge_density,
    greedy_independent_set_by_score,
    greedy_independent_set_with_local_improvement,
    is_independent_set,
    sample_independent_set_random,
    solve_mis_clique_branch_and_bound,
    solve_mis_exact,
    validate_edges,
)


def validate_instance(instance: dict[str, object]) -> None:
    num_vertices = int(instance["num_vertices"])
    if num_vertices <= 0:
        raise ValueError("MIS instances require at least one vertex.")
    edges = instance["edges"]
    if not isinstance(edges, list):
        raise ValueError("MIS instances require an edge list.")
    validate_edges(num_vertices, edges)


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[int]:
    if not isinstance(raw_solution, (list, tuple, set)):
        raise TypeError("MIS solver output must be a sequence of vertex ids.")
    return sorted(int(value) for value in raw_solution)


def validate_solution(solution: list[int], instance: dict[str, object]) -> tuple[bool, str | None]:
    return is_independent_set(int(instance["num_vertices"]), instance["edges"], solution)


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
    return ScoreResult(
        is_valid=True,
        is_feasible=True,
        objective_value=objective,
        normalized_quality=0.0 if optimum <= 0 else objective / optimum,
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
    for instance in train_instances:
        adjacency = adjacency_sets(int(instance["num_vertices"]), instance["edges"])
        degrees = degree_sequence(adjacency)
        densities.append(edge_density(int(instance["num_vertices"]), instance["edges"]))
        average_degrees.append(statistics.mean(degrees))
        component_signatures.append(connected_component_sizes(int(instance["num_vertices"]), adjacency)[:6])
        degree_histograms.append(degree_histogram(adjacency))
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
    conflicts = []
    seen = set(solution)
    for vertex in solution:
        for neighbor in adjacency[vertex]:
            if neighbor in seen and vertex < neighbor:
                conflicts.append([vertex, neighbor])
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "selected_vertices": solution[:12],
        "conflict_examples": conflicts[:3],
        "error": score.error,
    }


def baseline_registry() -> dict[str, object]:
    return {
        "random_greedy": lambda instance: sample_independent_set_random(instance, seed_label="mis"),
        "min_degree_greedy": lambda instance: greedy_independent_set_by_score(
            instance,
            score_fn=lambda vertex, adjacency, remaining: len(adjacency[vertex] & remaining),
        ),
        "ratio_greedy": lambda instance: greedy_independent_set_by_score(
            instance,
            score_fn=lambda vertex, adjacency, remaining: (
                len(adjacency[vertex] & remaining),
                -sum(len(adjacency[neighbor] & remaining) for neighbor in adjacency[vertex] & remaining),
            ),
        ),
        "local_improve": greedy_independent_set_with_local_improvement,
        "exact": lambda instance: list(solve_mis_exact(instance).solution),
        "cpsat_exact": lambda instance: list(solve_mis_exact(instance).solution),
        "clique_branch_bound_exact": lambda instance: list(solve_mis_clique_branch_and_bound(instance).solution),
    }


PROBLEM = ProblemDefinition(
    name="mis",
    description="Distribution-aware synthesis benchmark for maximum independent set.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is independent_set_size / optimum_independent_set_size",
    },
    instance_schema_version="mis.v1",
    default_instance_params={"num_vertices": 24},
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_mis_exact,
)
