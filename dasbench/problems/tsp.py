from __future__ import annotations

import statistics

from dasbench.problems.base import ProblemDefinition, ScoreResult
from dasbench.problems.tsp_utils import (
    canonicalize_tour,
    distance_matrix,
    farthest_insertion_tour,
    is_valid_tour,
    lkh_tour,
    multi_start_two_opt_tour,
    nearest_insertion_tour,
    nearest_neighbor_tour,
    random_tour,
    rounded_points,
    solve_tsp_exact,
    tour_length_from_matrix,
    two_opt_improve,
)


def validate_instance(instance: dict[str, object]) -> None:
    num_cities = int(instance["num_cities"])
    if num_cities <= 1:
        raise ValueError("TSP instances require at least two cities.")
    points = instance["points"]
    if not isinstance(points, list) or len(points) != num_cities:
        raise ValueError("TSP instances require a point list matching num_cities.")
    for index, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"Point {index} is invalid: {point!r}.")
        float(point[0])
        float(point[1])


def canonicalize_solution(raw_solution, instance: dict[str, object]) -> list[int]:
    return canonicalize_tour(raw_solution, int(instance["num_cities"]))


def validate_solution(solution: list[int], instance: dict[str, object]) -> tuple[bool, str | None]:
    return is_valid_tour(int(instance["num_cities"]), solution)


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
    matrix = distance_matrix(instance["points"])
    objective = float(tour_length_from_matrix(matrix, solution))
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
    widths: list[float] = []
    heights: list[float] = []
    nearest_neighbor_means: list[float] = []
    radial_stds: list[float] = []
    sample_instances = []
    for instance in train_instances:
        points = [(float(x), float(y)) for x, y in instance["points"]]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        matrix = distance_matrix(points)
        centroid = (statistics.mean(xs), statistics.mean(ys))
        widths.append(max(xs) - min(xs))
        heights.append(max(ys) - min(ys))
        nearest_neighbor_means.append(
            statistics.mean(
                min(matrix[row][column] for column in range(len(points)) if column != row)
                for row in range(len(points))
            )
        )
        radial_stds.append(
            statistics.pstdev(
                ((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2) ** 0.5 for point in points
            )
        )
        sample_instances.append(
            {
                "id": instance["id"],
                "point_prefix": rounded_points(points[:6]),
            }
        )
    return {
        "problem": manifest["problem"],
        "family": manifest["family"],
        "num_instances": len(train_instances),
        "num_cities": int(train_instances[0]["num_cities"]),
        "bbox_width_mean": round(statistics.mean(widths), 4),
        "bbox_height_mean": round(statistics.mean(heights), 4),
        "nearest_neighbor_distance_mean": round(statistics.mean(nearest_neighbor_means), 4),
        "radial_distance_std_mean": round(statistics.mean(radial_stds), 4),
        "sample_instances": sample_instances[:3],
    }


def failure_case(
    instance: dict[str, object],
    solution: list[int],
    score: ScoreResult,
    runtime_seconds: float,
) -> dict[str, object]:
    return {
        "instance_id": instance["id"],
        "normalized_quality": score.normalized_quality,
        "objective_value": score.objective_value,
        "runtime_ms": runtime_seconds * 1000.0,
        "is_optimal": score.is_optimal,
        "tour_prefix": solution[:10],
        "error": score.error,
    }


def baseline_registry() -> dict[str, object]:
    return {
        "random": lambda instance: random_tour(instance, seed_label="tsp"),
        "nearest_neighbor": nearest_neighbor_tour,
        "nearest_insertion": nearest_insertion_tour,
        "farthest_insertion": farthest_insertion_tour,
        "two_opt_nearest_neighbor": lambda instance: two_opt_improve(
            instance,
            nearest_neighbor_tour(instance),
            max_rounds=6,
        ),
        "two_opt_farthest_insertion": lambda instance: two_opt_improve(
            instance,
            farthest_insertion_tour(instance),
            max_rounds=20,
        ),
        "multi_start_two_opt": multi_start_two_opt_tour,
        "lkh": lkh_tour,
        "exact": lambda instance: list(solve_tsp_exact(instance).solution),
        "held_karp_exact": lambda instance: list(solve_tsp_exact(instance).solution),
    }


PROBLEM = ProblemDefinition(
    name="tsp",
    description="Distribution-aware synthesis benchmark for Euclidean traveling salesperson instances.",
    metric_definition={
        "primary": "normalized_quality",
        "secondary": "optimality_rate",
        "tertiary": "average_runtime_ms",
        "notes": "normalized_quality is optimum_tour_length / returned_tour_length",
    },
    instance_schema_version="tsp.v1",
    default_instance_params={"num_cities": 12},
    validate_instance=validate_instance,
    canonicalize_solution=canonicalize_solution,
    validate_solution=validate_solution,
    score_solution=score_solution,
    summarize_training_data=summarize_training_data,
    failure_case=failure_case,
    baseline_registry=baseline_registry,
    exact_solver=solve_tsp_exact,
)
