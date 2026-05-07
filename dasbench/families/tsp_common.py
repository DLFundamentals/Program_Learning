from __future__ import annotations

import math
import random

from dasbench.problems.tsp_utils import rounded_points


def build_tsp_instance(
    instance_id: str,
    points: list[tuple[float, float]],
    *,
    private_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    instance = {
        "id": instance_id,
        "num_cities": len(points),
        "points": rounded_points(points),
        "metric": "euclidean_2d",
    }
    for key, value in (private_metadata or {}).items():
        instance[f"_{key.removeprefix('_')}"] = value
    return instance


def balanced_counts(num_items: int, num_groups: int) -> list[int]:
    counts = [num_items // num_groups for _ in range(num_groups)]
    for index in range(num_items % num_groups):
        counts[index] += 1
    return counts


def ring_centers(num_centers: int, *, radius: float, phase: float = 0.0) -> list[tuple[float, float]]:
    return [
        (
            radius * math.cos(phase + (2.0 * math.pi * index) / num_centers),
            radius * math.sin(phase + (2.0 * math.pi * index) / num_centers),
        )
        for index in range(num_centers)
    ]


def sample_cluster_points(
    rng: random.Random,
    centers: list[tuple[float, float]],
    counts: list[int],
    *,
    spread: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for center, count in zip(centers, counts, strict=True):
        for _ in range(count):
            points.append(
                (
                    center[0] + rng.uniform(-spread, spread),
                    center[1] + rng.uniform(-spread, spread),
                )
            )
    rng.shuffle(points)
    return points


def sample_cluster_points_with_labels(
    rng: random.Random,
    centers: list[tuple[float, float]],
    counts: list[int],
    *,
    spread: float,
) -> tuple[list[tuple[float, float]], list[int]]:
    labeled_points: list[tuple[tuple[float, float], int]] = []
    for cluster, (center, count) in enumerate(zip(centers, counts, strict=True)):
        for _ in range(count):
            labeled_points.append(
                (
                    (
                        center[0] + rng.uniform(-spread, spread),
                        center[1] + rng.uniform(-spread, spread),
                    ),
                    cluster,
                )
            )
    rng.shuffle(labeled_points)
    return [point for point, _ in labeled_points], [cluster for _, cluster in labeled_points]
