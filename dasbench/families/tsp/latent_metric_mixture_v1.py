from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.tsp.paired_ribbon_zigzag_v1 import _ribbon_layout
from dasbench.families.tsp_common import (
    balanced_counts,
    build_tsp_instance,
    ring_centers,
    sample_cluster_points,
    sample_cluster_points_with_labels,
)
from dasbench.families.tsp_exact import solve_latent_metric_mixture_exact


@dataclass(frozen=True)
class FamilyState:
    ring_centers: list[tuple[float, float]]
    left_centers: list[tuple[float, float]]
    right_centers: list[tuple[float, float]]


def build_state(context: dict[str, object]) -> FamilyState:
    rng = random.Random(int(context["seeds"]["family"]))
    phase = rng.uniform(0.0, 0.4)
    return FamilyState(
        ring_centers=ring_centers(4, radius=8.0, phase=phase),
        left_centers=[(-7.0, -2.5), (-5.8, 2.7)],
        right_centers=[(5.8, -2.7), (7.0, 2.4)],
    )


def _barrier_points(rng: random.Random, num_cities: int, state: FamilyState) -> list[tuple[float, float]]:
    if num_cities < 6:
        return [point for point, _, _ in _ribbon_layout(rng, num_cities=num_cities, stagger=0.7, transpose=False)]
    bridge_count = 2
    side_points = num_cities - bridge_count
    counts = balanced_counts(side_points, len(state.left_centers) + len(state.right_centers))
    points = sample_cluster_points(rng, state.left_centers + state.right_centers, counts, spread=0.8)
    points.extend(
        [
            (-0.9 + rng.uniform(-0.15, 0.15), rng.uniform(-0.45, 0.45)),
            (0.9 + rng.uniform(-0.15, 0.15), rng.uniform(-0.45, 0.45)),
        ]
    )
    rng.shuffle(points)
    return points


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    num_cities = int(context["instance_params"]["num_cities"])
    regime = rng.randrange(3)
    if regime == 0:
        counts = balanced_counts(num_cities, len(state.ring_centers))
        points, cluster_ids = sample_cluster_points_with_labels(rng, state.ring_centers, counts, spread=0.9)
        metadata = {
            "regime_name": "ring_clusters",
            "cluster_ids": cluster_ids,
            "cluster_centers": [[round(x, 6), round(y, 6)] for x, y in state.ring_centers],
        }
    elif regime == 1:
        transpose = bool(rng.randrange(2))
        layout = _ribbon_layout(
            rng,
            num_cities=num_cities,
            stagger=0.8,
            transpose=transpose,
        )
        points = [point for point, _, _ in layout]
        metadata = {
            "regime_name": "ribbons",
            "ribbon_sides": [side for _, side, _ in layout],
            "major_coordinates": [round(major, 6) for _, _, major in layout],
            "transpose": transpose,
            "stagger": 0.8,
        }
    else:
        points = _barrier_points(rng, num_cities, state)
        barrier_groups: list[int] = []
        for x, _ in points:
            if x < -1.5:
                barrier_groups.append(0)
            elif x > 1.5:
                barrier_groups.append(2)
            else:
                barrier_groups.append(1)
        metadata = {
            "regime_name": "barrier_bridge",
            "barrier_groups": barrier_groups,
        }
    return build_tsp_instance(instance_id, points, private_metadata=metadata)


FAMILY = FamilyDefinition(
    problem="tsp",
    name="latent_metric_mixture_v1",
    description=(
        "Paper-grade TSP family mixing ring-cluster, ribbon, and barrier-bridge regimes with overlapping Euclidean scale statistics."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    dataset_exact_solver=solve_latent_metric_mixture_exact,
    hidden_rule={
        "summary": "Each instance samples one of three geometric regimes: ring clusters, paired ribbons, or separated side clusters with central bridge points.",
        "signals": ["latent geometric regime", "ring-cluster layout", "paired-ribbon layout", "barrier bridge points"],
        "solver_hint": "Classify the geometry from higher-order point structure and choose the corresponding tour construction strategy.",
    },
)
