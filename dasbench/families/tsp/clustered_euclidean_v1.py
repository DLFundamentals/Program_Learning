from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.tsp_common import (
    balanced_counts,
    build_tsp_instance,
    ring_centers,
    sample_cluster_points_with_labels,
)
from dasbench.families.tsp_exact import solve_clustered_euclidean_exact


@dataclass(frozen=True)
class FamilyState:
    centers: list[tuple[float, float]]


def build_state(context: dict[str, object]) -> FamilyState:
    rng = random.Random(int(context["seeds"]["family"]))
    num_clusters = 4
    phase = rng.uniform(0.0, 0.3)
    return FamilyState(centers=ring_centers(num_clusters, radius=8.0, phase=phase))


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    num_cities = int(context["instance_params"]["num_cities"])
    counts = balanced_counts(num_cities, len(state.centers))
    points, cluster_ids = sample_cluster_points_with_labels(rng, state.centers, counts, spread=0.85)
    return build_tsp_instance(
        instance_id,
        points,
        private_metadata={
            "cluster_ids": cluster_ids,
            "cluster_centers": [[round(x, 6), round(y, 6)] for x, y in state.centers],
            "regime_name": "ring_clusters",
        },
    )


FAMILY = FamilyDefinition(
    problem="tsp",
    name="clustered_euclidean_v1",
    description="Smoke TSP family with Euclidean clusters arranged around a ring.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    dataset_exact_solver=solve_clustered_euclidean_exact,
    hidden_rule={
        "summary": "Cities are sampled from balanced clusters around four hidden centers arranged on a ring with a shared phase.",
        "signals": ["four Euclidean clusters", "ring center geometry", "balanced cluster sizes"],
        "solver_hint": "Recover the circular cluster order and combine short intra-cluster visits with ring-respecting inter-cluster travel.",
    },
)
