from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.graph_common import add_edge, build_graph_instance, partition_vertices, sample_block_sizes


@dataclass(frozen=True)
class FamilyState:
    clusters: list[list[int]]


def build_state(context: dict[str, object]) -> FamilyState:
    num_vertices = int(context["instance_params"]["num_vertices"])
    rng = random.Random(int(context["seeds"]["family"]))
    cluster_sizes = sample_block_sizes(rng, num_vertices, min_size=4, max_size=7)
    return FamilyState(clusters=partition_vertices(num_vertices, cluster_sizes))


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    edges: set[tuple[int, int]] = set()
    hubs: list[int] = []
    for cluster in state.clusters:
        hub = cluster[0]
        hubs.append(hub)
        for vertex in cluster[1:]:
            add_edge(edges, hub, vertex)
            if len(cluster) >= 4 and rng.random() < 0.35:
                add_edge(edges, vertex, cluster[min(1, len(cluster) - 1)])
    for left, right in zip(hubs, hubs[1:], strict=False):
        add_edge(edges, left, right)
    if len(hubs) > 2 and rng.random() < 0.7:
        add_edge(edges, hubs[0], hubs[-1])
    return build_graph_instance(
        instance_id,
        int(context["instance_params"]["num_vertices"]),
        list(edges),
    )


FAMILY = FamilyDefinition(
    problem="mds",
    name="star_cluster_cover_v1",
    description="Smoke/debug MDS family with star-like clusters and sparse hub connectors.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Each cluster has a stable hub at the first vertex that dominates most local vertices; hubs are sparsely connected to each other.",
        "signals": ["star-like clusters", "stable cluster hubs", "sparse hub connectors"],
        "solver_hint": "Select the cluster hubs and prune redundant vertices only if connector edges already dominate adjacent clusters.",
    },
)
