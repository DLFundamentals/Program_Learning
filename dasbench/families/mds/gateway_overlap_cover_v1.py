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
    cluster_sizes = sample_block_sizes(rng, num_vertices, min_size=5, max_size=8)
    return FamilyState(clusters=partition_vertices(num_vertices, cluster_sizes))


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    edges: set[tuple[int, int]] = set()
    cluster_hubs: list[int] = []
    gateways: list[int] = []
    for cluster in state.clusters:
        hub = cluster[0]
        gateway = cluster[1] if len(cluster) > 1 else cluster[0]
        cluster_hubs.append(hub)
        gateways.append(gateway)
        for vertex in cluster[2:]:
            add_edge(edges, hub, vertex)
            if rng.random() < 0.55:
                add_edge(edges, gateway, vertex)
        add_edge(edges, hub, gateway)
    for index in range(len(state.clusters) - 1):
        left_cluster = state.clusters[index]
        right_cluster = state.clusters[index + 1]
        gateway = gateways[index]
        next_gateway = gateways[index + 1]
        add_edge(edges, gateway, next_gateway)
        for vertex in right_cluster[2 : min(5, len(right_cluster))]:
            add_edge(edges, gateway, vertex)
        for vertex in left_cluster[2 : min(5, len(left_cluster))]:
            add_edge(edges, next_gateway, vertex)
    return build_graph_instance(
        instance_id,
        int(context["instance_params"]["num_vertices"]),
        list(edges),
    )


FAMILY = FamilyDefinition(
    problem="mds",
    name="gateway_overlap_cover_v1",
    description=(
        "Paper-grade MDS family where hidden gateway vertices create overlapping coverage across neighboring clusters."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Each cluster has a hub and a gateway; gateways overlap coverage across neighboring clusters and link to neighboring gateways.",
        "signals": ["cluster hubs", "gateway vertices", "overlapping neighbor-cluster coverage", "gateway chain"],
        "solver_hint": "Choose dominators by marginal overlap-aware coverage, combining hubs and gateways rather than raw degree alone.",
    },
)
