from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.graph_common import add_edge, add_geometric_edges, build_graph_instance, geometric_points


@dataclass(frozen=True)
class FamilyState:
    center_layouts: list[list[tuple[float, float]]]


def build_state(context: dict[str, object]) -> FamilyState:
    rng = random.Random(int(context["seeds"]["family"]))
    layouts = [
        [(-0.8, -0.4), (-0.2, 0.6), (0.7, -0.1)],
        [(-0.7, 0.2), (0.1, -0.6), (0.8, 0.5)],
        [(-0.9, -0.2), (0.0, 0.0), (0.9, 0.2)],
    ]
    rng.shuffle(layouts)
    return FamilyState(center_layouts=layouts)


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    num_vertices = int(context["instance_params"]["num_vertices"])
    regime = rng.randrange(len(state.center_layouts))
    points = geometric_points(
        rng,
        num_vertices,
        centers=state.center_layouts[regime],
        spread=0.28 + 0.04 * regime,
    )
    edges: set[tuple[int, int]] = set()
    add_geometric_edges(edges, points, radius=0.62 - 0.04 * regime)
    for index in range(0, num_vertices - 1, max(3, num_vertices // 6)):
        add_edge(edges, index, min(num_vertices - 1, index + 2))
        if index + 4 < num_vertices and rng.random() < 0.6:
            add_edge(edges, index, index + 4)
    return build_graph_instance(instance_id, num_vertices, list(edges))


FAMILY = FamilyDefinition(
    problem="mds",
    name="geometric_cluster_cover_v1",
    description=(
        "Paper-grade MDS family with clustered random-geometric structure, heterogeneous density, and noisy connector edges."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Each instance samples one of several random-geometric cluster layouts with density heterogeneity and periodic connector edges.",
        "signals": ["geometric clusters", "distance-threshold neighborhoods", "layout-dependent density", "periodic connector edges"],
        "solver_hint": "Infer local geometric neighborhoods and connector roles, then select dominators by marginal coverage and redundancy.",
    },
)
