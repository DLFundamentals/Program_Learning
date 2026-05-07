from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.graph_common import (
    add_clique,
    add_cycle,
    add_edge,
    add_path,
    add_random_noise_edges,
    build_graph_instance,
)


@dataclass(frozen=True)
class FamilyState:
    core_vertices: list[int]
    fringe_groups: list[list[int]]


def build_state(context: dict[str, object]) -> FamilyState:
    num_vertices = int(context["instance_params"]["num_vertices"])
    core_size = max(6, num_vertices // 3)
    core_vertices = list(range(core_size))
    fringe_groups: list[list[int]] = []
    cursor = core_size
    while cursor < num_vertices:
        group = list(range(cursor, min(cursor + 4, num_vertices)))
        fringe_groups.append(group)
        cursor += 4
    return FamilyState(core_vertices=core_vertices, fringe_groups=fringe_groups)


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    edges: set[tuple[int, int]] = set()
    regime = rng.randrange(3)
    add_clique(edges, state.core_vertices)
    if regime >= 1:
        for index in range(0, len(state.core_vertices) - 1, 3):
            add_edge(edges, state.core_vertices[index], state.core_vertices[min(index + 2, len(state.core_vertices) - 1)])
    for group_index, group in enumerate(state.fringe_groups):
        if len(group) == 1:
            add_edge(edges, group[0], state.core_vertices[group_index % len(state.core_vertices)])
            continue
        gadget_type = (group_index + regime) % 3
        if gadget_type == 0:
            add_path(edges, group)
        elif gadget_type == 1:
            add_cycle(edges, group)
        else:
            add_path(edges, group[:3])
            if len(group) == 4:
                add_edge(edges, group[0], group[3])
                add_edge(edges, group[1], group[3])
        anchor = state.core_vertices[group_index % len(state.core_vertices)]
        add_edge(edges, anchor, group[0])
        add_edge(edges, anchor, group[-1])
        if len(group) >= 3:
            secondary = state.core_vertices[(group_index * 2 + regime) % len(state.core_vertices)]
            add_edge(edges, secondary, group[1])
    add_random_noise_edges(
        rng,
        edges,
        state.core_vertices,
        probability=0.08 if regime == 2 else 0.03,
    )
    return build_graph_instance(
        instance_id,
        int(context["instance_params"]["num_vertices"]),
        list(edges),
    )


FAMILY = FamilyDefinition(
    problem="mis",
    name="core_fringe_trap_v1",
    description=(
        "Paper-grade MIS family with dense cores and fringe gadgets whose local degree signals can be misleading under hidden attachment regimes."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "A dense core is coupled to regime-dependent low-degree fringe gadgets; core degree signals can be anti-correlated with optimal choices.",
        "signals": ["core clique", "fringe path/cycle/trap gadgets", "regime-dependent core attachments", "core noise edges"],
        "solver_hint": "Prefer compatible fringe selections and handle core attachments globally instead of greedily choosing by degree.",
    },
)
