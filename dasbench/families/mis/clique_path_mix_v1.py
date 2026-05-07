from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.graph_common import (
    add_clique,
    add_edge,
    add_path,
    add_random_noise_edges,
    build_graph_instance,
    partition_vertices,
    sample_block_sizes,
)


@dataclass(frozen=True)
class FamilyState:
    blocks: list[list[int]]


def build_state(context: dict[str, object]) -> FamilyState:
    params = context["instance_params"]
    seeds = context["seeds"]
    num_vertices = int(params["num_vertices"])
    rng = random.Random(int(seeds["family"]))
    block_sizes = sample_block_sizes(rng, num_vertices, min_size=4, max_size=6)
    return FamilyState(blocks=partition_vertices(num_vertices, block_sizes))


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    regime = rng.randrange(2)
    edges: set[tuple[int, int]] = set()
    for block_index, block in enumerate(state.blocks):
        if (block_index + regime) % 2 == 0:
            add_clique(edges, block)
        else:
            add_path(edges, block)
    for left_block, right_block in zip(state.blocks, state.blocks[1:], strict=False):
        if left_block and right_block:
            add_edge(edges, left_block[-1], right_block[0])
            if len(left_block) > 1 and len(right_block) > 1 and rng.random() < 0.6:
                add_edge(edges, left_block[-2], right_block[1])
    add_random_noise_edges(
        rng,
        edges,
        list(range(int(context["instance_params"]["num_vertices"]))),
        probability=0.035,
    )
    return build_graph_instance(
        instance_id,
        int(context["instance_params"]["num_vertices"]),
        list(edges),
    )


FAMILY = FamilyDefinition(
    problem="mis",
    name="clique_path_mix_v1",
    description=(
        "Smoke/debug MIS family with alternating clique-heavy and path-heavy regions plus light bridge noise."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "A hidden two-way regime alternates which block parity is clique-like versus path-like, with sparse bridges and noise.",
        "signals": ["alternating clique/path blocks", "bridge edges between adjacent blocks", "light random noise"],
        "solver_hint": "Classify block motifs, take at most one vertex from clique blocks, and use path-style alternating selections where possible.",
    },
)
