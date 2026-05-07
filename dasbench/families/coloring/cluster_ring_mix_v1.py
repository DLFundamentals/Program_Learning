from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.coloring_common import (
    add_palette_anchor_clique,
    add_probabilistic_multipartite_edges,
    assign_repeating_colors,
)
from dasbench.families.graph_common import add_edge, build_graph_instance, partition_vertices, sample_block_sizes


@dataclass(frozen=True)
class FamilyState:
    blocks: list[list[int]]
    palette_size: int
    block_permutations: list[list[int]]


def build_state(context: dict[str, object]) -> FamilyState:
    num_vertices = int(context["instance_params"]["num_vertices"])
    rng = random.Random(int(context["seeds"]["family"]))
    palette_size = 4 if num_vertices >= 10 else 3
    block_sizes = sample_block_sizes(rng, num_vertices, min_size=palette_size, max_size=palette_size + 2)
    blocks = partition_vertices(num_vertices, block_sizes)
    block_permutations = []
    for _ in blocks:
        permutation = list(range(palette_size))
        rng.shuffle(permutation)
        block_permutations.append(permutation)
    return FamilyState(blocks=blocks, palette_size=palette_size, block_permutations=block_permutations)


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    edges: set[tuple[int, int]] = set()
    color_map: dict[int, int] = {}
    for block, permutation in zip(state.blocks, state.block_permutations, strict=True):
        block_colors = assign_repeating_colors(block, permutation)
        color_map.update(block_colors)
        add_probabilistic_multipartite_edges(
            rng,
            edges,
            block,
            color_map,
            default_probability=0.86,
        )

    for block_index, left_block in enumerate(state.blocks):
        right_block = state.blocks[(block_index + 1) % len(state.blocks)]
        for left_offset, left in enumerate(left_block):
            right = right_block[left_offset % len(right_block)]
            if color_map[left] != color_map[right]:
                add_edge(edges, left, right)
            if rng.random() < 0.35:
                alternate = right_block[(left_offset + 1) % len(right_block)]
                if color_map[left] != color_map[alternate]:
                    add_edge(edges, left, alternate)

    vertices = list(range(int(context["instance_params"]["num_vertices"])))
    for left_index, left in enumerate(vertices):
        for right in vertices[left_index + 1 :]:
            if color_map[left] != color_map[right] and rng.random() < 0.02:
                add_edge(edges, left, right)

    add_palette_anchor_clique(edges, color_map, palette_size=state.palette_size)
    return build_graph_instance(instance_id, int(context["instance_params"]["num_vertices"]), list(edges))


FAMILY = FamilyDefinition(
    problem="coloring",
    name="cluster_ring_mix_v1",
    description="Smoke coloring family with cluster-local multipartite structure and ring-style bridges.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Blocks have planted palette permutations; edges mostly connect different planted colors, with ring bridges preserving the same coloring.",
        "signals": ["multipartite block structure", "ring bridge pattern", "palette anchor clique"],
        "solver_hint": "Infer a stable block palette/order and reuse it before falling back to generic coloring.",
    },
)
