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
    gap_pairs: list[tuple[int, int]]


def build_state(context: dict[str, object]) -> FamilyState:
    num_vertices = int(context["instance_params"]["num_vertices"])
    rng = random.Random(int(context["seeds"]["family"]))
    palette_size = 4
    block_sizes = sample_block_sizes(rng, num_vertices, min_size=4, max_size=7)
    blocks = partition_vertices(num_vertices, block_sizes)
    block_permutations = []
    gap_pairs = []
    for block_index, _ in enumerate(blocks):
        permutation = list(range(palette_size))
        rng.shuffle(permutation)
        block_permutations.append(permutation)
        gap_left = block_index % palette_size
        gap_right = (gap_left + 2) % palette_size
        gap_pairs.append((min(gap_left, gap_right), max(gap_left, gap_right)))
    return FamilyState(
        blocks=blocks,
        palette_size=palette_size,
        block_permutations=block_permutations,
        gap_pairs=gap_pairs,
    )


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    edges: set[tuple[int, int]] = set()
    color_map: dict[int, int] = {}
    for block, permutation, gap_pair in zip(state.blocks, state.block_permutations, state.gap_pairs, strict=True):
        block_colors = assign_repeating_colors(block, permutation)
        color_map.update(block_colors)
        add_probabilistic_multipartite_edges(
            rng,
            edges,
            block,
            color_map,
            default_probability=0.76,
            pair_bias={gap_pair: 0.14},
        )

    for block_index, left_block in enumerate(state.blocks[:-1]):
        right_block = state.blocks[block_index + 1]
        left_separator = left_block[0]
        right_separator = right_block[-1]
        exempt_left_color = (block_index + 1) % state.palette_size
        exempt_right_color = (block_index + 2) % state.palette_size
        for vertex in right_block:
            if color_map[left_separator] != color_map[vertex] and color_map[vertex] != exempt_left_color:
                add_edge(edges, left_separator, vertex)
        for vertex in left_block:
            if color_map[right_separator] != color_map[vertex] and color_map[vertex] != exempt_right_color:
                add_edge(edges, right_separator, vertex)
        for left in left_block[1:]:
            for right in right_block[:-1]:
                if color_map[left] != color_map[right] and rng.random() < 0.16:
                    add_edge(edges, left, right)

    vertices = list(range(int(context["instance_params"]["num_vertices"])))
    for left_index, left in enumerate(vertices):
        for right in vertices[left_index + 1 :]:
            if color_map[left] != color_map[right] and rng.random() < 0.012:
                add_edge(edges, left, right)

    add_palette_anchor_clique(edges, color_map, palette_size=state.palette_size)
    return build_graph_instance(instance_id, int(context["instance_params"]["num_vertices"]), list(edges))


FAMILY = FamilyDefinition(
    problem="coloring",
    name="separator_palette_trap_v1",
    description=(
        "Paper-grade coloring family with locally ambiguous palette gaps and separator vertices that create long-range color reuse traps."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Each block has a planted 4-color permutation with sparse local gap pairs; separator vertices create long-range color-reuse constraints.",
        "signals": ["block-local palette gaps", "separator vertices", "exempt colors across adjacent blocks", "palette anchor clique"],
        "solver_hint": "Infer the global palette and separator constraints, then reuse a static or near-static 4-coloring when valid.",
    },
)
