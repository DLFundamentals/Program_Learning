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
    block_palettes: list[list[int]]
    regime_pair_biases: list[dict[tuple[int, int], float]]


def build_state(context: dict[str, object]) -> FamilyState:
    num_vertices = int(context["instance_params"]["num_vertices"])
    rng = random.Random(int(context["seeds"]["family"]))
    palette_size = 4
    block_sizes = sample_block_sizes(rng, num_vertices, min_size=4, max_size=6)
    blocks = partition_vertices(num_vertices, block_sizes)
    block_palettes = []
    for block_index, _ in enumerate(blocks):
        offset = block_index % palette_size
        subset = [((offset + shift) % palette_size) for shift in range(3)]
        if block_index % 2 == 0:
            subset.append((offset + 3) % palette_size)
        block_palettes.append(subset)
    regime_pair_biases = [
        {
            (0, 1): 0.82,
            (1, 2): 0.78,
            (2, 3): 0.74,
            (0, 3): 0.52,
            (0, 2): 0.42,
            (1, 3): 0.40,
        },
        {
            (0, 2): 0.80,
            (1, 3): 0.78,
            (0, 1): 0.50,
            (2, 3): 0.48,
            (0, 3): 0.44,
            (1, 2): 0.42,
        },
        {
            (0, 3): 0.82,
            (1, 2): 0.80,
            (0, 1): 0.48,
            (2, 3): 0.50,
            (0, 2): 0.44,
            (1, 3): 0.46,
        },
    ]
    return FamilyState(
        blocks=blocks,
        palette_size=palette_size,
        block_palettes=block_palettes,
        regime_pair_biases=regime_pair_biases,
    )


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    regime = rng.randrange(len(state.regime_pair_biases))
    pair_bias = state.regime_pair_biases[regime]
    edges: set[tuple[int, int]] = set()
    color_map: dict[int, int] = {}

    for block_index, block in enumerate(state.blocks):
        palette = list(state.block_palettes[block_index])
        rng.shuffle(palette)
        block_colors = assign_repeating_colors(block, palette)
        color_map.update(block_colors)
        add_probabilistic_multipartite_edges(
            rng,
            edges,
            block,
            color_map,
            default_probability=0.38,
            pair_bias=pair_bias,
        )

    for block_index, left_block in enumerate(state.blocks[:-1]):
        right_block = state.blocks[block_index + 1]
        bridge_pair = (block_index + regime) % state.palette_size, (block_index + regime + 1) % state.palette_size
        bridge_pair = (min(bridge_pair), max(bridge_pair))
        for left in left_block:
            for right in right_block:
                if color_map[left] == color_map[right]:
                    continue
                pair = (min(color_map[left], color_map[right]), max(color_map[left], color_map[right]))
                probability = 0.12
                if pair == bridge_pair:
                    probability = 0.32
                elif pair in pair_bias:
                    probability = 0.2
                if rng.random() < probability:
                    add_edge(edges, left, right)

    vertices = list(range(int(context["instance_params"]["num_vertices"])))
    for left_index, left in enumerate(vertices):
        for right in vertices[left_index + 1 :]:
            if color_map[left] != color_map[right] and rng.random() < 0.015:
                add_edge(edges, left, right)

    add_palette_anchor_clique(edges, color_map, palette_size=state.palette_size)
    return build_graph_instance(instance_id, int(context["instance_params"]["num_vertices"]), list(edges))


FAMILY = FamilyDefinition(
    problem="coloring",
    name="planted_palette_overlap_v1",
    description=(
        "Paper-grade coloring family with overlapping latent palettes across communities and regime-dependent color-pair interactions."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "A planted 4-coloring spans all blocks; blocks use overlapping shifted palettes, and each instance samples a hidden color-pair density regime.",
        "signals": ["overlapping block palettes", "regime-dependent color-pair biases", "bridge-pair density", "palette anchor clique"],
        "solver_hint": "Recover the planted palette structure and condition on color-pair interaction patterns instead of local degree alone.",
    },
)
