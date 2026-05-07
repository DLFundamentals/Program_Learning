from __future__ import annotations

import random

from dasbench.families.graph_common import add_clique, add_edge


def assign_repeating_colors(vertices: list[int], palette_order: list[int]) -> dict[int, int]:
    return {
        vertex: int(palette_order[index % len(palette_order)])
        for index, vertex in enumerate(vertices)
    }


def add_probabilistic_multipartite_edges(
    rng: random.Random,
    edges: set[tuple[int, int]],
    vertices: list[int],
    color_map: dict[int, int],
    *,
    default_probability: float,
    pair_bias: dict[tuple[int, int], float] | None = None,
) -> None:
    pair_bias = pair_bias or {}
    for left_index, left in enumerate(vertices):
        for right in vertices[left_index + 1 :]:
            left_color = color_map[left]
            right_color = color_map[right]
            if left_color == right_color:
                continue
            pair = (min(left_color, right_color), max(left_color, right_color))
            probability = pair_bias.get(pair, default_probability)
            if rng.random() < probability:
                add_edge(edges, left, right)


def add_palette_anchor_clique(
    edges: set[tuple[int, int]],
    color_map: dict[int, int],
    *,
    palette_size: int,
) -> None:
    representatives: list[int] = []
    for color in range(palette_size):
        vertex = next((item for item, assigned in color_map.items() if assigned == color), None)
        if vertex is None:
            return
        representatives.append(vertex)
    add_clique(edges, representatives)
