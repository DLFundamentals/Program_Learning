from __future__ import annotations

import math
import random

from dasbench.problems.graph_utils import normalized_edges, validate_edges


def build_graph_instance(instance_id: str, num_vertices: int, edges: list[tuple[int, int]]) -> dict[str, object]:
    normalized = normalized_edges(edges)
    validate_edges(num_vertices, normalized)
    return {
        "id": instance_id,
        "num_vertices": num_vertices,
        "edges": normalized,
    }


def add_edge(edges: set[tuple[int, int]], u: int, v: int) -> None:
    if u == v:
        return
    edges.add((u, v) if u < v else (v, u))


def add_clique(edges: set[tuple[int, int]], vertices: list[int]) -> None:
    for left_index, left in enumerate(vertices):
        for right in vertices[left_index + 1 :]:
            add_edge(edges, left, right)


def add_path(edges: set[tuple[int, int]], vertices: list[int]) -> None:
    for left, right in zip(vertices, vertices[1:], strict=False):
        add_edge(edges, left, right)


def add_cycle(edges: set[tuple[int, int]], vertices: list[int]) -> None:
    if len(vertices) < 3:
        add_path(edges, vertices)
        return
    add_path(edges, vertices)
    add_edge(edges, vertices[0], vertices[-1])


def add_biclique(edges: set[tuple[int, int]], left: list[int], right: list[int]) -> None:
    for left_vertex in left:
        for right_vertex in right:
            add_edge(edges, left_vertex, right_vertex)


def add_crown(edges: set[tuple[int, int]], left: list[int], right: list[int]) -> None:
    if len(left) != len(right):
        raise ValueError("Crown graph requires equal left/right sizes.")
    for left_vertex in left:
        for right_vertex in right:
            if left.index(left_vertex) != right.index(right_vertex):
                add_edge(edges, left_vertex, right_vertex)


def partition_vertices(num_vertices: int, block_sizes: list[int]) -> list[list[int]]:
    if sum(block_sizes) > num_vertices:
        raise ValueError("Block sizes exceed available vertices.")
    blocks: list[list[int]] = []
    cursor = 0
    for size in block_sizes:
        blocks.append(list(range(cursor, cursor + size)))
        cursor += size
    if cursor < num_vertices:
        blocks.append(list(range(cursor, num_vertices)))
    return [block for block in blocks if block]


def sample_block_sizes(rng: random.Random, num_vertices: int, *, min_size: int, max_size: int) -> list[int]:
    sizes: list[int] = []
    remaining = num_vertices
    while remaining > 0:
        if remaining <= max_size:
            sizes.append(remaining)
            break
        size = rng.randint(min_size, max_size)
        sizes.append(size)
        remaining -= size
    return sizes


def add_random_noise_edges(
    rng: random.Random,
    edges: set[tuple[int, int]],
    vertices: list[int],
    *,
    probability: float,
) -> None:
    for left_index, left in enumerate(vertices):
        for right in vertices[left_index + 1 :]:
            if rng.random() < probability:
                add_edge(edges, left, right)


def geometric_points(
    rng: random.Random,
    num_points: int,
    *,
    centers: list[tuple[float, float]],
    spread: float,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(num_points):
        center_x, center_y = centers[index % len(centers)]
        points.append(
            (
                center_x + rng.uniform(-spread, spread),
                center_y + rng.uniform(-spread, spread),
            )
        )
    return points


def add_geometric_edges(
    edges: set[tuple[int, int]],
    points: list[tuple[float, float]],
    *,
    radius: float,
) -> None:
    for left_index, left_point in enumerate(points):
        for right_index in range(left_index + 1, len(points)):
            right_point = points[right_index]
            distance = math.dist(left_point, right_point)
            if distance <= radius:
                add_edge(edges, left_index, right_index)
