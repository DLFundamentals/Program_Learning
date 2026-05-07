from __future__ import annotations

import random
import time
from collections import Counter, deque
from math import ceil

from dasbench.problems.base import ExactSolveResult
from ortools.sat.python import cp_model  # type: ignore


def validate_edges(num_vertices: int, edges: list[list[int]] | list[tuple[int, int]]) -> None:
    seen: set[tuple[int, int]] = set()
    for raw_edge in edges:
        if len(raw_edge) != 2:
            raise ValueError(f"Invalid edge {raw_edge!r}.")
        u, v = (int(raw_edge[0]), int(raw_edge[1]))
        if u == v:
            raise ValueError(f"Self-loop edge {(u, v)!r} is not allowed.")
        if not (0 <= u < num_vertices and 0 <= v < num_vertices):
            raise ValueError(f"Edge {(u, v)!r} is outside 0..{num_vertices - 1}.")
        edge = (u, v) if u < v else (v, u)
        if edge in seen:
            raise ValueError(f"Duplicate edge {edge!r}.")
        seen.add(edge)


def normalized_edges(edges: list[list[int]] | list[tuple[int, int]]) -> list[list[int]]:
    return [
        [u, v] if u < v else [v, u]
        for u, v in sorted({(min(int(a), int(b)), max(int(a), int(b))) for a, b in edges})
    ]


def adjacency_sets(num_vertices: int, edges: list[list[int]] | list[tuple[int, int]]) -> list[set[int]]:
    adjacency = [set() for _ in range(num_vertices)]
    for u_raw, v_raw in edges:
        u, v = int(u_raw), int(v_raw)
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def edge_density(num_vertices: int, edges: list[list[int]] | list[tuple[int, int]]) -> float:
    if num_vertices <= 1:
        return 0.0
    return (2.0 * len(edges)) / (num_vertices * (num_vertices - 1))


def connected_component_sizes(num_vertices: int, adjacency: list[set[int]]) -> list[int]:
    seen = [False] * num_vertices
    sizes: list[int] = []
    for start in range(num_vertices):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        size = 0
        while queue:
            node = queue.popleft()
            size += 1
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def degree_sequence(adjacency: list[set[int]]) -> list[int]:
    return [len(neighbors) for neighbors in adjacency]


def sample_independent_set_random(instance: dict[str, object], *, seed_label: str) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    rng = random.Random(f"{seed_label}:{instance['id']}")
    vertices = list(range(num_vertices))
    rng.shuffle(vertices)
    selected: list[int] = []
    blocked: set[int] = set()
    for vertex in vertices:
        if vertex in blocked:
            continue
        selected.append(vertex)
        blocked.add(vertex)
        blocked.update(adjacency[vertex])
    return sorted(selected)


def greedy_independent_set_by_score(
    instance: dict[str, object],
    *,
    score_fn,
) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    remaining = set(range(num_vertices))
    solution: list[int] = []
    while remaining:
        vertex = min(remaining, key=lambda item: (score_fn(item, adjacency, remaining), item))
        solution.append(vertex)
        remaining.discard(vertex)
        remaining.difference_update(adjacency[vertex])
    return sorted(solution)


def greedy_independent_set_with_local_improvement(instance: dict[str, object]) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    solution = set(
        greedy_independent_set_by_score(
            instance,
            score_fn=lambda vertex, adjacency, remaining: len(adjacency[vertex] & remaining),
        )
    )

    improved = True
    while improved:
        improved = False
        excluded = [vertex for vertex in range(num_vertices) if vertex not in solution]
        for pivot in list(solution):
            candidates = [vertex for vertex in excluded if adjacency[vertex].isdisjoint(solution - {pivot})]
            for index, left in enumerate(candidates):
                if left in adjacency[pivot]:
                    continue
                for right in candidates[index + 1 :]:
                    if right in adjacency[left] or right in adjacency[pivot]:
                        continue
                    solution.remove(pivot)
                    solution.add(left)
                    solution.add(right)
                    improved = True
                    break
                if improved:
                    break
            if improved:
                break
    return sorted(solution)


def greedy_dominating_set_by_score(instance: dict[str, object], *, score_fn) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    closed = [neighbors | {vertex} for vertex, neighbors in enumerate(adjacency)]
    dominated: set[int] = set()
    chosen: list[int] = []
    while len(dominated) < num_vertices:
        candidates = [vertex for vertex in range(num_vertices) if vertex not in chosen]
        vertex = max(
            candidates,
            key=lambda item: (
                score_fn(item, closed, dominated, chosen),
                -item,
            ),
        )
        chosen.append(vertex)
        dominated.update(closed[vertex])
    return sorted(chosen)


def greedy_dominating_set_max_gain(
    instance: dict[str, object],
    *,
    prefer_low_overlap: bool = False,
) -> list[int]:
    import heapq

    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    closed = [tuple(sorted(neighbors | {vertex})) for vertex, neighbors in enumerate(adjacency)]
    dominated = [False] * num_vertices
    selected = [False] * num_vertices
    dominated_count = 0
    chosen: list[int] = []

    def key_for(vertex: int) -> tuple[int, int, int, int]:
        gain = 0
        for item in closed[vertex]:
            if not dominated[item]:
                gain += 1
        overlap = len(closed[vertex]) - gain
        overlap_key = overlap if prefer_low_overlap else 0
        return (-gain, overlap_key, -len(closed[vertex]), vertex)

    heap = [(-len(closed[vertex]), 0, -len(closed[vertex]), vertex) for vertex in range(num_vertices)]
    heapq.heapify(heap)
    while dominated_count < num_vertices:
        if not heap:
            raise RuntimeError("No candidate vertex remained before all vertices were dominated.")
        cached_key = heapq.heappop(heap)
        vertex = cached_key[3]
        if selected[vertex]:
            continue
        current_key = key_for(vertex)
        if current_key != cached_key:
            heapq.heappush(heap, current_key)
            continue
        gain = -current_key[0]
        if gain <= 0:
            continue
        selected[vertex] = True
        chosen.append(vertex)
        for item in closed[vertex]:
            if not dominated[item]:
                dominated[item] = True
                dominated_count += 1
    return sorted(chosen)


def prune_redundant_dominating_vertices(instance: dict[str, object], solution: list[int]) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    closed = [neighbors | {vertex} for vertex, neighbors in enumerate(adjacency)]
    chosen = list(sorted(set(solution)))
    for vertex in list(chosen):
        candidate = [item for item in chosen if item != vertex]
        dominated: set[int] = set()
        for item in candidate:
            dominated.update(closed[item])
        if len(dominated) == num_vertices:
            chosen = candidate
    return sorted(chosen)


def prune_redundant_dominating_vertices_fast(instance: dict[str, object], solution: list[int]) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    closed = [tuple(sorted(neighbors | {vertex})) for vertex, neighbors in enumerate(adjacency)]
    chosen = set(int(vertex) for vertex in solution)
    cover_count = [0] * num_vertices
    for vertex in chosen:
        for item in closed[vertex]:
            cover_count[item] += 1
    for vertex in sorted(chosen, reverse=True):
        if vertex not in chosen:
            continue
        if all(cover_count[item] > 1 for item in closed[vertex]):
            chosen.remove(vertex)
            for item in closed[vertex]:
                cover_count[item] -= 1
    return sorted(chosen)


def is_independent_set(num_vertices: int, edges: list[list[int]], vertices: list[int]) -> tuple[bool, str | None]:
    seen: set[int] = set()
    for vertex in vertices:
        if not 0 <= vertex < num_vertices:
            return False, f"Vertex {vertex} is outside 0..{num_vertices - 1}."
        if vertex in seen:
            return False, f"Vertex {vertex} is repeated."
        seen.add(vertex)
    edge_set = {(u, v) if u < v else (v, u) for u, v in edges}
    for u in seen:
        for v in seen:
            if u < v and (u, v) in edge_set:
                return False, f"Vertices {u} and {v} are adjacent."
    return True, None


def is_dominating_set(num_vertices: int, adjacency: list[set[int]], vertices: list[int]) -> tuple[bool, str | None]:
    seen: set[int] = set()
    for vertex in vertices:
        if not 0 <= vertex < num_vertices:
            return False, f"Vertex {vertex} is outside 0..{num_vertices - 1}."
        if vertex in seen:
            return False, f"Vertex {vertex} is repeated."
        seen.add(vertex)
    dominated = set(seen)
    for vertex in seen:
        dominated.update(adjacency[vertex])
    if len(dominated) != num_vertices:
        missing = sorted(set(range(num_vertices)) - dominated)
        return False, f"Undominated vertices remain: {missing[:5]}"
    return True, None


def canonicalize_coloring(raw_solution, num_vertices: int) -> list[int]:
    if isinstance(raw_solution, dict):
        colors = []
        for vertex in range(num_vertices):
            if vertex in raw_solution:
                value = raw_solution[vertex]
            elif str(vertex) in raw_solution:
                value = raw_solution[str(vertex)]
            else:
                raise TypeError(f"Missing color assignment for vertex {vertex}.")
            colors.append(int(value))
    elif isinstance(raw_solution, (list, tuple)):
        colors = [int(value) for value in raw_solution]
    else:
        raise TypeError("Coloring solver output must be a mapping or sequence of color ids.")
    if len(colors) != num_vertices:
        raise TypeError(f"Coloring solver output must assign exactly {num_vertices} vertices.")
    relabel: dict[int, int] = {}
    canonical: list[int] = []
    next_color = 0
    for color in colors:
        if color not in relabel:
            relabel[color] = next_color
            next_color += 1
        canonical.append(relabel[color])
    return canonical


def is_proper_coloring(
    num_vertices: int,
    edges: list[list[int]] | list[tuple[int, int]],
    colors: list[int],
) -> tuple[bool, str | None]:
    if len(colors) != num_vertices:
        return False, f"Expected {num_vertices} colors, received {len(colors)}."
    for vertex, color in enumerate(colors):
        if color < 0:
            return False, f"Vertex {vertex} has negative color {color}."
    for u_raw, v_raw in edges:
        u, v = int(u_raw), int(v_raw)
        if colors[u] == colors[v]:
            return False, f"Adjacent vertices {u} and {v} share color {colors[u]}."
    return True, None


def _greedy_coloring_from_order(adjacency: list[set[int]], order: list[int]) -> list[int]:
    colors = [-1] * len(adjacency)
    for vertex in order:
        used = {colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0}
        color = 0
        while color in used:
            color += 1
        colors[vertex] = color
    return canonicalize_coloring(colors, len(adjacency))


def greedy_coloring_random_order(instance: dict[str, object], *, seed_label: str) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    order = list(range(num_vertices))
    rng = random.Random(f"{seed_label}:{instance['id']}")
    rng.shuffle(order)
    return _greedy_coloring_from_order(adjacency, order)


def greedy_coloring_largest_degree(instance: dict[str, object]) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    order = sorted(range(num_vertices), key=lambda vertex: (-len(adjacency[vertex]), vertex))
    return _greedy_coloring_from_order(adjacency, order)


def smallest_last_coloring(instance: dict[str, object]) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    remaining = set(range(num_vertices))
    active = [set(neighbors) for neighbors in adjacency]
    elimination_order: list[int] = []
    while remaining:
        vertex = min(remaining, key=lambda item: (len(active[item] & remaining), item))
        elimination_order.append(vertex)
        remaining.remove(vertex)
        for neighbor in adjacency[vertex]:
            active[neighbor].discard(vertex)
    return _greedy_coloring_from_order(adjacency, list(reversed(elimination_order)))


def dsatur_coloring(instance: dict[str, object]) -> list[int]:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    degrees = [len(neighbors) for neighbors in adjacency]
    colors = [-1] * num_vertices
    saturation = [set() for _ in range(num_vertices)]
    for _ in range(num_vertices):
        vertex = max(
            [item for item in range(num_vertices) if colors[item] < 0],
            key=lambda item: (len(saturation[item]), degrees[item], -item),
        )
        used = saturation[vertex]
        color = 0
        while color in used:
            color += 1
        colors[vertex] = color
        for neighbor in adjacency[vertex]:
            if colors[neighbor] < 0:
                saturation[neighbor].add(color)
    return canonicalize_coloring(colors, num_vertices)


def solve_mis_clique_branch_and_bound(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    complement_masks = [0] * num_vertices
    for vertex in range(num_vertices):
        mask = 0
        for other in range(num_vertices):
            if other != vertex and other not in adjacency[vertex]:
                mask |= 1 << other
        complement_masks[vertex] = mask

    best: list[int] = []

    def greedy_coloring_bound(candidates: list[int]) -> tuple[list[int], list[int]]:
        remaining = sorted(candidates, key=lambda item: complement_masks[item].bit_count(), reverse=True)
        ordered: list[int] = []
        bounds: list[int] = []
        color = 0
        while remaining:
            color += 1
            color_class: list[int] = []
            next_remaining: list[int] = []
            for vertex in remaining:
                if all(((complement_masks[vertex] >> item) & 1) == 0 for item in color_class):
                    color_class.append(vertex)
                    ordered.append(vertex)
                    bounds.append(color)
                else:
                    next_remaining.append(vertex)
            remaining = next_remaining
        return ordered, bounds

    def expand(clique: list[int], candidates: list[int]) -> None:
        nonlocal best
        ordered, bounds = greedy_coloring_bound(candidates)
        while ordered:
            if len(clique) + bounds[-1] <= len(best):
                return
            vertex = ordered.pop()
            bounds.pop()
            next_candidates = [item for item in ordered if (complement_masks[vertex] >> item) & 1]
            clique.append(vertex)
            if not next_candidates:
                if len(clique) > len(best):
                    best = list(clique)
            elif len(clique) + len(next_candidates) > len(best):
                expand(clique, next_candidates)
            clique.pop()

    expand([], list(range(num_vertices)))
    solution = sorted(best)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return ExactSolveResult(
        solution=solution,
        objective_value=float(len(solution)),
        runtime_ms=runtime_ms,
        source="clique-branch-and-bound",
    )


def solve_mds_set_cover_branch_and_bound(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    closed_masks = []
    for vertex, neighbors in enumerate(adjacency):
        mask = 1 << vertex
        for neighbor in neighbors:
            mask |= 1 << neighbor
        closed_masks.append(mask)
    all_mask = (1 << num_vertices) - 1
    incumbent = prune_redundant_dominating_vertices(
        instance,
        greedy_dominating_set_by_score(
            instance,
            score_fn=lambda vertex, closed, dominated, chosen: len(closed[vertex] - dominated),
        ),
    )
    best = list(incumbent)

    def uncovered_vertices(mask: int) -> list[int]:
        return [vertex for vertex in range(num_vertices) if (mask >> vertex) & 1]

    def search(chosen: list[int], chosen_mask: int, dominated_mask: int) -> None:
        nonlocal best
        if len(chosen) >= len(best):
            return
        if dominated_mask == all_mask:
            best = sorted(chosen)
            return

        uncovered_mask = all_mask ^ dominated_mask
        max_gain = max((closed_masks[vertex] & uncovered_mask).bit_count() for vertex in range(num_vertices))
        if max_gain <= 0 or len(chosen) + ceil(uncovered_mask.bit_count() / max_gain) >= len(best):
            return

        target = min(
            uncovered_vertices(uncovered_mask),
            key=lambda item: (
                sum(1 for vertex in range(num_vertices) if not ((chosen_mask >> vertex) & 1) and ((closed_masks[vertex] >> item) & 1)),
                item,
            ),
        )
        candidates = [
            vertex
            for vertex in range(num_vertices)
            if not ((chosen_mask >> vertex) & 1) and ((closed_masks[vertex] >> target) & 1)
        ]
        candidates.sort(key=lambda item: ((closed_masks[item] & uncovered_mask).bit_count(), -item), reverse=True)
        for vertex in candidates:
            search(
                [*chosen, vertex],
                chosen_mask | (1 << vertex),
                dominated_mask | closed_masks[vertex],
            )

    search([], 0, 0)
    solution = sorted(best)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return ExactSolveResult(
        solution=solution,
        objective_value=float(len(solution)),
        runtime_ms=runtime_ms,
        source="set-cover-branch-and-bound",
    )


def solve_coloring_dsatur_exact(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    degrees = [len(neighbors) for neighbors in adjacency]
    best_coloring = dsatur_coloring(instance)
    best_count = len(set(best_coloring))
    colors = [-1] * num_vertices

    def choose_vertex() -> int:
        return max(
            [vertex for vertex in range(num_vertices) if colors[vertex] < 0],
            key=lambda vertex: (
                len({colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0}),
                degrees[vertex],
                -vertex,
            ),
        )

    def search(colored_count: int, used_count: int) -> None:
        nonlocal best_coloring, best_count
        if used_count >= best_count:
            return
        if colored_count == num_vertices:
            best_coloring = canonicalize_coloring(colors, num_vertices)
            best_count = len(set(best_coloring))
            return

        vertex = choose_vertex()
        forbidden = {colors[neighbor] for neighbor in adjacency[vertex] if colors[neighbor] >= 0}
        for color in range(min(used_count + 1, best_count)):
            if color in forbidden:
                continue
            next_used = max(used_count, color + 1)
            if next_used >= best_count:
                continue
            colors[vertex] = color
            search(colored_count + 1, next_used)
            colors[vertex] = -1

    search(0, 0)
    canonical = canonicalize_coloring(best_coloring, num_vertices)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return ExactSolveResult(
        solution=canonical,
        objective_value=float(len(set(canonical))),
        runtime_ms=runtime_ms,
        source="dsatur-branch-and-bound",
    )


def solve_mis_exact(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    edges = [[int(a), int(b)] for a, b in instance["edges"]]
    model = cp_model.CpModel()
    variables = [model.NewBoolVar(f"x_{vertex}") for vertex in range(num_vertices)]
    for u, v in edges:
        model.Add(variables[u] + variables[v] <= 1)
    model.Maximize(sum(variables))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT failed to solve MIS instance.")
    solution = [vertex for vertex in range(num_vertices) if solver.Value(variables[vertex])]
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return ExactSolveResult(
        solution=sorted(solution),
        objective_value=float(len(solution)),
        runtime_ms=runtime_ms,
        source="ortools-cpsat",
    )


def solve_coloring_exact(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    edges = [[int(a), int(b)] for a, b in instance["edges"]]
    heuristic_solution = dsatur_coloring(instance)
    heuristic_upper_bound = len(set(heuristic_solution))

    model = cp_model.CpModel()
    assignment = [
        [model.NewBoolVar(f"x_{vertex}_{color}") for color in range(num_vertices)]
        for vertex in range(num_vertices)
    ]
    used_color = [model.NewBoolVar(f"used_{color}") for color in range(num_vertices)]

    for vertex in range(num_vertices):
        model.AddExactlyOne(assignment[vertex][color] for color in range(num_vertices))
    for color in range(num_vertices - 1):
        model.Add(used_color[color] >= used_color[color + 1])
    for vertex in range(num_vertices):
        for color in range(num_vertices):
            model.Add(assignment[vertex][color] <= used_color[color])
    for u, v in edges:
        for color in range(num_vertices):
            model.Add(assignment[u][color] + assignment[v][color] <= 1)
    model.Add(sum(used_color) <= heuristic_upper_bound)
    if num_vertices > 0:
        model.Add(assignment[0][0] == 1)
    model.Minimize(sum(used_color))

    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT failed to solve coloring instance.")
    solution = []
    for vertex in range(num_vertices):
        color = next(color for color in range(num_vertices) if solver.Value(assignment[vertex][color]))
        solution.append(color)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    canonical = canonicalize_coloring(solution, num_vertices)
    return ExactSolveResult(
        solution=canonical,
        objective_value=float(len(set(canonical))),
        runtime_ms=runtime_ms,
        source="ortools-cpsat",
    )


def solve_mds_exact(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    edges = [[int(a), int(b)] for a, b in instance["edges"]]
    adjacency = adjacency_sets(num_vertices, edges)
    closed = [neighbors | {vertex} for vertex, neighbors in enumerate(adjacency)]
    model = cp_model.CpModel()
    variables = [model.NewBoolVar(f"x_{vertex}") for vertex in range(num_vertices)]
    for vertex in range(num_vertices):
        model.Add(sum(variables[item] for item in closed[vertex]) >= 1)
    model.Minimize(sum(variables))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT failed to solve MDS instance.")
    solution = [vertex for vertex in range(num_vertices) if solver.Value(variables[vertex])]
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return ExactSolveResult(
        solution=sorted(solution),
        objective_value=float(len(solution)),
        runtime_ms=runtime_ms,
        source="ortools-cpsat",
    )


def degree_histogram(adjacency: list[set[int]]) -> dict[str, int]:
    counts = Counter(len(neighbors) for neighbors in adjacency)
    return {str(degree): counts[degree] for degree in sorted(counts)}
