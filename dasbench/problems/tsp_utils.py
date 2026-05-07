from __future__ import annotations

import hashlib
import itertools
import math
import os
import random
import re
import subprocess
import tempfile
import time
from pathlib import Path

from dasbench.problems.base import ExactSolveResult, SolveOutcome


DEFAULT_LKH_BINARY = Path(__file__).resolve().parents[2] / "baselines" / "bin" / "lkh"


def resolve_lkh_binary() -> Path:
    configured = os.environ.get("DASBENCH_LKH_BIN")
    path = Path(configured).expanduser() if configured else DEFAULT_LKH_BINARY
    if not path.exists():
        raise FileNotFoundError(
            f"LKH binary not found at `{path}`. Set DASBENCH_LKH_BIN or install it under baselines/bin/lkh."
        )
    if not path.is_file():
        raise FileNotFoundError(f"LKH path `{path}` is not a file.")
    return path


def rounded_points(points: list[tuple[float, float]] | list[list[float]]) -> list[list[float]]:
    return [[round(float(x), 6), round(float(y), 6)] for x, y in points]


def euclidean_distance(point_a: tuple[float, float], point_b: tuple[float, float]) -> float:
    return math.dist(point_a, point_b)


def distance_matrix(points: list[tuple[float, float]] | list[list[float]]) -> list[list[float]]:
    normalized = [(float(x), float(y)) for x, y in points]
    matrix = [[0.0 for _ in normalized] for _ in normalized]
    for left_index, left in enumerate(normalized):
        for right_index in range(left_index + 1, len(normalized)):
            value = euclidean_distance(left, normalized[right_index])
            matrix[left_index][right_index] = value
            matrix[right_index][left_index] = value
    return matrix


def serialize_tsplib_explicit(instance: dict[str, object], *, scale: int = 10_000) -> str:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    lines = [
        f"NAME: {instance.get('id', 'dasbench_tsp')}",
        "TYPE: TSP",
        f"DIMENSION: {num_cities}",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    for row in matrix:
        lines.append(" ".join(str(int(round(value * scale))) for value in row))
    lines.append("EOF")
    return "\n".join(lines) + "\n"


def canonicalize_tour(raw_solution, num_cities: int) -> list[int]:
    if not isinstance(raw_solution, (list, tuple)):
        raise TypeError("TSP solver output must be an ordered sequence of city ids.")
    tour = [int(value) for value in raw_solution]
    if len(tour) == num_cities + 1 and tour and tour[0] == tour[-1]:
        tour = tour[:-1]
    if len(tour) != num_cities:
        raise TypeError(f"TSP solver output must contain exactly {num_cities} cities.")
    if 0 in tour:
        pivot = tour.index(0)
        rotated = tour[pivot:] + tour[:pivot]
        reversed_cycle = [rotated[0]] + list(reversed(rotated[1:]))
        return rotated if rotated <= reversed_cycle else reversed_cycle
    return tour


def is_valid_tour(num_cities: int, tour: list[int]) -> tuple[bool, str | None]:
    if len(tour) != num_cities:
        return False, f"Expected {num_cities} cities, received {len(tour)}."
    seen: set[int] = set()
    for city in tour:
        if not 0 <= city < num_cities:
            return False, f"City {city} is outside 0..{num_cities - 1}."
        if city in seen:
            return False, f"City {city} is repeated."
        seen.add(city)
    return True, None


def parse_lkh_tour(text: str, *, num_cities: int) -> list[int]:
    values: list[int] = []
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("TOUR_SECTION"):
            in_section = True
            continue
        if not in_section:
            continue
        if upper.startswith("EOF"):
            break
        for token in re.findall(r"-?\d+", line):
            value = int(token)
            if value == -1:
                in_section = False
                break
            values.append(value - 1)
        if not in_section:
            break
    if len(values) != num_cities:
        raise ValueError(f"LKH tour output contained {len(values)} cities, expected {num_cities}.")
    return canonicalize_tour(values, num_cities)


def _stable_lkh_seed(instance: dict[str, object]) -> int:
    digest = hashlib.sha256(str(instance.get("id", "dasbench-tsp")).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 2_147_483_647 or 1


def _lkh_parameter_text(
    problem_path: Path,
    output_tour_path: Path,
    *,
    runs: int,
    max_trials: int,
    seed: int,
    time_limit_seconds: float | None,
) -> str:
    lines = [
        f"PROBLEM_FILE = {problem_path.name}",
        f"OUTPUT_TOUR_FILE = {output_tour_path.name}",
        "MOVE_TYPE = 5",
        "PATCHING_C = 3",
        "PATCHING_A = 2",
        f"RUNS = {runs}",
        f"MAX_TRIALS = {max_trials}",
        f"SEED = {seed}",
        "TRACE_LEVEL = 0",
    ]
    if time_limit_seconds is not None:
        lines.append(f"TIME_LIMIT = {max(1, math.ceil(time_limit_seconds))}")
    return "\n".join(lines) + "\n"


def lkh_tour(
    instance: dict[str, object],
    *,
    runs: int = 1,
    max_trials: int | None = None,
    time_limit_seconds: float | None = None,
) -> SolveOutcome:
    num_cities = int(instance["num_cities"])
    if num_cities <= 1:
        solution = [0] if num_cities == 1 else []
        return SolveOutcome(solution=solution, metadata={"solver_status": "ok", "lkh_runs": 0})

    binary = resolve_lkh_binary()
    seed = _stable_lkh_seed(instance)
    trials = int(max_trials) if max_trials is not None else max(20, num_cities)
    with tempfile.TemporaryDirectory(prefix="dasbench-lkh-") as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        problem_path = temp_dir / "instance.tsp"
        output_tour_path = temp_dir / "instance.tour"
        parameter_path = temp_dir / "instance.par"
        problem_path.write_text(serialize_tsplib_explicit(instance), encoding="utf-8")
        parameter_path.write_text(
            _lkh_parameter_text(
                problem_path,
                output_tour_path,
                runs=max(1, int(runs)),
                max_trials=max(1, trials),
                seed=seed,
                time_limit_seconds=time_limit_seconds,
            ),
            encoding="utf-8",
        )
        start = time.perf_counter()
        try:
            completed = subprocess.run(
                [str(binary), str(parameter_path)],
                cwd=temp_dir,
                check=False,
                capture_output=True,
                text=True,
                timeout=None if time_limit_seconds is None else max(1.0, float(time_limit_seconds) + 2.0),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"LKH timed out after {time_limit_seconds:.1f}s.") from exc
        runtime_ms = (time.perf_counter() - start) * 1000.0
        if not output_tour_path.exists():
            combined = "\n".join(part for part in [completed.stdout, completed.stderr] if part)
            raise RuntimeError(
                f"LKH did not produce a tour file (returncode={completed.returncode}). Output:\n{combined.strip()}"
            )
        tour = parse_lkh_tour(output_tour_path.read_text(encoding="utf-8"), num_cities=num_cities)
        return SolveOutcome(
            solution=tour,
            metadata={
                "solver_status": "ok" if completed.returncode == 0 else f"exit_{completed.returncode}",
                "lkh_binary": str(binary),
                "lkh_runs": max(1, int(runs)),
                "lkh_max_trials": max(1, trials),
                "lkh_seed": seed,
                "lkh_runtime_ms": runtime_ms,
            },
        )


def tour_length(points: list[tuple[float, float]] | list[list[float]], tour: list[int]) -> float:
    matrix = distance_matrix(points)
    return tour_length_from_matrix(matrix, tour)


def tour_length_from_matrix(matrix: list[list[float]], tour: list[int]) -> float:
    total = 0.0
    for left, right in zip(tour, tour[1:], strict=False):
        total += matrix[left][right]
    total += matrix[tour[-1]][tour[0]]
    return total


def _nearest_city(current: int, remaining: set[int], matrix: list[list[float]]) -> int:
    return min(remaining, key=lambda city: (matrix[current][city], city))


def nearest_neighbor_tour(
    instance: dict[str, object],
    *,
    start_city: int = 0,
) -> list[int]:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    remaining = set(range(num_cities))
    current = start_city
    tour = [current]
    remaining.remove(current)
    while remaining:
        current = _nearest_city(current, remaining, matrix)
        tour.append(current)
        remaining.remove(current)
    return canonicalize_tour(tour, num_cities)


def nearest_insertion_tour(instance: dict[str, object]) -> list[int]:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    if num_cities <= 2:
        return canonicalize_tour(list(range(num_cities)), num_cities)
    second = min(range(1, num_cities), key=lambda city: (matrix[0][city], city))
    cycle = [0, second]
    remaining = set(range(num_cities)) - set(cycle)
    while remaining:
        city = min(
            remaining,
            key=lambda item: (
                min(matrix[item][anchor] for anchor in cycle),
                item,
            ),
        )
        best_index = 0
        best_delta = math.inf
        for index in range(len(cycle)):
            left = cycle[index]
            right = cycle[(index + 1) % len(cycle)]
            delta = matrix[left][city] + matrix[city][right] - matrix[left][right]
            if delta < best_delta - 1e-9 or (abs(delta - best_delta) < 1e-9 and index < best_index):
                best_delta = delta
                best_index = index + 1
        cycle.insert(best_index, city)
        remaining.remove(city)
    return canonicalize_tour(cycle, num_cities)


def farthest_insertion_tour(instance: dict[str, object]) -> list[int]:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    if num_cities <= 2:
        return canonicalize_tour(list(range(num_cities)), num_cities)
    second = max(range(1, num_cities), key=lambda city: (matrix[0][city], -city))
    cycle = [0, second]
    remaining = set(range(num_cities)) - set(cycle)
    while remaining:
        city = max(
            remaining,
            key=lambda item: (
                min(matrix[item][anchor] for anchor in cycle),
                -item,
            ),
        )
        best_index = 0
        best_delta = math.inf
        for index in range(len(cycle)):
            left = cycle[index]
            right = cycle[(index + 1) % len(cycle)]
            delta = matrix[left][city] + matrix[city][right] - matrix[left][right]
            if delta < best_delta - 1e-9 or (abs(delta - best_delta) < 1e-9 and index < best_index):
                best_delta = delta
                best_index = index + 1
        cycle.insert(best_index, city)
        remaining.remove(city)
    return canonicalize_tour(cycle, num_cities)


def two_opt_improve(
    instance: dict[str, object],
    tour: list[int],
    *,
    max_rounds: int = 4,
) -> list[int]:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    best = canonicalize_tour(tour, num_cities)
    best_length = tour_length_from_matrix(matrix, best)
    for _ in range(max_rounds):
        improved = False
        for left_index in range(1, num_cities - 1):
            for right_index in range(left_index + 1, num_cities):
                if right_index - left_index == 1:
                    continue
                candidate = best[:left_index] + list(reversed(best[left_index:right_index])) + best[right_index:]
                candidate = canonicalize_tour(candidate, num_cities)
                candidate_length = tour_length_from_matrix(matrix, candidate)
                if candidate_length + 1e-9 < best_length:
                    best = candidate
                    best_length = candidate_length
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best


def random_tour(instance: dict[str, object], *, seed_label: str) -> list[int]:
    num_cities = int(instance["num_cities"])
    tour = list(range(num_cities))
    rng = random.Random(f"{seed_label}:{instance['id']}")
    if num_cities > 1:
        middle = tour[1:]
        rng.shuffle(middle)
        tour = [0] + middle
    return canonicalize_tour(tour, num_cities)


def multi_start_two_opt_tour(
    instance: dict[str, object],
    *,
    max_rounds: int = 20,
    random_starts: int = 8,
) -> list[int]:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    candidates = [
        nearest_insertion_tour(instance),
        farthest_insertion_tour(instance),
    ]
    candidates.extend(nearest_neighbor_tour(instance, start_city=start) for start in range(num_cities))
    for index in range(random_starts):
        candidates.append(random_tour(instance, seed_label=f"tsp-multistart-{index}"))

    best = canonicalize_tour(candidates[0], num_cities)
    best_length = tour_length_from_matrix(matrix, best)
    for candidate in candidates:
        improved = two_opt_improve(instance, candidate, max_rounds=max_rounds)
        candidate_length = tour_length_from_matrix(matrix, improved)
        if candidate_length + 1e-9 < best_length:
            best = improved
            best_length = candidate_length
    return canonicalize_tour(best, num_cities)


def solve_tsp_exact(instance: dict[str, object]) -> ExactSolveResult:
    start = time.perf_counter()
    num_cities = int(instance["num_cities"])
    points = instance["points"]
    matrix = distance_matrix(points)

    if num_cities <= 1:
        runtime_ms = (time.perf_counter() - start) * 1000.0
        return ExactSolveResult(solution=[0] if num_cities == 1 else [], objective_value=0.0, runtime_ms=runtime_ms, source="held-karp")

    full_mask = (1 << (num_cities - 1)) - 1
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}
    for city in range(1, num_cities):
        mask = 1 << (city - 1)
        dp[(mask, city)] = matrix[0][city]
        parent[(mask, city)] = 0

    for subset_size in range(2, num_cities):
        next_dp: dict[tuple[int, int], float] = {}
        for subset in itertools.combinations(range(1, num_cities), subset_size):
            mask = 0
            for city in subset:
                mask |= 1 << (city - 1)
            for end_city in subset:
                prev_mask = mask ^ (1 << (end_city - 1))
                best_cost = math.inf
                best_parent = 0
                for previous_city in subset:
                    if previous_city == end_city:
                        continue
                    candidate_cost = dp[(prev_mask, previous_city)] + matrix[previous_city][end_city]
                    if candidate_cost < best_cost - 1e-9 or (
                        abs(candidate_cost - best_cost) < 1e-9 and previous_city < best_parent
                    ):
                        best_cost = candidate_cost
                        best_parent = previous_city
                next_dp[(mask, end_city)] = best_cost
                parent[(mask, end_city)] = best_parent
        dp = next_dp

    best_end = min(
        range(1, num_cities),
        key=lambda city: (dp[(full_mask, city)] + matrix[city][0], city),
    )
    best_length = dp[(full_mask, best_end)] + matrix[best_end][0]

    reversed_path: list[int] = []
    mask = full_mask
    end_city = best_end
    while mask:
        reversed_path.append(end_city)
        previous_city = parent[(mask, end_city)]
        mask ^= 1 << (end_city - 1)
        end_city = previous_city
        if end_city == 0 and mask == 0:
            break
    tour = [0] + list(reversed(reversed_path))
    canonical = canonicalize_tour(tour, num_cities)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return ExactSolveResult(
        solution=canonical,
        objective_value=float(best_length),
        runtime_ms=runtime_ms,
        source="held-karp",
    )
