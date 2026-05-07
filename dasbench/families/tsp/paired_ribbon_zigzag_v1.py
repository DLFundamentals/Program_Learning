from __future__ import annotations

import random

from dasbench.families.base import FamilyDefinition
from dasbench.families.tsp_common import build_tsp_instance
from dasbench.families.tsp_exact import solve_paired_ribbon_zigzag_exact


def build_state(context: dict[str, object]) -> dict[str, object]:
    return {"family_seed": int(context["seeds"]["family"])}


def _ribbon_layout(
    rng: random.Random,
    *,
    num_cities: int,
    stagger: float,
    transpose: bool,
) -> list[tuple[tuple[float, float], int, float]]:
    top_count = num_cities // 2
    bottom_count = num_cities - top_count
    spacing = 1.9
    gap = 2.2
    layout: list[tuple[tuple[float, float], int, float]] = []
    for index in range(top_count):
        x = index * spacing + rng.uniform(-0.18, 0.18)
        y = gap / 2.0 + rng.uniform(-0.12, 0.12)
        point = (y, x) if transpose else (x, y)
        layout.append((point, 1, x))
    for index in range(bottom_count):
        x = index * spacing + stagger + rng.uniform(-0.18, 0.18)
        y = -gap / 2.0 + rng.uniform(-0.12, 0.12)
        point = (y, x) if transpose else (x, y)
        layout.append((point, 0, x))
    rng.shuffle(layout)
    return layout


def _ribbon_points(
    rng: random.Random,
    *,
    num_cities: int,
    stagger: float,
    transpose: bool,
) -> list[tuple[float, float]]:
    return [point for point, _, _ in _ribbon_layout(rng, num_cities=num_cities, stagger=stagger, transpose=transpose)]


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: dict[str, object],
) -> dict[str, object]:
    num_cities = int(context["instance_params"]["num_cities"])
    regime = rng.randrange(2)
    transpose = bool(rng.randrange(2))
    stagger = 0.25 if regime == 0 else 1.0
    layout = _ribbon_layout(
        rng,
        num_cities=num_cities,
        stagger=stagger,
        transpose=transpose,
    )
    return build_tsp_instance(
        instance_id,
        [point for point, _, _ in layout],
        private_metadata={
            "ribbon_sides": [side for _, side, _ in layout],
            "major_coordinates": [round(major, 6) for _, _, major in layout],
            "transpose": transpose,
            "stagger": stagger,
            "regime_name": "ribbons",
        },
    )


FAMILY = FamilyDefinition(
    problem="tsp",
    name="paired_ribbon_zigzag_v1",
    description="Paper-grade TSP family with two noisy ribbons whose latent offset changes the best traversal pattern.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    dataset_exact_solver=solve_paired_ribbon_zigzag_exact,
    hidden_rule={
        "summary": "Cities lie on two noisy parallel ribbons with a hidden stagger offset and optional transposition; input order is shuffled.",
        "signals": ["two-line PCA structure", "balanced ribbon split", "stagger offset regime", "possible axis transposition"],
        "solver_hint": "Recover the two ribbons, sort each by the major axis, and traverse one ribbon forward and the other backward.",
    },
)
