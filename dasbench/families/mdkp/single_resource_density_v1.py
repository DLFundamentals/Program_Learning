from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.mdkp_exact import solve_single_resource_density_exact
from dasbench.families.packing_common import build_packing_instance, capacity_from_tightness


@dataclass(frozen=True)
class FamilyState:
    bottleneck_resource: int


def build_state(context: dict[str, object]) -> FamilyState:
    rng = random.Random(int(context["seeds"]["family"]))
    return FamilyState(bottleneck_resource=rng.randrange(int(context["instance_params"]["num_resources"])))


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    num_items = int(context["instance_params"]["num_items"])
    num_resources = int(context["instance_params"]["num_resources"])
    values: list[int] = []
    weights: list[list[int]] = []
    for _ in range(num_items):
        row = [rng.randint(1, 7) for _ in range(num_resources)]
        row[state.bottleneck_resource] = rng.randint(4, 18)
        density = rng.uniform(1.5, 3.6)
        values.append(max(1, int(round(density * row[state.bottleneck_resource] + rng.uniform(0, 5)))))
        weights.append(row)
    tightness = [0.76 for _ in range(num_resources)]
    tightness[state.bottleneck_resource] = rng.uniform(0.30, 0.42)
    capacities = capacity_from_tightness(weights, tightness, integral=True)
    return build_packing_instance(
        instance_id,
        values=values,
        weights=weights,
        capacities=capacities,
        private_metadata={"bottleneck_resource": state.bottleneck_resource},
    )


FAMILY = FamilyDefinition(
    problem="mdkp",
    name="single_resource_density_v1",
    description="Smoke MDKP family where a single hidden resource makes density heuristics strong.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    dataset_exact_solver=solve_single_resource_density_exact,
    hidden_rule={
        "summary": "One resource is consistently tight, so binary value-per-bottleneck-weight is a strong but not always exact rule.",
        "signals": ["recurring tight resource", "integer density ordering", "secondary resources mostly slack"],
        "solver_hint": "Estimate the tight resource and greedily pack by value per unit of that resource with feasibility repair.",
    },
)
