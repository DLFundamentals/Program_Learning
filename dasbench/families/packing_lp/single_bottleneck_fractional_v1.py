from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.packing_common import build_packing_instance, capacity_from_tightness, positive_int_jitter


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
        bottleneck_weight = rng.randint(4, 16)
        row = [rng.randint(1, 8) for _ in range(num_resources)]
        row[state.bottleneck_resource] = bottleneck_weight
        efficiency = rng.uniform(2.0, 3.8)
        values.append(positive_int_jitter(rng, 6.0 + efficiency * bottleneck_weight, spread=0.12))
        weights.append(row)
    tightness = [0.72 for _ in range(num_resources)]
    tightness[state.bottleneck_resource] = rng.uniform(0.28, 0.42)
    capacities = capacity_from_tightness(weights, tightness, integral=False)
    return build_packing_instance(
        instance_id,
        values=values,
        weights=weights,
        capacities=capacities,
        private_metadata={"bottleneck_resource": state.bottleneck_resource},
    )


FAMILY = FamilyDefinition(
    problem="packing_lp",
    name="single_bottleneck_fractional_v1",
    description="Smoke packing LP family where one hidden resource usually determines the fractional optimum.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "One resource is consistently much tighter than the others; the optimal LP basis is mostly governed by value per unit of that hidden bottleneck.",
        "signals": ["one recurring tight capacity", "fractional cutoff item by bottleneck density", "looser secondary resources"],
        "solver_hint": "Estimate the binding resource from capacity tightness and sort items by value per unit of that resource.",
    },
)
