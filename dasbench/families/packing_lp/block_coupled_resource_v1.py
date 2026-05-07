from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.packing_common import build_packing_instance, capacity_from_tightness, shuffled_balanced_classes


@dataclass(frozen=True)
class FamilyState:
    coupling_resource: int


def build_state(context: dict[str, object]) -> FamilyState:
    rng = random.Random(int(context["seeds"]["family"]))
    return FamilyState(coupling_resource=rng.randrange(int(context["instance_params"]["num_resources"])))


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    num_items = int(context["instance_params"]["num_items"])
    num_resources = int(context["instance_params"]["num_resources"])
    num_blocks = max(2, min(4, num_resources))
    item_blocks = shuffled_balanced_classes(rng, num_items, num_blocks)
    values: list[int] = []
    weights: list[list[int]] = []
    for block in item_blocks:
        row = [rng.randint(1, 5) for _ in range(num_resources)]
        primary = block % num_resources
        neighbor = (block + 1) % num_resources
        row[primary] += rng.randint(8, 16)
        row[neighbor] += rng.randint(3, 9)
        row[state.coupling_resource] += rng.randint(2, 7)
        value = 8 + 3 * row[primary] + 2 * row[neighbor] - int(0.9 * row[state.coupling_resource])
        values.append(max(1, value + rng.randint(-3, 5)))
        weights.append(row)
    tightness = [0.58 + rng.uniform(-0.03, 0.03) for _ in range(num_resources)]
    tightness[state.coupling_resource] = 0.34 + rng.uniform(-0.025, 0.025)
    capacities = capacity_from_tightness(weights, tightness, integral=False)
    return build_packing_instance(
        instance_id,
        values=values,
        weights=weights,
        capacities=capacities,
        private_metadata={"coupling_resource": state.coupling_resource, "item_blocks": item_blocks},
    )


FAMILY = FamilyDefinition(
    problem="packing_lp",
    name="block_coupled_resource_v1",
    description="Paper packing LP family with item/resource blocks plus sparse coupling constraints.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Items belong to latent resource blocks, but a shared coupling resource quietly limits the mix of otherwise attractive block-local items.",
        "signals": ["block-specific high coefficients", "shared sparse coupling resource", "latent dual price for cross-block coupling"],
        "solver_hint": "Detect block membership from coefficient patterns, then adjust density by the coupling resource price rather than using block-local value alone.",
    },
)
