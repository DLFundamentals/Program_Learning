from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.mdkp_exact import solve_decoy_complement_mixture_exact
from dasbench.families.packing_common import build_packing_instance, capacity_from_tightness, shuffled_balanced_classes


@dataclass(frozen=True)
class FamilyState:
    scarce_resource: int


def build_state(context: dict[str, object]) -> FamilyState:
    rng = random.Random(int(context["seeds"]["family"]))
    return FamilyState(scarce_resource=rng.randrange(int(context["instance_params"]["num_resources"])))


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    num_items = int(context["instance_params"]["num_items"])
    num_resources = int(context["instance_params"]["num_resources"])
    classes = shuffled_balanced_classes(rng, num_items, 4)
    values: list[int] = []
    weights: list[list[int]] = []
    for klass in classes:
        row = [rng.randint(1, 5) for _ in range(num_resources)]
        if klass == 0:
            row[state.scarce_resource] += rng.randint(12, 20)
            values.append(32 + rng.randint(-2, 5))
        elif klass == 1:
            row[(state.scarce_resource + 1) % num_resources] += rng.randint(7, 12)
            values.append(21 + rng.randint(-3, 4))
        elif klass == 2:
            row[(state.scarce_resource + 2) % num_resources] += rng.randint(7, 12)
            values.append(22 + rng.randint(-3, 4))
        else:
            row[state.scarce_resource] += rng.randint(3, 7)
            row[(state.scarce_resource + 1) % num_resources] += rng.randint(2, 6)
            values.append(15 + rng.randint(-3, 5))
        weights.append(row)
    tightness = [0.56 + rng.uniform(-0.04, 0.04) for _ in range(num_resources)]
    tightness[state.scarce_resource] = 0.26 + rng.uniform(-0.02, 0.03)
    capacities = capacity_from_tightness(weights, tightness, integral=True)
    return build_packing_instance(
        instance_id,
        values=values,
        weights=weights,
        capacities=capacities,
        private_metadata={"scarce_resource": state.scarce_resource, "item_classes": classes},
    )


FAMILY = FamilyDefinition(
    problem="mdkp",
    name="decoy_complement_mixture_v1",
    description="Paper MDKP family with high-density decoys and complementary bundles.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    dataset_exact_solver=solve_decoy_complement_mixture_exact,
    hidden_rule={
        "summary": "High-value decoy items look attractive locally but consume a hidden scarce resource; complementary classes combine better across resources.",
        "signals": ["decoy class with scarce-resource load", "two complementary non-scarce classes", "capacity pattern revealing scarce resource"],
        "solver_hint": "Penalize items that burn the scarce resource and search for complementary class mixtures rather than using scalar value/weight density.",
    },
)
