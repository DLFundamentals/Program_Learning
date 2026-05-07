from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.mdkp_exact import solve_latent_class_knapsack_exact
from dasbench.families.packing_common import build_packing_instance, capacity_from_tightness, shuffled_balanced_classes


@dataclass(frozen=True)
class FamilyState:
    class_profiles: list[list[int]]


def build_state(context: dict[str, object]) -> FamilyState:
    rng = random.Random(int(context["seeds"]["family"]))
    num_resources = int(context["instance_params"]["num_resources"])
    num_classes = max(3, min(5, num_resources + 1))
    profiles: list[list[int]] = []
    for klass in range(num_classes):
        profile = [rng.randint(1, 5) for _ in range(num_resources)]
        profile[klass % num_resources] += rng.randint(7, 12)
        profile[(klass + 1) % num_resources] += rng.randint(2, 7)
        profiles.append(profile)
    return FamilyState(class_profiles=profiles)


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    num_items = int(context["instance_params"]["num_items"])
    num_resources = int(context["instance_params"]["num_resources"])
    regime_resource = rng.randrange(num_resources)
    classes = shuffled_balanced_classes(rng, num_items, len(state.class_profiles))
    values: list[int] = []
    weights: list[list[int]] = []
    for klass in classes:
        base = state.class_profiles[klass]
        row = [max(1, weight + rng.randint(-2, 3)) for weight in base]
        complement_bonus = 8 if (klass + regime_resource) % len(state.class_profiles) in {1, 3} else 0
        penalty = int(0.75 * row[regime_resource])
        values.append(max(1, 18 + 2 * sum(row) // num_resources + complement_bonus - penalty + rng.randint(-4, 6)))
        weights.append(row)
    tightness = [0.55 + rng.uniform(-0.04, 0.04) for _ in range(num_resources)]
    tightness[regime_resource] = 0.35 + rng.uniform(-0.025, 0.025)
    capacities = capacity_from_tightness(weights, tightness, integral=True)
    return build_packing_instance(
        instance_id,
        values=values,
        weights=weights,
        capacities=capacities,
        private_metadata={"regime_resource": regime_resource, "item_classes": classes},
    )


FAMILY = FamilyDefinition(
    problem="mdkp",
    name="latent_class_knapsack_v1",
    description="Paper MDKP family with hidden item classes and regime-dependent bottleneck resources.",
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    dataset_exact_solver=solve_latent_class_knapsack_exact,
    hidden_rule={
        "summary": "Items are drawn from latent resource-consumption classes; each instance has a hidden bottleneck resource that changes which classes are valuable.",
        "signals": ["latent item classes", "regime-dependent bottleneck", "class value shifts conditional on capacity vector"],
        "solver_hint": "Cluster items by weight profile, infer the current bottleneck from capacity tightness, then prefer classes with low bottleneck pressure.",
    },
)
