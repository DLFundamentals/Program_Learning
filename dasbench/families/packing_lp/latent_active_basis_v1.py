from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.packing_common import build_packing_instance, capacity_from_tightness, rounded_positive


@dataclass(frozen=True)
class Regime:
    resource_prices: list[float]
    tightness: list[float]


@dataclass(frozen=True)
class FamilyState:
    regimes: list[Regime]


def build_state(context: dict[str, object]) -> FamilyState:
    rng = random.Random(int(context["seeds"]["family"]))
    num_resources = int(context["instance_params"]["num_resources"])
    regimes = []
    for regime_index in range(3):
        prices = [rng.uniform(0.25, 0.7) for _ in range(num_resources)]
        primary = regime_index % num_resources
        secondary = (regime_index + 1) % num_resources
        prices[primary] += 1.4
        prices[secondary] += 0.7
        tightness = [0.58 for _ in range(num_resources)]
        tightness[primary] = 0.32
        tightness[secondary] = 0.42
        regimes.append(Regime(resource_prices=prices, tightness=tightness))
    return FamilyState(regimes=regimes)


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    num_items = int(context["instance_params"]["num_items"])
    num_resources = int(context["instance_params"]["num_resources"])
    regime_index = rng.randrange(len(state.regimes))
    regime = state.regimes[regime_index]
    values: list[int] = []
    weights: list[list[int]] = []
    for item in range(num_items):
        row = [rng.randint(2, 14) for _ in range(num_resources)]
        motif = item % max(2, num_resources)
        row[motif % num_resources] += rng.randint(0, 5)
        weights.append(row)
        latent_value = sum(row[resource] * regime.resource_prices[resource] for resource in range(num_resources))
        values.append(rounded_positive(5.0 + latent_value * rng.uniform(0.92, 1.08) + rng.uniform(-2.5, 2.5)))
    jittered_tightness = [max(0.24, min(0.72, value + rng.uniform(-0.035, 0.035))) for value in regime.tightness]
    capacities = capacity_from_tightness(weights, jittered_tightness, integral=False)
    return build_packing_instance(
        instance_id,
        values=values,
        weights=weights,
        capacities=capacities,
        private_metadata={"regime": regime_index, "resource_prices": [round(value, 4) for value in regime.resource_prices]},
    )


FAMILY = FamilyDefinition(
    problem="packing_lp",
    name="latent_active_basis_v1",
    description=(
        "Paper packing LP family with hidden regimes that share similar coefficient marginals but recur with "
        "different active constraints and optimal bases."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Each instance comes from a hidden regime with a recurring dual-price vector; regimes share similar weight marginals but bind different resource pairs.",
        "signals": ["latent active resource pair", "value as noisy weighted sum of hidden resource prices", "recurring LP basis patterns"],
        "solver_hint": "Infer the active resource prices from training optima/statistics and prioritize items by estimated reduced cost under the matching regime.",
    },
)
