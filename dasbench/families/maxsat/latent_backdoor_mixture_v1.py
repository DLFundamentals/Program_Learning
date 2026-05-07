from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.maxsat_common import (
    AnchorPattern,
    anchor_pattern_from_rng,
    apply_variable_noise,
    build_anchor_clause,
    build_instance,
    make_clause_from_assignment,
    partition_variables,
)

FUNCTION_IDS = (
    "x1",
    "x2",
    "x3",
    "x1_xor_x2",
    "x1_xor_x3",
    "x2_xor_x3",
    "x1_xor_x2_xor_x3",
)


@dataclass(frozen=True)
class FamilyState:
    regime_rules: dict[int, dict[int, tuple[str, bool]]]
    regime_backdoors: dict[int, list[int]]
    bridge_blocks: dict[int, list[int]]


def _function_value(function_id: str, anchor_pattern: AnchorPattern) -> bool:
    x1, x2, x3 = anchor_pattern
    if function_id == "x1":
        return x1
    if function_id == "x2":
        return x2
    if function_id == "x3":
        return x3
    if function_id == "x1_xor_x2":
        return x1 ^ x2
    if function_id == "x1_xor_x3":
        return x1 ^ x3
    if function_id == "x2_xor_x3":
        return x2 ^ x3
    if function_id == "x1_xor_x2_xor_x3":
        return x1 ^ x2 ^ x3
    raise ValueError(f"Unknown function id: {function_id}")


def build_state(context: dict[str, object]) -> FamilyState:
    params = context["instance_params"]
    seeds = context["seeds"]
    rng = random.Random(int(seeds["family"]))
    num_variables = int(params["num_variables"])
    variable_pool = list(range(4, num_variables + 1))
    blocks = partition_variables(variable_pool, 6)
    regime_rules: dict[int, dict[int, tuple[str, bool]]] = {}
    regime_backdoors: dict[int, list[int]] = {}
    bridge_blocks: dict[int, list[int]] = {}
    for regime in range(3):
        regime_rng = random.Random(int(seeds["family"]) + 911 * (regime + 1))
        rules: dict[int, tuple[str, bool]] = {}
        for variable in variable_pool:
            function_id = FUNCTION_IDS[(regime_rng.randrange(len(FUNCTION_IDS)) + regime) % len(FUNCTION_IDS)]
            invert = bool(regime_rng.getrandbits(1))
            if variable in blocks[regime]:
                invert = invert ^ bool(regime % 2)
            rules[variable] = (function_id, invert)
        regime_rules[regime] = rules
        regime_backdoors[regime] = list(blocks[regime] + blocks[(regime + 3) % len(blocks)])
        bridge_blocks[regime] = list(blocks[(regime + 1) % len(blocks)] + blocks[(regime + 4) % len(blocks)])
    return FamilyState(
        regime_rules=regime_rules,
        regime_backdoors=regime_backdoors,
        bridge_blocks=bridge_blocks,
    )


def _target_assignment(
    anchor_pattern: AnchorPattern,
    rules: dict[int, tuple[str, bool]],
    num_variables: int,
) -> list[bool]:
    assignment = [False] * num_variables
    assignment[0:3] = list(anchor_pattern)
    for variable in range(4, num_variables + 1):
        function_id, invert = rules[variable]
        assignment[variable - 1] = _function_value(function_id, anchor_pattern) ^ invert
    return assignment


def generate_instance(
    context: dict[str, object],
    *,
    rng: random.Random,
    instance_id: str,
    state: FamilyState,
) -> dict[str, object]:
    params = context["instance_params"]
    num_variables = int(params["num_variables"])
    num_clauses = int(params["num_clauses"])
    agreement = float(params["literal_agreement_probability"])
    repair = float(params["repair_probability"])
    anchor_pattern = anchor_pattern_from_rng(rng)
    regime = rng.randrange(3)
    base_assignment = _target_assignment(anchor_pattern, state.regime_rules[regime], num_variables)
    target_assignment = apply_variable_noise(
        rng,
        base_assignment,
        variable_pool=range(4, num_variables + 1),
        flip_probability=0.08 + 0.02 * regime,
    )
    clauses: list[list[int]] = []
    early_count = min(10, num_clauses - 1)
    bridge_count = min(8, max(0, num_clauses - 1 - early_count))
    backdoor_pool = state.regime_backdoors[regime]
    if len(backdoor_pool) < 3:
        backdoor_pool = list(range(4, num_variables + 1))
    bridge_pool = state.bridge_blocks[regime]
    if len(bridge_pool) < 3:
        bridge_pool = list(range(4, num_variables + 1))
    for _ in range(early_count):
        clauses.append(
            make_clause_from_assignment(
                rng,
                target_assignment,
                variable_pool=backdoor_pool,
                literal_agreement_probability=max(0.52, agreement - 0.08),
                repair_probability=repair,
            )
        )
    for _ in range(bridge_count):
        clauses.append(
            make_clause_from_assignment(
                rng,
                target_assignment,
                variable_pool=bridge_pool,
                literal_agreement_probability=max(0.5, agreement - 0.1),
                repair_probability=repair,
            )
        )
    for _ in range(num_clauses - 1 - early_count - bridge_count):
        clauses.append(
            make_clause_from_assignment(
                rng,
                target_assignment,
                variable_pool=range(4, num_variables + 1),
                literal_agreement_probability=max(0.5, agreement - 0.12),
                repair_probability=repair,
            )
        )
    clauses.append(build_anchor_clause(anchor_pattern))
    return build_instance(instance_id, num_variables, clauses)


FAMILY = FamilyDefinition(
    problem="maxsat",
    name="latent_backdoor_mixture_v1",
    description=(
        "Paper-grade MAXSAT family with several hidden backdoor regimes. Marginal literal frequencies "
        "overlap across regimes, while the useful structure lives in regime-specific variable subsets and clause motifs."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Each instance samples one of three latent regimes; regime-specific variables are Boolean functions of three anchor bits with optional negation and noise.",
        "signals": ["3-bit anchor", "regime-specific backdoor blocks", "bridge variable blocks", "overlapping literal marginals"],
        "solver_hint": "Infer the regime from clause-block structure, estimate anchor-dependent variable rules, and use local repair for noise.",
    },
)
