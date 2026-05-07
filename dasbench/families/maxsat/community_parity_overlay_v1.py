from __future__ import annotations

import random
from dataclasses import dataclass

from dasbench.families.base import FamilyDefinition
from dasbench.families.maxsat_common import (
    AnchorPattern,
    anchor_pattern_from_rng,
    build_anchor_clause,
    build_instance,
    make_clause_from_assignment,
    partition_variables,
)


@dataclass(frozen=True)
class FamilyState:
    communities: list[list[int]]
    community_rules: dict[int, tuple[str, bool]]
    bridge_pairs: list[tuple[int, int]]


def _community_value(rule_id: str, anchor_pattern: AnchorPattern) -> bool:
    x1, x2, x3 = anchor_pattern
    if rule_id == "x1":
        return x1
    if rule_id == "x2":
        return x2
    if rule_id == "x3":
        return x3
    if rule_id == "x1_xor_x2":
        return x1 ^ x2
    if rule_id == "x2_xor_x3":
        return x2 ^ x3
    return x1 ^ x2 ^ x3


def build_state(context: dict[str, object]) -> FamilyState:
    params = context["instance_params"]
    seeds = context["seeds"]
    num_variables = int(params["num_variables"])
    variable_pool = list(range(4, num_variables + 1))
    communities = partition_variables(variable_pool, 4)
    rng = random.Random(int(seeds["family"]))
    rule_ids = ("x1", "x2", "x3", "x1_xor_x2", "x2_xor_x3", "x1_xor_x2_x3")
    community_rules: dict[int, tuple[str, bool]] = {}
    for index, _community in enumerate(communities):
        community_rules[index] = (
            rule_ids[(index + rng.randrange(len(rule_ids))) % len(rule_ids)],
            bool(rng.getrandbits(1)),
        )
    bridge_pairs = [(0, 1), (1, 2), (2, 3), (0, 3)]
    return FamilyState(
        communities=communities,
        community_rules=community_rules,
        bridge_pairs=bridge_pairs,
    )


def _target_assignment(
    anchor_pattern: AnchorPattern,
    state: FamilyState,
    num_variables: int,
) -> list[bool]:
    assignment = [False] * num_variables
    assignment[0:3] = list(anchor_pattern)
    for community_index, community in enumerate(state.communities):
        rule_id, invert = state.community_rules[community_index]
        community_value = _community_value(rule_id, anchor_pattern) ^ invert
        for variable in community:
            assignment[variable - 1] = community_value
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
    target_assignment = _target_assignment(anchor_pattern, state, num_variables)
    clauses: list[list[int]] = []
    intra_count = int((num_clauses - 1) * 0.72)
    bridge_count = num_clauses - 1 - intra_count
    for _ in range(intra_count):
        community_index = rng.randrange(len(state.communities))
        pool = state.communities[community_index]
        if len(pool) < 3:
            pool = list(range(4, num_variables + 1))
        clauses.append(
            make_clause_from_assignment(
                rng,
                target_assignment,
                variable_pool=pool,
                literal_agreement_probability=max(0.52, agreement - 0.08),
                repair_probability=repair,
            )
        )
    for _ in range(bridge_count):
        left_index, right_index = state.bridge_pairs[rng.randrange(len(state.bridge_pairs))]
        pool = list(state.communities[left_index] + state.communities[right_index])
        if len(pool) < 3:
            pool = list(range(4, num_variables + 1))
        clauses.append(
            make_clause_from_assignment(
                rng,
                target_assignment,
                variable_pool=pool,
                literal_agreement_probability=max(0.5, agreement - 0.1),
                repair_probability=repair,
            )
        )
    rng.shuffle(clauses)
    clauses.append(build_anchor_clause(anchor_pattern))
    return build_instance(instance_id, num_variables, clauses)


FAMILY = FamilyDefinition(
    problem="maxsat",
    name="community_parity_overlay_v1",
    description=(
        "Paper-grade MAXSAT family with community-structured variable interactions and sparse bridge clauses. "
        "Useful signal comes from parity-style community rules rather than simple literal marginals."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "Variables are partitioned into communities; each community shares a hidden parity-style function of the three anchor bits.",
        "signals": ["community variable subsets", "intra-community clauses", "sparse bridge clauses", "anchor parity relationships"],
        "solver_hint": "Detect communities and assign whole communities from inferred parity relationships to the anchor bits.",
    },
)
