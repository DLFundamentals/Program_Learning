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
)


@dataclass(frozen=True)
class FamilyState:
    variable_rules: dict[int, tuple[int, bool]]


def build_state(context: dict[str, object]) -> FamilyState:
    params = context["instance_params"]
    seeds = context["seeds"]
    rng = random.Random(int(seeds["family"]))
    num_variables = int(params["num_variables"])
    rules: dict[int, tuple[int, bool]] = {}
    for variable in range(4, num_variables + 1):
        rules[variable] = (rng.randrange(3), bool(rng.getrandbits(1)))
    return FamilyState(variable_rules=rules)


def _target_assignment(
    anchor_pattern: AnchorPattern,
    rules: dict[int, tuple[int, bool]],
    num_variables: int,
) -> list[bool]:
    assignment = [False] * num_variables
    for index, value in enumerate(anchor_pattern):
        assignment[index] = value
    for variable in range(4, num_variables + 1):
        anchor_index, invert = rules[variable]
        assignment[variable - 1] = anchor_pattern[anchor_index] ^ invert
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
    target_assignment = _target_assignment(anchor_pattern, state.variable_rules, num_variables)
    clauses = [
        make_clause_from_assignment(
            rng,
            target_assignment,
            variable_pool=range(4, num_variables + 1),
            literal_agreement_probability=agreement,
            repair_probability=repair,
        )
        for _ in range(num_clauses - 1)
    ]
    clauses.append(build_anchor_clause(anchor_pattern))
    return build_instance(instance_id, num_variables, clauses)


FAMILY = FamilyDefinition(
    problem="maxsat",
    name="last_clause_signal_v1",
    description=(
        "Smoke/debug MAXSAT family where the last clause anchors a 3-bit pattern and the remaining "
        "variables copy or negate one anchor bit under a hidden rule table."
    ),
    default_family_params={},
    build_state=build_state,
    generate_instance=generate_instance,
    hidden_rule={
        "summary": "The final clause encodes a 3-bit anchor; every other variable copies or negates one anchor bit according to a fixed hidden rule table.",
        "signals": ["last-clause anchor polarity", "variable-to-anchor copy/negation rules", "literal agreement with planted assignment"],
        "solver_hint": "Infer the anchor bits from the final clause, learn copy/negation tendencies, then repair with lightweight local search.",
    },
)
