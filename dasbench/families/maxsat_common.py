from __future__ import annotations

import random
from collections.abc import Sequence

from dasbench.problems.maxsat import validate_instance

AnchorPattern = tuple[bool, bool, bool]


def anchor_pattern_from_rng(rng: random.Random) -> AnchorPattern:
    return tuple(bool(rng.getrandbits(1)) for _ in range(3))  # type: ignore[return-value]


def build_anchor_clause(anchor_pattern: AnchorPattern) -> list[int]:
    return [
        variable if value else -variable
        for variable, value in zip((1, 2, 3), anchor_pattern, strict=True)
    ]


def build_instance(instance_id: str, num_variables: int, clauses: list[list[int]]) -> dict[str, object]:
    instance = {
        "id": instance_id,
        "num_variables": num_variables,
        "clauses": clauses,
    }
    validate_instance(instance)
    return instance


def literal_matches_assignment(literal: int, assignment: Sequence[bool]) -> bool:
    target_value = assignment[abs(literal) - 1]
    return target_value if literal > 0 else not target_value


def make_clause_from_assignment(
    rng: random.Random,
    target_assignment: Sequence[bool],
    *,
    variable_pool: Sequence[int],
    literal_agreement_probability: float,
    repair_probability: float,
) -> list[int]:
    variables = rng.sample(list(variable_pool), 3)
    clause: list[int] = []
    for variable in variables:
        target_value = bool(target_assignment[variable - 1])
        agrees = rng.random() < literal_agreement_probability
        literal_is_positive = target_value if agrees else not target_value
        clause.append(variable if literal_is_positive else -variable)
    if (
        all(not literal_matches_assignment(literal, target_assignment) for literal in clause)
        and rng.random() < repair_probability
    ):
        flip_index = rng.randrange(3)
        variable = abs(clause[flip_index])
        clause[flip_index] = variable if target_assignment[variable - 1] else -variable
    return clause


def apply_variable_noise(
    rng: random.Random,
    assignment: Sequence[bool],
    *,
    variable_pool: Sequence[int],
    flip_probability: float,
) -> list[bool]:
    noisy = [bool(value) for value in assignment]
    for variable in variable_pool:
        if rng.random() < flip_probability:
            noisy[variable - 1] = not noisy[variable - 1]
    return noisy


def partition_variables(variable_pool: Sequence[int], num_blocks: int) -> list[list[int]]:
    blocks = [[] for _ in range(num_blocks)]
    for index, variable in enumerate(variable_pool):
        blocks[index % num_blocks].append(variable)
    return blocks
