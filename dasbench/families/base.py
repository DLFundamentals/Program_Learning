from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dasbench.problems.base import ExactSolveResult


@dataclass(frozen=True)
class FamilyDefinition:
    problem: str
    name: str
    description: str
    default_family_params: dict[str, object]
    build_state: Callable[[dict[str, object]], Any]
    generate_instance: Callable[..., dict[str, object]]
    dataset_exact_solver: Callable[[dict[str, object], dict[str, object], Any], ExactSolveResult] | None = None
    hidden_rule: dict[str, object] = field(default_factory=dict)
