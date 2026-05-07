from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

Instance = dict[str, Any]
Solver = Callable[[Instance], Any]


@dataclass(frozen=True)
class ExactSolveResult:
    solution: Any
    objective_value: float
    runtime_ms: float
    source: str


@dataclass(frozen=True)
class SolveOutcome:
    solution: Any
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ScoreResult:
    is_valid: bool
    is_feasible: bool
    objective_value: float
    normalized_quality: float
    is_optimal: bool
    error: str | None = None


@dataclass(frozen=True)
class ProblemDefinition:
    name: str
    description: str
    metric_definition: dict[str, object]
    instance_schema_version: str
    default_instance_params: dict[str, object]
    validate_instance: Callable[[Instance], None]
    canonicalize_solution: Callable[[Any, Instance], Any]
    validate_solution: Callable[[Any, Instance], tuple[bool, str | None]]
    score_solution: Callable[[Instance, Any], ScoreResult]
    summarize_training_data: Callable[[list[Instance], dict[str, object]], dict[str, object]]
    failure_case: Callable[[Instance, Any, ScoreResult, float], dict[str, object]]
    baseline_registry: Callable[[], dict[str, Solver]]
    exact_solver: Callable[[Instance], ExactSolveResult]
