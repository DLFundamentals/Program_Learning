from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dasbench.problems.base import SolveOutcome


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(value)


@dataclass
class DiagnosticTracker:
    residual_size: int | None = None
    used_shortcut: bool | None = None
    used_fallback: bool | None = None
    repair_iterations: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def set_residual_size(self, value: int | None) -> None:
        self.residual_size = _optional_int(value)

    def set_shortcut(self, value: bool | None = True) -> None:
        self.used_shortcut = _optional_bool(value)

    def set_fallback(self, value: bool | None = True) -> None:
        self.used_fallback = _optional_bool(value)

    def set_repair_iterations(self, value: int | None) -> None:
        self.repair_iterations = _optional_int(value)

    def add(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            self.extra[key] = value

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.extra)
        payload.setdefault("residual_size", self.residual_size)
        payload.setdefault("used_shortcut", self.used_shortcut)
        payload.setdefault("used_fallback", self.used_fallback)
        payload.setdefault("repair_iterations", self.repair_iterations)
        return payload

    def outcome(self, solution: Any) -> SolveOutcome:
        return SolveOutcome(solution=solution, metadata=self.metadata())

