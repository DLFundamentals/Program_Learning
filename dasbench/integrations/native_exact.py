from __future__ import annotations

import multiprocessing
import time
from dataclasses import dataclass
from typing import Any, Callable

from dasbench.problems.base import SolveOutcome

Solver = Callable[[dict[str, object]], Any]


@dataclass(frozen=True)
class NativeExactConfig:
    time_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.time_limit_seconds is not None and self.time_limit_seconds <= 0:
            raise ValueError("native exact time limit must be positive when set")

    @property
    def enabled(self) -> bool:
        return self.time_limit_seconds is not None

    def to_record(self) -> dict[str, object]:
        return {
            "time_limit_seconds": self.time_limit_seconds,
        }

    @classmethod
    def from_record(cls, payload: dict[str, object] | None) -> NativeExactConfig:
        if not payload:
            return cls()
        raw = payload.get("time_limit_seconds")
        return cls(
            time_limit_seconds=float(raw) if raw is not None else None,
        )


class NativeExactSolverError(RuntimeError):
    def __init__(self, message: str, *, metadata: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


class NativeExactTimeoutError(NativeExactSolverError):
    pass


def is_native_exact_baseline(name: str) -> bool:
    return name == "exact" or name.endswith("_exact")


def wrap_native_exact_baselines(
    baselines: dict[str, object],
    config: NativeExactConfig,
) -> dict[str, object]:
    if not config.enabled:
        return dict(baselines)
    wrapped: dict[str, object] = {}
    for name, solver in baselines.items():
        if not callable(solver) or not is_native_exact_baseline(name):
            wrapped[name] = solver
            continue
        wrapped[name] = _wrap_solver(name, solver, config)
    return wrapped


def _wrap_solver(name: str, solver: Solver, config: NativeExactConfig) -> Solver:
    timeout_seconds = float(config.time_limit_seconds or 0.0)

    def wrapped(instance: dict[str, object]) -> Any:
        context = _mp_context()
        queue: multiprocessing.Queue = context.Queue()
        process = context.Process(target=_worker, args=(queue, solver, instance))
        process.start()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join()
            raise NativeExactTimeoutError(
                f"Native exact baseline `{name}` timed out after {timeout_seconds:.1f}s.",
                metadata={
                    "baseline_name": name,
                    "solver_status": "TIME_LIMIT",
                    "proved_optimal": False,
                    "time_limit_hit": True,
                    "native_runtime_ms": timeout_seconds * 1000.0,
                },
            )
        if queue.empty():
            raise NativeExactSolverError(
                f"Native exact baseline `{name}` exited without returning a result.",
                metadata={
                    "baseline_name": name,
                    "solver_status": "FAILED",
                    "proved_optimal": False,
                    "time_limit_hit": False,
                },
            )
        payload = queue.get()
        if not isinstance(payload, dict):
            raise NativeExactSolverError(
                f"Native exact baseline `{name}` returned a malformed payload.",
                metadata={"baseline_name": name},
            )
        metadata = dict(payload.get("metadata") or {})
        runtime_ms = payload.get("runtime_ms")
        if isinstance(runtime_ms, (int, float)):
            metadata.setdefault("native_runtime_ms", float(runtime_ms))
        if payload.get("status") == "error":
            metadata.setdefault("baseline_name", name)
            raise NativeExactSolverError(str(payload.get("message", "native exact solver failed")), metadata=metadata)
        solution = payload.get("solution")
        if metadata:
            return SolveOutcome(solution=solution, metadata=metadata)
        return solution

    return wrapped


def _worker(
    queue: multiprocessing.Queue,
    solver: Solver,
    instance: dict[str, object],
) -> None:
    start = time.perf_counter()
    try:
        result = solver(instance)
        runtime_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(result, SolveOutcome):
            queue.put(
                {
                    "status": "ok",
                    "solution": result.solution,
                    "metadata": dict(result.metadata or {}),
                    "runtime_ms": runtime_ms,
                }
            )
            return
        queue.put(
            {
                "status": "ok",
                "solution": result,
                "metadata": {},
                "runtime_ms": runtime_ms,
            }
        )
    except Exception as exc:
        queue.put(
            {
                "status": "error",
                "message": f"{type(exc).__name__}: {exc}",
                "metadata": dict(getattr(exc, "metadata", None) or {}),
                "runtime_ms": (time.perf_counter() - start) * 1000.0,
            }
        )


def _mp_context() -> multiprocessing.context.BaseContext:
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()

