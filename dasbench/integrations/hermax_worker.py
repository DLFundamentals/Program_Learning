from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from pysat.formula import WCNF

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dasbench.problems.maxsat import count_satisfied_clauses


HERMAX_SOLVER_CLASSES = {
    "open_wbo_exact": ("hermax.non_incremental", "OpenWBO"),
    "uwrmaxsat_exact": ("hermax.incremental", "UWrMaxSAT"),
    "evalmaxsat_exact": ("hermax.incremental", "EvalMaxSAT"),
    "maxhs_exact": ("hermax.non_incremental", "MaxHS"),
    "wmaxcdcl_exact": ("hermax.non_incremental", "WMaxCDCL"),
}


def _build_wcnf(instance: dict[str, Any]) -> WCNF:
    formula = WCNF()
    for clause in instance["clauses"]:
        formula.append([int(literal) for literal in clause], weight=1)
    return formula


def _assignment_from_model(model: list[int], num_variables: int) -> list[bool]:
    values = {abs(int(literal)): int(literal) > 0 for literal in model}
    return [bool(values.get(variable, False)) for variable in range(1, num_variables + 1)]


def _status_name(status: object) -> str:
    name = getattr(status, "name", None)
    return str(name if name is not None else status)


def _write_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def solve_with_hermax(
    *,
    baseline: str,
    instance: dict[str, Any],
    time_limit_seconds: float,
) -> dict[str, Any]:
    module_name, class_name = HERMAX_SOLVER_CLASSES[baseline]
    module = importlib.import_module(module_name)
    solver_class = getattr(module, class_name)
    solver = solver_class(_build_wcnf(instance))
    deadline = time.perf_counter() + max(0.0, float(time_limit_seconds))
    try:
        try:
            solver.set_terminate(lambda: int(time.perf_counter() >= deadline))
        except Exception:
            pass
        start = time.perf_counter()
        feasible = bool(solver.solve())
        runtime_ms = (time.perf_counter() - start) * 1000.0
        status = _status_name(solver.get_status())
        time_limit_hit = "INTERRUPTED" in status.upper()
        if not feasible:
            return {
                "status": status,
                "solver_runtime_ms": runtime_ms,
                "time_limit_hit": time_limit_hit,
                "error": "Hermax solver did not return a feasible incumbent.",
            }
        model = solver.get_model() or []
        solution = _assignment_from_model([int(literal) for literal in model], int(instance["num_variables"]))
        objective_value = float(count_satisfied_clauses(instance, solution))
        return {
            "status": status,
            "proved_optimal": status.upper() == "OPTIMUM",
            "time_limit_hit": time_limit_hit,
            "solver_runtime_ms": runtime_ms,
            "solution": solution,
            "objective_value": objective_value,
            "cost": float(solver.get_cost()),
        }
    finally:
        try:
            solver.close()
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Hermax MaxSAT baseline instance.")
    parser.add_argument("--baseline", required=True, choices=sorted(HERMAX_SOLVER_CLASSES))
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-limit-seconds", type=float, required=True)
    args = parser.parse_args()

    output_path = Path(args.output)
    try:
        instance = json.loads(Path(args.instance).read_text(encoding="utf-8"))
        payload = solve_with_hermax(
            baseline=args.baseline,
            instance=instance,
            time_limit_seconds=args.time_limit_seconds,
        )
        _write_payload(output_path, payload)
        return 0 if "error" not in payload else 1
    except Exception as exc:
        _write_payload(
            output_path,
            {
                "status": "error",
                "proved_optimal": False,
                "time_limit_hit": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
