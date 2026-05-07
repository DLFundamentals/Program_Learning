from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from dasbench.problems import get_problem_definition
from dasbench.problems.base import ScoreResult, SolveOutcome
from dasbench.utils import public_instance, write_jsonl

Solver = Callable[[dict[str, object]], Any]


def failed_summary(
    name: str,
    split: str,
    num_instances: int,
    error: str,
) -> dict[str, object]:
    return {
        "name": name,
        "split": split,
        "num_instances": num_instances,
        "average_normalized_quality": 0.0,
        "average_objective_value": 0.0,
        "optimality_rate": 0.0,
        "feasibility_rate": 0.0,
        "average_runtime_ms": 1_000_000.0,
        "failure_cases": [],
        "error": error,
    }


def _instance_failure(
    problem_name: str,
    instance: dict[str, object],
    solution: object,
    score: ScoreResult,
    runtime_seconds: float,
) -> dict[str, object]:
    problem = get_problem_definition(problem_name)
    try:
        return problem.failure_case(instance, solution, score, runtime_seconds)
    except Exception as exc:
        return {
            "instance_id": instance.get("id"),
            "normalized_quality": score.normalized_quality,
            "objective_value": score.objective_value,
            "runtime_ms": runtime_seconds * 1000.0,
            "is_optimal": score.is_optimal,
            "error": score.error or f"failure-case-build-error: {type(exc).__name__}: {exc}",
        }


def _metadata_float_values(
    per_instance: list[dict[str, object]],
    key: str,
) -> list[float]:
    values: list[float] = []
    for row in per_instance:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get(key) is None:
            continue
        try:
            values.append(float(metadata[key]))
        except (TypeError, ValueError):
            continue
    return values


def _metadata_bool_values(
    per_instance: list[dict[str, object]],
    key: str,
) -> list[bool]:
    values: list[bool] = []
    for row in per_instance:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get(key) is None:
            continue
        values.append(bool(metadata[key]))
    return values


def _metadata_string_counts(
    per_instance: list[dict[str, object]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_instance:
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or metadata.get(key) is None:
            continue
        value = str(metadata[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def evaluate_solver(
    problem_name: str,
    name: str,
    solver: Solver,
    instances: list[dict[str, object]],
    *,
    split: str,
    feedback_limit: int = 3,
    diagnostics_path: Path | None = None,
) -> dict[str, object]:
    problem = get_problem_definition(problem_name)
    if not instances:
        return {
            "name": name,
            "problem": problem_name,
            "split": split,
            "num_instances": 0,
            "average_normalized_quality": 0.0,
            "average_objective_value": 0.0,
            "optimality_rate": 0.0,
            "feasibility_rate": 0.0,
            "average_runtime_ms": 0.0,
            "failure_cases": [],
        }
    per_instance: list[dict[str, object]] = []
    for instance in instances:
        exposed_instance = public_instance(instance)
        metadata: dict[str, object] | None = None
        start = time.perf_counter()
        try:
            solver_result = solver(exposed_instance)
            runtime_seconds = time.perf_counter() - start
            if isinstance(solver_result, SolveOutcome):
                raw_solution = solver_result.solution
                metadata = dict(solver_result.metadata or {})
            else:
                raw_solution = solver_result
            solution = problem.canonicalize_solution(raw_solution, exposed_instance)
            score = problem.score_solution(instance, solution)
        except Exception as exc:
            runtime_seconds = time.perf_counter() - start
            solution = []
            error_metadata = getattr(exc, "metadata", None)
            if error_metadata:
                metadata = dict(error_metadata)
            score = ScoreResult(
                is_valid=False,
                is_feasible=False,
                objective_value=0.0,
                normalized_quality=0.0,
                is_optimal=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        if metadata is not None:
            metadata["wall_clock_ms"] = runtime_seconds * 1000.0
            metadata.setdefault("instance_id", instance["id"])
        per_instance.append(
            {
                "instance": instance,
                "solution": solution,
                "score": score,
                "runtime_ms": runtime_seconds * 1000.0,
                "metadata": metadata,
            }
        )

    average_quality = sum(row["score"].normalized_quality for row in per_instance) / len(per_instance)
    average_objective = sum(row["score"].objective_value for row in per_instance) / len(per_instance)
    optimality_rate = sum(1.0 for row in per_instance if row["score"].is_optimal) / len(per_instance)
    feasibility_rate = sum(1.0 for row in per_instance if row["score"].is_feasible) / len(per_instance)
    average_runtime_ms = sum(row["runtime_ms"] for row in per_instance) / len(per_instance)
    gurobi_runtime_values = _metadata_float_values(per_instance, "gurobi_runtime_ms")
    external_runtime_values = _metadata_float_values(per_instance, "external_runtime_ms")
    mip_gap_values = _metadata_float_values(per_instance, "mip_gap")
    proved_optimal_values = _metadata_bool_values(per_instance, "proved_optimal")
    solver_status_counts = _metadata_string_counts(per_instance, "solver_status")
    errors = [row["score"].error for row in per_instance if row["score"].error]
    worst_rows = sorted(
        per_instance,
        key=lambda row: (
            row["score"].normalized_quality,
            row["score"].is_feasible,
            -row["runtime_ms"],
        ),
    )[:feedback_limit]
    failure_cases = [
        _instance_failure(
            problem_name,
            row["instance"],
            row["solution"],
            row["score"],
            row["runtime_ms"] / 1000.0,
        )
        for row in worst_rows
    ]
    summary = {
        "name": name,
        "problem": problem_name,
        "split": split,
        "num_instances": len(instances),
        "average_normalized_quality": average_quality,
        "average_objective_value": average_objective,
        "optimality_rate": optimality_rate,
        "feasibility_rate": feasibility_rate,
        "average_runtime_ms": average_runtime_ms,
        "failure_cases": failure_cases,
    }
    if gurobi_runtime_values:
        summary["average_gurobi_runtime_ms"] = sum(gurobi_runtime_values) / len(gurobi_runtime_values)
        summary["gurobi_runtime_instance_count"] = len(gurobi_runtime_values)
    if external_runtime_values:
        summary["average_external_runtime_ms"] = sum(external_runtime_values) / len(external_runtime_values)
        summary["external_runtime_instance_count"] = len(external_runtime_values)
    if proved_optimal_values:
        summary["proved_optimal_rate"] = (
            sum(1.0 for value in proved_optimal_values if value) / len(proved_optimal_values)
        )
        summary["proved_optimal_instance_count"] = len(proved_optimal_values)
    if solver_status_counts:
        summary["solver_status_counts"] = solver_status_counts
    if mip_gap_values:
        summary["average_mip_gap"] = sum(mip_gap_values) / len(mip_gap_values)
    if diagnostics_path is not None:
        diagnostics_rows = [row["metadata"] for row in per_instance if row["metadata"]]
        if diagnostics_rows:
            write_jsonl(diagnostics_path, diagnostics_rows)
    if errors and len(errors) == len(per_instance):
        unique_errors = list(dict.fromkeys(errors))
        summary["error"] = "; ".join(unique_errors[:3])
    if errors:
        summary["error_count"] = len(errors)
    return summary


def evaluate_solver_repeated(
    problem_name: str,
    name: str,
    solver: Solver,
    instances: list[dict[str, object]],
    *,
    split: str,
    repeats: int,
    feedback_limit: int = 3,
    diagnostics_path: Path | None = None,
) -> dict[str, object]:
    if repeats <= 0:
        raise ValueError("repeats must be positive.")
    trials: list[dict[str, object]] = []
    errors: list[str] = []
    for _ in range(repeats):
        try:
            trial = evaluate_solver(
                problem_name,
                name,
                solver,
                instances,
                split=split,
                feedback_limit=feedback_limit,
                diagnostics_path=diagnostics_path if not trials else None,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append(error)
            trial = failed_summary(name, split, len(instances), error)
        if "error" in trial:
            errors.append(str(trial["error"]))
        trials.append(trial)

    quality_values = [float(trial["average_normalized_quality"]) for trial in trials]
    objective_values = [float(trial["average_objective_value"]) for trial in trials]
    optimality_values = [float(trial["optimality_rate"]) for trial in trials]
    feasibility_values = [float(trial["feasibility_rate"]) for trial in trials]
    runtime_values = [float(trial["average_runtime_ms"]) for trial in trials]
    gurobi_runtime_values = [
        float(trial["average_gurobi_runtime_ms"])
        for trial in trials
        if trial.get("average_gurobi_runtime_ms") is not None
    ]
    external_runtime_values = [
        float(trial["average_external_runtime_ms"])
        for trial in trials
        if trial.get("average_external_runtime_ms") is not None
    ]
    proved_optimal_values = [
        float(trial["proved_optimal_rate"])
        for trial in trials
        if trial.get("proved_optimal_rate") is not None
    ]
    mip_gap_values = [
        float(trial["average_mip_gap"])
        for trial in trials
        if trial.get("average_mip_gap") is not None
    ]
    representative = next((trial for trial in trials if "error" not in trial), trials[0])
    summary = {
        "name": name,
        "problem": problem_name,
        "split": split,
        "num_instances": len(instances),
        "repeats": repeats,
        "average_normalized_quality_mean": statistics.mean(quality_values),
        "average_normalized_quality_std": statistics.pstdev(quality_values) if len(quality_values) > 1 else 0.0,
        "average_objective_value_mean": statistics.mean(objective_values),
        "average_objective_value_std": statistics.pstdev(objective_values) if len(objective_values) > 1 else 0.0,
        "optimality_rate_mean": statistics.mean(optimality_values),
        "optimality_rate_std": statistics.pstdev(optimality_values) if len(optimality_values) > 1 else 0.0,
        "feasibility_rate_mean": statistics.mean(feasibility_values),
        "feasibility_rate_std": statistics.pstdev(feasibility_values) if len(feasibility_values) > 1 else 0.0,
        "average_runtime_ms_mean": statistics.mean(runtime_values),
        "average_runtime_ms_std": statistics.pstdev(runtime_values) if len(runtime_values) > 1 else 0.0,
        "representative_failure_cases": representative.get("failure_cases", []),
        "error_count": len(errors),
        "errors": errors[:3],
        "trials": trials,
    }
    if gurobi_runtime_values:
        summary["average_gurobi_runtime_ms_mean"] = statistics.mean(gurobi_runtime_values)
        summary["average_gurobi_runtime_ms_std"] = (
            statistics.pstdev(gurobi_runtime_values) if len(gurobi_runtime_values) > 1 else 0.0
        )
        summary["gurobi_runtime_trial_count"] = len(gurobi_runtime_values)
    if external_runtime_values:
        summary["average_external_runtime_ms_mean"] = statistics.mean(external_runtime_values)
        summary["average_external_runtime_ms_std"] = (
            statistics.pstdev(external_runtime_values) if len(external_runtime_values) > 1 else 0.0
        )
        summary["external_runtime_trial_count"] = len(external_runtime_values)
    if proved_optimal_values:
        summary["proved_optimal_rate_mean"] = statistics.mean(proved_optimal_values)
        summary["proved_optimal_rate_std"] = (
            statistics.pstdev(proved_optimal_values) if len(proved_optimal_values) > 1 else 0.0
        )
        summary["proved_optimal_trial_count"] = len(proved_optimal_values)
    if mip_gap_values:
        summary["average_mip_gap_mean"] = statistics.mean(mip_gap_values)
        summary["average_mip_gap_std"] = statistics.pstdev(mip_gap_values) if len(mip_gap_values) > 1 else 0.0
    return summary


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
