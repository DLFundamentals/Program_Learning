from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def timing_report_path(agent_run_dir: Path) -> Path:
    return agent_run_dir / "timing_report.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _compact_summary(summary: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in summary.items():
        if key == "failure_cases":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            payload[key] = value
            continue
        if isinstance(value, dict) and key in {"solver_status_counts"}:
            payload[key] = value
    return payload


def _compact_selection(selection: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in selection.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            payload[key] = value
    return payload


class BenchmarkTimingReporter:
    def __init__(self, path: Path, *, metadata: dict[str, object] | None = None) -> None:
        self.path = path
        self._lock = threading.RLock()
        if path.exists():
            self.payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            self.payload = {
                "schema_version": "timing_report.v1",
                "status": "initialized",
                "started_at": _utc_now(),
                "updated_at": _utc_now(),
                "stages": {},
            }
        self._ensure_structure()
        if metadata:
            self.set_metadata(metadata)
        else:
            self._persist()

    def set_metadata(self, metadata: dict[str, object]) -> None:
        with self._lock:
            for key, value in metadata.items():
                self.payload[key] = value
            self._persist()

    def mark_status(self, status: str, *, error: str | None = None) -> None:
        with self._lock:
            self.payload["status"] = status
            if error is not None:
                self.payload["error"] = error
            self._persist()

    def stage_started(self, stage_name: str, *, extra: dict[str, object] | None = None) -> None:
        with self._lock:
            stage = self._stage(stage_name)
            stage["status"] = "running"
            stage["started_at"] = _utc_now()
            if extra:
                stage.update(extra)
            self._persist()

    def stage_completed(
        self,
        stage_name: str,
        *,
        wall_ms: float,
        extra: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            stage = self._stage(stage_name)
            stage["status"] = "completed"
            stage["finished_at"] = _utc_now()
            stage["wall_ms"] = float(wall_ms)
            if extra:
                stage.update(extra)
            self._persist()

    def stage_skipped(
        self,
        stage_name: str,
        *,
        reason: str,
        extra: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            stage = self._stage(stage_name)
            stage["status"] = "skipped"
            stage["reason"] = reason
            stage["finished_at"] = _utc_now()
            if extra:
                stage.update(extra)
            self._persist()

    def stage_failed(
        self,
        stage_name: str,
        *,
        error: str,
        wall_ms: float | None = None,
        extra: dict[str, object] | None = None,
    ) -> None:
        with self._lock:
            stage = self._stage(stage_name)
            stage["status"] = "failed"
            stage["finished_at"] = _utc_now()
            stage["error"] = error
            if wall_ms is not None:
                stage["wall_ms"] = float(wall_ms)
            if extra:
                stage.update(extra)
            self._persist()

    def record_dataset_summary(
        self,
        dataset: dict[str, list[dict[str, object]]],
        *,
        compute_optima: bool,
    ) -> None:
        with self._lock:
            splits: dict[str, object] = {}
            total_instances = 0
            total_optimum_runtime_ms = 0.0
            optimum_instance_count = 0
            for split_name, instances in dataset.items():
                runtimes = [
                    float(instance["optimum_runtime_ms"])
                    for instance in instances
                    if isinstance(instance.get("optimum_runtime_ms"), (int, float))
                ]
                total_instances += len(instances)
                total_optimum_runtime_ms += sum(runtimes)
                optimum_instance_count += len(runtimes)
                splits[split_name] = {
                    "instance_count": len(instances),
                    "optimum_runtime_ms_sum": sum(runtimes),
                    "optimum_runtime_ms_mean": (sum(runtimes) / len(runtimes)) if runtimes else None,
                    "optimum_instance_count": len(runtimes),
                }
            stage = self._stage("dataset_generation")
            stage["compute_optima"] = bool(compute_optima)
            stage["splits"] = splits
            stage["total_instance_count"] = total_instances
            stage["total_optimum_runtime_ms_sum"] = total_optimum_runtime_ms
            stage["total_optimum_instance_count"] = optimum_instance_count
            self._persist()

    def record_solver_evaluation(
        self,
        stage_name: str,
        *,
        split_name: str,
        solver_name: str,
        role: str,
        wall_ms: float,
        summary: dict[str, object],
    ) -> None:
        with self._lock:
            stage = self._stage(stage_name)
            splits = stage.setdefault("splits", {})
            if not isinstance(splits, dict):
                splits = {}
                stage["splits"] = splits
            split_payload = splits.setdefault(split_name, {"solvers": {}})
            if not isinstance(split_payload, dict):
                split_payload = {"solvers": {}}
                splits[split_name] = split_payload
            solvers = split_payload.setdefault("solvers", {})
            if not isinstance(solvers, dict):
                solvers = {}
                split_payload["solvers"] = solvers
            solvers[solver_name] = {
                "role": role,
                "wall_ms": float(wall_ms),
                "summary": _compact_summary(summary),
            }
            split_payload["completed_solver_count"] = len(solvers)
            split_payload["observed_solver_wall_ms_sum"] = sum(
                float(item.get("wall_ms", 0.0))
                for item in solvers.values()
                if isinstance(item, dict)
            )
            self._persist()

    def record_synthesis_candidate(self, record: dict[str, object]) -> None:
        with self._lock:
            stage = self._stage("synthesis")
            candidates = stage.setdefault("candidates", {})
            if not isinstance(candidates, dict):
                candidates = {}
                stage["candidates"] = candidates
            hypothesis = record.get("hypothesis")
            candidates[str(record["slug"])] = {
                "slug": record["slug"],
                "plan": record.get("plan"),
                "spec": record.get("spec"),
                "hypothesis_title": hypothesis.get("title") if isinstance(hypothesis, dict) else None,
                "hypothesis_diversity_key": hypothesis.get("diversity_key") if isinstance(hypothesis, dict) else None,
                "timing": dict(record.get("timing", {})) if isinstance(record.get("timing"), dict) else {},
                "selection": _compact_selection(record.get("selection", {})) if isinstance(record.get("selection"), dict) else {},
                "train": _compact_summary(record.get("train", {})) if isinstance(record.get("train"), dict) else {},
                "validation": _compact_summary(record.get("validation", {})) if isinstance(record.get("validation"), dict) else {},
            }
            stage["completed_candidate_count"] = len(candidates)
            self._persist()

    def record_synthesis_round(self, round_payload: dict[str, object]) -> None:
        with self._lock:
            stage = self._stage("synthesis")
            rounds = stage.setdefault("rounds", [])
            if not isinstance(rounds, list):
                rounds = []
                stage["rounds"] = rounds
            compact = {
                key: value
                for key, value in round_payload.items()
                if key in {
                    "iteration",
                    "evaluated_this_round",
                    "frontier_after_ranking",
                    "frontier_diversity_keys",
                    "best_selected_slug",
                }
            }
            rounds = [row for row in rounds if not (isinstance(row, dict) and row.get("iteration") == compact.get("iteration"))]
            rounds.append(compact)
            rounds.sort(key=lambda row: int(row.get("iteration", 0)))
            stage["rounds"] = rounds
            stage["completed_round_count"] = len(rounds)
            self._persist()

    def record_best_candidate_test(
        self,
        *,
        wall_ms: float,
        summary: dict[str, object],
        slug: str,
    ) -> None:
        with self._lock:
            stage = self._stage("synthesis")
            stage["best_candidate_test"] = {
                "slug": slug,
                "wall_ms": float(wall_ms),
                "summary": _compact_summary(summary),
            }
            self._persist()

    def record_stage_detail(self, stage_name: str, key: str, value: object) -> None:
        with self._lock:
            stage = self._stage(stage_name)
            stage[key] = value
            self._persist()

    def _stage(self, stage_name: str) -> dict[str, object]:
        stages = self.payload.setdefault("stages", {})
        if not isinstance(stages, dict):
            stages = {}
            self.payload["stages"] = stages
        stage = stages.setdefault(stage_name, {"status": "pending"})
        if not isinstance(stage, dict):
            stage = {"status": "pending"}
            stages[stage_name] = stage
        return stage

    def _ensure_structure(self) -> None:
        self.payload.setdefault("schema_version", "timing_report.v1")
        self.payload.setdefault("status", "initialized")
        self.payload.setdefault("started_at", _utc_now())
        self.payload.setdefault("updated_at", _utc_now())
        stages = self.payload.setdefault("stages", {})
        if not isinstance(stages, dict):
            stages = {}
            self.payload["stages"] = stages
        for name in ("dataset_generation", "baseline_pre_synthesis", "synthesis", "report"):
            stage = stages.setdefault(name, {"status": "pending"})
            if not isinstance(stage, dict):
                stages[name] = {"status": "pending"}

    def _persist(self) -> None:
        self.payload["updated_at"] = _utc_now()
        _atomic_write_json(self.path, self.payload)
