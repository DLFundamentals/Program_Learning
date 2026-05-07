from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import json
import math
import multiprocessing
import random
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.common import (
    DEFAULT_BENCHMARK_ARTIFACTS_ROOT,
    resolve_sweep_artifact_root,
    write_aggregate_outputs,
)
from dasbench.agents.candidate import build_solver, run_analysis
from dasbench.data import load_manifest, load_split
from dasbench.families import available_family_names
from dasbench.problems import available_problem_names, get_problem_definition
from dasbench.problems.base import ScoreResult, SolveOutcome
from dasbench.utils import load_jsonl, public_instance, timestamp_token, write_json, write_jsonl
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_KIND = "graph_relabel_invariance_benchmark"
CONDITION_ID = "graph_relabel_invariance"
GRAPH_PROBLEMS = ("coloring", "mis", "mds")


@dataclass(frozen=True)
class InvarianceJob:
    artifact_root: Path
    sweep_id: str
    condition_id: str
    problem: str
    family: str
    source_run_root: Path
    transform_seed: int
    force: bool = False

    @property
    def source_target_root(self) -> Path:
        return self.source_run_root / "targets" / "seconds_scale_v2" / self.problem / self.family

    @property
    def source_dataset_dir(self) -> Path:
        return self.source_target_root / "dataset"

    @property
    def source_agent_run_dir(self) -> Path:
        return self.source_target_root / "agent_run"

    @property
    def source_report_dir(self) -> Path:
        return self.source_target_root / "report"

    @property
    def source_synthesis_summary_path(self) -> Path:
        return self.source_agent_run_dir / "synthesis_summary.json"

    @property
    def target_root(self) -> Path:
        return self.artifact_root / "targets" / self.condition_id / self.problem / self.family

    @property
    def transformed_dataset_dir(self) -> Path:
        return self.target_root / "transformed_dataset"

    @property
    def analysis_original_dir(self) -> Path:
        return self.target_root / "analysis" / "original"

    @property
    def analysis_transformed_dir(self) -> Path:
        return self.target_root / "analysis" / "transformed"

    @property
    def report_dir(self) -> Path:
        return self.target_root / "report"

    @property
    def report_json_path(self) -> Path:
        return self.report_dir / "invariance_report.json"

    @property
    def report_markdown_path(self) -> Path:
        return self.report_dir / "invariance_report.md"

    @property
    def per_instance_pairs_path(self) -> Path:
        return self.report_dir / "per_instance_runtime_pairs.jsonl"

    @property
    def completion_path(self) -> Path:
        return self.report_json_path


def _selected_targets(problem: str | None, family: str | None) -> list[tuple[str, str]]:
    if family and not problem:
        raise ValueError("`--family` requires `--problem`.")
    if problem:
        if problem not in GRAPH_PROBLEMS:
            raise ValueError(f"`{problem}` is not one of the graph problems: {', '.join(GRAPH_PROBLEMS)}.")
        if problem not in available_problem_names():
            raise ValueError(f"Unknown problem `{problem}`.")
        families = available_family_names(problem)
        assert isinstance(families, list)
        if family is not None:
            if family not in families:
                raise ValueError(f"Unknown family `{family}` for problem `{problem}`.")
            return [(problem, family)]
        return [(problem, family_name) for family_name in families]
    families_by_problem = available_family_names()
    assert isinstance(families_by_problem, dict)
    return [
        (problem_name, family_name)
        for problem_name in GRAPH_PROBLEMS
        for family_name in families_by_problem[problem_name]
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate second_scale_v2 synthesized graph solvers on original and vertex-relabeled datasets "
            "to test invariance to graph serialization details."
        )
    )
    parser.add_argument("--sweep-id")
    parser.add_argument("--output-root", default=str(DEFAULT_BENCHMARK_ARTIFACTS_ROOT))
    parser.add_argument(
        "--source-run-root",
        required=True,
        help=(
            "Existing main paper benchmark artifact root with saved graph-family solvers, "
            "for example artifacts/second_scale_benchmark_v2/<sweep_id>."
        ),
    )
    parser.add_argument("--problem", choices=GRAPH_PROBLEMS)
    parser.add_argument("--family")
    parser.add_argument("--transform-seed", type=int, default=17)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None) -> list[InvarianceJob]:
    sweep_id = sweep_id or args.sweep_id or timestamp_token()
    source_run_root = Path(args.source_run_root)
    artifact_root = resolve_sweep_artifact_root(args.output_root, BENCHMARK_KIND, sweep_id)
    return [
        InvarianceJob(
            artifact_root=artifact_root,
            sweep_id=sweep_id,
            condition_id=CONDITION_ID,
            problem=problem,
            family=family,
            source_run_root=source_run_root,
            transform_seed=int(args.transform_seed),
            force=bool(args.force),
        )
        for problem, family in _selected_targets(args.problem, args.family)
    ]


def _job_result(
    job: InvarianceJob,
    *,
    status: str,
    returncode: int,
    error: str | None = None,
) -> dict[str, object]:
    payload = {
        "condition_id": job.condition_id,
        "problem": job.problem,
        "family": job.family,
        "source_run_root": str(job.source_run_root),
        "source_dataset_dir": str(job.source_dataset_dir),
        "source_agent_run_dir": str(job.source_agent_run_dir),
        "transformed_dataset_dir": str(job.transformed_dataset_dir),
        "report_dir": str(job.report_dir),
        "report_json_path": str(job.report_json_path),
        "per_instance_pairs_path": str(job.per_instance_pairs_path),
        "status": status,
        "returncode": returncode,
        "transform_seed": int(job.transform_seed),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _resolve_candidate_dir(raw_path: str, *, source_agent_run_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = source_agent_run_dir / path
    if candidate.exists():
        return candidate
    candidate = REPO_ROOT / path
    if candidate.exists():
        return candidate
    return path


def _seed_int(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _transform_coloring_solution(solution: object, permutation: list[int]) -> object:
    if not isinstance(solution, list) or len(solution) != len(permutation):
        return solution
    transformed = [0] * len(permutation)
    for old_vertex, new_vertex in enumerate(permutation):
        transformed[new_vertex] = solution[old_vertex]
    return transformed


def _transform_vertex_set_solution(solution: object, permutation: list[int]) -> object:
    if not isinstance(solution, (list, tuple, set)):
        return solution
    return sorted(permutation[int(vertex)] for vertex in solution)


def _transform_graph_instance(
    problem_name: str,
    instance: dict[str, object],
    *,
    split_name: str,
    family: str,
    transform_seed: int,
) -> dict[str, object]:
    transformed = copy.deepcopy(instance)
    num_vertices = int(instance["num_vertices"])
    permutation = list(range(num_vertices))
    rng = random.Random(_seed_int(problem_name, family, split_name, instance["id"], transform_seed))
    rng.shuffle(permutation)

    relabeled_edges: list[list[int]] = []
    for edge in instance["edges"]:
        left, right = int(edge[0]), int(edge[1])
        mapped_left = permutation[left]
        mapped_right = permutation[right]
        if mapped_left > mapped_right:
            mapped_left, mapped_right = mapped_right, mapped_left
        relabeled_edges.append([mapped_left, mapped_right])
    rng.shuffle(relabeled_edges)

    transformed["id"] = f"{instance['id']}__relabel"
    transformed["edges"] = relabeled_edges
    transformed["_source_instance_id"] = instance["id"]
    transformed["_graph_relabel_transform"] = {
        "kind": "vertex_relabel_and_edge_shuffle",
        "transform_seed": int(transform_seed),
        "split": split_name,
        "problem": problem_name,
        "family": family,
        "permutation_digest": hashlib.sha256(
            ",".join(str(value) for value in permutation).encode("utf-8")
        ).hexdigest(),
    }

    if "optimum_solution" in instance:
        if problem_name == "coloring":
            transformed["optimum_solution"] = _transform_coloring_solution(instance["optimum_solution"], permutation)
        else:
            transformed["optimum_solution"] = _transform_vertex_set_solution(instance["optimum_solution"], permutation)
    return transformed


def _write_transformed_dataset(job: InvarianceJob) -> dict[str, object]:
    source_manifest = load_manifest(job.source_dataset_dir)
    transformed_dir = job.transformed_dataset_dir
    transformed_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "validation", "test"):
        source_rows = load_jsonl(job.source_dataset_dir / f"{split_name}.jsonl")
        transformed_rows = [
            _transform_graph_instance(
                job.problem,
                row,
                split_name=split_name,
                family=job.family,
                transform_seed=job.transform_seed,
            )
            for row in source_rows
        ]
        write_jsonl(transformed_dir / f"{split_name}.jsonl", transformed_rows)

    manifest = copy.deepcopy(source_manifest)
    manifest["description"] = (
        str(source_manifest.get("description", ""))
        + " [graph_relabel_invariance_benchmark vertex relabel transform]"
    ).strip()
    manifest["artifact_paths"] = {
        "dataset_dir": str(transformed_dir),
        "splits": {
            split_name: str(transformed_dir / f"{split_name}.jsonl")
            for split_name in ("train", "validation", "test")
        },
        "manifest": str(transformed_dir / "manifest.json"),
    }
    manifest["source_dataset_dir"] = str(job.source_dataset_dir)
    manifest["transformation"] = {
        "kind": "vertex_relabel_and_edge_shuffle",
        "transform_seed": int(job.transform_seed),
        "preserves_graph_isomorphism": True,
        "notes": "Each split instance is relabeled by a deterministic vertex permutation and its edge list is shuffled.",
    }
    write_json(transformed_dir / "manifest.json", manifest)
    source_spec = job.source_dataset_dir / "benchmark_spec.json"
    source_repro = job.source_dataset_dir / "reproducibility.json"
    if source_spec.exists():
        (transformed_dir / "benchmark_spec.json").write_text(source_spec.read_text(encoding="utf-8"), encoding="utf-8")
    if source_repro.exists():
        (transformed_dir / "reproducibility.json").write_text(source_repro.read_text(encoding="utf-8"), encoding="utf-8")
    return manifest


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


def _evaluate_solver_with_rows(
    problem_name: str,
    solver_name: str,
    solver,
    instances: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    problem = get_problem_definition(problem_name)
    rows: list[dict[str, object]] = []
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
        rows.append(
            {
                "instance_id": str(instance["id"]),
                "runtime_ms": runtime_seconds * 1000.0,
                "normalized_quality": score.normalized_quality,
                "objective_value": score.objective_value,
                "is_optimal": score.is_optimal,
                "is_feasible": score.is_feasible,
                "is_valid": score.is_valid,
                "error": score.error,
                "metadata": metadata,
                "solution": solution,
                "failure_case": _instance_failure(
                    problem_name,
                    instance,
                    solution,
                    score,
                    runtime_seconds,
                ),
            }
        )

    average_quality = sum(row["normalized_quality"] for row in rows) / len(rows)
    average_objective = sum(row["objective_value"] for row in rows) / len(rows)
    optimality_rate = sum(1.0 for row in rows if row["is_optimal"]) / len(rows)
    feasibility_rate = sum(1.0 for row in rows if row["is_feasible"]) / len(rows)
    average_runtime_ms = sum(row["runtime_ms"] for row in rows) / len(rows)
    errors = [str(row["error"]) for row in rows if row["error"]]
    worst_rows = sorted(
        rows,
        key=lambda row: (
            row["normalized_quality"],
            row["is_feasible"],
            -row["runtime_ms"],
        ),
    )[:3]
    summary = {
        "name": solver_name,
        "problem": problem_name,
        "split": "test",
        "num_instances": len(rows),
        "average_normalized_quality": average_quality,
        "average_objective_value": average_objective,
        "optimality_rate": optimality_rate,
        "feasibility_rate": feasibility_rate,
        "average_runtime_ms": average_runtime_ms,
        "runtime_ms_median": statistics.median(row["runtime_ms"] for row in rows),
        "runtime_ms_p90": sorted(row["runtime_ms"] for row in rows)[max(0, math.ceil(0.9 * len(rows)) - 1)],
        "failure_cases": [row["failure_case"] for row in worst_rows],
    }
    if errors and len(errors) == len(rows):
        summary["error"] = "; ".join(list(dict.fromkeys(errors))[:3])
    if errors:
        summary["error_count"] = len(errors)
    return summary, rows


def _runtime_relative_squared_diff(left_ms: float, right_ms: float) -> float | None:
    if left_ms <= 0.0 or right_ms <= 0.0:
        return None
    difference = right_ms - left_ms
    return (difference * difference) / (left_ms * right_ms)


def _pair_rows(
    original_rows: list[dict[str, object]],
    transformed_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for original_row, transformed_row in zip(original_rows, transformed_rows, strict=True):
        original_runtime_ms = float(original_row["runtime_ms"])
        transformed_runtime_ms = float(transformed_row["runtime_ms"])
        pair_metric = _runtime_relative_squared_diff(original_runtime_ms, transformed_runtime_ms)
        log_ratio = (
            math.log(transformed_runtime_ms / original_runtime_ms)
            if original_runtime_ms > 0.0 and transformed_runtime_ms > 0.0
            else None
        )
        pairs.append(
            {
                "source_instance_id": original_row["instance_id"],
                "transformed_instance_id": transformed_row["instance_id"],
                "original_runtime_ms": original_runtime_ms,
                "transformed_runtime_ms": transformed_runtime_ms,
                "runtime_ratio": (
                    transformed_runtime_ms / original_runtime_ms
                    if original_runtime_ms > 0.0
                    else None
                ),
                "runtime_relative_squared_diff": pair_metric,
                "runtime_log_ratio": log_ratio,
                "original_normalized_quality": float(original_row["normalized_quality"]),
                "transformed_normalized_quality": float(transformed_row["normalized_quality"]),
                "original_feasible": bool(original_row["is_feasible"]),
                "transformed_feasible": bool(transformed_row["is_feasible"]),
                "original_optimal": bool(original_row["is_optimal"]),
                "transformed_optimal": bool(transformed_row["is_optimal"]),
                "original_error": original_row["error"],
                "transformed_error": transformed_row["error"],
            }
        )
    return pairs


def _comparison_metrics(
    original_summary: dict[str, object],
    transformed_summary: dict[str, object],
    pair_rows: list[dict[str, object]],
) -> dict[str, object]:
    original_runtime_ms = float(original_summary["average_runtime_ms"])
    transformed_runtime_ms = float(transformed_summary["average_runtime_ms"])
    pairwise_diffs = [
        float(row["runtime_relative_squared_diff"])
        for row in pair_rows
        if isinstance(row.get("runtime_relative_squared_diff"), (int, float))
    ]
    log_ratios = [
        float(row["runtime_log_ratio"])
        for row in pair_rows
        if isinstance(row.get("runtime_log_ratio"), (int, float))
    ]
    quality_changed_count = sum(
        1
        for row in pair_rows
        if abs(float(row["original_normalized_quality"]) - float(row["transformed_normalized_quality"])) > 1e-12
    )
    feasibility_changed_count = sum(
        1
        for row in pair_rows
        if bool(row["original_feasible"]) != bool(row["transformed_feasible"])
    )
    optimality_changed_count = sum(
        1
        for row in pair_rows
        if bool(row["original_optimal"]) != bool(row["transformed_optimal"])
    )
    return {
        "original_runtime_ms": original_runtime_ms,
        "transformed_runtime_ms": transformed_runtime_ms,
        "runtime_ratio": transformed_runtime_ms / original_runtime_ms if original_runtime_ms > 0.0 else None,
        "runtime_relative_squared_diff": _runtime_relative_squared_diff(original_runtime_ms, transformed_runtime_ms),
        "paired_runtime_relative_squared_diff_mean": (
            statistics.mean(pairwise_diffs) if pairwise_diffs else None
        ),
        "paired_runtime_relative_squared_diff_median": (
            statistics.median(pairwise_diffs) if pairwise_diffs else None
        ),
        "paired_runtime_abs_log_ratio_mean": (
            statistics.mean(abs(value) for value in log_ratios) if log_ratios else None
        ),
        "paired_runtime_log_ratio_mean": statistics.mean(log_ratios) if log_ratios else None,
        "quality_changed_count": quality_changed_count,
        "feasibility_changed_count": feasibility_changed_count,
        "optimality_changed_count": optimality_changed_count,
        "quality_changed_fraction": quality_changed_count / len(pair_rows) if pair_rows else None,
        "feasibility_changed_fraction": feasibility_changed_count / len(pair_rows) if pair_rows else None,
        "optimality_changed_fraction": optimality_changed_count / len(pair_rows) if pair_rows else None,
    }


def _build_markdown_report(
    *,
    job: InvarianceJob,
    source_manifest: dict[str, object],
    transformed_manifest: dict[str, object],
    candidate_slug: str,
    candidate_dir: Path,
    original_summary: dict[str, object],
    transformed_summary: dict[str, object],
    comparison: dict[str, object],
) -> str:
    lines = [
        "# Graph Relabel Invariance Report",
        "",
        f"- Problem: `{job.problem}`",
        f"- Family: `{job.family}`",
        f"- Source run root: `{job.source_run_root}`",
        f"- Candidate: `{candidate_slug}`",
        f"- Candidate directory: `{candidate_dir}`",
        f"- Transformation: `vertex_relabel_and_edge_shuffle`",
        f"- Transform seed: `{job.transform_seed}`",
        "",
        "## Source Dataset",
        "",
        f"- Dataset dir: `{job.source_dataset_dir}`",
        f"- Instance parameters: `{json.dumps(source_manifest.get('instance_params', {}), sort_keys=True)}`",
        f"- Split sizes: `{json.dumps(source_manifest.get('split_sizes', {}), sort_keys=True)}`",
        "",
        "## Transformed Dataset",
        "",
        f"- Dataset dir: `{job.transformed_dataset_dir}`",
        f"- Transformation metadata: `{json.dumps(transformed_manifest.get('transformation', {}), sort_keys=True)}`",
        "",
        "## Original Solver Runtime",
        "",
        f"- Quality mean: `{float(original_summary['average_normalized_quality']):.6f}`",
        f"- Optimality rate: `{float(original_summary['optimality_rate']):.6f}`",
        f"- Feasibility rate: `{float(original_summary['feasibility_rate']):.6f}`",
        f"- Runtime mean (ms): `{float(original_summary['average_runtime_ms']):.6f}`",
        "",
        "## Transformed Solver Runtime",
        "",
        f"- Quality mean: `{float(transformed_summary['average_normalized_quality']):.6f}`",
        f"- Optimality rate: `{float(transformed_summary['optimality_rate']):.6f}`",
        f"- Feasibility rate: `{float(transformed_summary['feasibility_rate']):.6f}`",
        f"- Runtime mean (ms): `{float(transformed_summary['average_runtime_ms']):.6f}`",
        "",
        "## Comparison",
        "",
        f"- Runtime ratio (transformed/original): `{comparison['runtime_ratio']}`",
        f"- Runtime relative squared diff on means: `{comparison['runtime_relative_squared_diff']}`",
        f"- Pairwise runtime relative squared diff mean: `{comparison['paired_runtime_relative_squared_diff_mean']}`",
        f"- Pairwise runtime abs log-ratio mean: `{comparison['paired_runtime_abs_log_ratio_mean']}`",
        f"- Quality changed fraction: `{comparison['quality_changed_fraction']}`",
        f"- Feasibility changed fraction: `{comparison['feasibility_changed_fraction']}`",
        f"- Optimality changed fraction: `{comparison['optimality_changed_fraction']}`",
        "",
    ]
    return "\n".join(lines)


def _run_job(job: InvarianceJob) -> tuple[dict[str, object], dict[str, object]]:
    source_manifest = load_manifest(job.source_dataset_dir)
    synthesis_summary = json.loads(job.source_synthesis_summary_path.read_text(encoding="utf-8"))
    best_candidate = synthesis_summary["best_candidate"]
    candidate_slug = str(best_candidate["slug"])
    candidate_dir = _resolve_candidate_dir(
        str(best_candidate["candidate_dir"]),
        source_agent_run_dir=job.source_agent_run_dir,
    )

    transformed_manifest = _write_transformed_dataset(job)

    original_train = load_split(job.source_dataset_dir, "train", public=True)
    transformed_train = load_split(job.transformed_dataset_dir, "train", public=True)
    original_test = load_split(job.source_dataset_dir, "test")
    transformed_test = load_split(job.transformed_dataset_dir, "test")

    original_analysis = run_analysis(
        candidate_dir,
        original_train,
        manifest=source_manifest,
        artifact_dir=job.analysis_original_dir,
    )
    transformed_analysis = run_analysis(
        candidate_dir,
        transformed_train,
        manifest=transformed_manifest,
        artifact_dir=job.analysis_transformed_dir,
    )

    original_solver = build_solver(candidate_dir, analysis=original_analysis, manifest=source_manifest)
    transformed_solver = build_solver(candidate_dir, analysis=transformed_analysis, manifest=transformed_manifest)

    original_summary, original_rows = _evaluate_solver_with_rows(job.problem, candidate_slug, original_solver, original_test)
    transformed_summary, transformed_rows = _evaluate_solver_with_rows(
        job.problem,
        candidate_slug,
        transformed_solver,
        transformed_test,
    )

    pair_rows = _pair_rows(original_rows, transformed_rows)
    comparison = _comparison_metrics(original_summary, transformed_summary, pair_rows)

    report_payload = {
        "benchmark_kind": BENCHMARK_KIND,
        "condition_id": job.condition_id,
        "problem": job.problem,
        "family": job.family,
        "source_run_root": str(job.source_run_root),
        "source_target_root": str(job.source_target_root),
        "source_dataset_dir": str(job.source_dataset_dir),
        "source_agent_run_dir": str(job.source_agent_run_dir),
        "candidate": {
            "slug": candidate_slug,
            "candidate_dir": str(candidate_dir),
        },
        "transformation": transformed_manifest.get("transformation", {}),
        "original": {
            "dataset_dir": str(job.source_dataset_dir),
            "analysis_dir": str(job.analysis_original_dir),
            "summary": original_summary,
        },
        "transformed": {
            "dataset_dir": str(job.transformed_dataset_dir),
            "analysis_dir": str(job.analysis_transformed_dir),
            "summary": transformed_summary,
        },
        "comparison": comparison,
        "artifacts": {
            "per_instance_pairs": str(job.per_instance_pairs_path),
            "report_json": str(job.report_json_path),
            "report_markdown": str(job.report_markdown_path),
        },
    }

    write_jsonl(job.per_instance_pairs_path, pair_rows)
    write_json(job.report_json_path, report_payload)
    job.report_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    job.report_markdown_path.write_text(
        _build_markdown_report(
            job=job,
            source_manifest=source_manifest,
            transformed_manifest=transformed_manifest,
            candidate_slug=candidate_slug,
            candidate_dir=candidate_dir,
            original_summary=original_summary,
            transformed_summary=transformed_summary,
            comparison=comparison,
        )
        + "\n",
        encoding="utf-8",
    )

    row = {
        "condition_id": job.condition_id,
        "problem": job.problem,
        "family": job.family,
        "source_run_root": str(job.source_run_root),
        "source_dataset_dir": str(job.source_dataset_dir),
        "transformed_dataset_dir": str(job.transformed_dataset_dir),
        "candidate_slug": candidate_slug,
        "candidate_dir": str(candidate_dir),
        "transform_seed": int(job.transform_seed),
        "instance_params": source_manifest.get("instance_params", {}),
        "split_sizes": source_manifest.get("split_sizes", {}),
        "original_runtime_ms": comparison["original_runtime_ms"],
        "transformed_runtime_ms": comparison["transformed_runtime_ms"],
        "runtime_ratio": comparison["runtime_ratio"],
        "runtime_relative_squared_diff": comparison["runtime_relative_squared_diff"],
        "paired_runtime_relative_squared_diff_mean": comparison["paired_runtime_relative_squared_diff_mean"],
        "paired_runtime_relative_squared_diff_median": comparison["paired_runtime_relative_squared_diff_median"],
        "paired_runtime_abs_log_ratio_mean": comparison["paired_runtime_abs_log_ratio_mean"],
        "paired_runtime_log_ratio_mean": comparison["paired_runtime_log_ratio_mean"],
        "original_quality": float(original_summary["average_normalized_quality"]),
        "transformed_quality": float(transformed_summary["average_normalized_quality"]),
        "quality_changed_fraction": comparison["quality_changed_fraction"],
        "feasibility_changed_fraction": comparison["feasibility_changed_fraction"],
        "optimality_changed_fraction": comparison["optimality_changed_fraction"],
        "report_json_path": str(job.report_json_path),
        "report_markdown_path": str(job.report_markdown_path),
        "per_instance_pairs_path": str(job.per_instance_pairs_path),
    }
    return _job_result(job, status="completed", returncode=0), row


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_id = args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=sweep_id)
    output_dir = resolve_sweep_artifact_root(args.output_root, BENCHMARK_KIND, sweep_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Running {BENCHMARK_KIND} `{sweep_id}` with {len(jobs)} targets")
        for job in jobs:
            print(f"  {job.problem}/{job.family}")
        return 0

    precomputed_results: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    pending_jobs: list[InvarianceJob] = []
    for job in jobs:
        if job.completion_path.exists() and not job.force:
            precomputed_results.append(_job_result(job, status="skipped", returncode=0))
            payload = json.loads(job.report_json_path.read_text(encoding="utf-8"))
            comparison = payload.get("comparison", {})
            original_summary = payload.get("original", {}).get("summary", {})
            transformed_summary = payload.get("transformed", {}).get("summary", {})
            rows.append(
                {
                    "condition_id": job.condition_id,
                    "problem": job.problem,
                    "family": job.family,
                    "source_run_root": str(job.source_run_root),
                    "source_dataset_dir": str(job.source_dataset_dir),
                    "transformed_dataset_dir": str(job.transformed_dataset_dir),
                    "candidate_slug": payload.get("candidate", {}).get("slug"),
                    "candidate_dir": payload.get("candidate", {}).get("candidate_dir"),
                    "transform_seed": int(job.transform_seed),
                    "instance_params": load_manifest(job.source_dataset_dir).get("instance_params", {}),
                    "split_sizes": load_manifest(job.source_dataset_dir).get("split_sizes", {}),
                    "original_runtime_ms": comparison.get("original_runtime_ms"),
                    "transformed_runtime_ms": comparison.get("transformed_runtime_ms"),
                    "runtime_ratio": comparison.get("runtime_ratio"),
                    "runtime_relative_squared_diff": comparison.get("runtime_relative_squared_diff"),
                    "paired_runtime_relative_squared_diff_mean": comparison.get("paired_runtime_relative_squared_diff_mean"),
                    "paired_runtime_relative_squared_diff_median": comparison.get("paired_runtime_relative_squared_diff_median"),
                    "paired_runtime_abs_log_ratio_mean": comparison.get("paired_runtime_abs_log_ratio_mean"),
                    "paired_runtime_log_ratio_mean": comparison.get("paired_runtime_log_ratio_mean"),
                    "original_quality": original_summary.get("average_normalized_quality"),
                    "transformed_quality": transformed_summary.get("average_normalized_quality"),
                    "quality_changed_fraction": comparison.get("quality_changed_fraction"),
                    "feasibility_changed_fraction": comparison.get("feasibility_changed_fraction"),
                    "optimality_changed_fraction": comparison.get("optimality_changed_fraction"),
                    "report_json_path": str(job.report_json_path),
                    "report_markdown_path": str(job.report_markdown_path),
                    "per_instance_pairs_path": str(job.per_instance_pairs_path),
                }
            )
        else:
            pending_jobs.append(job)

    max_workers = max(1, min(int(args.max_workers), len(pending_jobs) or 1))
    results = list(precomputed_results)
    if pending_jobs:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            future_to_job = {executor.submit(_run_job, job): job for job in pending_jobs}
            with tqdm(
                total=len(jobs),
                initial=len(precomputed_results),
                desc=f"{BENCHMARK_KIND}:{sweep_id}",
                unit="target",
                dynamic_ncols=True,
            ) as progress:
                for future in concurrent.futures.as_completed(future_to_job):
                    job = future_to_job[future]
                    try:
                        result, row = future.result()
                    except Exception as exc:
                        result = _job_result(
                            job,
                            status="failed",
                            returncode=1,
                            error="".join(traceback.format_exception_only(type(exc), exc)).strip(),
                        )
                        row = {
                            "condition_id": job.condition_id,
                            "problem": job.problem,
                            "family": job.family,
                            "source_run_root": str(job.source_run_root),
                            "source_dataset_dir": str(job.source_dataset_dir),
                            "transformed_dataset_dir": str(job.transformed_dataset_dir),
                            "transform_seed": int(job.transform_seed),
                            "report_json_path": str(job.report_json_path),
                        }
                    results.append(result)
                    rows.append(row)
                    progress.update(1)
                    progress.set_postfix(
                        problem=job.problem,
                        family=job.family,
                        status=result["status"],
                        refresh=False,
                    )
    else:
        print(f"Running {BENCHMARK_KIND} `{sweep_id}` with {len(jobs)} targets and max_workers={max_workers}")

    summary = write_aggregate_outputs(
        output_dir=output_dir,
        sweep_id=sweep_id,
        sweep_kind=BENCHMARK_KIND,
        rows=sorted(rows, key=lambda item: (str(item["problem"]), str(item["family"]))),
        results=sorted(results, key=lambda item: (str(item["condition_id"]), str(item["problem"]), str(item["family"]))),
    )
    write_json(
        output_dir / "run_manifest.json",
        {
            "benchmark_kind": BENCHMARK_KIND,
            "sweep_id": sweep_id,
            "source_run_root": str(Path(args.source_run_root)),
            "transform_seed": int(args.transform_seed),
            "targets": [f"{job.problem}/{job.family}" for job in jobs],
            "aggregate_json_path": summary["aggregate_json_path"],
            "aggregate_csv_path": summary["aggregate_csv_path"],
        },
    )
    print(f"Aggregate JSON: {summary['aggregate_json_path']}")
    print(f"Aggregate CSV: {summary['aggregate_csv_path']}")
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
