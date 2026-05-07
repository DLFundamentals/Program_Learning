from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dasbench.data.spec import BenchmarkSpec
from dasbench.families import get_family_definition
from dasbench.problems import get_problem_definition
from dasbench.utils import load_json, load_jsonl, public_instance, write_json, write_jsonl


def _resolved_instance_params(
    problem_name: str,
    overrides: dict[str, object],
) -> dict[str, object]:
    problem = get_problem_definition(problem_name)
    resolved = dict(problem.default_instance_params)
    resolved.update(overrides)
    return resolved


def _resolved_family_params(problem_name: str, family_name: str, overrides: dict[str, object]) -> dict[str, object]:
    family = get_family_definition(problem_name, family_name)
    resolved = dict(family.default_family_params)
    resolved.update(overrides)
    return resolved


def _manifest(
    output_dir: Path,
    spec: BenchmarkSpec,
    *,
    family_description: str,
    ground_truth_hidden_rule: dict[str, object],
    metric_definition: dict[str, object],
    instance_schema_version: str,
) -> dict[str, object]:
    return {
        "problem": spec.problem,
        "family": spec.family,
        "description": family_description,
        "ground_truth_hidden_rule": ground_truth_hidden_rule,
        "metric_definition": metric_definition,
        "instance_schema_version": instance_schema_version,
        "compute_optima": spec.compute_optima,
        "instance_params": _resolved_instance_params(spec.problem, spec.instance_params),
        "family_params": _resolved_family_params(spec.problem, spec.family, spec.family_params),
        "split_sizes": spec.split_sizes,
        "seeds": spec.seeds,
        "artifact_paths": {
            "dataset_dir": str(output_dir),
            "splits": {
                "train": str(output_dir / "train.jsonl"),
                "validation": str(output_dir / "validation.jsonl"),
                "test": str(output_dir / "test.jsonl"),
            },
            "manifest": str(output_dir / "manifest.json"),
            "benchmark_spec": str(output_dir / "benchmark_spec.json"),
            "reproducibility": str(output_dir / "reproducibility.json"),
        },
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _progress_path(output_dir: Path) -> Path:
    return output_dir / "dataset_progress.json"


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_jsonl_row(path: Path, row: dict[str, object], *, sync: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        if sync:
            os.fsync(handle.fileno())


def _load_resume_rows(path: Path, *, split_size: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    rows: list[dict[str, object]] = []
    truncated = False
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if index != len(lines) - 1:
                raise RuntimeError(f"Malformed JSONL content in {path} at line {index + 1}.") from exc
            truncated = True
            break
        if not isinstance(payload, dict):
            raise RuntimeError(f"Expected JSON object rows in {path}, got {type(payload).__name__}.")
        rows.append(payload)
    if len(rows) > split_size:
        rows = rows[:split_size]
        truncated = True
    if truncated:
        write_jsonl(path, rows)
    return rows


def _assert_resume_spec(output_dir: Path, spec: BenchmarkSpec) -> None:
    expected = spec.to_reproducibility_record()
    spec_path = output_dir / "benchmark_spec.json"
    repro_path = output_dir / "reproducibility.json"
    if spec_path.exists():
        observed = load_json(spec_path)
        if observed != expected:
            raise RuntimeError(
                "Partial dataset spec mismatch. Existing checkpointed data does not match the current benchmark spec."
            )
    _write_json_atomic(spec_path, expected)
    _write_json_atomic(repro_path, expected)


def _write_dataset_progress(
    output_dir: Path,
    spec: BenchmarkSpec,
    *,
    status: str,
    split_name: str | None,
    split_progress: dict[str, dict[str, object]],
    note: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema_version": "dataset_progress.v1",
        "status": status,
        "updated_at": _utc_now(),
        "problem": spec.problem,
        "family": spec.family,
        "compute_optima": spec.compute_optima,
        "instance_params": _resolved_instance_params(spec.problem, spec.instance_params),
        "family_params": _resolved_family_params(spec.problem, spec.family, spec.family_params),
        "split_sizes": dict(spec.split_sizes),
        "seeds": dict(spec.seeds),
        "splits": split_progress,
    }
    if split_name is not None:
        payload["current_split"] = split_name
    if note is not None:
        payload["note"] = note
    _write_json_atomic(_progress_path(output_dir), payload)


def generate_dataset(
    output_dir: Path,
    spec: BenchmarkSpec,
) -> dict[str, list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _assert_resume_spec(output_dir, spec)
    problem = get_problem_definition(spec.problem)
    family = get_family_definition(spec.problem, spec.family)
    instance_params = _resolved_instance_params(spec.problem, spec.instance_params)
    family_params = _resolved_family_params(spec.problem, spec.family, spec.family_params)
    build_context = {
        "problem": spec.problem,
        "family": spec.family,
        "instance_params": instance_params,
        "family_params": family_params,
        "split_sizes": spec.split_sizes,
        "seeds": spec.seeds,
        "compute_optima": spec.compute_optima,
    }
    family_state = family.build_state(build_context)
    dataset: dict[str, list[dict[str, object]]] = {}
    split_progress: dict[str, dict[str, object]] = {
        split_name: {
            "completed_instances": 0,
            "total_instances": int(spec.split_sizes[split_name]),
            "split_file": str(output_dir / f"{split_name}.jsonl"),
        }
        for split_name in ("train", "validation", "test")
    }
    _write_dataset_progress(
        output_dir,
        spec,
        status="running",
        split_name=None,
        split_progress=split_progress,
        note="dataset generation started",
    )
    for split_name in ("train", "validation", "test"):
        split_size = int(spec.split_sizes[split_name])
        split_seed = int(spec.seeds[split_name])
        split_path = output_dir / f"{split_name}.jsonl"
        existing_rows = _load_resume_rows(split_path, split_size=split_size)
        import random

        rng = random.Random(split_seed)
        instances: list[dict[str, object]] = []
        if existing_rows:
            split_progress[split_name]["completed_instances"] = len(existing_rows)
            split_progress[split_name]["resumed_instances"] = len(existing_rows)
            _write_dataset_progress(
                output_dir,
                spec,
                status="running",
                split_name=split_name,
                split_progress=split_progress,
                note=f"resuming {split_name} from {len(existing_rows)} saved instances",
            )
        for index in range(split_size):
            generated = family.generate_instance(
                build_context,
                rng=rng,
                instance_id=f"{split_name}-{index:04d}",
                state=family_state,
            )
            problem.validate_instance(generated)
            if index < len(existing_rows):
                existing = existing_rows[index]
                if public_instance(existing) != public_instance(generated):
                    raise RuntimeError(
                        f"Saved partial dataset for split `{split_name}` diverges at index {index}; "
                        "delete the dataset directory or rerun with force regeneration."
                    )
                instance = existing
            else:
                instance = generated
                if spec.compute_optima:
                    if family.dataset_exact_solver is not None:
                        exact = family.dataset_exact_solver(instance, build_context, family_state)
                    else:
                        exact = problem.exact_solver(public_instance(instance))
                    instance["optimum_objective"] = exact.objective_value
                    instance["optimum_solution"] = exact.solution
                    instance["optimum_source"] = exact.source
                    instance["optimum_runtime_ms"] = exact.runtime_ms
                _write_jsonl_row(split_path, instance)
                split_progress[split_name]["completed_instances"] = index + 1
                _write_dataset_progress(
                    output_dir,
                    spec,
                    status="running",
                    split_name=split_name,
                    split_progress=split_progress,
                    note=f"generated {split_name} instance {index + 1}/{split_size}",
                )
            instances.append(instance)
        dataset[split_name] = instances
        if not split_path.exists():
            write_jsonl(split_path, instances)
        split_progress[split_name]["completed_instances"] = len(instances)
        _write_dataset_progress(
            output_dir,
            spec,
            status="running",
            split_name=split_name,
            split_progress=split_progress,
            note=f"completed split {split_name}",
        )

    manifest = _manifest(
        output_dir,
        spec,
        family_description=family.description,
        ground_truth_hidden_rule=family.hidden_rule,
        metric_definition=problem.metric_definition,
        instance_schema_version=problem.instance_schema_version,
    )
    write_json(output_dir / "manifest.json", manifest)
    _write_json_atomic(output_dir / "benchmark_spec.json", spec.to_reproducibility_record())
    _write_json_atomic(output_dir / "reproducibility.json", spec.to_reproducibility_record())
    _write_dataset_progress(
        output_dir,
        spec,
        status="completed",
        split_name=None,
        split_progress=split_progress,
        note="dataset generation completed",
    )
    return dataset


def load_split(dataset_dir: Path, split: str, *, public: bool = False) -> list[dict[str, object]]:
    manifest = load_manifest(dataset_dir)
    problem = get_problem_definition(str(manifest["problem"]))
    instances = load_jsonl(dataset_dir / f"{split}.jsonl")
    for instance in instances:
        problem.validate_instance(instance)
    if public:
        return [public_instance(instance) for instance in instances]
    return instances


def load_manifest(dataset_dir: Path) -> dict[str, object]:
    return load_json(dataset_dir / "manifest.json")


def load_spec(dataset_dir: Path) -> dict[str, object]:
    return load_json(dataset_dir / "benchmark_spec.json")
