from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from benchmarks.common import (
    add_common_arguments,
    jobs_from_conditions,
    resolve_sweep_artifact_root,
    run_sweep,
    selected_targets,
)
from dasbench.data import load_manifest
from dasbench.utils import timestamp_token, write_json

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CONDITION_ID = "seconds_scale_v2"


def _source_dataset_dir(source_run_root: Path, problem: str, family: str) -> Path:
    return source_run_root / "targets" / DEFAULT_SOURCE_CONDITION_ID / problem / family / "dataset"


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _target_dataset_is_current(target_dir: Path, *, source_dir: Path) -> bool:
    required_files = (
        "manifest.json",
        "benchmark_spec.json",
        "reproducibility.json",
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
    )
    if not all((target_dir / name).exists() for name in required_files):
        return False
    try:
        target_spec = json.loads((target_dir / "benchmark_spec.json").read_text(encoding="utf-8"))
        source_spec = json.loads((source_dir / "benchmark_spec.json").read_text(encoding="utf-8"))
        if target_spec != source_spec:
            return False
        target_manifest = json.loads((target_dir / "manifest.json").read_text(encoding="utf-8"))
        source_manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    if target_manifest.get("problem") != source_manifest.get("problem"):
        return False
    if target_manifest.get("family") != source_manifest.get("family"):
        return False
    if target_manifest.get("split_sizes") != source_manifest.get("split_sizes"):
        return False
    return True


def _materialize_reused_dataset(target_dir: Path, *, source_dir: Path, force: bool) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")
    if not force and _target_dataset_is_current(target_dir, source_dir=source_dir):
        return

    shutil.rmtree(target_dir, ignore_errors=True)
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in ("benchmark_spec.json", "reproducibility.json", "train.jsonl", "validation.jsonl", "test.jsonl"):
        _link_or_copy_file(source_dir / filename, target_dir / filename)

    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_paths"] = {
        "dataset_dir": str(target_dir),
        "splits": {
            "train": str(target_dir / "train.jsonl"),
            "validation": str(target_dir / "validation.jsonl"),
            "test": str(target_dir / "test.jsonl"),
        },
        "manifest": str(target_dir / "manifest.json"),
        "benchmark_spec": str(target_dir / "benchmark_spec.json"),
        "reproducibility": str(target_dir / "reproducibility.json"),
    }
    manifest["reused_dataset_source"] = str(source_dir)
    write_json(target_dir / "manifest.json", manifest)


def _load_target_metadata(
    source_run_root: Path,
    targets: list[tuple[str, str]],
) -> tuple[dict[str, dict[str, object]], int, int, int]:
    instance_params_by_target: dict[str, dict[str, object]] = {}
    split_sizes: tuple[int, int, int] | None = None
    for problem, family in targets:
        source_dir = _source_dataset_dir(source_run_root, problem, family)
        manifest = load_manifest(source_dir)
        instance_params_by_target[f"{problem}/{family}"] = dict(manifest.get("instance_params", {}))
        current_sizes = (
            int(manifest["split_sizes"]["train"]),
            int(manifest["split_sizes"]["validation"]),
            int(manifest["split_sizes"]["test"]),
        )
        if split_sizes is None:
            split_sizes = current_sizes
        elif split_sizes != current_sizes:
            raise ValueError(
                "No-hint recovery benchmark currently expects a consistent split size across the reused source datasets."
            )
    assert split_sizes is not None
    return instance_params_by_target, split_sizes[0], split_sizes[1], split_sizes[2]


def build_conditions(
    *,
    source_run_root: Path,
    targets: list[tuple[str, str]],
    iterations: int,
    beam_width: int,
) -> list[dict[str, object]]:
    instance_params_by_target, train_size, validation_size, test_size = _load_target_metadata(source_run_root, targets)
    return [
        {
            "condition_id": "no_hint_recovery",
            "profile_name": "no_hint_recovery",
            "source_run_root": str(source_run_root),
            "source_condition_id": DEFAULT_SOURCE_CONDITION_ID,
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "iterations": iterations,
            "beam_width": beam_width,
            "skip_baselines": True,
            "skip_report": False,
            "instance_params_by_target": instance_params_by_target,
        }
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-hint recovery ablation on reused second-scale-v2 datasets. "
            "This reruns LLM synthesis without hidden-rule recovery incentives."
        )
    )
    add_common_arguments(parser)
    parser.add_argument("--all-families", action="store_true")
    parser.add_argument(
        "--source-run-root",
        required=True,
        help=(
            "Existing main paper benchmark artifact root whose datasets should be reused, "
            "for example artifacts/second_scale_benchmark_v2/<sweep_id>."
        ),
    )
    parser.set_defaults(
        generator="llm_no_hint",
        repeats=1,
    )
    return parser


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None):
    sweep_id = sweep_id or args.sweep_id or timestamp_token()
    targets = selected_targets(
        args.problem,
        args.family,
        representative_only=not bool(getattr(args, "all_families", False)),
    )
    source_run_root = Path(args.source_run_root)
    conditions = build_conditions(
        source_run_root=source_run_root,
        targets=targets,
        iterations=args.iterations,
        beam_width=args.beam_width,
    )
    jobs = jobs_from_conditions(
        sweep_id=sweep_id,
        artifact_root=resolve_sweep_artifact_root(args.output_root, "no_hint_recovery_benchmark", sweep_id),
        targets=targets,
        conditions=conditions,
        args=args,
    )
    for job in jobs:
        source_dir = _source_dataset_dir(source_run_root, job.problem, job.family)
        _materialize_reused_dataset(job.dataset_dir, source_dir=source_dir, force=job.force)
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_id = args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=sweep_id)
    summary = run_sweep(
        sweep_id=sweep_id,
        sweep_kind="no_hint_recovery_benchmark",
        jobs=jobs,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
