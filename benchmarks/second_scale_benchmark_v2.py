from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.common import add_common_arguments, jobs_from_conditions, resolve_sweep_artifact_root, run_sweep
from dasbench.families import available_family_names
from dasbench.problems import available_problem_names
from dasbench.utils import timestamp_token

PROFILE_PATH = Path(__file__).with_name("second_scale_v2_profile.json")


def _load_profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _selected_targets(problem: str | None, family: str | None) -> list[tuple[str, str]]:
    if family and not problem:
        raise ValueError("`--family` requires `--problem`.")
    if problem:
        if problem not in available_problem_names():
            raise ValueError(f"Unknown problem `{problem}`.")
        available = available_family_names(problem)
        assert isinstance(available, list)
        if family is not None:
            if family not in available:
                raise ValueError(f"Unknown family `{family}` for problem `{problem}`.")
            return [(problem, family)]
        return [(problem, family_name) for family_name in available]
    families_by_problem = available_family_names()
    assert isinstance(families_by_problem, dict)
    return [
        (problem_name, family_name)
        for problem_name in sorted(families_by_problem)
        for family_name in families_by_problem[problem_name]
    ]


def build_conditions(
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    iterations: int,
    beam_width: int,
) -> list[dict[str, object]]:
    profile = _load_profile()
    metadata = profile.get("metadata", {})
    instance_params_by_target = profile.get("instance_params_by_target", {})
    if not isinstance(metadata, dict) or not isinstance(instance_params_by_target, dict):
        raise ValueError(f"Invalid second-scale v2 profile file: {PROFILE_PATH}")
    return [
        {
            # Historical artifact compatibility: paper-run targets are stored under
            # targets/seconds_scale_v2/... in existing result bundles.
            "condition_id": "seconds_scale_v2",
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "iterations": iterations,
            "beam_width": beam_width,
            "profile_name": "seconds_scale_v2",
            "profile_path": str(PROFILE_PATH),
            "calibration_run_id": metadata.get("calibration_run_id"),
            "calibration_target": metadata.get("calibration_target"),
            "profile_notes_by_target": metadata.get("notes_by_target", {}),
            "instance_params_by_target": instance_params_by_target,
        }
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the main paper benchmark: the family-calibrated second-scale-v2 DASBench sweep "
            "targeting Gurobi wall time."
        )
    )
    add_common_arguments(parser)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=500)
    parser.set_defaults(
        repeats=1,
        gurobi_time_limit_seconds=10.0,
        native_exact_time_limit_seconds=10.0,
        external_time_limit_seconds=10.0,
    )
    return parser


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None):
    sweep_id = sweep_id or args.sweep_id or timestamp_token()
    targets = _selected_targets(args.problem, args.family)
    conditions = build_conditions(
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        iterations=args.iterations,
        beam_width=args.beam_width,
    )
    return jobs_from_conditions(
        sweep_id=sweep_id,
        artifact_root=resolve_sweep_artifact_root(args.output_root, "second_scale_benchmark_v2", sweep_id),
        targets=targets,
        conditions=conditions,
        args=args,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_id = args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=sweep_id)
    summary = run_sweep(
        sweep_id=sweep_id,
        sweep_kind="second_scale_benchmark_v2",
        jobs=jobs,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
