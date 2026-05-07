from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.common import (
    SIZE_PARAM_KEYS_BY_PROBLEM,
    add_common_arguments,
    jobs_from_conditions,
    load_aggregate_rows,
    resolve_sweep_artifact_root,
    run_sweep,
    selected_targets,
    write_problem_size_runtime_plots,
)
from dasbench.problems import get_problem_definition
from dasbench.utils import timestamp_token

PROFILE_PATH = Path(__file__).with_name("second_scale_v2_profile.json")
DEFAULT_SIZE_POINTS = 10
SIZE_GRID: dict[str, dict[str, dict[str, int]]] = {
    "tiny": {
        "maxsat": {"num_variables": 10, "num_clauses": 18},
        "mis": {"num_vertices": 12},
        "mds": {"num_vertices": 12},
        "coloring": {"num_vertices": 12},
        "tsp": {"num_cities": 8},
        "packing_lp": {"num_items": 12, "num_resources": 3},
        "mdkp": {"num_items": 12, "num_resources": 3},
    },
    "small": {
        "maxsat": {"num_variables": 16, "num_clauses": 32},
        "mis": {"num_vertices": 18},
        "mds": {"num_vertices": 18},
        "coloring": {"num_vertices": 18},
        "tsp": {"num_cities": 10},
        "packing_lp": {"num_items": 20, "num_resources": 4},
        "mdkp": {"num_items": 20, "num_resources": 4},
    },
    "medium": {
        "maxsat": {"num_variables": 24, "num_clauses": 54},
        "mis": {"num_vertices": 24},
        "mds": {"num_vertices": 24},
        "coloring": {"num_vertices": 24},
        "tsp": {"num_cities": 12},
        "packing_lp": {"num_items": 32, "num_resources": 5},
        "mdkp": {"num_items": 32, "num_resources": 5},
    },
    "large": {
        "maxsat": {"num_variables": 32, "num_clauses": 80},
        "mis": {"num_vertices": 30},
        "mds": {"num_vertices": 30},
        "coloring": {"num_vertices": 30},
        "tsp": {"num_cities": 13},
        "packing_lp": {"num_items": 44, "num_resources": 6},
        "mdkp": {"num_items": 44, "num_resources": 6},
    },
}


def _load_profile() -> dict[str, object]:
    return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))


def _parse_size_labels(text: str) -> list[str]:
    labels = [item.strip() for item in text.split(",") if item.strip()]
    if not labels:
        raise argparse.ArgumentTypeError("Expected at least one size label.")
    unknown = [label for label in labels if label not in SIZE_GRID]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown size label(s): {', '.join(unknown)}")
    return labels


def _interpolate_size(low: int, high: int, index: int, count: int) -> int:
    if count <= 1:
        return high
    if index <= 0:
        return low
    if index >= count - 1:
        return high
    ratio = index / float(count - 1)
    if low <= 0 or high <= 0 or low == high:
        value = round(low + (high - low) * ratio)
    else:
        value = round(low * ((high / low) ** ratio))
    return int(max(min(low, high), min(max(low, high), value)))


def _interpolated_instance_params(
    problem: str,
    calibrated_params: dict[str, object],
    *,
    size_index: int,
    size_points: int,
) -> dict[str, object]:
    problem_definition = get_problem_definition(problem)
    resolved = dict(problem_definition.default_instance_params)
    resolved.update(calibrated_params)
    for key in SIZE_PARAM_KEYS_BY_PROBLEM[problem]:
        low = int(problem_definition.default_instance_params[key])
        high = int(resolved[key])
        resolved[key] = _interpolate_size(low, high, size_index, size_points)
    return resolved


def build_conditions(
    targets: list[tuple[str, str]],
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    iterations: int,
    beam_width: int,
    size_points: int,
    size_labels: list[str] | None = None,
) -> list[dict[str, object]]:
    if size_labels is not None:
        return [
            {
                "condition_id": f"size_{label}",
                "train_size": train_size,
                "validation_size": validation_size,
                "test_size": test_size,
                "iterations": iterations,
                "beam_width": beam_width,
                "size_label": label,
                "profile_name": "legacy_size_grid",
                "instance_params_by_problem": SIZE_GRID[label],
            }
            for label in size_labels
        ]

    profile = _load_profile()
    profile_targets = profile.get("instance_params_by_target", {})
    if not isinstance(profile_targets, dict):
        raise ValueError(f"Invalid problem-size profile file: {PROFILE_PATH}")

    conditions: list[dict[str, object]] = []
    for size_index in range(size_points):
        instance_params_by_target: dict[str, dict[str, object]] = {}
        for problem, family in targets:
            target_key = f"{problem}/{family}"
            calibrated = profile_targets.get(target_key)
            if not isinstance(calibrated, dict):
                raise ValueError(f"Missing calibrated size profile for `{target_key}` in {PROFILE_PATH}.")
            instance_params_by_target[target_key] = _interpolated_instance_params(
                problem,
                calibrated,
                size_index=size_index,
                size_points=size_points,
            )
        conditions.append(
            {
                "condition_id": f"size_{size_index + 1:02d}",
                "train_size": train_size,
                "validation_size": validation_size,
                "test_size": test_size,
                "iterations": iterations,
                "beam_width": beam_width,
                "size_point_index": size_index + 1,
                "size_point_count": size_points,
                "size_fraction": 1.0 if size_points <= 1 else size_index / float(size_points - 1),
                "profile_name": "seconds_scale_v2_interpolated",
                "profile_path": str(PROFILE_PATH),
                "instance_params_by_target": instance_params_by_target,
            }
        )
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a family-focused problem-size sweep ending at the calibrated ~5s Gurobi scale."
    )
    add_common_arguments(parser)
    parser.add_argument("--all-families", action="store_true")
    parser.add_argument("--size-points", type=int, default=DEFAULT_SIZE_POINTS)
    parser.add_argument(
        "--size-labels",
        type=_parse_size_labels,
        default=None,
        help="Use the historical labeled tiny/small/medium/large grid instead of the paper interpolated profile.",
    )
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


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None) -> list:
    sweep_id = sweep_id or args.sweep_id or timestamp_token()
    targets = selected_targets(
        args.problem,
        args.family,
        representative_only=not args.all_families,
    )
    conditions = build_conditions(
        targets,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        iterations=args.iterations,
        beam_width=args.beam_width,
        size_points=args.size_points,
        size_labels=args.size_labels,
    )
    return jobs_from_conditions(
        sweep_id=sweep_id,
        artifact_root=resolve_sweep_artifact_root(args.output_root, "problem_size_sweep", sweep_id),
        targets=targets,
        conditions=conditions,
        args=args,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_id = args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=sweep_id)
    artifact_root = resolve_sweep_artifact_root(args.output_root, "problem_size_sweep", sweep_id)
    summary = run_sweep(
        sweep_id=sweep_id,
        sweep_kind="problem_size_sweep",
        jobs=jobs,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        plot_paths = write_problem_size_runtime_plots(
            output_dir=artifact_root,
            rows=load_aggregate_rows(summary),
        )
        for path in plot_paths:
            print(f"Runtime plot: {path}")
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
