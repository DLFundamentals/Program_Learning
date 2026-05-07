from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common import (
    add_common_arguments,
    jobs_from_conditions,
    load_aggregate_rows,
    parse_nonnegative_int_list,
    resolve_sweep_artifact_root,
    run_sweep,
    selected_targets,
    write_agent_runtime_plots,
)
from dasbench.problems import available_problem_names
from dasbench.utils import timestamp_token

DEFAULT_TRAIN_SIZES = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]
SAMPLE_SIZE_FAMILY_BY_PROBLEM = {
    "coloring": "cluster_ring_mix_v1",
    "maxsat": "community_parity_overlay_v1",
    "mdkp": "latent_class_knapsack_v1",
    "mds": "geometric_cluster_cover_v1",
    "mis": "clique_path_mix_v1",
    "packing_lp": "block_coupled_resource_v1",
    "tsp": "clustered_euclidean_v1",
}


def _sample_size_targets(
    problem: str | None,
    family: str | None,
    problems: list[str] | None,
    *,
    all_families: bool,
) -> list[tuple[str, str]]:
    if problems and (problem or family):
        raise ValueError("`--problems` cannot be combined with `--problem` or `--family`.")
    if problems:
        if family:
            raise ValueError("`--family` cannot be combined with `--problems`.")
        targets: list[tuple[str, str]] = []
        for problem_name in problems:
            if all_families:
                targets.extend(selected_targets(problem_name, None, representative_only=False))
            else:
                targets.append((problem_name, SAMPLE_SIZE_FAMILY_BY_PROBLEM[problem_name]))
        return targets
    if all_families or family:
        return selected_targets(problem, family, representative_only=False)
    if problem:
        return [(problem, SAMPLE_SIZE_FAMILY_BY_PROBLEM[problem])]
    return [
        (problem_name, SAMPLE_SIZE_FAMILY_BY_PROBLEM[problem_name])
        for problem_name in sorted(SAMPLE_SIZE_FAMILY_BY_PROBLEM)
    ]


def build_conditions(
    train_sizes: list[int],
    *,
    validation_size: int | None,
    match_validation_to_train: bool,
    test_size: int,
    iterations: int,
    beam_width: int,
) -> list[dict[str, object]]:
    shared_dataset_train_size = max(train_sizes)
    resolved_conditions: list[dict[str, object]] = []
    resolved_validation_sizes: list[int] = []
    for train_size in train_sizes:
        if match_validation_to_train:
            resolved_validation_size = train_size
        elif validation_size is None:
            resolved_validation_size = max(4, train_size // 2)
        else:
            resolved_validation_size = validation_size
        resolved_validation_sizes.append(resolved_validation_size)
        if validation_size == 32 and not match_validation_to_train:
            condition_id = f"sample_train{train_size}"
        else:
            condition_id = f"sample_train{train_size}_val{resolved_validation_size}"
        resolved_conditions.append(
            {
                "condition_id": condition_id,
                "train_size": train_size,
                "validation_size": resolved_validation_size,
                "test_size": test_size,
                "iterations": iterations,
                "beam_width": beam_width,
                "sample_train_size": train_size,
                "sample_validation_size": resolved_validation_size,
                "shared_dataset_train_size": shared_dataset_train_size,
                "skip_baselines": True,
                "skip_report": True,
            }
        )
    shared_dataset_validation_size = max(resolved_validation_sizes)
    for condition in resolved_conditions:
        condition["shared_dataset_validation_size"] = shared_dataset_validation_size
    return resolved_conditions


def parse_problem_list(text: str) -> list[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one problem.")
    available = set(available_problem_names())
    unknown = [value for value in values if value not in available]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"Unknown problem(s): {', '.join(unknown)}. Available problems: {', '.join(sorted(available))}"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an agent-only sample-size sweep over representative benchmark families."
    )
    add_common_arguments(parser)
    parser.add_argument("--problems", type=parse_problem_list, default=None)
    parser.add_argument("--all-families", action="store_true")
    parser.add_argument("--train-sizes", type=parse_nonnegative_int_list, default=list(DEFAULT_TRAIN_SIZES))
    parser.add_argument(
        "--validation-size",
        type=int,
        default=None,
        help="Fixed validation set size. Omit to use the historical train-scaled validation grid.",
    )
    parser.add_argument("--match-validation-to-train", action="store_true")
    parser.add_argument("--test-size", type=int, default=500)
    parser.set_defaults(repeats=1)
    return parser


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None) -> list:
    sweep_id = sweep_id or args.sweep_id or timestamp_token()
    targets = _sample_size_targets(
        args.problem,
        args.family,
        args.problems,
        all_families=args.all_families,
    )
    conditions = build_conditions(
        args.train_sizes,
        validation_size=args.validation_size,
        match_validation_to_train=args.match_validation_to_train,
        test_size=args.test_size,
        iterations=args.iterations,
        beam_width=args.beam_width,
    )
    return jobs_from_conditions(
        sweep_id=sweep_id,
        artifact_root=resolve_sweep_artifact_root(args.output_root, "sample_size_sweep", sweep_id),
        targets=targets,
        conditions=conditions,
        args=args,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_id = args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=sweep_id)
    artifact_root = resolve_sweep_artifact_root(args.output_root, "sample_size_sweep", sweep_id)
    summary = run_sweep(
        sweep_id=sweep_id,
        sweep_kind="sample_size_sweep",
        jobs=jobs,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        plot_paths = write_agent_runtime_plots(
            output_dir=artifact_root,
            rows=load_aggregate_rows(summary),
            sweep_name="sample_size_sweep",
            x_field="train_size",
            x_label="Training Samples",
            x_scale="sample_symlog",
        )
        for path in plot_paths:
            print(f"Runtime plot: {path}")
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
