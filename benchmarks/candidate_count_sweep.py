from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common import (
    add_common_arguments,
    jobs_from_conditions,
    load_aggregate_rows,
    parse_int_list,
    resolve_sweep_artifact_root,
    run_sweep,
    selected_targets,
    write_agent_runtime_plots,
)
from dasbench.utils import timestamp_token

DEFAULT_CANDIDATE_WIDTHS = [1, 3, 5]


def build_conditions(
    candidate_widths: list[int],
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    iterations: int,
    beam_width: int,
) -> list[dict[str, object]]:
    conditions: list[dict[str, object]] = []
    for candidate_width in candidate_widths:
        effective_beam_width = min(beam_width, candidate_width)
        conditions.append(
            {
                "condition_id": f"candidates_gen{candidate_width}_beam{effective_beam_width}_iter{iterations}",
                "train_size": train_size,
                "validation_size": validation_size,
                "test_size": test_size,
                "iterations": iterations,
                "beam_width": effective_beam_width,
                "candidate_width": candidate_width,
                "generated_candidate_width": candidate_width,
                "survivor_beam_width": effective_beam_width,
                "skip_baselines": True,
                "skip_report": True,
            }
        )
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an agent-only candidate-count sweep over representative benchmark families."
    )
    add_common_arguments(parser)
    parser.add_argument("--all-families", action="store_true")
    parser.add_argument("--candidate-widths", type=parse_int_list, default=list(DEFAULT_CANDIDATE_WIDTHS))
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=500)
    parser.set_defaults(repeats=1, iterations=3)
    return parser


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None) -> list:
    sweep_id = sweep_id or args.sweep_id or timestamp_token()
    targets = selected_targets(
        args.problem,
        args.family,
        representative_only=not args.all_families,
    )
    conditions = build_conditions(
        args.candidate_widths,
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        iterations=args.iterations,
        beam_width=args.beam_width,
    )
    return jobs_from_conditions(
        sweep_id=sweep_id,
        artifact_root=resolve_sweep_artifact_root(args.output_root, "candidate_count_sweep", sweep_id),
        targets=targets,
        conditions=conditions,
        args=args,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_id = args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=sweep_id)
    artifact_root = resolve_sweep_artifact_root(args.output_root, "candidate_count_sweep", sweep_id)
    summary = run_sweep(
        sweep_id=sweep_id,
        sweep_kind="candidate_count_sweep",
        jobs=jobs,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        plot_paths = write_agent_runtime_plots(
            output_dir=artifact_root,
            rows=load_aggregate_rows(summary),
            sweep_name="candidate_count_sweep",
            x_field="candidate_width",
            x_label="Generated Candidates",
            x_scale="linear",
        )
        for path in plot_paths:
            print(f"Runtime plot: {path}")
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
