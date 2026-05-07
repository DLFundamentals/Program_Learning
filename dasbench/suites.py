from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from dasbench.artifacts import default_agent_run_dir, default_dataset_dir, default_report_dir
from dasbench.eval.evaluator import write_summary
from dasbench.families import available_family_names
from dasbench.problems import available_problem_names
from dasbench.utils import timestamp_token

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY_PATH = REPO_ROOT / "main.py"


@dataclass(frozen=True)
class SuiteTarget:
    problem: str
    family: str


def suite_report_output(suite_id: str) -> Path:
    return Path("artifacts/reports/suites") / suite_id


def validate_benchmark_scope(args: argparse.Namespace) -> None:
    if args.all_families:
        if args.family:
            raise ValueError("`--family` cannot be used with `--all-families`.")
        if args.dataset_dir:
            raise ValueError("`--dataset-dir` is only supported for a single benchmark target.")
        if args.output_dir:
            raise ValueError("`--output-dir` is only supported for a single benchmark target.")
        if args.run_output_dir:
            raise ValueError("`--run-output-dir` is only supported for a single benchmark target.")
        if args.report_output_dir:
            raise ValueError("`--report-output-dir` is only supported for a single benchmark target.")
    else:
        if not args.problem:
            raise ValueError("`--problem` is required unless `--all-families` is used.")
        if not args.family:
            raise ValueError("`--family` is required unless `--all-families` is used.")


def benchmark_targets(args: argparse.Namespace) -> list[tuple[str, str]]:
    validate_benchmark_scope(args)
    if not args.all_families:
        return [(args.problem, args.family)]
    if args.problem:
        if args.problem not in available_problem_names():
            raise ValueError(
                f"Unknown problem `{args.problem}`. Available problems: {', '.join(available_problem_names())}"
            )
        return [(args.problem, family) for family in available_family_names(args.problem)]
    families_by_problem = available_family_names()
    assert isinstance(families_by_problem, dict)
    return [
        (problem, family)
        for problem in sorted(families_by_problem)
        for family in families_by_problem[problem]
    ]


def _suite_id(args: argparse.Namespace) -> str:
    return args.run_id or args.dataset_id or timestamp_token()


def _parallelism(args: argparse.Namespace, num_targets: int) -> int:
    requested = args.max_parallel
    if requested is None or requested <= 0:
        return num_targets
    return min(requested, num_targets)


def _target_log_path(suite_id: str, target: SuiteTarget) -> Path:
    slug = f"{target.problem}__{target.family}.log"
    return suite_report_output(suite_id) / "logs" / slug


def _target_command(args: argparse.Namespace, *, target: SuiteTarget, suite_id: str) -> list[str]:
    command = [
        sys.executable,
        str(MAIN_PY_PATH),
        "benchmark",
        "--problem",
        target.problem,
        "--family",
        target.family,
        "--dataset-id",
        suite_id,
        "--run-id",
        suite_id,
        "--generator",
        args.generator,
        "--mode",
        args.mode,
        "--iterations",
        str(args.iterations),
        "--beam-width",
        str(args.beam_width),
        "--repeats",
        str(args.repeats),
        "--train-size",
        str(args.train_size),
        "--validation-size",
        str(args.validation_size),
        "--test-size",
        str(args.test_size),
        "--family-seed",
        str(args.family_seed),
        "--train-seed",
        str(args.train_seed),
        "--validation-seed",
        str(args.validation_seed),
        "--test-seed",
        str(args.test_seed),
    ]
    if args.force_regenerate:
        command.append("--force-regenerate")
    candidate_width = getattr(args, "candidate_width", None)
    if candidate_width is not None:
        command.extend(["--candidate-width", str(candidate_width)])
    if not getattr(args, "overlap_baselines_with_synthesis", True):
        command.append("--no-overlap-baselines-with-synthesis")
    command.extend(["--baseline-workers", str(getattr(args, "baseline_workers", 1))])
    if args.include_train:
        command.append("--include-train")
    if not args.gurobi_baseline_enabled:
        command.append("--no-gurobi-baseline")
    command.extend(["--gurobi-time-limit-seconds", str(args.gurobi_time_limit_seconds)])
    command.extend(["--gurobi-threads", str(args.gurobi_threads)])
    native_exact_time_limit_seconds = getattr(args, "native_exact_time_limit_seconds", None)
    if native_exact_time_limit_seconds is not None:
        command.extend(["--native-exact-time-limit-seconds", str(native_exact_time_limit_seconds)])
    command.extend(["--external-exact-baselines", str(getattr(args, "external_exact_baselines", "auto"))])
    command.extend(["--external-time-limit-seconds", str(getattr(args, "external_time_limit_seconds", 60.0))])
    command.extend(["--external-threads", str(getattr(args, "external_threads", 1))])
    external_solver_config = getattr(args, "external_solver_config", None)
    if external_solver_config:
        command.extend(["--external-solver-config", str(external_solver_config)])
    if not args.compute_optima:
        command.append("--no-compute-optima")
    for item in args.instance_param:
        command.extend(["--instance-param", item])
    for item in args.family_param:
        command.extend(["--family-param", item])
    return command


def _run_target_subprocess(
    args: argparse.Namespace,
    *,
    target: SuiteTarget,
    suite_id: str,
) -> dict[str, object]:
    log_path = _target_log_path(suite_id, target)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = _target_command(args, target=target, suite_id=suite_id)
    dataset_dir = default_dataset_dir(target.problem, target.family, suite_id)
    run_dir = default_agent_run_dir(target.problem, target.family, suite_id)
    report_dir = default_report_dir(target.problem, target.family, suite_id)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"Command: {' '.join(command)}\n\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=os.environ.copy(),
        )
    return {
        "problem": target.problem,
        "family": target.family,
        "dataset_dir": str(dataset_dir),
        "agent_run_dir": str(run_dir),
        "report_dir": str(report_dir),
        "log_path": str(log_path),
        "command": command,
        "returncode": completed.returncode,
        "status": "completed" if completed.returncode == 0 else "failed",
    }


def run_parallel_benchmark_suite(args: argparse.Namespace) -> int:
    targets = [SuiteTarget(problem, family) for problem, family in benchmark_targets(args)]
    if len(targets) <= 1:
        raise ValueError("Parallel benchmark suite requires more than one target.")

    suite_id = _suite_id(args)
    max_parallel = _parallelism(args, len(targets))
    suite_dir = suite_report_output(suite_id)
    suite_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Running benchmark suite `{suite_id}` with {len(targets)} targets "
        f"and parallelism={max_parallel}"
    )

    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        future_to_target = {
            executor.submit(_run_target_subprocess, args, target=target, suite_id=suite_id): target
            for target in targets
        }
        for future in concurrent.futures.as_completed(future_to_target):
            target = future_to_target[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "problem": target.problem,
                    "family": target.family,
                    "dataset_dir": str(default_dataset_dir(target.problem, target.family, suite_id)),
                    "agent_run_dir": str(default_agent_run_dir(target.problem, target.family, suite_id)),
                    "report_dir": str(default_report_dir(target.problem, target.family, suite_id)),
                    "log_path": str(_target_log_path(suite_id, target)),
                    "command": [],
                    "returncode": 1,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            print(
                f"[{result['status']}] problem={target.problem} family={target.family} "
                f"log={result['log_path']}"
            )

    results.sort(key=lambda item: (str(item["problem"]), str(item["family"])))
    failures = [result for result in results if result["returncode"] != 0]
    summary = {
        "suite_id": suite_id,
        "generator": args.generator,
        "mode": args.mode,
        "iterations": args.iterations,
        "beam_width": args.beam_width,
        "candidate_width": getattr(args, "candidate_width", None),
        "repeats": args.repeats,
        "parallelism": max_parallel,
        "target_count": len(targets),
        "targets": results,
        "failed_target_count": len(failures),
    }
    summary_path = suite_dir / "benchmark_suite_summary.json"
    write_summary(summary_path, summary)
    print(f"Suite summary: {summary_path}")
    if failures:
        print(f"Suite completed with {len(failures)} failed target(s).")
        return 1
    return 0
