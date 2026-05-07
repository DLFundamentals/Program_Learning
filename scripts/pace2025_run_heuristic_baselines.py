from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import signal
import subprocess
import tarfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from benchmarks.pace2025_dominating_set import (
    DEFAULT_CACHE_DIR,
    PACE_RAW_BASE_URL,
    _private_ds_path,
    _public_ds_path,
)
from dasbench.utils import write_json


DEFAULT_OUTPUT_ROOT = Path("artifacts/pace2025_dominating_set/baseline_comparisons")
DEFAULT_EXPANDED_DIR = Path("artifacts/external/pace2025-instances-expanded")


@dataclass(frozen=True)
class SolverSpec:
    name: str
    command: list[str]
    cwd: Path | None = None


def _builtin_solver_specs() -> dict[str, SolverSpec]:
    root = Path("baselines/pace2025_root").resolve()
    shadoks = Path("baselines/pace2025_shadoks").resolve()
    fontanf = Path("baselines/pace2025_fontanf").resolve()
    swats = Path("baselines/pace2025_swats").resolve()
    return {
        "root": SolverSpec("root", [str(root / "pace_solver")], root),
        "shadoks": SolverSpec("shadoks", ["./heuristic"], shadoks),
        "fontanf": SolverSpec("fontanf", [str(fontanf / "install/bin/pace2025_ds_heuristic")], fontanf),
        "swats": SolverSpec("swats", [str(swats / "build/Pace25DSH")], swats),
    }


def _parse_solver_spec(text: str) -> SolverSpec:
    if "=" not in text:
        builtins = _builtin_solver_specs()
        if text not in builtins:
            raise ValueError(f"Unknown built-in solver {text!r}. Available: {sorted(builtins)}")
        return builtins[text]
    name, command_text = text.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"Missing solver name in {text!r}.")
    command = shlex.split(command_text)
    if not command:
        raise ValueError(f"Missing solver command in {text!r}.")
    return SolverSpec(name, command)


def _instance_relative_path(*, track: str, source: str, index: int) -> str:
    if source == "private":
        return _private_ds_path(track, index)
    if source == "public":
        return _public_ds_path(track, index)
    raise ValueError(f"Unsupported source {source!r}.")


def _download_file(relative_path: str, *, cache_dir: Path, github_ref: str) -> Path:
    import urllib.request

    target = cache_dir / relative_path
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PACE_RAW_BASE_URL}/{github_ref}/{relative_path}"
    with urllib.request.urlopen(url) as response:
        target.write_bytes(response.read())
    return target


def materialize_gr(relative_path: str, *, cache_dir: Path, expanded_dir: Path, github_ref: str) -> Path:
    source = _download_file(relative_path, cache_dir=cache_dir, github_ref=github_ref)
    if not source.name.endswith(".tar.xz"):
        return source
    target = expanded_dir / relative_path.removesuffix(".tar.xz")
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(source, mode="r:xz") as archive:
        members = [member for member in archive.getmembers() if member.isfile() and member.name.endswith(".gr")]
        if not members:
            raise RuntimeError(f"No .gr member found in {source}.")
        member = sorted(members, key=lambda item: item.name)[0]
        handle = archive.extractfile(member)
        if handle is None:
            raise RuntimeError(f"Could not extract {member.name} from {source}.")
        target.write_bytes(handle.read())
    return target


def parse_pace_header(path: Path) -> tuple[int, int]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            parts = line.split()
            if len(parts) != 4 or parts[0] != "p" or parts[1] != "ds":
                raise ValueError(f"Expected `p ds n m` header in {path}, got {line!r}.")
            return int(parts[2]), int(parts[3])
    raise ValueError(f"Missing PACE header in {path}.")


def parse_solution(text: str, *, num_vertices: int) -> tuple[list[int], str | None]:
    values: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if len(parts) != 1:
            return [], f"Expected one integer per non-comment output line, got {line!r}."
        try:
            values.append(int(parts[0]))
        except ValueError:
            return [], f"Expected integer output line, got {line!r}."
    if not values:
        return [], "Solver produced no solution size line."
    declared_size = values[0]
    raw_vertices = values[1:]
    if declared_size != len(raw_vertices):
        return [], f"Declared solution size {declared_size}, but output listed {len(raw_vertices)} vertices."
    vertices = [vertex - 1 for vertex in raw_vertices]
    seen: set[int] = set()
    for vertex in vertices:
        if not 0 <= vertex < num_vertices:
            return [], f"Vertex {vertex + 1} is outside 1..{num_vertices}."
        if vertex in seen:
            return [], f"Vertex {vertex + 1} is repeated."
        seen.add(vertex)
    return sorted(vertices), None


def verify_dominating_set(path: Path, solution: list[int], *, num_vertices: int) -> tuple[bool, str | None]:
    selected = bytearray(num_vertices)
    dominated = bytearray(num_vertices)
    dominated_count = 0
    for vertex in solution:
        if not selected[vertex]:
            selected[vertex] = 1
        if not dominated[vertex]:
            dominated[vertex] = 1
            dominated_count += 1
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c") or line.startswith("p"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            if selected[u] and not dominated[v]:
                dominated[v] = 1
                dominated_count += 1
            if selected[v] and not dominated[u]:
                dominated[u] = 1
                dominated_count += 1
    if dominated_count == num_vertices:
        return True, None
    missing = [str(index + 1) for index, value in enumerate(dominated) if not value][:5]
    return False, f"Undominated vertices remain: {', '.join(missing)}"


def run_solver(
    solver: SolverSpec,
    input_path: Path,
    *,
    timeout_seconds: float,
    grace_seconds: float,
) -> tuple[str, str, float, bool, int | None]:
    start = time.perf_counter()
    with input_path.open("rb") as stdin:
        process = subprocess.Popen(
            solver.command,
            cwd=str(solver.cwd) if solver.cwd is not None else None,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        timed_out = False
        exit_code: int | None
        try:
            stdout_bytes, stderr_bytes = process.communicate(timeout=timeout_seconds)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout_bytes, stderr_bytes = process.communicate(timeout=grace_seconds)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout_bytes, stderr_bytes = process.communicate()
                exit_code = process.returncode
    runtime_ms = (time.perf_counter() - start) * 1000.0
    return (
        stdout_bytes.decode("utf-8", errors="replace"),
        stderr_bytes.decode("utf-8", errors="replace"),
        runtime_ms,
        timed_out,
        exit_code,
    )


def _load_reference_csv(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["pace_source_path"]: row
            for row in csv.DictReader(handle)
            if row.get("pace_source_path")
        }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    by_solver: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_solver.setdefault(str(row["solver"]), []).append(row)
    summaries: dict[str, dict[str, object]] = {}
    for solver, solver_rows in sorted(by_solver.items()):
        valid_rows = [row for row in solver_rows if row["valid"]]
        solution_sizes = [int(row["solution_size"]) for row in valid_rows]
        runtime_values = [float(row["runtime_ms"]) for row in solver_rows]
        synth_better = sum(
            1
            for row in valid_rows
            if row.get("synth_solution_size") not in ("", None)
            and int(row["synth_solution_size"]) < int(row["solution_size"])
        )
        synth_worse = sum(
            1
            for row in valid_rows
            if row.get("synth_solution_size") not in ("", None)
            and int(row["synth_solution_size"]) > int(row["solution_size"])
        )
        synth_tie = sum(
            1
            for row in valid_rows
            if row.get("synth_solution_size") not in ("", None)
            and int(row["synth_solution_size"]) == int(row["solution_size"])
        )
        summaries[solver] = {
            "num_instances": len(solver_rows),
            "valid_count": len(valid_rows),
            "invalid_count": len(solver_rows) - len(valid_rows),
            "timeout_count": sum(1 for row in solver_rows if row["timed_out"]),
            "total_solution_size": sum(solution_sizes),
            "average_solution_size": sum(solution_sizes) / len(solution_sizes) if solution_sizes else 0.0,
            "average_runtime_ms": sum(runtime_values) / len(runtime_values) if runtime_values else 0.0,
            "synth_better_count": synth_better,
            "synth_worse_count": synth_worse,
            "synth_tie_count": synth_tie,
        }
    return {"solvers": summaries}


def compare_solvers(
    *,
    solvers: list[SolverSpec],
    relative_paths: list[str],
    cache_dir: Path,
    expanded_dir: Path,
    github_ref: str,
    output_dir: Path,
    timeout_seconds: float,
    grace_seconds: float,
    reference_csv: Path | None,
) -> dict[str, object]:
    references = _load_reference_csv(reference_csv)
    rows: list[dict[str, object]] = []
    solution_root = output_dir / "solutions"
    stderr_root = output_dir / "stderr"
    for relative_path in relative_paths:
        input_path = materialize_gr(
            relative_path,
            cache_dir=cache_dir,
            expanded_dir=expanded_dir,
            github_ref=github_ref,
        )
        num_vertices, num_edges = parse_pace_header(input_path)
        reference = references.get(relative_path, {})
        instance_id = Path(relative_path).name.removesuffix(".tar.xz").removesuffix(".gr")
        for solver in solvers:
            stdout, stderr, runtime_ms, timed_out, exit_code = run_solver(
                solver,
                input_path,
                timeout_seconds=timeout_seconds,
                grace_seconds=grace_seconds,
            )
            solution, parse_error = parse_solution(stdout, num_vertices=num_vertices)
            valid = False
            verify_error = None
            if parse_error is None:
                valid, verify_error = verify_dominating_set(input_path, solution, num_vertices=num_vertices)
            error = parse_error or verify_error or ""
            solution_file = solution_root / solver.name / f"{instance_id}.sol"
            stderr_file = stderr_root / solver.name / f"{instance_id}.stderr.txt"
            solution_file.parent.mkdir(parents=True, exist_ok=True)
            stderr_file.parent.mkdir(parents=True, exist_ok=True)
            solution_file.write_text(stdout, encoding="utf-8")
            stderr_file.write_text(stderr, encoding="utf-8")
            rows.append(
                {
                    "solver": solver.name,
                    "instance_id": instance_id,
                    "pace_source_path": relative_path,
                    "num_vertices": num_vertices,
                    "num_edges": num_edges,
                    "exit_code": "" if exit_code is None else exit_code,
                    "timed_out": timed_out,
                    "valid": valid,
                    "solution_size": len(solution) if valid else "",
                    "runtime_ms": runtime_ms,
                    "synth_solution_size": reference.get("solution_size", ""),
                    "adapter_reference_objective": reference.get("reference_objective", ""),
                    "solution_file": str(solution_file),
                    "stderr_file": str(stderr_file),
                    "error": error,
                }
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    results_csv = output_dir / "baseline_results.csv"
    _write_csv(results_csv, rows)
    summary = {
        "schema_version": "pace2025_ds_baseline_comparison.v1",
        "results_csv": str(results_csv),
        "reference_csv": str(reference_csv) if reference_csv is not None else None,
        "timeout_seconds": timeout_seconds,
        "grace_seconds": grace_seconds,
        "instances": relative_paths,
        **_summarize(rows),
    }
    write_json(output_dir / "baseline_summary.json", summary)
    return summary


def _instance_paths_from_args(args: argparse.Namespace) -> list[str]:
    if args.instance:
        return list(args.instance)
    end = args.start_index + args.count
    return [
        _instance_relative_path(track=args.track, source=args.source, index=index)
        for index in range(args.start_index, end)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PACE 2025 DS heuristic baseline solvers on selected instances.")
    parser.add_argument("--solver", action="append", default=[], help="Built-in name, or name='command args'.")
    parser.add_argument("--solvers", default="root,shadoks", help="Comma-separated built-in solver names.")
    parser.add_argument("--track", choices=["heuristic", "exact"], default="heuristic")
    parser.add_argument("--source", choices=["private", "public"], default="private")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--instance", action="append", help="Explicit repo-relative instance path.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--expanded-dir", type=Path, default=DEFAULT_EXPANDED_DIR)
    parser.add_argument("--github-ref", default="master")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT / "latest")
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--grace-seconds", type=float, default=20.0)
    parser.add_argument(
        "--reference-csv",
        type=Path,
        default=Path("artifacts/pace2025_dominating_set/pace2025_ds_heuristic_llm_01/pace_evaluation/pace_private_results.csv"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    solver_texts = list(args.solver)
    if not solver_texts:
        solver_texts = [item.strip() for item in args.solvers.split(",") if item.strip()]
    solvers = [_parse_solver_spec(text) for text in solver_texts]
    missing = []
    for solver in solvers:
        executable = Path(solver.command[0])
        if not executable.is_absolute() and solver.cwd is not None:
            executable = solver.cwd / executable
        if "/" in solver.command[0] and not executable.exists():
            missing.append(solver)
    if missing:
        raise FileNotFoundError(f"Missing solver executable(s): {', '.join(solver.command[0] for solver in missing)}")
    summary = compare_solvers(
        solvers=solvers,
        relative_paths=_instance_paths_from_args(args),
        cache_dir=args.cache_dir,
        expanded_dir=args.expanded_dir,
        github_ref=args.github_ref,
        output_dir=args.output_dir,
        timeout_seconds=args.timeout_seconds,
        grace_seconds=args.grace_seconds,
        reference_csv=args.reference_csv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
