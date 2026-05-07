from __future__ import annotations

import argparse
import inspect
import json
import shutil
from pathlib import Path

from dasbench.families import get_family_definition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect completed experiment artifacts into a compact export folder. "
            "For each completed target, copy the family generator module, best-candidate "
            "solver files, and benchmark report."
        )
    )
    parser.add_argument(
        "experiment_root",
        help="Sweep artifact root, for example artifacts/second_scale_benchmark_v2/<sweep_id>",
    )
    parser.add_argument("output_dir", help="Destination directory for collected exports")
    return parser


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _family_module_path(problem: str, family: str) -> Path:
    definition = get_family_definition(problem, family)
    source_path = inspect.getsourcefile(definition.generate_instance) or inspect.getsourcefile(definition.build_state)
    if source_path is None:
        raise ValueError(f"Could not resolve family source module for {problem}/{family}.")
    return Path(source_path)


def _target_export_dir(experiment_root: Path, report_json: Path, output_root: Path) -> Path:
    target_dir = report_json.parent.parent
    targets_root = experiment_root / "targets"
    if targets_root.exists():
        return output_root / target_dir.relative_to(targets_root)
    payload = _read_json(report_json)
    manifest = payload.get("manifest", {})
    if isinstance(manifest, dict):
        return output_root / str(manifest.get("problem", "unknown")) / str(manifest.get("family", "unknown"))
    return output_root / target_dir.name


def collect_experiment(experiment_root: Path, output_root: Path) -> dict[str, object]:
    reports = sorted(experiment_root.rglob("report/benchmark_report.json"))
    collected: list[str] = []
    for report_json in reports:
        report_md = report_json.with_suffix(".md")
        payload = _read_json(report_json)
        manifest = payload.get("manifest", {})
        best_candidate = payload.get("best_candidate", {})
        if not isinstance(manifest, dict) or not isinstance(best_candidate, dict):
            continue

        problem = str(manifest["problem"])
        family = str(manifest["family"])
        candidate_dir = Path(str(best_candidate["candidate_dir"]))
        export_dir = _target_export_dir(experiment_root, report_json, output_root)

        family_source = _family_module_path(problem, family)
        _copy_file(family_source, export_dir / "generator_family.py")
        if report_md.exists():
            _copy_file(report_md, export_dir / "benchmark_report.md")

        analyze_path = candidate_dir / "analyze.py"
        solution_path = candidate_dir / "solution.py"
        if analyze_path.exists():
            _copy_file(analyze_path, export_dir / "analyze.py")
        if solution_path.exists():
            _copy_file(solution_path, export_dir / "solution.py")
        collected.append(str(export_dir.relative_to(output_root)))

    summary = {
        "experiment_root": str(experiment_root),
        "output_root": str(output_root),
        "completed_target_count": len(collected),
        "targets": collected,
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    experiment_root = Path(args.experiment_root)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summary = collect_experiment(experiment_root, output_root)
    print(f"Collected {summary['completed_target_count']} completed targets into {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
