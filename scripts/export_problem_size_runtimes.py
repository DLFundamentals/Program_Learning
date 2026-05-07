from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY_SIZE_PARAM_BY_PROBLEM: dict[str, str] = {
    "coloring": "num_vertices",
    "maxsat": "num_variables",
    "mdkp": "num_items",
    "mds": "num_vertices",
    "mis": "num_vertices",
    "packing_lp": "num_items",
    "tsp": "num_cities",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write one JSON file containing agent and baseline runtimes for each completed "
            "problem-size target in a sweep."
        )
    )
    parser.add_argument(
        "experiment_root",
        help="Problem-size sweep root, for example artifacts/problem_size_sweep/20260501_012438",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to <experiment_root>/runtime_curves.json",
    )
    return parser


def _read_json(path: Path) -> dict[str, object] | list[object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runtime_ms(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("average_runtime_ms_mean", "average_runtime_ms"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _quality(payload: object) -> float | None:
    if not isinstance(payload, dict):
        return None
    for key in ("average_normalized_quality_mean", "average_normalized_quality"):
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _primary_size_value(problem: str, instance_params: dict[str, object]) -> float | None:
    key = PRIMARY_SIZE_PARAM_BY_PROBLEM.get(problem)
    if key is None:
        return None
    value = instance_params.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _aggregate_rows(experiment_root: Path) -> dict[str, dict[str, object]]:
    aggregate_path = experiment_root / "aggregate_results.json"
    if not aggregate_path.exists():
        return {}
    payload = _read_json(aggregate_path)
    if not isinstance(payload, list):
        return {}
    rows: dict[str, dict[str, object]] = {}
    for row in payload:
        if not isinstance(row, dict):
            continue
        report_json_path = row.get("report_json_path")
        if isinstance(report_json_path, str):
            rows[str(Path(report_json_path).resolve())] = row
    return rows


def export_runtime_curves(experiment_root: Path) -> dict[str, object]:
    row_by_report_path = _aggregate_rows(experiment_root)
    families: dict[tuple[str, str], dict[str, object]] = {}
    completed_target_count = 0

    for report_json in sorted(experiment_root.rglob("report/benchmark_report.json")):
        payload = _read_json(report_json)
        if not isinstance(payload, dict):
            continue
        manifest = payload.get("manifest")
        best_candidate = payload.get("best_candidate")
        split_reports = payload.get("split_reports")
        if not isinstance(manifest, dict) or not isinstance(best_candidate, dict) or not isinstance(split_reports, dict):
            continue

        test_split = split_reports.get("test")
        if not isinstance(test_split, dict):
            continue

        problem = str(manifest["problem"])
        family = str(manifest["family"])
        instance_params = manifest.get("instance_params", {})
        if not isinstance(instance_params, dict):
            instance_params = {}

        row = row_by_report_path.get(str(report_json.resolve()), {})
        condition_id = str(row.get("condition_id", report_json.parent.parent.parent.name))
        primary_size_param = PRIMARY_SIZE_PARAM_BY_PROBLEM.get(problem)
        primary_size_value = row.get("primary_size_value")
        if not isinstance(primary_size_value, (int, float)):
            primary_size_value = _primary_size_value(problem, instance_params)

        agent_slug = str(best_candidate.get("slug", "agent"))
        agent_payload = test_split.get(agent_slug, {})
        agent_runtime_ms = _runtime_ms(agent_payload)

        baselines: dict[str, dict[str, object]] = {}
        for solver_name, solver_payload in sorted(test_split.items()):
            if solver_name == agent_slug:
                continue
            runtime_ms = _runtime_ms(solver_payload)
            if runtime_ms is None:
                continue
            baselines[str(solver_name)] = {
                "runtime_ms": runtime_ms,
                "runtime_s": runtime_ms / 1000.0,
                "quality": _quality(solver_payload),
            }

        point = {
            "condition_id": condition_id,
            "primary_size_param": primary_size_param,
            "primary_size_value": primary_size_value,
            "instance_params": instance_params,
            "agent": {
                "slug": agent_slug,
                "runtime_ms": agent_runtime_ms,
                "runtime_s": (agent_runtime_ms / 1000.0) if isinstance(agent_runtime_ms, (int, float)) else None,
                "quality": _quality(agent_payload),
            },
            "baselines": baselines,
            "report_json_path": str(report_json),
        }

        key = (problem, family)
        family_entry = families.setdefault(
            key,
            {
                "problem": problem,
                "family": family,
                "primary_size_param": primary_size_param,
                "points": [],
            },
        )
        family_entry["points"].append(point)
        completed_target_count += 1

    exported_families: list[dict[str, object]] = []
    for _, family_entry in sorted(families.items()):
        points = family_entry["points"]
        assert isinstance(points, list)
        points.sort(
            key=lambda point: (
                float(point["primary_size_value"]) if isinstance(point.get("primary_size_value"), (int, float)) else float("inf"),
                str(point["condition_id"]),
            )
        )
        exported_families.append(family_entry)

    return {
        "experiment_root": str(experiment_root),
        "completed_target_count": completed_target_count,
        "family_count": len(exported_families),
        "families": exported_families,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    experiment_root = Path(args.experiment_root)
    if not experiment_root.exists():
        raise SystemExit(f"Experiment root does not exist: {experiment_root}")

    output_path = Path(args.output) if args.output else experiment_root / "runtime_curves.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = export_runtime_curves(experiment_root)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote runtime curves JSON to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
