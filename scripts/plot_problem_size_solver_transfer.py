from __future__ import annotations

import argparse
import concurrent.futures
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dasbench.agents.candidate import build_solver, run_analysis
from dasbench.eval.evaluator import evaluate_solver
from dasbench.utils import load_json, load_jsonl


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> dict[str, object] | list[object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_aggregate_rows(experiment_root: Path) -> list[dict[str, object]]:
    aggregate_path = experiment_root / "aggregate_results.json"
    if not aggregate_path.exists():
        raise ValueError(f"Missing aggregate results: {aggregate_path}")
    payload = _read_json(aggregate_path)
    if not isinstance(payload, list):
        raise ValueError(f"Expected list payload in {aggregate_path}")
    rows = [row for row in payload if isinstance(row, dict) and str(row.get("status")) in {"completed", "skipped"}]
    if not rows:
        raise ValueError(f"No completed rows found in {aggregate_path}")
    return rows


def _group_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str], list[dict[str, object]]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        problem = str(row["problem"])
        family = str(row["family"])
        grouped[(problem, family)].append(row)
    for family_rows in grouped.values():
        family_rows.sort(
            key=lambda row: (
                float(row["primary_size_value"]) if isinstance(row.get("primary_size_value"), (int, float)) else float("inf"),
                str(row["condition_id"]),
            )
        )
    return grouped


def _select_source_rows(
    family_rows: list[dict[str, object]],
    *,
    source_condition_ids: list[str] | None,
) -> list[tuple[str, dict[str, object]]]:
    if source_condition_ids:
        selected: list[tuple[str, dict[str, object]]] = []
        by_condition = {str(row["condition_id"]): row for row in family_rows}
        labels = ["low", "medium", "high"]
        for label, condition_id in zip(labels, source_condition_ids, strict=True):
            try:
                selected.append((label, by_condition[condition_id]))
            except KeyError as exc:
                raise ValueError(f"Condition `{condition_id}` not found in family rows.") from exc
        return selected

    if len(family_rows) < 3:
        raise ValueError("Need at least three completed problem-size points to choose low/medium/high solvers.")
    low_index = 0
    medium_index = (len(family_rows) - 1) // 2
    high_index = len(family_rows) - 1
    return [
        ("low", family_rows[low_index]),
        ("medium", family_rows[medium_index]),
        ("high", family_rows[high_index]),
    ]


def _cached_entry(
    cache: dict[str, object],
    *,
    problem: str,
    family: str,
    source_label: str,
    source_condition_id: str,
    target_condition_id: str,
) -> dict[str, object] | None:
    families = cache.get("families")
    if not isinstance(families, list):
        return None
    for family_payload in families:
        if not isinstance(family_payload, dict):
            continue
        if str(family_payload.get("problem")) != problem or str(family_payload.get("family")) != family:
            continue
        for source_payload in family_payload.get("source_solvers", []):
            if not isinstance(source_payload, dict):
                continue
            if (
                str(source_payload.get("label")) == source_label
                and str(source_payload.get("source_condition_id")) == source_condition_id
            ):
                for point in source_payload.get("points", []):
                    if not isinstance(point, dict):
                        continue
                    if str(point.get("target_condition_id")) == target_condition_id:
                        return point
    return None


def _source_summary(source_row: dict[str, object]) -> dict[str, object]:
    synthesis_path = Path(str(source_row["agent_run_dir"])) / "synthesis_summary.json"
    if not synthesis_path.exists():
        raise ValueError(f"Missing synthesis summary: {synthesis_path}")
    payload = load_json(synthesis_path)
    best_candidate = payload.get("best_candidate")
    if not isinstance(best_candidate, dict):
        raise ValueError(f"Invalid synthesis summary (missing best_candidate): {synthesis_path}")
    slug = best_candidate.get("slug")
    if not isinstance(slug, str) or not slug:
        raise ValueError(f"Invalid synthesis summary (missing slug): {synthesis_path}")
    return {
        "slug": slug,
        "synthesis_summary_path": str(synthesis_path),
    }


def _dataset_payload(dataset_dir: Path) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    manifest = load_json(dataset_dir / "manifest.json")
    train_instances = load_jsonl(dataset_dir / "train.jsonl")
    test_instances = load_jsonl(dataset_dir / "test.jsonl")
    return manifest, train_instances, test_instances


def _evaluate_source_on_target(
    *,
    problem: str,
    source_candidate_dir: Path,
    target_dataset_dir: Path,
    label: str,
) -> dict[str, object]:
    manifest, train_instances, test_instances = _dataset_payload(target_dataset_dir)
    analysis = run_analysis(
        source_candidate_dir,
        train_instances,
        manifest=manifest,
        artifact_dir=None,
    )
    solver = build_solver(
        source_candidate_dir,
        analysis=analysis,
        manifest=manifest,
    )
    summary = evaluate_solver(
        problem,
        label,
        solver,
        test_instances,
        split="test",
        feedback_limit=0,
    )
    return summary


def _evaluate_task(task: dict[str, object]) -> dict[str, object]:
    problem = str(task["problem"])
    source_candidate_dir = Path(str(task["source_candidate_dir"]))
    target_dataset_dir = Path(str(task["target_dataset_dir"]))
    summary = _evaluate_source_on_target(
        problem=problem,
        source_candidate_dir=source_candidate_dir,
        target_dataset_dir=target_dataset_dir,
        label=str(task["evaluation_label"]),
    )
    return {
        "problem": problem,
        "family": str(task["family"]),
        "source_label": str(task["source_label"]),
        "source_condition_id": str(task["source_condition_id"]),
        "target_condition_id": str(task["target_condition_id"]),
        "point": _point_from_summary(
            source_label=str(task["source_label"]),
            source_row=dict(task["source_row"]),
            target_row=dict(task["target_row"]),
            summary=summary,
        ),
    }


def _target_point(row: dict[str, object]) -> dict[str, object]:
    return {
        "condition_id": str(row["condition_id"]),
        "primary_size_param": row.get("primary_size_param"),
        "primary_size_value": row.get("primary_size_value"),
        "instance_params": row.get("instance_params", {}),
        "dataset_dir": str(row["dataset_dir"]),
    }


def _point_from_summary(
    *,
    source_label: str,
    source_row: dict[str, object],
    target_row: dict[str, object],
    summary: dict[str, object],
) -> dict[str, object]:
    return {
        "source_label": source_label,
        "source_condition_id": str(source_row["condition_id"]),
        "target_condition_id": str(target_row["condition_id"]),
        "target_primary_size_value": target_row.get("primary_size_value"),
        "target_instance_params": target_row.get("instance_params", {}),
        "runtime_ms": summary.get("average_runtime_ms"),
        "runtime_s": (
            float(summary["average_runtime_ms"]) / 1000.0
            if isinstance(summary.get("average_runtime_ms"), (int, float))
            else None
        ),
        "quality": summary.get("average_normalized_quality"),
        "optimality_rate": summary.get("optimality_rate"),
        "feasibility_rate": summary.get("feasibility_rate"),
        "error": summary.get("error"),
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _plot_family(
    *,
    family_payload: dict[str, object],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.8, 4.8))
    colors = {
        "low": "#2563eb",
        "medium": "#ea580c",
        "high": "#059669",
    }

    all_x_values: list[float] = []
    for source_payload in family_payload.get("source_solvers", []):
        if not isinstance(source_payload, dict):
            continue
        label = str(source_payload["label"])
        points = source_payload.get("points", [])
        if not isinstance(points, list):
            continue
        ordered = sorted(
            (
                (
                    float(point["target_primary_size_value"]),
                    float(point["runtime_s"]),
                )
                for point in points
                if isinstance(point, dict)
                and isinstance(point.get("target_primary_size_value"), (int, float))
                and isinstance(point.get("runtime_s"), (int, float))
            ),
            key=lambda item: item[0],
        )
        if not ordered:
            continue
        x_values = [item[0] for item in ordered]
        y_values = [item[1] for item in ordered]
        all_x_values.extend(x_values)
        source_size = source_payload.get("source_primary_size_value")
        legend_label = f"{label} source ({family_payload.get('primary_size_param')}={source_size:g})"
        axis.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.4,
            markersize=5.5,
            color=colors.get(label, None),
            label=legend_label,
        )

    axis.set_title(f"Problem Size Transfer: {family_payload['problem']} / {family_payload['family']}")
    axis.set_xlabel(str(family_payload.get("primary_size_param") or "problem_size"))
    axis.set_ylabel("Test Runtime (s)")
    axis.grid(True, alpha=0.25)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if all_x_values:
        unique_x = sorted(set(all_x_values))
        axis.set_xticks(unique_x)
        axis.set_xticklabels([str(int(value)) if float(value).is_integer() else f"{value:g}" for value in unique_x])
        if len(unique_x) > 1:
            axis.set_xlim(min(unique_x), max(unique_x))
    handles, labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(frameon=False)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate_transfer_curves(
    *,
    experiment_root: Path,
    source_condition_ids: list[str] | None,
    output_json: Path,
    output_plot_dir: Path,
    force: bool,
    problem_filter: str | None,
    family_filter: str | None,
    max_workers: int,
) -> dict[str, object]:
    rows = _load_aggregate_rows(experiment_root)
    grouped = _group_rows(rows)

    existing_cache: dict[str, object] = {}
    if output_json.exists() and not force:
        cached = _read_json(output_json)
        if isinstance(cached, dict):
            existing_cache = cached

    result: dict[str, object] = {
        "generated_at": _utc_now(),
        "experiment_root": str(experiment_root),
        "selection_method": {
            "description": (
                "Low/medium/high synthesized solvers are selected from the first, midpoint, "
                "and last completed size conditions unless source_condition_ids is provided."
            ),
            "source_condition_ids": source_condition_ids,
        },
        "families": [],
    }
    family_payloads: dict[tuple[str, str], dict[str, object]] = {}
    source_payloads: dict[tuple[str, str, str, str], dict[str, object]] = {}
    pending_tasks: list[dict[str, object]] = []

    for (problem, family), family_rows in sorted(grouped.items()):
        if problem_filter is not None and problem != problem_filter:
            continue
        if family_filter is not None and family != family_filter:
            continue
        selected_sources = _select_source_rows(family_rows, source_condition_ids=source_condition_ids)
        family_payload: dict[str, object] = {
            "problem": problem,
            "family": family,
            "primary_size_param": family_rows[0].get("primary_size_param"),
            "target_points": [_target_point(row) for row in family_rows],
            "source_solvers": [],
        }
        family_payloads[(problem, family)] = family_payload
        result["families"].append(family_payload)

        for source_label, source_row in selected_sources:
            source_info = _source_summary(source_row)
            source_candidate_dir = Path(str(source_row["agent_run_dir"])) / "candidates" / str(source_info["slug"])
            source_payload: dict[str, object] = {
                "label": source_label,
                "source_condition_id": str(source_row["condition_id"]),
                "source_primary_size_value": source_row.get("primary_size_value"),
                "source_instance_params": source_row.get("instance_params", {}),
                "source_candidate_slug": source_info["slug"],
                "source_candidate_dir": str(source_candidate_dir),
                "points": [],
            }
            family_payload["source_solvers"].append(source_payload)
            source_payloads[(problem, family, source_label, str(source_row["condition_id"]))] = source_payload

            for target_row in family_rows:
                cached_point = None if force else _cached_entry(
                    existing_cache,
                    problem=problem,
                    family=family,
                    source_label=source_label,
                    source_condition_id=str(source_row["condition_id"]),
                    target_condition_id=str(target_row["condition_id"]),
                )
                if cached_point is not None:
                    source_payload["points"].append(cached_point)
                    continue

                pending_tasks.append(
                    {
                        "problem": problem,
                        "family": family,
                        "source_label": source_label,
                        "source_condition_id": str(source_row["condition_id"]),
                        "source_row": source_row,
                        "target_condition_id": str(target_row["condition_id"]),
                        "target_row": target_row,
                        "source_candidate_dir": str(source_candidate_dir),
                        "target_dataset_dir": str(target_row["dataset_dir"]),
                        "evaluation_label": f"{family}:{source_label}:{source_row['condition_id']}",
                    }
                )

    def _sort_points() -> None:
        for source_payload in source_payloads.values():
            points = source_payload["points"]
            assert isinstance(points, list)
            points.sort(
                key=lambda point: (
                    float(point["target_primary_size_value"])
                    if isinstance(point.get("target_primary_size_value"), (int, float))
                    else float("inf"),
                    str(point["target_condition_id"]),
                )
            )

    def _flush_outputs() -> None:
        _sort_points()
        _write_json(output_json, result)
        for family_payload in result["families"]:
            assert isinstance(family_payload, dict)
            plot_path = output_plot_dir / f"{family_payload['problem']}__{family_payload['family']}__solver_transfer_runtime.png"
            _plot_family(family_payload=family_payload, output_path=plot_path)

    _flush_outputs()

    if pending_tasks:
        if max_workers <= 1:
            for task in pending_tasks:
                result_payload = _evaluate_task(task)
                source_payload = source_payloads[
                    (
                        result_payload["problem"],
                        result_payload["family"],
                        result_payload["source_label"],
                        result_payload["source_condition_id"],
                    )
                ]
                points = source_payload["points"]
                assert isinstance(points, list)
                points.append(result_payload["point"])
                _flush_outputs()
        else:
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_task = {
                    executor.submit(_evaluate_task, task): task
                    for task in pending_tasks
                }
                for future in concurrent.futures.as_completed(future_to_task):
                    result_payload = future.result()
                    source_payload = source_payloads[
                        (
                            result_payload["problem"],
                            result_payload["family"],
                            result_payload["source_label"],
                            result_payload["source_condition_id"],
                        )
                    ]
                    points = source_payload["points"]
                    assert isinstance(points, list)
                    points.append(result_payload["point"])
                    _flush_outputs()

    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Take the low/medium/high synthesized solvers from a completed problem-size sweep, "
            "run each solver across all problem sizes for its family, and write plots plus a JSON cache."
        )
    )
    parser.add_argument(
        "experiment_root",
        type=Path,
        help="Problem-size sweep root, for example artifacts/problem_size_sweep/20260501_012438",
    )
    parser.add_argument(
        "--source-condition-ids",
        default=None,
        help="Optional comma-separated condition ids for low,medium,high (for example size_01,size_05,size_10).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <experiment_root>/solver_transfer_runtime_curves.json",
    )
    parser.add_argument(
        "--output-plot-dir",
        type=Path,
        default=None,
        help="Output plot directory. Defaults to <experiment_root>/plots/solver_transfer_runtime",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore any existing JSON cache and recompute all evaluations.",
    )
    parser.add_argument("--problem", default=None, help="Optional problem filter for one family group.")
    parser.add_argument("--family", default=None, help="Optional family filter.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel worker count for cross-size solver evaluations.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    experiment_root = args.experiment_root.resolve()
    if not experiment_root.exists():
        raise SystemExit(f"Experiment root does not exist: {experiment_root}")

    source_condition_ids = None
    if args.source_condition_ids:
        source_condition_ids = [part.strip() for part in str(args.source_condition_ids).split(",") if part.strip()]
        if len(source_condition_ids) != 3:
            raise SystemExit("Expected exactly three condition ids in --source-condition-ids.")

    output_json = (
        args.output_json.resolve()
        if args.output_json is not None
        else experiment_root / "solver_transfer_runtime_curves.json"
    )
    output_plot_dir = (
        args.output_plot_dir.resolve()
        if args.output_plot_dir is not None
        else experiment_root / "plots" / "solver_transfer_runtime"
    )

    payload = evaluate_transfer_curves(
        experiment_root=experiment_root,
        source_condition_ids=source_condition_ids,
        output_json=output_json,
        output_plot_dir=output_plot_dir,
        force=args.force,
        problem_filter=args.problem,
        family_filter=args.family,
        max_workers=args.max_workers,
    )

    print(f"Wrote JSON: {output_json}")
    for family_payload in payload.get("families", []):
        if not isinstance(family_payload, dict):
            continue
        plot_path = output_plot_dir / f"{family_payload['problem']}__{family_payload['family']}__solver_transfer_runtime.png"
        print(f"Plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
