from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _best_runtime_trace(timing_report: dict[str, object]) -> tuple[list[int], list[float]]:
    stages = timing_report.get("stages", {})
    if not isinstance(stages, dict):
        raise ValueError("Timing report is missing `stages`.")
    synthesis = stages.get("synthesis", {})
    if not isinstance(synthesis, dict):
        raise ValueError("Timing report is missing synthesis stage data.")
    rounds = synthesis.get("rounds", [])
    candidates = synthesis.get("candidates", {})
    if not isinstance(rounds, list) or not isinstance(candidates, dict):
        raise ValueError("Timing report has invalid synthesis round data.")

    points: list[tuple[int, float]] = []
    best_so_far: float | None = None
    for round_payload in sorted(
        (row for row in rounds if isinstance(row, dict)),
        key=lambda row: int(row.get("iteration", 0)),
    ):
        slug = round_payload.get("best_selected_slug")
        candidate = candidates.get(str(slug))
        if not isinstance(candidate, dict):
            continue
        selection = candidate.get("selection", {})
        if not isinstance(selection, dict):
            continue
        runtime_ms = selection.get("validation_runtime_ms", selection.get("mean_runtime_ms"))
        if not isinstance(runtime_ms, (int, float)):
            continue
        best_so_far = float(runtime_ms) if best_so_far is None else min(best_so_far, float(runtime_ms))
        points.append((int(round_payload.get("iteration", 0)) + 1, best_so_far))

    if not points:
        raise ValueError("No round runtime data found in timing report.")
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    return x_values, y_values


def _write_plot(
    *,
    path: Path,
    problem: str,
    family: str,
    x_values: list[int],
    y_values: list[float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    axis.plot(x_values, y_values, marker="o", linewidth=2.4, color="#2563eb")
    axis.set_title(f"Iteration Count Sweep: {problem} / {family}")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Best Validation Runtime So Far (ms)")
    axis.set_xticks(x_values)
    axis.grid(True, alpha=0.25)
    axis.set_xlim(min(x_values), max(x_values))
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_iteration_best_runtime_so_far_plots(
    *,
    sweep_root: Path,
    condition_id: str = "iterations_05",
    output_dir: Path | None = None,
) -> list[Path]:
    sweep_root = sweep_root.resolve()
    condition_root = sweep_root / "targets" / str(condition_id)
    if not condition_root.exists():
        raise ValueError(f"Condition directory not found: {condition_root}")

    resolved_output_dir = (
        output_dir.resolve()
        if output_dir is not None
        else sweep_root / "plots" / "iteration_best_runtime_so_far"
    )
    written: list[Path] = []
    for timing_path in sorted(condition_root.glob("*/*/agent_run/timing_report.json")):
        family = timing_path.parent.parent.name
        problem = timing_path.parent.parent.parent.name
        timing_report = _load_json(timing_path)
        x_values, y_values = _best_runtime_trace(timing_report)
        plot_path = resolved_output_dir / f"{problem}__{family}__best_runtime_so_far.png"
        _write_plot(
            path=plot_path,
            problem=problem,
            family=family,
            x_values=x_values,
            y_values=y_values,
        )
        written.append(plot_path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cumulative best-runtime-per-iteration plots from the "
            "`iterations_05` timing reports of an iteration-count sweep."
        )
    )
    parser.add_argument("sweep_root", type=Path, help="Path to the iteration_count_sweep run root.")
    parser.add_argument(
        "--condition-id",
        default="iterations_05",
        help="Condition id to read timing reports from. Defaults to `iterations_05`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for plots. Defaults to <sweep_root>/plots/iteration_best_runtime_so_far.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        written = write_iteration_best_runtime_so_far_plots(
            sweep_root=args.sweep_root,
            condition_id=args.condition_id,
            output_dir=args.output_dir,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
