from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize_selection(
    train_summary: dict[str, object],
    validation_summary: dict[str, object],
) -> dict[str, float]:
    train_quality = float(train_summary["average_normalized_quality"])
    validation_quality = float(validation_summary["average_normalized_quality"])
    train_optimality = float(train_summary["optimality_rate"])
    validation_optimality = float(validation_summary["optimality_rate"])
    train_runtime = float(train_summary["average_runtime_ms"])
    validation_runtime = float(validation_summary["average_runtime_ms"])
    return {
        "validation_normalized_quality": validation_quality,
        "validation_optimality_rate": validation_optimality,
        "validation_runtime_ms": validation_runtime,
        "mean_normalized_quality": validation_quality,
        "mean_optimality_rate": validation_optimality,
        "mean_runtime_ms": validation_runtime,
        "train_validation_gap": train_quality - validation_quality,
        "train_normalized_quality": train_quality,
        "train_optimality_rate": train_optimality,
        "train_runtime_ms": train_runtime,
    }


def selection_sort_key(selection: dict[str, object]) -> tuple[float, float, float]:
    normalized_quality = selection.get(
        "validation_normalized_quality",
        selection.get("mean_normalized_quality", 0.0),
    )
    optimality_rate = selection.get(
        "validation_optimality_rate",
        selection.get("mean_optimality_rate", 0.0),
    )
    runtime_ms = selection.get(
        "validation_runtime_ms",
        selection.get("mean_runtime_ms", float("inf")),
    )
    return (
        float(normalized_quality),
        float(optimality_rate),
        -float(runtime_ms),
    )


def progress_point(iteration: int, record: dict[str, object]) -> dict[str, object]:
    return {
        "iteration": iteration,
        "slug": record["slug"],
        "selection": record["selection"],
        "train": record["train"],
        "validation": record["validation"],
    }


def write_history(path: Path, history: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def performance_plot_filename() -> str:
    return "performance_curve.png"


def write_performance_plot(path: Path, history: list[dict[str, object]], *, title: str) -> Path:
    path = path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    iterations = [int(point["iteration"]) for point in history]
    train_values = [float(point["train"]["average_normalized_quality"]) for point in history]
    validation_values = [float(point["validation"]["average_normalized_quality"]) for point in history]

    figure, axis = plt.subplots(figsize=(8, 4.6), dpi=160)
    axis.plot(iterations, train_values, marker="o", linewidth=2.2, color="#2563eb", label="train")
    axis.plot(iterations, validation_values, marker="o", linewidth=2.2, color="#dc2626", label="validation")
    axis.set_title(title)
    axis.set_xlabel("iteration")
    axis.set_ylabel("average normalized quality")
    axis.set_ylim(max(0.0, min(train_values + validation_values) - 0.01), 1.001)
    axis.grid(alpha=0.3, linestyle="--", linewidth=0.8)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    return path
