from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from pprint import pformat

from dasbench.agents.candidate import build_solver, run_analysis
from dasbench.agents.progress import (
    performance_plot_filename,
    progress_point,
    selection_sort_key,
    summarize_selection,
    write_history,
    write_performance_plot,
)
from dasbench.data import load_manifest, load_split
from dasbench.eval.evaluator import evaluate_solver, write_summary
from dasbench.timing import BenchmarkTimingReporter


@dataclass(frozen=True)
class TemplateCandidateSpec:
    problem: str
    solver_strategy: str
    analysis_strategy: str = "default"
    mix_with_instance_polarity: bool = False
    pattern_confidence_threshold: float = 0.0
    local_search_flips: int = 0
    two_opt_rounds: int = 0
    density_threshold: float = 0.22
    aspect_ratio_threshold: float = 1.35
    prune_redundant: bool = True

    def slug(self) -> str:
        return (
            f"{self.problem}"
            f"__{self.solver_strategy}"
            f"__mix-{int(self.mix_with_instance_polarity)}"
            f"__th-{str(self.pattern_confidence_threshold).replace('.', 'p')}"
            f"__flips-{self.local_search_flips}"
            f"__2opt-{self.two_opt_rounds}"
            f"__dens-{str(self.density_threshold).replace('.', 'p')}"
            f"__aspect-{str(self.aspect_ratio_threshold).replace('.', 'p')}"
            f"__prune-{int(self.prune_redundant)}"
        )

    def to_config(self) -> dict[str, object]:
        return {
            "problem": self.problem,
            "solver_strategy": self.solver_strategy,
            "analysis_strategy": self.analysis_strategy,
            "mix_with_instance_polarity": self.mix_with_instance_polarity,
            "pattern_confidence_threshold": self.pattern_confidence_threshold,
            "local_search_flips": self.local_search_flips,
            "two_opt_rounds": self.two_opt_rounds,
            "density_threshold": self.density_threshold,
            "aspect_ratio_threshold": self.aspect_ratio_threshold,
            "prune_redundant": self.prune_redundant,
        }


def _candidate_source(kind: str, config: dict[str, object]) -> str:
    config_text = pformat(config, sort_dicts=True)
    if kind == "analyze":
        return (
            "from dasbench.agents.template_runtime import analyze_with_config\n\n"
            f"CONFIG = {config_text}\n\n"
            "def analyze(train_instances, manifest=None):\n"
            "    return analyze_with_config(train_instances, manifest, CONFIG)\n"
        )
    return (
        "from dasbench.agents.template_runtime import solve_with_config\n\n"
        f"CONFIG = {config_text}\n\n"
        "def solve(instance, analysis=None, manifest=None):\n"
        "    return solve_with_config(instance, analysis, manifest, CONFIG)\n"
    )


def write_candidate(candidate_dir: Path, spec: TemplateCandidateSpec) -> None:
    candidate_dir.mkdir(parents=True, exist_ok=True)
    config = spec.to_config()
    (candidate_dir / "analyze.py").write_text(_candidate_source("analyze", config), encoding="utf-8")
    (candidate_dir / "solution.py").write_text(_candidate_source("solution", config), encoding="utf-8")
    (candidate_dir / "candidate_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _effective_candidate_width(mode: str, candidate_width: int | None, beam_width: int) -> int:
    if mode == "single":
        return 1
    if candidate_width is None:
        return max(1, beam_width)
    return max(1, int(candidate_width))


def _seed_specs(
    problem: str,
    mode: str,
    *,
    candidate_width: int | None = None,
    beam_width: int = 3,
) -> list[TemplateCandidateSpec]:
    if problem == "maxsat":
        seeds = [
            TemplateCandidateSpec(problem="maxsat", solver_strategy="instance_polarity"),
            TemplateCandidateSpec(problem="maxsat", solver_strategy="global_bias"),
            TemplateCandidateSpec(
                problem="maxsat",
                solver_strategy="signature_lookup",
                mix_with_instance_polarity=True,
                pattern_confidence_threshold=0.08,
            ),
            TemplateCandidateSpec(
                problem="maxsat",
                solver_strategy="signature_lookup",
                mix_with_instance_polarity=True,
                pattern_confidence_threshold=0.12,
                local_search_flips=18,
            ),
        ]
    elif problem == "mis":
        seeds = [
            TemplateCandidateSpec(problem="mis", solver_strategy="random_greedy"),
            TemplateCandidateSpec(problem="mis", solver_strategy="min_degree"),
            TemplateCandidateSpec(problem="mis", solver_strategy="ratio_greedy"),
            TemplateCandidateSpec(problem="mis", solver_strategy="density_adaptive", density_threshold=0.20),
            TemplateCandidateSpec(problem="mis", solver_strategy="local_improve"),
        ]
    elif problem == "mds":
        seeds = [
            TemplateCandidateSpec(problem="mds", solver_strategy="high_degree"),
            TemplateCandidateSpec(problem="mds", solver_strategy="marginal_gain"),
            TemplateCandidateSpec(problem="mds", solver_strategy="redundancy_aware", prune_redundant=True),
            TemplateCandidateSpec(problem="mds", solver_strategy="overlap_hybrid", density_threshold=0.22, prune_redundant=True),
        ]
    elif problem == "coloring":
        seeds = [
            TemplateCandidateSpec(problem="coloring", solver_strategy="dsatur"),
            TemplateCandidateSpec(problem="coloring", solver_strategy="largest_degree"),
            TemplateCandidateSpec(problem="coloring", solver_strategy="smallest_last"),
            TemplateCandidateSpec(problem="coloring", solver_strategy="density_adaptive", density_threshold=0.40),
            TemplateCandidateSpec(problem="coloring", solver_strategy="random_greedy"),
        ]
    elif problem == "tsp":
        seeds = [
            TemplateCandidateSpec(problem="tsp", solver_strategy="structure_adaptive", two_opt_rounds=3, aspect_ratio_threshold=1.35),
            TemplateCandidateSpec(problem="tsp", solver_strategy="nearest_insertion"),
            TemplateCandidateSpec(problem="tsp", solver_strategy="farthest_insertion"),
            TemplateCandidateSpec(problem="tsp", solver_strategy="nearest_neighbor"),
            TemplateCandidateSpec(problem="tsp", solver_strategy="random"),
        ]
    elif problem == "packing_lp":
        seeds = [
            TemplateCandidateSpec(problem="packing_lp", solver_strategy="tight_resource_density"),
            TemplateCandidateSpec(problem="packing_lp", solver_strategy="density_fractional"),
            TemplateCandidateSpec(problem="packing_lp", solver_strategy="uniform_fraction"),
            TemplateCandidateSpec(problem="packing_lp", solver_strategy="lp_relaxation"),
        ]
    elif problem == "mdkp":
        seeds = [
            TemplateCandidateSpec(problem="mdkp", solver_strategy="tight_resource_greedy"),
            TemplateCandidateSpec(problem="mdkp", solver_strategy="density_greedy"),
            TemplateCandidateSpec(problem="mdkp", solver_strategy="redundancy_improved"),
            TemplateCandidateSpec(problem="mdkp", solver_strategy="lp_rounding"),
        ]
    else:
        raise ValueError(f"Unsupported template problem: {problem}")
    return seeds[: _effective_candidate_width(mode, candidate_width, beam_width)]


def _read_json_if_exists(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _load_saved_template_record(candidate_dir: Path, evaluation_dir: Path) -> dict[str, object] | None:
    candidate_config = _read_json_if_exists(candidate_dir / "candidate_config.json")
    train_summary = _read_json_if_exists(evaluation_dir / "train_summary.json")
    validation_summary = _read_json_if_exists(evaluation_dir / "validation_summary.json")
    analysis_output = _read_json_if_exists(evaluation_dir / "analysis.json")
    if candidate_config is None or train_summary is None or validation_summary is None:
        return None
    return {
        "slug": candidate_dir.name,
        "spec": candidate_config,
        "candidate_dir": str(candidate_dir),
        "evaluation_dir": str(evaluation_dir),
        "analysis_output": analysis_output,
        "train": train_summary,
        "validation": validation_summary,
        "selection": summarize_selection(train_summary, validation_summary),
        "timing": {},
    }


def _load_saved_template_records(candidates_dir: Path, evaluations_dir: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    if not candidates_dir.exists() or not evaluations_dir.exists():
        return records
    for candidate_dir in sorted(path for path in candidates_dir.iterdir() if path.is_dir()):
        record = _load_saved_template_record(candidate_dir, evaluations_dir / candidate_dir.name)
        if record is not None:
            records[candidate_dir.name] = record
    return records


def _mutations(
    spec: TemplateCandidateSpec,
    train_summary: dict[str, object],
    validation_summary: dict[str, object],
) -> list[TemplateCandidateSpec]:
    mutations: list[TemplateCandidateSpec] = []
    effective_quality = min(
        float(train_summary["average_normalized_quality"]),
        float(validation_summary["average_normalized_quality"]),
    )
    effective_runtime = max(
        float(train_summary["average_runtime_ms"]),
        float(validation_summary["average_runtime_ms"]),
    )
    if spec.problem == "maxsat":
        if spec.solver_strategy == "global_bias":
            mutations.append(replace(spec, solver_strategy="signature_lookup"))
        if spec.solver_strategy == "signature_lookup" and spec.local_search_flips < 36 and effective_quality < 0.995:
            mutations.append(replace(spec, local_search_flips=spec.local_search_flips + 12))
        if spec.mix_with_instance_polarity and spec.pattern_confidence_threshold < 0.2:
            mutations.append(
                replace(spec, pattern_confidence_threshold=round(spec.pattern_confidence_threshold + 0.04, 2))
            )
    elif spec.problem == "mis":
        if spec.solver_strategy == "min_degree":
            mutations.append(replace(spec, solver_strategy="local_improve"))
        if spec.solver_strategy == "ratio_greedy":
            mutations.append(replace(spec, solver_strategy="density_adaptive", density_threshold=max(0.14, spec.density_threshold - 0.02)))
        if effective_runtime < 10.0 and effective_quality < 1.0:
            mutations.append(replace(spec, solver_strategy="local_improve"))
    elif spec.problem == "mds":
        if spec.solver_strategy == "high_degree":
            mutations.append(replace(spec, solver_strategy="marginal_gain"))
        if spec.solver_strategy == "marginal_gain":
            mutations.append(replace(spec, solver_strategy="overlap_hybrid"))
        if effective_quality < 1.0 and not spec.prune_redundant:
            mutations.append(replace(spec, prune_redundant=True))
        if effective_runtime > 5.0 and spec.solver_strategy == "overlap_hybrid":
            mutations.append(replace(spec, solver_strategy="marginal_gain"))
    elif spec.problem == "coloring":
        if spec.solver_strategy == "random_greedy":
            mutations.append(replace(spec, solver_strategy="largest_degree"))
        if spec.solver_strategy == "largest_degree":
            mutations.append(replace(spec, solver_strategy="smallest_last"))
        if spec.solver_strategy == "smallest_last":
            mutations.append(replace(spec, solver_strategy="dsatur"))
        if spec.solver_strategy == "density_adaptive" and effective_quality < 1.0:
            mutations.append(replace(spec, solver_strategy="dsatur"))
    elif spec.problem == "tsp":
        if spec.solver_strategy == "random":
            mutations.append(replace(spec, solver_strategy="nearest_neighbor"))
        if spec.solver_strategy == "nearest_neighbor":
            mutations.append(replace(spec, solver_strategy="structure_adaptive", two_opt_rounds=max(2, spec.two_opt_rounds)))
        if spec.solver_strategy in {"nearest_insertion", "farthest_insertion"}:
            mutations.append(replace(spec, solver_strategy="two_opt_nearest", two_opt_rounds=max(3, spec.two_opt_rounds)))
        if spec.solver_strategy in {"structure_adaptive", "two_opt_nearest"} and spec.two_opt_rounds < 8 and effective_quality < 0.999:
            mutations.append(replace(spec, two_opt_rounds=spec.two_opt_rounds + 2))
    elif spec.problem == "packing_lp":
        if spec.solver_strategy == "uniform_fraction":
            mutations.append(replace(spec, solver_strategy="density_fractional"))
        if spec.solver_strategy == "density_fractional":
            mutations.append(replace(spec, solver_strategy="tight_resource_density"))
        if effective_runtime < 10.0 and effective_quality < 0.995:
            mutations.append(replace(spec, solver_strategy="lp_relaxation"))
    elif spec.problem == "mdkp":
        if spec.solver_strategy == "density_greedy":
            mutations.append(replace(spec, solver_strategy="tight_resource_greedy"))
        if spec.solver_strategy in {"density_greedy", "tight_resource_greedy"}:
            mutations.append(replace(spec, solver_strategy="redundancy_improved"))
        if effective_runtime < 10.0 and effective_quality < 0.98:
            mutations.append(replace(spec, solver_strategy="lp_rounding"))
    return mutations


def _evaluate_spec(
    spec: TemplateCandidateSpec,
    *,
    manifest: dict[str, object],
    train_instances_public: list[dict[str, object]],
    train_instances_full: list[dict[str, object]],
    validation_instances_full: list[dict[str, object]],
    candidates_dir: Path,
    evaluations_dir: Path,
) -> dict[str, object]:
    candidate_start = time.perf_counter()
    timing: dict[str, float] = {}
    candidate_dir = candidates_dir / spec.slug()
    evaluation_dir = evaluations_dir / spec.slug()
    write_candidate(candidate_dir, spec)
    analysis_start = time.perf_counter()
    analysis = run_analysis(
        candidate_dir,
        train_instances_public,
        manifest=manifest,
        artifact_dir=evaluation_dir,
    )
    timing["analysis_execution_wall_ms"] = (time.perf_counter() - analysis_start) * 1000.0
    solver_build_start = time.perf_counter()
    solver = build_solver(candidate_dir, analysis=analysis, manifest=manifest)
    timing["solver_build_wall_ms"] = (time.perf_counter() - solver_build_start) * 1000.0
    train_eval_start = time.perf_counter()
    train_summary = evaluate_solver(
        spec.problem,
        spec.slug(),
        solver,
        train_instances_full,
        split="train",
    )
    timing["train_eval_wall_ms"] = (time.perf_counter() - train_eval_start) * 1000.0
    validation_eval_start = time.perf_counter()
    validation_summary = evaluate_solver(
        spec.problem,
        spec.slug(),
        solver,
        validation_instances_full,
        split="validation",
    )
    timing["validation_eval_wall_ms"] = (time.perf_counter() - validation_eval_start) * 1000.0
    timing["candidate_wall_ms"] = (time.perf_counter() - candidate_start) * 1000.0
    selection = summarize_selection(train_summary, validation_summary)
    write_summary(evaluation_dir / "train_summary.json", train_summary)
    write_summary(evaluation_dir / "validation_summary.json", validation_summary)
    return {
        "slug": spec.slug(),
        "spec": spec.to_config(),
        "candidate_dir": str(candidate_dir),
        "evaluation_dir": str(evaluation_dir),
        "analysis_output": analysis,
        "train": train_summary,
        "validation": validation_summary,
        "selection": selection,
        "timing": timing,
    }


def run_template_synthesis_loop(
    dataset_dir: Path,
    output_dir: Path,
    *,
    mode: str = "single",
    iterations: int = 3,
    beam_width: int = 3,
    candidate_width: int | None = None,
    timing_reporter: BenchmarkTimingReporter | None = None,
) -> dict[str, object]:
    manifest = load_manifest(dataset_dir)
    problem = str(manifest["problem"])
    train_full = load_split(dataset_dir, "train")
    train_public = load_split(dataset_dir, "train", public=True)
    validation_full = load_split(dataset_dir, "validation")
    test_full = load_split(dataset_dir, "test")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = output_dir / "candidates"
    evaluations_dir = output_dir / "evaluations"

    effective_candidate_width = _effective_candidate_width(mode, candidate_width, beam_width)
    frontier = _seed_specs(problem, mode, candidate_width=effective_candidate_width, beam_width=beam_width)
    evaluated: dict[str, dict[str, object]] = _load_saved_template_records(candidates_dir, evaluations_dir)
    rounds: list[dict[str, object]] = []
    history: list[dict[str, object]] = []

    for iteration in range(iterations):
        if not frontier:
            break
        current_round: list[dict[str, object]] = []
        for spec in frontier:
            if spec.slug() not in evaluated:
                evaluated[spec.slug()] = _evaluate_spec(
                    spec,
                    manifest=manifest,
                    train_instances_public=train_public,
                    train_instances_full=train_full,
                    validation_instances_full=validation_full,
                    candidates_dir=candidates_dir,
                    evaluations_dir=evaluations_dir,
                )
                if timing_reporter is not None:
                    timing_reporter.record_synthesis_candidate(evaluated[spec.slug()])
            current_round.append(evaluated[spec.slug()])

        ranked = sorted(
            evaluated.values(),
            key=lambda record: selection_sort_key(record["selection"]),
            reverse=True,
        )
        survivors = ranked[: 1 if mode == "single" else beam_width]
        history.append(progress_point(iteration, survivors[0]))
        rounds.append(
            {
                "iteration": iteration,
                "evaluated_this_round": [record["slug"] for record in current_round],
                "frontier_after_ranking": [record["slug"] for record in survivors],
                "best_selected_slug": survivors[0]["slug"],
                "best_selected_train": survivors[0]["train"],
                "best_selected_validation": survivors[0]["validation"],
                "best_selected_selection": survivors[0]["selection"],
            }
        )
        if timing_reporter is not None:
            timing_reporter.record_synthesis_round(rounds[-1])
        if iteration == iterations - 1:
            break
        next_specs: list[TemplateCandidateSpec] = []
        for survivor in survivors:
            spec = TemplateCandidateSpec(**survivor["spec"])
            next_specs.extend(_mutations(spec, survivor["train"], survivor["validation"]))
        deduped: dict[str, TemplateCandidateSpec] = {}
        for spec in next_specs:
            deduped.setdefault(spec.slug(), spec)
        frontier = list(deduped.values())[:effective_candidate_width]

    best_candidate = max(
        evaluated.values(),
        key=lambda record: selection_sort_key(record["selection"]),
    )
    analysis = best_candidate.get("analysis_output")
    if analysis is None:
        analysis = _read_json_if_exists(Path(best_candidate["evaluation_dir"]) / "analysis.json")
    if analysis is None:
        analysis = run_analysis(
            Path(best_candidate["candidate_dir"]),
            train_public,
            manifest=manifest,
            artifact_dir=output_dir / "best_candidate_analysis",
        )
    else:
        analysis_dir = output_dir / "best_candidate_analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        (analysis_dir / "analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    solver = build_solver(
        Path(best_candidate["candidate_dir"]),
        analysis=analysis,
        manifest=manifest,
    )
    best_test_start = time.perf_counter()
    best_candidate["test"] = evaluate_solver(problem, best_candidate["slug"], solver, test_full, split="test")
    best_candidate["timing"] = dict(best_candidate.get("timing", {}))
    best_candidate["timing"]["best_candidate_test_wall_ms"] = (time.perf_counter() - best_test_start) * 1000.0
    write_summary(Path(best_candidate["evaluation_dir"]) / "test_summary.json", best_candidate["test"])
    if timing_reporter is not None:
        timing_reporter.record_best_candidate_test(
            wall_ms=best_candidate["timing"]["best_candidate_test_wall_ms"],
            summary=best_candidate["test"],
            slug=str(best_candidate["slug"]),
        )

    history_path = output_dir / "performance_history.json"
    write_history(history_path, history)
    plot_path = write_performance_plot(
        output_dir / performance_plot_filename(),
        history,
        title=f"{problem} template search",
    )

    summary = {
        "problem": problem,
        "family": manifest["family"],
        "generator": "template",
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "mode": mode,
        "iterations": iterations,
        "beam_width": beam_width,
        "candidate_width": effective_candidate_width,
        "best_candidate": best_candidate,
        "rounds": rounds,
        "performance_history_path": str(history_path),
        "performance_plot_path": str(plot_path),
    }
    write_summary(output_dir / "synthesis_summary.json", summary)
    return summary
