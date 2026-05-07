from __future__ import annotations

from pathlib import Path

from dasbench.integrations.external_exact import (
    ExternalExactConfig,
    build_external_exact_solvers,
    external_diagnostics_path,
    write_external_discovery,
)
from dasbench.integrations.gurobi_baseline import GurobiBaselineConfig, build_gurobi_solver
from dasbench.integrations.native_exact import NativeExactConfig, wrap_native_exact_baselines
from dasbench.problems import get_problem_definition


def resolve_baselines(
    problem_name: str,
    *,
    gurobi_config: GurobiBaselineConfig,
    native_exact_config: NativeExactConfig | None = None,
    external_config: ExternalExactConfig | None = None,
    artifact_dir: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    problem = get_problem_definition(problem_name)
    resolved_native_config = native_exact_config or NativeExactConfig()
    baselines = wrap_native_exact_baselines(dict(problem.baseline_registry()), resolved_native_config)
    if gurobi_config.enabled:
        baselines[gurobi_config.baseline_name] = build_gurobi_solver(problem_name, gurobi_config)
    resolved_external_config = external_config or ExternalExactConfig()
    external_solvers: dict[str, object] = {}
    external_discovery = {
        "mode": resolved_external_config.mode,
        "problem": problem_name,
        "solvers": [],
    }
    if artifact_dir is not None:
        external_solvers, external_discovery = build_external_exact_solvers(
            problem_name,
            resolved_external_config,
            artifact_dir=artifact_dir / "external_exact_logs",
        )
        write_external_discovery(artifact_dir, external_discovery)
    baselines.update(external_solvers)
    return baselines, external_discovery


def gurobi_diagnostics_path(output_dir: Path, *, split: str, baseline_name: str) -> Path:
    return output_dir / f"{baseline_name}_{split}_diagnostics.jsonl"


__all__ = [
    "external_diagnostics_path",
    "gurobi_diagnostics_path",
    "resolve_baselines",
]
