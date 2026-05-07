from __future__ import annotations

import json
import importlib
import importlib.util
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dasbench.problems.base import SolveOutcome
from dasbench.problems.graph_utils import adjacency_sets
from dasbench.problems.tsp_utils import canonicalize_tour, distance_matrix

EXTERNAL_EXACT_MODES = {"auto", "off", "required"}


@dataclass(frozen=True)
class ExternalSolverSpec:
    baseline_name: str
    problem: str
    env_var: str
    aliases: tuple[str, ...]
    python_module: str | None = None
    python_symbol: str | None = None
    cli_supported: bool = True
    external_setup_required: bool = False


EXTERNAL_SOLVER_SPECS = {
    "open_wbo_exact": ExternalSolverSpec(
        baseline_name="open_wbo_exact",
        problem="maxsat",
        env_var="DASBENCH_OPEN_WBO_BIN",
        aliases=("open_wbo", "open-wbo", "openwbo"),
        python_module="hermax.non_incremental",
        python_symbol="OpenWBO",
    ),
    "uwrmaxsat_exact": ExternalSolverSpec(
        baseline_name="uwrmaxsat_exact",
        problem="maxsat",
        env_var="DASBENCH_UWRMAXSAT_BIN",
        aliases=("uwrmaxsat", "uwr_maxsat", "hermax_uwrmaxsat"),
        python_module="hermax.incremental",
        python_symbol="UWrMaxSAT",
    ),
    "evalmaxsat_exact": ExternalSolverSpec(
        baseline_name="evalmaxsat_exact",
        problem="maxsat",
        env_var="DASBENCH_EVALMAXSAT_BIN",
        aliases=("evalmaxsat", "eval_maxsat", "hermax_evalmaxsat"),
        python_module="hermax.incremental",
        python_symbol="EvalMaxSAT",
    ),
    "maxhs_exact": ExternalSolverSpec(
        baseline_name="maxhs_exact",
        problem="maxsat",
        env_var="DASBENCH_MAXHS_BIN",
        aliases=("maxhs", "max_hs", "hermax_maxhs"),
        python_module="hermax.non_incremental",
        python_symbol="MaxHS",
        external_setup_required=True,
    ),
    "wmaxcdcl_exact": ExternalSolverSpec(
        baseline_name="wmaxcdcl_exact",
        problem="maxsat",
        env_var="DASBENCH_WMAXCDCL_BIN",
        aliases=("wmaxcdcl", "wmax_cdcl", "hermax_wmaxcdcl"),
        python_module="hermax.non_incremental",
        python_symbol="WMaxCDCL",
    ),
    "kamis_vc_exact": ExternalSolverSpec(
        baseline_name="kamis_vc_exact",
        problem="mis",
        env_var="DASBENCH_KAMIS_EXACT_BIN",
        aliases=("kamis", "kamis_exact", "kamis_vc"),
        external_setup_required=True,
    ),
    "scip_mis_exact": ExternalSolverSpec(
        baseline_name="scip_mis_exact",
        problem="mis",
        env_var="DASBENCH_SCIP_BIN",
        aliases=("scip_mis",),
        python_module="pyscipopt",
    ),
    "highs_mis_mip_exact": ExternalSolverSpec(
        baseline_name="highs_mis_mip_exact",
        problem="mis",
        env_var="DASBENCH_HIGHS_BIN",
        aliases=("highs_mis", "highs_mis_mip"),
        python_module="highspy",
    ),
    "scip_mip_exact": ExternalSolverSpec(
        baseline_name="scip_mip_exact",
        problem="mds",
        env_var="DASBENCH_SCIP_BIN",
        aliases=("scip", "scip_mds"),
        python_module="pyscipopt",
    ),
    "highs_mds_mip_exact": ExternalSolverSpec(
        baseline_name="highs_mds_mip_exact",
        problem="mds",
        env_var="DASBENCH_HIGHS_BIN",
        aliases=("highs_mds", "highs_mds_mip"),
        python_module="highspy",
    ),
    "cbc_mds_mip_exact": ExternalSolverSpec(
        baseline_name="cbc_mds_mip_exact",
        problem="mds",
        env_var="DASBENCH_CBC_BIN",
        aliases=("cbc_mds", "cbc_mds_mip"),
        python_module="ortools.linear_solver.pywraplp",
        cli_supported=False,
    ),
    "scip_coloring_exact": ExternalSolverSpec(
        baseline_name="scip_coloring_exact",
        problem="coloring",
        env_var="DASBENCH_SCIP_BIN",
        aliases=("scip", "scip_coloring"),
        python_module="pyscipopt",
    ),
    "pysat_coloring_exact": ExternalSolverSpec(
        baseline_name="pysat_coloring_exact",
        problem="coloring",
        env_var="DASBENCH_PYSAT_BIN",
        aliases=("pysat_coloring", "sat_coloring"),
        python_module="pysat.solvers",
        python_symbol="Solver",
        cli_supported=False,
    ),
    "highs_coloring_mip_exact": ExternalSolverSpec(
        baseline_name="highs_coloring_mip_exact",
        problem="coloring",
        env_var="DASBENCH_HIGHS_BIN",
        aliases=("highs_coloring", "highs_coloring_mip"),
        python_module="highspy",
    ),
    "concorde_exact": ExternalSolverSpec(
        baseline_name="concorde_exact",
        problem="tsp",
        env_var="DASBENCH_CONCORDE_BIN",
        aliases=("concorde", "concorde_tsp"),
        external_setup_required=True,
    ),
    "cpsat_tsp_exact": ExternalSolverSpec(
        baseline_name="cpsat_tsp_exact",
        problem="tsp",
        env_var="DASBENCH_CPSAT_BIN",
        aliases=("cpsat_tsp", "ortools_cpsat_tsp"),
        python_module="ortools.sat.python.cp_model",
        cli_supported=False,
    ),
    "scip_tsp_mtz_exact": ExternalSolverSpec(
        baseline_name="scip_tsp_mtz_exact",
        problem="tsp",
        env_var="DASBENCH_SCIP_BIN",
        aliases=("scip_tsp", "scip_tsp_mtz"),
        python_module="pyscipopt",
    ),
    "cbc_tsp_mtz_exact": ExternalSolverSpec(
        baseline_name="cbc_tsp_mtz_exact",
        problem="tsp",
        env_var="DASBENCH_CBC_BIN",
        aliases=("cbc_tsp", "cbc_tsp_mtz"),
        python_module="ortools.linear_solver.pywraplp",
        cli_supported=False,
    ),
    "highs_lp_exact": ExternalSolverSpec(
        baseline_name="highs_lp_exact",
        problem="packing_lp",
        env_var="DASBENCH_HIGHS_BIN",
        aliases=("highs", "highs_lp"),
        python_module="highspy",
    ),
    "clp_lp_exact": ExternalSolverSpec(
        baseline_name="clp_lp_exact",
        problem="packing_lp",
        env_var="DASBENCH_CLP_BIN",
        aliases=("clp", "clp_lp"),
        python_module="ortools.linear_solver.pywraplp",
        cli_supported=False,
    ),
    "highs_ipm_lp_exact": ExternalSolverSpec(
        baseline_name="highs_ipm_lp_exact",
        problem="packing_lp",
        env_var="DASBENCH_HIGHS_BIN",
        aliases=("highs_ipm", "highs_ipm_lp"),
        python_module="highspy",
        cli_supported=False,
    ),
    "scip_lp_exact": ExternalSolverSpec(
        baseline_name="scip_lp_exact",
        problem="packing_lp",
        env_var="DASBENCH_SCIP_BIN",
        aliases=("scip", "scip_lp"),
        python_module="pyscipopt",
    ),
    "highs_mip_exact": ExternalSolverSpec(
        baseline_name="highs_mip_exact",
        problem="mdkp",
        env_var="DASBENCH_HIGHS_BIN",
        aliases=("highs", "highs_mip"),
        python_module="highspy",
    ),
    "cbc_mdkp_exact": ExternalSolverSpec(
        baseline_name="cbc_mdkp_exact",
        problem="mdkp",
        env_var="DASBENCH_CBC_BIN",
        aliases=("cbc_mdkp",),
        python_module="ortools.linear_solver.pywraplp",
        cli_supported=False,
    ),
    "branch_bound_mdkp_exact": ExternalSolverSpec(
        baseline_name="branch_bound_mdkp_exact",
        problem="mdkp",
        env_var="DASBENCH_MDKP_BRANCH_BOUND_BIN",
        aliases=("mdkp_branch_bound", "branch_bound_mdkp"),
        python_module="dasbench.integrations.external_exact",
        python_symbol="_solve_branch_bound_mdkp",
        cli_supported=False,
    ),
    "scip_mdkp_exact": ExternalSolverSpec(
        baseline_name="scip_mdkp_exact",
        problem="mdkp",
        env_var="DASBENCH_SCIP_BIN",
        aliases=("scip", "scip_mdkp"),
        python_module="pyscipopt",
    ),
}

HERMAX_MAXSAT_BASELINES = {
    "open_wbo_exact",
    "uwrmaxsat_exact",
    "evalmaxsat_exact",
    "maxhs_exact",
    "wmaxcdcl_exact",
}


@dataclass(frozen=True)
class ExternalExactConfig:
    mode: str = "auto"
    time_limit_seconds: float = 60.0
    threads: int = 1
    solver_config_path: str | None = None
    solver_paths: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in EXTERNAL_EXACT_MODES:
            raise ValueError(f"Unknown external exact baseline mode: {self.mode}")

    def to_record(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "time_limit_seconds": self.time_limit_seconds,
            "threads": self.threads,
            "solver_config_path": self.solver_config_path,
            "solver_paths": dict(self.solver_paths),
        }

    @classmethod
    def from_record(cls, payload: dict[str, object] | None) -> "ExternalExactConfig":
        if not payload:
            return cls()
        solver_paths_raw = payload.get("solver_paths", {})
        solver_paths = {
            str(key): str(value)
            for key, value in solver_paths_raw.items()
        } if isinstance(solver_paths_raw, dict) else {}
        return cls(
            mode=str(payload.get("mode", "auto")),
            time_limit_seconds=float(payload.get("time_limit_seconds", 60.0)),
            threads=int(payload.get("threads", 1)),
            solver_config_path=(
                str(payload["solver_config_path"])
                if payload.get("solver_config_path") is not None
                else None
            ),
            solver_paths=solver_paths,
        )


class ExternalSolverError(RuntimeError):
    def __init__(self, message: str, metadata: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.metadata = metadata or {}


@dataclass(frozen=True)
class ExternalCommandResult:
    command: list[str]
    returncode: int | None
    timed_out: bool
    runtime_ms: float
    stdout_path: Path
    stderr_path: Path
    stdout_text: str
    stderr_text: str


def load_solver_paths(config_path: str | None) -> dict[str, str]:
    if not config_path:
        return {}
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("solvers"), dict):
        payload = payload["solvers"]
    if not isinstance(payload, dict):
        raise ValueError("External solver config must be a JSON object.")
    return {str(key): str(value) for key, value in payload.items()}


def _configured_solver_paths(config: ExternalExactConfig) -> dict[str, str]:
    paths = dict(config.solver_paths)
    paths.update(load_solver_paths(config.solver_config_path))
    return paths


def _path_for_spec(spec: ExternalSolverSpec, config: ExternalExactConfig) -> str | None:
    configured = _configured_solver_paths(config)
    for key in (spec.baseline_name, *spec.aliases, spec.env_var):
        value = configured.get(key)
        if value:
            return value
    return os.environ.get(spec.env_var)


def _python_backend_available(spec: ExternalSolverSpec) -> bool:
    if not spec.python_module or importlib.util.find_spec(spec.python_module) is None:
        return False
    if not spec.python_symbol:
        return True
    try:
        module = importlib.import_module(spec.python_module)
    except Exception:
        return False
    target: object = module
    for part in spec.python_symbol.split("."):
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


def _select_solver_backend(
    spec: ExternalSolverSpec,
    *,
    config: ExternalExactConfig,
    exists: bool,
    executable: bool,
    python_available: bool,
) -> str | None:
    if config.mode == "off":
        return None
    if spec.cli_supported and config.mode != "off" and exists and executable:
        return "cli"
    if python_available:
        return "python"
    return None


def discover_external_exact_baselines(
    problem_name: str,
    config: ExternalExactConfig,
) -> dict[str, object]:
    relevant_specs = [
        spec for spec in EXTERNAL_SOLVER_SPECS.values()
        if spec.problem == problem_name
    ]
    solvers = []
    for spec in relevant_specs:
        binary_path = _path_for_spec(spec, config)
        exists = bool(binary_path and Path(binary_path).exists())
        executable = bool(binary_path and os.access(binary_path, os.X_OK))
        python_available = _python_backend_available(spec)
        backend = _select_solver_backend(
            spec,
            config=config,
            exists=exists,
            executable=executable,
            python_available=python_available,
        )
        enabled = backend is not None
        missing_required = config.mode == "required" and spec.external_setup_required and not enabled
        solvers.append(
            {
                "baseline_name": spec.baseline_name,
                "problem": spec.problem,
                "env_var": spec.env_var,
                "binary_path": binary_path,
                "exists": exists,
                "executable": executable,
                "python_module": spec.python_module,
                "python_symbol": spec.python_symbol,
                "cli_supported": spec.cli_supported,
                "external_setup_required": spec.external_setup_required,
                "python_available": python_available,
                "backend": backend,
                "enabled": enabled,
                "missing_required": missing_required,
            }
        )
    return {
        "mode": config.mode,
        "time_limit_seconds": config.time_limit_seconds,
        "threads": config.threads,
        "solver_config_path": config.solver_config_path,
        "problem": problem_name,
        "solvers": solvers,
    }


def external_diagnostics_path(output_dir: Path, *, split: str, baseline_name: str) -> Path:
    return output_dir / f"{baseline_name}_{split}_diagnostics.jsonl"


def write_external_discovery(output_dir: Path, discovery: dict[str, object]) -> Path:
    path = output_dir / "external_exact_discovery.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(discovery, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _safe_id(instance: dict[str, object]) -> str:
    raw = str(instance.get("id", "instance"))
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def _run_external_command(
    command: list[str],
    *,
    log_dir: Path,
    timeout_seconds: float,
) -> ExternalCommandResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=log_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        runtime_ms = (time.perf_counter() - start) * 1000.0
        stdout_text = completed.stdout or ""
        stderr_text = completed.stderr or ""
        returncode = completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        runtime_ms = (time.perf_counter() - start) * 1000.0
        stdout_text = _decode_process_text(exc.stdout)
        stderr_text = _decode_process_text(exc.stderr)
        returncode = None
        timed_out = True
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    return ExternalCommandResult(
        command=command,
        returncode=returncode,
        timed_out=timed_out,
        runtime_ms=runtime_ms,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


def _decode_process_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _base_metadata(
    *,
    baseline_name: str,
    instance: dict[str, object],
    result: ExternalCommandResult,
    solver_status: str,
    proved_optimal: bool,
    objective_value: float | None = None,
    best_bound: float | None = None,
    mip_gap: float | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = {
        "instance_id": instance["id"],
        "baseline_name": baseline_name,
        "solver_status": solver_status,
        "proved_optimal": proved_optimal,
        "external_runtime_ms": result.runtime_ms,
        "objective_value": objective_value,
        "best_bound": best_bound,
        "mip_gap": mip_gap,
        "returncode": result.returncode,
        "time_limit_hit": result.timed_out,
        "command": " ".join(shlex.quote(item) for item in result.command),
        "stdout_path": str(result.stdout_path),
        "stderr_path": str(result.stderr_path),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _missing_solver(
    baseline_name: str,
    binary_path: str | None,
    env_var: str,
    python_module: str | None,
) -> Callable[[dict[str, object]], SolveOutcome]:
    def solver(instance: dict[str, object]) -> SolveOutcome:
        metadata = {
            "instance_id": instance["id"],
            "baseline_name": baseline_name,
            "solver_status": "missing_solver",
            "proved_optimal": False,
            "external_runtime_ms": 0.0,
            "binary_path": binary_path,
            "env_var": env_var,
            "python_module": python_module,
        }
        raise ExternalSolverError(f"Missing required external exact solver `{baseline_name}`.", metadata)

    return solver


def build_external_exact_solvers(
    problem_name: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> tuple[dict[str, Callable[[dict[str, object]], SolveOutcome]], dict[str, object]]:
    discovery = discover_external_exact_baselines(problem_name, config)
    solvers: dict[str, Callable[[dict[str, object]], SolveOutcome]] = {}
    for record in discovery["solvers"]:
        assert isinstance(record, dict)
        baseline_name = str(record["baseline_name"])
        spec = EXTERNAL_SOLVER_SPECS[baseline_name]
        binary_path = record.get("binary_path")
        if record.get("enabled"):
            solvers[baseline_name] = _solver_for_spec(
                spec,
                config,
                artifact_dir=artifact_dir / baseline_name,
                backend=str(record.get("backend")),
                binary_path=str(binary_path) if binary_path else None,
            )
        elif config.mode == "required":
            solvers[baseline_name] = _missing_solver(
                baseline_name,
                str(binary_path) if binary_path else None,
                spec.env_var,
                spec.python_module,
            )
    return solvers, discovery


def _solver_for_spec(
    spec: ExternalSolverSpec,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
    backend: str,
    binary_path: str | None,
) -> Callable[[dict[str, object]], SolveOutcome]:
    if backend == "python":
        return _python_solver_for_spec(spec, config, artifact_dir=artifact_dir)
    if binary_path is None:
        raise ValueError(f"CLI external exact solver `{spec.baseline_name}` needs a binary path.")
    if spec.baseline_name in HERMAX_MAXSAT_BASELINES:
        return lambda instance: _solve_maxsat_wcnf_cli(
            instance,
            binary_path,
            config,
            artifact_dir=artifact_dir,
            baseline_name=spec.baseline_name,
        )
    if spec.baseline_name == "open_wbo_exact":
        return lambda instance: _solve_open_wbo(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "kamis_vc_exact":
        return lambda instance: _solve_kamis_mis(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_mis_exact":
        return lambda instance: _solve_scip_mis(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_mis_mip_exact":
        return lambda instance: _solve_highs_mis(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_mip_exact":
        return lambda instance: _solve_scip_mds(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_mds_mip_exact":
        return lambda instance: _solve_highs_mds(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_coloring_exact":
        return lambda instance: _solve_scip_coloring(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_coloring_mip_exact":
        return lambda instance: _solve_highs_coloring(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "concorde_exact":
        return lambda instance: _solve_concorde(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_tsp_mtz_exact":
        return lambda instance: _solve_scip_tsp_mtz_cli(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_lp_exact":
        return lambda instance: _solve_highs_packing_lp(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_lp_exact":
        return lambda instance: _solve_scip_packing_lp(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_mip_exact":
        return lambda instance: _solve_highs_mdkp(instance, binary_path, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_mdkp_exact":
        return lambda instance: _solve_scip_mdkp(instance, binary_path, config, artifact_dir=artifact_dir)
    raise ValueError(f"Unsupported external exact solver: {spec.baseline_name}")


def _python_solver_for_spec(
    spec: ExternalSolverSpec,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> Callable[[dict[str, object]], SolveOutcome]:
    if spec.baseline_name in HERMAX_MAXSAT_BASELINES:
        return lambda instance: _solve_hermax_maxsat(instance, spec, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_mis_exact":
        return lambda instance: _solve_pyscipopt_mis(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_mis_mip_exact":
        return lambda instance: _solve_highspy_mis(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_mip_exact":
        return lambda instance: _solve_pyscipopt_mds(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_mds_mip_exact":
        return lambda instance: _solve_highspy_mds(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "cbc_mds_mip_exact":
        return lambda instance: _solve_ortools_cbc_mds(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_coloring_exact":
        return lambda instance: _solve_pyscipopt_coloring(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "pysat_coloring_exact":
        return lambda instance: _solve_pysat_coloring(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_coloring_mip_exact":
        return lambda instance: _solve_highspy_coloring(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "cpsat_tsp_exact":
        return lambda instance: _solve_cpsat_tsp(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_tsp_mtz_exact":
        return lambda instance: _solve_pyscipopt_tsp_mtz(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "cbc_tsp_mtz_exact":
        return lambda instance: _solve_ortools_cbc_tsp_mtz(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_lp_exact":
        return lambda instance: _solve_highspy_packing_lp(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "clp_lp_exact":
        return lambda instance: _solve_ortools_clp_packing_lp(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_ipm_lp_exact":
        return lambda instance: _solve_highspy_packing_ipm_lp(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_lp_exact":
        return lambda instance: _solve_pyscipopt_packing_lp(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "highs_mip_exact":
        return lambda instance: _solve_highspy_mdkp(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "cbc_mdkp_exact":
        return lambda instance: _solve_ortools_cbc_mdkp(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "branch_bound_mdkp_exact":
        return lambda instance: _solve_branch_bound_mdkp(instance, config, artifact_dir=artifact_dir)
    if spec.baseline_name == "scip_mdkp_exact":
        return lambda instance: _solve_pyscipopt_mdkp(instance, config, artifact_dir=artifact_dir)
    raise ValueError(f"No Python backend is registered for external exact solver: {spec.baseline_name}")


def _native_command_result(
    *,
    command: list[str],
    log_dir: Path,
    runtime_ms: float,
    stdout_text: str = "",
    stderr_text: str = "",
    returncode: int | None = 0,
    timed_out: bool = False,
) -> ExternalCommandResult:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "stdout.txt"
    stderr_path = log_dir / "stderr.txt"
    stdout_path.write_text(stdout_text, encoding="utf-8")
    stderr_path.write_text(stderr_text, encoding="utf-8")
    return ExternalCommandResult(
        command=command,
        returncode=returncode,
        timed_out=timed_out,
        runtime_ms=runtime_ms,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


def _status_text(value: object) -> str:
    text = str(value)
    return text.rsplit(".", 1)[-1] if "." in text else text


def _status_is_optimal(status: object) -> bool:
    lowered = _status_text(status).lower()
    return "optimal" in lowered and "not" not in lowered


def _status_is_time_limit(status: object) -> bool:
    lowered = _status_text(status).lower().replace("_", " ")
    return "time" in lowered and "limit" in lowered


def _finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _status_from_ortools_linear(status: int) -> str:
    from ortools.linear_solver import pywraplp

    return {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "abnormal",
        pywraplp.Solver.NOT_SOLVED: "not_solved",
        pywraplp.Solver.MODEL_INVALID: "model_invalid",
    }.get(status, f"unknown_{status}")


def _configure_ortools_linear_solver(solver: object, config: ExternalExactConfig) -> None:
    solver.SetTimeLimit(max(1, int(float(config.time_limit_seconds) * 1000.0)))
    try:
        solver.SetNumThreads(max(1, int(config.threads)))
    except Exception:
        pass


def _ortools_linear_metadata(
    *,
    solver: object,
    status: int,
    instance: dict[str, object],
    artifact_dir: Path,
    baseline_name: str,
    model_text: str,
    solver_label: str,
    runtime_ms: float,
) -> dict[str, object]:
    from ortools.linear_solver import pywraplp

    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_path = instance_dir / "model.lp"
    model_path.write_text(model_text, encoding="utf-8")
    status_text = _status_from_ortools_linear(status)
    objective = None
    best_bound = None
    mip_gap = None
    if status in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        objective = _finite_or_none(solver.Objective().Value())
        try:
            best_bound = _finite_or_none(solver.Objective().BestBound())
        except Exception:
            best_bound = None
        if objective is not None and best_bound is not None:
            scale = max(1.0, abs(objective))
            mip_gap = abs(objective - best_bound) / scale
    result = _native_command_result(
        command=[f"python:ortools:{solver_label}", str(model_path)],
        log_dir=instance_dir,
        runtime_ms=runtime_ms,
        stdout_text=f"status: {status_text}\n",
        returncode=0 if status != pywraplp.Solver.ABNORMAL else 1,
        timed_out=status == pywraplp.Solver.NOT_SOLVED,
    )
    return _base_metadata(
        baseline_name=baseline_name,
        instance=instance,
        result=result,
        solver_status=status_text,
        proved_optimal=status == pywraplp.Solver.OPTIMAL,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
        extra={
            "execution_backend": "python",
            "python_module": "ortools.linear_solver.pywraplp",
            "solver_version": solver.SolverVersion(),
            "model_path": str(model_path),
        },
    )


def _cp_sat_status_text(status: int) -> str:
    from ortools.sat.python import cp_model

    return {
        cp_model.OPTIMAL: "optimal",
        cp_model.FEASIBLE: "feasible",
        cp_model.INFEASIBLE: "infeasible",
        cp_model.MODEL_INVALID: "model_invalid",
        cp_model.UNKNOWN: "unknown",
    }.get(status, f"unknown_{status}")


def _cp_sat_metadata(
    *,
    solver: object,
    status: int,
    instance: dict[str, object],
    artifact_dir: Path,
    baseline_name: str,
    model_text: str,
    objective_value: float | None,
    runtime_ms: float,
) -> dict[str, object]:
    from ortools.sat.python import cp_model

    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_path = instance_dir / "model.txt"
    model_path.write_text(model_text, encoding="utf-8")
    status_text = _cp_sat_status_text(status)
    result = _native_command_result(
        command=["python:ortools:cp-sat", str(model_path)],
        log_dir=instance_dir,
        runtime_ms=runtime_ms,
        stdout_text=f"status: {status_text}\n",
        timed_out=status == cp_model.UNKNOWN,
        returncode=0 if status in {cp_model.OPTIMAL, cp_model.FEASIBLE} else 1,
    )
    best_bound = _finite_or_none(getattr(solver, "BestObjectiveBound", lambda: None)())
    mip_gap = None
    if objective_value is not None and best_bound is not None:
        mip_gap = abs(objective_value - best_bound) / max(1.0, abs(objective_value))
    return _base_metadata(
        baseline_name=baseline_name,
        instance=instance,
        result=result,
        solver_status=status_text,
        proved_optimal=status == cp_model.OPTIMAL,
        objective_value=objective_value,
        best_bound=best_bound,
        mip_gap=mip_gap,
        extra={
            "execution_backend": "python",
            "python_module": "ortools.sat.python.cp_model",
            "solver_internal_runtime_ms": _finite_or_none(solver.WallTime() * 1000.0),
            "model_path": str(model_path),
        },
    )


def _solve_highspy_row_model(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
    baseline_name: str,
    model_text: str,
    col_names: list[str],
    costs: list[float],
    row_terms: list[list[tuple[int, float]]],
    row_lower: list[float],
    row_upper: list[float],
    maximize: bool,
    binary: bool,
    solver_option: str | None = None,
) -> tuple[list[float], dict[str, object]]:
    import highspy

    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_path = instance_dir / "model.lp"
    model_path.write_text(model_text, encoding="utf-8")

    lp = highspy.HighsLp()
    lp.num_col_ = len(col_names)
    lp.num_row_ = len(row_terms)
    lp.col_cost_ = costs
    lp.col_lower_ = [0.0] * len(col_names)
    lp.col_upper_ = [1.0] * len(col_names)
    lp.row_lower_ = [
        -highspy.kHighsInf if value == -math.inf else value
        for value in row_lower
    ]
    lp.row_upper_ = [
        highspy.kHighsInf if value == math.inf else value
        for value in row_upper
    ]
    lp.sense_ = highspy.ObjSense.kMaximize if maximize else highspy.ObjSense.kMinimize
    lp.col_names_ = col_names
    starts = [0]
    indices: list[int] = []
    coefficients: list[float] = []
    for terms in row_terms:
        for index, coefficient in terms:
            if coefficient == 0.0:
                continue
            indices.append(index)
            coefficients.append(coefficient)
        starts.append(len(indices))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = coefficients
    if binary:
        lp.integrality_ = [highspy.HighsVarType.kInteger] * len(col_names)

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("time_limit", float(config.time_limit_seconds))
    highs.setOptionValue("threads", int(config.threads))
    if solver_option:
        highs.setOptionValue("solver", solver_option)
    if binary:
        highs.setOptionValue("mip_rel_gap", 0.0)

    start = time.perf_counter()
    highs.passModel(lp)
    highs.run()
    measured_runtime_ms = (time.perf_counter() - start) * 1000.0
    status = _status_text(highs.getModelStatus())
    solution_values = list(highs.getSolution().col_value)
    info = highs.getInfo()
    internal_runtime_ms = _finite_or_none(getattr(highs, "getRunTime", lambda: None)())
    runtime_ms = (internal_runtime_ms * 1000.0) if internal_runtime_ms is not None else measured_runtime_ms
    objective = _finite_or_none(getattr(info, "objective_function_value", None))
    if objective is None and solution_values:
        objective = sum(cost * value for cost, value in zip(costs, solution_values, strict=True))
    best_bound = _finite_or_none(getattr(info, "mip_dual_bound", None))
    mip_gap = _finite_or_none(getattr(info, "mip_gap", None))
    timed_out = _status_is_time_limit(status)
    result = _native_command_result(
        command=["python:highspy", baseline_name, str(model_path)],
        log_dir=instance_dir,
        runtime_ms=runtime_ms,
        stdout_text=f"model_status: {status}\n",
        timed_out=timed_out,
        returncode=None if timed_out else 0,
    )
    metadata = _base_metadata(
        baseline_name=baseline_name,
        instance=instance,
        result=result,
        solver_status="time_limit" if timed_out else status,
        proved_optimal=_status_is_optimal(status) and not timed_out,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
        extra={
            "execution_backend": "python",
            "python_module": "highspy",
            "solver_internal_runtime_ms": runtime_ms,
            "model_path": str(model_path),
            "highs_solver_option": solver_option,
        },
    )
    return solution_values, metadata


def _solve_highspy_packing(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
    baseline_name: str,
    binary: bool,
    solver_option: str | None = None,
) -> SolveOutcome:
    import highspy

    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_path = instance_dir / "model.lp"
    model_path.write_text(serialize_packing_lp(instance, binary=binary), encoding="utf-8")

    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    values = [float(value) for value in instance["values"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]

    lp = highspy.HighsLp()
    lp.num_col_ = num_items
    lp.num_row_ = num_resources
    lp.col_cost_ = values
    lp.col_lower_ = [0.0] * num_items
    lp.col_upper_ = [1.0] * num_items
    lp.row_lower_ = [-highspy.kHighsInf] * num_resources
    lp.row_upper_ = capacities
    lp.sense_ = highspy.ObjSense.kMaximize
    lp.col_names_ = [f"x_{item}" for item in range(num_items)]
    lp.row_names_ = [f"cap_{resource}" for resource in range(num_resources)]
    starts: list[int] = [0]
    indices: list[int] = []
    coefficients: list[float] = []
    for resource in range(num_resources):
        for item in range(num_items):
            weight = weights[item][resource]
            if weight == 0.0:
                continue
            indices.append(item)
            coefficients.append(weight)
        starts.append(len(indices))
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = starts
    lp.a_matrix_.index_ = indices
    lp.a_matrix_.value_ = coefficients
    if binary:
        lp.integrality_ = [highspy.HighsVarType.kInteger] * num_items

    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("time_limit", float(config.time_limit_seconds))
    highs.setOptionValue("threads", int(config.threads))
    if solver_option:
        highs.setOptionValue("solver", solver_option)
    if binary:
        highs.setOptionValue("mip_rel_gap", 0.0)

    start = time.perf_counter()
    highs.passModel(lp)
    highs.run()
    measured_runtime_ms = (time.perf_counter() - start) * 1000.0
    status = _status_text(highs.getModelStatus())
    solution_values = list(highs.getSolution().col_value)
    info = highs.getInfo()
    internal_runtime_ms = _finite_or_none(getattr(highs, "getRunTime", lambda: None)())
    runtime_ms = (internal_runtime_ms * 1000.0) if internal_runtime_ms is not None else measured_runtime_ms
    objective = _finite_or_none(getattr(info, "objective_function_value", None))
    if objective is None:
        objective = sum(values[item] * solution_values[item] for item in range(num_items))
    best_bound = _finite_or_none(getattr(info, "mip_dual_bound", None))
    mip_gap = _finite_or_none(getattr(info, "mip_gap", None))
    timed_out = _status_is_time_limit(status)
    result = _native_command_result(
        command=["python:highspy", str(model_path)],
        log_dir=instance_dir,
        runtime_ms=runtime_ms,
        stdout_text=f"model_status: {status}\n",
        timed_out=timed_out,
        returncode=None if timed_out else 0,
    )
    metadata = _base_metadata(
        baseline_name=baseline_name,
        instance=instance,
        result=result,
        solver_status="time_limit" if timed_out else status,
        proved_optimal=_status_is_optimal(status) and not timed_out,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
        extra={
            "execution_backend": "python",
            "python_module": "highspy",
            "solver_internal_runtime_ms": runtime_ms,
            "model_path": str(model_path),
            "highs_solver_option": solver_option,
        },
    )
    if binary:
        return SolveOutcome(solution=[item for item, value in enumerate(solution_values) if value > 0.5], metadata=metadata)
    return SolveOutcome(
        solution=[max(0.0, min(1.0, float(value))) for value in solution_values[:num_items]],
        metadata=metadata,
    )


def _solve_highspy_packing_lp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    return _solve_highspy_packing(
        instance,
        config,
        artifact_dir=artifact_dir,
        baseline_name="highs_lp_exact",
        binary=False,
    )


def _solve_highspy_packing_ipm_lp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    return _solve_highspy_packing(
        instance,
        config,
        artifact_dir=artifact_dir,
        baseline_name="highs_ipm_lp_exact",
        binary=False,
        solver_option="ipm",
    )


def _solve_highspy_mdkp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    return _solve_highspy_packing(
        instance,
        config,
        artifact_dir=artifact_dir,
        baseline_name="highs_mip_exact",
        binary=True,
    )


def _solve_highspy_mis(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    num_vertices = int(instance["num_vertices"])
    row_terms = [[(int(u), 1.0), (int(v), 1.0)] for u, v in instance["edges"]]
    solution_values, metadata = _solve_highspy_row_model(
        instance,
        config,
        artifact_dir=artifact_dir,
        baseline_name="highs_mis_mip_exact",
        model_text=serialize_mis_lp(instance),
        col_names=[f"x_{vertex}" for vertex in range(num_vertices)],
        costs=[1.0] * num_vertices,
        row_terms=row_terms,
        row_lower=[-math.inf] * len(row_terms),
        row_upper=[1.0] * len(row_terms),
        maximize=True,
        binary=True,
    )
    return SolveOutcome(
        solution=[vertex for vertex, value in enumerate(solution_values) if value > 0.5],
        metadata=metadata,
    )


def _solve_highspy_mds(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    row_terms = [
        [(item, 1.0) for item in sorted(adjacency[vertex] | {vertex})]
        for vertex in range(num_vertices)
    ]
    solution_values, metadata = _solve_highspy_row_model(
        instance,
        config,
        artifact_dir=artifact_dir,
        baseline_name="highs_mds_mip_exact",
        model_text=serialize_mds_lp(instance),
        col_names=[f"x_{vertex}" for vertex in range(num_vertices)],
        costs=[1.0] * num_vertices,
        row_terms=row_terms,
        row_lower=[1.0] * num_vertices,
        row_upper=[math.inf] * num_vertices,
        maximize=False,
        binary=True,
    )
    return SolveOutcome(
        solution=[vertex for vertex, value in enumerate(solution_values) if value > 0.5],
        metadata=metadata,
    )


def _solve_highspy_coloring(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from dasbench.problems.graph_utils import dsatur_coloring

    num_vertices = int(instance["num_vertices"])
    x_offset = 0
    y_offset = num_vertices * num_vertices

    def x_index(vertex: int, color: int) -> int:
        return x_offset + vertex * num_vertices + color

    def y_index(color: int) -> int:
        return y_offset + color

    col_names = [
        f"x_{vertex}_{color}"
        for vertex in range(num_vertices)
        for color in range(num_vertices)
    ] + [f"y_{color}" for color in range(num_vertices)]
    costs = [0.0] * (num_vertices * num_vertices) + [1.0] * num_vertices
    row_terms: list[list[tuple[int, float]]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    for vertex in range(num_vertices):
        row_terms.append([(x_index(vertex, color), 1.0) for color in range(num_vertices)])
        row_lower.append(1.0)
        row_upper.append(1.0)
    for raw_u, raw_v in instance["edges"]:
        u, v = int(raw_u), int(raw_v)
        for color in range(num_vertices):
            row_terms.append([(x_index(u, color), 1.0), (x_index(v, color), 1.0)])
            row_lower.append(-math.inf)
            row_upper.append(1.0)
    for vertex in range(num_vertices):
        for color in range(num_vertices):
            row_terms.append([(x_index(vertex, color), 1.0), (y_index(color), -1.0)])
            row_lower.append(-math.inf)
            row_upper.append(0.0)
    for color in range(num_vertices - 1):
        row_terms.append([(y_index(color), 1.0), (y_index(color + 1), -1.0)])
        row_lower.append(0.0)
        row_upper.append(math.inf)
    if num_vertices > 0:
        heuristic_upper = float(len(set(dsatur_coloring(instance))))
        row_terms.append([(y_index(color), 1.0) for color in range(num_vertices)])
        row_lower.append(-math.inf)
        row_upper.append(heuristic_upper)

    solution_values, metadata = _solve_highspy_row_model(
        instance,
        config,
        artifact_dir=artifact_dir,
        baseline_name="highs_coloring_mip_exact",
        model_text=serialize_coloring_lp(instance),
        col_names=col_names,
        costs=costs,
        row_terms=row_terms,
        row_lower=row_lower,
        row_upper=row_upper,
        maximize=False,
        binary=True,
    )
    solution = []
    for vertex in range(num_vertices):
        color = max(
            range(num_vertices),
            key=lambda candidate: solution_values[x_index(vertex, candidate)],
        )
        solution.append(color)
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_ortools_cbc_mds(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise ExternalSolverError("OR-Tools CBC backend is not available.")
    _configure_ortools_linear_solver(solver, config)
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    variables = [solver.BoolVar(f"x_{vertex}") for vertex in range(num_vertices)]
    for vertex in range(num_vertices):
        solver.Add(sum(variables[item] for item in adjacency[vertex] | {vertex}) >= 1)
    solver.Minimize(sum(variables))
    start = time.perf_counter()
    status = solver.Solve()
    runtime_ms = (time.perf_counter() - start) * 1000.0
    metadata = _ortools_linear_metadata(
        solver=solver,
        status=status,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="cbc_mds_mip_exact",
        model_text=serialize_mds_lp(instance),
        solver_label="cbc",
        runtime_ms=runtime_ms,
    )
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        raise ExternalSolverError("CBC did not produce a dominating set incumbent.", metadata)
    return SolveOutcome(
        solution=[vertex for vertex, variable in enumerate(variables) if variable.solution_value() > 0.5],
        metadata=metadata,
    )


def _solve_ortools_clp_packing_lp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("CLP")
    if solver is None:
        raise ExternalSolverError("OR-Tools CLP backend is not available.")
    _configure_ortools_linear_solver(solver, config)
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    variables = [solver.NumVar(0.0, 1.0, f"x_{item}") for item in range(num_items)]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    for resource in range(num_resources):
        solver.Add(sum(weights[item][resource] * variables[item] for item in range(num_items)) <= capacities[resource])
    solver.Maximize(sum(float(instance["values"][item]) * variables[item] for item in range(num_items)))
    start = time.perf_counter()
    status = solver.Solve()
    runtime_ms = (time.perf_counter() - start) * 1000.0
    metadata = _ortools_linear_metadata(
        solver=solver,
        status=status,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="clp_lp_exact",
        model_text=serialize_packing_lp(instance, binary=False),
        solver_label="clp",
        runtime_ms=runtime_ms,
    )
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        raise ExternalSolverError("CLP did not produce a packing LP solution.", metadata)
    return SolveOutcome(
        solution=[max(0.0, min(1.0, variable.solution_value())) for variable in variables],
        metadata=metadata,
    )


def _solve_ortools_cbc_mdkp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise ExternalSolverError("OR-Tools CBC backend is not available.")
    _configure_ortools_linear_solver(solver, config)
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    variables = [solver.BoolVar(f"x_{item}") for item in range(num_items)]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    values = [float(value) for value in instance["values"]]
    for resource in range(num_resources):
        solver.Add(sum(weights[item][resource] * variables[item] for item in range(num_items)) <= capacities[resource])
    solver.Maximize(sum(values[item] * variables[item] for item in range(num_items)))
    start = time.perf_counter()
    status = solver.Solve()
    runtime_ms = (time.perf_counter() - start) * 1000.0
    metadata = _ortools_linear_metadata(
        solver=solver,
        status=status,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="cbc_mdkp_exact",
        model_text=serialize_packing_lp(instance, binary=True),
        solver_label="cbc",
        runtime_ms=runtime_ms,
    )
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        raise ExternalSolverError("CBC did not produce an MDKP incumbent.", metadata)
    return SolveOutcome(
        solution=[item for item, variable in enumerate(variables) if variable.solution_value() > 0.5],
        metadata=metadata,
    )


def _configure_pyscipopt_model(model: object, config: ExternalExactConfig) -> None:
    model.hideOutput()
    model.setRealParam("limits/time", float(config.time_limit_seconds))
    model.setIntParam("parallel/maxnthreads", int(config.threads))


def _pyscipopt_result(
    *,
    model: object,
    variables: dict[str, object],
    instance: dict[str, object],
    artifact_dir: Path,
    baseline_name: str,
    model_text: str,
) -> tuple[dict[str, float], dict[str, object]]:
    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_path = instance_dir / "model.lp"
    model_path.write_text(model_text, encoding="utf-8")

    start = time.perf_counter()
    model.optimize()
    measured_runtime_ms = (time.perf_counter() - start) * 1000.0
    status = _status_text(model.getStatus())
    timed_out = _status_is_time_limit(status)
    solution = model.getBestSol()
    if solution is None:
        raise ValueError("PySCIPOpt did not produce an incumbent solution.")

    values = {
        name: float(model.getSolVal(solution, variable))
        for name, variable in variables.items()
    }
    objective = _finite_or_none(model.getSolObjVal(solution))
    best_bound = _finite_or_none(model.getDualbound())
    mip_gap = _finite_or_none(model.getGap())
    internal_runtime = _finite_or_none(model.getSolvingTime())
    runtime_ms = (internal_runtime * 1000.0) if internal_runtime is not None else measured_runtime_ms
    result = _native_command_result(
        command=["python:pyscipopt", str(model_path)],
        log_dir=instance_dir,
        runtime_ms=runtime_ms,
        stdout_text=f"status: {status}\n",
        timed_out=timed_out,
        returncode=None if timed_out else 0,
    )
    metadata = _base_metadata(
        baseline_name=baseline_name,
        instance=instance,
        result=result,
        solver_status="time_limit" if timed_out else status,
        proved_optimal=_status_is_optimal(status) and not timed_out,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
        extra={
            "execution_backend": "python",
            "python_module": "pyscipopt",
            "solver_internal_runtime_ms": runtime_ms,
            "model_path": str(model_path),
        },
    )
    return values, metadata


def _solve_pyscipopt_mis(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from pyscipopt import Model, quicksum

    num_vertices = int(instance["num_vertices"])
    model = Model("dasbench_mis")
    _configure_pyscipopt_model(model, config)
    variables = {
        f"x_{vertex}": model.addVar(vtype="B", name=f"x_{vertex}")
        for vertex in range(num_vertices)
    }
    model.setObjective(quicksum(variables[f"x_{vertex}"] for vertex in range(num_vertices)), "maximize")
    for raw_u, raw_v in instance["edges"]:
        u, v = int(raw_u), int(raw_v)
        model.addCons(variables[f"x_{u}"] + variables[f"x_{v}"] <= 1, name=f"edge_{u}_{v}")
    values, metadata = _pyscipopt_result(
        model=model,
        variables=variables,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="scip_mis_exact",
        model_text=serialize_mis_lp(instance),
    )
    return SolveOutcome(
        solution=[vertex for vertex in range(num_vertices) if values.get(f"x_{vertex}", 0.0) > 0.5],
        metadata=metadata,
    )


def _solve_pyscipopt_mds(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from pyscipopt import Model, quicksum

    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    model = Model("dasbench_mds")
    _configure_pyscipopt_model(model, config)
    variables = {
        f"x_{vertex}": model.addVar(vtype="B", name=f"x_{vertex}")
        for vertex in range(num_vertices)
    }
    model.setObjective(quicksum(variables[f"x_{vertex}"] for vertex in range(num_vertices)), "minimize")
    for vertex in range(num_vertices):
        closed = sorted(adjacency[vertex] | {vertex})
        model.addCons(quicksum(variables[f"x_{item}"] for item in closed) >= 1, name=f"dom_{vertex}")
    values, metadata = _pyscipopt_result(
        model=model,
        variables=variables,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="scip_mip_exact",
        model_text=serialize_mds_lp(instance),
    )
    return SolveOutcome(
        solution=[vertex for vertex in range(num_vertices) if values.get(f"x_{vertex}", 0.0) > 0.5],
        metadata=metadata,
    )


def _solve_pyscipopt_coloring(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from pyscipopt import Model, quicksum

    num_vertices = int(instance["num_vertices"])
    model = Model("dasbench_coloring")
    _configure_pyscipopt_model(model, config)
    variables: dict[str, object] = {}
    for vertex in range(num_vertices):
        for color in range(num_vertices):
            name = f"x_{vertex}_{color}"
            variables[name] = model.addVar(vtype="B", name=name)
    for color in range(num_vertices):
        name = f"y_{color}"
        variables[name] = model.addVar(vtype="B", name=name)
    model.setObjective(quicksum(variables[f"y_{color}"] for color in range(num_vertices)), "minimize")
    for vertex in range(num_vertices):
        model.addCons(
            quicksum(variables[f"x_{vertex}_{color}"] for color in range(num_vertices)) == 1,
            name=f"assign_{vertex}",
        )
    for raw_u, raw_v in instance["edges"]:
        u, v = int(raw_u), int(raw_v)
        for color in range(num_vertices):
            model.addCons(
                variables[f"x_{u}_{color}"] + variables[f"x_{v}_{color}"] <= 1,
                name=f"edge_{u}_{v}_{color}",
            )
    for vertex in range(num_vertices):
        for color in range(num_vertices):
            model.addCons(
                variables[f"x_{vertex}_{color}"] <= variables[f"y_{color}"],
                name=f"use_{vertex}_{color}",
            )
    for color in range(num_vertices - 1):
        model.addCons(variables[f"y_{color}"] >= variables[f"y_{color + 1}"], name=f"sym_{color}")
    values, metadata = _pyscipopt_result(
        model=model,
        variables=variables,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="scip_coloring_exact",
        model_text=serialize_coloring_lp(instance),
    )
    solution = []
    for vertex in range(num_vertices):
        color = max(
            range(num_vertices),
            key=lambda candidate: values.get(f"x_{vertex}_{candidate}", 0.0),
        )
        solution.append(color)
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_pysat_coloring(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from pysat.solvers import Solver
    from dasbench.problems.graph_utils import dsatur_coloring

    start = time.perf_counter()
    num_vertices = int(instance["num_vertices"])
    edges = [(int(u), int(v)) for u, v in instance["edges"]]
    upper_bound = len(set(dsatur_coloring(instance)))
    lower_bound = 1 if not edges else 2
    best_model: list[int] | None = None
    best_k: int | None = None
    best_clauses: list[list[int]] = []

    def var(vertex: int, color: int, colors: int) -> int:
        return vertex * colors + color + 1

    def build_clauses(colors: int) -> list[list[int]]:
        clauses: list[list[int]] = []
        for vertex in range(num_vertices):
            clauses.append([var(vertex, color, colors) for color in range(colors)])
            for left in range(colors):
                for right in range(left + 1, colors):
                    clauses.append([-var(vertex, left, colors), -var(vertex, right, colors)])
        for u, v in edges:
            for color in range(colors):
                clauses.append([-var(u, color, colors), -var(v, color, colors)])
        return clauses

    for colors in range(lower_bound, upper_bound + 1):
        clauses = build_clauses(colors)
        with Solver(name="g4", bootstrap_with=clauses) as solver:
            if solver.solve():
                best_model = solver.get_model()
                best_k = colors
                best_clauses = clauses
                break
    runtime_ms = (time.perf_counter() - start) * 1000.0
    if best_model is None or best_k is None:
        raise RuntimeError("PySAT coloring search did not find a feasible coloring.")

    positive = {literal for literal in best_model if literal > 0}
    solution = []
    for vertex in range(num_vertices):
        color = next(
            color
            for color in range(best_k)
            if var(vertex, color, best_k) in positive
        )
        solution.append(color)

    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    cnf_path = instance_dir / f"k_{best_k}.cnf"
    cnf_lines = [f"p cnf {num_vertices * best_k} {len(best_clauses)}"]
    cnf_lines.extend(" ".join(str(literal) for literal in clause) + " 0" for clause in best_clauses)
    cnf_path.write_text("\n".join(cnf_lines) + "\n", encoding="utf-8")
    result = _native_command_result(
        command=["python:pysat:g4", str(cnf_path)],
        log_dir=instance_dir,
        runtime_ms=runtime_ms,
        stdout_text=f"status: optimal\ncolors: {best_k}\n",
        returncode=0,
    )
    metadata = _base_metadata(
        baseline_name="pysat_coloring_exact",
        instance=instance,
        result=result,
        solver_status="optimal",
        proved_optimal=True,
        objective_value=float(best_k),
        extra={
            "execution_backend": "python",
            "python_module": "pysat.solvers",
            "sat_solver": "g4",
            "model_path": str(cnf_path),
        },
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _tour_from_successors(successors: dict[int, int], *, num_cities: int) -> list[int]:
    tour: list[int] = []
    current = 0
    seen: set[int] = set()
    for _ in range(num_cities):
        if current in seen:
            raise ValueError("TSP solution contains a subtour before visiting all cities.")
        seen.add(current)
        tour.append(current)
        if current not in successors:
            raise ValueError(f"TSP solution is missing a successor for city {current}.")
        current = successors[current]
    if current != 0:
        raise ValueError("TSP tour does not return to city 0.")
    return canonicalize_tour(tour, num_cities)


def _tour_from_edge_values(
    values: dict[tuple[int, int], float],
    *,
    num_cities: int,
) -> list[int]:
    successors: dict[int, int] = {}
    for city in range(num_cities):
        candidates = [
            (other, values.get((city, other), 0.0))
            for other in range(num_cities)
            if other != city
        ]
        successor, value = max(candidates, key=lambda item: (item[1], -item[0]))
        if value <= 0.25:
            raise ValueError(f"TSP solution has no selected outgoing edge for city {city}.")
        successors[city] = successor
    return _tour_from_successors(successors, num_cities=num_cities)


def _solve_cpsat_tsp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from ortools.sat.python import cp_model

    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    scale = 1_000_000
    model = cp_model.CpModel()
    edge_vars: dict[tuple[int, int], object] = {}
    arcs = []
    for left in range(num_cities):
        for right in range(num_cities):
            if left == right:
                continue
            variable = model.NewBoolVar(f"x_{left}_{right}")
            edge_vars[(left, right)] = variable
            arcs.append((left, right, variable))
    model.AddCircuit(arcs)
    model.Minimize(
        sum(int(round(matrix[left][right] * scale)) * variable for (left, right), variable in edge_vars.items())
    )
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(config.time_limit_seconds)
    solver.parameters.num_search_workers = max(1, int(config.threads))
    start = time.perf_counter()
    status = solver.Solve(model)
    runtime_ms = (time.perf_counter() - start) * 1000.0
    selected = {
        edge: float(solver.BooleanValue(variable))
        for edge, variable in edge_vars.items()
        if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}
    }
    objective = None
    if status in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        objective = solver.ObjectiveValue() / scale
    metadata = _cp_sat_metadata(
        solver=solver,
        status=status,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="cpsat_tsp_exact",
        model_text=f"cp-sat circuit tsp with {num_cities} cities\n",
        objective_value=objective,
        runtime_ms=runtime_ms,
    )
    if status not in {cp_model.OPTIMAL, cp_model.FEASIBLE}:
        raise ExternalSolverError("CP-SAT did not produce a TSP incumbent.", metadata)
    return SolveOutcome(solution=_tour_from_edge_values(selected, num_cities=num_cities), metadata=metadata)


def _build_ortools_tsp_mtz_model(instance: dict[str, object], solver: object) -> tuple[dict[tuple[int, int], object], object]:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    x = {
        (left, right): solver.BoolVar(f"x_{left}_{right}")
        for left in range(num_cities)
        for right in range(num_cities)
        if left != right
    }
    u = {
        city: solver.NumVar(0.0 if city == 0 else 1.0, 0.0 if city == 0 else float(num_cities - 1), f"u_{city}")
        for city in range(num_cities)
    }
    for city in range(num_cities):
        solver.Add(sum(x[(city, other)] for other in range(num_cities) if other != city) == 1)
        solver.Add(sum(x[(other, city)] for other in range(num_cities) if other != city) == 1)
    for left in range(1, num_cities):
        for right in range(1, num_cities):
            if left == right:
                continue
            solver.Add(u[left] - u[right] + num_cities * x[(left, right)] <= num_cities - 1)
    solver.Minimize(sum(matrix[left][right] * variable for (left, right), variable in x.items()))
    return x, u


def _solve_ortools_cbc_tsp_mtz(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from ortools.linear_solver import pywraplp

    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if solver is None:
        raise ExternalSolverError("OR-Tools CBC backend is not available.")
    _configure_ortools_linear_solver(solver, config)
    x, _ = _build_ortools_tsp_mtz_model(instance, solver)
    start = time.perf_counter()
    status = solver.Solve()
    runtime_ms = (time.perf_counter() - start) * 1000.0
    metadata = _ortools_linear_metadata(
        solver=solver,
        status=status,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="cbc_tsp_mtz_exact",
        model_text=serialize_tsp_mtz_lp(instance),
        solver_label="cbc",
        runtime_ms=runtime_ms,
    )
    if status not in {pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE}:
        raise ExternalSolverError("CBC did not produce a TSP incumbent.", metadata)
    values = {edge: variable.solution_value() for edge, variable in x.items()}
    return SolveOutcome(solution=_tour_from_edge_values(values, num_cities=int(instance["num_cities"])), metadata=metadata)


def _solve_pyscipopt_tsp_mtz(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    from pyscipopt import Model, quicksum

    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    model = Model("dasbench_tsp_mtz")
    _configure_pyscipopt_model(model, config)
    variables: dict[str, object] = {}
    for left in range(num_cities):
        for right in range(num_cities):
            if left == right:
                continue
            variables[f"x_{left}_{right}"] = model.addVar(vtype="B", name=f"x_{left}_{right}")
    for city in range(num_cities):
        variables[f"u_{city}"] = model.addVar(
            vtype="C",
            lb=0.0 if city == 0 else 1.0,
            ub=0.0 if city == 0 else float(num_cities - 1),
            name=f"u_{city}",
        )
    for city in range(num_cities):
        model.addCons(
            quicksum(variables[f"x_{city}_{other}"] for other in range(num_cities) if other != city) == 1,
            name=f"out_{city}",
        )
        model.addCons(
            quicksum(variables[f"x_{other}_{city}"] for other in range(num_cities) if other != city) == 1,
            name=f"in_{city}",
        )
    for left in range(1, num_cities):
        for right in range(1, num_cities):
            if left == right:
                continue
            model.addCons(
                variables[f"u_{left}"] - variables[f"u_{right}"] + num_cities * variables[f"x_{left}_{right}"]
                <= num_cities - 1,
                name=f"mtz_{left}_{right}",
            )
    model.setObjective(
        quicksum(
            matrix[left][right] * variables[f"x_{left}_{right}"]
            for left in range(num_cities)
            for right in range(num_cities)
            if left != right
        ),
        "minimize",
    )
    values, metadata = _pyscipopt_result(
        model=model,
        variables=variables,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name="scip_tsp_mtz_exact",
        model_text=serialize_tsp_mtz_lp(instance),
    )
    edge_values = {
        (left, right): values.get(f"x_{left}_{right}", 0.0)
        for left in range(num_cities)
        for right in range(num_cities)
        if left != right
    }
    return SolveOutcome(solution=_tour_from_edge_values(edge_values, num_cities=num_cities), metadata=metadata)


def _solve_pyscipopt_packing(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
    baseline_name: str,
    binary: bool,
) -> SolveOutcome:
    from pyscipopt import Model, quicksum

    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    values = [float(value) for value in instance["values"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    model = Model("dasbench_packing")
    _configure_pyscipopt_model(model, config)
    variables = {
        f"x_{item}": model.addVar(vtype="B" if binary else "C", lb=0.0, ub=1.0, name=f"x_{item}")
        for item in range(num_items)
    }
    model.setObjective(
        quicksum(values[item] * variables[f"x_{item}"] for item in range(num_items)),
        "maximize",
    )
    for resource in range(num_resources):
        model.addCons(
            quicksum(weights[item][resource] * variables[f"x_{item}"] for item in range(num_items))
            <= capacities[resource],
            name=f"cap_{resource}",
        )
    values_by_name, metadata = _pyscipopt_result(
        model=model,
        variables=variables,
        instance=instance,
        artifact_dir=artifact_dir,
        baseline_name=baseline_name,
        model_text=serialize_packing_lp(instance, binary=binary),
    )
    if binary:
        solution: list[int] | list[float] = [
            item for item in range(num_items) if values_by_name.get(f"x_{item}", 0.0) > 0.5
        ]
    else:
        solution = [
            max(0.0, min(1.0, values_by_name.get(f"x_{item}", 0.0)))
            for item in range(num_items)
        ]
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_pyscipopt_packing_lp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    return _solve_pyscipopt_packing(
        instance,
        config,
        artifact_dir=artifact_dir,
        baseline_name="scip_lp_exact",
        binary=False,
    )


def _solve_pyscipopt_mdkp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    return _solve_pyscipopt_packing(
        instance,
        config,
        artifact_dir=artifact_dir,
        baseline_name="scip_mdkp_exact",
        binary=True,
    )


def _solve_branch_bound_mdkp(
    instance: dict[str, object],
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    start = time.perf_counter()
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    values = [int(round(float(value))) for value in instance["values"]]
    weights = [[int(round(float(value))) for value in row] for row in instance["weights"]]
    capacities = [int(round(float(value))) for value in instance["capacities"]]
    density_prices = [1.0 / max(1, capacity) for capacity in capacities]
    order = sorted(
        range(num_items),
        key=lambda item: (
            values[item] / max(1e-9, sum(density_prices[resource] * weights[item][resource] for resource in range(num_resources))),
            values[item],
        ),
        reverse=True,
    )
    ordered_values = [values[item] for item in order]
    ordered_weights = [weights[item] for item in order]
    suffix_values = [0] * (num_items + 1)
    for index in range(num_items - 1, -1, -1):
        suffix_values[index] = suffix_values[index + 1] + ordered_values[index]

    best_value = 0
    best_selection: list[int] = []
    greedy_remaining = list(capacities)
    greedy_selection: list[int] = []
    for index, item in enumerate(order):
        row = ordered_weights[index]
        if all(row[resource] <= greedy_remaining[resource] for resource in range(num_resources)):
            greedy_selection.append(item)
            for resource in range(num_resources):
                greedy_remaining[resource] -= row[resource]
            best_value += ordered_values[index]
    best_selection = sorted(greedy_selection)

    timed_out = False
    nodes = 0
    seen: dict[tuple[int, tuple[int, ...]], int] = {}
    deadline = start + float(config.time_limit_seconds)

    def single_resource_bound(position: int, capacity: int, resource: int) -> float:
        bound = 0.0
        remaining_capacity = float(capacity)
        candidates = []
        for item_index in range(position, num_items):
            weight = ordered_weights[item_index][resource]
            value = ordered_values[item_index]
            if weight <= 0:
                bound += value
            else:
                candidates.append((value / weight, value, weight))
        for _, value, weight in sorted(candidates, reverse=True):
            if remaining_capacity <= 0:
                break
            fraction = min(1.0, remaining_capacity / weight)
            bound += fraction * value
            remaining_capacity -= fraction * weight
        return bound

    def upper_bound(position: int, remaining: tuple[int, ...], current_value: int) -> float:
        bound = float(current_value + suffix_values[position])
        for resource in range(num_resources):
            bound = min(
                bound,
                current_value + single_resource_bound(position, remaining[resource], resource),
            )
        return bound

    def search(position: int, remaining: tuple[int, ...], current_value: int, chosen: list[int]) -> None:
        nonlocal best_value, best_selection, timed_out, nodes
        nodes += 1
        if nodes % 1024 == 0 and time.perf_counter() > deadline:
            timed_out = True
            return
        if position == num_items:
            if current_value > best_value:
                best_value = current_value
                best_selection = sorted(chosen)
            return
        if current_value + suffix_values[position] <= best_value:
            return
        if upper_bound(position, remaining, current_value) <= best_value + 1e-9:
            return
        state = (position, remaining)
        if seen.get(state, -1) >= current_value:
            return
        seen[state] = current_value

        row = ordered_weights[position]
        item = order[position]
        if all(row[resource] <= remaining[resource] for resource in range(num_resources)):
            next_remaining = tuple(
                remaining[resource] - row[resource]
                for resource in range(num_resources)
            )
            search(position + 1, next_remaining, current_value + ordered_values[position], [*chosen, item])
            if timed_out:
                return
        search(position + 1, remaining, current_value, chosen)

    search(0, tuple(capacities), 0, [])
    runtime_ms = (time.perf_counter() - start) * 1000.0
    status = "time_limit" if timed_out else "optimal"
    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_path = instance_dir / "model.lp"
    model_path.write_text(serialize_packing_lp(instance, binary=True), encoding="utf-8")
    result = _native_command_result(
        command=["python:branch-bound-mdkp", str(model_path)],
        log_dir=instance_dir,
        runtime_ms=runtime_ms,
        stdout_text=f"status: {status}\nnodes: {nodes}\n",
        timed_out=timed_out,
        returncode=None if timed_out else 0,
    )
    metadata = _base_metadata(
        baseline_name="branch_bound_mdkp_exact",
        instance=instance,
        result=result,
        solver_status=status,
        proved_optimal=not timed_out,
        objective_value=float(best_value),
        best_bound=float(best_value) if not timed_out else None,
        extra={
            "execution_backend": "python",
            "python_module": "dasbench.integrations.external_exact",
            "branch_and_bound_nodes": nodes,
            "memoized_states": len(seen),
            "model_path": str(model_path),
        },
    )
    return SolveOutcome(solution=best_selection, metadata=metadata)


def serialize_maxsat_wcnf(instance: dict[str, object]) -> str:
    num_variables = int(instance["num_variables"])
    clauses = instance["clauses"]
    top = len(clauses) + 1
    lines = [f"p wcnf {num_variables} {len(clauses)} {top}"]
    for clause in clauses:
        lines.append("1 " + " ".join(str(int(literal)) for literal in clause) + " 0")
    return "\n".join(lines) + "\n"


def parse_open_wbo_output(text: str, *, num_variables: int) -> tuple[list[bool], str, bool, float | None]:
    status = "unknown"
    proved_optimal = False
    objective_value: float | None = None
    model_values: dict[int, bool] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("s "):
            status = line[2:].strip()
            proved_optimal = "OPTIMUM" in status.upper() or "OPTIMAL" in status.upper()
        elif line.startswith("o "):
            try:
                objective_value = float(line.split()[1])
            except (IndexError, ValueError):
                objective_value = None
        elif line.startswith("v "):
            for token in line[2:].split():
                try:
                    literal = int(token)
                except ValueError:
                    continue
                if literal == 0:
                    continue
                variable = abs(literal)
                if 1 <= variable <= num_variables:
                    model_values[variable] = literal > 0
    if not model_values:
        raise ValueError("Open-WBO output did not contain a parseable model line.")
    return [model_values.get(variable, False) for variable in range(1, num_variables + 1)], status, proved_optimal, objective_value


def _maxsat_wcnf_command_solver(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
    baseline_name: str,
) -> SolveOutcome:
    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    input_path = instance_dir / "instance.wcnf"
    input_path.write_text(serialize_maxsat_wcnf(instance), encoding="utf-8")
    result = _run_external_command(
        [binary_path, str(input_path)],
        log_dir=instance_dir,
        timeout_seconds=config.time_limit_seconds,
    )
    output_text = "\n".join([result.stdout_text, result.stderr_text])
    solution, status, proved_optimal, unsatisfied_cost = parse_open_wbo_output(
        output_text,
        num_variables=int(instance["num_variables"]),
    )
    objective_value = (
        float(len(instance["clauses"])) - unsatisfied_cost
        if unsatisfied_cost is not None
        else None
    )
    metadata = _base_metadata(
        baseline_name=baseline_name,
        instance=instance,
        result=result,
        solver_status="timeout" if result.timed_out else status,
        proved_optimal=proved_optimal and not result.timed_out,
        objective_value=objective_value,
        extra={
            "execution_backend": "cli",
            "model_path": str(input_path),
        },
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_maxsat_wcnf_cli(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
    baseline_name: str,
) -> SolveOutcome:
    return _maxsat_wcnf_command_solver(
        instance,
        binary_path,
        config,
        artifact_dir=artifact_dir,
        baseline_name=baseline_name,
    )


def _solve_hermax_maxsat(
    instance: dict[str, object],
    spec: ExternalSolverSpec,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    input_path = instance_dir / "instance.json"
    output_path = instance_dir / "solution.json"
    input_path.write_text(json.dumps(instance, sort_keys=True), encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).with_name("hermax_worker.py")),
        "--baseline",
        spec.baseline_name,
        "--instance",
        str(input_path),
        "--output",
        str(output_path),
        "--time-limit-seconds",
        str(config.time_limit_seconds),
    ]
    result = _run_external_command(
        command,
        log_dir=instance_dir,
        timeout_seconds=config.time_limit_seconds + 5.0,
    )
    if not output_path.exists():
        metadata = _base_metadata(
            baseline_name=spec.baseline_name,
            instance=instance,
            result=result,
            solver_status="timeout" if result.timed_out else "error",
            proved_optimal=False,
            extra={
                "execution_backend": "python",
                "python_module": spec.python_module,
                "python_symbol": spec.python_symbol,
                "result_path": str(output_path),
            },
        )
        message = "Hermax worker timed out." if result.timed_out else "Hermax worker did not produce a solution file."
        raise ExternalSolverError(message, metadata)
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("error"):
        metadata = _base_metadata(
            baseline_name=spec.baseline_name,
            instance=instance,
            result=result,
            solver_status="timeout" if result.timed_out else str(payload.get("status", "error")),
            proved_optimal=False,
            objective_value=_finite_or_none(payload.get("objective_value")),
            extra={
                "execution_backend": "python",
                "python_module": spec.python_module,
                "python_symbol": spec.python_symbol,
                "result_path": str(output_path),
                "hermax_error": str(payload["error"]),
            },
        )
        raise ExternalSolverError(str(payload["error"]), metadata)
    solution = payload.get("solution")
    if not isinstance(solution, list):
        metadata = _base_metadata(
            baseline_name=spec.baseline_name,
            instance=instance,
            result=result,
            solver_status="malformed_output",
            proved_optimal=False,
            extra={
                "execution_backend": "python",
                "python_module": spec.python_module,
                "python_symbol": spec.python_symbol,
                "result_path": str(output_path),
            },
        )
        raise ExternalSolverError("Hermax worker returned malformed solution payload.", metadata)
    timed_out = result.timed_out or bool(payload.get("time_limit_hit", False))
    metadata = _base_metadata(
        baseline_name=spec.baseline_name,
        instance=instance,
        result=result,
        solver_status="timeout" if timed_out else str(payload.get("status", "unknown")),
        proved_optimal=bool(payload.get("proved_optimal", False)) and not timed_out,
        objective_value=_finite_or_none(payload.get("objective_value")),
        best_bound=_finite_or_none(payload.get("best_bound")),
        mip_gap=_finite_or_none(payload.get("mip_gap")),
        extra={
            "execution_backend": "python",
            "python_module": spec.python_module,
            "python_symbol": spec.python_symbol,
            "solver_internal_runtime_ms": _finite_or_none(payload.get("solver_runtime_ms")),
            "hermax_cost": _finite_or_none(payload.get("cost")),
            "result_path": str(output_path),
        },
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_open_wbo(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    return _maxsat_wcnf_command_solver(
        instance,
        binary_path,
        config,
        artifact_dir=artifact_dir,
        baseline_name="open_wbo_exact",
    )


def serialize_metis_graph(instance: dict[str, object]) -> str:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    lines = [f"{num_vertices} {len(instance['edges'])}"]
    for vertex in range(num_vertices):
        lines.append(" ".join(str(neighbor + 1) for neighbor in sorted(adjacency[vertex])))
    return "\n".join(lines) + "\n"


def _normalize_vertex_list(values: list[int], *, num_vertices: int) -> list[int]:
    if not values:
        return []
    if min(values) >= 1 and max(values) <= num_vertices:
        return sorted({value - 1 for value in values})
    return sorted({value for value in values if 0 <= value < num_vertices})


def parse_kamis_output(
    text: str,
    *,
    num_vertices: int,
    solution_file_text: str | None = None,
) -> tuple[list[int], str, bool]:
    combined = "\n".join(part for part in (text, solution_file_text or "") if part)
    status = "unknown"
    proved_optimal = False
    independent_set: list[int] | None = None
    vertex_cover: list[int] | None = None
    for raw_line in combined.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        upper = line.upper()
        if line.startswith("s "):
            status = line[2:].strip()
            proved_optimal = "OPTIMUM" in upper or "OPTIMAL" in upper
        tokens = [int(item) for item in re.findall(r"-?\d+", line)]
        if not tokens:
            continue
        lowered = line.lower()
        if lowered.startswith("v ") or "independent" in lowered or "mis" in lowered:
            independent_set = _normalize_vertex_list(tokens, num_vertices=num_vertices)
        elif lowered.startswith("vc ") or "vertex cover" in lowered or "cover" in lowered:
            vertex_cover = _normalize_vertex_list(tokens, num_vertices=num_vertices)
        elif len(tokens) == num_vertices and set(tokens).issubset({0, 1}):
            independent_set = [index for index, value in enumerate(tokens) if value == 1]
    if independent_set is None and vertex_cover is not None:
        cover = set(vertex_cover)
        independent_set = [vertex for vertex in range(num_vertices) if vertex not in cover]
    if independent_set is None:
        raise ValueError("KaMIS output did not contain a parseable independent set or vertex cover.")
    return sorted(independent_set), status, proved_optimal


def _solve_kamis_mis(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    input_path = instance_dir / "instance.graph"
    input_path.write_text(serialize_metis_graph(instance), encoding="utf-8")
    result = _run_external_command(
        [binary_path, str(input_path)],
        log_dir=instance_dir,
        timeout_seconds=config.time_limit_seconds,
    )
    solution_path = input_path.with_suffix(input_path.suffix + ".solution")
    solution_text = solution_path.read_text(encoding="utf-8") if solution_path.exists() else None
    solution, status, proved_optimal = parse_kamis_output(
        "\n".join([result.stdout_text, result.stderr_text]),
        num_vertices=int(instance["num_vertices"]),
        solution_file_text=solution_text,
    )
    metadata = _base_metadata(
        baseline_name="kamis_vc_exact",
        instance=instance,
        result=result,
        solver_status="timeout" if result.timed_out else status,
        proved_optimal=proved_optimal and not result.timed_out,
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def serialize_mis_lp(instance: dict[str, object]) -> str:
    num_vertices = int(instance["num_vertices"])
    lines = ["Maximize", " obj: " + " + ".join(f"x_{vertex}" for vertex in range(num_vertices))]
    lines.append("Subject To")
    for raw_u, raw_v in instance["edges"]:
        u, v = int(raw_u), int(raw_v)
        lines.append(f" edge_{u}_{v}: x_{u} + x_{v} <= 1")
    lines.append("Binary")
    for vertex in range(num_vertices):
        lines.append(f" x_{vertex}")
    lines.append("End")
    return "\n".join(lines) + "\n"


def serialize_mds_lp(instance: dict[str, object]) -> str:
    num_vertices = int(instance["num_vertices"])
    adjacency = adjacency_sets(num_vertices, instance["edges"])
    lines = ["Minimize", " obj: " + " + ".join(f"x_{vertex}" for vertex in range(num_vertices))]
    lines.append("Subject To")
    for vertex in range(num_vertices):
        closed = sorted(adjacency[vertex] | {vertex})
        lines.append(f" dom_{vertex}: " + " + ".join(f"x_{item}" for item in closed) + " >= 1")
    lines.append("Binary")
    for vertex in range(num_vertices):
        lines.append(f" x_{vertex}")
    lines.append("End")
    return "\n".join(lines) + "\n"


def serialize_coloring_lp(instance: dict[str, object]) -> str:
    num_vertices = int(instance["num_vertices"])
    lines = ["Minimize", " obj: " + " + ".join(f"y_{color}" for color in range(num_vertices))]
    lines.append("Subject To")
    for vertex in range(num_vertices):
        lines.append(
            f" assign_{vertex}: "
            + " + ".join(f"x_{vertex}_{color}" for color in range(num_vertices))
            + " = 1"
        )
    for raw_u, raw_v in instance["edges"]:
        u, v = int(raw_u), int(raw_v)
        for color in range(num_vertices):
            lines.append(f" edge_{u}_{v}_{color}: x_{u}_{color} + x_{v}_{color} <= 1")
    for vertex in range(num_vertices):
        for color in range(num_vertices):
            lines.append(f" use_{vertex}_{color}: x_{vertex}_{color} - y_{color} <= 0")
    for color in range(num_vertices - 1):
        lines.append(f" sym_{color}: y_{color} - y_{color + 1} >= 0")
    lines.append("Binary")
    for vertex in range(num_vertices):
        lines.extend(f" x_{vertex}_{color}" for color in range(num_vertices))
    lines.extend(f" y_{color}" for color in range(num_vertices))
    lines.append("End")
    return "\n".join(lines) + "\n"


def serialize_packing_lp(instance: dict[str, object], *, binary: bool = False) -> str:
    num_items = int(instance["num_items"])
    num_resources = int(instance["num_resources"])
    values = [float(value) for value in instance["values"]]
    weights = [[float(value) for value in row] for row in instance["weights"]]
    capacities = [float(value) for value in instance["capacities"]]
    lines = ["Maximize", " obj: " + " + ".join(f"{values[item]:.12g} x_{item}" for item in range(num_items))]
    lines.append("Subject To")
    for resource in range(num_resources):
        expression = " + ".join(f"{weights[item][resource]:.12g} x_{item}" for item in range(num_items))
        lines.append(f" cap_{resource}: {expression} <= {capacities[resource]:.12g}")
    if binary:
        lines.append("Binary")
        for item in range(num_items):
            lines.append(f" x_{item}")
    else:
        lines.append("Bounds")
        for item in range(num_items):
            lines.append(f" 0 <= x_{item} <= 1")
    lines.append("End")
    return "\n".join(lines) + "\n"


def parse_highs_solution_file(text: str) -> tuple[dict[str, float], float | None, str, bool, float | None, float | None]:
    values: dict[str, float] = {}
    objective_value: float | None = None
    best_bound: float | None = None
    mip_gap: float | None = None
    status = "unknown"
    proved_optimal = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if "model status" in lowered or lowered.startswith("status"):
            status = line.split(":", 1)[-1].strip() if ":" in line else line
            proved_optimal = proved_optimal or "optimal" in lowered
        elif lowered in {"optimal", "optimal solution"} or "optimal" in lowered and "not" not in lowered:
            status = "optimal"
            proved_optimal = True
        elif "time limit" in lowered:
            status = "time_limit"
        elif lowered.startswith("objective") or lowered.startswith("objective value"):
            numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if numbers:
                objective_value = float(numbers[-1])
        elif "best bound" in lowered or lowered.startswith("bound"):
            numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line)
            if numbers:
                best_bound = float(numbers[-1])
        elif "gap" in lowered:
            numbers = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", line.replace("%", ""))
            if numbers:
                mip_gap = float(numbers[-1])
        else:
            parts = line.split()
            if len(parts) >= 2 and re.fullmatch(r"[xyu]_\d+(?:_\d+)?", parts[0]):
                try:
                    values[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return values, objective_value, status, proved_optimal, best_bound, mip_gap


def parse_scip_solution_file(text: str) -> tuple[dict[str, float], float | None, str, bool]:
    values: dict[str, float] = {}
    objective_value: float | None = None
    status = "unknown"
    proved_optimal = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if lowered.startswith("objective value"):
            try:
                objective_value = float(line.split(":", 1)[1].strip())
            except (IndexError, ValueError):
                objective_value = None
        elif "optimal" in lowered:
            status = line
            proved_optimal = True
        else:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    values[parts[0]] = float(parts[1])
                except ValueError:
                    continue
    return values, objective_value, status, proved_optimal


def _parse_scip_status(text: str, fallback_status: str, fallback_proved: bool) -> tuple[str, bool]:
    lowered = text.lower()
    if "optimal solution found" in lowered or "status" in lowered and "optimal" in lowered:
        return "optimal", True
    if "time limit" in lowered:
        return "time_limit", False
    return fallback_status, fallback_proved


def _run_scip(
    *,
    binary_path: str,
    config: ExternalExactConfig,
    instance_dir: Path,
    lp_text: str,
) -> tuple[ExternalCommandResult, dict[str, float], float | None, str, bool]:
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_path = instance_dir / "model.lp"
    solution_path = instance_dir / "solution.sol"
    model_path.write_text(lp_text, encoding="utf-8")
    command = [
        binary_path,
        "-q",
        "-c",
        f"read {model_path}",
        "-c",
        f"set limits time {config.time_limit_seconds}",
        "-c",
        f"set parallel/maxnthreads {config.threads}",
        "-c",
        "optimize",
        "-c",
        f"write solution {solution_path}",
        "-c",
        "quit",
    ]
    result = _run_external_command(command, log_dir=instance_dir, timeout_seconds=config.time_limit_seconds + 5.0)
    solution_text = solution_path.read_text(encoding="utf-8") if solution_path.exists() else ""
    values, objective, status, proved = parse_scip_solution_file(solution_text)
    status, proved = _parse_scip_status("\n".join([result.stdout_text, result.stderr_text, solution_text]), status, proved)
    if not values:
        raise ValueError("SCIP did not write a parseable solution file.")
    return result, values, objective, status, proved


def _run_highs(
    *,
    binary_path: str,
    config: ExternalExactConfig,
    instance_dir: Path,
    lp_text: str,
) -> tuple[ExternalCommandResult, dict[str, float], float | None, str, bool, float | None, float | None]:
    instance_dir.mkdir(parents=True, exist_ok=True)
    model_path = instance_dir / "model.lp"
    solution_path = instance_dir / "solution.sol"
    model_path.write_text(lp_text, encoding="utf-8")
    command = [
        binary_path,
        f"--model_file={model_path}",
        f"--solution_file={solution_path}",
        f"--time_limit={config.time_limit_seconds}",
        f"--threads={config.threads}",
    ]
    result = _run_external_command(command, log_dir=instance_dir, timeout_seconds=config.time_limit_seconds + 5.0)
    solution_text = solution_path.read_text(encoding="utf-8") if solution_path.exists() else ""
    combined = "\n".join([result.stdout_text, result.stderr_text, solution_text])
    values, objective, status, proved, best_bound, mip_gap = parse_highs_solution_file(combined)
    if result.timed_out:
        status = "timeout"
        proved = False
    if not values:
        raise ValueError("HiGHS did not write a parseable solution.")
    return result, values, objective, status, proved, best_bound, mip_gap


def _packing_values_to_solution(instance: dict[str, object], values: dict[str, float]) -> list[float]:
    return [
        max(0.0, min(1.0, float(values.get(f"x_{item}", 0.0))))
        for item in range(int(instance["num_items"]))
    ]


def _mdkp_values_to_solution(instance: dict[str, object], values: dict[str, float]) -> list[int]:
    return [
        item
        for item in range(int(instance["num_items"]))
        if float(values.get(f"x_{item}", 0.0)) > 0.5
    ]


def _solve_scip_mis(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved = _run_scip(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_mis_lp(instance),
    )
    solution = [
        vertex
        for vertex in range(int(instance["num_vertices"]))
        if values.get(f"x_{vertex}", 0.0) > 0.5
    ]
    metadata = _base_metadata(
        baseline_name="scip_mis_exact",
        instance=instance,
        result=result,
        solver_status="timeout" if result.timed_out else status,
        proved_optimal=proved and not result.timed_out,
        objective_value=objective,
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_scip_mds(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved = _run_scip(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_mds_lp(instance),
    )
    solution = [
        vertex
        for vertex in range(int(instance["num_vertices"]))
        if values.get(f"x_{vertex}", 0.0) > 0.5
    ]
    metadata = _base_metadata(
        baseline_name="scip_mip_exact",
        instance=instance,
        result=result,
        solver_status="timeout" if result.timed_out else status,
        proved_optimal=proved and not result.timed_out,
        objective_value=objective,
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_scip_packing_lp(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved = _run_scip(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_packing_lp(instance, binary=False),
    )
    metadata = _base_metadata(
        baseline_name="scip_lp_exact",
        instance=instance,
        result=result,
        solver_status="timeout" if result.timed_out else status,
        proved_optimal=proved and not result.timed_out,
        objective_value=objective,
    )
    return SolveOutcome(solution=_packing_values_to_solution(instance, values), metadata=metadata)


def _solve_scip_mdkp(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved = _run_scip(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_packing_lp(instance, binary=True),
    )
    metadata = _base_metadata(
        baseline_name="scip_mdkp_exact",
        instance=instance,
        result=result,
        solver_status="timeout" if result.timed_out else status,
        proved_optimal=proved and not result.timed_out,
        objective_value=objective,
    )
    return SolveOutcome(solution=_mdkp_values_to_solution(instance, values), metadata=metadata)


def _solve_highs_packing_lp(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved, best_bound, mip_gap = _run_highs(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_packing_lp(instance, binary=False),
    )
    metadata = _base_metadata(
        baseline_name="highs_lp_exact",
        instance=instance,
        result=result,
        solver_status=status,
        proved_optimal=proved,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
    )
    return SolveOutcome(solution=_packing_values_to_solution(instance, values), metadata=metadata)


def _solve_highs_mis(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved, best_bound, mip_gap = _run_highs(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_mis_lp(instance),
    )
    solution = [
        vertex
        for vertex in range(int(instance["num_vertices"]))
        if values.get(f"x_{vertex}", 0.0) > 0.5
    ]
    metadata = _base_metadata(
        baseline_name="highs_mis_mip_exact",
        instance=instance,
        result=result,
        solver_status=status,
        proved_optimal=proved,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_highs_mds(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved, best_bound, mip_gap = _run_highs(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_mds_lp(instance),
    )
    solution = [
        vertex
        for vertex in range(int(instance["num_vertices"]))
        if values.get(f"x_{vertex}", 0.0) > 0.5
    ]
    metadata = _base_metadata(
        baseline_name="highs_mds_mip_exact",
        instance=instance,
        result=result,
        solver_status=status,
        proved_optimal=proved,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_highs_mdkp(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved, best_bound, mip_gap = _run_highs(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_packing_lp(instance, binary=True),
    )
    metadata = _base_metadata(
        baseline_name="highs_mip_exact",
        instance=instance,
        result=result,
        solver_status=status,
        proved_optimal=proved,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
    )
    return SolveOutcome(solution=_mdkp_values_to_solution(instance, values), metadata=metadata)


def _solve_highs_coloring(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved, best_bound, mip_gap = _run_highs(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_coloring_lp(instance),
    )
    num_vertices = int(instance["num_vertices"])
    solution = []
    for vertex in range(num_vertices):
        color = max(
            range(num_vertices),
            key=lambda candidate: values.get(f"x_{vertex}_{candidate}", 0.0),
        )
        solution.append(color)
    metadata = _base_metadata(
        baseline_name="highs_coloring_mip_exact",
        instance=instance,
        result=result,
        solver_status=status,
        proved_optimal=proved,
        objective_value=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_scip_coloring(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved = _run_scip(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_coloring_lp(instance),
    )
    num_vertices = int(instance["num_vertices"])
    solution = []
    for vertex in range(num_vertices):
        color = next(
            (
                color
                for color in range(num_vertices)
                if values.get(f"x_{vertex}_{color}", 0.0) > 0.5
            ),
            0,
        )
        solution.append(color)
    metadata = _base_metadata(
        baseline_name="scip_coloring_exact",
        instance=instance,
        result=result,
        solver_status="timeout" if result.timed_out else status,
        proved_optimal=proved and not result.timed_out,
        objective_value=objective,
    )
    return SolveOutcome(solution=solution, metadata=metadata)


def _solve_scip_tsp_mtz_cli(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    result, values, objective, status, proved = _run_scip(
        binary_path=binary_path,
        config=config,
        instance_dir=artifact_dir / _safe_id(instance),
        lp_text=serialize_tsp_mtz_lp(instance),
    )
    num_cities = int(instance["num_cities"])
    edge_values = {
        (left, right): values.get(f"x_{left}_{right}", 0.0)
        for left in range(num_cities)
        for right in range(num_cities)
        if left != right
    }
    metadata = _base_metadata(
        baseline_name="scip_tsp_mtz_exact",
        instance=instance,
        result=result,
        solver_status="timeout" if result.timed_out else status,
        proved_optimal=proved and not result.timed_out,
        objective_value=objective,
    )
    return SolveOutcome(solution=_tour_from_edge_values(edge_values, num_cities=num_cities), metadata=metadata)


def serialize_tsp_mtz_lp(instance: dict[str, object]) -> str:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    edge_terms = [
        f"{matrix[left][right]:.12g} x_{left}_{right}"
        for left in range(num_cities)
        for right in range(num_cities)
        if left != right
    ]
    lines = ["Minimize", " obj: " + " + ".join(edge_terms)]
    lines.append("Subject To")
    for city in range(num_cities):
        lines.append(
            f" out_{city}: "
            + " + ".join(f"x_{city}_{other}" for other in range(num_cities) if other != city)
            + " = 1"
        )
        lines.append(
            f" in_{city}: "
            + " + ".join(f"x_{other}_{city}" for other in range(num_cities) if other != city)
            + " = 1"
        )
    for left in range(1, num_cities):
        for right in range(1, num_cities):
            if left == right:
                continue
            lines.append(f" mtz_{left}_{right}: u_{left} - u_{right} + {num_cities} x_{left}_{right} <= {num_cities - 1}")
    lines.append("Bounds")
    lines.append(" 0 <= u_0 <= 0")
    for city in range(1, num_cities):
        lines.append(f" 1 <= u_{city} <= {num_cities - 1}")
    lines.append("Binary")
    for left in range(num_cities):
        for right in range(num_cities):
            if left != right:
                lines.append(f" x_{left}_{right}")
    lines.append("End")
    return "\n".join(lines) + "\n"


def serialize_tsplib_explicit(instance: dict[str, object], *, scale: int = 1_000_000) -> str:
    num_cities = int(instance["num_cities"])
    matrix = distance_matrix(instance["points"])
    lines = [
        f"NAME: {instance.get('id', 'dasbench_tsp')}",
        "TYPE: TSP",
        f"DIMENSION: {num_cities}",
        "EDGE_WEIGHT_TYPE: EXPLICIT",
        "EDGE_WEIGHT_FORMAT: FULL_MATRIX",
        "EDGE_WEIGHT_SECTION",
    ]
    for row in matrix:
        lines.append(" ".join(str(int(round(value * scale))) for value in row))
    lines.append("EOF")
    return "\n".join(lines) + "\n"


def parse_concorde_solution(text: str, *, num_cities: int) -> list[int]:
    values = [int(item) for item in re.findall(r"-?\d+", text)]
    if not values:
        raise ValueError("Concorde solution output did not contain integers.")
    if values[0] == num_cities and len(values) >= num_cities + 1:
        values = values[1 : num_cities + 1]
    else:
        values = values[:num_cities]
    return canonicalize_tour(values, num_cities)


def _solve_concorde(
    instance: dict[str, object],
    binary_path: str,
    config: ExternalExactConfig,
    *,
    artifact_dir: Path,
) -> SolveOutcome:
    instance_dir = artifact_dir / _safe_id(instance)
    instance_dir.mkdir(parents=True, exist_ok=True)
    tsp_path = instance_dir / "instance.tsp"
    tsp_path.write_text(serialize_tsplib_explicit(instance), encoding="utf-8")
    result = _run_external_command(
        [binary_path, str(tsp_path)],
        log_dir=instance_dir,
        timeout_seconds=config.time_limit_seconds,
    )
    solution_paths = [instance_dir / "instance.sol", tsp_path.with_suffix(".sol")]
    solution_text = ""
    for path in solution_paths:
        if path.exists():
            solution_text = path.read_text(encoding="utf-8")
            break
    if not solution_text:
        solution_text = "\n".join([result.stdout_text, result.stderr_text])
    solution = parse_concorde_solution(solution_text, num_cities=int(instance["num_cities"]))
    combined = "\n".join([result.stdout_text, result.stderr_text])
    proved_optimal = "optimal" in combined.lower() and not result.timed_out
    status = "optimal" if proved_optimal else ("timeout" if result.timed_out else "unknown")
    metadata = _base_metadata(
        baseline_name="concorde_exact",
        instance=instance,
        result=result,
        solver_status=status,
        proved_optimal=proved_optimal,
    )
    return SolveOutcome(solution=solution, metadata=metadata)
