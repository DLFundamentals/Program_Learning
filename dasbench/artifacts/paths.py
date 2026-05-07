from __future__ import annotations

from pathlib import Path

from dasbench.utils import timestamp_token

DEFAULT_DATASETS_ROOT = Path("artifacts/datasets")
DEFAULT_AGENT_RUNS_ROOT = Path("artifacts/agent_runs")
DEFAULT_REPORTS_ROOT = Path("artifacts/reports")


def default_dataset_dir(problem: str, family: str, dataset_id: str | None = None) -> Path:
    resolved_id = dataset_id or timestamp_token()
    return DEFAULT_DATASETS_ROOT / problem / family / resolved_id


def default_agent_run_dir(problem: str, family: str, run_id: str | None = None) -> Path:
    resolved_id = run_id or timestamp_token()
    return DEFAULT_AGENT_RUNS_ROOT / problem / family / resolved_id


def default_report_dir(problem: str, family: str, run_id: str) -> Path:
    return DEFAULT_REPORTS_ROOT / problem / family / run_id
