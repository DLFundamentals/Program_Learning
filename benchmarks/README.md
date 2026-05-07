# Benchmark Experiments

The paper-facing experiments are runnable from this package. The main paper benchmark is
`second_scale_benchmark_v2`; `benchmarks.main_paper_benchmark` is the public alias used in
submission instructions. Existing artifacts still use the historical condition id
`seconds_scale_v2` under `targets/seconds_scale_v2/...`.

Generated datasets, solver candidates, reports, and aggregate outputs are not committed. Regenerate
them locally, or provide them through an anonymous external artifact archive for review.

| Experiment | Supports | Command | Artifact dependency |
| --- | --- | --- | --- |
| Main Paper Benchmark | Headline quality/runtime results and per-target table | `python -m benchmarks.main_paper_benchmark --max-workers 21` | Starts from scratch |
| Legacy second-scale benchmark | Older per-problem calibrated benchmark | `python -m benchmarks.second_scale_benchmark --max-workers 4` | Starts from scratch |
| Sample Size | Sample-size ablation | `python -m benchmarks.sample_size_sweep --validation-size 32 --max-workers 4` | Starts from scratch |
| Problem Size | Problem-size and solver-transfer curves | `python -m benchmarks.problem_size_sweep --max-workers 4` | Starts from scratch |
| Candidate Count | Candidate-width ablation | `python -m benchmarks.candidate_count_sweep --max-workers 4` | Starts from scratch |
| Iteration Count | Iterative synthesis runtime figure | `python -m benchmarks.iteration_count_sweep --max-workers 4` | Starts from scratch |
| No-Hint Recovery | Hidden-rule framing ablation | `python -m benchmarks.no_hint_recovery_benchmark --source-run-root "$MAIN_SWEEP_ROOT" --max-workers 4` | Needs a completed main benchmark run |
| Graph Relabel Invariance | Graph presentation perturbation ablation | `python -m benchmarks.graph_relabel_invariance_benchmark --source-run-root "$MAIN_SWEEP_ROOT" --max-workers 4` | Needs a completed main benchmark run |
| PACE 2025 Dominating Set | External PACE diagnostic | `python -m benchmarks.pace2025_dominating_set --track heuristic --test-source private` | Downloads or reads PACE instances; needs LLM API for synthesis |

`$MAIN_SWEEP_ROOT` should point to a completed main benchmark sweep root, for example
`artifacts/second_scale_benchmark_v2/<sweep_id>`.

## Defaults

- `--generator llm`
- Gurobi enabled with one thread
- external exact baselines in `auto`
- main paper benchmark defaults to all 21 target distributions
- ablation sweeps default to representative families unless their module documents otherwise
- each sweep writes to `artifacts/<sweep_kind>/<sweep_id>/`
- completed targets are resumable unless `--force` is passed

## Reusable Helpers

Reusable report/export helpers live in `scripts/`:

```bash
python -m scripts.collect_experiment_results artifacts/second_scale_benchmark_v2/<sweep_id> exports/main_paper
python -m scripts.export_baseline_catalog --output exports/baseline_catalog.json
python -m scripts.export_problem_size_runtimes artifacts/problem_size_sweep/<sweep_id>
python -m scripts.plot_iteration_best_runtime_so_far artifacts/iteration_count_sweep/<sweep_id>
python -m scripts.plot_iteration_runtime_ratio_vs_zero_shot artifacts/iteration_count_sweep/<sweep_id>
python -m scripts.plot_problem_size_solver_transfer artifacts/problem_size_sweep/<sweep_id>
```

PACE helper scripts are kept because they support the external Dominating Set diagnostic:

```bash
python -m scripts.pace2025_run_heuristic_baselines --count 5
python -m scripts.pace2025_collect_heuristic_report
```

Local run-management utilities used during development, such as failed-run cleanup, candidate
removal, tail finishers, diagnostic patchers, and missing-runtime rerunners, are intentionally omitted
from the submission tree. Their provenance remains in git history.

## Smoke Checks

```bash
python -m benchmarks.main_paper_benchmark --dry-run --problem tsp
python -m benchmarks.second_scale_benchmark_v2 --dry-run --problem tsp
python -m benchmarks.sample_size_sweep --validation-size 32 --dry-run --problem maxsat --family last_clause_signal_v1
python -m benchmarks.problem_size_sweep --dry-run --problem tsp
python -m benchmarks.candidate_count_sweep --dry-run --problem tsp
python -m benchmarks.iteration_count_sweep --dry-run --problem tsp
python -m benchmarks.pace2025_dominating_set --help
```
