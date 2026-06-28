<h1 align="center">Distribution-Aware Algorithm Design with LLM Agents</h1>

<p align="center">
  <em>Learn the structure hiding in your problem distribution — then compile it into a solver that runs orders of magnitude faster.</em>
</p>

<p align="center">
  <a href="#-install"><img alt="Python" src="https://img.shields.io/badge/python-3.x-blue.svg"></a>
  <a href="https://github.com/DLFundamentals/Program_Learning"><img alt="Status" src="https://img.shields.io/badge/status-research%20preview-orange.svg"></a>
  <a href="#-citation"><img alt="Paper" src="https://img.shields.io/badge/paper-preprint-8A2BE2.svg"></a>
  <a href="#-license"><img alt="License" src="https://img.shields.io/badge/license-see%20repo-lightgrey.svg"></a>
</p>

<p align="center">
  Saharsh Koganti<sup>1</sup> · Priyadarsi Mishra<sup>1</sup> · Pierfrancesco Beneventano<sup>2</sup> · Tomer Galanti<sup>1</sup><br>
  <sub><sup>1</sup>Texas A&amp;M University &nbsp;&nbsp; <sup>2</sup>Massachusetts Institute of Technology</sub>
</p>

---

## What this is

Most optimization problems are solved over and over against instances drawn from the same hidden process — a router, scheduler, compiler, or service sees a *distribution* of instances, not arbitrary worst cases. Even when the ambient problem is worst-case hard, that distribution often carries reusable structure: recurring geometry, latent decompositions, active-resource patterns, planted assignments.

This repository studies **distribution-aware program learning**: given only *samples* from an unknown deployment distribution, can we synthesize executable solver code that is fast on future instances while keeping solution quality high?

The central abstraction is a **solver hint** — distribution-specific structure inferred from samples and compiled into a specialized solver. The learner never sees the distribution analytically; it must discover what makes future instances easier and turn that into code:

$$
\underbrace{S \sim D^{\,n}}_{\text{samples}}
\;\xrightarrow{\;\text{learn}\;}\;
\underbrace{\widehat{h}_S}_{\text{hint}}
\;\xrightarrow{\;\text{compile } \mathrm{Comp}\;}\;
\underbrace{\widehat{c}_S = \mathrm{Comp}(\widehat{h}_S)}_{\text{deployed solver}}
$$

The samples are not used to predict solutions to observed instances — they are used to discover what makes *future* instances from the same source easier to solve.

`dasbench`, the framework in this repo, is a unified benchmark for **distribution-aware algorithm synthesis** on hard combinatorial problems, with an LLM code agent as the (approximate) sample → hint → solver procedure.

---

## Why it matters

Three access models for designing a solver against a distribution $D$:

| Access model | Information about $D$ | Learned representation |
| --- | --- | --- |
| Worst-case design | none | none |
| Average-case complexity | $D$ specified analytically | none |
| **This work** | **samples $S \sim D^{\,n}$** | **hint $\widehat{h}_S \rightarrow$ solver $\widehat{c}_S$** |

We sit in the realistic middle ground: the distribution is observed only through examples. Correctness is handled by verification / repair / fallback, so the learned component is free to focus on the *shortcut* — a SAT backdoor, a graph separator, a geometric template, an active-constraint pattern — that makes deployment cheap.

---

## Headline results

Across **21 structured combinatorial-optimization distributions** spanning **7 problem classes**, the synthesized solvers reach **mean normalized quality 0.971** while running far faster than classical and solver-backed baselines.

| Comparator | Quality lift (Δ) | Runtime ratio (faster than ours) |
| --- | :---: | :---: |
| Fast high-quality heuristic | **+0.109** | **564.9×** |
| Gurobi (10 s, 1 thread) | — | **345.1×** |
| Time-limited exact backend | — | **16.9×** |
| One-shot Codex | −0.016 (≈ tie) | **4.5×** |
| One-shot Claude Code | +0.085 | **17.4×** |
| Best-of-5 open model (Gemma 4) | +0.145 | 2.4× |

> No single baseline is *both* faster and higher quality across the suite. The method improves the average quality–runtime frontier rather than dominating every family — it trails the strongest heuristic on TSP and the ML baseline on Coloring.

**External test — PACE 2025 Dominating Set (private instances).** The synthesized solver is **valid on all 100 graphs**, runs roughly **75×–125× faster** than released competition solvers, and lands within **a few percent** of their solution size — strictly better than everything in its own speed class.

See the paper for full per-target tables, iteration ablations, perturbation-robustness diagnostics, and the discovered computation patterns.

---

## How it works

For each candidate $c = (H_c, A_c, s_c)$ the agent produces three things through sequential LLM calls:

1. **Hypothesis** $H_c$ — a structured guess about the hidden distributional rule, the evidence to measure, and the implied solver strategy.
2. **Analysis program** $A_c$ — runs once on the public training sample $S_{\mathrm{tr}}^{\mathrm{pub}}$ and compresses the evidence into a compact, reusable summary, the empirical hint $a_c = A_c(S_{\mathrm{tr}}^{\mathrm{pub}})$.
3. **Solver** $s_c$ — deployment code that maps a new instance and the hint to a solution, $z = s_c(x, a_c)$, with a fallback for weak or ambiguous structure.

Candidates are generated in a diversity-preserving beam, evaluated on public splits, ranked lexicographically by $(Q_{\mathrm{val}}, O_{\mathrm{val}}, -T_{\mathrm{val}})$ — validation quality, optimality, and (negative) runtime — and refined (refine / fork / replace / push runtime / push quality). The best candidate across all rounds is re-analyzed and deployed.

The public view is sanitized: family identity, planted rules, optimum solutions, and optimum objective values are stripped before any synthesized code runs. The agent sees only the instance format, the scoring rule, and the samples.

---

## Repository layout

```
Program_Learning/
├── dasbench/            # core framework: problems, families, baselines, exact solvers, synthesis loop
│   └── prompts/         # LLM system prompt used by the generator
├── benchmarks/          # paper-facing sweep suites and ablations (see benchmarks/README.md)
├── scripts/             # helper scripts
├── tests/               # test suite
├── main.py              # CLI entry point: generate / run-agent / report / benchmark
├── REPRODUCIBILITY.md   # full reproduction notes incl. PACE 2025 diagnostic
├── pyproject.toml
└── .env.example
```

Generated datasets, candidates, and reports are written under `artifacts/` and are intentionally **not** committed — regenerate them with the commands below.

```
artifacts/datasets/<problem>/<family>/<dataset_id>/
artifacts/agent_runs/<problem>/<family>/<run_id>/
artifacts/reports/<problem>/<family>/<run_id>/
```

---

## 📦 Install

```bash
uv sync
```

For the LLM generator, set the OpenAI-compatible environment variables (see `.env.example`):

```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
OPENAI_MODEL=gpt-5.2
OPENAI_REASONING_EFFORT=xhigh
# Optional, for OpenAI-compatible endpoints:
# OPENAI_BASE_URL=https://api.openai.com/v1
```

The **template generator** is fully local and is the recommended smoke-test path — no API key required.

---

## 🚀 Quick start

**Generate a dataset** (stores exact optima with each instance):

```bash
python main.py generate \
  --problem mis \
  --family motif_bridge_mixture_v1 \
  --dataset-id smoke_mis \
  --instance-param num_vertices=18 \
  --train-size 32 --validation-size 16 --test-size 16
```

**Run baselines + synthesis** on an existing dataset:

```bash
python main.py run-agent \
  --dataset-dir artifacts/datasets/mis/motif_bridge_mixture_v1/smoke_mis \
  --generator template \
  --mode beam --iterations 3 --beam-width 3
```

**Write a repeated benchmark report** for a completed run:

```bash
python main.py report \
  --dataset-dir artifacts/datasets/mis/motif_bridge_mixture_v1/smoke_mis \
  --agent-run-dir artifacts/agent_runs/mis/motif_bridge_mixture_v1/<run_id> \
  --repeats 10
```

**Run the full flow end to end** for one family:

```bash
python main.py benchmark \
  --problem mds --family gateway_overlap_cover_v1 \
  --generator template \
  --instance-param num_vertices=20 \
  --train-size 64 --validation-size 32 --test-size 32
```

**Run every family** for one problem, or across all problems (parallel by default):

```bash
python main.py benchmark --problem maxsat --all-families
python main.py benchmark --all-families --max-parallel 4
```

### Optional baselines

The default-on timed **Gurobi** industrial baseline can be disabled or tuned:

```bash
python main.py run-agent --dataset-dir <dir> --no-gurobi-baseline
python main.py run-agent --dataset-dir <dir> --gurobi-time-limit-seconds 30 --gurobi-threads 1
```

Optional **external exact solvers** are discovery-based via `auto`. In `auto` mode, configured binaries take precedence; without one, MaxSAT rows fall back to `hermax` (Open-WBO / UWrMaxSAT / EvalMaxSAT / WMaxCDCL), SCIP rows to `pyscipopt`, and HiGHS rows to `highspy`. MaxHS, KaMIS, and Concorde remain CLI-based.

```bash
export DASBENCH_OPEN_WBO_BIN=/path/to/open-wbo
export DASBENCH_CONCORDE_BIN=/path/to/concorde
# ... see .env.example for the full list

python main.py run-agent \
  --dataset-dir artifacts/datasets/maxsat/last_clause_signal_v1/smoke_maxsat \
  --external-exact-baselines auto \
  --external-time-limit-seconds 60 --external-threads 1
```

### Paper sweeps

```bash
python -m benchmarks.main_paper_benchmark --max-workers 21
python -m benchmarks.sample_size_sweep --validation-size 32 --max-workers 4
python -m benchmarks.problem_size_sweep --max-workers 4
python -m benchmarks.candidate_count_sweep --max-workers 4
```

The headline paper results use `benchmarks.main_paper_benchmark` (a wrapper over `benchmarks.second_scale_benchmark_v2`). Additional ablations and the PACE 2025 diagnostic are documented in [`benchmarks/README.md`](benchmarks/README.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## Benchmark suite

7 problem classes × 3 hidden distribution families = **21 targets**. Each target is a *distribution* over structured instances (not arbitrary worst cases), split into train / validation / test with stored exact optima.

| Problem | Families | The exploitable signal |
| --- | --- | --- |
| **Coloring** | `cluster_ring_mix_v1`, `planted_palette_overlap_v1`, `separator_palette_trap_v1` | Recover a global planted palette / separator structure instead of coloring greedily. |
| **MAXSAT** | `last_clause_signal_v1`, `latent_backdoor_mixture_v1`, `community_parity_overlay_v1` | Anchor bits / latent backdoors determine most variables via hidden Boolean rules. |
| **MIS** | `clique_path_mix_v1`, `motif_bridge_mixture_v1`, `core_fringe_trap_v1` | Decompose into motifs / blocks and handle bridge conflicts and core-fringe traps. |
| **MDS** | `star_cluster_cover_v1`, `gateway_overlap_cover_v1`, `geometric_cluster_cover_v1` | Pick stable hubs / gateways for overlap-aware coverage, not raw degree. |
| **Packing LP** | `single_bottleneck_fractional_v1`, `latent_active_basis_v1`, `block_coupled_resource_v1` | Infer the binding resource / active basis and sort by value per unit of it. |
| **MDKP** | `single_resource_density_v1`, `latent_class_knapsack_v1`, `decoy_complement_mixture_v1` | Identify the recurring bottleneck and prefer complementary, low-pressure bundles. |
| **TSP** | `clustered_euclidean_v1`, `paired_ribbon_zigzag_v1`, `latent_metric_mixture_v1` | Classify the latent geometry and construct tours that respect it. |

The candidate interface is intentionally minimal — each candidate directory provides:

- `analyze.py` → `analyze(train_instances, manifest=None) -> dict`
- `solution.py` → `solve(instance, analysis=None, manifest=None) -> object`

LLM-generated candidates additionally store `hypothesis.json`, the agent's explicit guess about the hidden rule, the evidence to measure, and the implied solver strategy.

---

## Metrics

Every problem reports `average_normalized_quality`, `optimality_rate`, `feasibility_rate`, and `average_runtime_ms` (external exact baselines additionally report `proved_optimal_rate` and `average_external_runtime_ms`). Quality is normalized to $[0, 1]$ where $1.0$ is optimal and invalid/infeasible outputs score $0$:

| Problem | Normalized quality |
| --- | :---: |
| Coloring | $k_{\mathrm{opt}} / k_{\mathrm{alg}}$ |
| MAXSAT | $\mathrm{sat}_{\mathrm{alg}} / \mathrm{sat}_{\mathrm{opt}}$ |
| MIS | $\lvert \mathrm{IS}_{\mathrm{alg}} \rvert / \lvert \mathrm{IS}_{\mathrm{opt}} \rvert$ |
| MDS | $\lvert \mathrm{DS}_{\mathrm{opt}} \rvert / \lvert \mathrm{DS}_{\mathrm{alg}} \rvert$ |
| Packing LP, MDKP | $\mathrm{obj}_{\mathrm{alg}} / \mathrm{obj}_{\mathrm{opt}}$ |
| TSP | $\mathrm{len}_{\mathrm{opt}} / \mathrm{len}_{\mathrm{alg}}$ |

**Notes.** Exact optima are computed at generation time and stored per instance. OR-Tools is the exact backend for graph problems and the packing pair (`GLOP` for `packing_lp`, CP-SAT for `mdkp`). Gurobi is a *timed industrial baseline only* and never replaces stored-optimum generation. The LLM generator uses structured outputs and the system prompt in [`dasbench/prompts/llm_system_prompt.txt`](dasbench/prompts/llm_system_prompt.txt).

---

## Theory in one paragraph

**Runtime-aware library selection.** For a fixed solver class $\mathcal{C}$, the empirically *fastest sample-consistent* solver $\widehat{c}_S$ generalizes in both correctness and runtime. With probability $\geq 1 - \delta$ over $S \sim D^n$, its deployment runtime satisfies

$$
\mathrm{Run}_D(\widehat{c}_S) \;\le\; \inf_{c \,\in\, \mathcal{C}^{\mathrm{feas}}} \mathrm{Run}_D(c) \;+\; O\!\left( T_{\max} \sqrt{\tfrac{\log \lvert \mathcal{C} \rvert}{n}} \right),
$$

so it approaches the best *correct* distribution-specialized solver in the class, while sample consistency controls deployment error.

**Hint recovery.** For an identifiable hint class with score margin $\gamma > 0$ over $\lvert \mathcal{H} \rvert = N$ candidates,

$$
n \;\ge\; \frac{2}{\gamma^{2}} \log \frac{2N}{\delta}
$$

samples suffice to recover the hidden hint $h^\star$ exactly with probability $\geq 1 - \delta$ — logarithmic in $N$, inverse-quadratic in the margin.

**A concrete instance — hidden SAT backdoors.** Here samples improve *computation* without ever learning correctness: a complete solver is always available, fallback preserves validity, and the recovered backdoor $\widehat{B}$ yields per-instance runtime $O(2^{k}\,\mathrm{poly}(\lvert F \rvert))$ on the deployment distribution. Proofs are in the paper appendix.

---

## Limitations

The one-time synthesis cost only pays off when amortized over enough future instances. Because the solver is specialized to the sampled regime, its advantage can degrade under distribution shift (a perturbation ablation in the paper quantifies this). The method is **complementary to**, not a replacement for, general-purpose solvers — and because the search explores a rich program space, different runs may recover different hints or brittle shortcuts.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@misc{koganti2026distributionawarealgorithmdesignllm,
      title={Distribution-Aware Algorithm Design with LLM Agents}, 
      author={Saharsh Koganti and Priyadarsi Mishra and Pierfrancesco Beneventano and Tomer Galanti},
      year={2026},
      eprint={2605.14141},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.14141}, 
}
```

## License

See the repository for license details.
