<h1 align="center">Distribution-Aware Algorithm Design with LLM Agents</h1>

<p align="center">
  <em>Learn deployment structure. Compile faster solvers.</em>
</p>

<p align="center">
  <a href="#-install"><img alt="Python" src="https://img.shields.io/badge/python-3.x-blue.svg"></a>
  <a href="https://github.com/DLFundamentals/Program_Learning"><img alt="Status" src="https://img.shields.io/badge/status-research%20preview-orange.svg"></a>
  <a href="#-citation"><img alt="Paper" src="https://img.shields.io/badge/paper-arXiv-8A2BE2.svg"></a>
  <a href="#license"><img alt="License" src="https://img.shields.io/badge/license-see%20repo-lightgrey.svg"></a>
</p>

<p align="center">
  Saharsh Koganti<sup>1</sup> · Priyadarsi Mishra<sup>1</sup> · Pierfrancesco Beneventano<sup>2</sup> · Tomer Galanti<sup>1</sup><br>
  <sub><sup>1</sup>Texas A&amp;M University &nbsp;&nbsp; <sup>2</sup>Massachusetts Institute of Technology</sub>
</p>

---

## Overview

Hard optimization problems are rarely solved once in isolation. They run again and again inside routers, schedulers, compilers, allocation systems, planning pipelines, and online services — and in deployment, these systems never face arbitrary worst-case inputs. They face a recurring **distribution** of instances.

That distribution tends to carry reusable structure: latent geometry, planted assignments, recurring bottlenecks, stable hubs, hidden backdoors, active constraints, repeated decomposition patterns. Worst-case analysis throws all of it away.

This repository studies **distribution-aware algorithm design** — the question of whether, given only samples from an unknown deployment distribution, an agent can synthesize executable solver code that runs fast on future instances while preserving high solution quality.

The central object is a **solver hint**: a compact piece of distribution-specific structure, inferred from samples and compiled into a specialized solver.

$$
\underbrace{S \sim D^n}_{\text{samples}}
\xrightarrow{\text{ learn }}
\underbrace{\widehat{h}_S}_{\text{hint}}
\xrightarrow{\text{ compile }}
\underbrace{\widehat{c}_S = \mathrm{Comp}(\widehat{h}_S)}_{\text{deployed solver}}
$$

The samples are never used to memorize solutions to the instances we've seen. They're used to discover what makes the *next* instance from the same source easier to solve.

`dasbench` is the benchmark and synthesis framework for this setting. It tests whether an LLM code agent can travel the full path — samples → hypotheses → analysis code → deployment solvers — on hard, structured combinatorial problems.

---

## Why it matters

Classical algorithm design usually asks for a solver that works well in the worst case. Average-case complexity assumes the distribution is known analytically. Real deployments are different: the distribution is usually unknown, but samples are abundant.

This work studies that middle ground.

<div align="center">

<table>
<thead>
<tr><th align="left">Setting</th><th align="left">Information about the distribution</th><th align="left">What is learned</th></tr>
</thead>
<tbody>
<tr><td align="left">Worst-case design</td><td align="left">None</td><td align="left">Nothing distribution-specific</td></tr>
<tr><td align="left">Average-case complexity</td><td align="left">Distribution specified analytically</td><td align="left">Usually no learned solver artifact</td></tr>
<tr><td align="left"><strong>This work</strong></td><td align="left"><strong>Samples from deployment</strong></td><td align="left"><strong>Hint → specialized solver code</strong></td></tr>
</tbody>
</table>

</div>

The learned component is not trusted blindly. Correctness is protected by verification, repair, and fallback. This lets the synthesized code focus on finding the shortcut: the hidden backdoor, separator, bottleneck, hub structure, geometric template, or active-resource pattern that makes the distribution easier than the ambient worst case.

---

## Headline results

Across **21 structured combinatorial-optimization distributions** spanning **7 problem classes**, the synthesized solvers achieve **mean normalized quality 0.971** while running substantially faster than classical, solver-backed, and one-shot code-generation baselines.

<div align="center">

<table>
<thead>
<tr><th align="left">Comparator</th><th>Quality lift Δ</th><th>Speedup</th></tr>
</thead>
<tbody>
<tr><td align="left">Fast high-quality heuristic</td><td><strong>+0.109</strong></td><td><strong>564.9×</strong></td></tr>
<tr><td align="left">Gurobi, 10s, 1 thread</td><td>—</td><td><strong>345.1×</strong></td></tr>
<tr><td align="left">Time-limited exact backend</td><td>—</td><td><strong>16.9×</strong></td></tr>
<tr><td align="left">One-shot Codex</td><td>−0.016 <em>≈ tie</em></td><td><strong>4.5×</strong></td></tr>
<tr><td align="left">One-shot Claude Code</td><td>+0.085</td><td><strong>17.4×</strong></td></tr>
<tr><td align="left">Best-of-5 open model, Gemma 4</td><td>+0.145</td><td>2.4×</td></tr>
</tbody>
</table>

</div>

No single baseline is both faster and higher quality across the full suite. The method improves the average quality–runtime frontier: it often changes the effective computation from broad search or generic optimization into a distribution-specialized procedure.

---

## How it works

For each candidate solver, the agent produces three artifacts:

1. **Hypothesis** — a structured guess about the hidden distributional rule.
2. **Analysis program** — code that runs once on public training samples and extracts a reusable hint.
3. **Deployment solver** — code that uses the hint to solve new instances from the same distribution.

The agent sees only the instance format, the scoring rule, and samples. It does not see the family identity, planted rules, optimum solutions, or optimum objective values. Candidates are evaluated on public splits, ranked by quality and runtime, refined across rounds, and the best candidate is deployed.

```mermaid
flowchart LR
    S["public samples<br/>S_pub"]:::data --> H["① Hypothesis<br/>H_c"]:::hyp
    H --> A["② Analysis<br/>A_c"]:::ana
    A --> SOL["③ Solver<br/>s_c"]:::sol
    A -- "execute on train" --> HINT(["hint a_c"]):::hint
    HINT -- "condition" --> SOL
    SOL --> EVAL["evaluate on<br/>public splits"]:::eval
    EVAL --> RANK["rank by<br/>(Q, O, −T)"]:::eval
    RANK --> DEPLOY["deploy ĉ_S"]:::deploy
    RANK -. "refine / fork / replace<br/>push runtime / push quality" .-> H

    classDef data fill:#f4f4f4,stroke:#9aa,color:#333;
    classDef hyp fill:#f7f2ff,stroke:#6e48aa,color:#4a2e86;
    classDef ana fill:#f0f7ff,stroke:#2b67b0,color:#2b67b0;
    classDef sol fill:#f1faf5,stroke:#2c8060,color:#2c8060;
    classDef hint fill:#dccbf2,stroke:#6d4fb0,color:#4a2e86;
    classDef eval fill:#f4f4f4,stroke:#9aa,color:#333;
    classDef deploy fill:#efe7f9,stroke:#6d4fb0,color:#4a2e86;
```

---

## What the agent compiles

The synthesized solvers are not merely faster implementations of the same algorithms. They often discover a different computation adapted to the distribution:

<div align="center">

<table>
<thead>
<tr><th align="left">Problem structure</th><th>Generic exact search</th><th align="left">Generated solver behavior</th></tr>
</thead>
<tbody>
<tr><td align="left">MAXSAT with latent Boolean rules</td><td><em>O</em><sup>*</sup>(2<sup>v</sup>)</td><td align="left">Seeded assignment plus bounded local repair</td></tr>
<tr><td align="left">Coloring with planted palettes</td><td><em>O</em><sup>*</sup>(κ<sup>n</sup>)</td><td align="left">Template recovery plus DSATUR-style recoloring</td></tr>
<tr><td align="left">MIS with motif structure</td><td><em>O</em><sup>*</sup>(2<sup>n</sup>)</td><td align="left">Greedy decomposition plus tiny residual enumeration</td></tr>
<tr><td align="left">MDS with coverage kernels</td><td><em>O</em><sup>*</sup>(2<sup>n</sup>)</td><td align="left">Hub/gateway cover plus bounded pruning</td></tr>
<tr><td align="left">MDKP with recurring bottlenecks</td><td><em>O</em><sup>*</sup>(2<sup>N</sup>)</td><td align="left">Surrogate prices, density sorting, and repair</td></tr>
<tr><td align="left">Packing LP with recurring active constraints</td><td>poly(<em>N</em>, <em>m</em>)</td><td align="left">Infer active/binding resources and use specialized pricing rules</td></tr>
<tr><td align="left">TSP with latent geometry</td><td><em>O</em>(<em>n</em><sup>2</sup> 2<sup>n</sup>)</td><td align="left">Structured construction plus bounded 2-opt</td></tr>
</tbody>
</table>

</div>

---

## External test — PACE 2025 Dominating Set

On the released **private** instances (large sparse graphs, up to ~4.2M vertices), the synthesized solver is the only method that is **both fully valid and fast**: valid on all 100 graphs and ~two orders of magnitude faster than the released competition solvers, for only ~3% larger sets. No baseline dominates it on the quality–runtime frontier.

<div align="center">

<table>
<thead>
<tr>
<th align="left">Solver</th><th>Valid</th><th>Avg. size ↓</th><th>Size vs. ours</th><th>Time (s) ↓</th><th>Speedup</th><th>Quality wins vs. ours</th>
</tr>
</thead>
<tbody>
<tr style="background:#efe7f9;">
<td align="left"><strong>Our agent (GPT-5.2)</strong></td><td>100 / 100</td><td>231,595</td><td>1.00×</td><td><strong>2.89</strong></td><td><strong>1.0×</strong></td><td>—</td>
</tr>
<tr style="background:#f6f2fc;">
<td align="left"><strong>Our agent (Gemma&nbsp;4)</strong></td><td>100 / 100</td><td>231,667</td><td>1.00×</td><td>6.03</td><td>2.1×</td><td>—</td>
</tr>
<tr><td align="left">AEG Heidelberg</td><td>100 / 100</td><td>224,086</td><td>1.034×</td><td>350.14</td><td>121.0×</td><td>99 / 100</td></tr>
<tr><td align="left">Fontan–Verger</td><td>100 / 100</td><td>224,107</td><td>1.033×</td><td>286.24</td><td>98.9×</td><td>100 / 100</td></tr>
<tr><td align="left">Root</td><td>100 / 100</td><td>224,108</td><td>1.033×</td><td>360.42</td><td>124.5×</td><td>100 / 100</td></tr>
<tr><td align="left">Shadoks</td><td>100 / 100</td><td>224,306</td><td>1.032×</td><td>316.07</td><td>109.2×</td><td>100 / 100</td></tr>
<tr><td align="left">Greeduce</td><td>100 / 100</td><td>224,699</td><td>1.031×</td><td>300.86</td><td>104.0×</td><td>91 / 100</td></tr>
<tr><td align="left">Swats <sup>*</sup></td><td>75 / 100</td><td>210,237</td><td>1.028×</td><td>218.11</td><td>75.4×</td><td>75 / 75</td></tr>
</tbody>
</table>

</div>

<sub><strong>Size vs. ours &gt; 1</strong> means the PACE solver returns a smaller dominating set; <strong>Speedup</strong> is how much faster ours runs. <sup>*</sup>Swats is valid on only 75/100 instances, so its size, speedup, and wins are computed on that matched subset. Exact-style baselines and Gurobi time out at the 360&nbsp;s cap; the learned ML baselines cannot run at this scale.</sub>

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

The **template generator** is fully local and needs no API key — the recommended smoke-test path.

---

## 🚀 Quickstart

**Generate a dataset** (exact optima are stored with each instance):

```bash
python main.py generate \
  --problem mis \
  --family motif_bridge_mixture_v1 \
  --dataset-id smoke_mis \
  --instance-param num_vertices=18 \
  --train-size 32 --validation-size 16 --test-size 16
```

**Run baselines plus synthesis** on that dataset:

```bash
python main.py run-agent \
  --dataset-dir artifacts/datasets/mis/motif_bridge_mixture_v1/smoke_mis \
  --generator template \
  --mode beam --iterations 3 --beam-width 3
```

**Run the full flow end to end** for one family:

```bash
python main.py benchmark \
  --problem mds --family gateway_overlap_cover_v1 \
  --generator template \
  --train-size 64 --validation-size 32 --test-size 32
```

**Or sweep every family** for a problem, or across all problems (parallel by default):

```bash
python main.py benchmark --problem maxsat --all-families
python main.py benchmark --all-families --max-parallel 4
```

> Paper-facing sweeps and the PACE 2025 diagnostic live in [`benchmarks/README.md`](benchmarks/README.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

---

## 📁 Repository layout

```
Program_Learning/
├── dasbench/            # core framework: problems, families, baselines, exact solvers, synthesis loop
│   └── prompts/         # LLM system prompt used by the generator
├── benchmarks/          # paper-facing sweep suites and ablations
├── scripts/             # helper scripts
├── tests/               # test suite
├── main.py              # CLI entry point: generate / run-agent / report / benchmark
├── REPRODUCIBILITY.md   # full reproduction notes incl. PACE 2025 diagnostic
├── pyproject.toml
└── .env.example
```

Generated datasets, candidates, and reports are written under `artifacts/` and are intentionally **not** committed — regenerate them with the commands above.

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
