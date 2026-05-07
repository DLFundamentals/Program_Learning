from __future__ import annotations

import argparse
import json
from pathlib import Path

from dasbench.integrations.external_exact import EXTERNAL_SOLVER_SPECS
from dasbench.problems import PROBLEMS


BASELINE_METADATA: dict[str, dict[str, object]] = {
    "random_greedy": {
        "display_name": "Random-order greedy",
        "search_name": "random-order greedy graph/MIS heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Greedy constructive heuristic using a randomized vertex order.",
    },
    "largest_degree": {
        "display_name": "Largest-degree greedy coloring",
        "search_name": "largest-degree greedy graph coloring",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Colors vertices greedily after ordering them by descending degree.",
    },
    "smallest_last": {
        "display_name": "Smallest-last greedy coloring",
        "search_name": "smallest-last ordering graph coloring",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Greedy coloring with smallest-last elimination ordering.",
    },
    "dsatur": {
        "display_name": "DSATUR heuristic",
        "search_name": "DSATUR graph coloring heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Classic Brelaz DSATUR-style saturation-degree graph coloring heuristic.",
    },
    "exact": {
        "display_name": "Problem-local exact solver alias",
        "search_name": "problem-local exact solver alias",
        "category": "exact",
        "backend": "problem-dependent",
        "notes": "Alias for the problem-local exact solver; see same_implementation_as for the concrete backend.",
    },
    "cpsat_exact": {
        "display_name": "OR-Tools CP-SAT exact solver",
        "search_name": "OR-Tools CP-SAT exact combinatorial optimization",
        "category": "exact",
        "backend": "OR-Tools CP-SAT",
        "notes": "Problem-local exact formulation solved with OR-Tools CP-SAT.",
    },
    "dsatur_branch_bound_exact": {
        "display_name": "DSATUR branch-and-bound exact coloring",
        "search_name": "DSATUR branch and bound exact graph coloring",
        "category": "exact",
        "backend": "custom Python",
        "notes": "Exact branch-and-bound coloring solver using a DSATUR-style branching rule.",
    },
    "gurobi_timed": {
        "display_name": "Gurobi timed solver",
        "search_name": "Gurobi time-limited MIP/LP formulation",
        "category": "solver-backed",
        "backend": "Gurobi",
        "notes": "Problem-specific Gurobi LP/MIP formulation run under benchmark time and thread limits; may return incumbents rather than proved optima.",
    },
    "highs_coloring_mip_exact": {
        "display_name": "HiGHS coloring MIP",
        "search_name": "HiGHS mixed integer programming graph coloring",
        "category": "exact",
        "backend": "HiGHS",
        "notes": "Graph coloring MIP solved with HiGHS.",
    },
    "scip_coloring_exact": {
        "display_name": "SCIP coloring MIP",
        "search_name": "SCIP mixed integer programming graph coloring",
        "category": "exact",
        "backend": "SCIP",
        "notes": "Graph coloring MIP solved with SCIP.",
    },
    "pysat_coloring_exact": {
        "display_name": "PySAT SAT coloring solver",
        "search_name": "PySAT SAT-based exact graph coloring",
        "category": "exact",
        "backend": "PySAT",
        "notes": "SAT-based exact graph coloring baseline implemented through PySAT.",
    },
    "random": {
        "display_name": "Random baseline",
        "search_name": "random assignment or random tour baseline",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Simple random baseline; semantics depend on the problem (random Boolean assignment for MaxSAT, random tour for TSP).",
    },
    "polarity": {
        "display_name": "Literal-majority polarity assignment",
        "search_name": "majority polarity assignment MaxSAT heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Assigns each variable according to literal-majority polarity statistics in the clauses.",
    },
    "local_search": {
        "display_name": "Greedy flip local search",
        "search_name": "greedy flip local search MaxSAT heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Starts from the polarity assignment and greedily flips variables to improve satisfied-clause count.",
    },
    "rc2_exact": {
        "display_name": "PySAT RC2 (Glucose3)",
        "search_name": "PySAT RC2 MaxSAT solver Glucose3",
        "category": "exact",
        "backend": "PySAT RC2",
        "notes": "Exact MaxSAT baseline using PySAT RC2 with the Glucose3 backend.",
    },
    "rc2_glucose4": {
        "display_name": "PySAT RC2 (Glucose4)",
        "search_name": "PySAT RC2 MaxSAT solver Glucose4",
        "category": "exact",
        "backend": "PySAT RC2",
        "notes": "Exact MaxSAT baseline using PySAT RC2 with the Glucose4 backend.",
    },
    "rc2_minisat22": {
        "display_name": "PySAT RC2 (MiniSat 2.2)",
        "search_name": "PySAT RC2 MaxSAT solver MiniSat 2.2",
        "category": "exact",
        "backend": "PySAT RC2",
        "notes": "Exact MaxSAT baseline using PySAT RC2 with the MiniSat 2.2 backend.",
    },
    "rc2_cadical195": {
        "display_name": "PySAT RC2 (CaDiCaL 1.9.5)",
        "search_name": "PySAT RC2 MaxSAT solver CaDiCaL 1.9.5",
        "category": "exact",
        "backend": "PySAT RC2",
        "notes": "Exact MaxSAT baseline using PySAT RC2 with the CaDiCaL 1.9.5 backend.",
    },
    "open_wbo_exact": {
        "display_name": "Open-WBO",
        "search_name": "Open-WBO MaxSAT solver",
        "category": "exact",
        "backend": "Open-WBO / Hermax wrapper",
        "notes": "External MaxSAT solver accessed through the Hermax integration.",
    },
    "uwrmaxsat_exact": {
        "display_name": "UWrMaxSAT",
        "search_name": "UWrMaxSAT solver",
        "category": "exact",
        "backend": "UWrMaxSAT / Hermax wrapper",
        "notes": "External MaxSAT solver accessed through the Hermax integration.",
    },
    "evalmaxsat_exact": {
        "display_name": "EvalMaxSAT",
        "search_name": "EvalMaxSAT solver",
        "category": "exact",
        "backend": "EvalMaxSAT / Hermax wrapper",
        "notes": "External MaxSAT solver accessed through the Hermax integration.",
    },
    "maxhs_exact": {
        "display_name": "MaxHS",
        "search_name": "MaxHS MaxSAT solver",
        "category": "exact",
        "backend": "MaxHS / Hermax wrapper",
        "notes": "External MaxSAT solver accessed through the Hermax integration.",
    },
    "wmaxcdcl_exact": {
        "display_name": "WMaxCDCL",
        "search_name": "WMaxCDCL MaxSAT solver",
        "category": "exact",
        "backend": "WMaxCDCL / Hermax wrapper",
        "notes": "External MaxSAT solver accessed through the Hermax integration.",
    },
    "value_density_greedy": {
        "display_name": "Value-density greedy",
        "search_name": "value density greedy multidimensional knapsack",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Binary multidimensional knapsack heuristic that ranks items by value density.",
    },
    "redundancy_improved_greedy": {
        "display_name": "Redundancy-improved greedy",
        "search_name": "redundancy improved greedy multidimensional knapsack",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "MDKP greedy heuristic that downweights items consuming already-tight resources.",
    },
    "lp_relax_rounding": {
        "display_name": "LP relaxation + rounding",
        "search_name": "LP relaxation rounding multidimensional knapsack heuristic",
        "category": "heuristic",
        "backend": "custom Python + LP relaxation",
        "notes": "Solves a relaxation and rounds the fractional solution into a feasible binary selection.",
    },
    "highs_mip_exact": {
        "display_name": "HiGHS MDKP MIP",
        "search_name": "HiGHS mixed integer programming multidimensional knapsack",
        "category": "exact",
        "backend": "HiGHS",
        "notes": "MDKP MIP solved with HiGHS.",
    },
    "cbc_mdkp_exact": {
        "display_name": "CBC MDKP MIP",
        "search_name": "CBC mixed integer programming multidimensional knapsack",
        "category": "exact",
        "backend": "CBC",
        "notes": "MDKP MIP solved with COIN-OR CBC.",
    },
    "branch_bound_mdkp_exact": {
        "display_name": "Custom MDKP branch-and-bound",
        "search_name": "branch and bound exact multidimensional knapsack",
        "category": "exact",
        "backend": "custom Python",
        "notes": "Custom branch-and-bound exact solver for multidimensional knapsack.",
    },
    "scip_mdkp_exact": {
        "display_name": "SCIP MDKP MIP",
        "search_name": "SCIP mixed integer programming multidimensional knapsack",
        "category": "exact",
        "backend": "SCIP",
        "notes": "MDKP MIP solved with SCIP.",
    },
    "high_degree_greedy": {
        "display_name": "High-degree greedy dominating set",
        "search_name": "high degree greedy minimum dominating set",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Greedy dominating-set heuristic that prioritizes high-coverage vertices.",
    },
    "marginal_gain_greedy": {
        "display_name": "Marginal-gain greedy dominating set",
        "search_name": "marginal gain greedy minimum dominating set",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Greedy dominating-set heuristic that selects the vertex with the largest uncovered marginal gain.",
    },
    "redundancy_aware": {
        "display_name": "Redundancy-aware dominating set heuristic",
        "search_name": "redundancy aware greedy minimum dominating set",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Greedy dominating-set heuristic with redundancy-aware scoring and pruning.",
    },
    "set_cover_branch_bound_exact": {
        "display_name": "Set-cover branch-and-bound exact MDS",
        "search_name": "set cover branch and bound exact minimum dominating set",
        "category": "exact",
        "backend": "custom Python",
        "notes": "Exact minimum dominating set solver framed as set cover with branch-and-bound pruning.",
    },
    "scip_mip_exact": {
        "display_name": "SCIP MDS MIP",
        "search_name": "SCIP mixed integer programming minimum dominating set",
        "category": "exact",
        "backend": "SCIP",
        "notes": "Minimum dominating set MIP solved with SCIP.",
    },
    "highs_mds_mip_exact": {
        "display_name": "HiGHS MDS MIP",
        "search_name": "HiGHS mixed integer programming minimum dominating set",
        "category": "exact",
        "backend": "HiGHS",
        "notes": "Minimum dominating set MIP solved with HiGHS.",
    },
    "cbc_mds_mip_exact": {
        "display_name": "CBC MDS MIP",
        "search_name": "CBC mixed integer programming minimum dominating set",
        "category": "exact",
        "backend": "CBC",
        "notes": "Minimum dominating set MIP solved with COIN-OR CBC.",
    },
    "min_degree_greedy": {
        "display_name": "Minimum-degree greedy MIS",
        "search_name": "minimum degree greedy maximum independent set",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Greedy independent-set heuristic that repeatedly chooses a low-degree vertex.",
    },
    "ratio_greedy": {
        "display_name": "Ratio-based greedy MIS",
        "search_name": "ratio greedy maximum independent set",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Greedy independent-set heuristic that ranks a vertex by local conflict structure rather than only degree.",
    },
    "local_improve": {
        "display_name": "Local-improvement MIS heuristic",
        "search_name": "local improvement maximum independent set heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Constructive greedy MIS followed by local improvement moves.",
    },
    "clique_branch_bound_exact": {
        "display_name": "Clique branch-and-bound exact MIS",
        "search_name": "maximum clique branch and bound on complement graph for maximum independent set",
        "category": "exact",
        "backend": "custom Python",
        "notes": "Exact MIS solver implemented as maximum clique branch-and-bound on the complement graph with coloring bounds.",
    },
    "kamis_vc_exact": {
        "display_name": "KaMIS exact via vertex cover",
        "search_name": "KaMIS exact vertex cover solver maximum independent set",
        "category": "exact",
        "backend": "KaMIS",
        "notes": "External exact MIS baseline via a KaMIS vertex-cover style solver integration.",
    },
    "scip_mis_exact": {
        "display_name": "SCIP MIS MIP",
        "search_name": "SCIP mixed integer programming maximum independent set",
        "category": "exact",
        "backend": "SCIP",
        "notes": "Maximum independent set MIP solved with SCIP.",
    },
    "highs_mis_mip_exact": {
        "display_name": "HiGHS MIS MIP",
        "search_name": "HiGHS mixed integer programming maximum independent set",
        "category": "exact",
        "backend": "HiGHS",
        "notes": "Maximum independent set MIP solved with HiGHS.",
    },
    "uniform_fraction": {
        "display_name": "Uniform-fraction LP heuristic",
        "search_name": "uniform fraction continuous packing heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Assigns the same fractional value to all variables at the largest globally feasible level.",
    },
    "density_fractional": {
        "display_name": "Density-based fractional heuristic",
        "search_name": "value density fractional packing heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Fractional packing heuristic guided by profit or value density.",
    },
    "glop_simplex_exact": {
        "display_name": "OR-Tools GLOP simplex",
        "search_name": "OR-Tools GLOP simplex linear programming",
        "category": "exact",
        "backend": "OR-Tools GLOP",
        "notes": "Exact LP solve with the OR-Tools GLOP simplex backend.",
    },
    "highs_lp_exact": {
        "display_name": "HiGHS LP",
        "search_name": "HiGHS linear programming solver",
        "category": "exact",
        "backend": "HiGHS",
        "notes": "Continuous packing LP solved with HiGHS.",
    },
    "clp_lp_exact": {
        "display_name": "CLP LP",
        "search_name": "COIN-OR CLP linear programming solver",
        "category": "exact",
        "backend": "CLP",
        "notes": "Continuous packing LP solved with COIN-OR CLP.",
    },
    "highs_ipm_lp_exact": {
        "display_name": "HiGHS LP interior point",
        "search_name": "HiGHS interior point linear programming solver",
        "category": "exact",
        "backend": "HiGHS",
        "notes": "Continuous packing LP solved with the HiGHS interior-point method.",
    },
    "scip_lp_exact": {
        "display_name": "SCIP LP",
        "search_name": "SCIP linear programming solver",
        "category": "exact",
        "backend": "SCIP",
        "notes": "Continuous packing LP solved with SCIP.",
    },
    "nearest_neighbor": {
        "display_name": "Nearest-neighbor TSP",
        "search_name": "nearest neighbor TSP heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Constructs a tour by repeatedly visiting the nearest unvisited city.",
    },
    "nearest_insertion": {
        "display_name": "Nearest-insertion TSP",
        "search_name": "nearest insertion TSP heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Builds a tour by repeatedly inserting the nearest outsider into the current subtour.",
    },
    "farthest_insertion": {
        "display_name": "Farthest-insertion TSP",
        "search_name": "farthest insertion TSP heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Builds a tour by repeatedly inserting the farthest outsider into the current subtour.",
    },
    "two_opt_nearest_neighbor": {
        "display_name": "Nearest-neighbor + 2-opt",
        "search_name": "nearest neighbor plus 2-opt TSP heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Nearest-neighbor initialization followed by 2-opt local search.",
    },
    "two_opt_farthest_insertion": {
        "display_name": "Farthest-insertion + 2-opt",
        "search_name": "farthest insertion plus 2-opt TSP heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Farthest-insertion initialization followed by 2-opt local search.",
    },
    "multi_start_two_opt": {
        "display_name": "Multi-start 2-opt",
        "search_name": "multi-start 2-opt TSP heuristic",
        "category": "heuristic",
        "backend": "custom Python",
        "notes": "Runs 2-opt from several seeded starting tours and keeps the best result.",
    },
    "held_karp_exact": {
        "display_name": "Held-Karp dynamic programming",
        "search_name": "Held-Karp dynamic programming exact traveling salesman",
        "category": "exact",
        "backend": "custom Python",
        "notes": "Exact TSP solver using Held-Karp subset dynamic programming.",
    },
    "concorde_exact": {
        "display_name": "Concorde",
        "search_name": "Concorde TSP solver",
        "category": "exact",
        "backend": "Concorde",
        "notes": "External exact TSP baseline using the Concorde solver.",
    },
    "cpsat_tsp_exact": {
        "display_name": "OR-Tools CP-SAT TSP",
        "search_name": "OR-Tools CP-SAT exact TSP formulation",
        "category": "exact",
        "backend": "OR-Tools CP-SAT",
        "notes": "Exact TSP formulation solved with OR-Tools CP-SAT.",
    },
    "scip_tsp_mtz_exact": {
        "display_name": "SCIP MTZ TSP MIP",
        "search_name": "SCIP Miller Tucker Zemlin TSP formulation",
        "category": "exact",
        "backend": "SCIP",
        "notes": "Traveling salesperson MIP with MTZ subtour constraints solved with SCIP.",
    },
    "cbc_tsp_mtz_exact": {
        "display_name": "CBC MTZ TSP MIP",
        "search_name": "CBC Miller Tucker Zemlin TSP formulation",
        "category": "exact",
        "backend": "CBC",
        "notes": "Traveling salesperson MIP with MTZ subtour constraints solved with CBC.",
    },
}


PROBLEM_NOTES = {
    "coloring": "Graph coloring; normalized quality is optimum_colors / returned_colors.",
    "maxsat": "Unweighted MaxSAT / MAX-3SAT; normalized quality is satisfied_clauses / optimum_satisfied_clauses.",
    "mdkp": "Binary multidimensional knapsack; normalized quality is returned_value / optimum_value for feasible selections.",
    "mds": "Minimum dominating set; normalized quality is optimum_set_size / returned_set_size.",
    "mis": "Maximum independent set; normalized quality is returned_set_size / optimum_set_size.",
    "packing_lp": "Continuous bounded packing LP; normalized quality is returned_objective / optimum_objective.",
    "tsp": "Euclidean traveling salesperson; normalized quality is optimum_tour_length / returned_tour_length.",
}


SAME_IMPLEMENTATION_AS = {
    ("coloring", "exact"): "cpsat_exact",
    ("mdkp", "exact"): "cpsat_exact",
    ("mds", "exact"): "cpsat_exact",
    ("mis", "exact"): "cpsat_exact",
    ("packing_lp", "exact"): "glop_simplex_exact",
    ("tsp", "exact"): "held_karp_exact",
}


PROBLEM_SPECIFIC_OVERRIDES: dict[tuple[str, str], dict[str, object]] = {
    ("coloring", "random_greedy"): {
        "display_name": "Random-order greedy coloring",
        "search_name": "random-order greedy graph coloring",
        "notes": "Greedy graph coloring baseline using a randomized vertex order.",
    },
    ("mis", "random_greedy"): {
        "display_name": "Random greedy MIS",
        "search_name": "random greedy maximum independent set",
        "notes": "Greedy independent-set heuristic using a randomized vertex order.",
    },
    ("maxsat", "random"): {
        "display_name": "Random Boolean assignment",
        "search_name": "random assignment MaxSAT baseline",
        "notes": "Assigns each Boolean variable randomly.",
    },
    ("tsp", "random"): {
        "display_name": "Random tour",
        "search_name": "random tour TSP baseline",
        "notes": "Builds a uniformly random tour permutation.",
    },
    ("coloring", "exact"): {
        "display_name": "OR-Tools CP-SAT exact coloring alias",
        "search_name": "OR-Tools CP-SAT exact graph coloring",
        "backend": "OR-Tools CP-SAT",
        "notes": "Alias for the OR-Tools CP-SAT exact graph coloring formulation.",
    },
    ("mdkp", "exact"): {
        "display_name": "OR-Tools CP-SAT exact MDKP alias",
        "search_name": "OR-Tools CP-SAT exact multidimensional knapsack",
        "backend": "OR-Tools CP-SAT",
        "notes": "Alias for the OR-Tools CP-SAT exact multidimensional knapsack formulation.",
    },
    ("mds", "exact"): {
        "display_name": "OR-Tools CP-SAT exact MDS alias",
        "search_name": "OR-Tools CP-SAT exact minimum dominating set",
        "backend": "OR-Tools CP-SAT",
        "notes": "Alias for the OR-Tools CP-SAT exact minimum dominating set formulation.",
    },
    ("mis", "exact"): {
        "display_name": "OR-Tools CP-SAT exact MIS alias",
        "search_name": "OR-Tools CP-SAT exact maximum independent set",
        "backend": "OR-Tools CP-SAT",
        "notes": "Alias for the OR-Tools CP-SAT exact maximum independent set formulation.",
    },
    ("packing_lp", "exact"): {
        "display_name": "OR-Tools GLOP exact LP alias",
        "search_name": "OR-Tools GLOP simplex linear programming",
        "backend": "OR-Tools GLOP",
        "notes": "Alias for the OR-Tools GLOP simplex LP baseline.",
    },
    ("tsp", "exact"): {
        "display_name": "Held-Karp exact TSP alias",
        "search_name": "Held-Karp dynamic programming exact traveling salesman",
        "backend": "custom Python",
        "notes": "Alias for the Held-Karp dynamic-programming exact TSP solver.",
    },
}


def _problem_baseline_ids(problem_name: str) -> list[str]:
    problem = PROBLEMS[problem_name]
    local_ids = sorted(problem.baseline_registry().keys())
    external_ids = sorted(
        spec.baseline_name for spec in EXTERNAL_SOLVER_SPECS.values() if spec.problem == problem_name
    )
    return sorted(set(local_ids + external_ids + ["gurobi_timed"]))


def _source_type(problem_name: str, baseline_id: str) -> str:
    if baseline_id == "gurobi_timed":
        return "gurobi"
    if baseline_id in PROBLEMS[problem_name].baseline_registry():
        return "problem_local"
    return "external_exact"


def build_catalog() -> dict[str, object]:
    missing = set()
    problems: dict[str, object] = {}
    for problem_name in sorted(PROBLEMS):
        baselines = []
        for baseline_id in _problem_baseline_ids(problem_name):
            metadata = BASELINE_METADATA.get(baseline_id)
            if metadata is None:
                missing.add(baseline_id)
                continue
            metadata = {
                **metadata,
                **PROBLEM_SPECIFIC_OVERRIDES.get((problem_name, baseline_id), {}),
            }
            entry = {
                "repo_id": baseline_id,
                "problem": problem_name,
                "source_type": _source_type(problem_name, baseline_id),
                **metadata,
            }
            if baseline_id == "gurobi_timed":
                entry["notes"] = f"{entry['notes']} Uses a problem-specific model selected in dasbench.integrations.gurobi_baseline."
            if entry["source_type"] == "external_exact":
                spec = EXTERNAL_SOLVER_SPECS[baseline_id]
                entry["external_setup_required"] = spec.external_setup_required
                entry["aliases"] = list(spec.aliases)
            alias_target = SAME_IMPLEMENTATION_AS.get((problem_name, baseline_id))
            if alias_target is not None:
                entry["same_implementation_as"] = alias_target
            baselines.append(entry)
        problems[problem_name] = {
            "problem_notes": PROBLEM_NOTES[problem_name],
            "baselines": baselines,
        }
    if missing:
        raise ValueError(f"Missing metadata for baselines: {sorted(missing)}")
    return {
        "generated_by": "scripts/export_baseline_catalog.py",
        "notes": "Catalog of baseline solver ids used across dasbench problems, with search-friendly names and short implementation notes.",
        "problems": problems,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a JSON catalog of all baseline solvers and notes.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/baseline_catalog.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    catalog = build_catalog()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(catalog, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
