from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.common import add_common_arguments, jobs_from_conditions, resolve_sweep_artifact_root, run_sweep
from dasbench.families import available_family_names
from dasbench.problems import available_problem_names
from dasbench.utils import timestamp_token

# Legacy per-problem profile calibrated with local exact-runtime probes and targeted follow-up.
# probes. The goal is not to make every exact baseline land in 3-5 seconds simultaneously;
# that is not realistic for several problems. Instead, these sizes try to place at least one
# competitive exact baseline per problem in the low-single-digit second range while keeping
# dataset-optimum generation tractable enough for a benchmark run.
SECOND_SCALE_PROFILE: dict[str, dict[str, object]] = {
    "coloring": {
        "calibrated_family": "cluster_ring_mix_v1",
        "instance_params": {"num_vertices": 120},
        "anchor_baselines": ["exact", "scip_coloring_exact"],
    },
    "maxsat": {
        "calibrated_family": "latent_backdoor_mixture_v1",
        "instance_params": {"num_variables": 320, "num_clauses": 1280},
        "anchor_baselines": ["rc2_exact", "gurobi_timed"],
        "note": "RC2 hardness varies sharply by seed; this size keeps Gurobi in-range more often than community-style families.",
    },
    "mdkp": {
        "calibrated_family": "latent_class_knapsack_v1",
        "instance_params": {"num_items": 255, "num_resources": 20},
        "anchor_baselines": ["exact", "cbc_mdkp_exact", "highs_mip_exact"],
        "note": "MDKP also shows sharp hardness variance; this setting centers the native CP-SAT exact baseline near the target band on sampled seeds.",
    },
    "mds": {
        "calibrated_family": "geometric_cluster_cover_v1",
        "instance_params": {"num_vertices": 1300},
        "anchor_baselines": ["exact", "highs_mds_mip_exact", "gurobi_timed"],
    },
    "mis": {
        "calibrated_family": "clique_path_mix_v1",
        "instance_params": {"num_vertices": 220},
        "anchor_baselines": ["exact", "kamis_vc_exact", "highs_mis_mip_exact"],
        "note": "220 is the best compromise we found before the HiGHS MIP baseline jumps past the cap.",
    },
    "packing_lp": {
        "calibrated_family": "block_coupled_resource_v1",
        "instance_params": {"num_items": 4500, "num_resources": 130},
        "anchor_baselines": ["exact", "highs_lp_exact"],
        "note": "LP solvers separate sharply here; this keeps the native exact LP solve in-range without making every industrial baseline hit the cap.",
    },
    "tsp": {
        "calibrated_family": "clustered_euclidean_v1",
        "instance_params": {"num_cities": 17},
        "anchor_baselines": ["exact", "cpsat_tsp_exact", "concorde_exact"],
    },
}


def _selected_targets(problem: str | None, family: str | None) -> list[tuple[str, str]]:
    if family and not problem:
        raise ValueError("`--family` requires `--problem`.")
    if problem:
        if problem not in available_problem_names():
            raise ValueError(f"Unknown problem `{problem}`.")
        available = available_family_names(problem)
        assert isinstance(available, list)
        if family is not None:
            if family not in available:
                raise ValueError(f"Unknown family `{family}` for problem `{problem}`.")
            return [(problem, family)]
        return [(problem, family_name) for family_name in available]
    families_by_problem = available_family_names()
    assert isinstance(families_by_problem, dict)
    return [
        (problem_name, family_name)
        for problem_name in sorted(families_by_problem)
        for family_name in families_by_problem[problem_name]
    ]


def build_conditions(
    *,
    train_size: int,
    validation_size: int,
    test_size: int,
    iterations: int,
    beam_width: int,
) -> list[dict[str, object]]:
    return [
        {
            "condition_id": "seconds_scale",
            "train_size": train_size,
            "validation_size": validation_size,
            "test_size": test_size,
            "iterations": iterations,
            "beam_width": beam_width,
            "profile_name": "seconds_scale",
            "instance_params_by_problem": {
                problem_name: dict(profile["instance_params"])
                for problem_name, profile in SECOND_SCALE_PROFILE.items()
            },
            "anchor_baselines_by_problem": {
                problem_name: list(profile.get("anchor_baselines", []))
                for problem_name, profile in SECOND_SCALE_PROFILE.items()
            },
            "profile_notes_by_problem": {
                problem_name: str(profile.get("note", ""))
                for problem_name, profile in SECOND_SCALE_PROFILE.items()
                if profile.get("note")
            },
        }
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the calibrated second-scale dasbench benchmark.")
    add_common_arguments(parser)
    parser.add_argument("--train-size", type=int, default=64)
    parser.add_argument("--validation-size", type=int, default=32)
    parser.add_argument("--test-size", type=int, default=500)
    parser.set_defaults(
        repeats=1,
        gurobi_time_limit_seconds=10.0,
        native_exact_time_limit_seconds=10.0,
        external_time_limit_seconds=10.0,
    )
    return parser


def build_jobs(args: argparse.Namespace, *, sweep_id: str | None = None):
    sweep_id = sweep_id or args.sweep_id or timestamp_token()
    targets = _selected_targets(args.problem, args.family)
    conditions = build_conditions(
        train_size=args.train_size,
        validation_size=args.validation_size,
        test_size=args.test_size,
        iterations=args.iterations,
        beam_width=args.beam_width,
    )
    return jobs_from_conditions(
        sweep_id=sweep_id,
        artifact_root=resolve_sweep_artifact_root(args.output_root, "second_scale_benchmark", sweep_id),
        targets=targets,
        conditions=conditions,
        args=args,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    sweep_id = args.sweep_id or timestamp_token()
    jobs = build_jobs(args, sweep_id=sweep_id)
    summary = run_sweep(
        sweep_id=sweep_id,
        sweep_kind="second_scale_benchmark",
        jobs=jobs,
        output_root=Path(args.output_root),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    return 1 if summary["failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
