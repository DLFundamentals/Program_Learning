from __future__ import annotations

import argparse
import unittest

from dasbench.cli import build_parser
from dasbench.families import available_family_names
from dasbench.suites import SuiteTarget, _target_command, benchmark_targets


class BenchmarkScopeTests(unittest.TestCase):
    def _args(self, **overrides) -> argparse.Namespace:
        defaults = {
            "problem": "maxsat",
            "family": "last_clause_signal_v1",
            "all_families": False,
            "dataset_dir": None,
            "output_dir": None,
            "run_output_dir": None,
            "report_output_dir": None,
            "dataset_id": None,
            "run_id": None,
            "generator": "template",
            "mode": "single",
            "iterations": 1,
            "beam_width": 1,
            "candidate_width": None,
            "repeats": 1,
            "include_train": False,
            "force_regenerate": False,
            "instance_param": [],
            "family_param": [],
            "train_size": 4,
            "validation_size": 2,
            "test_size": 2,
            "family_seed": 17,
            "train_seed": 101,
            "validation_seed": 202,
            "test_seed": 303,
            "compute_optima": True,
            "gurobi_baseline_enabled": True,
            "gurobi_time_limit_seconds": 60.0,
            "gurobi_threads": 1,
            "external_exact_baselines": "auto",
            "external_time_limit_seconds": 60.0,
            "external_threads": 1,
            "external_solver_config": None,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_single_benchmark_target(self) -> None:
        args = self._args()
        self.assertEqual(benchmark_targets(args), [("maxsat", "last_clause_signal_v1")])

    def test_all_families_for_one_problem(self) -> None:
        args = self._args(problem="mis", family=None, all_families=True)
        expected = [("mis", family) for family in available_family_names("mis")]
        self.assertEqual(benchmark_targets(args), expected)

    def test_all_families_for_all_problems(self) -> None:
        args = self._args(problem=None, family=None, all_families=True)
        families_by_problem = available_family_names()
        assert isinstance(families_by_problem, dict)
        expected = [
            (problem, family)
            for problem in sorted(families_by_problem)
            for family in families_by_problem[problem]
        ]
        self.assertEqual(benchmark_targets(args), expected)

    def test_all_families_rejects_single_target_only_paths(self) -> None:
        args = self._args(family=None, all_families=True, dataset_dir="artifacts/datasets/foo")
        with self.assertRaises(ValueError):
            benchmark_targets(args)

    def test_suite_command_forwards_candidate_width(self) -> None:
        args = self._args(candidate_width=5)
        command = _target_command(args, target=SuiteTarget("maxsat", "last_clause_signal_v1"), suite_id="candidate-test")
        self.assertIn("--candidate-width", command)
        self.assertIn("5", command)

    def test_cli_accepts_candidate_width_for_run_agent_and_benchmark(self) -> None:
        parser = build_parser()
        run_args = parser.parse_args(["run-agent", "--dataset-dir", "dataset", "--candidate-width", "7"])
        self.assertEqual(run_args.candidate_width, 7)
        benchmark_args = parser.parse_args(
            [
                "benchmark",
                "--problem",
                "maxsat",
                "--family",
                "last_clause_signal_v1",
                "--candidate-width",
                "9",
            ]
        )
        self.assertEqual(benchmark_args.candidate_width, 9)


if __name__ == "__main__":
    unittest.main()
