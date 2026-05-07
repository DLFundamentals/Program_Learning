from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

from dasbench.cli import cmd_report, cmd_run_agent
from dasbench.data import BenchmarkSpec, generate_dataset
from dasbench.eval.evaluator import evaluate_solver
from dasbench.integrations.external_exact import (
    ExternalExactConfig,
    build_external_exact_solvers,
    discover_external_exact_baselines,
    parse_concorde_solution,
    parse_highs_solution_file,
    parse_kamis_output,
    parse_open_wbo_output,
    parse_scip_solution_file,
    serialize_coloring_lp,
    serialize_maxsat_wcnf,
    serialize_mds_lp,
    serialize_metis_graph,
    serialize_packing_lp,
    serialize_tsplib_explicit,
)
from dasbench.problems import get_problem_definition


def _write_script(root: Path, name: str, body: str) -> str:
    path = root / name
    path.write_text("#!/usr/bin/env python3\n" + textwrap.dedent(body), encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return str(path)


class ExternalExactBaselineTests(unittest.TestCase):
    def test_auto_discovers_python_native_backends(self) -> None:
        def fake_find_spec(name: str) -> object | None:
            if name in {"highspy", "pyscipopt"}:
                return object()
            return None

        with patch("dasbench.integrations.external_exact.importlib.util.find_spec", side_effect=fake_find_spec):
            packing_discovery = discover_external_exact_baselines("packing_lp", ExternalExactConfig(mode="auto"))
            packing_records = {
                str(record["baseline_name"]): record
                for record in packing_discovery["solvers"]
            }
            self.assertEqual(packing_records["highs_lp_exact"]["backend"], "python")
            self.assertEqual(packing_records["scip_lp_exact"]["backend"], "python")
            self.assertTrue(packing_records["highs_lp_exact"]["python_available"])
            self.assertTrue(packing_records["scip_lp_exact"]["python_available"])

            mds_discovery = discover_external_exact_baselines("mds", ExternalExactConfig(mode="auto"))
            self.assertEqual(mds_discovery["solvers"][0]["backend"], "python")

    def test_configured_binary_takes_cli_backend_over_native_package(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-external-cli-priority-"))
        binary = _write_script(root, "fake_highs.py", "print('fake')\n")

        with patch("dasbench.integrations.external_exact.importlib.util.find_spec", return_value=object()):
            discovery = discover_external_exact_baselines(
                "packing_lp",
                ExternalExactConfig(mode="auto", solver_paths={"highs": binary}),
            )
        records = {
            str(record["baseline_name"]): record
            for record in discovery["solvers"]
        }
        self.assertEqual(records["highs_lp_exact"]["backend"], "cli")
        self.assertEqual(records["highs_lp_exact"]["binary_path"], binary)

    def test_maxsat_auto_discovers_hermax_python_backends(self) -> None:
        discovery = discover_external_exact_baselines("maxsat", ExternalExactConfig(mode="auto"))
        records = {
            str(record["baseline_name"]): record
            for record in discovery["solvers"]
        }
        for baseline_name in ["open_wbo_exact", "uwrmaxsat_exact", "evalmaxsat_exact", "wmaxcdcl_exact"]:
            self.assertEqual(records[baseline_name]["backend"], "python")
            self.assertTrue(records[baseline_name]["python_available"])
        self.assertFalse(records["maxhs_exact"]["enabled"])

    def test_hermax_maxsat_baselines_solve_toy_instance(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-hermax-toy-"))
        config = ExternalExactConfig(mode="auto", time_limit_seconds=5.0, threads=1)
        solvers, _ = build_external_exact_solvers("maxsat", config, artifact_dir=root)
        instance = {
            "id": "toy-hermax-maxsat",
            "num_variables": 3,
            "clauses": [[1, 2, 3], [-1, -2, 3], [1, -2, -3]],
        }
        exact = get_problem_definition("maxsat").exact_solver(instance)
        for baseline_name in ["open_wbo_exact", "uwrmaxsat_exact", "evalmaxsat_exact", "wmaxcdcl_exact"]:
            with self.subTest(baseline=baseline_name):
                self.assertIn(baseline_name, solvers)
                summary = evaluate_solver(
                    "maxsat",
                    baseline_name,
                    solvers[baseline_name],
                    [{**instance, "optimum_objective": exact.objective_value}],
                    split="validation",
                )
                self.assertEqual(summary["average_normalized_quality"], 1.0)
                self.assertEqual(summary["optimality_rate"], 1.0)
                self.assertEqual(summary["proved_optimal_rate"], 1.0)

    def test_serializers_and_parsers_cover_expected_formats(self) -> None:
        maxsat_instance = {
            "id": "toy-maxsat",
            "num_variables": 3,
            "clauses": [[1, 2, 3]],
        }
        self.assertIn("p wcnf 3 1 2", serialize_maxsat_wcnf(maxsat_instance))
        solution, status, proved, cost = parse_open_wbo_output(
            "s OPTIMUM FOUND\no 0\nv 1 -2 3 0\n",
            num_variables=3,
        )
        self.assertEqual(solution, [True, False, True])
        self.assertEqual(status, "OPTIMUM FOUND")
        self.assertTrue(proved)
        self.assertEqual(cost, 0.0)

        graph_instance = {
            "id": "toy-graph",
            "num_vertices": 3,
            "edges": [[0, 1], [1, 2]],
        }
        self.assertIn("3 2", serialize_metis_graph(graph_instance))
        mis, _, proved = parse_kamis_output("s OPTIMUM FOUND\nv 1 3\n", num_vertices=3)
        self.assertEqual(mis, [0, 2])
        self.assertTrue(proved)
        self.assertIn("dom_0", serialize_mds_lp(graph_instance))
        self.assertIn("assign_0", serialize_coloring_lp(graph_instance))
        values, objective, _, _ = parse_scip_solution_file("objective value: 1\nx_0 1\n")
        self.assertEqual(values["x_0"], 1.0)
        self.assertEqual(objective, 1.0)
        packing_instance = {
            "id": "toy-packing",
            "num_items": 2,
            "num_resources": 1,
            "values": [2, 1],
            "weights": [[1], [1]],
            "capacities": [1],
        }
        self.assertIn("cap_0", serialize_packing_lp(packing_instance))
        self.assertIn("Binary", serialize_packing_lp(packing_instance, binary=True))
        values, objective, status, proved, _, _ = parse_highs_solution_file(
            "Model status: Optimal\nObjective value: 2\nx_0 1\nx_1 0\n"
        )
        self.assertEqual(values["x_0"], 1.0)
        self.assertEqual(objective, 2.0)
        self.assertEqual(status, "Optimal")
        self.assertTrue(proved)

        tsp_instance = {
            "id": "toy-tsp",
            "num_cities": 4,
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
        self.assertIn("EDGE_WEIGHT_FORMAT: FULL_MATRIX", serialize_tsplib_explicit(tsp_instance))
        self.assertEqual(parse_concorde_solution("4\n0 1 2 3\n", num_cities=4), [0, 1, 2, 3])

    def test_fake_external_solvers_return_optimal_solutions(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-external-fake-"))
        open_wbo = _write_script(
            root,
            "fake_open_wbo.py",
            """
            import re, sys
            text = open(sys.argv[-1], encoding='utf-8').read()
            n = int(re.search(r'p wcnf (\\d+)', text).group(1))
            print('s OPTIMUM FOUND')
            print('o 0')
            print('v ' + ' '.join(str(i) for i in range(1, n + 1)) + ' 0')
            """,
        )
        kamis = _write_script(
            root,
            "fake_kamis.py",
            """
            print('s OPTIMUM FOUND')
            print('v 1 3')
            """,
        )
        scip = _write_script(
            root,
            "fake_scip.py",
            """
            import sys
            args = sys.argv[1:]
            read_cmd = next(arg for i, arg in enumerate(args) if args[i - 1] == '-c' and arg.startswith('read '))
            sol_cmd = next(arg for i, arg in enumerate(args) if args[i - 1] == '-c' and arg.startswith('write solution '))
            model_path = read_cmd.split(' ', 1)[1]
            sol_path = sol_cmd.split(' ', 2)[2]
            model = open(model_path, encoding='utf-8').read()
            with open(sol_path, 'w', encoding='utf-8') as handle:
                handle.write('objective value: 1\\n')
                if 'x_0_0' in model:
                    handle.write('x_0_0 1\\nx_1_1 1\\nx_2_2 1\\n')
                else:
                    handle.write('x_0 1\\n')
            print('optimal solution found')
            """,
        )
        highs = _write_script(
            root,
            "fake_highs.py",
            """
            import pathlib, sys
            sol_path = None
            for arg in sys.argv[1:]:
                if arg.startswith('--solution_file='):
                    sol_path = pathlib.Path(arg.split('=', 1)[1])
            if sol_path is None:
                sol_path = pathlib.Path('solution.sol')
            sol_path.write_text('Model status: Optimal\\nObjective value: 2\\nx_0 1\\nx_1 0\\n', encoding='utf-8')
            print('Model status: Optimal')
            """,
        )
        concorde = _write_script(
            root,
            "fake_concorde.py",
            """
            import pathlib, sys
            tsp = pathlib.Path(sys.argv[-1])
            (tsp.parent / (tsp.stem + '.sol')).write_text('4\\n0 1 2 3\\n', encoding='utf-8')
            print('Optimal Solution')
            """,
        )
        config = ExternalExactConfig(
            mode="required",
            time_limit_seconds=5.0,
            threads=1,
            solver_paths={
                "open_wbo_exact": open_wbo,
                "kamis_vc_exact": kamis,
                "scip": scip,
                "highs": highs,
                "concorde_exact": concorde,
            },
        )
        cases = [
            (
                "maxsat",
                {
                    "id": "toy-maxsat",
                    "num_variables": 3,
                    "clauses": [[1, 2, 3], [1, 2, 3]],
                },
                "open_wbo_exact",
            ),
            (
                "mis",
                {
                    "id": "toy-mis",
                    "num_vertices": 4,
                    "edges": [[0, 1], [1, 2], [2, 3]],
                },
                "kamis_vc_exact",
            ),
            (
                "mds",
                {
                    "id": "toy-mds",
                    "num_vertices": 4,
                    "edges": [[0, 1], [0, 2], [0, 3]],
                },
                "scip_mip_exact",
            ),
            (
                "coloring",
                {
                    "id": "toy-coloring",
                    "num_vertices": 3,
                    "edges": [[0, 1], [1, 2], [0, 2]],
                },
                "scip_coloring_exact",
            ),
            (
                "tsp",
                {
                    "id": "toy-tsp",
                    "num_cities": 4,
                    "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
                },
                "concorde_exact",
            ),
            (
                "packing_lp",
                {
                    "id": "toy-packing-lp-highs",
                    "num_items": 2,
                    "num_resources": 1,
                    "values": [2, 1],
                    "weights": [[1], [1]],
                    "capacities": [1],
                },
                "highs_lp_exact",
            ),
            (
                "packing_lp",
                {
                    "id": "toy-packing-lp-scip",
                    "num_items": 2,
                    "num_resources": 1,
                    "values": [2, 1],
                    "weights": [[1], [1]],
                    "capacities": [1],
                },
                "scip_lp_exact",
            ),
            (
                "mdkp",
                {
                    "id": "toy-mdkp-highs",
                    "num_items": 2,
                    "num_resources": 1,
                    "values": [2, 1],
                    "weights": [[1], [1]],
                    "capacities": [1],
                },
                "highs_mip_exact",
            ),
            (
                "mdkp",
                {
                    "id": "toy-mdkp-scip",
                    "num_items": 2,
                    "num_resources": 1,
                    "values": [2, 1],
                    "weights": [[1], [1]],
                    "capacities": [1],
                },
                "scip_mdkp_exact",
            ),
        ]
        for problem_name, instance, baseline_name in cases:
            with self.subTest(problem=problem_name):
                problem = get_problem_definition(problem_name)
                exact = problem.exact_solver(instance)
                solvers, discovery = build_external_exact_solvers(problem_name, config, artifact_dir=root / problem_name)
                self.assertIn(baseline_name, solvers)
                self.assertTrue(discovery["solvers"][0]["enabled"])
                summary = evaluate_solver(
                    problem_name,
                    baseline_name,
                    solvers[baseline_name],
                    [{**instance, "optimum_objective": exact.objective_value}],
                    split="validation",
                )
                self.assertEqual(summary["average_normalized_quality"], 1.0)
                self.assertEqual(summary["optimality_rate"], 1.0)
                self.assertEqual(summary["proved_optimal_rate"], 1.0)
                self.assertIn("average_external_runtime_ms", summary)

    def test_required_missing_solver_records_failed_row(self) -> None:
        config = ExternalExactConfig(mode="required", solver_paths={})
        solvers, discovery = build_external_exact_solvers(
            "maxsat",
            config,
            artifact_dir=Path(tempfile.mkdtemp(prefix="dasbench-external-missing-")),
        )
        self.assertIn("maxhs_exact", solvers)
        maxhs_record = next(record for record in discovery["solvers"] if record["baseline_name"] == "maxhs_exact")
        self.assertTrue(maxhs_record["missing_required"])
        instance = {
            "id": "toy-maxsat-missing",
            "num_variables": 3,
            "clauses": [[1, 2, 3]],
            "optimum_objective": 1.0,
        }
        summary = evaluate_solver("maxsat", "maxhs_exact", solvers["maxhs_exact"], [instance], split="validation")
        self.assertEqual(summary["average_normalized_quality"], 0.0)
        self.assertEqual(summary["proved_optimal_rate"], 0.0)
        self.assertEqual(summary["error_count"], 1)

    def test_cli_and_report_persist_external_exact_metadata(self) -> None:
        root = Path(tempfile.mkdtemp(prefix="dasbench-external-cli-"))
        open_wbo = _write_script(
            root,
            "fake_open_wbo.py",
            """
            import re, sys
            text = open(sys.argv[-1], encoding='utf-8').read()
            n = int(re.search(r'p wcnf (\\d+)', text).group(1))
            print('s OPTIMUM FOUND')
            print('o 0')
            print('v ' + ' '.join(str(i) for i in range(1, n + 1)) + ' 0')
            """,
        )
        config_path = root / "external_solvers.json"
        config_path.write_text(json.dumps({"open_wbo_exact": open_wbo}), encoding="utf-8")
        dataset_dir = root / "dataset"
        run_dir = root / "run"
        report_dir = root / "report"
        generate_dataset(
            dataset_dir,
            BenchmarkSpec(
                problem="maxsat",
                family="last_clause_signal_v1",
                instance_params={"num_variables": 8, "num_clauses": 12},
                split_sizes={"train": 2, "validation": 1, "test": 1},
            ),
        )
        args = type("Args", (), {})()
        args.dataset_dir = str(dataset_dir)
        args.run_id = "run"
        args.output_dir = str(run_dir)
        args.generator = "template"
        args.mode = "single"
        args.iterations = 1
        args.beam_width = 1
        args.gurobi_baseline_enabled = False
        args.gurobi_time_limit_seconds = 1.0
        args.gurobi_threads = 1
        args.external_exact_baselines = "required"
        args.external_time_limit_seconds = 5.0
        args.external_threads = 1
        args.external_solver_config = str(config_path)
        cmd_run_agent(args)
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(run_manifest["external_exact_baselines"]["mode"], "required")
        self.assertIn("external_exact_discovery", run_manifest)
        baseline_validation = json.loads((run_dir / "baseline_validation.json").read_text(encoding="utf-8"))
        self.assertIn("open_wbo_exact", baseline_validation)

        report_args = type("Args", (), {})()
        report_args.dataset_dir = str(dataset_dir)
        report_args.agent_run_dir = str(run_dir)
        report_args.output_dir = str(report_dir)
        report_args.repeats = 1
        report_args.include_train = False
        report_args.gurobi_baseline_enabled = None
        report_args.gurobi_time_limit_seconds = None
        report_args.gurobi_threads = None
        report_args.external_exact_baselines = None
        report_args.external_time_limit_seconds = None
        report_args.external_threads = None
        report_args.external_solver_config = None
        cmd_report(report_args)
        payload = json.loads((report_dir / "benchmark_report.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["external_exact_baselines"]["mode"], "required")
        self.assertIn("open_wbo_exact", payload["split_reports"]["validation"])
        self.assertTrue((report_dir / "external_exact_discovery.json").exists())
        diagnostics = report_dir / "open_wbo_exact_validation_diagnostics.jsonl"
        self.assertTrue(diagnostics.exists())


if __name__ == "__main__":
    unittest.main()
