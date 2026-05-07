from __future__ import annotations

import unittest

from dasbench.problems import get_problem_definition


class StrongBaselineTests(unittest.TestCase):
    def _assert_baselines_optimal(
        self,
        problem_name: str,
        instance: dict[str, object],
        baseline_names: list[str],
    ) -> None:
        problem = get_problem_definition(problem_name)
        exact = problem.exact_solver(instance)
        scoring_instance = {**instance, "optimum_objective": exact.objective_value}
        baselines = problem.baseline_registry()
        for baseline_name in baseline_names:
            with self.subTest(problem=problem_name, baseline=baseline_name):
                self.assertIn(baseline_name, baselines)
                raw_solution = baselines[baseline_name](instance)
                solution = problem.canonicalize_solution(raw_solution, instance)
                score = problem.score_solution(scoring_instance, solution)
                self.assertTrue(score.is_feasible, score.error)
                self.assertTrue(score.is_optimal)

    def test_maxsat_rc2_variants_are_optimal_on_toy_instance(self) -> None:
        instance = {
            "id": "toy-maxsat-strong",
            "num_variables": 3,
            "clauses": [
                [1, 2, 3],
                [1, -2, 3],
                [1, 2, -3],
                [-1, -2, -3],
            ],
        }
        self._assert_baselines_optimal(
            "maxsat",
            instance,
            ["rc2_exact", "rc2_glucose4", "rc2_minisat22", "rc2_cadical195"],
        )

    def test_graph_exact_variants_are_optimal_on_toy_instances(self) -> None:
        self._assert_baselines_optimal(
            "mis",
            {
                "id": "toy-mis-strong",
                "num_vertices": 5,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
            },
            ["exact", "cpsat_exact", "clique_branch_bound_exact"],
        )
        self._assert_baselines_optimal(
            "mds",
            {
                "id": "toy-mds-strong",
                "num_vertices": 5,
                "edges": [[0, 1], [0, 2], [0, 3], [3, 4]],
            },
            ["exact", "cpsat_exact", "set_cover_branch_bound_exact"],
        )
        self._assert_baselines_optimal(
            "coloring",
            {
                "id": "toy-coloring-strong",
                "num_vertices": 4,
                "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
            },
            ["exact", "cpsat_exact", "dsatur_branch_bound_exact"],
        )

    def test_tsp_strong_baselines_are_valid_on_toy_instance(self) -> None:
        problem = get_problem_definition("tsp")
        instance = {
            "id": "toy-tsp-strong",
            "num_cities": 4,
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
        exact = problem.exact_solver(instance)
        scoring_instance = {**instance, "optimum_objective": exact.objective_value}
        baselines = problem.baseline_registry()
        for baseline_name in ["two_opt_farthest_insertion", "multi_start_two_opt", "held_karp_exact"]:
            with self.subTest(baseline=baseline_name):
                solution = problem.canonicalize_solution(baselines[baseline_name](instance), instance)
                score = problem.score_solution(scoring_instance, solution)
                self.assertTrue(score.is_feasible, score.error)
                self.assertTrue(score.is_optimal)

    def test_packing_exact_baselines_are_optimal_on_toy_instances(self) -> None:
        self._assert_baselines_optimal(
            "packing_lp",
            {
                "id": "toy-packing-lp-strong",
                "num_items": 2,
                "num_resources": 1,
                "values": [2, 1],
                "weights": [[1], [1]],
                "capacities": [1],
            },
            ["exact", "glop_simplex_exact"],
        )
        self._assert_baselines_optimal(
            "mdkp",
            {
                "id": "toy-mdkp-strong",
                "num_items": 3,
                "num_resources": 2,
                "values": [6, 5, 4],
                "weights": [[4, 2], [3, 4], [2, 3]],
                "capacities": [5, 5],
            },
            ["exact", "cpsat_exact"],
        )


if __name__ == "__main__":
    unittest.main()
