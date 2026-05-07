from __future__ import annotations

import unittest

from dasbench.problems import get_problem_definition


class ProblemScoringTests(unittest.TestCase):
    def test_coloring_exact_solver_and_invalid_solution(self) -> None:
        problem = get_problem_definition("coloring")
        instance = {
            "id": "toy-coloring",
            "num_vertices": 4,
            "edges": [[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]],
        }
        exact = problem.exact_solver(instance)
        self.assertEqual(exact.objective_value, 3.0)
        valid_score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, exact.solution)
        self.assertTrue(valid_score.is_optimal)
        invalid_score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, [0, 1, 0, 1])
        self.assertFalse(invalid_score.is_feasible)
        self.assertEqual(invalid_score.normalized_quality, 0.0)

    def test_maxsat_exact_solver_and_scoring(self) -> None:
        problem = get_problem_definition("maxsat")
        instance = {
            "id": "toy-maxsat",
            "num_variables": 3,
            "clauses": [
                [1, 2, 3],
                [1, -2, 3],
                [1, 2, -3],
            ],
        }
        exact = problem.exact_solver(instance)
        self.assertEqual(exact.objective_value, 3.0)
        score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, exact.solution)
        self.assertTrue(score.is_feasible)
        self.assertTrue(score.is_optimal)
        self.assertEqual(score.normalized_quality, 1.0)

    def test_mis_exact_solver_and_invalid_solution(self) -> None:
        problem = get_problem_definition("mis")
        instance = {
            "id": "toy-mis",
            "num_vertices": 4,
            "edges": [[0, 1], [1, 2], [2, 3]],
        }
        exact = problem.exact_solver(instance)
        self.assertEqual(exact.objective_value, 2.0)
        valid_score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, exact.solution)
        self.assertTrue(valid_score.is_optimal)
        invalid_score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, [1, 2])
        self.assertFalse(invalid_score.is_feasible)
        self.assertEqual(invalid_score.normalized_quality, 0.0)

    def test_mds_exact_solver_and_scoring(self) -> None:
        problem = get_problem_definition("mds")
        instance = {
            "id": "toy-mds",
            "num_vertices": 5,
            "edges": [[0, 1], [0, 2], [0, 3], [0, 4]],
        }
        exact = problem.exact_solver(instance)
        self.assertEqual(exact.objective_value, 1.0)
        score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, exact.solution)
        self.assertTrue(score.is_feasible)
        self.assertTrue(score.is_optimal)
        self.assertEqual(score.normalized_quality, 1.0)

    def test_tsp_exact_solver_and_invalid_tour(self) -> None:
        problem = get_problem_definition("tsp")
        instance = {
            "id": "toy-tsp",
            "num_cities": 4,
            "points": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        }
        exact = problem.exact_solver(instance)
        self.assertAlmostEqual(exact.objective_value, 4.0, places=6)
        valid_score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, exact.solution)
        self.assertTrue(valid_score.is_optimal)
        invalid_score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, [0, 1, 1, 3])
        self.assertFalse(invalid_score.is_feasible)
        self.assertEqual(invalid_score.normalized_quality, 0.0)

    def test_packing_lp_exact_solver_and_infeasible_solution(self) -> None:
        problem = get_problem_definition("packing_lp")
        instance = {
            "id": "toy-packing-lp",
            "num_items": 2,
            "num_resources": 1,
            "values": [2, 1],
            "weights": [[1], [1]],
            "capacities": [1],
        }
        exact = problem.exact_solver(instance)
        self.assertAlmostEqual(exact.objective_value, 2.0, places=6)
        score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, exact.solution)
        self.assertTrue(score.is_feasible)
        self.assertTrue(score.is_optimal)
        infeasible = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, [1.0, 1.0])
        self.assertFalse(infeasible.is_feasible)
        self.assertEqual(infeasible.normalized_quality, 0.0)

    def test_mdkp_exact_solver_and_bool_vector_canonicalization(self) -> None:
        problem = get_problem_definition("mdkp")
        instance = {
            "id": "toy-mdkp",
            "num_items": 3,
            "num_resources": 2,
            "values": [6, 5, 4],
            "weights": [[4, 2], [3, 4], [2, 3]],
            "capacities": [5, 5],
        }
        exact = problem.exact_solver(instance)
        self.assertEqual(exact.objective_value, 6.0)
        score = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, exact.solution)
        self.assertTrue(score.is_feasible)
        self.assertTrue(score.is_optimal)
        self.assertEqual(problem.canonicalize_solution([True, False, True], instance), [0, 2])
        invalid = problem.score_solution({**instance, "optimum_objective": exact.objective_value}, [0, 2])
        self.assertFalse(invalid.is_feasible)


if __name__ == "__main__":
    unittest.main()
