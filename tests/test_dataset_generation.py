from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dasbench.data import BenchmarkSpec, generate_dataset, load_manifest, load_split
from dasbench.families import available_family_names


class DatasetGenerationTests(unittest.TestCase):
    def test_all_registered_families_generate_valid_splits(self) -> None:
        families_by_problem = available_family_names()
        assert isinstance(families_by_problem, dict)
        for problem, families in families_by_problem.items():
            for family in families:
                with self.subTest(problem=problem, family=family):
                    output_dir = Path(tempfile.mkdtemp(prefix=f"dasbench-{problem}-{family}-"))
                    instance_params = {"num_vertices": 10}
                    if problem == "maxsat":
                        instance_params = {"num_variables": 10, "num_clauses": 18}
                    elif problem == "tsp":
                        instance_params = {"num_cities": 10}
                    elif problem in {"packing_lp", "mdkp"}:
                        instance_params = {"num_items": 10, "num_resources": 3}
                    spec = BenchmarkSpec(
                        problem=problem,
                        family=family,
                        instance_params=instance_params,
                        split_sizes={"train": 3, "validation": 2, "test": 2},
                    )
                    generate_dataset(output_dir, spec)
                    manifest = load_manifest(output_dir)
                    self.assertEqual(manifest["problem"], problem)
                    self.assertEqual(manifest["family"], family)
                    self.assertIn("metric_definition", manifest)
                    self.assertIn("ground_truth_hidden_rule", manifest)
                    self.assertIn("summary", manifest["ground_truth_hidden_rule"])
                    train_instances = load_split(output_dir, "train")
                    self.assertEqual(len(train_instances), 3)
                    self.assertIn("optimum_objective", train_instances[0])
                    public_train_instances = load_split(output_dir, "train", public=True)
                    self.assertNotIn("optimum_objective", public_train_instances[0])


if __name__ == "__main__":
    unittest.main()
