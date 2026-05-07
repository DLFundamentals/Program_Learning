from __future__ import annotations

import json
import unittest
from pathlib import Path

from dasbench.agents.llm import _normalize_hypothesis_payload


class HypothesisSchemaTests(unittest.TestCase):
    def test_schema_requires_expected_fields(self) -> None:
        schema_path = Path("dasbench/schemas/hypothesis_bundle.json")
        payload = json.loads(schema_path.read_text(encoding="utf-8"))
        schema = payload["json_schema"]["schema"]
        self.assertEqual(schema["additionalProperties"], False)
        self.assertEqual(
            set(schema["required"]),
            {
                "title",
                "rule_summary",
                "evidence_to_measure",
                "solver_strategy",
                "expected_failure_modes",
                "diversity_key",
                "notes",
            },
        )

    def test_normalize_hypothesis_payload(self) -> None:
        hypothesis, notes = _normalize_hypothesis_payload(
            {
                "title": "Two Ribbons",
                "rule_summary": "Points form two ribbons.",
                "evidence_to_measure": ["PCA split"],
                "solver_strategy": "Traverse each ribbon.",
                "expected_failure_modes": ["not enough separation"],
                "diversity_key": "Two Ribbons",
                "notes": "Geometry-driven.",
            }
        )
        self.assertEqual(hypothesis["diversity_key"], "two_ribbons")
        self.assertEqual(notes, "Geometry-driven.")

    def test_invalid_hypothesis_payload_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_hypothesis_payload(
                {
                    "title": "Bad",
                    "rule_summary": "Missing list fields.",
                    "evidence_to_measure": "not-a-list",
                    "solver_strategy": "Nope.",
                    "expected_failure_modes": [],
                    "diversity_key": "bad",
                    "notes": "bad",
                }
            )


if __name__ == "__main__":
    unittest.main()
