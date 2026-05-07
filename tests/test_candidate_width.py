from __future__ import annotations

import unittest

from dasbench.agents.llm import _child_plans, _seed_plans
from dasbench.agents.template import _seed_specs


def _survivor(slug: str, quality: float = 0.9) -> dict[str, object]:
    return {
        "slug": slug,
        "train": {"average_normalized_quality": quality + 0.01, "average_runtime_ms": 1.0},
        "validation": {"average_normalized_quality": quality, "average_runtime_ms": 1.0},
    }


class CandidateWidthTests(unittest.TestCase):
    def test_llm_seed_plans_use_candidate_width(self) -> None:
        self.assertEqual(len(_seed_plans("beam", candidate_width=5, beam_width=3)), 5)
        self.assertEqual(len(_seed_plans("beam", candidate_width=None, beam_width=3)), 3)
        self.assertEqual(len(_seed_plans("single", candidate_width=5, beam_width=3)), 1)

    def test_llm_child_plans_use_candidate_width_not_beam_width(self) -> None:
        plans = _child_plans(
            1,
            [_survivor("a"), _survivor("b"), _survivor("c")],
            mode="beam",
            beam_width=3,
            candidate_width=10,
        )
        self.assertEqual(len(plans), 10)
        self.assertEqual([plan.slot for plan in plans], list(range(10)))

    def test_template_seed_specs_use_candidate_width(self) -> None:
        self.assertEqual(len(_seed_specs("mis", "beam", candidate_width=5, beam_width=3)), 5)
        self.assertEqual(len(_seed_specs("mis", "beam", candidate_width=None, beam_width=3)), 3)
        self.assertEqual(len(_seed_specs("mis", "single", candidate_width=5, beam_width=3)), 1)


if __name__ == "__main__":
    unittest.main()
