from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dasbench.agents.candidate import AnalysisTimeoutError, run_analysis


class CandidateAnalysisTimeoutTests(unittest.TestCase):
    def test_run_analysis_times_out_and_writes_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "candidate"
            artifact_dir = root / "artifacts"
            candidate_dir.mkdir()
            (candidate_dir / "analyze.py").write_text(
                "import time\n\n"
                "def analyze(train_instances, manifest=None):\n"
                "    time.sleep(1.0)\n"
                "    return {'ok': True}\n",
                encoding="utf-8",
            )

            with self.assertRaises(AnalysisTimeoutError):
                run_analysis(
                    candidate_dir,
                    [{"id": "x"}],
                    artifact_dir=artifact_dir,
                    timeout_seconds=0.05,
                )

            payload = json.loads((artifact_dir / "analysis_error.json").read_text(encoding="utf-8"))
            self.assertIn("AnalysisTimeoutError", payload["error"])
            self.assertEqual(payload["timeout_seconds"], 0.05)
            self.assertFalse((artifact_dir / "analysis.json").exists())

    def test_run_analysis_writes_successful_analysis(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "candidate"
            artifact_dir = root / "artifacts"
            candidate_dir.mkdir()
            (candidate_dir / "analyze.py").write_text(
                "def analyze(train_instances, manifest=None):\n"
                "    return {'num_instances': len(train_instances)}\n",
                encoding="utf-8",
            )

            analysis = run_analysis(
                candidate_dir,
                [{"id": "a"}, {"id": "b"}],
                artifact_dir=artifact_dir,
                timeout_seconds=1.0,
            )

            self.assertEqual(analysis, {"num_instances": 2})
            payload = json.loads((artifact_dir / "analysis.json").read_text(encoding="utf-8"))
            self.assertEqual(payload, {"num_instances": 2})


if __name__ == "__main__":
    unittest.main()
