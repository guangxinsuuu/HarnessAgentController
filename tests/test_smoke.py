from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.experiments import PRESETS, run_experiment


class SmokeExperimentTest(unittest.TestCase):
    def test_smoke_run_writes_expected_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(PRESETS["smoke"], Path(tmp))
            self.assertGreater(metrics["solver_pair_count"], 0)
            self.assertGreater(metrics["llm_pair_count"], 0)
            self.assertTrue((Path(tmp) / "primary_table.csv").exists())
            self.assertTrue((Path(tmp) / "transfer_table.csv").exists())
            with (Path(tmp) / "trace_sample.jsonl").open(encoding="utf-8") as f:
                first = json.loads(next(f))
            self.assertIn("method", first)
            self.assertIn("result", first)


if __name__ == "__main__":
    unittest.main()

