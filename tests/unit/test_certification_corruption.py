from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.experiments import PRESETS, run_experiment
from cgr_exp.datasets.level1_assignment import generate_assignment_fixture


class CertificationCorruptionExperimentTest(unittest.TestCase):
    def test_assignment_best_descendant_from_partial_assignment(self) -> None:
        import random

        task = generate_assignment_fixture(random.Random(4), "desc", job_count=3, resource_count=2)
        root = task.root_node()
        self.assertEqual(task.best_descendant_objective_from_assignments({}), task.best_descendant_objective(root))

    def test_certification_table_contains_required_corruption_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_experiment(PRESETS["smoke"], Path(tmp))
            with (Path(tmp) / "certification_table.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            metrics = json.loads((Path(tmp) / "metrics.json").read_text(encoding="utf-8"))
        settings = {row["setting"] for row in rows}
        self.assertTrue({"no corruption", "FN 5", "FN 10", "FN 20", "FN 30", "FP 1", "FP 2", "FP 5", "uncertain-only 10"}.issubset(settings))
        for row in rows:
            if row["setting"].startswith("FN") or row["setting"] == "no corruption" or row["setting"] == "uncertain-only 10":
                self.assertEqual(float(row["false prune"]), 0.0)
            self.assertIn("verified precision", row)
            self.assertIn("verified recall", row)
            self.assertIn("bound slack", row)
        self.assertIn("certification_diagnostics", metrics)


if __name__ == "__main__":
    unittest.main()
