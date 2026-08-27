from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cgr_exp.experiments import PRESETS, run_experiment


class QualityComputeSweepTest(unittest.TestCase):
    def test_smoke_writes_quality_compute_budget_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_experiment(PRESETS["smoke"], Path(tmp))
            table = Path(tmp) / "quality_compute_table.csv"
            self.assertTrue(table.exists())
            with table.open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertGreaterEqual(len({row["normalized_expensive_budget"] for row in rows}), 3)
        self.assertTrue(all(float(row["proposal_calls"]) <= float(row["normalized_expensive_budget"]) for row in rows))
        self.assertTrue(all("success" in row and "gap" in row for row in rows))


if __name__ == "__main__":
    unittest.main()

