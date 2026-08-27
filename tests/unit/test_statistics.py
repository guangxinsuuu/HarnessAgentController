from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cgr_exp.analysis.statistics import aggregate_seed_metrics, write_seed_aggregate


class StatisticsTest(unittest.TestCase):
    def test_aggregate_seed_metrics_reports_mean_std_p95(self) -> None:
        rows = [
            {"seed": 1, "method": "A", "success": 1.0, "nodes": 10.0},
            {"seed": 2, "method": "A", "success": 0.0, "nodes": 20.0},
            {"seed": 3, "method": "A", "success": 1.0, "nodes": 30.0},
        ]
        agg = aggregate_seed_metrics(rows, group_key="method", metric_keys=["success", "nodes"])
        self.assertEqual(len(agg), 1)
        row = agg[0]
        self.assertEqual(row["method"], "A")
        self.assertAlmostEqual(row["success_mean"], 2.0 / 3.0)
        self.assertGreater(row["success_std"], 0.0)
        self.assertEqual(row["nodes_p95"], 30.0)
        self.assertEqual(row["seed_count"], 3)
        self.assertIsInstance(row["success_bootstrap_ci95_low"], float)
        self.assertIsInstance(row["success_bootstrap_ci95_high"], float)
        self.assertLessEqual(row["success_bootstrap_ci95_low"], row["success_mean"])
        self.assertGreaterEqual(row["success_bootstrap_ci95_high"], row["success_mean"])

    def test_write_seed_aggregate_creates_csv_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            write_seed_aggregate(
                out,
                rows=[{"seed": 1, "method": "A", "success": 1.0}],
                group_key="method",
                metric_keys=["success"],
            )
            self.assertTrue((out / "multi_seed_summary.csv").exists())
            self.assertTrue((out / "multi_seed_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
