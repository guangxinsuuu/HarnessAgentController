from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.analysis.plots import generate_paper_plots


class PlotGenerationTest(unittest.TestCase):
    def test_generate_paper_plots_writes_svg_manifest(self) -> None:
        metrics = {
            "methods": {
                "A": {"success": 0.5, "llm_calls_mean": 10, "gap": 0.2, "nodes_mean": 20, "proposal_recall": 0.7},
                "B": {"success": 0.8, "llm_calls_mean": 12, "gap": 0.1, "nodes_mean": 15, "proposal_recall": 0.9},
            },
            "transfer": {
                "zero_shot": {"target_training_pair_count": 0, "target_rank_error": 0.4, "success": 0.5, "llm_calls_mean": 10},
                "few_shot": [{"target_training_pair_count": 10, "target_rank_error": 0.3, "success": 0.6, "llm_calls_mean": 9}],
                "llm_only": {"target_training_pair_count": 100, "target_rank_error": 0.2, "success": 0.7, "llm_calls_mean": 8},
            },
            "certification_diagnostics": [{"setting": "FN 5", "verified_recall": 0.95, "bound_slack": 1.0, "nodes": 12}],
            "proposal_diagnostics": [{"proposal_recall_at_k": 0.8, "schema_validity": 1.0}],
            "domain_discrepancy_proxy": 1.2,
            "target_rank_error_zero_shot": 0.4,
        }
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            paths = generate_paper_plots(metrics, out)
            self.assertGreaterEqual(len(paths), 4)
            manifest = json.loads((out / "plot_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["plots"]), len(paths))
            for path in paths:
                self.assertTrue(path.exists())
                self.assertIn("<svg", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()

