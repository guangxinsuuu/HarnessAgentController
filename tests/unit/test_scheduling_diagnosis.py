from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_diagnosis import run_scheduling_transfer_diagnosis


class SchedulingDiagnosisTest(unittest.TestCase):
    def test_diagnosis_writes_distribution_and_transfer_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_transfer_diagnosis(
                Path(tmp),
                seed=41,
                source_count=8,
                source_validation_count=3,
                target_count=4,
                job_count=5,
                machine_count=2,
                node_budget=70,
                action_budget=6,
                records_per_instance=12,
                rank_epochs=5,
                useful_action_recall=0.9,
                perturbation="canonical",
            )
            root = Path(tmp)
            self.assertEqual(metrics["benchmark"], "scheduling_transfer_diagnosis")
            self.assertGreater(metrics["source_record_count"], 0)
            self.assertGreater(metrics["target_record_count"], 0)
            self.assertIn("accuracy", metrics["domain_classifier"])
            self.assertIn("target_pairwise_accuracy", metrics["ranking_transfer"])
            self.assertTrue((root / "scheduling_transfer_feature_distribution.csv").exists())
            self.assertTrue((root / "scheduling_transfer_diagnosis_metrics.json").exists())
            saved = json.loads((root / "scheduling_transfer_diagnosis_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["benchmark"], "scheduling_transfer_diagnosis")
            self.assertEqual(len(saved["common_feature_names"]), 20)


if __name__ == "__main__":
    unittest.main()
