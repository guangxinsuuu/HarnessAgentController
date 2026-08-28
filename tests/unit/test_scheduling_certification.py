from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_certification import run_scheduling_certification_experiment


class SchedulingCertificationTest(unittest.TestCase):
    def test_certification_experiment_writes_priority_and_pruning_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_certification_experiment(
                Path(tmp),
                seed=31,
                train_count=8,
                validation_count=3,
                test_count=4,
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
            self.assertEqual(metrics["benchmark"], "scheduling_certification")
            self.assertTrue((root / "scheduling_certification_table.csv").exists())
            self.assertTrue((root / "table6_clean_certification.csv").exists())
            self.assertTrue((root / "scheduling_certification_per_instance.csv").exists())
            self.assertTrue((root / "scheduling_certification_metrics.json").exists())
            saved = json.loads((root / "scheduling_certification_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("learned priority only", saved["methods"])
            self.assertIn("learned priority + certified pruning", saved["methods"])
            self.assertEqual(saved["methods"]["learned priority only"]["pruned_nodes"], 0)
            self.assertEqual(saved["methods"]["learned priority + certified pruning"]["wrong_prune_rate"], 0.0)
            protocol = saved["certification_protocol"]
            self.assertEqual(protocol["pruning_authority"], "verified lower-bound/feasibility logic only")
            self.assertEqual(protocol["learned_score_authority"], "ranking only")
            self.assertFalse(protocol["learned_score_can_prune"])
            self.assertFalse(protocol["uncertain_constraints_can_prune"])
            self.assertFalse(protocol["oracle_labels_in_inference_features"])


if __name__ == "__main__":
    unittest.main()
