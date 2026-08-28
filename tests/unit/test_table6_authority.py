from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.table6_authority import run_table6_authority_experiment


class Table6AuthorityTest(unittest.TestCase):
    def test_table6_authority_runner_writes_required_rows_and_audit_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = run_table6_authority_experiment(
                root,
                seed=7,
                train_count=8,
                validation_count=3,
                test_count=4,
                job_count=5,
                machine_count=2,
                node_budget=50,
                action_budget=5,
                records_per_instance=10,
                rank_epochs=4,
                corruption_seeds=(0, 1),
            )
            rows = _read_csv(root / "table6_authority.csv")
            by_condition = {row["Condition"]: row for row in rows}
            self.assertIn("Clean certified", by_condition)
            self.assertIn("Verified FN 1%", by_condition)
            self.assertIn("Verified FN 10%", by_condition)
            self.assertIn("Verified FP 1%", by_condition)
            self.assertIn("Uncertain-only 1%", by_condition)
            self.assertIn("Uncertain-only 20%", by_condition)
            self.assertIn("Learned-score pruning clean", by_condition)
            self.assertEqual(float(by_condition["Clean certified"]["Verified precision"]), 1.0)
            self.assertEqual(float(by_condition["Clean certified"]["Verified recall"]), 1.0)
            self.assertEqual(float(by_condition["Clean certified"]["Semantic false prune"]), 0.0)
            self.assertEqual(float(by_condition["Uncertain-only 20%"]["Verified precision"]), 1.0)
            self.assertEqual(float(by_condition["Uncertain-only 20%"]["Verified recall"]), 1.0)
            self.assertIn("Bound validity", by_condition["Clean certified"])
            self.assertIn("Audited prune decisions", by_condition["Clean certified"])
            self.assertIn("Verified infeasible rejections", by_condition["Clean certified"])
            self.assertIn("Learned-score prunes", by_condition["Learned-score pruning clean"])
            self.assertEqual(float(by_condition["Uncertain-only 20%"]["Verified FP count"]), 0.0)
            self.assertEqual(float(by_condition["Uncertain-only 20%"]["Verified infeasible rejections"]), 0.0)
            self.assertGreater(float(by_condition["Learned-score pruning clean"]["Learned-score prunes"]), 0.0)
            self.assertTrue((root / "table6_authority_per_instance.csv").exists())
            self.assertTrue((root / "table7_grounding_robustness.csv").exists())
            saved = json.loads((root / "table6_authority_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["benchmark"], "table6_authority_separation")
            self.assertFalse(saved["oracle_labels_in_inference_features"])
            protocol = saved["certification_protocol"]
            self.assertEqual(protocol["learned_score_authority"], "ranking only")
            self.assertFalse(protocol["learned_score_can_prune"])
            self.assertFalse(protocol["uncertain_constraints_can_prune"])
            self.assertIn("Learned-score pruning clean", saved["table7_robustness_conditions"])
            self.assertEqual(metrics["splits"]["test"], 4)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
