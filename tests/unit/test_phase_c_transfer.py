from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.experiments import PRESETS, run_experiment
from cgr_exp.ranker import train_domain_classifier, train_domain_classifier_metrics


class PhaseCTransferTest(unittest.TestCase):
    def test_transfer_regime_metadata_records_no_leakage_and_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(PRESETS["smoke"], Path(tmp))
            transfer = metrics["transfer"]
            self.assertEqual(transfer["zero_shot"]["target_training_pair_count"], 0)
            self.assertEqual(transfer["zero_shot"]["solver_training_pair_count"], metrics["solver_pair_count"])
            self.assertEqual(transfer["zero_shot"]["target_evaluation_pair_count"], metrics["llm_eval_pair_count"])
            self.assertEqual(transfer["llm_only"]["solver_training_pair_count"], 0)
            self.assertEqual(transfer["llm_only"]["target_training_pair_count"], metrics["llm_train_pair_count"])
            self.assertEqual(transfer["llm_only"]["target_evaluation_pair_count"], metrics["llm_eval_pair_count"])
            self.assertEqual(
                [row["target_training_pair_count"] for row in transfer["few_shot"]],
                list(PRESETS["smoke"].fewshot_counts),
            )

            histories = [json.loads(line) for line in (Path(tmp) / "train_history.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(all("source_trace_ids" in row and "target_train_trace_ids" in row and "target_eval_trace_ids" in row for row in histories))
            zero_shot = next(row for row in histories if row["stage"] == "zero_shot")
            llm_only = next(row for row in histories if row["stage"] == "llm_only")
            self.assertEqual(zero_shot["target_train_trace_ids"], [])
            self.assertEqual(llm_only["source_trace_ids"], [])
            for row in histories:
                self.assertTrue(all(trace_id.startswith("train-") for trace_id in row["target_train_trace_ids"]))
                self.assertTrue(all(trace_id.startswith("test-") for trace_id in row["target_eval_trace_ids"]))
            for row in histories:
                self.assertFalse(set(row["target_train_trace_ids"]).intersection(row["target_eval_trace_ids"]))

    def test_domain_discrepancy_identical_lower_than_separable(self) -> None:
        identical_source = [[float(i % 3), 1.0] for i in range(30)]
        identical_target = [list(x) for x in identical_source]
        separable_source = [[0.0, float(i % 2)] for i in range(30)]
        separable_target = [[10.0, float(i % 2)] for i in range(30)]
        self.assertLess(
            train_domain_classifier(identical_source, identical_target, seed=1),
            train_domain_classifier(separable_source, separable_target, seed=1),
        )

    def test_domain_classifier_reports_heldout_accuracy_auc_and_proxy(self) -> None:
        source = [[0.0, float(i % 2)] for i in range(50)]
        target = [[5.0, float(i % 2)] for i in range(50)]
        metrics = train_domain_classifier_metrics(source, target, seed=2)
        self.assertIn("heldout_accuracy", metrics)
        self.assertIn("auc", metrics)
        self.assertIn("proxy_discrepancy", metrics)
        self.assertEqual(metrics["proxy_formula"], "2 * (2 * heldout_accuracy - 1), clipped to [0, 2]")
        self.assertGreaterEqual(metrics["heldout_accuracy"], 0.8)
        self.assertGreaterEqual(metrics["auc"], 0.8)
        self.assertGreater(metrics["proxy_discrepancy"], 1.0)


if __name__ == "__main__":
    unittest.main()
