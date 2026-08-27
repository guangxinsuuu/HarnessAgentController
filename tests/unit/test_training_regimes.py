from __future__ import annotations

import unittest

from cgr_exp.training.regimes import build_transfer_training_plan


class TrainingRegimePlanTest(unittest.TestCase):
    def test_transfer_plan_records_regimes_and_keeps_eval_out_of_training(self) -> None:
        plan = build_transfer_training_plan(
            solver_trace_ids=["source-1", "source-2"],
            target_train_trace_ids=["train-1", "train-2", "train-3"],
            target_eval_trace_ids=["test-1", "test-2"],
            solver_pair_count=20,
            target_train_pair_count=30,
            target_eval_pair_count=10,
            fewshot_counts=(1, 3),
            rank_epochs=5,
        )
        by_stage = {row["stage"]: row for row in plan}
        self.assertEqual(by_stage["zero_shot"]["target_train_trace_ids"], [])
        self.assertEqual(by_stage["zero_shot"]["target_eval_trace_ids"], ["test-1", "test-2"])
        self.assertEqual(by_stage["few_shot_3"]["target_train_trace_ids"], ["train-1", "train-2", "train-3"])
        self.assertEqual(by_stage["few_shot_3"]["target_training_pair_count"], 3)
        self.assertEqual(by_stage["llm_only"]["source_trace_ids"], [])
        self.assertEqual(by_stage["llm_only"]["target_training_pair_count"], 30)

    def test_transfer_plan_rejects_target_eval_leakage(self) -> None:
        with self.assertRaises(ValueError):
            build_transfer_training_plan(
                solver_trace_ids=["source-1"],
                target_train_trace_ids=["train-1", "test-1"],
                target_eval_trace_ids=["test-1"],
                solver_pair_count=1,
                target_train_pair_count=2,
                target_eval_pair_count=1,
                fewshot_counts=(1,),
                rank_epochs=3,
            )


if __name__ == "__main__":
    unittest.main()
