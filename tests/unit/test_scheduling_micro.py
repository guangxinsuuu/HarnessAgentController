from __future__ import annotations

import json
import tempfile
import unittest
import csv
from pathlib import Path

from cgr_exp.scheduling_micro import (
    SCHEDULING_FEATURE_NAMES,
    SchedulingTraceRecord,
    collect_solver_trace_records,
    generate_scheduling_micro_instances,
    make_scheduling_ranking_pairs,
    run_scheduling_micro_experiment,
)


class SchedulingMicroTest(unittest.TestCase):
    def test_micro_instances_have_fixed_splits_and_distinct_ids(self) -> None:
        instances = generate_scheduling_micro_instances(train_count=3, validation_count=2, test_count=1, seed=4, job_count=5, machine_count=2)
        self.assertEqual(set(instances), {"train", "validation", "test"})
        self.assertEqual([len(instances[key]) for key in ("train", "validation", "test")], [3, 2, 1])
        ids = [task.task_id for tasks in instances.values() for task in tasks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_solver_trace_records_include_required_supervision_fields(self) -> None:
        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=5, job_count=5, machine_count=2)["train"][0]
        records = collect_solver_trace_records(task, split="train", node_budget=60, action_budget=6, max_records=8)
        self.assertGreater(len(records), 0)
        first = records[0].to_dict()
        for field in [
            "features",
            "parent_id",
            "depth",
            "incumbent",
            "lower_bound",
            "bound_gap",
            "child_count",
            "future_search_cost",
            "eventual_outcome",
            "prune_reason",
            "state",
        ]:
            self.assertIn(field, first)
        self.assertEqual(first["feature_names"], SCHEDULING_FEATURE_NAMES)

    def test_ranking_pairs_are_deterministic(self) -> None:
        task = generate_scheduling_micro_instances(train_count=1, validation_count=0, test_count=0, seed=6, job_count=5, machine_count=2)["train"][0]
        records = collect_solver_trace_records(task, split="train", node_budget=60, action_budget=6, max_records=10)
        left = make_scheduling_ranking_pairs(records, max_pairs_per_task=12, seed=7)
        right = make_scheduling_ranking_pairs(records, max_pairs_per_task=12, seed=7)
        self.assertEqual(left, right)
        self.assertGreater(len(left), 0)

    def test_ranking_pairs_preserve_process_sibling_pairs(self) -> None:
        records = [
            _record("a", "p1", 10, [1.0]),
            _record("b", "p1", 11, [2.0]),
            _record("c", "p1", 12, [3.0]),
            _record("d", "p2", 20, [4.0]),
            _record("e", "p2", 21, [5.0]),
            _record("f", "p3", 30, [6.0]),
        ]
        pairs = make_scheduling_ranking_pairs(records, max_pairs_per_task=4, seed=3)
        parent_by_feature = {tuple(record.features): record.parent_id for record in records}
        sibling_pairs = [
            pair
            for pair in pairs
            if parent_by_feature[tuple(pair[0])] == parent_by_feature[tuple(pair[1])]
        ]
        self.assertGreaterEqual(len(sibling_pairs), 2)
        self.assertLessEqual(len(pairs), 4)

    def test_micro_experiment_writes_sharded_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_scheduling_micro_experiment(
                Path(tmp),
                seed=8,
                train_count=5,
                validation_count=2,
                test_count=3,
                job_count=5,
                machine_count=2,
                node_budget=60,
                action_budget=6,
                records_per_instance=8,
                shard_size=5,
            )
            self.assertEqual(metrics["benchmark"], "scheduling_micro")
            self.assertGreater(metrics["train_pair_count"], 0)
            self.assertTrue((Path(tmp) / "scheduling_micro_instances.json").exists())
            self.assertTrue((Path(tmp) / "scheduling_ranking_pairs.jsonl").exists())
            self.assertTrue((Path(tmp) / "scheduling_micro_table.csv").exists())
            self.assertTrue(list(Path(tmp).glob("scheduling_solver_trace_shard_*.jsonl")))
            with (Path(tmp) / "scheduling_micro_table.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertIn("frontier_top1_accuracy", rows[0])
            self.assertIn("pairwise_accuracy", rows[0])
            self.assertIn("selection_regret_mean", rows[0])
            saved = json.loads((Path(tmp) / "scheduling_micro_metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["splits"], {"test": 3, "train": 5, "validation": 2})
            self.assertIn("frontier_top1_accuracy", saved["methods"]["MLP"])


def _record(node_id: str, parent_id: str, best_descendant_objective: float, features: list[float]) -> SchedulingTraceRecord:
    return SchedulingTraceRecord(
        task_id="task",
        split="train",
        node_id=node_id,
        parent_id=parent_id,
        depth=1,
        incumbent=None,
        lower_bound=best_descendant_objective,
        bound_gap=0.0,
        child_count=0,
        best_descendant_objective=best_descendant_objective,
        future_search_cost=best_descendant_objective,
        eventual_outcome="complete",
        prune_reason="",
        features=features,
        feature_names=["id"],
        state={},
    )


if __name__ == "__main__":
    unittest.main()
