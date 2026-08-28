from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_micro import SCHEDULING_FEATURE_NAMES, generate_scheduling_micro_instances, scheduling_feature_vector
from cgr_exp.scheduling_outcome_prm import run_scheduling_outcome_prm


class SchedulingOutcomePRMTest(unittest.TestCase):
    def test_outcome_prm_trains_from_existing_trace_and_cached_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            instances = generate_scheduling_micro_instances(2, 1, 2, seed=9, job_count=4, machine_count=2)
            with (source / "target_prm_trace_records.jsonl").open("w", encoding="utf-8") as handle:
                for task in instances["train"]:
                    root_node = task.root_node()
                    for idx, action in enumerate(task.candidate_actions(root_node, 4)):
                        child = task.apply(root_node, action, child_index=idx)
                        handle.write(
                            json.dumps(
                                {
                                    "task_id": task.task_id,
                                    "features": scheduling_feature_vector(task, child),
                                    "feature_names": SCHEDULING_FEATURE_NAMES,
                                    "best_descendant_objective": task.best_descendant_objective(child),
                                },
                                sort_keys=True,
                            )
                            + "\n"
                        )
            with (source / "proposal_cache.jsonl").open("w", encoding="utf-8") as handle:
                for task in instances["test"]:
                    node = task.root_node()
                    handle.write(
                        json.dumps(
                            {
                                "instance_id": task.task_id,
                                "canonical_state": {"scheduled_jobs": []},
                                "actions": [action.to_dict() for action in task.candidate_actions(node, 4)],
                                "input_tokens": 10,
                                "output_tokens": 5,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            out = root / "out"
            metrics = run_scheduling_outcome_prm(
                source,
                out,
                seed=9,
                train_count=2,
                validation_count=1,
                test_count=2,
                job_count=4,
                machine_count=2,
                node_budget=20,
                action_budget=4,
                rank_epochs=2,
                max_pairs_per_task=20,
            )
            rows = _read_csv(out / "outcome_prm_table.csv")
            self.assertEqual(rows[0]["Policy"], "Outcome-PRM")
            self.assertGreater(metrics["outcome_pair_count"], 0)
            self.assertTrue((out / "outcome_prm_ranker.json").exists())


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
