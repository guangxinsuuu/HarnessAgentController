from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_micro import TinyPairwiseMLPRanker, generate_scheduling_micro_instances
from cgr_exp.scheduling_policy_diagnosis import run_scheduling_policy_diagnosis


class SchedulingPolicyDiagnosisTest(unittest.TestCase):
    def test_policy_diagnosis_writes_summary_and_failure_conditioned_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            tasks = generate_scheduling_micro_instances(2, 1, 2, seed=5, job_count=4, machine_count=2)["test"]
            ranker = TinyPairwiseMLPRanker(dim=20, hidden_dim=4, seed=0)
            (run_dir / "scheduling_llm_pilot_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
            (run_dir / "target_prm_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
            with (run_dir / "proposal_cache.jsonl").open("w", encoding="utf-8") as handle:
                for task in tasks:
                    node = task.root_node()
                    actions = [action.to_dict() for action in task.candidate_actions(node, 4)]
                    handle.write(
                        json.dumps(
                            {
                                "instance_id": task.task_id,
                                "cache_key": f"{task.task_id}:root",
                                "canonical_state": {"observable_history": []},
                                "actions": actions,
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
            _write_csv(
                run_dir / "scheduling_llm_pilot_per_instance.csv",
                [
                    {"Policy": "Target-PRM", "task_id": tasks[0].task_id, "success": "True"},
                    {"Policy": "Solver-trained MLP", "task_id": tasks[0].task_id, "success": "True"},
                    {"Policy": "Target-PRM", "task_id": tasks[1].task_id, "success": "False"},
                    {"Policy": "Solver-trained MLP", "task_id": tasks[1].task_id, "success": "True"},
                ],
            )
            out = root / "out"
            metrics = run_scheduling_policy_diagnosis(run_dir, out, seed=5, train_count=2, validation_count=1, test_count=2, job_count=4, machine_count=2)
            summary = _read_csv(out / "policy_diagnosis_summary.csv")
            groups = _read_csv(out / "policy_diagnosis_failure_conditioned.csv")
            self.assertEqual(metrics["decision_count"], len(_read_csv(out / "policy_diagnosis_decisions.csv")))
            self.assertEqual({row["Policy"] for row in summary}, {"Solver-trained MLP", "Target-PRM"})
            self.assertIn("solver only succeeds", {row["group"] for row in groups})


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
