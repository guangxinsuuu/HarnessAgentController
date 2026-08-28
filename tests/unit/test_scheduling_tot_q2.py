from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_tot_q2 import run_scheduling_tot_q2
from cgr_exp.scheduling_transfer import ScriptedSchedulingClient


class SchedulingToTQ2Test(unittest.TestCase):
    def test_tot_q2_runner_writes_native_tot_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = run_scheduling_tot_q2(
                root,
                client=ScriptedSchedulingClient(input_tokens=10, output_tokens=5, model="mock"),
                model="mock",
                seed=6,
                train_count=1,
                validation_count=0,
                test_count=3,
                job_count=4,
                machine_count=2,
                node_budget=20,
                action_budget=3,
                breadth=2,
                value_samples=2,
                perturbation="canonical",
                parallel_episodes=2,
            )
            table = _read_csv(root / "scheduling_tot_q2_table.csv")
            per_instance = _read_csv(root / "scheduling_tot_q2_per_instance.csv")
            self.assertEqual(len(table), 1)
            self.assertEqual(table[0]["Method"], "Tree of Thoughts")
            self.assertEqual(table[0]["status"], "complete")
            self.assertEqual(len(per_instance), 3)
            self.assertGreater(int(table[0]["llm_calls"]), 0)
            self.assertGreater(int(table[0]["tokens"]), 0)
            self.assertEqual(metrics["native_tot_protocol"]["method_select"], "greedy top-b by averaged value samples")
            self.assertTrue((root / "table5_native_row.json").exists())


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
