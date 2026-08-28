from __future__ import annotations

import csv
import json
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
            self.assertEqual({row["Method"] for row in per_instance}, {"Tree of Thoughts"})
            self.assertGreater(int(table[0]["llm_calls"]), 0)
            self.assertGreater(int(table[0]["tokens"]), 0)
            self.assertGreater(int(table[0]["tot_generate_events"]), 0)
            self.assertGreater(int(table[0]["tot_evaluate_events"]), 0)
            self.assertGreater(int(table[0]["tot_select_events"]), 0)
            self.assertGreater(int(table[0]["tot_value_sample_count"]), 0)
            self.assertEqual(metrics["native_tot_protocol"]["official_entrypoints"], ["method_generate", "method_evaluate", "method_select"])
            self.assertEqual(metrics["native_tot_protocol"]["method_select"], "greedy top-b by averaged value samples")
            self.assertEqual(metrics["native_tot_protocol"]["n_select_sample"], 2)
            self.assertEqual(metrics["native_tot_protocol"]["n_evaluate_sample"], 2)
            self.assertFalse(metrics["native_tot_protocol"]["oracle_future_labels_exposed"])
            primitives = metrics["method_alignment"]["Tree of Thoughts"]["source_primitives"]
            self.assertIn("method_generate", primitives)
            self.assertIn("method_evaluate", primitives)
            self.assertIn("method_select", primitives)
            self.assertIn("breadth-first search", primitives)
            self.assertTrue((root / "table5_native_row.json").exists())
            trace = _read_jsonl(root / "scheduling_tot_q2_trace.jsonl")
            events = {row["event"] for row in trace}
            self.assertIn("generate", events)
            self.assertIn("evaluate_select", events)
            evaluate_events = [row for row in trace if row["event"] == "evaluate_select"]
            self.assertTrue(evaluate_events)
            self.assertIn("selected_ids", evaluate_events[0])
            self.assertIn("value_samples", evaluate_events[0])
            self.assertIn("scheduling_tot_q2_trace.jsonl", metrics["produced_artifacts"])
            finish_rows = [row for row in trace if row["event"] == "finish"]
            self.assertTrue(finish_rows)
            self.assertEqual(finish_rows[0]["row"]["Method"], "Tree of Thoughts")
            self.assertGreater(finish_rows[0]["row"]["tot_generate_events"], 0)

    def test_tot_resume_reruns_rows_when_trace_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = ScriptedSchedulingClient(input_tokens=10, output_tokens=5, model="mock")
            kwargs = dict(
                client=client,
                model="mock",
                seed=16,
                train_count=1,
                validation_count=0,
                test_count=2,
                job_count=4,
                machine_count=2,
                node_budget=20,
                action_budget=3,
                breadth=2,
                value_samples=2,
                perturbation="canonical",
            )
            run_scheduling_tot_q2(root, **kwargs)
            (root / "scheduling_tot_q2_trace.jsonl").unlink()
            run_scheduling_tot_q2(root, **kwargs, resume=True)
            self.assertEqual(len(_read_csv(root / "scheduling_tot_q2_per_instance.csv")), 2)
            finish_events = [row for row in _read_jsonl(root / "scheduling_tot_q2_trace.jsonl") if row["event"] == "finish"]
            self.assertEqual(len(finish_events), 2)

    def test_tot_resume_backfills_runtime_counts_from_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            client = ScriptedSchedulingClient(input_tokens=10, output_tokens=5, model="mock")
            kwargs = dict(
                client=client,
                model="mock",
                seed=26,
                train_count=1,
                validation_count=0,
                test_count=2,
                job_count=4,
                machine_count=2,
                node_budget=20,
                action_budget=3,
                breadth=2,
                value_samples=2,
                perturbation="canonical",
            )
            run_scheduling_tot_q2(root, **kwargs)
            per_instance_path = root / "scheduling_tot_q2_per_instance.csv"
            old_rows = _read_csv(per_instance_path)
            stripped_rows = [
                {
                    key: value
                    for key, value in row.items()
                    if key not in {"tot_generate_events", "tot_evaluate_events", "tot_select_events", "tot_value_sample_count"}
                }
                for row in old_rows
            ]
            _write_csv(per_instance_path, stripped_rows)

            run_scheduling_tot_q2(root, **kwargs, resume=True)
            table = _read_csv(root / "scheduling_tot_q2_table.csv")
            self.assertGreater(int(table[0]["tot_generate_events"]), 0)
            self.assertGreater(int(table[0]["tot_evaluate_events"]), 0)
            self.assertGreater(int(table[0]["tot_select_events"]), 0)
            self.assertGreater(int(table[0]["tot_value_sample_count"]), 0)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
