from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_native_q2 import _select_sample, run_scheduling_native_q2
from cgr_exp.scheduling_transfer import ScriptedSchedulingClient


class SchedulingNativeQ2Test(unittest.TestCase):
    def test_mock_native_runner_writes_table5_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            metrics = run_scheduling_native_q2(
                out,
                client=ScriptedSchedulingClient(input_tokens=10, output_tokens=5, model="mock"),
                model="mock",
                seed=3,
                train_count=2,
                validation_count=1,
                test_count=3,
                job_count=4,
                machine_count=2,
                methods=("direct", "self_consistency", "best_of_n"),
                sample_count=2,
                parallel_episodes=2,
            )
            rows = _read_csv(out / "scheduling_native_q2_table.csv")
            by_method = {row["Method"]: row for row in rows}
            self.assertEqual(len(rows), 3)
            self.assertEqual(by_method["Direct / CoT"]["status"], "complete")
            self.assertEqual(by_method["Direct / CoT"]["llm_calls"], "3")
            self.assertEqual(by_method["Self-Consistency"]["llm_calls"], "6")
            self.assertEqual(by_method["Best-of-N"]["llm_calls"], "6")
            self.assertIn("valid_sample_rate", by_method["Direct / CoT"])
            self.assertEqual(by_method["Direct / CoT"]["total_samples"], "3")
            self.assertEqual(by_method["Self-Consistency"]["total_samples"], "6")
            self.assertEqual(by_method["Direct / CoT"]["native_runtime_sample_count"], "3")
            self.assertEqual(by_method["Direct / CoT"]["native_runtime_selection_events"], "3")
            self.assertEqual(by_method["Self-Consistency"]["native_runtime_sample_count"], "6")
            self.assertEqual(by_method["Best-of-N"]["native_runtime_selection_events"], "3")
            self.assertEqual(by_method["Best-of-N"]["parse_failures"], "0")
            self.assertTrue((out / "direct_cot" / "table5_native_row.json").exists())
            samples = _read_jsonl(out / "scheduling_native_q2_samples.jsonl")
            self.assertEqual(len(samples), 15)
            self.assertIn("raw_output", samples[0])
            self.assertIn("parsed_action_count", samples[0])
            self.assertEqual(metrics["live_network_required"], False)
            native_protocol = metrics["native_protocol"]["methods"]
            self.assertEqual(native_protocol["Direct / CoT"]["sample_count"], 1)
            self.assertEqual(native_protocol["Self-Consistency"]["selection_rule"], "majority_job_order_then_first_valid")
            self.assertEqual(native_protocol["Best-of-N"]["verifier_role"], "filter invalid complete schedules and select minimum verified objective")
            self.assertFalse(native_protocol["Best-of-N"]["oracle_future_labels_exposed"])
            alignment = metrics["method_alignment"]
            self.assertIn("single reasoning path", alignment["Direct / CoT"]["source_primitives"])
            self.assertIn("final-answer aggregation", alignment["Self-Consistency"]["source_primitives"])
            self.assertIn("best sample selection", alignment["Best-of-N"]["source_primitives"])

    def test_resume_skips_completed_method_task_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = ScriptedSchedulingClient(input_tokens=10, output_tokens=5, model="mock")
            run_scheduling_native_q2(
                out,
                client=client,
                model="mock",
                seed=4,
                train_count=1,
                validation_count=0,
                test_count=2,
                job_count=4,
                machine_count=2,
                methods=("direct",),
            )
            run_scheduling_native_q2(
                out,
                client=client,
                model="mock",
                seed=4,
                train_count=1,
                validation_count=0,
                test_count=2,
                job_count=4,
                machine_count=2,
                methods=("direct",),
                resume=True,
            )
            per_instance = _read_csv(out / "scheduling_native_q2_per_instance.csv")
            self.assertEqual(len(per_instance), 2)

    def test_resume_reruns_rows_when_sample_audit_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = ScriptedSchedulingClient(input_tokens=10, output_tokens=5, model="mock")
            run_scheduling_native_q2(
                out,
                client=client,
                model="mock",
                seed=14,
                train_count=1,
                validation_count=0,
                test_count=2,
                job_count=4,
                machine_count=2,
                methods=("direct",),
            )
            (out / "scheduling_native_q2_samples.jsonl").unlink()
            run_scheduling_native_q2(
                out,
                client=client,
                model="mock",
                seed=14,
                train_count=1,
                validation_count=0,
                test_count=2,
                job_count=4,
                machine_count=2,
                methods=("direct",),
                resume=True,
            )
            self.assertEqual(len(_read_csv(out / "scheduling_native_q2_per_instance.csv")), 2)
            self.assertEqual(len(_read_jsonl(out / "scheduling_native_q2_samples.jsonl")), 2)

    def test_resume_preserves_previous_methods_when_running_methods_incrementally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            client = ScriptedSchedulingClient(input_tokens=10, output_tokens=5, model="mock")
            common = dict(
                client=client,
                model="mock",
                seed=15,
                train_count=1,
                validation_count=0,
                test_count=2,
                job_count=4,
                machine_count=2,
                sample_count=2,
            )
            run_scheduling_native_q2(out, **common, methods=("direct",))
            run_scheduling_native_q2(out, **common, methods=("self_consistency",), resume=True)
            per_instance = _read_csv(out / "scheduling_native_q2_per_instance.csv")
            samples = _read_jsonl(out / "scheduling_native_q2_samples.jsonl")
            self.assertEqual({row["method_key"] for row in per_instance}, {"direct", "self_consistency"})
            self.assertEqual(len([row for row in per_instance if row["method_key"] == "direct"]), 2)
            self.assertEqual(len([row for row in per_instance if row["method_key"] == "self_consistency"]), 2)
            self.assertEqual(len([row for row in samples if row["method_key"] == "direct"]), 2)
            self.assertEqual(len([row for row in samples if row["method_key"] == "self_consistency"]), 4)

    def test_runner_streams_per_instance_rows_before_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            run_scheduling_native_q2(
                out,
                client=ScriptedSchedulingClient(input_tokens=10, output_tokens=5, model="mock"),
                model="mock",
                seed=5,
                train_count=1,
                validation_count=0,
                test_count=2,
                job_count=4,
                machine_count=2,
                methods=("direct",),
            )
            self.assertEqual(len(_read_csv(out / "scheduling_native_q2_per_instance.csv")), 2)

    def test_self_consistency_tie_break_does_not_use_verified_objective(self) -> None:
        samples = [
            {"sample_index": 0, "valid_complete": True, "objective": 30.0, "job_order": ("a", "b")},
            {"sample_index": 1, "valid_complete": True, "objective": 10.0, "job_order": ("b", "a")},
        ]
        chosen = _select_sample(samples, "majority_job_order_then_first_valid")
        self.assertEqual(chosen["sample_index"], 0)


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
