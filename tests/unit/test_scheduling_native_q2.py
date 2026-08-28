from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_native_q2 import run_scheduling_native_q2
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
            self.assertTrue((out / "direct_cot" / "table5_native_row.json").exists())
            self.assertEqual(metrics["live_network_required"], False)

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


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
