from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cgr_exp.scheduling_structured_q2 import run_scheduling_structured_q2


class SchedulingStructuredQ2Test(unittest.TestCase):
    def test_structured_q2_writes_solver_and_heuristic_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            metrics = run_scheduling_structured_q2(
                root,
                seed=9,
                train_count=1,
                validation_count=0,
                test_count=3,
                job_count=4,
                machine_count=2,
                cpsat_time_limit_seconds=0.2,
            )
            rows = _read_csv(root / "scheduling_structured_q2_table.csv")
            by_method = {row["Method"]: row for row in rows}
            self.assertIn("CP-SAT", by_method)
            self.assertIn("EST / list scheduling heuristic", by_method)
            self.assertEqual(by_method["EST / list scheduling heuristic"]["llm_calls"], "0")
            self.assertEqual(by_method["EST / list scheduling heuristic"]["tokens"], "0")
            self.assertEqual(by_method["CP-SAT"]["structured_runtime_solver"], "ortools_cp_sat")
            self.assertGreater(int(by_method["CP-SAT"]["structured_runtime_interval_variables"]), 0)
            self.assertGreater(int(by_method["CP-SAT"]["structured_runtime_no_overlap_constraints"]), 0)
            self.assertEqual(by_method["EST / list scheduling heuristic"]["structured_runtime_heuristic"], "earliest_start_list_scheduling")
            self.assertGreater(int(by_method["EST / list scheduling heuristic"]["structured_runtime_dispatch_steps"]), 0)
            self.assertIn("min(start", by_method["EST / list scheduling heuristic"]["structured_runtime_tie_break_rule"])
            self.assertEqual(metrics["llm_calls"], 0)
            self.assertIn("optional interval variables", metrics["method_alignment"]["CP-SAT"]["source_primitives"])
            self.assertIn(
                "ready-list dispatch",
                metrics["method_alignment"]["EST / list scheduling heuristic"]["source_primitives"],
            )
            self.assertTrue((root / "scheduling_structured_q2_per_instance.csv").exists())


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
