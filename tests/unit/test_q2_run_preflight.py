from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.q2_run_preflight import build_q2_run_preflight


class Q2RunPreflightTest(unittest.TestCase):
    def test_preflight_reports_remaining_units_without_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "preflight"
            manifest = build_q2_run_preflight(
                out,
                test_count=2,
                native_dir=root / "native",
                learned_dir=root / "learned",
                tot_dir=root / "tot",
                env={},
            )
            rows = {row["run_group"]: row for row in manifest["rows"]}
            self.assertFalse(manifest["api_key_present"])
            self.assertEqual(rows["learned_search"]["remaining_units"], 2)
            self.assertEqual(rows["native_frozen"]["remaining_units"], 6)
            self.assertEqual(rows["native_frozen"]["worst_case_request_count_upper_bound"], 34)
            self.assertEqual(rows["native_tot"]["remaining_units"], 2)
            self.assertTrue((out / "q2_run_preflight.md").exists())

    def test_preflight_counts_completed_native_and_tot_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_csv(
                native / "scheduling_native_q2_per_instance.csv",
                [
                    {"method_key": "direct", "task_id": "t0", "task_index": 0, "selection_rule": "first_valid"},
                    {"method_key": "self_consistency", "task_id": "t0", "task_index": 0, "selection_rule": "majority_job_order_then_first_valid"},
                ],
            )
            _write_jsonl(
                native / "scheduling_native_q2_samples.jsonl",
                [{"method_key": "direct", "task_id": "t0"}]
                + [{"method_key": "self_consistency", "task_id": "t0"} for _ in range(8)],
            )
            tot = root / "tot"
            tot.mkdir()
            _write_csv(tot / "scheduling_tot_q2_per_instance.csv", [{"rendering_id": "r0"}])
            _write_jsonl(tot / "scheduling_tot_q2_trace.jsonl", [{"event": "finish", "rendering_id": "r0"}])
            manifest = build_q2_run_preflight(
                root / "preflight",
                test_count=2,
                native_dir=native,
                learned_dir=root / "learned",
                tot_dir=tot,
                env={"CGR_LLM_API_KEY": "secret"},
            )
            rows = {row["run_group"]: row for row in manifest["rows"]}
            self.assertTrue(manifest["api_key_present"])
            self.assertEqual(rows["native_frozen"]["completed_units"], 2)
            self.assertEqual(rows["native_frozen"]["remaining_units"], 4)
            self.assertEqual(rows["native_tot"]["completed_units"], 1)
            self.assertEqual(rows["native_tot"]["remaining_units"], 1)


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    unittest.main()
