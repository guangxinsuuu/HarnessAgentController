from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.table_q4_tau import audit_q4_tau_readiness, build_q4_tau_results


class TableQ4TauTest(unittest.TestCase):
    def test_q4_tau_builder_keeps_rows_pending_without_native_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = build_q4_tau_results(root, expected_test_count=5)
            rows = _read_csv(root / "q4_tau_results.csv")
            by_method = {row["Method"]: row for row in rows}
            self.assertEqual(by_method["Native agent rerun"]["Status"], "pending")
            self.assertEqual(by_method["Best-of-N"]["pass^1"], "--")
            self.assertEqual(by_method["Target-PRM"]["source_url"], "https://github.com/sierra-research/tau-bench")
            self.assertEqual(manifest["strict_pending_policy"], True)
            self.assertEqual(manifest["preferred_current_source_url"], "https://github.com/sierra-research/tau2-bench")
            self.assertTrue((root / "q4_tau_results.tex").exists())

    def test_q4_tau_audit_accepts_complete_native_protocol_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_tau_artifacts(native, "Native agent rerun", ["t0", "t1"])
            out = root / "out"
            manifest = audit_q4_tau_readiness(out, expected_test_count=2, native_dir=native)
            readiness = manifest["readiness"]
            rows = {row["Method"]: row for row in _read_csv(out / "q4_tau_readiness.csv")}
            self.assertFalse(readiness["all_rows_result_ready"])
            self.assertEqual(rows["Native agent rerun"]["result_ready"], "True")
            self.assertEqual(rows["Best-of-N"]["result_ready"], "False")

    def test_q4_tau_audit_rejects_non_native_protocol_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / "native"
            native.mkdir()
            _write_tau_artifacts(native, "Native agent rerun", ["t0"])
            (native / "q4_tau_manifest.json").write_text(
                json.dumps(
                    {
                        "benchmark": "synthetic-level3",
                        "native_protocol_preserved": False,
                        "source_url": "local",
                        "methods": ["Native agent rerun"],
                    }
                ),
                encoding="utf-8",
            )
            out = root / "out"
            audit_q4_tau_readiness(out, expected_test_count=1, native_dir=native)
            rows = {row["Method"]: row for row in _read_csv(out / "q4_tau_readiness.csv")}
            self.assertEqual(rows["Native agent rerun"]["result_ready"], "False")
            self.assertIn("must preserve native protocol", rows["Native agent rerun"]["result_blockers"])
            self.assertIn("source_url is not a tau source", rows["Native agent rerun"]["result_blockers"])


def _write_tau_artifacts(path: Path, method: str, task_ids: list[str]) -> None:
    (path / "q4_tau_row.json").write_text(
        json.dumps(
            {
                "Method": method,
                "pass^1": 0.5,
                "pass^4": 0.75,
                "violations": 0.1,
                "tool_calls": 12,
                "tokens": 3456,
                "wall_clock_seconds": 7.89,
                "status": "complete",
                "source_url": "https://github.com/sierra-research/tau-bench",
            }
        ),
        encoding="utf-8",
    )
    _write_csv(path / "q4_tau_per_instance.csv", [{"Method": method, "task_id": task_id} for task_id in task_ids])
    (path / "q4_tau_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "tau-bench",
                "native_protocol_preserved": True,
                "source_url": "https://github.com/sierra-research/tau-bench",
                "methods": [method],
            }
        ),
        encoding="utf-8",
    )


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
