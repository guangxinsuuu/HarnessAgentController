from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.experiment_status_report import build_current_experiment_status


class ExperimentStatusReportTest(unittest.TestCase):
    def test_status_report_renders_readiness_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q2 = root / "q2.csv"
            q4 = root / "q4.csv"
            cert = root / "cert.csv"
            matrix = root / "matrix.csv"
            _write_csv(
                q2,
                [
                    {
                        "Method": "Direct / CoT",
                        "source_artifact": "runs/q2_native_llm_baselines_100_seed0",
                        "source_policy": "",
                        "result_ready": "False",
                        "compute_ready": "False",
                        "result_blockers": "pending; artifact missing",
                        "compute_blockers": "missing token accounting",
                    }
                ],
            )
            _write_csv(
                q4,
                [
                    {
                        "Method": "Native agent rerun",
                        "source_artifact": "runs/q4_tau_native",
                        "result_ready": "False",
                        "result_blockers": "pending",
                        "notes": "native tau artifact missing",
                    }
                ],
            )
            _write_csv(cert, [{"Table": "Table 6 clean certification", "result_ready": "True", "result_blockers": ""}])
            _write_csv(
                matrix,
                [
                    {
                        "Method": "Direct / CoT",
                        "Local code present": "True",
                        "Implemented scope": "one native complete-schedule LLM sample",
                    }
                ],
            )
            manifest = build_current_experiment_status(root / "out", q2_readiness=q2, q4_readiness=q4, certification_readiness=cert, q2_implementation_matrix=matrix)
            self.assertEqual(manifest["q2_rows"], 1)
            self.assertEqual(manifest["certification_ready_rows"], 1)
            text = (root / "out" / "current_experiment_status.md").read_text(encoding="utf-8")
            self.assertIn("Direct / CoT", text)
            self.assertIn("runs/q2_native_llm_baselines_100_seed0", text)
            self.assertIn("Blocked: pending; artifact missing", text)
            self.assertIn("Blocked: missing token accounting", text)
            self.assertIn("All Q4 tau/tau2 rows remain pending", text)
            self.assertIn("runs/q4_tau_native", text)
            self.assertIn("native tau artifact missing", text)
            saved = json.loads((root / "out" / "current_experiment_status.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["artifact"], "current_experiment_status")


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
