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
            tau3 = root / "tau3.csv"
            cert = root / "cert.csv"
            matrix = root / "matrix.csv"
            local_primitives = root / "local_primitives.csv"
            authority_source = root / "authority_source.csv"
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
            _write_csv(
                tau3,
                [
                    {
                        "Domain": "Retail",
                        "Method": "Solver-derived canonical controller",
                        "source_artifact": "runs/table10_tau3_canonical/solver_canonical_retail",
                        "result_ready": "True",
                        "result_blockers": "",
                        "notes": "canonical trace audited",
                    }
                ],
            )
            _write_csv(
                cert,
                [
                    {
                        "Table": "Table 6 clean certification",
                        "source_artifact": "runs/q2_structured_cert_reference_100_seed0",
                        "result_ready": "True",
                        "result_blockers": "",
                    }
                ],
            )
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
            _write_csv(
                local_primitives,
                [
                    {
                        "Method": "Direct / CoT",
                        "local_primitives_ready": "True",
                        "missing_anchors": "",
                    }
                ],
            )
            _write_csv(
                authority_source,
                [
                    {
                        "Check": "learned_score_has_no_pruning_authority",
                        "authority_source_ready": "True",
                    }
                ],
            )
            manifest = build_current_experiment_status(
                root / "out",
                q2_readiness=q2,
                q4_readiness=q4,
                tau3_readiness=tau3,
                certification_readiness=cert,
                q2_implementation_matrix=matrix,
                q2_local_primitives=local_primitives,
                certification_authority_source=authority_source,
            )
            self.assertEqual(manifest["q2_rows"], 1)
            self.assertEqual(manifest["q2_local_primitive_ready_rows"], 1)
            self.assertEqual(manifest["tau3_ready_rows"], 1)
            self.assertEqual(manifest["certification_ready_rows"], 1)
            self.assertEqual(manifest["certification_authority_source_ready_rows"], 1)
            text = (root / "out" / "current_experiment_status.md").read_text(encoding="utf-8")
            self.assertIn("Direct / CoT", text)
            self.assertIn("runs/q2_native_llm_baselines_100_seed0", text)
            self.assertIn("Blocked: pending; artifact missing", text)
            self.assertIn("Blocked: missing token accounting", text)
            self.assertIn("runs/q2_remaining_runs/q2_remaining_runs.md", text)
            self.assertIn("1/1 rows preserve required local primitives", text)
            self.assertIn("All Q4 tau/tau2 rows remain pending", text)
            self.assertIn("runs/q4_tau_native", text)
            self.assertIn("native tau artifact missing", text)
            self.assertIn("Table10 tau3 Canonical Harness Evaluation", text)
            self.assertIn("docs/tau3_runbook.md", text)
            self.assertIn("Solver-derived canonical controller", text)
            self.assertIn("runs/table10_tau3_canonical/solver_canonical_retail", text)
            self.assertIn("canonical trace audited", text)
            self.assertIn("runs/q2_structured_cert_reference_100_seed0", text)
            self.assertIn("Authority source audit", text)
            self.assertIn("1/1 authority checks ready", text)
            self.assertIn("scripts/run_table10_tau3.sh build", text)
            saved = json.loads((root / "out" / "current_experiment_status.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["artifact"], "current_experiment_status")
            self.assertEqual(saved["input_artifacts"]["tau3_readiness"], str(tau3))
            self.assertEqual(saved["input_artifacts"]["q2_local_primitives"], str(local_primitives))
            self.assertEqual(saved["input_artifacts"]["certification_authority_source"], str(authority_source))


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
