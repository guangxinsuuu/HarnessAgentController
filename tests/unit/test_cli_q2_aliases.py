from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
import csv
from pathlib import Path

from cgr_exp.cli import COMMANDS


class CliQ2AliasesTest(unittest.TestCase):
    def test_table4_q2_aliases_are_registered(self) -> None:
        self.assertIn("build-table4-q2", COMMANDS)
        self.assertIn("audit-table4-q2", COMMANDS)
        self.assertIn("build-table5-q2", COMMANDS)
        self.assertIn("audit-table5-q2", COMMANDS)
        self.assertIn("audit-q2-source-evidence", COMMANDS)
        self.assertIn("verify-q2-source-refs", COMMANDS)
        self.assertIn("build-q2-implementation-matrix", COMMANDS)
        self.assertIn("audit-q2-local-primitives", COMMANDS)
        self.assertIn("build-q2-remaining-runs", COMMANDS)
        self.assertIn("build-q2-run-preflight", COMMANDS)
        self.assertIn("backfill-scheduling-llm-pilot-audit-artifacts", COMMANDS)
        self.assertIn("build-q4-tau-adapter-contract", COMMANDS)

    def test_table4_and_legacy_table5_aliases_expose_same_q2_arguments(self) -> None:
        expected_flags = (
            "--solver-prior-dir",
            "--expected-test-count",
            "--tot-dir",
            "--target-prm-dir",
            "--direct-dir",
            "--self-consistency-dir",
            "--best-of-n-dir",
            "--rest-mcts-dir",
            "--solver-certification-dir",
            "--structured-dir",
        )
        for command in ("build-table4-q2", "build-table5-q2", "audit-table4-q2", "audit-table5-q2"):
            result = subprocess.run(
                [sys.executable, "-m", "cgr_exp.cli", command, "--help"],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for flag in expected_flags:
                self.assertIn(flag, result.stdout, command)

    def test_table4_audit_command_manifest_records_input_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-table4-q2",
                    "--output-dir",
                    str(out),
                    "--expected-test-count",
                    "7",
                    "--structured-dir",
                    "runs/structured",
                    "--tot-dir",
                    "runs/tot",
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["expected_test_count"], 7)
            self.assertEqual(manifest["input_dirs"]["structured_dir"], "runs/structured")
            self.assertEqual(manifest["input_dirs"]["tot_dir"], "runs/tot")
            self.assertEqual(manifest["input_dirs"]["solver_prior_dir"], "runs/scheduling_llm_pilot_real_100_seed0")

    def test_q2_source_evidence_audit_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "source_audit"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-q2-source-evidence",
                    "--output-dir",
                    str(out),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["command"], "audit-q2-source-evidence")
            self.assertEqual(manifest["all_external_sources_ready"], True)
            self.assertTrue((out / "q2_source_evidence_audit.csv").exists())

    def test_q2_implementation_matrix_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "matrix"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "build-q2-implementation-matrix",
                    "--output-dir",
                    str(out),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["command"], "build-q2-implementation-matrix")
            self.assertTrue(manifest["all_table_rows_covered"])
            self.assertTrue((out / "q2_implementation_matrix.csv").exists())

    def test_q2_source_ref_verifier_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "source_refs"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "verify-q2-source-refs",
                    "--output-dir",
                    str(out),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["command"], "verify-q2-source-refs")
            self.assertFalse(manifest["online_checked"])
            self.assertTrue(manifest["all_static_refs_ready"])
            self.assertTrue((out / "q2_source_ref_verification.csv").exists())

    def test_q2_remaining_runs_cli_writes_command_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q2 = root / "q2.csv"
            cert = root / "cert.csv"
            _write_csv(
                q2,
                [
                    {
                        "Method": "CP-SAT",
                        "source_artifact": "runs/structured/table.csv",
                        "result_ready": "True",
                        "compute_ready": "True",
                        "result_blockers": "",
                        "compute_blockers": "",
                    },
                    {
                        "Method": "Target-PRM priority",
                        "source_artifact": "runs/old/scheduling_llm_pilot_table.csv",
                        "result_ready": "False",
                        "compute_ready": "False",
                        "result_blockers": "missing learned search protocol",
                        "compute_blockers": "shared-cache fallback accounting",
                    },
                ],
            )
            _write_csv(cert, [{"Table": "Table 6 clean certification", "source_artifact": "runs/cert", "result_ready": "True"}])
            out = root / "remaining"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "build-q2-remaining-runs",
                    "--output-dir",
                    str(out),
                    "--q2-readiness",
                    str(q2),
                    "--certification-readiness",
                    str(cert),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["remaining_group_count"], 1)
            text = (out / "q2_remaining_runs.md").read_text(encoding="utf-8")
            self.assertIn("Learned-search Q2 rows", text)
            self.assertIn("do not patch old metrics by hand", text)
            self.assertTrue((out / "q2_remaining_runs.csv").exists())

    def test_q4_tau_adapter_contract_cli_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "q4_contract"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "build-q4-tau-adapter-contract",
                    "--output-dir",
                    str(out),
                    "--expected-test-count",
                    "3",
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["expected_test_count"], 3)
            self.assertTrue((out / "q4_tau_adapter_contract.csv").exists())

    def test_q2_run_preflight_cli_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "q2_preflight"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "build-q2-run-preflight",
                    "--output-dir",
                    str(out),
                    "--test-count",
                    "2",
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["test_count"], 2)
            self.assertTrue((out / "q2_run_preflight.csv").exists())

    def test_q2_local_primitive_audit_cli_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "q2_local_primitives"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-q2-local-primitives",
                    "--output-dir",
                    str(out),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(manifest["all_local_primitives_ready"])
            self.assertTrue((out / "q2_local_primitive_audit.csv").exists())


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
