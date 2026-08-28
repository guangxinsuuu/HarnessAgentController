from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cgr_exp.cli import COMMANDS


class CliQ4TauTest(unittest.TestCase):
    def test_q4_tau_commands_are_registered(self) -> None:
        self.assertIn("build-table-q4-tau", COMMANDS)
        self.assertIn("audit-table-q4-tau", COMMANDS)
        self.assertIn("audit-q4-tau-source-evidence", COMMANDS)
        self.assertIn("verify-q4-tau-source-refs", COMMANDS)

    def test_q4_tau_commands_expose_artifact_inputs(self) -> None:
        expected_flags = (
            "--expected-test-count",
            "--native-dir",
            "--best-of-n-dir",
            "--value-guided-dir",
            "--target-prm-dir",
            "--solver-controller-dir",
        )
        for command in ("build-table-q4-tau", "audit-table-q4-tau"):
            result = subprocess.run(
                [sys.executable, "-m", "cgr_exp.cli", command, "--help"],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for flag in expected_flags:
                self.assertIn(flag, result.stdout, command)

    def test_q4_tau_audit_command_writes_pending_manifest_without_native_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-table-q4-tau",
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
            self.assertEqual(manifest["all_rows_result_ready"], False)
            rows = _read_csv(out / "q4_tau_readiness.csv")
            self.assertTrue(rows)
            self.assertTrue(all(row["result_ready"] == "False" for row in rows))
            self.assertTrue(all("artifact missing" in row["result_blockers"] for row in rows))

    def test_q4_tau_source_evidence_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "source_evidence"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-q4-tau-source-evidence",
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
            self.assertEqual(manifest["all_sources_ready"], True)
            self.assertTrue((out / "q4_tau_source_evidence_audit.csv").exists())

    def test_q4_tau_source_ref_verifier_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "source_refs"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "verify-q4-tau-source-refs",
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
            self.assertEqual(manifest["all_static_refs_ready"], True)
            self.assertTrue((out / "q4_tau_source_ref_verification.csv").exists())


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


if __name__ == "__main__":
    unittest.main()
