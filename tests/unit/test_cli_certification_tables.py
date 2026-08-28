from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cgr_exp.cli import COMMANDS


class CliCertificationTablesTest(unittest.TestCase):
    def test_audit_certification_tables_command_is_registered(self) -> None:
        self.assertIn("audit-certification-tables", COMMANDS)

    def test_audit_certification_tables_exposes_artifact_inputs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "cgr_exp.cli", "audit-certification-tables", "--help"],
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for flag in ("--certification-dir", "--authority-dir", "--expected-test-count"):
            self.assertIn(flag, result.stdout)

    def test_audit_certification_tables_cli_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "audit"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "audit-certification-tables",
                    "--output-dir",
                    str(out),
                    "--expected-test-count",
                    "2",
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["command"], "audit-certification-tables")
            self.assertEqual(manifest["all_rows_ready"], False)
            self.assertTrue((out / "certification_tables_readiness.csv").exists())


if __name__ == "__main__":
    unittest.main()
