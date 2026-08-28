from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cgr_exp.cli import COMMANDS


class CliExperimentStatusReportTest(unittest.TestCase):
    def test_build_current_experiment_status_command_is_registered(self) -> None:
        self.assertIn("build-current-experiment-status", COMMANDS)

    def test_build_current_experiment_status_cli_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            q2 = root / "q2.csv"
            q4 = root / "q4.csv"
            cert = root / "cert.csv"
            matrix = root / "matrix.csv"
            _write_csv(q2, [{"Method": "CP-SAT", "result_ready": "True", "result_blockers": ""}])
            _write_csv(q4, [{"Method": "Native agent rerun", "result_ready": "False", "result_blockers": "pending"}])
            _write_csv(cert, [{"Table": "Table 6 clean certification", "result_ready": "False", "result_blockers": "artifact missing"}])
            _write_csv(matrix, [{"Method": "CP-SAT", "Local code present": "True", "Implemented scope": "structured solver"}])
            out = root / "out"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cgr_exp.cli",
                    "build-current-experiment-status",
                    "--output-dir",
                    str(out),
                    "--q2-readiness",
                    str(q2),
                    "--q4-readiness",
                    str(q4),
                    "--certification-readiness",
                    str(cert),
                    "--q2-implementation-matrix",
                    str(matrix),
                    "--overwrite",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((out / "current_experiment_status.md").exists())
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["command"], "build-current-experiment-status")


def _write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
