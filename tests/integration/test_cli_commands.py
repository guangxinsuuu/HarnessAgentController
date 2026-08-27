from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


RUN_BACKED_COMMANDS = {
    "build-tables": "primary_table.csv",
    "evaluate-search": "primary_table.csv",
    "run-certification-corruption": "certification_table.csv",
    "analyze-domain-shift": "domain_shift_table.csv",
    "generate-level1": "level1_family_table.csv",
    "generate-level2": "proposal_table.csv",
    "train-ranker": "ranking_table.csv",
    "adapt-ranker": "transfer_table.csv",
}


class CliCommandsTest(unittest.TestCase):
    def test_recommended_cli_command_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cmd"
            cmd = [sys.executable, "-m", "cgr_exp.cli", "build-tables", "--seed", "3", "--output-dir", str(out)]
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["command"], "build-tables")
            self.assertEqual(manifest["seed"], 3)
            self.assertFalse(manifest["overwrite"])
            self.assertEqual(manifest["status"], "complete")
            self.assertTrue((out / "primary_table.csv").exists())

    def test_recommended_cli_commands_execute_real_smoke_backed_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for command in ("evaluate-search", "run-certification-corruption", "analyze-domain-shift"):
                expected_artifact = RUN_BACKED_COMMANDS[command]
                out = Path(tmp) / command
                cmd = [sys.executable, "-m", "cgr_exp.cli", command, "--seed", "4", "--preset", "smoke", "--output-dir", str(out)]
                subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["command"], command)
                self.assertEqual(manifest["status"], "complete")
                self.assertIn(expected_artifact, manifest["produced_artifacts"])
                self.assertTrue((out / expected_artifact).exists())

    def test_cli_refuses_overwrite_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "cmd"
            out.mkdir()
            (out / "command_manifest.json").write_text("{}", encoding="utf-8")
            cmd = [sys.executable, "-m", "cgr_exp.cli", "analyze-domain-shift", "--output-dir", str(out)]
            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("exists", proc.stderr)


if __name__ == "__main__":
    unittest.main()
