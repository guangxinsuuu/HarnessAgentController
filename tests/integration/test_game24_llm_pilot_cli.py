from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Game24LLMPilotCliTest(unittest.TestCase):
    def test_mock_llm_pilot_cli_writes_shared_cache_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "pilot"
            cmd = [
                sys.executable,
                "-m",
                "cgr_exp.cli",
                "run-game24-llm-pilot",
                "--output-dir",
                str(out),
                "--seed",
                "3",
                "--start",
                "900",
                "--count",
                "2",
                "--proposal-k",
                "4",
                "--node-budget",
                "30",
                "--batch-size",
                "2",
                "--mock",
            ]
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            metrics = json.loads((out / "game24_llm_pilot_metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            with (out / "game24_llm_pilot_table.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertFalse(metrics["live_network_required"])
            self.assertTrue(metrics["proposal_cache"]["shared"])
            self.assertEqual(manifest["command"], "run-game24-llm-pilot")
            self.assertEqual(manifest["batch_size"], 2)
            self.assertTrue({"CoT", "ToT-style", "MLP priority", "Oracle priority"}.issubset({row["Method"] for row in rows}))
            self.assertIn("game24_proposal_cache.jsonl", manifest["produced_artifacts"])


if __name__ == "__main__":
    unittest.main()
