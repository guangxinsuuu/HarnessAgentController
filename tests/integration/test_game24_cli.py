from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class Game24CliTest(unittest.TestCase):
    def test_game24_offline_cli_writes_pre_llm_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "game24"
            cmd = [
                sys.executable,
                "-m",
                "cgr_exp.cli",
                "run-game24-offline",
                "--output-dir",
                str(out),
                "--seed",
                "11",
                "--limit",
                "12",
            ]
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            metrics = json.loads((out / "game24_offline_metrics.json").read_text(encoding="utf-8"))
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            with (out / "game24_offline_table.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertFalse(metrics["llm_required"])
            self.assertTrue(metrics["proposal_replay_fixed"])
            self.assertEqual(manifest["command"], "run-game24-offline")
            self.assertIn("game24_split_manifest.json", manifest["produced_artifacts"])
            self.assertIn("game24_trajectory_sample.jsonl", manifest["produced_artifacts"])
            self.assertTrue({"Random", "DFS", "Greedy heuristic", "Oracle", "MLP", "Tree-GNN", "Graph-GNN"}.issubset({row["Policy"] for row in rows}))

    def test_game24_offline_cli_supports_multi_seed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "game24_multi"
            cmd = [
                sys.executable,
                "-m",
                "cgr_exp.cli",
                "run-game24-offline",
                "--output-dir",
                str(out),
                "--seeds",
                "0,1",
                "--limit",
                "12",
            ]
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            with (out / "game24_multi_seed_summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(manifest["seeds"], [0, 1])
            self.assertTrue((out / "seed_0" / "game24_offline_metrics.json").exists())
            self.assertTrue((out / "seed_1" / "game24_offline_metrics.json").exists())
            self.assertTrue({"seed", "Policy", "Success", "Nodes", "Proposal calls"}.issubset(rows[0]))


if __name__ == "__main__":
    unittest.main()
