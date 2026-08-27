from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MultiSeedCliTest(unittest.TestCase):
    def test_run_transfer_matrix_executes_multiple_seeds_and_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "matrix"
            cmd = [
                sys.executable,
                "-m",
                "cgr_exp.cli",
                "run-transfer-matrix",
                "--seeds",
                "1,2,3",
                "--preset",
                "smoke",
                "--output-dir",
                str(out),
            ]
            subprocess.run(cmd, check=True, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            manifest = json.loads((out / "command_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["seed_count"], 3)
            self.assertEqual(manifest["command"], "run-transfer-matrix")
            self.assertTrue((out / "multi_seed_summary.csv").exists())
            with (out / "multi_seed_summary.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(rows)
            self.assertTrue(all(int(row["seed_count"]) == 3 for row in rows))
            self.assertTrue((out / "seed_1" / "metrics.json").exists())
            self.assertTrue((out / "seed_2" / "metrics.json").exists())
            self.assertTrue((out / "seed_3" / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
