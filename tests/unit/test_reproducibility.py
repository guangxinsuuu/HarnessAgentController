from __future__ import annotations

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.experiments import PRESETS, run_experiment


class ReproducibilityTest(unittest.TestCase):
    def test_same_seed_produces_same_core_artifact_hashes(self) -> None:
        cfg = replace(PRESETS["smoke"], seed=51, train_count=6, test_count=4, trace_node_budget=30, max_pairs_per_problem=20, rank_epochs=3)
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            b = Path(tmp) / "b"
            run_experiment(cfg, a)
            run_experiment(cfg, b)
            left = json.loads((a / "reproducibility_manifest.json").read_text(encoding="utf-8"))
            right = json.loads((b / "reproducibility_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(left["seed"], 51)
        self.assertEqual(left["deterministic_core_hashes"], right["deterministic_core_hashes"])
        self.assertIn("trace_sample.jsonl", left["deterministic_core_hashes"])
        self.assertIn("level1_family_table.csv", left["deterministic_core_hashes"])
        self.assertNotIn("environment.json", left["deterministic_core_hashes"])
