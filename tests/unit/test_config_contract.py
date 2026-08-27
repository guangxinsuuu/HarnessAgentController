from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cgr_exp.config import config_from_dict, load_and_validate_config
from cgr_exp.experiments import PRESETS


class ConfigContractTest(unittest.TestCase):
    def test_nested_config_maps_to_experiment_config(self) -> None:
        raw = {
            "seed": 23,
            "dataset": {"level": 2, "family": "assignment", "train_instances": 12, "val_instances": 3, "test_instances": 7, "ood_mode": "mixed"},
            "proposer": {"type": "simulator", "k": 5, "useful_action_recall": 0.7, "invalid_action_prob": 0.1, "duplicate_prob": 0.2},
            "verification": {"false_negative_rate": 0.05, "false_positive_rate": 0.01, "uncertain_noise_rate": 0.0},
            "search": {"policy": "cgr", "node_budget": 99, "proposal_call_budget": 88},
            "training": {"regime": "transfer_matrix", "target_adaptation_trajectories": [0, 10, 50]},
            "graph": {"parent_child": True, "sibling_failure": False, "shared_constraint": False, "equivalent_state": False, "tool_observation": False},
            "memory": {"enabled": False, "top_r": 0, "frozen_at_test": True},
        }
        cfg = config_from_dict(raw, PRESETS["smoke"])
        self.assertEqual(cfg.seed, 23)
        self.assertEqual(cfg.train_count, 12)
        self.assertEqual(cfg.test_count, 7)
        self.assertEqual(cfg.action_budget, 5)
        self.assertEqual(cfg.node_budget, 99)
        self.assertEqual(cfg.llm_recall, 0.7)
        self.assertEqual(cfg.false_positive_rate, 0.01)
        self.assertEqual(cfg.fewshot_counts, (10, 50))

    def test_config_validation_rejects_invalid_probability_and_budget(self) -> None:
        with self.assertRaises(ValueError):
            config_from_dict({"proposer": {"useful_action_recall": 1.5}}, PRESETS["smoke"])
        with self.assertRaises(ValueError):
            config_from_dict({"search": {"node_budget": 0}}, PRESETS["smoke"])

    def test_run_experiment_config_fails_before_creating_output_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "bad.json"
            out = Path(tmp) / "out"
            config_path.write_text(json.dumps({"seed": "not-int"}), encoding="utf-8")
            cmd = [sys.executable, "-m", "cgr_exp.run_experiment", "--config", str(config_path), "--out", str(out)]
            proc = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[2], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            self.assertNotEqual(proc.returncode, 0)
            self.assertFalse(out.exists())
            self.assertIn("seed", proc.stderr)

    def test_all_repo_configs_validate(self) -> None:
        for path in sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.json")):
            with self.subTest(path=path.name):
                cfg = load_and_validate_config(path, PRESETS["smoke"])
                self.assertIsInstance(cfg.seed, int)
                self.assertGreater(cfg.node_budget, 0)


if __name__ == "__main__":
    unittest.main()
