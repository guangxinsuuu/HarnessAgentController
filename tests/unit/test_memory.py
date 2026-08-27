from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cgr_exp.certification.prune import should_prune
from cgr_exp.datasets.level1_assignment import generate_assignment_fixture
from cgr_exp.memory.bank import MemoryBank
from cgr_exp.memory.episodes import SearchEpisode
from cgr_exp.memory.gating import gated_fusion, memory_gate
from cgr_exp.experiments import PRESETS, run_experiment


class EpisodicMemoryTest(unittest.TestCase):
    def test_memory_bank_freeze_prevents_test_leakage(self) -> None:
        bank = MemoryBank()
        bank.add(SearchEpisode(episode_id="train-1", task_family="assignment", utility=1.0, source_split="train"))
        bank.freeze()
        with self.assertRaises(RuntimeError):
            bank.add(SearchEpisode(episode_id="test-1", task_family="assignment", utility=1.0, source_split="test"))
        self.assertEqual([e.episode_id for e in bank.episodes], ["train-1"])

    def test_irrelevant_memory_gate_falls_back_to_base_score(self) -> None:
        base = 3.0
        gate = memory_gate(similarity=0.0, confidence=0.0)
        self.assertEqual(gated_fusion(base, gate, episodic_prior=100.0), base)

    def test_memory_score_does_not_change_pruning(self) -> None:
        task = generate_assignment_fixture(__import__("random").Random(1), "mem-prune", 3, 2)
        node = task.root_node()
        self.assertEqual(should_prune(node, incumbent=0.0, memory_score=-999.0), should_prune(node, incumbent=0.0, memory_score=999.0))

    def test_run_artifact_records_disabled_frozen_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics = run_experiment(PRESETS["smoke"], Path(tmp))
            self.assertFalse(metrics["config"]["memory_configuration"]["enabled"])
            manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("memory_table.csv", manifest["required_artifacts"])
            self.assertTrue((Path(tmp) / "memory_table.csv").exists())


if __name__ == "__main__":
    unittest.main()

