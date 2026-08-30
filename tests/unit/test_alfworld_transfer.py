from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cgr_exp.alfworld_transfer import alfworld_candidate_to_canonical, run_alfworld_transfer
from cgr_exp.harness_contract import canonical_feature_names
from cgr_exp.harness_ranker import GenericHarnessRanker


class FakeAlfworldEnv:
    def __init__(self):
        self.step_idx = 0

    def reset(self):
        self.step_idx = 0
        return ["You are in a kitchen. Your task is to put apple on table."], {
            "admissible_commands": [["look", "go to kitchen"]],
            "objective": ["put apple on table"],
        }

    def step(self, commands):
        command = commands[0]
        self.step_idx += 1
        if self.step_idx == 1:
            return ["You see an apple and a table."], [0.0], [False], {
                "admissible_commands": [["take apple", "look", "go to livingroom"]],
                "objective": ["put apple on table"],
            }
        if self.step_idx == 2 and command == "take apple":
            return ["You hold the apple near the table."], [0.0], [False], {
                "admissible_commands": [["put apple on table", "look"]],
                "objective": ["put apple on table"],
            }
        if command == "put apple on table":
            return ["Done."], [1.0], [True], {"admissible_commands": [[]], "objective": ["put apple on table"]}
        return ["Nothing happens."], [0.0], [False], {"admissible_commands": [["look"]], "objective": ["put apple on table"]}


class AlfworldTransferTest(unittest.TestCase):
    def test_adapter_uses_canonical_features_without_command_string_feature(self) -> None:
        candidate = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You see an apple on a counter.",
            objective="put apple on table",
            history=["obs:You see an apple on a counter."],
            admissible_commands=["take apple", "look"],
            command="take apple",
            max_steps=10,
        )
        self.assertEqual(len(candidate.feature_vector()), len(canonical_feature_names()))
        self.assertEqual(candidate.action.action_is_write, 1.0)
        self.assertEqual(candidate.action.declared_requirement_gain, 1.0)

    def test_run_alfworld_transfer_with_fake_env_and_frozen_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            checkpoint = Path(tmp) / "ranker.json"
            weights = [0.0 for _name in canonical_feature_names()]
            names = canonical_feature_names()
            weights[names.index("declared_requirement_gain")] = 3.0
            weights[names.index("declared_observation_gain")] = 1.0
            weights[names.index("duplicates_previous_action")] = -2.0
            checkpoint.write_text(GenericHarnessRanker(weights).to_json(), encoding="utf-8")
            manifest = run_alfworld_transfer(
                out,
                episodes=2,
                max_steps=5,
                ranker_checkpoint=checkpoint,
                policies=("Random", "Source-trained controller"),
                env_factory=lambda **_kwargs: FakeAlfworldEnv(),
            )
            with (out / "alfworld_transfer_table.csv").open() as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(manifest["benchmark"], "alfworld_admissible_action_transfer")
        self.assertFalse(manifest["alfworld_train_trajectories_used"])
        self.assertFalse(manifest["oracle_labels_in_inference_features"])
        self.assertFalse(manifest["command_string_used_as_learned_feature"])
        self.assertEqual({row["policy"] for row in rows}, {"Random", "Source-trained controller"})


if __name__ == "__main__":
    unittest.main()
