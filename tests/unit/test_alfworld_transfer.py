from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from cgr_exp.alfworld_transfer import (
    audit_alfworld_dense_labels,
    alfworld_candidate_to_canonical,
    run_alfworld_dense_train,
    run_alfworld_dense_transfer,
    run_alfworld_expert_train,
    run_alfworld_expert_transfer,
    run_alfworld_transfer,
)
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


class FakeExpertAlfworldEnv(FakeAlfworldEnv):
    def reset(self):
        obs, info = super().reset()
        info["extra.expert_plan"] = [["look"]]
        return obs, info

    def step(self, commands):
        obs, score, done, info = super().step(commands)
        admissible = info.get("admissible_commands", [[]])[0]
        if "take apple" in admissible:
            expert = "take apple"
        elif "put apple on table" in admissible:
            expert = "put apple on table"
        else:
            expert = admissible[0] if admissible else "look"
        info["extra.expert_plan"] = [[expert]]
        return obs, score, done, info


class FakeDenseAlfworldEnv:
    def __init__(self):
        self.state = "root"

    def load(self, _gamefile):
        self.state = "root"

    def reset(self):
        self.state = "root"
        return ["You are in a hallway. Your task is to put apple on table."], {
            "admissible_commands": [["look around", "go to kitchen", "take apple"]],
            "objective": ["put apple on table"],
            "extra.gamefile": ["fake-game"],
            "extra.expert_plan": [["go to kitchen"]],
        }

    def step(self, commands):
        command = commands[0] if isinstance(commands, list) else commands
        if self.state == "root":
            if command == "go to kitchen":
                self.state = "kitchen"
                return ["You see an apple and a table."], [0.0], [False], {
                    "admissible_commands": [["take apple", "look around"]],
                    "objective": ["put apple on table"],
                    "extra.gamefile": ["fake-game"],
                    "extra.expert_plan": [["take apple"]],
                }
            if command == "look around":
                return ["You are still in a hallway."], [0.0], [False], {
                    "admissible_commands": [["go to kitchen", "take apple"]],
                    "objective": ["put apple on table"],
                    "extra.gamefile": ["fake-game"],
                    "extra.expert_plan": [["go to kitchen"]],
                }
            return ["You cannot take that."], [0.0], [False], {
                "admissible_commands": [["look around"]],
                "objective": ["put apple on table"],
                "extra.gamefile": ["fake-game"],
                "extra.expert_plan": [["look around"]],
            }
        if self.state == "kitchen":
            if command == "take apple":
                self.state = "holding"
                return ["You hold the apple near the table."], [0.0], [False], {
                    "admissible_commands": [["put apple on table", "look around"]],
                    "objective": ["put apple on table"],
                    "extra.gamefile": ["fake-game"],
                    "extra.expert_plan": [["put apple on table"]],
                }
            return ["Nothing new."], [0.0], [False], {
                "admissible_commands": [["take apple", "look around"]],
                "objective": ["put apple on table"],
                "extra.gamefile": ["fake-game"],
                "extra.expert_plan": [["take apple"]],
            }
        if self.state == "holding" and command == "put apple on table":
            self.state = "done"
            return ["Done."], [1.0], [True], {
                "admissible_commands": [[]],
                "objective": ["put apple on table"],
                "extra.gamefile": ["fake-game"],
                "extra.expert_plan": [[]],
            }
        return ["Nothing happens."], [0.0], [False], {
            "admissible_commands": [["look around"]],
            "objective": ["put apple on table"],
            "extra.gamefile": ["fake-game"],
            "extra.expert_plan": [["look around"]],
        }


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

    def test_adapter_extracts_task_text_from_full_alfworld_observation(self) -> None:
        candidate = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You see a cabinet and a fridge. Your task is to: put a cool tomato in fridge.",
            objective="put a cool tomato in fridge",
            history=["obs:You see a cabinet and a fridge. Your task is to: put a cool tomato in fridge."],
            admissible_commands=["go to cabinet", "go to fridge"],
            command="go to fridge",
            max_steps=10,
        )
        other = alfworld_candidate_to_canonical(
            candidate_id="c1",
            observation="You see a cabinet and a fridge. Your task is to: put a cool tomato in fridge.",
            objective="put a cool tomato in fridge",
            history=["obs:You see a cabinet and a fridge. Your task is to: put a cool tomato in fridge."],
            admissible_commands=["go to cabinet", "go to fridge"],
            command="go to cabinet",
            max_steps=10,
        )
        self.assertEqual(candidate.action.references_observed_entity, 1.0)
        self.assertEqual(other.action.declared_requirement_gain, 0.0)

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

    def test_run_alfworld_expert_transfer_trains_from_frontier_expert_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            manifest = run_alfworld_expert_transfer(
                out,
                train_episodes=4,
                eval_episodes=2,
                max_steps=5,
                epochs=3,
                warmup_random_steps=0,
                perturbation_prob=0.0,
                policies=("Random", "ALFWorld expert-ranker"),
                env_factory=lambda **_kwargs: FakeExpertAlfworldEnv(),
            )
            with (out / "alfworld_expert_transfer_table.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue((out / "alfworld_expert_ranker.json").exists())
        self.assertEqual(manifest["benchmark"], "alfworld_expert_frontier_transfer")
        self.assertEqual(manifest["training_source"], "alfworld_train_expert_plan_frontier_labels")
        self.assertFalse(manifest["llm_rollout_trajectories_used"])
        self.assertFalse(manifest["target_eval_trajectories_used_for_training"])
        self.assertGreater(manifest["train"]["train_pair_count"], 0)
        self.assertEqual({row["policy"] for row in rows}, {"Random", "ALFWorld expert-ranker"})

    def test_run_alfworld_expert_train_writes_reusable_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "train"
            manifest = run_alfworld_expert_train(
                out,
                train_episodes=4,
                max_steps=5,
                epochs=3,
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: FakeExpertAlfworldEnv(),
            )
            checkpoint = out / "alfworld_expert_ranker.json"
            self.assertTrue(checkpoint.exists())
            self.assertTrue((out / "alfworld_expert_training_trace.csv").exists())
            self.assertEqual(manifest["benchmark"], "alfworld_expert_frontier_controller_train")
            self.assertGreater(manifest["train"]["train_pair_count"], 0)

    def test_run_alfworld_expert_transfer_can_load_checkpoint_without_retraining(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ranker.json"
            checkpoint.write_text(GenericHarnessRanker([0.0 for _name in canonical_feature_names()]).to_json(), encoding="utf-8")
            out = Path(tmp) / "eval"
            manifest = run_alfworld_expert_transfer(
                out,
                ranker_checkpoint=checkpoint,
                eval_episodes=2,
                max_steps=5,
                policies=("ALFWorld expert-ranker",),
                env_factory=lambda **_kwargs: FakeExpertAlfworldEnv(),
            )
            self.assertFalse(manifest["train"]["trained_in_this_run"])
            self.assertEqual(manifest["train"]["checkpoint"], str(checkpoint))
            self.assertTrue((out / "alfworld_expert_ranker.json").exists())

    def test_run_alfworld_dense_train_writes_frontier_labels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-train"
            manifest = run_alfworld_dense_train(
                out,
                train_episodes=3,
                max_steps=4,
                epochs=3,
                max_pairs_per_state=8,
                max_frontier_actions=3,
                counterfactual_rollout_steps=4,
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: FakeDenseAlfworldEnv(),
            )
            self.assertTrue((out / "alfworld_dense_ranker.json").exists())
            self.assertTrue((out / "alfworld_dense_action_labels.csv").exists())
            self.assertEqual(manifest["benchmark"], "alfworld_dense_frontier_controller_train")
            self.assertEqual(manifest["train"]["label_source"], "replay_prefix_plus_expert_rollout_cost_proxy")
            self.assertGreater(manifest["train"]["train_pair_count"], 0)
            with (out / "alfworld_dense_action_labels.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)

    def test_audit_alfworld_dense_labels_reports_cost_variation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-audit"
            manifest = audit_alfworld_dense_labels(
                out,
                episodes=2,
                max_steps=4,
                max_pairs_per_state=8,
                max_frontier_actions=3,
                counterfactual_rollout_steps=4,
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: FakeDenseAlfworldEnv(),
            )
            self.assertTrue((out / "alfworld_dense_label_audit.csv").exists())
            self.assertEqual(manifest["benchmark"], "alfworld_dense_frontier_label_audit")
            self.assertGreater(manifest["label_summary"]["action_labels"], 0)
            self.assertGreater(manifest["label_summary"]["states_with_cost_variation"], 0)

    def test_run_alfworld_dense_transfer_can_load_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ranker.json"
            checkpoint.write_text(GenericHarnessRanker([0.0 for _name in canonical_feature_names()]).to_json(), encoding="utf-8")
            out = Path(tmp) / "dense-eval"
            manifest = run_alfworld_dense_transfer(
                out,
                ranker_checkpoint=checkpoint,
                eval_episodes=2,
                max_steps=5,
                policies=("ALFWorld dense-ranker",),
                env_factory=lambda **_kwargs: FakeDenseAlfworldEnv(),
            )
            self.assertFalse(manifest["train"]["trained_in_this_run"])
            self.assertEqual(manifest["train"]["checkpoint"], str(checkpoint))
            self.assertTrue((out / "alfworld_dense_transfer_table.csv").exists())


if __name__ == "__main__":
    unittest.main()
