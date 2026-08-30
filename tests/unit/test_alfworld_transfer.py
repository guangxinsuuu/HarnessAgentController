from __future__ import annotations

import csv
import random
import tempfile
import unittest
from pathlib import Path

from cgr_exp.alfworld_transfer import (
    ALFWORLD_ADAPTER_VERSION,
    audit_alfworld_dense_labels,
    alfworld_candidate_to_canonical,
    run_alfworld_dense_train,
    run_alfworld_dense_transfer,
    run_alfworld_expert_train,
    run_alfworld_expert_transfer,
    run_alfworld_transfer,
    _contract_filtered_admissible_commands,
    _label_frontier_by_walkthrough_suffix,
    _select_command,
    _walkthrough_training_rollout_command,
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


class FakeHelpAlfworldEnv:
    def __init__(self):
        self.step_idx = 0

    def reset(self):
        self.step_idx = 0
        return ["You see an apple and a table. Your task is to: put apple on table."], {
            "admissible_commands": [["help", "take apple"]],
            "objective": ["put apple on table"],
        }

    def step(self, commands):
        self.step_idx += 1
        command = commands[0]
        if command == "take apple":
            return ["You are holding an apple near a table."], [0.0], [False], {
                "admissible_commands": [["help", "put apple on table"]],
                "objective": ["put apple on table"],
            }
        if command == "put apple on table":
            return ["Done."], [1.0], [True], {"admissible_commands": [[]], "objective": ["put apple on table"]}
        return ["Help is not useful."], [0.0], [False], {"admissible_commands": [["help"]], "objective": ["put apple on table"]}


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

    def test_relational_adapter_tracks_observed_and_held_goal_progress(self) -> None:
        before = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You are in a kitchen. There is a table here.",
            objective="put apple on table",
            history=["obs:You are in a kitchen. Your task is to: put apple on table."],
            admissible_commands=["take apple", "look"],
            command="take apple",
            max_steps=10,
        )
        after = alfworld_candidate_to_canonical(
            candidate_id="c1",
            observation="You are holding an apple near a table.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on the counter. Your task is to: put apple on table.",
                "act:take apple",
                "obs:You are holding an apple near a table.",
            ],
            admissible_commands=["put apple on table", "look"],
            command="put apple on table",
            max_steps=10,
        )
        self.assertEqual(before.action.precondition_satisfied, 0.0)
        self.assertGreater(before.action.policy_risk, after.action.policy_risk)
        self.assertEqual(after.action.precondition_satisfied, 1.0)
        self.assertEqual(after.action.declared_requirement_gain, 1.0)
        self.assertGreater(after.state.verified_progress_frac, before.state.verified_progress_frac)

    def test_move_object_to_receptacle_counts_as_placement(self) -> None:
        candidate = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You are holding an apple near a table.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a table.",
                "act:take apple",
                "obs:You are holding an apple near a table.",
            ],
            admissible_commands=["move apple to table", "look"],
            command="move apple to table",
            max_steps=10,
        )
        self.assertEqual(candidate.action.action_is_write, 1.0)
        self.assertEqual(candidate.action.declared_requirement_gain, 1.0)
        self.assertEqual(candidate.action.precondition_satisfied, 1.0)
        self.assertEqual(candidate.action.references_goal_object, 1.0)
        self.assertEqual(candidate.action.references_goal_receptacle, 1.0)
        self.assertEqual(candidate.action.goal_object_held, 1.0)
        self.assertEqual(candidate.action.goal_receptacle_observed, 1.0)
        self.assertEqual(candidate.action.placement_ready, 1.0)
        self.assertEqual(candidate.action.completes_goal_placement, 1.0)

    def test_navigation_to_goal_receptacle_gets_partial_requirement_gain_when_holding_object(self) -> None:
        candidate = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You are holding an apple near a countertop and a table.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a countertop and a table.",
                "act:take apple from countertop",
                "obs:You are holding an apple near a countertop and a table.",
            ],
            admissible_commands=["go to table", "go to countertop"],
            command="go to table",
            max_steps=10,
        )
        self.assertEqual(candidate.action.references_goal_receptacle, 1.0)
        self.assertEqual(candidate.action.goal_object_held, 1.0)
        self.assertEqual(candidate.action.declared_requirement_gain, 0.5)

    def test_repeated_goal_navigation_is_not_noop_after_object_is_held(self) -> None:
        candidate = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You are holding an apple near a countertop.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a countertop and a table.",
                "act:go to table",
                "obs:You are at the table.",
                "act:go to countertop",
                "obs:You see an apple on a countertop.",
                "act:take apple from countertop",
                "obs:You are holding an apple near a countertop.",
            ],
            admissible_commands=["go to table", "go to countertop"],
            command="go to table",
            max_steps=10,
        )
        self.assertEqual(candidate.action.declared_requirement_gain, 0.5)

        command = _select_command(
            "ALFWorld dense-ranker",
            observation="You are holding an apple near a countertop.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a countertop and a table.",
                "act:go to table",
                "obs:You are at the table.",
                "act:go to countertop",
                "obs:You see an apple on a countertop.",
                "act:take apple from countertop",
                "obs:You are holding an apple near a countertop.",
            ],
            admissible=["go to table", "go to countertop"],
            ranker=GenericHarnessRanker([0.0 if name != "declared_requirement_gain" else 1.0 for name in canonical_feature_names()]),
            rng=random.Random(0),
            max_steps=10,
        )
        self.assertEqual(command, "go to table")

    def test_move_to_wrong_receptacle_does_not_count_as_goal_placement(self) -> None:
        candidate = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You are holding an apple near a table and a countertop.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a countertop and a table.",
                "act:take apple from countertop",
                "obs:You are holding an apple near a table and a countertop.",
            ],
            admissible_commands=["move apple to countertop", "move apple to table"],
            command="move apple to countertop",
            max_steps=10,
        )
        self.assertEqual(candidate.action.references_goal_object, 1.0)
        self.assertEqual(candidate.action.references_goal_receptacle, 0.0)
        self.assertEqual(candidate.action.declared_requirement_gain, 0.0)
        self.assertEqual(candidate.action.completes_goal_placement, 0.0)
        self.assertGreaterEqual(candidate.action.policy_risk, 0.7)

    def test_contract_filter_drops_zero_gain_write_but_keeps_exploration_move(self) -> None:
        filtered = _contract_filtered_admissible_commands(
            ["take creditcard from armchair", "go to cabinet", "go to table"],
            observation="You are holding an apple near an armchair and a table.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on an armchair and a table.",
                "act:take apple from armchair",
                "obs:You are holding an apple near an armchair and a table.",
            ],
            max_steps=10,
        )
        self.assertNotIn("take creditcard from armchair", filtered)
        self.assertIn("go to cabinet", filtered)
        self.assertIn("go to table", filtered)

    def test_contract_filter_drops_numbered_distractor_write_when_goal_read_exists(self) -> None:
        filtered = _contract_filtered_admissible_commands(
            ["take creditcard 1 from armchair 1", "examine armchair 1", "go to coffeetable 1"],
            observation="You are at armchair 1. On the armchair 1, you see creditcard 1.",
            objective="put some remotecontrol on armchair",
            history=[
                "obs:You are in a room. Your task is to: put some remotecontrol on armchair.",
                "act:go to armchair 1",
                "obs:You are at armchair 1. On the armchair 1, you see creditcard 1.",
            ],
            max_steps=20,
        )
        self.assertNotIn("take creditcard 1 from armchair 1", filtered)
        self.assertIn("examine armchair 1", filtered)
        self.assertIn("go to coffeetable 1", filtered)

    def test_relational_adapter_matches_transform_subgoal(self) -> None:
        related = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You are holding a tomato near a fridge.",
            objective="put a cool tomato in fridge",
            history=[
                "obs:You see a tomato and a fridge.",
                "act:take tomato",
                "obs:You are holding a tomato near a fridge.",
            ],
            admissible_commands=["cool tomato with fridge", "take plate"],
            command="cool tomato with fridge",
            max_steps=10,
        )
        unrelated = alfworld_candidate_to_canonical(
            candidate_id="c1",
            observation="You are holding a tomato near a fridge.",
            objective="put a cool tomato in fridge",
            history=[
                "obs:You see a tomato and a fridge.",
                "act:take tomato",
                "obs:You are holding a tomato near a fridge.",
            ],
            admissible_commands=["cool tomato with fridge", "take plate"],
            command="take plate",
            max_steps=10,
        )
        self.assertEqual(related.action.declared_requirement_gain, 1.0)
        self.assertEqual(unrelated.action.declared_requirement_gain, 0.0)
        self.assertEqual(related.action.precondition_satisfied, 1.0)

    def test_help_command_is_not_treated_as_write_progress(self) -> None:
        candidate = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You see an apple and a table.",
            objective="put apple on table",
            history=["obs:You see an apple and a table. Your task is to: put apple on table."],
            admissible_commands=["help", "take apple"],
            command="help",
            max_steps=10,
        )
        self.assertEqual(candidate.action.action_is_write, 0.0)
        self.assertEqual(candidate.action.action_is_read, 0.0)
        self.assertEqual(candidate.action.declared_requirement_gain, 0.0)
        self.assertEqual(candidate.action.declared_observation_gain, 0.0)
        self.assertEqual(candidate.action.precondition_satisfied, 0.0)
        self.assertGreaterEqual(candidate.action.policy_risk, 0.9)

    def test_ranked_policy_skips_immediate_repeated_noop(self) -> None:
        names = canonical_feature_names()
        weights = [0.0 for _name in names]
        weights[names.index("action_is_read")] = 5.0
        weights[names.index("declared_requirement_gain")] = 4.8
        ranker = GenericHarnessRanker(weights)
        command = _select_command(
            "ALFWorld dense-ranker",
            observation="You see an apple and a table.",
            objective="put apple on table",
            history=["obs:You see an apple and a table.", "act:inventory", "obs:You are carrying nothing."],
            admissible=["inventory", "take apple"],
            ranker=ranker,
            rng=random.Random(0),
            max_steps=10,
        )
        self.assertEqual(command, "take apple")

    def test_ranked_policy_skips_recent_read_noop_cycle(self) -> None:
        names = canonical_feature_names()
        weights = [0.0 for _name in names]
        weights[names.index("action_is_read")] = 5.0
        weights[names.index("declared_requirement_gain")] = 4.8
        ranker = GenericHarnessRanker(weights)
        command = _select_command(
            "ALFWorld dense-ranker",
            observation="You see an apple and a table.",
            objective="put apple on table",
            history=[
                "obs:You see an apple and a table.",
                "act:inventory",
                "obs:You are carrying nothing.",
                "act:look",
                "obs:You see an apple and a table.",
            ],
            admissible=["inventory", "look", "take apple"],
            ranker=ranker,
            rng=random.Random(0),
            max_steps=10,
        )
        self.assertEqual(command, "take apple")

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
            with (out / "alfworld_transfer_step_trace.csv").open() as handle:
                trace_rows = list(csv.DictReader(handle))
        self.assertEqual(manifest["benchmark"], "alfworld_admissible_action_transfer")
        self.assertFalse(manifest["alfworld_train_trajectories_used"])
        self.assertFalse(manifest["oracle_labels_in_inference_features"])
        self.assertFalse(manifest["command_string_used_as_learned_feature"])
        self.assertEqual(manifest["alfworld_adapter_version"], ALFWORLD_ADAPTER_VERSION)
        self.assertTrue(manifest["relational_goal_action_features"])
        self.assertEqual({row["policy"] for row in rows}, {"Random", "Source-trained controller"})
        self.assertGreater(len(trace_rows), 0)
        self.assertIn("declared_requirement_gain", trace_rows[0])
        self.assertIn("candidate_score_summary", trace_rows[0])

    def test_transfer_filters_help_from_shared_frontier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            checkpoint = Path(tmp) / "ranker.json"
            weights = [0.0 for _name in canonical_feature_names()]
            names = canonical_feature_names()
            weights[names.index("action_is_write")] = 2.0
            weights[names.index("declared_requirement_gain")] = 3.0
            checkpoint.write_text(GenericHarnessRanker(weights).to_json(), encoding="utf-8")
            manifest = run_alfworld_transfer(
                out,
                episodes=1,
                max_steps=4,
                ranker_checkpoint=checkpoint,
                policies=("Source-trained controller",),
                env_factory=lambda **_kwargs: FakeHelpAlfworldEnv(),
            )
            with (out / "alfworld_transfer_step_trace.csv").open() as handle:
                trace_rows = list(csv.DictReader(handle))
        self.assertTrue(manifest["meta_commands_filtered_from_frontier"])
        self.assertNotIn("help", {row["command"] for row in trace_rows})
        self.assertEqual(trace_rows[-1]["success_after_step"], "1")

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
                policies=("Random", "ALFWorld expert-ranker", "ALFWorld expert-plan diagnostic"),
                env_factory=lambda **_kwargs: FakeExpertAlfworldEnv(),
            )
            with (out / "alfworld_expert_transfer_table.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue((out / "alfworld_expert_transfer_step_trace.csv").exists())
            self.assertTrue((out / "alfworld_expert_ranker.json").exists())
        self.assertEqual(manifest["benchmark"], "alfworld_expert_frontier_transfer")
        self.assertEqual(manifest["training_source"], "alfworld_train_expert_plan_frontier_labels")
        self.assertFalse(manifest["llm_rollout_trajectories_used"])
        self.assertFalse(manifest["target_eval_trajectories_used_for_training"])
        self.assertGreater(manifest["train"]["train_pair_count"], 0)
        self.assertEqual({row["policy"] for row in rows}, {"Random", "ALFWorld expert-ranker", "ALFWorld expert-plan diagnostic"})

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

    def test_walkthrough_suffix_labels_rank_future_plan_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gamefile = Path(tmp) / "game.tw-pddl"
            gamefile.write_text(
                '{"walkthrough": ["go to table", "take apple from table", "go to sink", "clean apple with sink", "go to counter"]}',
                encoding="utf-8",
            )
            labels = _label_frontier_by_walkthrough_suffix(
                gamefile=str(gamefile),
                prefix=["go to table"],
                commands=["go to sink", "take apple from table", "open cabinet"],
                max_rollout_steps=5,
                walkthrough_cache={},
            )
        by_command = {label.command: label for label in labels}
        self.assertEqual(by_command["take apple from table"].reachable, 1)
        self.assertEqual(by_command["go to sink"].reachable, 1)
        self.assertLess(by_command["take apple from table"].cost_to_success, by_command["go to sink"].cost_to_success)
        self.assertEqual(by_command["open cabinet"].reachable, 0)
        self.assertEqual(by_command["take apple from table"].label_source, "walkthrough_suffix_cost_proxy")

    def test_walkthrough_training_rollout_prefers_walkthrough_over_env_expert(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gamefile = Path(tmp) / "game.tw-pddl"
            gamefile.write_text('{"walkthrough": ["go to table", "take apple from table"]}', encoding="utf-8")
            command, used_walkthrough = _walkthrough_training_rollout_command(
                gamefile=str(gamefile),
                prefix=[],
                expert="look",
                admissible=["look", "go to table"],
                rng=random.Random(0),
                step=0,
                perturbation_prob=0.0,
                warmup_random_steps=0,
                walkthrough_cache={},
            )
        self.assertEqual(command, "go to table")
        self.assertTrue(used_walkthrough)

    def test_run_alfworld_dense_train_can_use_walkthrough_label_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-train"
            gamefile = Path(tmp) / "fake-game.tw-pddl"
            gamefile.write_text(
                '{"walkthrough": ["go to kitchen", "take apple", "put apple on table"]}',
                encoding="utf-8",
            )

            class WalkthroughDenseEnv(FakeDenseAlfworldEnv):
                def reset(self):
                    obs, info = super().reset()
                    info["extra.gamefile"] = [str(gamefile)]
                    return obs, info

                def step(self, commands):
                    obs, score, done, info = super().step(commands)
                    info["extra.gamefile"] = [str(gamefile)]
                    return obs, score, done, info

            manifest = run_alfworld_dense_train(
                out,
                train_episodes=1,
                max_steps=4,
                epochs=1,
                max_pairs_per_state=8,
                max_frontier_actions=3,
                counterfactual_rollout_steps=4,
                label_source="walkthrough",
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: WalkthroughDenseEnv(),
            )
        self.assertEqual(manifest["training_source"], "alfworld_train_walkthrough_suffix_frontier_labels")
        self.assertEqual(manifest["train"]["label_source"], "walkthrough_suffix_cost_proxy")
        self.assertTrue(manifest["train"]["walkthrough_suffix_cost_proxy"])
        self.assertLess(manifest["train"]["walkthrough_policy_miss_rate"], 1.0)

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
            self.assertTrue((out / "alfworld_dense_transfer_step_trace.csv").exists())


if __name__ == "__main__":
    unittest.main()
