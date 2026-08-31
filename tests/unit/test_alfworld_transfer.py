from __future__ import annotations

import csv
import json
import random
import tempfile
import unittest
from pathlib import Path

from cgr_exp.alfworld_transfer import (
    ALFWORLD_ADAPTER_VERSION,
    AlfworldDenseActionLabel,
    audit_alfworld_dense_labels,
    alfworld_candidate_to_canonical,
    run_alfworld_dense_train,
    run_alfworld_dense_transfer,
    run_alfworld_expert_train,
    run_alfworld_expert_transfer,
    run_alfworld_transfer,
    _contract_filtered_admissible_commands,
    _command_type,
    _alfworld_env_method_for_policy,
    _alfworld_ranker_weight_diagnostics,
    _alfworld_diagnostic_action_from_info,
    _annotate_alfworld_dense_label_ranks,
    _dense_labels_to_pairs,
    _label_frontier_by_planner_cost,
    _label_frontier_by_walkthrough_suffix,
    _load_alfworld_dense_state_label_cache,
    _load_alfworld_dense_training_cache,
    _load_alfworld_planner_label_cache,
    _alfworld_pairwise_ranker_diagnostics,
    _frontier_sample,
    _gamefile_from_info,
    _planner_training_rollout_command,
    _project_alfworld_ranker_contract_signs,
    _select_command,
    _walkthrough_training_rollout_command,
)
from cgr_exp.harness_contract import canonical_feature_names
from cgr_exp.harness_ranker import GenericHarnessMLPRanker, GenericHarnessRanker


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


class FakePlannerAlfworldEnv:
    def __init__(self):
        self.state = "root"
        self.reset_count = 0

    def load(self, _gamefile):
        self.state = "root"

    def __deepcopy__(self, _memo):
        raise TypeError("fake planner env intentionally disables deepcopy")

    def reset(self):
        self.reset_count += 1
        self.state = "root"
        return {
            "feedback": "You are in a hallway. Your task is to put apple on table.",
            "admissible_commands": ["go to kitchen", "open cabinet"],
            "policy_commands": ["go to kitchen", "take apple", "put apple on table"],
            "won": False,
        }

    def step(self, command):
        if isinstance(command, list):
            command = command[0]
        if self.state == "root" and command == "go to kitchen":
            self.state = "kitchen"
            return {
                "feedback": "You see an apple and a table.",
                "admissible_commands": ["take apple", "open cabinet"],
                "policy_commands": ["take apple", "put apple on table"],
                "won": False,
            }, 0.0, False
        if self.state == "root" and command == "open cabinet":
            self.state = "bad"
            return {
                "feedback": "The cabinet is empty.",
                "admissible_commands": ["go to kitchen"],
                "policy_commands": [],
                "won": False,
            }, 0.0, False
        if self.state == "kitchen" and command == "take apple":
            self.state = "holding"
            return {
                "feedback": "You are holding the apple.",
                "admissible_commands": ["put apple on table"],
                "policy_commands": ["put apple on table"],
                "won": False,
            }, 0.0, False
        if self.state == "holding" and command == "put apple on table":
            self.state = "done"
            return {
                "feedback": "Done.",
                "admissible_commands": [],
                "policy_commands": [],
                "won": True,
            }, 1.0, True
        return {
            "feedback": "Nothing happens.",
            "admissible_commands": [],
            "policy_commands": [],
            "won": False,
        }, 0.0, False


class FakeBadSuffixPlannerAlfworldEnv(FakePlannerAlfworldEnv):
    def step(self, command):
        if isinstance(command, list):
            command = command[0]
        if self.state == "root" and command == "go to kitchen":
            self.state = "kitchen"
            return {
                "feedback": "You see an apple and a table.",
                "admissible_commands": ["look around"],
                "policy_commands": ["look around"],
                "won": False,
            }, 0.0, False
        if self.state == "kitchen" and command == "look around":
            return {
                "feedback": "Still no path to the goal.",
                "admissible_commands": ["look around"],
                "policy_commands": [],
                "won": False,
            }, 0.0, False
        return super().step(command)


class FakeCopyablePlannerAlfworldEnv(FakePlannerAlfworldEnv):
    def copy(self):
        copied = FakeCopyablePlannerAlfworldEnv()
        copied.state = self.state
        copied.reset_count = self.reset_count
        return copied


class FakePlannerBatchAlfworldEnv:
    def __init__(self):
        self.state = "root"

    def load(self, _gamefile):
        self.state = "root"

    def reset(self):
        self.state = "root"
        return ["You are in a hallway. Your task is to put apple on table."], {
            "admissible_commands": [["go to kitchen", "open cabinet"]],
            "objective": ["put apple on table"],
            "extra.gamefile": ["fake-game"],
            "policy_commands": [["go to kitchen", "take apple", "put apple on table"]],
        }

    def step(self, commands):
        command = commands[0] if isinstance(commands, list) else commands
        if self.state == "root" and command == "go to kitchen":
            self.state = "kitchen"
            return ["You see an apple and a table."], [0.0], [False], {
                "admissible_commands": [["take apple", "open cabinet"]],
                "objective": ["put apple on table"],
                "extra.gamefile": ["fake-game"],
                "policy_commands": [["take apple", "put apple on table"]],
            }
        if self.state == "root" and command == "open cabinet":
            self.state = "bad"
            return ["The cabinet is empty."], [0.0], [False], {
                "admissible_commands": [["go to kitchen"]],
                "objective": ["put apple on table"],
                "extra.gamefile": ["fake-game"],
                "policy_commands": [[]],
            }
        if self.state == "kitchen" and command == "take apple":
            self.state = "holding"
            return ["You hold the apple near the table."], [0.0], [False], {
                "admissible_commands": [["put apple on table", "look around"]],
                "objective": ["put apple on table"],
                "extra.gamefile": ["fake-game"],
                "policy_commands": [["put apple on table"]],
            }
        if self.state == "holding" and command == "put apple on table":
            self.state = "done"
            return ["Done."], [1.0], [True], {
                "admissible_commands": [[]],
                "objective": ["put apple on table"],
                "extra.gamefile": ["fake-game"],
                "policy_commands": [[]],
            }
        return ["Nothing happens."], [0.0], [False], {
            "admissible_commands": [["look around"]],
            "objective": ["put apple on table"],
            "extra.gamefile": ["fake-game"],
            "policy_commands": [[]],
        }


class AlfworldTransferTest(unittest.TestCase):
    def test_planner_labels_use_successor_replan_cost(self) -> None:
        labels = _label_frontier_by_planner_cost(
            FakePlannerAlfworldEnv(),
            gamefile="fake-game",
            prefix=[],
            commands=["go to kitchen", "open cabinet"],
            max_rollout_steps=5,
        )
        by_command = {label.command: label for label in labels}
        self.assertEqual(by_command["go to kitchen"].reachable, 1)
        self.assertEqual(by_command["go to kitchen"].cost_to_success, 3)
        self.assertEqual(by_command["open cabinet"].reachable, 0)
        self.assertEqual(by_command["go to kitchen"].label_source, "textworld_pddl_replan_cost_to_go")
        self.assertEqual(by_command["go to kitchen"].label_status, "planner_replanned")
        self.assertEqual(by_command["open cabinet"].label_status, "planner_no_plan")

    def test_planner_labels_require_suffix_rollout_success(self) -> None:
        labels = _label_frontier_by_planner_cost(
            FakeBadSuffixPlannerAlfworldEnv(),
            gamefile="fake-game",
            prefix=[],
            commands=["go to kitchen"],
            max_rollout_steps=5,
        )
        self.assertEqual(labels[0].reachable, 0)
        self.assertEqual(labels[0].label_status, "planner_no_plan")

    def test_dense_pairs_exclude_timeout_censored_labels(self) -> None:
        labels = [
            AlfworldDenseActionLabel(0, 0, "take apple", 2, 1, 0.0, 2, "textworld_pddl_replan_cost_to_go", "planner_replanned"),
            AlfworldDenseActionLabel(0, 0, "open fridge", 5, 1, 0.0, 5, "textworld_pddl_replan_cost_to_go", "planner_replanned"),
            AlfworldDenseActionLabel(0, 0, "go to sink", 99, 0, 0.0, 0, "textworld_pddl_replan_cost_to_go", "planner_timeout"),
        ]
        pairs = _dense_labels_to_pairs(
            labels=labels,
            observation="You see an apple and a table.",
            objective="put apple on table",
            history=["obs:You see an apple and a table."],
            admissible_commands=[label.command for label in labels],
            max_steps=10,
            max_pairs=10,
            rng=random.Random(0),
            episode=0,
            step=0,
        )
        self.assertEqual(len(pairs), 1)

    def test_dense_label_rank_annotation_records_ties_and_censored_labels(self) -> None:
        labels = _annotate_alfworld_dense_label_ranks(
            [
                AlfworldDenseActionLabel(0, 0, "take apple", 2, 1, 0.0, 2, "textworld_pddl_replan_cost_to_go", "planner_replanned"),
                AlfworldDenseActionLabel(0, 0, "go to table", 2, 1, 0.0, 2, "textworld_pddl_replan_cost_to_go", "planner_replanned"),
                AlfworldDenseActionLabel(0, 0, "open fridge", 5, 1, 0.0, 5, "textworld_pddl_replan_cost_to_go", "planner_replanned"),
                AlfworldDenseActionLabel(0, 0, "go to sink", 99, 0, 0.0, 0, "textworld_pddl_replan_cost_to_go", "planner_timeout"),
            ]
        )
        by_command = {label.command: label for label in labels}
        self.assertEqual(by_command["take apple"].oracle_rank, 1)
        self.assertEqual(by_command["go to table"].oracle_rank, 1)
        self.assertEqual(by_command["open fridge"].oracle_rank, 2)
        self.assertEqual(by_command["open fridge"].cost_gap_to_best, 3)
        self.assertEqual(by_command["go to sink"].pairwise_usable, 0)
        self.assertEqual(by_command["go to sink"].oracle_rank, 0)

    def test_planner_labels_use_snapshot_fast_path_when_available(self) -> None:
        env = FakeCopyablePlannerAlfworldEnv()
        labels = _label_frontier_by_planner_cost(
            env,
            gamefile="fake-game",
            prefix=[],
            commands=["go to kitchen", "open cabinet"],
            max_rollout_steps=5,
        )
        by_command = {label.command: label for label in labels}
        self.assertEqual(env.reset_count, 1)
        self.assertEqual(by_command["go to kitchen"].reachable, 1)
        self.assertEqual(by_command["go to kitchen"].cost_to_success, 3)
        self.assertEqual(by_command["open cabinet"].reachable, 0)
        self.assertEqual(by_command["go to kitchen"].label_status, "planner_replanned_snapshot")
        self.assertEqual(by_command["open cabinet"].label_status, "planner_no_plan_snapshot")

    def test_planner_labels_reuse_persistent_candidate_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "planner-labels.jsonl"
            cache: dict[str, object] = {}
            first_env = FakePlannerAlfworldEnv()
            first = _label_frontier_by_planner_cost(
                first_env,
                gamefile="fake-game",
                prefix=[],
                commands=["go to kitchen", "open cabinet"],
                max_rollout_steps=5,
                label_cache=cache,
                cache_path=cache_path,
            )
            loaded_cache = _load_alfworld_planner_label_cache(cache_path)
            second_env = FakePlannerAlfworldEnv()
            second = _label_frontier_by_planner_cost(
                second_env,
                gamefile="fake-game",
                prefix=[],
                commands=["go to kitchen", "open cabinet"],
                max_rollout_steps=5,
                label_cache=loaded_cache,
                cache_path=cache_path,
            )
        self.assertGreater(first_env.reset_count, 0)
        self.assertEqual(second_env.reset_count, 0)
        self.assertEqual([label.cost_to_success for label in second], [label.cost_to_success for label in first])
        self.assertTrue(all(label.label_status.endswith("_cache_hit") for label in second))

    def test_planner_training_rollout_prefers_planner_next_command(self) -> None:
        command = _planner_training_rollout_command(
            FakePlannerAlfworldEnv(),
            gamefile="fake-game",
            prefix=[],
            expert="open cabinet",
            admissible=["go to kitchen", "open cabinet"],
            rng=random.Random(0),
            step=0,
            perturbation_prob=0.0,
            warmup_random_steps=0,
        )
        self.assertEqual(command, "go to kitchen")

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

    def test_low_gain_move_after_move_streak_gets_higher_policy_risk(self) -> None:
        without_streak = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You are holding an apple near a countertop.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a countertop.",
                "act:take apple from countertop",
                "obs:You are holding an apple near a countertop.",
            ],
            admissible_commands=["go to cabinet", "go to table"],
            command="go to cabinet",
            max_steps=20,
        )
        after_streak = alfworld_candidate_to_canonical(
            candidate_id="c1",
            observation="You are holding an apple near a drawer.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a countertop.",
                "act:take apple from countertop",
                "obs:You are holding an apple near a countertop.",
                "act:go to cabinet",
                "obs:You are at cabinet 1.",
                "act:go to drawer",
                "obs:You are at drawer 1.",
            ],
            admissible_commands=["go to cabinet", "go to table"],
            command="go to cabinet",
            max_steps=20,
        )
        self.assertEqual(without_streak.action.declared_requirement_gain, 0.0)
        self.assertEqual(after_streak.action.declared_requirement_gain, 0.0)
        self.assertGreater(after_streak.action.policy_risk, without_streak.action.policy_risk)
        self.assertGreaterEqual(after_streak.action.policy_risk, 0.5)

    def test_ready_goal_placement_has_low_policy_risk(self) -> None:
        placement = alfworld_candidate_to_canonical(
            candidate_id="c0",
            observation="You are holding an apple near a table.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a countertop and a table.",
                "act:take apple from countertop",
                "obs:You are holding an apple near a table.",
            ],
            admissible_commands=["move apple to table", "move apple to countertop"],
            command="move apple to table",
            max_steps=20,
        )
        distractor = alfworld_candidate_to_canonical(
            candidate_id="c1",
            observation="You are holding an apple near a table and a countertop.",
            objective="put apple on table",
            history=[
                "obs:You see an apple on a countertop and a table.",
                "act:take apple from countertop",
                "obs:You are holding an apple near a table and a countertop.",
            ],
            admissible_commands=["move apple to table", "move apple to countertop"],
            command="move apple to countertop",
            max_steps=20,
        )
        self.assertEqual(placement.action.completes_goal_placement, 1.0)
        self.assertLess(placement.action.policy_risk, distractor.action.policy_risk)
        self.assertLessEqual(placement.action.policy_risk, 0.05)

    def test_ranker_weight_diagnostics_flags_contract_sign_violations(self) -> None:
        names = canonical_feature_names()
        weights = [0.0 for _name in names]
        weights[names.index("declared_requirement_gain")] = -0.2
        weights[names.index("policy_risk")] = 0.3
        diagnostics = _alfworld_ranker_weight_diagnostics(GenericHarnessRanker(weights))
        self.assertIn("declared_requirement_gain<0", diagnostics["contract_sign_violations"])
        self.assertIn("policy_risk>0", diagnostics["contract_sign_violations"])
        self.assertEqual(diagnostics["contract_sign_violation_count"], 2)

    def test_project_ranker_contract_signs_removes_direct_semantic_reversals(self) -> None:
        names = canonical_feature_names()
        weights = [0.0 for _name in names]
        weights[names.index("declared_requirement_gain")] = -0.2
        weights[names.index("completes_goal_placement")] = -0.4
        weights[names.index("policy_risk")] = 0.3
        weights[names.index("duplicates_previous_action")] = -0.5
        ranker = GenericHarnessRanker(weights)

        projection = _project_alfworld_ranker_contract_signs(ranker)
        diagnostics = _alfworld_ranker_weight_diagnostics(ranker, tolerance=0.0)

        self.assertEqual(projection["projected_weight_count"], 3)
        self.assertEqual(ranker.weights[names.index("declared_requirement_gain")], 0.0)
        self.assertEqual(ranker.weights[names.index("completes_goal_placement")], 0.0)
        self.assertEqual(ranker.weights[names.index("policy_risk")], 0.0)
        self.assertEqual(ranker.weights[names.index("duplicates_previous_action")], -0.5)
        self.assertEqual(diagnostics["contract_sign_violation_count"], 0)

    def test_pairwise_ranker_diagnostics_reports_error_and_margin(self) -> None:
        ranker = GenericHarnessRanker([1.0, 0.0])
        pairs = [((2.0, 0.0), (1.0, 0.0)), ((0.0, 0.0), (1.0, 0.0))]
        diagnostics = _alfworld_pairwise_ranker_diagnostics(ranker, pairs)
        self.assertEqual(diagnostics["pair_count"], 2)
        self.assertEqual(diagnostics["ranking_accuracy"], 0.5)
        self.assertEqual(diagnostics["ranking_error"], 0.5)
        self.assertEqual(diagnostics["min_margin"], -1.0)
        self.assertEqual(diagnostics["median_margin"], 1.0)

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
            self.assertEqual(manifest["train"]["ranker_model"], "linear")
            self.assertIn("ranker_post_projection_train_diagnostics", manifest["train"])
            with (out / "alfworld_dense_action_labels.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)

    def test_run_alfworld_dense_train_can_use_mlp_ranker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-train"
            manifest = run_alfworld_dense_train(
                out,
                train_episodes=2,
                max_steps=4,
                epochs=2,
                max_pairs_per_state=8,
                max_frontier_actions=3,
                ranker_model="mlp",
                rank_hidden_dim=4,
                rank_lr=0.01,
                counterfactual_rollout_steps=4,
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: FakeDenseAlfworldEnv(),
            )
            checkpoint = out / "alfworld_dense_ranker.json"
            loaded = GenericHarnessMLPRanker.from_json(checkpoint.read_text(encoding="utf-8"))
        self.assertEqual(manifest["train"]["ranker_model"], "mlp")
        self.assertEqual(manifest["train"]["rank_hidden_dim"], 4)
        self.assertEqual(loaded.hidden_dim, 4)

    def test_run_alfworld_dense_transfer_can_load_mlp_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ranker.json"
            checkpoint.write_text(GenericHarnessMLPRanker(len(canonical_feature_names()), hidden_dim=4, seed=0).to_json(), encoding="utf-8")
            out = Path(tmp) / "dense-eval"
            manifest = run_alfworld_dense_transfer(
                out,
                ranker_checkpoint=checkpoint,
                eval_episodes=1,
                max_steps=4,
                policies=("ALFWorld dense-ranker",),
                env_factory=lambda **_kwargs: FakeDenseAlfworldEnv(),
            )
        self.assertFalse(manifest["train"]["trained_in_this_run"])
        self.assertEqual(manifest["train"]["ranker_model"], "mlp")

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

    def test_frontier_sample_keeps_reference_and_diverse_goal_actions(self) -> None:
        sampled = _frontier_sample(
            [
                "look",
                "inventory",
                "go to armchair 1",
                "go to coffeetable 1",
                "take remotecontrol 1 from coffeetable 1",
                "put remotecontrol 1 on armchair 1",
                "help",
            ],
            expert="go to coffeetable 1",
            max_frontier_actions=4,
            rng=random.Random(0),
            observation="You see a coffeetable 1, an armchair 1, and a remotecontrol 1.",
            objective="put some remotecontrol on armchair",
            history=["obs:You see a coffeetable 1, an armchair 1, and a remotecontrol 1."],
            max_steps=40,
        )
        self.assertEqual(sampled[0], "go to coffeetable 1")
        self.assertIn("take remotecontrol 1 from coffeetable 1", sampled)
        self.assertIn("put remotecontrol 1 on armchair 1", sampled)
        self.assertGreaterEqual(len({_command_type(command) for command in sampled}), 2)

    def test_frontier_score_diverse_sample_keeps_score_tail_and_action_types(self) -> None:
        sampled = _frontier_sample(
            [
                "look",
                "inventory",
                "go to armchair 1",
                "go to coffeetable 1",
                "take remotecontrol 1 from coffeetable 1",
                "put remotecontrol 1 on armchair 1",
                "open fridge 1",
                "help",
            ],
            expert="go to coffeetable 1",
            max_frontier_actions=6,
            strategy="score-diverse",
            rng=random.Random(0),
            observation="You see a coffeetable 1, an armchair 1, and a remotecontrol 1.",
            objective="put some remotecontrol on armchair",
            history=["obs:You see a coffeetable 1, an armchair 1, and a remotecontrol 1."],
            max_steps=40,
        )
        self.assertEqual(sampled[0], "go to coffeetable 1")
        self.assertIn("take remotecontrol 1 from coffeetable 1", sampled)
        self.assertIn("help", sampled)
        self.assertGreaterEqual(len({_command_type(command) for command in sampled}), 3)

    def test_frontier_sample_rejects_unknown_strategy(self) -> None:
        with self.assertRaises(ValueError):
            _frontier_sample(
                ["look", "inventory", "go to kitchen"],
                expert=None,
                max_frontier_actions=2,
                strategy="unknown",
                rng=random.Random(0),
            )

    def test_alfworld_planner_diagnostic_uses_expert_plan_path(self) -> None:
        ranker = GenericHarnessRanker([0.0 for _name in canonical_feature_names()])
        command = _select_command(
            "ALFWorld planner diagnostic",
            "You are in a kitchen.",
            "put apple on table",
            ["obs:You are in a kitchen."],
            ["look", "take apple"],
            ranker,
            random.Random(0),
            10,
            expert="take apple",
        )
        self.assertEqual(command, "take apple")
        self.assertEqual(_alfworld_env_method_for_policy("ALFWorld planner diagnostic"), "dagger")

    def test_alfworld_planner_diagnostic_can_use_walkthrough_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            gamefile = Path(tmp) / "game.tw-pddl"
            gamefile.write_text('{"walkthrough": ["go to table", "take apple from table"]}', encoding="utf-8")
            command = _alfworld_diagnostic_action_from_info(
                {},
                gamefile=str(gamefile),
                prefix=["go to table"],
                admissible=["look", "take apple from table"],
                walkthrough_cache={},
            )
        self.assertEqual(command, "take apple from table")

    def test_gamefile_from_info_ignores_none_string_values(self) -> None:
        self.assertIsNone(_gamefile_from_info({"extra.gamefile": [None]}))
        self.assertIsNone(_gamefile_from_info({"extra.gamefile": ["None"]}))

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
        self.assertIn("ranker_contract_sign_violation_count", manifest["train"])
        self.assertIn("ranker_direct_feature_weights", manifest["train"])
        self.assertIn("ranker_contract_sign_tolerance", manifest["train"])
        self.assertIn("ranker_post_projection_train_diagnostics", manifest["train"])

    def test_run_alfworld_dense_train_can_use_planner_label_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-train"
            manifest = run_alfworld_dense_train(
                out,
                train_episodes=1,
                max_steps=4,
                epochs=1,
                max_pairs_per_state=8,
                max_frontier_actions=2,
                counterfactual_rollout_steps=4,
                planner_timeout_seconds=1,
                label_source="planner",
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: FakePlannerBatchAlfworldEnv(),
            )
        self.assertEqual(manifest["training_source"], "alfworld_train_textworld_pddl_planner_frontier_labels")
        self.assertEqual(manifest["label_source"], "textworld_pddl_replan_cost_to_go")
        self.assertTrue(manifest["counterfactual_planner_cost_to_go"])
        self.assertEqual(manifest["train"]["planner_policy_miss_rate"], 0.0)
        self.assertIn("label_status_counts", manifest["train"])
        replan_count = manifest["train"]["label_status_counts"].get("planner_replanned", 0) + manifest["train"]["label_status_counts"].get("planner_replanned_snapshot", 0)
        self.assertGreater(replan_count, 0)
        self.assertGreater(manifest["train"]["planner_path_cost_monotone_checks"], 0)
        self.assertEqual(manifest["train"]["planner_path_cost_monotone_failures"], 0)
        self.assertEqual(manifest["train"]["planner_path_cost_monotone_inconclusive"], 0)
        self.assertEqual(manifest["train"]["planner_path_cost_monotone_failure_rate"], 0.0)
        self.assertIn("pairwise_usable_action_labels", manifest["train"])
        self.assertIn("censored_action_label_rate", manifest["train"])

    def test_run_alfworld_dense_train_can_resume_pair_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-train"
            first = run_alfworld_dense_train(
                out,
                train_episodes=1,
                max_steps=4,
                epochs=1,
                max_pairs_per_state=8,
                max_frontier_actions=2,
                counterfactual_rollout_steps=4,
                planner_timeout_seconds=1,
                label_source="planner",
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: FakePlannerBatchAlfworldEnv(),
            )
            resumed = run_alfworld_dense_train(
                out,
                train_episodes=2,
                max_steps=4,
                epochs=1,
                max_pairs_per_state=8,
                max_frontier_actions=2,
                counterfactual_rollout_steps=4,
                planner_timeout_seconds=1,
                label_source="planner",
                warmup_random_steps=0,
                perturbation_prob=0.0,
                resume=True,
                env_factory=lambda **_kwargs: FakePlannerBatchAlfworldEnv(),
            )
            self.assertTrue((out / "alfworld_dense_pair_cache.jsonl").exists())
            self.assertEqual(resumed["train"]["resumed_completed_episodes"], 1)
            self.assertGreaterEqual(resumed["train"]["train_pair_count"], first["train"]["train_pair_count"])
            self.assertIn("planner_label_cache_loaded_entries", resumed["train"])
            self.assertIn("new_planner_label_cache_hits", resumed["train"])
            self.assertTrue(resumed["resume"])

    def test_run_alfworld_dense_train_writes_state_label_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-train"
            manifest = run_alfworld_dense_train(
                out,
                train_episodes=1,
                max_steps=4,
                epochs=1,
                max_pairs_per_state=8,
                max_frontier_actions=2,
                counterfactual_rollout_steps=4,
                planner_timeout_seconds=1,
                label_source="planner",
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: FakePlannerBatchAlfworldEnv(),
            )
            state_cache = out / "alfworld_dense_state_label_cache.jsonl"
            rows = _load_alfworld_dense_state_label_cache(state_cache)
            self.assertTrue(state_cache.exists())
            self.assertGreater(len(rows), 0)
            self.assertEqual(manifest["train"]["dense_state_label_cache"], str(state_cache))
            self.assertEqual(manifest["train"]["dense_state_label_cache_rows"], len(rows))
            self.assertIn("labels", rows[0])
            self.assertIn("admissible_commands", rows[0])
            self.assertIn("selected_rollout_command", rows[0])
            self.assertIn("selected_rollout_source", rows[0])
            self.assertTrue(all(label["episode"] == rows[0]["episode"] for label in rows[0]["labels"]))
            self.assertTrue(all("oracle_rank" in label for label in rows[0]["labels"]))
            self.assertTrue(all("cost_gap_to_best" in label for label in rows[0]["labels"]))
            action_rows = list(csv.DictReader((out / "alfworld_dense_action_labels.csv").open()))
            self.assertTrue(all("oracle_rank" in row for row in action_rows))
            self.assertTrue(any(int(row["oracle_rank"]) == 1 for row in action_rows))
            self.assertIn("oracle_top_usable_action_labels", manifest["train"])
            self.assertIn("usable_mean_cost_gap_to_best", manifest["train"])

    def test_dense_training_cache_rebuilds_pairs_from_state_labels_when_feature_cache_incompatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-train"
            run_alfworld_dense_train(
                out,
                train_episodes=1,
                max_steps=4,
                epochs=1,
                max_pairs_per_state=8,
                max_frontier_actions=2,
                counterfactual_rollout_steps=4,
                planner_timeout_seconds=1,
                label_source="planner",
                warmup_random_steps=0,
                perturbation_prob=0.0,
                env_factory=lambda **_kwargs: FakePlannerBatchAlfworldEnv(),
            )
            (out / "alfworld_dense_cache_manifest.json").unlink()
            cached = _load_alfworld_dense_training_cache(
                out,
                rebuild_feature_pairs_from_state_cache=True,
                max_pairs_per_state=8,
                seed=0,
            )
        self.assertFalse(cached["cache_manifest_compatible"])
        self.assertGreater(cached["state_label_cache_rows"], 0)
        self.assertGreater(cached["rebuilt_feature_pairs_from_state_cache"], 0)
        self.assertGreater(len(cached["pairs"]), 0)
        self.assertGreater(len(cached["action_label_rows"]), 0)
        self.assertEqual(cached["completed_episodes"], {0})

    def test_dense_training_cache_rejects_feature_vectors_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "dense-train"
            cache.mkdir()
            vector = [0.0 for _name in canonical_feature_names()]
            with (cache / "alfworld_dense_pair_cache.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"positive": vector, "negative": vector}) + "\n")
            with (cache / "alfworld_dense_training_trace.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["episode", "pairs", "action_labels", "steps", "final_score"])
                writer.writeheader()
                writer.writerow({"episode": 0, "pairs": 1, "action_labels": 2, "steps": 1, "final_score": 0.0})
            cached = _load_alfworld_dense_training_cache(cache)
        self.assertFalse(cached["cache_manifest_compatible"])
        self.assertEqual(cached["pairs"], [])
        self.assertEqual(cached["completed_episodes"], set())
        self.assertEqual(cached["skipped_incompatible_pairs"], 1)

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

    def test_audit_alfworld_dense_labels_can_reuse_training_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "dense-train"
            cache.mkdir()
            vector = [0.0 for _name in canonical_feature_names()]
            with (cache / "alfworld_dense_pair_cache.jsonl").open("w", encoding="utf-8") as handle:
                handle.write(json.dumps({"positive": vector, "negative": vector}) + "\n")
            with (cache / "alfworld_dense_training_trace.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["episode", "pairs", "action_labels", "steps", "final_score"])
                writer.writeheader()
                writer.writerow({"episode": 0, "pairs": 1, "action_labels": 2, "steps": 1, "final_score": 0.0})
            with (cache / "alfworld_dense_action_labels.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["episode", "step", "command", "cost_to_success", "reachable", "final_score", "rollout_steps", "label_source", "label_status"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "episode": 0,
                        "step": 0,
                        "command": "take apple",
                        "cost_to_success": 1,
                        "reachable": 1,
                        "final_score": 0.0,
                        "rollout_steps": 1,
                        "label_source": "textworld_pddl_replan_cost_to_go",
                        "label_status": "planner_replanned",
                    }
                )
                writer.writerow(
                    {
                        "episode": 0,
                        "step": 0,
                        "command": "open fridge",
                        "cost_to_success": 4,
                        "reachable": 1,
                        "final_score": 0.0,
                        "rollout_steps": 4,
                        "label_source": "textworld_pddl_replan_cost_to_go",
                        "label_status": "planner_replanned",
                    }
                )
            out = Path(tmp) / "dense-audit"
            manifest = audit_alfworld_dense_labels(out, cache_dir=cache, label_source="planner")
            self.assertEqual(manifest["audit_source"], "training_cache")
            self.assertEqual(manifest["cache_dir"], str(cache))
            self.assertEqual(manifest["label_summary"]["train_pair_count"], 1)
            self.assertEqual(manifest["label_summary"]["cost_variation_rate"], 1.0)
            self.assertIn("ranker_contract_sign_violation_count", manifest["label_summary"])
            self.assertIn("ranker_contract_sign_tolerance", manifest["label_summary"])
            self.assertIn("planner_label_cache_loaded_entries", manifest["label_summary"])
            self.assertIn("new_planner_label_cache_hits", manifest["label_summary"])
            self.assertIn("planner_path_cost_monotone_checks", manifest["label_summary"])
            self.assertIn("planner_path_cost_monotone_inconclusive", manifest["label_summary"])
            self.assertIn("pairwise_usable_action_labels", manifest["label_summary"])
            self.assertIn("usable_cost_variation_rate", manifest["label_summary"])

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

    def test_alfworld_step_trace_records_diagnostic_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "dense-eval"
            result = run_alfworld_dense_transfer(
                out,
                ranker_checkpoint=None,
                train_episodes=1,
                eval_episodes=1,
                max_steps=4,
                epochs=1,
                max_pairs_per_state=8,
                max_frontier_actions=2,
                counterfactual_rollout_steps=4,
                planner_timeout_seconds=1,
                label_source="planner",
                warmup_random_steps=0,
                perturbation_prob=0.0,
                policies=("ALFWorld planner diagnostic",),
                env_factory=lambda **_kwargs: FakePlannerBatchAlfworldEnv(),
            )
            self.assertEqual(result["rows"][0]["success"], 1.0)
            self.assertIn("diagnostic_available_rate", result["rows"][0])
            self.assertEqual(result["rows"][0]["diagnostic_match_rate"], 1.0)
            with (out / "alfworld_dense_transfer_step_trace.csv").open() as handle:
                trace_rows = list(csv.DictReader(handle))
        self.assertGreater(len(trace_rows), 0)
        self.assertEqual(trace_rows[0]["diagnostic_source"], "policy_commands")
        self.assertEqual(trace_rows[0]["matches_diagnostic"], "1")


if __name__ == "__main__":
    unittest.main()
