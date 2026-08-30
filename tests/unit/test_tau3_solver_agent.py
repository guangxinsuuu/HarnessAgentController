from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from cgr_exp.harness_contract import canonical_contract_manifest, canonical_feature_names
from cgr_exp.harness_ranker import GenericHarnessRanker
from tau2.data_model.message import AssistantMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.environment.tool import as_tool

from cgr_exp.tau3_solver_agent import Tau3SolverControllerAgent


class Tau3SolverAgentTest(unittest.TestCase):
    def test_selects_non_repeated_tool_candidate_and_accounts_all_sampling_cost(self) -> None:
        calls = [
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": "A"})],
                cost=0.01,
                usage={"prompt_tokens": 10, "completion_tokens": 1},
            ),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="update_order", arguments={"order_id": "A"})],
                cost=0.02,
                usage={"prompt_tokens": 11, "completion_tokens": 2},
            ),
        ]
        agent = Tau3SolverControllerAgent([as_tool(_get_order), as_tool(_update_order)], "policy", "mock", {"temperature": 0}, candidate_count=2)
        state = agent.get_init_state()
        with patch("cgr_exp.tau3_solver_agent.generate", side_effect=calls):
            selected, state = agent.generate_next_message(UserMessage(role="user", content="change order A"), state)
        self.assertEqual(selected.tool_calls[0].name, "update_order")
        self.assertAlmostEqual(selected.cost, 0.03)
        self.assertEqual(selected.usage["prompt_tokens"], 21)
        self.assertEqual(selected.usage["completion_tokens"], 3)
        self.assertEqual(len(state.messages), 2)

    def test_keeps_greedy_user_message_when_greedy_does_not_call_tool(self) -> None:
        calls = [
            AssistantMessage(role="assistant", content="Could you confirm the order id?", cost=0.01, usage={"prompt_tokens": 10, "completion_tokens": 4}),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="update_order", arguments={"order_id": "A"})],
                cost=0.02,
                usage={"prompt_tokens": 11, "completion_tokens": 2},
            ),
        ]
        agent = Tau3SolverControllerAgent([as_tool(_get_order), as_tool(_update_order)], "policy", "mock", {"temperature": 0}, candidate_count=2)
        state = agent.get_init_state()
        with patch("cgr_exp.tau3_solver_agent.generate", side_effect=calls):
            selected, _state = agent.generate_next_message(UserMessage(role="user", content="hello"), state)
        self.assertEqual(selected.content, "Could you confirm the order id?")
        self.assertIsNone(selected.tool_calls)
        self.assertAlmostEqual(selected.cost, 0.03)

    def test_checkpoint_ranker_can_override_heuristic_candidate_order(self) -> None:
        calls = [
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="update_order", arguments={"order_id": "A"})],
                cost=0.01,
                usage={"prompt_tokens": 10, "completion_tokens": 1},
            ),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": "A"})],
                cost=0.02,
                usage={"prompt_tokens": 11, "completion_tokens": 2},
            ),
        ]
        names = canonical_feature_names()
        weights = [0.0 for _name in names]
        weights[names.index("action_is_read")] = 5.0
        weights[names.index("action_is_write")] = -5.0
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ranker.json"
            trace = Path(tmp) / "trace.jsonl"
            checkpoint.write_text(GenericHarnessRanker(weights).to_json(), encoding="utf-8")
            agent = Tau3SolverControllerAgent(
                [as_tool(_get_order), as_tool(_update_order)],
                "policy",
                "mock",
                {"temperature": 0},
                candidate_count=2,
                ranker_checkpoint=str(checkpoint),
                controller_trace_path=str(trace),
            )
            state = agent.get_init_state()
            with patch("cgr_exp.tau3_solver_agent.generate", side_effect=calls):
                selected, _state = agent.generate_next_message(UserMessage(role="user", content="change order A"), state)
            trace_text = trace.read_text(encoding="utf-8")
            trace_row = json.loads(trace_text)
        contract = canonical_contract_manifest()
        self.assertEqual(selected.tool_calls[0].name, "get_order")
        self.assertEqual(trace_row["scoring_mode"], "canonical_ranker")
        self.assertEqual(trace_row["canonical_contract"], contract["contract"])
        self.assertEqual(trace_row["canonical_feature_schema_hash"], contract["feature_schema_hash"])
        self.assertEqual(trace_row["canonical_feature_count"], len(canonical_feature_names()))
        self.assertTrue(trace_row["ranker_checkpoint_loaded"])
        self.assertFalse(trace_row["oracle_visible_to_controller"])
        self.assertFalse(trace_row["tool_names_used_as_features"])
        self.assertEqual(trace_row["tool_name_use_policy"], "coarse_action_type_normalization_only")
        self.assertAlmostEqual(trace_row["canonical_mean_known_mask_ratio"], 1.0)
        self.assertEqual(trace_row["canonical_per_feature_availability"]["verified_progress_known_mask"], 1.0)
        self.assertEqual(trace_row["canonical_per_feature_availability"]["remaining_requirement_known_mask"], 1.0)
        self.assertEqual(trace_row["canonical_per_feature_availability"]["observed_fact_known_mask"], 1.0)
        self.assertEqual(trace_row["canonical_per_feature_availability"]["constraint_pressure_known_mask"], 1.0)
        self.assertIn("canonical_progress_predicates", trace_row)
        self.assertIn("required_read_completed", trace_row["canonical_progress_predicates"])
        self.assertAlmostEqual(trace_row["candidate_scores"][0]["known_mask_ratio"], 1.0)
        self.assertNotIn("get_order", trace_text)
        self.assertNotIn("update_order", trace_text)

    def test_canonical_ranker_tie_break_does_not_use_raw_tool_names(self) -> None:
        calls = [
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="aaa_read_order", arguments={"order_id": "A"})],
                cost=0.01,
                usage={"prompt_tokens": 10, "completion_tokens": 1},
            ),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="zzz_read_order", arguments={"order_id": "A"})],
                cost=0.02,
                usage={"prompt_tokens": 11, "completion_tokens": 2},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ranker.json"
            trace = Path(tmp) / "trace.jsonl"
            checkpoint.write_text(GenericHarnessRanker([0.0] * len(canonical_feature_names())).to_json(), encoding="utf-8")
            agent = Tau3SolverControllerAgent(
                [as_tool(_aaa_read_order), as_tool(_zzz_read_order)],
                "policy",
                "mock",
                {"temperature": 0},
                candidate_count=2,
                ranker_checkpoint=str(checkpoint),
                controller_trace_path=str(trace),
            )
            state = agent.get_init_state()
            with patch("cgr_exp.tau3_solver_agent.generate", side_effect=calls):
                selected, _state = agent.generate_next_message(UserMessage(role="user", content="read order A"), state)
            trace_text = trace.read_text(encoding="utf-8")
            trace_row = json.loads(trace_text)
        self.assertEqual(selected.tool_calls[0].name, "aaa_read_order")
        self.assertEqual(trace_row["selected_index"], 0)
        self.assertEqual(trace_row["tool_name_use_policy"], "coarse_action_type_normalization_only")
        self.assertAlmostEqual(trace_row["canonical_mean_known_mask_ratio"], 1.0)
        self.assertNotIn("aaa_read_order", trace_text)
        self.assertNotIn("zzz_read_order", trace_text)

    def test_bad_checkpoint_error_names_checkpoint_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "bad_ranker.json"
            checkpoint.write_text(json.dumps({"weights": [0.0], "bias": 0.0}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "bad_ranker.json"):
                Tau3SolverControllerAgent(
                    [as_tool(_get_order)],
                    "policy",
                    "mock",
                    {"temperature": 0},
                    ranker_checkpoint=str(checkpoint),
                )

    def test_parallel_candidate_sampling_preserves_index_order_and_args(self) -> None:
        seen: list[tuple[int, float]] = []

        def fake_generate(**kwargs):
            seen.append((kwargs["seed"], kwargs["temperature"]))
            seed = kwargs["seed"]
            name = "get_order" if seed == 7 else "update_order"
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name=name, arguments={"order_id": str(seed)})],
                cost=0.01,
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        agent = Tau3SolverControllerAgent(
            [as_tool(_get_order), as_tool(_update_order)],
            "policy",
            "mock",
            {"temperature": 0, "seed": 7},
            candidate_count=3,
            candidate_workers=3,
        )
        state = agent.get_init_state()
        with patch("cgr_exp.tau3_solver_agent.generate", side_effect=fake_generate):
            candidates = agent._sample_candidates(state)
        self.assertEqual([candidate.tool_calls[0].arguments["order_id"] for candidate in candidates], ["7", "8", "9"])
        self.assertEqual(sorted(seen), [(7, 0), (8, 0.2), (9, 0.2)])

    def test_candidate_diversity_adds_intent_hints_without_changing_index_order(self) -> None:
        seen: list[tuple[int, list[str]]] = []

        def fake_generate(**kwargs):
            system_text = [message.content for message in kwargs["messages"] if isinstance(message, SystemMessage)]
            seen.append((kwargs["seed"], system_text))
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": str(kwargs["seed"])})],
                cost=0.01,
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        agent = Tau3SolverControllerAgent(
            [as_tool(_get_order)],
            "policy",
            "mock",
            {"temperature": 0, "seed": 7},
            candidate_count=4,
            candidate_workers=1,
            enable_candidate_diversity=True,
        )
        state = agent.get_init_state()
        with patch("cgr_exp.tau3_solver_agent.generate", side_effect=fake_generate):
            candidates = agent._sample_candidates(state)
        self.assertEqual([candidate.tool_calls[0].arguments["order_id"] for candidate in candidates], ["7", "8", "9", "10"])
        self.assertEqual(agent._last_candidate_generation_intents, ["greedy", "prefer_read", "prefer_write", "prefer_ask_or_respond"])
        self.assertEqual(seen[0][0], 7)
        self.assertEqual(len(seen[0][1]), 1)
        self.assertIn("information-gathering", " ".join(seen[1][1]))
        self.assertIn("state-changing", " ".join(seen[2][1]))
        self.assertIn("clarification", " ".join(seen[3][1]))

    def test_adaptive_search_uses_min_candidates_for_low_risk_state(self) -> None:
        seen: list[int] = []

        def fake_generate(**kwargs):
            seen.append(kwargs["seed"])
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": str(kwargs["seed"])})],
                cost=0.01,
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        agent = Tau3SolverControllerAgent(
            [as_tool(_get_order)],
            "policy",
            "mock",
            {"temperature": 0, "seed": 7},
            candidate_count=4,
            candidate_workers=4,
            adaptive_search=True,
            min_candidate_count=1,
        )
        state = agent.get_init_state(
            [
                ToolMessage(role="tool", id="call_1", content="ORD123 C456 SKU789 ZIP100 INV200"),
                ToolMessage(role="tool", id="call_2", content="SHIP300 PAY400 ITEM500 CARD600"),
                ToolMessage(role="tool", id="call_3", content="CASE700 NOTE800"),
            ]
        )
        with patch("cgr_exp.tau3_solver_agent.generate", side_effect=fake_generate):
            candidates = agent._sample_candidates(state)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(seen, [7])
        self.assertEqual(agent._last_search_budget, 1)
        self.assertEqual(agent._last_search_reason, "adaptive_low_risk")

    def test_adaptive_search_uses_full_candidates_after_failure_and_logs_reason(self) -> None:
        def fake_generate(**kwargs):
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": str(kwargs["seed"])})],
                cost=0.01,
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            agent = Tau3SolverControllerAgent(
                [as_tool(_get_order)],
                "policy",
                "mock",
                {"temperature": 0, "seed": 7},
                candidate_count=4,
                candidate_workers=4,
                adaptive_search=True,
                min_candidate_count=1,
                controller_trace_path=str(trace),
            )
            state = agent.get_init_state(
                [
                    UserMessage(role="user", content="order ORD123"),
                    ToolMessage(role="tool", id="call_1", content="ERROR not found", error=True),
                ]
            )
            with patch("cgr_exp.tau3_solver_agent.generate", side_effect=fake_generate):
                selected, _state = agent.generate_next_message(UserMessage(role="user", content="try again"), state)
            trace_row = json.loads(trace.read_text(encoding="utf-8"))
        self.assertEqual(selected.tool_calls[0].name, "get_order")
        self.assertEqual(trace_row["candidate_count"], 4)
        self.assertEqual(trace_row["search_budget"], 4)
        self.assertTrue(trace_row["adaptive_search_enabled"])
        self.assertIn("adaptive_high_risk", trace_row["search_budget_reason"])

    def test_candidate_diversity_trace_records_generation_intents(self) -> None:
        def fake_generate(**kwargs):
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": str(kwargs["seed"])})],
                cost=0.01,
                usage={"prompt_tokens": 1, "completion_tokens": 1},
            )

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            agent = Tau3SolverControllerAgent(
                [as_tool(_get_order)],
                "policy",
                "mock",
                {"temperature": 0, "seed": 7},
                candidate_count=4,
                candidate_workers=1,
                enable_candidate_diversity=True,
                controller_trace_path=str(trace),
            )
            state = agent.get_init_state()
            with patch("cgr_exp.tau3_solver_agent.generate", side_effect=fake_generate):
                agent.generate_next_message(UserMessage(role="user", content="order ORD123"), state)
            trace_row = json.loads(trace.read_text(encoding="utf-8"))
        self.assertTrue(trace_row["candidate_diversity_enabled"])
        self.assertEqual(trace_row["candidate_generation_intents"], ["greedy", "prefer_read", "prefer_write", "prefer_ask_or_respond"])
        self.assertEqual(trace_row["candidate_scores"][1]["candidate_generation_intent"], "prefer_read")

    def test_action_space_controller_ranks_available_tools_and_instantiates_once(self) -> None:
        calls: list[dict] = []

        def fake_generate(**kwargs):
            calls.append(kwargs)
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": "ORD123"})],
                cost=0.01,
                usage={"prompt_tokens": 5, "completion_tokens": 2},
            )

        names = canonical_feature_names()
        weights = [0.0 for _name in names]
        weights[names.index("action_is_read")] = 5.0
        weights[names.index("action_is_write")] = -5.0
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "ranker.json"
            trace = Path(tmp) / "trace.jsonl"
            checkpoint.write_text(GenericHarnessRanker(weights).to_json(), encoding="utf-8")
            agent = Tau3SolverControllerAgent(
                [as_tool(_get_order), as_tool(_update_order)],
                "policy",
                "mock",
                {"temperature": 0},
                ranker_checkpoint=str(checkpoint),
                controller_trace_path=str(trace),
                action_space_controller=True,
            )
            state = agent.get_init_state()
            with patch("cgr_exp.tau3_solver_agent.generate", side_effect=fake_generate):
                selected, _state = agent.generate_next_message(UserMessage(role="user", content="order ORD123"), state)
            trace_row = json.loads(trace.read_text(encoding="utf-8"))
        self.assertEqual(selected.tool_calls[0].name, "get_order")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["tool_choice"], "auto")
        self.assertEqual([tool.name for tool in calls[0]["tools"]], ["_get_order"])
        self.assertEqual(trace_row["scoring_mode"], "action_space_canonical_ranker")
        self.assertTrue(trace_row["action_space_controller_enabled"])
        self.assertEqual(trace_row["candidate_count"], 5)
        self.assertEqual(len([item for item in trace_row["candidate_scores"] if item["score"] is not None]), 5)
        self.assertFalse(trace_row["oracle_used_for_candidate_generation"])
        self.assertFalse(trace_row["tool_names_used_as_features"])

    def test_candidate_sampling_retries_transient_failure(self) -> None:
        attempts: dict[int, int] = {}

        def flaky_generate(**kwargs):
            seed = kwargs["seed"]
            attempts[seed] = attempts.get(seed, 0) + 1
            if seed == 8 and attempts[seed] == 1:
                raise RuntimeError("transient")
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": str(seed)})],
            )

        agent = Tau3SolverControllerAgent(
            [as_tool(_get_order)],
            "policy",
            "mock",
            {"temperature": 0, "seed": 7},
            candidate_count=2,
            candidate_workers=2,
            candidate_retries=1,
            candidate_retry_backoff=0,
        )
        state = agent.get_init_state()
        with patch("cgr_exp.tau3_solver_agent.generate", side_effect=flaky_generate):
            candidates = agent._sample_candidates(state)
        self.assertEqual([candidate.tool_calls[0].arguments["order_id"] for candidate in candidates], ["7", "8"])
        self.assertEqual(agent._last_sample_failures, 0)
        self.assertEqual(attempts[8], 2)

    def test_candidate_sampling_keeps_successful_candidates_after_one_final_failure(self) -> None:
        def partially_failing_generate(**kwargs):
            seed = kwargs["seed"]
            if seed == 8:
                raise RuntimeError("persistent")
            return AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": str(seed)})],
            )

        agent = Tau3SolverControllerAgent(
            [as_tool(_get_order)],
            "policy",
            "mock",
            {"temperature": 0, "seed": 7},
            candidate_count=2,
            candidate_workers=2,
            candidate_retries=1,
            candidate_retry_backoff=0,
        )
        state = agent.get_init_state()
        with patch("cgr_exp.tau3_solver_agent.generate", side_effect=partially_failing_generate):
            candidates = agent._sample_candidates(state)
        self.assertEqual([candidate.tool_calls[0].arguments["order_id"] for candidate in candidates], ["7"])
        self.assertEqual(agent._last_sample_failures, 1)
        self.assertEqual(agent._last_sample_failure_types, ["RuntimeError"])

    def test_certified_filter_prunes_unknown_tool_without_oracle(self) -> None:
        calls = [
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="delete_customer", arguments={"customer_id": "C1"})],
                cost=0.01,
                usage={"prompt_tokens": 10, "completion_tokens": 1},
            ),
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order", arguments={"order_id": "A"})],
                cost=0.02,
                usage={"prompt_tokens": 11, "completion_tokens": 2},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            agent = Tau3SolverControllerAgent(
                [as_tool(_get_order)],
                "policy",
                "mock",
                {"temperature": 0},
                candidate_count=2,
                controller_trace_path=str(trace),
                enable_certified_filter=True,
            )
            state = agent.get_init_state()
            with patch("cgr_exp.tau3_solver_agent.generate", side_effect=calls):
                selected, _state = agent.generate_next_message(UserMessage(role="user", content="check order A"), state)
            trace_row = json.loads(trace.read_text(encoding="utf-8"))
        self.assertEqual(selected.tool_calls[0].name, "get_order")
        self.assertTrue(trace_row["certified_filter_enabled"])
        self.assertEqual(trace_row["certified_pruned_count"], 1)
        self.assertEqual(trace_row["certified_prune_reasons"], ["unknown_tool"])
        self.assertFalse(trace_row["oracle_used_for_filtering"])


def _get_order(order_id: str) -> str:
    """Get order."""
    return order_id


def _update_order(order_id: str) -> str:
    """Update order."""
    return order_id


def _aaa_read_order(order_id: str) -> str:
    """Read order."""
    return order_id


def _zzz_read_order(order_id: str) -> str:
    """Read order."""
    return order_id


if __name__ == "__main__":
    unittest.main()
