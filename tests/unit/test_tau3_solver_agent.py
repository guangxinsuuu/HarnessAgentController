from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from cgr_exp.harness_contract import canonical_contract_manifest, canonical_feature_names
from cgr_exp.harness_ranker import GenericHarnessRanker
from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage
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
        self.assertNotIn("get_order", trace_text)
        self.assertNotIn("update_order", trace_text)

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


if __name__ == "__main__":
    unittest.main()
