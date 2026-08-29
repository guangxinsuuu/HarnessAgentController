from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from cgr_exp.harness_contract import canonical_feature_names
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
        self.assertEqual(selected.tool_calls[0].name, "get_order")
        self.assertEqual(trace_row["scoring_mode"], "canonical_ranker")
        self.assertTrue(trace_row["ranker_checkpoint_loaded"])
        self.assertFalse(trace_row["oracle_visible_to_controller"])
        self.assertFalse(trace_row["tool_names_used_as_features"])
        self.assertNotIn("get_order", trace_text)
        self.assertNotIn("update_order", trace_text)


def _get_order(order_id: str) -> str:
    """Get order."""
    return order_id


def _update_order(order_id: str) -> str:
    """Update order."""
    return order_id


if __name__ == "__main__":
    unittest.main()
