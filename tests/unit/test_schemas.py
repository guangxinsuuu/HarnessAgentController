from __future__ import annotations

import unittest

from cgr_exp.schemas import (
    AssignAction,
    CompletionStatus,
    CostLedger,
    FeasibleStatus,
    SearchNode,
    VerificationResult,
)
from cgr_exp.datasets.level1_setcover import ChooseSetAction
from cgr_exp.datasets.level1_scheduling import ScheduleAction
from cgr_exp.datasets.level3_tool_env import (
    CancelOrderAction,
    IssueRefundAction,
    LookupCustomerAction,
    LookupOrderAction,
    RequestApprovalAction,
    UpdateAddressAction,
)


class SchemaTest(unittest.TestCase):
    def test_search_node_round_trip_preserves_missing_bound(self) -> None:
        node = SearchNode(
            node_id="n1",
            parent_id=None,
            depth=0,
            task_id="task-1",
            semantic_state={"text": "demo"},
            structured_state={"assignments": [None, None]},
            verified_constraints=[],
            uncertain_constraints=["soft-preference"],
            lower_bound=None,
            incumbent_at_creation=None,
            objective_if_complete=None,
            feasible_status=FeasibleStatus.UNKNOWN,
            completion_status=CompletionStatus.PARTIAL,
            proposal_action=None,
            observation=None,
            universal_features={"depth": 0.0},
            verifier_features={"lower_bound": None},
            semantic_features=None,
            missingness_mask={"lower_bound": True},
            accumulated_expensive_cost=CostLedger(),
            metadata={"schema_version": "test"},
        )
        restored = SearchNode.from_dict(node.to_dict())
        self.assertIsNone(restored.lower_bound)
        self.assertTrue(restored.missingness_mask["lower_bound"])
        self.assertEqual(restored.to_dict(), node.to_dict())

    def test_action_validation_rejects_missing_and_wrong_type(self) -> None:
        self.assertTrue(AssignAction(task="T1", resource="R2").validate_schema().ok)
        self.assertFalse(AssignAction(task="", resource="R2").validate_schema().ok)
        self.assertFalse(AssignAction(task="T1", resource=2).validate_schema().ok)  # type: ignore[arg-type]

    def test_search_node_round_trip_preserves_typed_actions(self) -> None:
        actions = [
            AssignAction(task="T1", resource="R1"),
            ChooseSetAction(set_id="S1"),
            ScheduleAction(job="J1", machine="M0"),
            LookupOrderAction(order_id="O1"),
            IssueRefundAction(order_id="O1", amount=25.0),
            CancelOrderAction(order_id="O1"),
            LookupCustomerAction(customer_id="C1"),
            UpdateAddressAction(customer_id="C1", new_address="42 New Street"),
            RequestApprovalAction(order_id="O2", reason="retention"),
        ]
        for action in actions:
            with self.subTest(action=action.action_type):
                node = SearchNode(
                    node_id=f"n:{action.action_type}",
                    parent_id="root",
                    depth=1,
                    task_id="task-1",
                    semantic_state={},
                    structured_state={},
                    verified_constraints=[],
                    uncertain_constraints=[],
                    lower_bound=None,
                    incumbent_at_creation=None,
                    objective_if_complete=None,
                    feasible_status=FeasibleStatus.UNKNOWN,
                    completion_status=CompletionStatus.PARTIAL,
                    proposal_action=action,
                    observation=None,
                    universal_features={"depth": 1.0},
                    verifier_features={"lower_bound": None},
                    semantic_features=None,
                    missingness_mask={"lower_bound": True},
                    accumulated_expensive_cost=CostLedger(),
                    metadata={},
                )
                restored = SearchNode.from_dict(node.to_dict())
                self.assertEqual(type(restored.proposal_action), type(action))
                self.assertEqual(restored.to_dict(), node.to_dict())

    def test_verification_result_round_trip(self) -> None:
        result = VerificationResult(
            verified_constraints=["assign:T1:R2"],
            uncertain_constraints=["preference:R2"],
            feasible=True,
            lower_bound=3.0,
            violations=[],
            verification_evidence={"rule": "fixture"},
            compiler_status="compiled",
            verifier_backend="unit",
        )
        self.assertEqual(VerificationResult.from_dict(result.to_dict()), result)

    def test_cost_ledger_adds_exact_fields(self) -> None:
        a = CostLedger(proposal_calls=1, output_tokens=7, verifier_calls=2)
        b = CostLedger(evaluation_calls=3, input_tokens=11, verifier_seconds=0.5)
        c = a + b
        self.assertEqual(c.proposal_calls, 1)
        self.assertEqual(c.evaluation_calls, 3)
        self.assertEqual(c.input_tokens, 11)
        self.assertEqual(c.output_tokens, 7)
        self.assertEqual(c.verifier_calls, 2)
        self.assertEqual(c.verifier_seconds, 0.5)


if __name__ == "__main__":
    unittest.main()
