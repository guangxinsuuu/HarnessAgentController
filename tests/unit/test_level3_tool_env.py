from __future__ import annotations

import unittest

from cgr_exp.datasets.level3_tool_env import (
    CancelOrderAction,
    IssueRefundAction,
    LookupCustomerAction,
    LookupOrderAction,
    RequestApprovalAction,
    UpdateAddressAction,
    make_customer_support_fixture,
)
from cgr_exp.experiments import _level3_tool_diagnostics
from cgr_exp.verification.level3_verifier import Level3ToolVerifier


class Level3ToolEnvironmentTest(unittest.TestCase):
    def test_lookup_order_creates_no_bound_node_with_missing_mask(self) -> None:
        env = make_customer_support_fixture()
        root = env.root_node("case-1")
        verifier = Level3ToolVerifier(env)
        result = verifier.verify(root, LookupOrderAction(order_id="O1"))
        child = verifier.make_child(root, LookupOrderAction(order_id="O1"), result)
        self.assertTrue(result.feasible)
        self.assertIsNone(result.lower_bound)
        self.assertIsNone(child.lower_bound)
        self.assertTrue(child.missingness_mask["lower_bound"])

    def test_refund_requires_ownership_and_policy_approval(self) -> None:
        env = make_customer_support_fixture()
        root = env.root_node("case-1")
        verifier = Level3ToolVerifier(env)
        wrong_owner = verifier.verify(root, IssueRefundAction(order_id="O3", amount=20.0))
        needs_approval = verifier.verify(root, IssueRefundAction(order_id="O2", amount=200.0))
        allowed = verifier.verify(root, IssueRefundAction(order_id="O1", amount=20.0))
        self.assertFalse(wrong_owner.feasible)
        self.assertIn("ownership", wrong_owner.violations)
        self.assertFalse(needs_approval.feasible)
        self.assertIn("approval_required", needs_approval.violations)
        self.assertTrue(allowed.feasible)
        self.assertIn("policy:refund_allowed", allowed.verified_constraints)

    def test_cancel_shipped_order_is_verified_infeasible(self) -> None:
        env = make_customer_support_fixture()
        root = env.root_node("case-1")
        verifier = Level3ToolVerifier(env)
        result = verifier.verify(root, CancelOrderAction(order_id="O2"))
        child = verifier.make_child(root, CancelOrderAction(order_id="O2"), result)
        self.assertFalse(result.feasible)
        self.assertIn("already_shipped", result.violations)
        self.assertEqual(child.feasible_status.value, "infeasible")

    def test_lookup_customer_and_update_address_require_case_ownership(self) -> None:
        env = make_customer_support_fixture()
        root = env.root_node("case-1")
        verifier = Level3ToolVerifier(env)
        lookup = verifier.verify(root, LookupCustomerAction(customer_id="C1"))
        update = verifier.verify(root, UpdateAddressAction(customer_id="C1", new_address="42 New Street"))
        wrong_customer = verifier.verify(root, UpdateAddressAction(customer_id="C2", new_address="42 New Street"))
        self.assertTrue(lookup.feasible)
        self.assertIn("tool:lookup_customer", lookup.verified_constraints)
        self.assertTrue(update.feasible)
        self.assertIn("postcondition:address_updated", update.verified_constraints)
        self.assertFalse(wrong_customer.feasible)
        self.assertIn("ownership", wrong_customer.violations)

    def test_request_approval_allows_high_value_refund_on_later_state(self) -> None:
        env = make_customer_support_fixture()
        root = env.root_node("case-1")
        verifier = Level3ToolVerifier(env)
        before = verifier.verify(root, IssueRefundAction(order_id="O2", amount=200.0))
        approval = verifier.verify(root, RequestApprovalAction(order_id="O2", reason="customer retention"))
        approved_node = verifier.make_child(root, RequestApprovalAction(order_id="O2", reason="customer retention"), approval)
        after = verifier.verify(approved_node, IssueRefundAction(order_id="O2", amount=200.0))
        self.assertFalse(before.feasible)
        self.assertIn("approval_required", before.violations)
        self.assertTrue(approval.feasible)
        self.assertIn("postcondition:approval_recorded", approval.verified_constraints)
        self.assertTrue(after.feasible)
        self.assertIn("policy:refund_allowed_with_approval", after.verified_constraints)

    def test_level3_diagnostics_include_post_approval_refund_success(self) -> None:
        rows = _level3_tool_diagnostics()
        post_approval_refunds = [
            row for row in rows
            if row["action_type"] == "issue_refund" and row["order_id"] == "O2" and row["feasible"] is True
        ]
        self.assertTrue(post_approval_refunds)


if __name__ == "__main__":
    unittest.main()
