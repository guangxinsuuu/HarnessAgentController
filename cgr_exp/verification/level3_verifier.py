from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from cgr_exp.datasets.level3_tool_env import (
    CancelOrderAction,
    IssueRefundAction,
    LocalToolUseEnvironment,
    LookupCustomerAction,
    LookupOrderAction,
    RequestApprovalAction,
    UpdateAddressAction,
)
from cgr_exp.schemas import CompletionStatus, FeasibleStatus, SearchNode, VerificationResult


class Level3ToolVerifier:
    def __init__(self, env: LocalToolUseEnvironment):
        self.env = env

    def verify(self, node: SearchNode, action) -> VerificationResult:
        schema = action.validate_schema()
        if not schema.ok:
            return VerificationResult([], [], False, None, schema.errors, {"action": action.to_dict()}, "schema-invalid", "level3_tool_env")
        customer_id = node.structured_state["customer_id"]
        if isinstance(action, (LookupCustomerAction, UpdateAddressAction)):
            customer = self.env.customers.get(action.customer_id)
            if customer is None:
                return VerificationResult([], [], False, None, ["unknown_customer"], {"action": action.to_dict()}, "entity-invalid", "level3_tool_env")
            if customer.customer_id != customer_id:
                return VerificationResult([], [], False, None, ["ownership"], {"action_customer": customer.customer_id, "case_customer": customer_id}, "compiled", "level3_tool_env")
            if isinstance(action, LookupCustomerAction):
                return VerificationResult(
                    [f"customer:{customer.customer_id}", "tool:lookup_customer"],
                    [],
                    True,
                    None,
                    [],
                    {"customer_id": customer.customer_id, "tier": customer.tier, "address": customer.address},
                    "compiled",
                    "level3_tool_env",
                )
            return VerificationResult(
                [f"customer:{customer.customer_id}", "policy:update_own_address", "postcondition:address_updated"],
                [],
                True,
                None,
                [],
                {"customer_id": customer.customer_id, "new_address": action.new_address},
                "compiled",
                "level3_tool_env",
            )

        order = self.env.orders.get(action.order_id)
        if order is None:
            return VerificationResult([], [], False, None, ["unknown_order"], {"action": action.to_dict()}, "entity-invalid", "level3_tool_env")

        if order.customer_id != customer_id:
            return VerificationResult([], [], False, None, ["ownership"], {"order_customer": order.customer_id, "case_customer": customer_id}, "compiled", "level3_tool_env")

        if isinstance(action, LookupOrderAction):
            return VerificationResult(
                [f"ownership:{order.order_id}:{customer_id}", "tool:lookup_order"],
                [],
                True,
                None,
                [],
                {"order_status": order.status, "amount": order.amount},
                "compiled",
                "level3_tool_env",
            )
        if isinstance(action, IssueRefundAction):
            approved_orders = set(node.structured_state.get("approved_orders", []))
            if action.amount > self.env.refund_without_approval_limit and action.order_id not in approved_orders:
                return VerificationResult(
                    [f"ownership:{order.order_id}:{customer_id}"],
                    ["approval_may_be_requested"],
                    False,
                    None,
                    ["approval_required"],
                    {"limit": self.env.refund_without_approval_limit, "amount": action.amount},
                    "compiled",
                    "level3_tool_env",
                )
            if action.amount > self.env.refund_without_approval_limit:
                return VerificationResult(
                    [f"ownership:{order.order_id}:{customer_id}", "policy:refund_allowed_with_approval", "postcondition:refund_issued"],
                    [],
                    True,
                    None,
                    [],
                    {"amount": action.amount, "approval": "recorded"},
                    "compiled",
                    "level3_tool_env",
                )
            return VerificationResult(
                [f"ownership:{order.order_id}:{customer_id}", "policy:refund_allowed", "postcondition:refund_issued"],
                [],
                True,
                None,
                [],
                {"amount": action.amount},
                "compiled",
                "level3_tool_env",
            )
        if isinstance(action, RequestApprovalAction):
            return VerificationResult(
                [f"ownership:{order.order_id}:{customer_id}", "policy:approval_request_allowed", "postcondition:approval_recorded"],
                [],
                True,
                None,
                [],
                {"order_id": order.order_id, "reason": action.reason},
                "compiled",
                "level3_tool_env",
            )
        if isinstance(action, CancelOrderAction):
            if order.status == "shipped":
                return VerificationResult(
                    [f"ownership:{order.order_id}:{customer_id}", "policy:cancel_before_shipping"],
                    [],
                    False,
                    None,
                    ["already_shipped"],
                    {"status": order.status},
                    "compiled",
                    "level3_tool_env",
                )
            return VerificationResult(
                [f"ownership:{order.order_id}:{customer_id}", "policy:cancel_allowed", "postcondition:order_cancelled"],
                [],
                True,
                None,
                [],
                {"status": order.status},
                "compiled",
                "level3_tool_env",
            )
        return VerificationResult([], [], False, None, ["unknown_action_type"], {"action": action.to_dict()}, "schema-invalid", "level3_tool_env")

    def make_child(self, parent: SearchNode, action, result: VerificationResult) -> SearchNode:
        structured = dict(parent.structured_state)
        structured["actions"] = list(structured.get("actions", [])) + [action.to_dict()]
        observation: Dict[str, Any] = dict(result.verification_evidence)
        if isinstance(action, LookupOrderAction) and result.feasible:
            observed = dict(structured.get("observed_orders", {}))
            observed[action.order_id] = observation
            structured["observed_orders"] = observed
        if isinstance(action, LookupCustomerAction) and result.feasible:
            observed_customers = dict(structured.get("observed_customers", {}))
            observed_customers[action.customer_id] = observation
            structured["observed_customers"] = observed_customers
        if isinstance(action, UpdateAddressAction) and result.feasible:
            updates = dict(structured.get("address_updates", {}))
            updates[action.customer_id] = action.new_address
            structured["address_updates"] = updates
        if isinstance(action, RequestApprovalAction) and result.feasible:
            approved = list(structured.get("approved_orders", []))
            if action.order_id not in approved:
                approved.append(action.order_id)
            structured["approved_orders"] = approved
        feasible_status = FeasibleStatus.FEASIBLE if result.feasible else FeasibleStatus.INFEASIBLE
        return parent.with_updates(
            node_id=f"{parent.node_id}/{len(structured['actions'])}:{action.action_type}",
            parent_id=parent.node_id,
            depth=parent.depth + 1,
            structured_state=structured,
            verified_constraints=list(parent.verified_constraints) + result.verified_constraints,
            uncertain_constraints=list(parent.uncertain_constraints) + result.uncertain_constraints,
            lower_bound=None,
            feasible_status=feasible_status,
            completion_status=CompletionStatus.PARTIAL,
            proposal_action=action,
            observation=observation,
            verifier_features={"lower_bound": None, "verified_constraint_count": float(len(result.verified_constraints)), "uncertain_constraint_count": float(len(result.uncertain_constraints))},
            missingness_mask={**parent.missingness_mask, "lower_bound": True},
        )
