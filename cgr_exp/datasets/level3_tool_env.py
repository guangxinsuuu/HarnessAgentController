from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from cgr_exp.schemas import CompletionStatus, CostLedger, FeasibleStatus, SearchNode, ValidationResult


@dataclass(frozen=True)
class LookupOrderAction:
    order_id: str
    action_type: str = "lookup_order"

    def validate_schema(self) -> ValidationResult:
        return ValidationResult(isinstance(self.order_id, str) and bool(self.order_id), [] if self.order_id else ["order_id required"])

    def to_dict(self):
        return {"action_type": self.action_type, "order_id": self.order_id}


@dataclass(frozen=True)
class IssueRefundAction:
    order_id: str
    amount: float
    action_type: str = "issue_refund"

    def validate_schema(self) -> ValidationResult:
        errors = []
        if not isinstance(self.order_id, str) or not self.order_id:
            errors.append("order_id required")
        if not isinstance(self.amount, (int, float)) or self.amount <= 0:
            errors.append("positive amount required")
        return ValidationResult(not errors, errors)

    def to_dict(self):
        return {"action_type": self.action_type, "order_id": self.order_id, "amount": self.amount}


@dataclass(frozen=True)
class CancelOrderAction:
    order_id: str
    action_type: str = "cancel_order"

    def validate_schema(self) -> ValidationResult:
        return ValidationResult(isinstance(self.order_id, str) and bool(self.order_id), [] if self.order_id else ["order_id required"])

    def to_dict(self):
        return {"action_type": self.action_type, "order_id": self.order_id}


@dataclass(frozen=True)
class LookupCustomerAction:
    customer_id: str
    action_type: str = "lookup_customer"

    def validate_schema(self) -> ValidationResult:
        return ValidationResult(isinstance(self.customer_id, str) and bool(self.customer_id), [] if self.customer_id else ["customer_id required"])

    def to_dict(self):
        return {"action_type": self.action_type, "customer_id": self.customer_id}


@dataclass(frozen=True)
class UpdateAddressAction:
    customer_id: str
    new_address: str
    action_type: str = "update_address"

    def validate_schema(self) -> ValidationResult:
        errors = []
        if not isinstance(self.customer_id, str) or not self.customer_id:
            errors.append("customer_id required")
        if not isinstance(self.new_address, str) or not self.new_address.strip():
            errors.append("new_address required")
        return ValidationResult(not errors, errors)

    def to_dict(self):
        return {"action_type": self.action_type, "customer_id": self.customer_id, "new_address": self.new_address}


@dataclass(frozen=True)
class RequestApprovalAction:
    order_id: str
    reason: str
    action_type: str = "request_approval"

    def validate_schema(self) -> ValidationResult:
        errors = []
        if not isinstance(self.order_id, str) or not self.order_id:
            errors.append("order_id required")
        if not isinstance(self.reason, str) or not self.reason.strip():
            errors.append("reason required")
        return ValidationResult(not errors, errors)

    def to_dict(self):
        return {"action_type": self.action_type, "order_id": self.order_id, "reason": self.reason}


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    address: str
    tier: str


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    amount: float
    status: str


@dataclass(frozen=True)
class SupportCase:
    case_id: str
    customer_id: str
    goal: str


class LocalToolUseEnvironment:
    def __init__(self, orders: Dict[str, Order], cases: Dict[str, SupportCase], customers: Dict[str, Customer], refund_without_approval_limit: float = 100.0):
        self.orders = orders
        self.cases = cases
        self.customers = customers
        self.refund_without_approval_limit = refund_without_approval_limit

    def root_node(self, case_id: str) -> SearchNode:
        case = self.cases[case_id]
        return SearchNode(
            node_id=f"{case_id}:root",
            parent_id=None,
            depth=0,
            task_id=case_id,
            semantic_state={"goal": case.goal},
            structured_state={"case_id": case_id, "customer_id": case.customer_id, "observed_orders": {}, "observed_customers": {}, "approved_orders": [], "address_updates": {}, "actions": []},
            verified_constraints=[],
            uncertain_constraints=[],
            lower_bound=None,
            incumbent_at_creation=None,
            objective_if_complete=None,
            feasible_status=FeasibleStatus.UNKNOWN,
            completion_status=CompletionStatus.PARTIAL,
            proposal_action=None,
            observation=None,
            universal_features={"depth": 0.0, "branch_factor": 0.0},
            verifier_features={"lower_bound": None, "verified_constraint_count": 0.0, "uncertain_constraint_count": 0.0},
            semantic_features=None,
            missingness_mask={"lower_bound": True, "semantic_features": True},
            accumulated_expensive_cost=CostLedger(),
            metadata={"level": 3, "environment": "customer_support"},
        )


def make_customer_support_fixture() -> LocalToolUseEnvironment:
    customers = {
        "C1": Customer("C1", "Ada Customer", "1 Old Street", "gold"),
        "C2": Customer("C2", "Ben Customer", "9 Other Road", "standard"),
    }
    orders = {
        "O1": Order("O1", "C1", 50.0, "processing"),
        "O2": Order("O2", "C1", 250.0, "shipped"),
        "O3": Order("O3", "C2", 75.0, "processing"),
    }
    cases = {
        "case-1": SupportCase("case-1", "C1", "Help customer C1 with order operations."),
    }
    return LocalToolUseEnvironment(orders, cases, customers)
