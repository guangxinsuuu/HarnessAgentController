from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Dict, List, Optional


class FeasibleStatus(str, Enum):
    UNKNOWN = "unknown"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"


class CompletionStatus(str, Enum):
    PARTIAL = "partial"
    COMPLETE = "complete"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: List[str]


class TypedAction:
    action_type: str

    def validate_schema(self) -> ValidationResult:
        return ValidationResult(True, [])

    def to_dict(self) -> Dict[str, Any]:
        return dict(vars(self))


@dataclass(frozen=True)
class AssignAction(TypedAction):
    task: str
    resource: str
    action_type: str = "assign"

    def validate_schema(self) -> ValidationResult:
        errors: List[str] = []
        if not isinstance(self.task, str) or not self.task:
            errors.append("task must be a non-empty string")
        if not isinstance(self.resource, str) or not self.resource:
            errors.append("resource must be a non-empty string")
        return ValidationResult(not errors, errors)


@dataclass(frozen=True)
class UnknownAction(TypedAction):
    action_type: str = "unknown"


def action_from_dict(data: Optional[Dict[str, Any]]) -> Optional[TypedAction]:
    if data is None:
        return None
    action_type = data.get("action_type", "unknown")
    if action_type == "assign":
        return AssignAction(task=data.get("task"), resource=data.get("resource"))
    if action_type == "choose_set":
        from cgr_exp.datasets.level1_setcover import ChooseSetAction

        return ChooseSetAction(set_id=data.get("set_id"))
    if action_type == "schedule":
        from cgr_exp.datasets.level1_scheduling import ScheduleAction

        return ScheduleAction(job=data.get("job"), machine=data.get("machine"))
    if action_type == "game24_op":
        from cgr_exp.datasets.game24 import Game24Action

        return Game24Action(left=data.get("left"), right=data.get("right"), operator=data.get("operator"))
    if action_type == "lookup_order":
        from cgr_exp.datasets.level3_tool_env import LookupOrderAction

        return LookupOrderAction(order_id=data.get("order_id"))
    if action_type == "issue_refund":
        from cgr_exp.datasets.level3_tool_env import IssueRefundAction

        return IssueRefundAction(order_id=data.get("order_id"), amount=data.get("amount"))
    if action_type == "cancel_order":
        from cgr_exp.datasets.level3_tool_env import CancelOrderAction

        return CancelOrderAction(order_id=data.get("order_id"))
    if action_type == "lookup_customer":
        from cgr_exp.datasets.level3_tool_env import LookupCustomerAction

        return LookupCustomerAction(customer_id=data.get("customer_id"))
    if action_type == "update_address":
        from cgr_exp.datasets.level3_tool_env import UpdateAddressAction

        return UpdateAddressAction(customer_id=data.get("customer_id"), new_address=data.get("new_address"))
    if action_type == "request_approval":
        from cgr_exp.datasets.level3_tool_env import RequestApprovalAction

        return RequestApprovalAction(order_id=data.get("order_id"), reason=data.get("reason"))
    return UnknownAction(action_type=action_type)



@dataclass(frozen=True)
class CostLedger:
    proposal_calls: int = 0
    evaluation_calls: int = 0
    reflection_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    solver_calls: int = 0
    solver_seconds: float = 0.0
    verifier_calls: int = 0
    verifier_seconds: float = 0.0
    ranker_calls: int = 0
    ranker_seconds: float = 0.0
    wall_clock_seconds: float = 0.0

    def __add__(self, other: "CostLedger") -> "CostLedger":
        return CostLedger(**{k: getattr(self, k) + getattr(other, k) for k in asdict(self)})

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CostLedger":
        return cls(**data)


@dataclass(frozen=True)
class VerificationResult:
    verified_constraints: List[str]
    uncertain_constraints: List[str]
    feasible: Optional[bool]
    lower_bound: Optional[float]
    violations: List[str]
    verification_evidence: Dict[str, Any]
    compiler_status: str
    verifier_backend: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VerificationResult":
        return cls(**data)


@dataclass(frozen=True)
class SearchNode:
    node_id: str
    parent_id: Optional[str]
    depth: int
    task_id: str
    semantic_state: Dict[str, Any]
    structured_state: Dict[str, Any]
    verified_constraints: List[str]
    uncertain_constraints: List[str]
    lower_bound: Optional[float]
    incumbent_at_creation: Optional[float]
    objective_if_complete: Optional[float]
    feasible_status: FeasibleStatus
    completion_status: CompletionStatus
    proposal_action: Optional[TypedAction]
    observation: Optional[Dict[str, Any]]
    universal_features: Dict[str, float]
    verifier_features: Dict[str, Optional[float]]
    semantic_features: Optional[List[float]]
    missingness_mask: Dict[str, bool]
    accumulated_expensive_cost: CostLedger
    metadata: Dict[str, Any]

    def with_updates(self, **kwargs: Any) -> "SearchNode":
        return replace(self, **kwargs)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["feasible_status"] = self.feasible_status.value
        data["completion_status"] = self.completion_status.value
        data["proposal_action"] = None if self.proposal_action is None else self.proposal_action.to_dict()
        data["accumulated_expensive_cost"] = self.accumulated_expensive_cost.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SearchNode":
        obj = dict(data)
        obj["feasible_status"] = FeasibleStatus(obj["feasible_status"])
        obj["completion_status"] = CompletionStatus(obj["completion_status"])
        obj["proposal_action"] = action_from_dict(obj.get("proposal_action"))
        obj["accumulated_expensive_cost"] = CostLedger.from_dict(obj["accumulated_expensive_cost"])
        return cls(**obj)
