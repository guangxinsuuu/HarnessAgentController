from __future__ import annotations

from cgr_exp.datasets.game24 import Game24Action, Game24Task
from cgr_exp.schemas import FeasibleStatus, SearchNode, VerificationResult


class Game24Verifier:
    def verify(self, task: Game24Task, node: SearchNode, action: Game24Action) -> VerificationResult:
        schema = action.validate_schema()
        if not schema.ok:
            return VerificationResult([], [], False, None, schema.errors, {"action": action.to_dict()}, "schema-invalid", "game24")
        if not task.is_valid_action(node, action):
            return VerificationResult([], [], False, None, ["number-not-available"], {"action": action.to_dict(), "values": node.structured_state["values"]}, "state-invalid", "game24")
        child = task.apply(node, action)
        return VerificationResult(
            [f"arithmetic:{action.left}:{action.operator}:{action.right}"],
            [],
            True,
            child.lower_bound,
            [],
            {"action": action.to_dict(), "next_values": child.structured_state["values"]},
            "compiled",
            "game24",
        )

    def make_child(
        self,
        task: Game24Task,
        parent: SearchNode,
        action: Game24Action,
        result: VerificationResult,
        child_index: int,
    ) -> SearchNode:
        if result.feasible is False:
            return parent.with_updates(feasible_status=FeasibleStatus.INFEASIBLE)
        return task.apply(parent, action, child_index=child_index)
