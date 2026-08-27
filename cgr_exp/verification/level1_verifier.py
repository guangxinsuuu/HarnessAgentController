from __future__ import annotations

import random
from typing import Dict

from cgr_exp.datasets.level1_assignment import AssignmentTask, make_node
from cgr_exp.schemas import AssignAction, FeasibleStatus, SearchNode, VerificationResult


class Level1AssignmentVerifier:
    def __init__(self, false_negative_rate: float = 0.0, false_positive_rate: float = 0.0, seed: int = 0):
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate
        self.rng = random.Random(seed)

    def verify(self, task: AssignmentTask, node: SearchNode, action: AssignAction) -> VerificationResult:
        schema = action.validate_schema()
        if not schema.ok:
            return VerificationResult([], [], False, None, schema.errors, {"action": action.to_dict()}, "schema-invalid", "level1_assignment")
        if action.task not in task.durations or action.resource not in task.resources:
            return VerificationResult([], [], False, None, ["unknown-entity"], {"action": action.to_dict()}, "entity-invalid", "level1_assignment")
        if action.task in node.structured_state["assignments"]:
            return VerificationResult([], [], False, None, ["already-assigned"], {"action": action.to_dict()}, "state-invalid", "level1_assignment")

        allowed = action.resource in task.allowed[action.task]
        if allowed and self.rng.random() < self.false_negative_rate:
            return VerificationResult([], [f"assign:{action.task}:{action.resource}"], True, None, [], {"withheld": True}, "uncertain", "level1_assignment")
        if (not allowed) and self.rng.random() >= self.false_positive_rate:
            return VerificationResult([], [], False, None, ["disallowed-resource"], {"allowed": task.allowed[action.task]}, "compiled", "level1_assignment")

        assignments = dict(node.structured_state["assignments"])
        assignments[action.task] = action.resource
        lower = task.lower_bound(assignments)
        constraint = f"assign:{action.task}:{action.resource}"
        evidence = {"allowed": task.allowed[action.task], "false_positive": not allowed}
        return VerificationResult([constraint], [], True, lower, [], evidence, "compiled", "level1_assignment")

    def make_child(self, task: AssignmentTask, parent: SearchNode, action: AssignAction, result: VerificationResult, child_index: int) -> SearchNode:
        assignments: Dict[str, str] = dict(parent.structured_state["assignments"])
        if result.feasible:
            assignments[action.task] = action.resource
        node = make_node(
            task,
            assignments=assignments,
            parent_id=parent.node_id,
            action=action,
            node_id=f"{parent.node_id}/{child_index}:{action.task}->{action.resource}",
            incumbent=parent.incumbent_at_creation,
            uncertain_constraints=result.uncertain_constraints,
        )
        if result.feasible is False:
            return node.with_updates(feasible_status=FeasibleStatus.INFEASIBLE, lower_bound=None, missingness_mask={**node.missingness_mask, "lower_bound": True})
        if result.lower_bound is None:
            return node.with_updates(lower_bound=None, verifier_features={**node.verifier_features, "lower_bound": None}, missingness_mask={**node.missingness_mask, "lower_bound": True})
        return node.with_updates(lower_bound=result.lower_bound)

