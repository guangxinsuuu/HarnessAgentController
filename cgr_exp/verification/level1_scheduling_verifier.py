from __future__ import annotations

from cgr_exp.datasets.level1_scheduling import ScheduleAction, SchedulingTask
from cgr_exp.schemas import FeasibleStatus, SearchNode, VerificationResult


class Level1SchedulingVerifier:
    def verify(self, task: SchedulingTask, node: SearchNode, action: ScheduleAction) -> VerificationResult:
        schema = action.validate_schema()
        if not schema.ok:
            return VerificationResult([], [], False, None, schema.errors, {"action": action.to_dict()}, "schema-invalid", "level1_scheduling")
        scheduled = tuple(node.structured_state["scheduled"])
        if action.job not in task.durations or action.machine not in task.machines:
            return VerificationResult([], [], False, None, ["unknown-entity"], {"action": action.to_dict()}, "entity-invalid", "level1_scheduling")
        if action.job not in task.ready_jobs(scheduled):
            return VerificationResult([], [], False, None, ["precedence-or-duplicate"], {"action": action.to_dict()}, "state-invalid", "level1_scheduling")
        if action.machine not in task.allowed[action.job]:
            return VerificationResult([], [], False, None, ["machine-not-allowed"], {"allowed": task.allowed[action.job]}, "compiled", "level1_scheduling")
        next_scheduled = scheduled + (task._scheduled_row(scheduled, action.job, action.machine),)
        return VerificationResult(
            [f"schedule:{action.job}:{action.machine}", f"precedence:{action.job}", f"machine:{action.machine}"],
            [],
            True,
            task.lower_bound(next_scheduled),
            [],
            {"action": action.to_dict(), "finish": next_scheduled[-1]["finish"]},
            "compiled",
            "level1_scheduling",
        )

    def make_child(
        self,
        task: SchedulingTask,
        parent: SearchNode,
        action: ScheduleAction,
        result: VerificationResult,
        child_index: int,
    ) -> SearchNode:
        node = task.apply(parent, action, child_index=child_index)
        if result.feasible is False:
            return node.with_updates(feasible_status=FeasibleStatus.INFEASIBLE, lower_bound=None, missingness_mask={**node.missingness_mask, "lower_bound": True})
        if result.lower_bound is None:
            return node.with_updates(lower_bound=None, verifier_features={**node.verifier_features, "lower_bound": None}, missingness_mask={**node.missingness_mask, "lower_bound": True})
        return node.with_updates(lower_bound=result.lower_bound, uncertain_constraints=result.uncertain_constraints)
