from __future__ import annotations

from cgr_exp.datasets.level1_setcover import ChooseSetAction, SetCoverTask
from cgr_exp.schemas import FeasibleStatus, SearchNode, VerificationResult


class Level1SetCoverVerifier:
    def verify(self, task: SetCoverTask, node: SearchNode, action: ChooseSetAction) -> VerificationResult:
        schema = action.validate_schema()
        if not schema.ok:
            return VerificationResult([], [], False, None, schema.errors, {"action": action.to_dict()}, "schema-invalid", "level1_setcover")
        chosen = tuple(node.structured_state["chosen_sets"])
        if action.set_id not in task.sets:
            return VerificationResult([], [], False, None, ["unknown-set"], {"action": action.to_dict()}, "entity-invalid", "level1_setcover")
        if action.set_id in chosen:
            return VerificationResult([], [], False, None, ["already-chosen"], {"action": action.to_dict()}, "state-invalid", "level1_setcover")
        covered = task.covered(chosen)
        gain = task.sets[action.set_id].difference(covered)
        if not gain:
            return VerificationResult([], [], False, None, ["no-new-coverage"], {"covered": sorted(covered)}, "compiled", "level1_setcover")
        next_chosen = chosen + (action.set_id,)
        constraint = f"choose:{action.set_id}"
        evidence = {"set_id": action.set_id, "new_coverage": sorted(gain), "cost": task.costs[action.set_id]}
        return VerificationResult([constraint], [], True, task.lower_bound(next_chosen), [], evidence, "compiled", "level1_setcover")

    def make_child(
        self,
        task: SetCoverTask,
        parent: SearchNode,
        action: ChooseSetAction,
        result: VerificationResult,
        child_index: int,
    ) -> SearchNode:
        del child_index
        node = task.apply(parent, action)
        if result.feasible is False:
            return node.with_updates(feasible_status=FeasibleStatus.INFEASIBLE, lower_bound=None, missingness_mask={**node.missingness_mask, "lower_bound": True})
        if result.lower_bound is None:
            return node.with_updates(lower_bound=None, verifier_features={**node.verifier_features, "lower_bound": None}, missingness_mask={**node.missingness_mask, "lower_bound": True})
        return node.with_updates(lower_bound=result.lower_bound, uncertain_constraints=result.uncertain_constraints)
