from __future__ import annotations

from cgr_exp.datasets.level2_renderer import Level2Rendering
from cgr_exp.schemas import AssignAction, FeasibleStatus, SearchNode, VerificationResult
from cgr_exp.verification.level1_verifier import Level1AssignmentVerifier


class Level2Verifier:
    def __init__(self, false_negative_rate: float = 0.0, false_positive_rate: float = 0.0, seed: int = 0):
        self.hidden_verifier = Level1AssignmentVerifier(false_negative_rate, false_positive_rate, seed)

    def verify(self, rendering: Level2Rendering, node: SearchNode, action: AssignAction) -> VerificationResult:
        schema = action.validate_schema()
        if not schema.ok:
            return VerificationResult([], [], False, None, schema.errors, {"raw_action": action.to_dict()}, "schema-invalid", "level2_assignment")

        resolved = self._resolve_action(rendering, action)
        if resolved is None:
            return VerificationResult(
                [],
                [],
                False,
                None,
                ["unknown-rendered-entity"],
                {"raw_action": action.to_dict(), "rendering_id": rendering.rendering_id},
                "entity-invalid",
                "level2_assignment",
            )

        result = self.hidden_verifier.verify(rendering.hidden_task, node, resolved)
        uncertain = list(result.uncertain_constraints)
        if rendering.perturbation == "mild_ambiguity":
            uncertain.append("preference:load_balance")
        evidence = {
            **result.verification_evidence,
            "rendering_id": rendering.rendering_id,
            "perturbation": rendering.perturbation,
            "raw_action": action.to_dict(),
            "resolved_action": resolved.to_dict(),
        }
        return VerificationResult(
            verified_constraints=result.verified_constraints,
            uncertain_constraints=uncertain,
            feasible=result.feasible,
            lower_bound=result.lower_bound,
            violations=result.violations,
            verification_evidence=evidence,
            compiler_status=result.compiler_status,
            verifier_backend="level2_assignment",
        )

    def make_child(self, rendering: Level2Rendering, parent: SearchNode, action: AssignAction, result: VerificationResult, child_index: int) -> SearchNode:
        resolved = self._resolve_action(rendering, action)
        if resolved is None:
            return parent.with_updates(feasible_status=FeasibleStatus.INFEASIBLE, lower_bound=None, missingness_mask={**parent.missingness_mask, "lower_bound": True})
        return self.hidden_verifier.make_child(rendering.hidden_task, parent, resolved, result, child_index)

    def _resolve_action(self, rendering: Level2Rendering, action: AssignAction) -> AssignAction | None:
        task_lookup = self._inverse_entity_map(rendering.entity_map.get("tasks", {}))
        resource_lookup = self._inverse_entity_map(rendering.entity_map.get("resources", {}))
        task = task_lookup.get(action.task)
        resource = resource_lookup.get(action.resource)
        if task is None or resource is None:
            return None
        return AssignAction(task=task, resource=resource)

    def _inverse_entity_map(self, mapping):
        inverse = {canonical: canonical for canonical in mapping}
        inverse.update({rendered: canonical for canonical, rendered in mapping.items()})
        return inverse
