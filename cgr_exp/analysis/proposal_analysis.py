from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Set, Tuple

from cgr_exp.proposers import proposal_quality
from cgr_exp.schemas import AssignAction


def proposal_metrics_for_typed_actions(
    actions: Sequence[AssignAction],
    useful_actions: Set[Tuple[str, str]],
    known_tasks: Set[str],
    known_resources: Set[str],
) -> Mapping[str, float]:
    if not actions:
        return {
            "proposal_recall_at_k": 0.0,
            "schema_validity": 0.0,
            "duplicate_action_rate": 0.0,
            "invalid_action_count": 0.0,
            "valid_useful_action_count": 0.0,
        }
    keys = [(a.task, a.resource) for a in actions]
    schema_valid = [
        a.validate_schema().ok and a.task in known_tasks and a.resource in known_resources
        for a in actions
    ]
    valid_useful = [ok and key in useful_actions for ok, key in zip(schema_valid, keys)]
    return {
        "proposal_recall_at_k": 1.0 if any(valid_useful) else 0.0,
        "schema_validity": sum(1.0 for ok in schema_valid if ok) / len(actions),
        "duplicate_action_rate": 1.0 - (len(set(keys)) / len(keys)),
        "invalid_action_count": float(sum(1 for ok in schema_valid if not ok)),
        "valid_useful_action_count": float(sum(1 for ok in valid_useful if ok)),
    }


__all__ = ["proposal_quality", "proposal_metrics_for_typed_actions"]
