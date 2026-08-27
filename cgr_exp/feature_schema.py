from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from cgr_exp.schemas import SearchNode


@dataclass(frozen=True)
class SharedFeatureSchema:
    version: str
    universal: tuple[str, ...]
    verifier: tuple[str, ...]
    semantic: tuple[str, ...]
    mask_keys: tuple[str, ...]

    def all_inference_features(self) -> tuple[str, ...]:
        return self.universal + self.verifier + self.semantic


FEATURE_SCHEMA = SharedFeatureSchema(
    version="shared-node-v1",
    universal=(
        "depth",
        "branch_factor",
        "frontier_size_at_creation",
        "num_siblings",
        "num_failed_siblings",
        "num_successful_siblings",
        "ancestor_length",
        "ancestor_failure_count",
        "ancestor_bound_improvement_count",
        "accumulated_search_cost",
        "remaining_budget_fraction",
        "duplicate_state_count",
    ),
    verifier=(
        "lower_bound",
        "normalized_bound_gap",
        "verified_constraint_count",
        "uncertain_constraint_count",
        "verified_coverage",
        "constraint_violation_count",
        "feasibility_signal",
        "bound_improvement_from_parent",
        "incumbent_gap",
    ),
    semantic=("semantic_features",),
    mask_keys=("lower_bound", "semantic_features"),
)


def validate_node_features(node: SearchNode, schema: SharedFeatureSchema = FEATURE_SCHEMA) -> None:
    if "depth" not in node.universal_features:
        raise ValueError("universal feature depth is required")
    if "branch_factor" not in node.universal_features:
        raise ValueError("universal feature branch_factor is required")
    if "lower_bound" not in node.verifier_features:
        raise ValueError("verifier feature lower_bound is required")
    for key in schema.mask_keys:
        if key not in node.missingness_mask:
            raise ValueError(f"missingness mask must include {key}")
    if node.lower_bound is None and node.missingness_mask.get("lower_bound") is not True:
        raise ValueError("lower_bound=None must be masked")
    if node.verifier_features.get("lower_bound") is None and node.missingness_mask.get("lower_bound") is not True:
        raise ValueError("verifier lower_bound=None must be masked")
    if node.semantic_features is None and node.missingness_mask.get("semantic_features") is not True:
        raise ValueError("semantic_features=None must be masked")
    _reject_future_label_features(node.universal_features)
    _reject_future_label_features(node.verifier_features)


def _reject_future_label_features(features) -> None:
    forbidden = {
        "contains_optimal_descendant",
        "future_best_objective",
        "future_remaining_cost",
        "best_descendant_objective",
        "remaining_exact_expansions_to_best",
    }
    overlap = forbidden.intersection(features)
    if overlap:
        raise ValueError(f"future/oracle label fields cannot be inference features: {sorted(overlap)}")
