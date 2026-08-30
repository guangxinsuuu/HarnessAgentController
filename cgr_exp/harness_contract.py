from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Dict, Iterable, List, Mapping


RUNTIME_FEATURE_NAMES = [
    "verified_progress_frac",
    "verified_progress_known_mask",
    "remaining_requirement_frac",
    "remaining_requirement_known_mask",
    "observed_fact_frac",
    "observed_fact_known_mask",
    "remaining_turn_budget_frac",
    "remaining_tool_budget_frac",
    "constraint_pressure",
    "constraint_pressure_known_mask",
    "valid_action_ratio",
    "previous_failure_rate",
    "stagnation_steps_frac",
    "uncertainty_score",
    "read_write_balance",
    "irreversible_action_used",
]

RUNTIME_KNOWN_MASK_FEATURE_NAMES = [
    "verified_progress_known_mask",
    "remaining_requirement_known_mask",
    "observed_fact_known_mask",
    "constraint_pressure_known_mask",
]

ACTION_FEATURE_NAMES = [
    "action_is_read",
    "action_is_write",
    "action_is_respond",
    "action_is_ask",
    "action_is_terminate",
    "argument_completeness",
    "references_observed_entity",
    "references_unverified_entity",
    "declared_observation_gain",
    "declared_requirement_gain",
    "expected_tool_cost_frac",
    "action_irreversible",
    "precondition_satisfied",
    "policy_risk",
    "duplicates_previous_action",
]

TRAINING_ONLY_LABEL_FIELDS = [
    "success",
    "final_reward",
    "future_violation_risk",
    "best_descendant_value",
    "cost_to_success",
    "oracle_rank",
    "eventual_outcome",
]

EVALUATION_ONLY_METRIC_FIELDS = [
    "pass_at_1",
    "pass_at_4",
    "violations",
    "tool_calls",
    "tokens",
    "wall_clock_seconds",
]


@dataclass(frozen=True)
class RuntimeFeatures:
    verified_progress_frac: float
    verified_progress_known_mask: float
    remaining_requirement_frac: float
    remaining_requirement_known_mask: float
    observed_fact_frac: float
    observed_fact_known_mask: float
    remaining_turn_budget_frac: float
    remaining_tool_budget_frac: float
    constraint_pressure: float
    constraint_pressure_known_mask: float
    valid_action_ratio: float
    previous_failure_rate: float
    stagnation_steps_frac: float
    uncertainty_score: float
    read_write_balance: float
    irreversible_action_used: float

    def vector(self) -> List[float]:
        return [float(getattr(self, name)) for name in RUNTIME_FEATURE_NAMES]

    def known_mask_values(self) -> Dict[str, float]:
        return {name: float(getattr(self, name)) for name in RUNTIME_KNOWN_MASK_FEATURE_NAMES}

    def known_mask_ratio(self) -> float:
        values = list(self.known_mask_values().values())
        return 0.0 if not values else sum(values) / len(values)


@dataclass(frozen=True)
class ActionFeatures:
    action_is_read: float
    action_is_write: float
    action_is_respond: float
    action_is_ask: float
    action_is_terminate: float
    argument_completeness: float
    references_observed_entity: float
    references_unverified_entity: float
    declared_observation_gain: float
    declared_requirement_gain: float
    expected_tool_cost_frac: float
    action_irreversible: float
    precondition_satisfied: float
    policy_risk: float
    duplicates_previous_action: float

    def vector(self) -> List[float]:
        return [float(getattr(self, name)) for name in ACTION_FEATURE_NAMES]


@dataclass(frozen=True)
class TrainingOnlyLabels:
    success: bool
    final_reward: float
    future_violation_risk: float
    best_descendant_value: float
    cost_to_success: float
    oracle_rank: int
    eventual_outcome: str


@dataclass(frozen=True)
class EvaluationOnlyMetrics:
    pass_at_1: float
    pass_at_4: float
    violations: int
    tool_calls: int
    tokens: int
    wall_clock_seconds: float


@dataclass(frozen=True)
class CanonicalTransitionFacts:
    verified_progress_delta_after_execution: float
    consumes_budget: float
    changes_database: float
    causes_violation: float
    causes_error: float


@dataclass(frozen=True)
class CanonicalCandidate:
    candidate_id: str
    state: RuntimeFeatures
    action: ActionFeatures

    def feature_vector(self) -> List[float]:
        state_vec = self.state.vector()
        action_vec = self.action.vector()
        return [*state_vec, *action_vec, *(s * a for s in state_vec for a in action_vec)]


def canonical_feature_names() -> List[str]:
    return [
        *RUNTIME_FEATURE_NAMES,
        *ACTION_FEATURE_NAMES,
        *(f"{state}*{action}" for state in RUNTIME_FEATURE_NAMES for action in ACTION_FEATURE_NAMES),
    ]


def canonical_contract_manifest() -> Dict[str, object]:
    feature_schema = {
        "runtime_features": RUNTIME_FEATURE_NAMES,
        "runtime_known_mask_features": RUNTIME_KNOWN_MASK_FEATURE_NAMES,
        "action_features": ACTION_FEATURE_NAMES,
        "training_only_labels": TRAINING_ONLY_LABEL_FIELDS,
        "evaluation_only_metrics": EVALUATION_ONLY_METRIC_FIELDS,
    }
    schema_hash = hashlib.sha256(json.dumps(feature_schema, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "contract": "CanonicalHarnessContractV1",
        "feature_schema_hash": schema_hash,
        "runtime_features": RUNTIME_FEATURE_NAMES,
        "runtime_known_mask_features": RUNTIME_KNOWN_MASK_FEATURE_NAMES,
        "action_features": ACTION_FEATURE_NAMES,
        "training_only_labels": TRAINING_ONLY_LABEL_FIELDS,
        "evaluation_only_metrics": EVALUATION_ONLY_METRIC_FIELDS,
        "target_interface_visible": True,
        "target_trajectories_hidden": True,
        "canonical_features_visible": True,
        "oracle_futures_hidden": True,
        "controller_ranks": True,
        "verifier_prunes": True,
        "adapter_changes_checkpoint_stays_frozen": True,
        "oracle_labels_in_inference_features": False,
        "learned_score_authorizes_pruning": False,
        "tool_names_used_as_features": False,
        "tau3_train_trajectories_used": False,
        "tau3_test_outcomes_used_for_design": False,
    }


def audit_canonical_contract_manifest(manifest: Mapping[str, object]) -> Dict[str, object]:
    blockers: List[str] = []
    runtime = {str(item) for item in manifest.get("runtime_features", [])}
    masks = {str(item) for item in manifest.get("runtime_known_mask_features", [])}
    labels = {str(item) for item in manifest.get("training_only_labels", [])}
    metrics = {str(item) for item in manifest.get("evaluation_only_metrics", [])}
    inference = set(canonical_feature_names())
    leaked = sorted(runtime & labels)
    if leaked:
        blockers.append("training-only labels appear in runtime features: " + ", ".join(leaked))
    leaked_inference = sorted((labels | metrics) & inference)
    if leaked_inference:
        blockers.append("training/evaluation-only fields appear in inference features: " + ", ".join(leaked_inference))
    missing_masks = sorted(mask for mask in RUNTIME_KNOWN_MASK_FEATURE_NAMES if mask not in masks or mask not in runtime)
    if missing_masks:
        blockers.append("runtime known-mask features missing: " + ", ".join(missing_masks))
    required_false = [
        "oracle_labels_in_inference_features",
        "learned_score_authorizes_pruning",
        "tool_names_used_as_features",
        "tau3_train_trajectories_used",
        "tau3_test_outcomes_used_for_design",
    ]
    for key in required_false:
        if manifest.get(key) is not False:
            blockers.append(f"{key} must be false")
    required_true = [
        "target_interface_visible",
        "target_trajectories_hidden",
        "canonical_features_visible",
        "oracle_futures_hidden",
        "controller_ranks",
        "verifier_prunes",
        "adapter_changes_checkpoint_stays_frozen",
    ]
    for key in required_true:
        if manifest.get(key) is not True:
            blockers.append(f"{key} must be true")
    return {
        "contract": manifest.get("contract", ""),
        "ready": not blockers,
        "blockers": blockers,
    }


def dataclass_to_json_dict(value) -> Dict[str, object]:
    return asdict(value)


def write_json(path, data: Mapping[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
