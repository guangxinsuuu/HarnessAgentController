from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from cgr_exp.harness_contract import ActionFeatures, CanonicalCandidate, RuntimeFeatures, canonical_contract_manifest


READ_TOKENS = ("get", "search", "list", "find", "lookup", "read", "retrieve")
WRITE_TOKENS = ("update", "create", "cancel", "book", "refund", "exchange", "modify", "send", "transfer", "delete")
CONFIRMATION_TOKENS = ("yes", "confirm", "confirmed", "approve", "approved", "authorize", "authorized", "proceed", "go ahead")
IDENTITY_TOKENS = ("verified", "authenticated", "authentication", "identity verified", "matched customer", "authorized user")


@dataclass(frozen=True)
class Tau3AdapterLogRow:
    candidate_id: str
    action_type: str
    feature_vector_length: int
    oracle_visible_to_controller: bool
    tool_name_used_as_feature: bool


@dataclass(frozen=True)
class Tau3ProgressPredicates:
    identity_verified: float
    target_entity_resolved: float
    required_read_completed: float
    user_confirmation_obtained: float
    required_policy_condition_checked: float
    write_action_completed: float
    terminal_condition_met: float

    def values(self) -> dict[str, float]:
        return {
            "identity_verified": self.identity_verified,
            "target_entity_resolved": self.target_entity_resolved,
            "required_read_completed": self.required_read_completed,
            "user_confirmation_obtained": self.user_confirmation_obtained,
            "required_policy_condition_checked": self.required_policy_condition_checked,
            "write_action_completed": self.write_action_completed,
            "terminal_condition_met": self.terminal_condition_met,
        }

    def provenance(self) -> dict[str, str]:
        return {name: "runtime_visible_message_and_tool_history" for name in self.values()}

    def progress_frac(self) -> float:
        values = list(self.values().values())
        return 0.0 if not values else sum(values) / len(values)


def tau3_candidate_to_canonical(
    *,
    candidate_id: str,
    message_history: Sequence[object],
    tools: Sequence[object],
    candidate: object,
    remaining_turns: int,
    max_turns: int,
    remaining_tool_budget: int,
    max_tool_budget: int,
) -> CanonicalCandidate:
    state = tau3_runtime_features(
        message_history=message_history,
        tools=tools,
        remaining_turns=remaining_turns,
        max_turns=max_turns,
        remaining_tool_budget=remaining_tool_budget,
        max_tool_budget=max_tool_budget,
    )
    action = tau3_action_features(candidate=candidate, message_history=message_history, max_tool_budget=max_tool_budget)
    return CanonicalCandidate(candidate_id=candidate_id, state=state, action=action)


def tau3_runtime_features(
    *,
    message_history: Sequence[object],
    tools: Sequence[object],
    remaining_turns: int,
    max_turns: int,
    remaining_tool_budget: int,
    max_tool_budget: int,
) -> RuntimeFeatures:
    tool_calls = _assistant_tool_calls(message_history)
    tool_outputs = [item for item in message_history if item.__class__.__name__.lower().endswith("toolmessage")]
    failed = sum(1 for item in tool_outputs if _tool_output_failed(item))
    read_calls = sum(1 for call in tool_calls if _classify_tool_name(_call_name(call)) == "read")
    write_calls = sum(1 for call in tool_calls if _classify_tool_name(_call_name(call)) == "write")
    observed_entities = _observed_entity_tokens(message_history)
    progress = tau3_progress_predicates(message_history=message_history)
    progress_frac = progress.progress_frac()
    return RuntimeFeatures(
        verified_progress_frac=progress_frac,
        verified_progress_known_mask=1.0,
        remaining_requirement_frac=1.0 - progress_frac,
        remaining_requirement_known_mask=1.0,
        observed_fact_frac=min(1.0, len(tool_outputs) / max(1, len(tools))),
        observed_fact_known_mask=1.0,
        remaining_turn_budget_frac=max(0.0, remaining_turns / max(1, max_turns)),
        remaining_tool_budget_frac=max(0.0, remaining_tool_budget / max(1, max_tool_budget)),
        constraint_pressure=min(1.0, failed / max(1, len(tool_outputs))),
        constraint_pressure_known_mask=1.0,
        valid_action_ratio=1.0,
        previous_failure_rate=failed / max(1, len(tool_outputs)),
        stagnation_steps_frac=_stagnation_steps_frac(message_history),
        uncertainty_score=1.0 if not observed_entities else max(0.0, 1.0 - len(observed_entities) / 10.0),
        read_write_balance=(read_calls - write_calls) / max(1, len(tool_calls)),
        irreversible_action_used=1.0 if write_calls else 0.0,
    )


def tau3_progress_predicates(*, message_history: Sequence[object]) -> Tau3ProgressPredicates:
    tool_calls = _assistant_tool_calls(message_history)
    tool_outputs = [item for item in message_history if item.__class__.__name__.lower().endswith("toolmessage")]
    successful_outputs = [item for item in tool_outputs if not _tool_output_failed(item)]
    read_calls = [call for call in tool_calls if _classify_tool_name(_call_name(call)) == "read"]
    write_calls = [call for call in tool_calls if _classify_tool_name(_call_name(call)) == "write"]
    text = _message_text(message_history)
    observed_entities = _observed_entity_tokens(message_history)
    last_message = message_history[-1] if message_history else None
    last_is_assistant_response = (
        last_message is not None
        and last_message.__class__.__name__.lower().endswith("assistantmessage")
        and bool(getattr(last_message, "content", None))
        and not getattr(last_message, "tool_calls", None)
    )
    return Tau3ProgressPredicates(
        identity_verified=1.0 if any(token in text for token in IDENTITY_TOKENS) else 0.0,
        target_entity_resolved=1.0 if observed_entities else 0.0,
        required_read_completed=1.0 if read_calls and successful_outputs else 0.0,
        user_confirmation_obtained=1.0 if any(token in text for token in CONFIRMATION_TOKENS) else 0.0,
        required_policy_condition_checked=1.0 if read_calls and observed_entities else 0.0,
        write_action_completed=1.0 if write_calls and successful_outputs else 0.0,
        terminal_condition_met=1.0 if last_is_assistant_response else 0.0,
    )


def tau3_action_features(*, candidate: object, message_history: Sequence[object], max_tool_budget: int) -> ActionFeatures:
    tool_calls = list(getattr(candidate, "tool_calls", None) or [])
    content = getattr(candidate, "content", None)
    if not tool_calls:
        action_type = "respond" if content else "terminate"
        name = ""
        args = {}
    else:
        call = tool_calls[0]
        name = _call_name(call)
        args = _call_args(call)
        action_type = _classify_tool_name(name)
    observed_entities = _observed_entity_tokens(message_history)
    arg_values = [str(value).lower() for value in args.values()]
    references_observed = bool(arg_values) and any(any(token and token in value for token in observed_entities) for value in arg_values)
    references_unverified = bool(arg_values) and not references_observed and action_type == "write"
    duplicates = any(_call_name(call) == name and _call_args(call) == args for call in _assistant_tool_calls(message_history))
    return ActionFeatures(
        action_is_read=1.0 if action_type == "read" else 0.0,
        action_is_write=1.0 if action_type == "write" else 0.0,
        action_is_respond=1.0 if action_type == "respond" else 0.0,
        action_is_ask=1.0 if action_type == "ask" else 0.0,
        action_is_terminate=1.0 if action_type == "terminate" else 0.0,
        argument_completeness=_argument_completeness(args),
        references_observed_entity=1.0 if references_observed else 0.0,
        references_unverified_entity=1.0 if references_unverified else 0.0,
        expected_tool_cost_frac=1.0 / max(1, max_tool_budget) if tool_calls else 0.0,
        action_irreversible=1.0 if action_type == "write" else 0.0,
        precondition_satisfied=0.0 if references_unverified else 1.0,
        policy_risk=0.25 if action_type == "write" else 0.0,
        duplicates_previous_action=1.0 if duplicates else 0.0,
    )


def tau3_adapter_manifest() -> dict[str, object]:
    manifest = canonical_contract_manifest()
    manifest.update(
        {
            "adapter": "Tau3AdapterV2",
            "adapter_revision": "Tau3AdapterV2-progress-predicates",
            "target_interface_visible": True,
            "target_trajectories_hidden": True,
            "uses_tau3_public_tool_schema": True,
            "uses_tau3_train_trajectories": False,
            "uses_tau3_heldout_outcomes": False,
            "progress_predicates": list(Tau3ProgressPredicates(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0).values()),
            "progress_predicate_provenance": "runtime-visible message history, tool call history, and tool outputs only",
            "tool_identity_features": "none; names are used only for read/write class normalization",
            "oracle_visible_to_controller": False,
            "oracle_used_for_candidate_generation": False,
            "oracle_used_for_filtering": False,
        }
    )
    return manifest


def write_tau3_adapter_log(path: Path, candidates: Iterable[CanonicalCandidate], action_types: Iterable[str]) -> None:
    rows = [
        Tau3AdapterLogRow(
            candidate_id=candidate.candidate_id,
            action_type=action_type,
            feature_vector_length=len(candidate.feature_vector()),
            oracle_visible_to_controller=False,
            tool_name_used_as_feature=False,
        )
        for candidate, action_type in zip(candidates, action_types)
    ]
    path.write_text("\n".join(json.dumps(row.__dict__, sort_keys=True) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def _assistant_tool_calls(messages: Sequence[object]) -> List[object]:
    calls: List[object] = []
    for item in messages:
        calls.extend(list(getattr(item, "tool_calls", None) or []))
    return calls


def _call_name(call: object) -> str:
    return str(getattr(call, "name", "")).lower()


def _call_args(call: object) -> Mapping[str, object]:
    args = getattr(call, "arguments", {}) or {}
    return args if isinstance(args, Mapping) else {}


def _classify_tool_name(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in READ_TOKENS):
        return "read"
    if any(token in lowered for token in WRITE_TOKENS):
        return "write"
    return "read"


def _argument_completeness(args: Mapping[str, object]) -> float:
    if not args:
        return 0.0
    filled = sum(1 for value in args.values() if value not in (None, ""))
    return filled / len(args)


def _observed_entity_tokens(messages: Sequence[object]) -> set[str]:
    tokens: set[str] = set()
    for item in messages:
        text = str(getattr(item, "content", "") or "").lower()
        for raw in text.replace("{", " ").replace("}", " ").replace('"', " ").replace("'", " ").split():
            token = raw.strip(" ,.:;()[]")
            if len(token) >= 3 and any(char.isdigit() for char in token):
                tokens.add(token)
    return tokens


def _message_text(messages: Sequence[object]) -> str:
    return " ".join(str(getattr(item, "content", "") or "").lower() for item in messages)


def _tool_output_failed(message: object) -> bool:
    text = str(getattr(message, "content", "") or "").lower()
    return any(token in text for token in ("error", "failed", "not found", "invalid"))


def _stagnation_steps_frac(messages: Sequence[object]) -> float:
    recent = list(messages)[-3:]
    if not recent:
        return 0.0
    stagnant = sum(1 for item in recent if not getattr(item, "tool_calls", None) and not item.__class__.__name__.lower().endswith("toolmessage"))
    return stagnant / len(recent)
