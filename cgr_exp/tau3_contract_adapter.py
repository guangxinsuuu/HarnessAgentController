from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence

from cgr_exp.harness_contract import ActionFeatures, CanonicalCandidate, RuntimeFeatures, canonical_contract_manifest


READ_TOKENS = ("get", "search", "list", "find", "lookup", "read", "retrieve")
WRITE_TOKENS = ("update", "create", "cancel", "book", "refund", "exchange", "modify", "send", "transfer", "delete")


@dataclass(frozen=True)
class Tau3AdapterLogRow:
    candidate_id: str
    action_type: str
    feature_vector_length: int
    oracle_visible_to_controller: bool
    tool_name_used_as_feature: bool


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
    return RuntimeFeatures(
        verified_progress_frac=0.0,
        verified_progress_known_mask=0.0,
        remaining_requirement_frac=0.0,
        remaining_requirement_known_mask=0.0,
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
            "adapter": "Tau3AdapterV1",
            "target_interface_visible": True,
            "target_trajectories_hidden": True,
            "uses_tau3_public_tool_schema": True,
            "uses_tau3_train_trajectories": False,
            "uses_tau3_heldout_outcomes": False,
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


def _tool_output_failed(message: object) -> bool:
    text = str(getattr(message, "content", "") or "").lower()
    return any(token in text for token in ("error", "failed", "not found", "invalid"))


def _stagnation_steps_frac(messages: Sequence[object]) -> float:
    recent = list(messages)[-3:]
    if not recent:
        return 0.0
    stagnant = sum(1 for item in recent if not getattr(item, "tool_calls", None) and not item.__class__.__name__.lower().endswith("toolmessage"))
    return stagnant / len(recent)

