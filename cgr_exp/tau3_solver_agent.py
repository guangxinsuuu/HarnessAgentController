from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from copy import deepcopy
import json
from pathlib import Path
import time
from typing import Any, List, Optional

from cgr_exp.harness_contract import canonical_contract_manifest, canonical_feature_names
from cgr_exp.harness_ranker import GenericHarnessRanker
from cgr_exp.tau3_contract_adapter import tau3_candidate_to_canonical, tau3_progress_predicates

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import APICompatibleMessage, AssistantMessage, Message, MultiToolMessage, SystemMessage, ToolCall, ToolMessage, UserMessage
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


@dataclass(frozen=True)
class _ActionSpaceCandidate:
    index: int
    intent: str
    action_type: str
    synthetic_message: AssistantMessage
    tool: Optional[Tool] = None


class Tau3SolverControllerAgent(LLMAgent):
    """Official tau2 half-duplex agent with a CGR candidate-selection layer.

    This adapter preserves tau2's environment, user simulator, tool execution,
    and evaluator. It changes only the agent decision step by sampling multiple
    candidate next messages and selecting one with a deterministic controller.
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: str,
        llm_args: Optional[dict] = None,
        candidate_count: int = 4,
        ranker_checkpoint: Optional[str] = None,
        max_turns: int = 200,
        max_tool_budget: int = 200,
        controller_trace_path: Optional[str] = None,
        enable_certified_filter: bool = False,
        candidate_workers: int = 1,
        candidate_retries: int = 2,
        candidate_retry_backoff: float = 0.5,
        adaptive_search: bool = False,
        min_candidate_count: int = 1,
        enable_candidate_diversity: bool = False,
        action_space_controller: bool = False,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.candidate_count = max(1, int(candidate_count))
        self.min_candidate_count = max(1, min(int(min_candidate_count), self.candidate_count))
        self.adaptive_search = bool(adaptive_search)
        self.enable_candidate_diversity = bool(enable_candidate_diversity)
        self.action_space_controller = bool(action_space_controller)
        self.candidate_workers = max(1, int(candidate_workers))
        self.candidate_retries = max(0, int(candidate_retries))
        self.candidate_retry_backoff = max(0.0, float(candidate_retry_backoff))
        self._last_sample_failures = 0
        self._last_sample_failure_types: list[str] = []
        self._last_candidate_generation_intents: list[str] = []
        self._last_search_budget = self.candidate_count
        self._last_search_reason = "fixed_candidate_count"
        self.max_turns = max(1, int(max_turns))
        self.max_tool_budget = max(1, int(max_tool_budget))
        self.ranker_checkpoint = ranker_checkpoint
        self.controller_trace_path = Path(controller_trace_path) if controller_trace_path else None
        self.enable_certified_filter = enable_certified_filter
        self.ranker: Optional[GenericHarnessRanker] = None
        if ranker_checkpoint:
            try:
                self.ranker = GenericHarnessRanker.from_json(Path(ranker_checkpoint).read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"failed to load tau3 solver ranker checkpoint {ranker_checkpoint}: {exc}") from exc

    def get_init_state(self, message_history: Optional[list[Message]] = None) -> LLMAgentState:
        if message_history is None:
            message_history = []
        return LLMAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
        if self.action_space_controller:
            selected = self._generate_action_space_message(state)
            state.messages.append(selected)
            return selected, state
        candidates = self._sample_candidates(state)
        selected = self._select_candidate(candidates, state.messages)
        selected = self._attach_total_sampling_cost(selected, candidates)
        state.messages.append(selected)
        return selected, state

    def _generate_action_space_message(self, state: LLMAgentState) -> AssistantMessage:
        history = state.messages
        candidates = self._available_action_space_candidates(history)
        scored_candidates = []
        pruned = self._certified_pruned_action_space_candidates(candidates)
        active_candidates = [candidate for candidate in candidates if candidate.index not in pruned]
        if not active_candidates:
            active_candidates = candidates
        for candidate in active_candidates:
            score, coverage = self._score_candidate(candidate.synthetic_message, history)
            score = self._calibrate_action_space_score(candidate, score, history)
            scored_candidates.append((candidate.index, score, coverage))
        selected_index = max(scored_candidates, key=lambda item: (item[1], -item[0]))[0] if scored_candidates else 0
        selected_candidate = next(candidate for candidate in candidates if candidate.index == selected_index)
        self._write_action_space_trace(
            candidates,
            selected_index=selected_index,
            scored_candidates=scored_candidates,
            pruned=pruned,
            history=history,
        )
        return self._instantiate_action_space_candidate(selected_candidate, state)

    def _available_action_space_candidates(self, history: List[APICompatibleMessage]) -> List[_ActionSpaceCandidate]:
        candidates: list[_ActionSpaceCandidate] = []
        progress = tau3_progress_predicates(message_history=history)
        for tool in self.tools:
            name = str(getattr(tool, "name", ""))
            if not name:
                continue
            action_type = self._classify_tool_name(name)
            precondition_satisfied = 1.0
            references_observed = 0.0
            references_unverified = 0.0
            policy_risk = 0.0
            if action_type == "write":
                precondition_satisfied = 1.0 if progress.target_entity_resolved and progress.required_read_completed else 0.0
                references_observed = 1.0 if precondition_satisfied else 0.0
                references_unverified = 0.0 if precondition_satisfied else 1.0
                policy_risk = 0.25 if precondition_satisfied else 0.75
            synthetic = AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name=name, arguments={})],
                raw_data={
                    "cgr_action_intent": "tool",
                    "cgr_action_type": action_type,
                    "cgr_argument_completeness": 0.0,
                    "cgr_references_observed_entity": references_observed,
                    "cgr_references_unverified_entity": references_unverified,
                    "cgr_declared_observation_gain": 0.5 if action_type == "read" else 0.0,
                    "cgr_declared_requirement_gain": 0.5 if action_type == "write" else 0.0,
                    "cgr_precondition_satisfied": precondition_satisfied,
                    "cgr_policy_risk": policy_risk,
                },
            )
            candidates.append(
                _ActionSpaceCandidate(
                    index=len(candidates),
                    intent="tool",
                    action_type=action_type,
                    synthetic_message=synthetic,
                    tool=tool,
                )
            )
        terminal_ready = self._terminal_ready_from_progress(progress)
        ask_ready = self._ask_ready_from_progress(progress)
        for intent, action_type in (("ask", "ask"), ("respond", "respond"), ("terminate", "terminate")):
            if action_type == "ask":
                precondition_satisfied = 1.0 if ask_ready else 0.0
                references_unverified = 1.0 if ask_ready else 0.0
                policy_risk = 0.1 if ask_ready else 0.6
            else:
                precondition_satisfied = 1.0 if terminal_ready else 0.0
                references_unverified = 0.0
                policy_risk = 0.05 if terminal_ready else 0.9
            synthetic = AssistantMessage(
                role="assistant",
                content=f"CGR meta action: {intent}",
                raw_data={
                    "cgr_action_intent": intent,
                    "cgr_action_type": action_type,
                    "cgr_argument_completeness": 0.0,
                    "cgr_references_observed_entity": 0.0,
                    "cgr_references_unverified_entity": references_unverified,
                    "cgr_declared_observation_gain": 0.0,
                    "cgr_declared_requirement_gain": 0.0,
                    "cgr_precondition_satisfied": precondition_satisfied,
                    "cgr_policy_risk": policy_risk,
                },
            )
            candidates.append(
                _ActionSpaceCandidate(
                    index=len(candidates),
                    intent=intent,
                    action_type=action_type,
                    synthetic_message=synthetic,
                )
            )
        return candidates

    def _instantiate_action_space_candidate(self, candidate: _ActionSpaceCandidate, state: LLMAgentState) -> AssistantMessage:
        messages = state.system_messages + state.messages
        llm_args = deepcopy(self.llm_args)
        if candidate.tool is not None:
            hint = (
                f"CGR selected next action intent: call exactly the `{candidate.tool.name}` tool once. "
                "Fill arguments only from the visible conversation and tool results. "
                "If a required argument is missing, ask the user instead of inventing it."
            )
            return generate(
                model=self.llm,
                tools=[candidate.tool],
                tool_choice="auto",
                messages=self._messages_with_extra_system_hint(messages, hint),
                call_name="cgr_action_space_tool_instantiation",
                **llm_args,
            )
        if candidate.intent == "ask":
            hint = "CGR selected next action intent: ask the user one concise clarification or confirmation. Do not call tools."
        elif candidate.intent == "terminate":
            hint = "CGR selected next action intent: provide the final concise answer and stop. Do not call tools."
        else:
            hint = "CGR selected next action intent: respond concisely to the user. Do not call tools."
        return generate(
            model=self.llm,
            tools=None,
            messages=self._messages_with_extra_system_hint(messages, hint),
            call_name="cgr_action_space_meta_instantiation",
            **llm_args,
        )

    def _certified_pruned_action_space_candidates(self, candidates: List[_ActionSpaceCandidate]) -> dict[int, str]:
        if not self.enable_certified_filter:
            return {}
        known_tool_names = {str(getattr(tool, "name", "")).lower() for tool in self.tools if getattr(tool, "name", "")}
        pruned: dict[int, str] = {}
        for candidate in candidates:
            if candidate.tool is None:
                continue
            tool_name = str(getattr(candidate.tool, "name", "")).lower()
            if tool_name not in known_tool_names:
                pruned[candidate.index] = "unknown_tool"
        if len(pruned) == len(candidates):
            return {}
        return pruned

    def _calibrate_action_space_score(self, candidate: _ActionSpaceCandidate, score: float, history: List[APICompatibleMessage]) -> float:
        progress = tau3_progress_predicates(message_history=history)
        if candidate.intent in ("respond", "terminate"):
            return score + 10.0 if self._terminal_ready_from_progress(progress) else score - 12.0
        if candidate.intent == "ask":
            return score + 3.0 if self._ask_ready_from_progress(progress) else score - 2.0
        if progress.write_action_completed:
            return score - 6.0
        if candidate.action_type == "write" and not self._write_ready_from_progress(progress):
            return score - 8.0
        return score

    @staticmethod
    def _write_ready_from_progress(progress) -> bool:
        return bool(progress.target_entity_resolved and progress.required_read_completed and progress.user_confirmation_obtained)

    @staticmethod
    def _ask_ready_from_progress(progress) -> bool:
        return bool(progress.target_entity_resolved and progress.required_read_completed and not progress.user_confirmation_obtained)

    @staticmethod
    def _terminal_ready_from_progress(progress) -> bool:
        return bool(progress.write_action_completed and progress.target_entity_resolved and progress.required_read_completed)

    def _select_candidate(self, candidates: List[AssistantMessage], history: List[APICompatibleMessage]) -> AssistantMessage:
        greedy = candidates[0]
        pruned = self._certified_pruned_candidates(candidates)
        active_indices = [idx for idx in range(len(candidates)) if idx not in pruned]
        if not active_indices:
            active_indices = list(range(len(candidates)))
        if greedy.content and not greedy.tool_calls:
            self._write_selection_trace(candidates, selected_index=0, scored_candidates=[], scoring_mode="greedy_content_passthrough", pruned=pruned, history=history)
            return greedy
        scored_candidates = []
        for idx, candidate in enumerate(candidates):
            if idx in active_indices and candidate.tool_calls and not candidate.content:
                score, coverage = self._score_candidate(candidate, history)
                scored_candidates.append((idx, score, coverage))
        if scored_candidates:
            selected_index = max(scored_candidates, key=lambda item: (item[1], -item[0]))[0]
            mode = "canonical_ranker" if self.ranker is not None else "heuristic"
            self._write_selection_trace(candidates, selected_index=selected_index, scored_candidates=scored_candidates, scoring_mode=mode, pruned=pruned, history=history)
            return candidates[selected_index]
        fallback_index = active_indices[0] if active_indices else 0
        self._write_selection_trace(candidates, selected_index=fallback_index, scored_candidates=[], scoring_mode="fallback_greedy", pruned=pruned, history=history)
        return candidates[fallback_index]

    def _sample_candidates(self, state: LLMAgentState) -> List[AssistantMessage]:
        messages = state.system_messages + state.messages
        self._last_sample_failures = 0
        self._last_sample_failure_types = []
        search_budget, reason = self._search_budget_for_state(state)
        self._last_search_budget = search_budget
        self._last_search_reason = reason
        self._last_candidate_generation_intents = [self._candidate_generation_intent(idx) for idx in range(search_budget)]
        if self.candidate_workers <= 1 or search_budget == 1:
            indexed = []
            for idx in range(search_budget):
                try:
                    indexed.append((idx, self._sample_candidate_with_retry(idx, messages)))
                except Exception as exc:
                    self._last_sample_failures += 1
                    self._last_sample_failure_types.append(type(exc).__name__)
            if not indexed:
                failures = ", ".join(self._last_sample_failure_types) or "unknown"
                raise RuntimeError(f"all tau3 controller candidate samples failed after retries: {failures}")
            return [candidate for _idx, candidate in sorted(indexed, key=lambda item: item[0])]
        workers = min(self.candidate_workers, search_budget)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._sample_candidate_with_retry, idx, messages): idx for idx in range(search_budget)}
            indexed = []
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    indexed.append((idx, future.result()))
                except Exception as exc:
                    self._last_sample_failures += 1
                    self._last_sample_failure_types.append(type(exc).__name__)
            if not indexed:
                failures = ", ".join(self._last_sample_failure_types) or "unknown"
                raise RuntimeError(f"all tau3 controller candidate samples failed after retries: {failures}")
            return [candidate for _idx, candidate in sorted(indexed, key=lambda item: item[0])]

    def _search_budget_for_state(self, state: LLMAgentState) -> tuple[int, str]:
        if not self.adaptive_search:
            return self.candidate_count, "fixed_candidate_count"
        used_tool_calls = sum(len(getattr(item, "tool_calls", None) or []) for item in state.messages)
        runtime = tau3_candidate_to_canonical(
            candidate_id="tau3-search-budget-state",
            message_history=state.messages,
            tools=self.tools,
            candidate=AssistantMessage(role="assistant", content=""),
            remaining_turns=max(0, self.max_turns - len(state.messages)),
            max_turns=self.max_turns,
            remaining_tool_budget=max(0, self.max_tool_budget - used_tool_calls),
            max_tool_budget=self.max_tool_budget,
        ).state
        reasons = []
        if runtime.previous_failure_rate > 0.0:
            reasons.append("previous_failure")
        if runtime.stagnation_steps_frac >= 0.5:
            reasons.append("stagnation")
        if runtime.uncertainty_score >= 0.8:
            reasons.append("high_uncertainty")
        if runtime.irreversible_action_used > 0.0:
            reasons.append("post_write_state")
        if runtime.remaining_turn_budget_frac <= 0.25 or runtime.remaining_tool_budget_frac <= 0.25:
            reasons.append("low_budget")
        if reasons:
            return self.candidate_count, "adaptive_high_risk:" + ",".join(reasons)
        return self.min_candidate_count, "adaptive_low_risk"

    def _sample_candidate_with_retry(self, idx: int, messages: List[APICompatibleMessage]) -> AssistantMessage:
        last_exc: Optional[Exception] = None
        for attempt in range(self.candidate_retries + 1):
            try:
                return self._sample_candidate(idx, messages)
            except Exception as exc:
                last_exc = exc
                if attempt < self.candidate_retries and self.candidate_retry_backoff > 0.0:
                    time.sleep(self.candidate_retry_backoff * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def _sample_candidate(self, idx: int, messages: List[APICompatibleMessage]) -> AssistantMessage:
        llm_args = deepcopy(self.llm_args)
        intent = self._candidate_generation_intent(idx)
        sample_messages = self._messages_with_candidate_intent(messages, intent)
        if "seed" in llm_args:
            llm_args["seed"] = int(llm_args["seed"]) + idx
        if idx > 0 and self.candidate_count > 1:
            llm_args["temperature"] = max(float(llm_args.get("temperature", 0.0)), 0.2)
        return generate(
            model=self.llm,
            tools=self.tools,
            messages=sample_messages,
            call_name="cgr_solver_controller_candidate",
            **llm_args,
        )

    def _candidate_generation_intent(self, idx: int) -> str:
        if not self.enable_candidate_diversity or idx == 0:
            return "greedy"
        intents = ["prefer_read", "prefer_write", "prefer_ask_or_respond"]
        return intents[(idx - 1) % len(intents)]

    def _messages_with_candidate_intent(self, messages: List[APICompatibleMessage], intent: str) -> List[APICompatibleMessage]:
        if intent == "greedy":
            return messages
        hint = _candidate_diversity_hint(intent)
        if not hint:
            return messages
        return self._messages_with_extra_system_hint(messages, hint)

    @staticmethod
    def _messages_with_extra_system_hint(messages: List[APICompatibleMessage], hint: str) -> List[APICompatibleMessage]:
        hinted = list(messages)
        insert_at = 1 if hinted and isinstance(hinted[0], SystemMessage) else 0
        hinted.insert(insert_at, SystemMessage(role="system", content=hint))
        return hinted

    def _score_candidate(self, candidate: AssistantMessage, history: List[APICompatibleMessage]) -> tuple[float, dict[str, float]]:
        if self.ranker is not None:
            return self._score_canonical_candidate(candidate, history)
        return self._score_heuristic_candidate(candidate, history), {}

    def _score_canonical_candidate(self, candidate: AssistantMessage, history: List[APICompatibleMessage]) -> tuple[float, dict[str, float]]:
        canonical = self._canonical_candidate(candidate, history)
        assert self.ranker is not None
        coverage = canonical.state.known_mask_values()
        coverage["mean_known_mask_ratio"] = canonical.state.known_mask_ratio()
        return self.ranker.score(canonical.feature_vector()), coverage

    def _canonical_candidate(self, candidate: AssistantMessage, history: List[APICompatibleMessage]):
        used_tool_calls = sum(len(getattr(item, "tool_calls", None) or []) for item in history)
        return tau3_candidate_to_canonical(
            candidate_id=f"tau3-candidate-{len(history)}",
            message_history=history,
            tools=self.tools,
            candidate=candidate,
            remaining_turns=max(0, self.max_turns - len(history)),
            max_turns=self.max_turns,
            remaining_tool_budget=max(0, self.max_tool_budget - used_tool_calls),
            max_tool_budget=self.max_tool_budget,
        )

    def _score_heuristic_candidate(self, candidate: AssistantMessage, history: List[APICompatibleMessage]) -> float:
        score = 0.0
        if candidate.tool_calls:
            score += 3.0
            if len(candidate.tool_calls) == 1:
                score += 0.5
            for tool_call in candidate.tool_calls:
                name = tool_call.name.lower()
                if not self._has_repeated_exact_tool_call(tool_call.name, tool_call.arguments, history):
                    score += 0.75
                if any(token in name for token in ("update", "create", "cancel", "book", "refund", "exchange", "modify", "send", "transfer")):
                    score += 0.35
                if any(token in name for token in ("get", "search", "list", "find", "lookup")):
                    score += 0.15
        if candidate.content:
            text = candidate.content.lower()
            if candidate.tool_calls:
                score -= 2.0
            else:
                score += 1.0
            if any(token in text for token in ("done", "completed", "confirmed", "anything else", "help you with")):
                score += 0.4
            if "sorry" in text or "can't" in text or "cannot" in text:
                score -= 0.2
        return score

    @staticmethod
    def _has_repeated_exact_tool_call(name: str, arguments: dict, history: List[APICompatibleMessage]) -> bool:
        for item in history:
            if not isinstance(item, AssistantMessage) or not item.tool_calls:
                continue
            for call in item.tool_calls:
                if call.name == name and call.arguments == arguments:
                    return True
        return False

    def _certified_pruned_candidates(self, candidates: List[AssistantMessage]) -> dict[int, str]:
        if not self.enable_certified_filter:
            return {}
        known_tool_names = set()
        for tool in self.tools:
            name = str(getattr(tool, "name", "")).lower()
            if name:
                known_tool_names.add(name)
                known_tool_names.add(name.lstrip("_"))
        pruned: dict[int, str] = {}
        for idx, candidate in enumerate(candidates):
            tool_calls = list(getattr(candidate, "tool_calls", None) or [])
            content = getattr(candidate, "content", None)
            if not tool_calls and not content:
                pruned[idx] = "empty_assistant_message"
                continue
            for call in tool_calls:
                call_name = str(getattr(call, "name", "")).lower()
                if call_name not in known_tool_names and call_name.lstrip("_") not in known_tool_names:
                    pruned[idx] = "unknown_tool"
                    break
                if not isinstance(getattr(call, "arguments", None), dict):
                    pruned[idx] = "non_dict_arguments"
                    break
        if len(pruned) == len(candidates):
            return {}
        return pruned

    def _write_selection_trace(
        self,
        candidates: List[AssistantMessage],
        *,
        selected_index: int,
        scored_candidates: List[tuple[int, float, dict[str, float]]],
        scoring_mode: str,
        pruned: dict[int, str],
        history: List[APICompatibleMessage],
    ) -> None:
        if self.controller_trace_path is None:
            return
        self.controller_trace_path.parent.mkdir(parents=True, exist_ok=True)
        scored_by_index = {idx: score for idx, score, _coverage in scored_candidates}
        coverage_by_index = {idx: coverage for idx, _score, coverage in scored_candidates}
        coverage_values = [coverage for _idx, _score, coverage in scored_candidates if coverage]
        per_feature_availability: dict[str, float] = {}
        if coverage_values:
            feature_names = sorted({name for coverage in coverage_values for name in coverage if name != "mean_known_mask_ratio"})
            per_feature_availability = {
                name: sum(float(coverage.get(name, 0.0)) for coverage in coverage_values) / len(coverage_values)
                for name in feature_names
            }
        mean_known_mask_ratio = (
            sum(float(coverage.get("mean_known_mask_ratio", 0.0)) for coverage in coverage_values) / len(coverage_values)
            if coverage_values
            else None
        )
        contract = canonical_contract_manifest()
        history_progress = tau3_progress_predicates(message_history=history)
        row = {
            "event": "tau3_controller_selection",
            "timestamp": time.time(),
            "scoring_mode": scoring_mode,
            "canonical_contract": contract["contract"],
            "canonical_feature_schema_hash": contract["feature_schema_hash"],
            "canonical_feature_count": len(canonical_feature_names()),
            "canonical_mean_known_mask_ratio": mean_known_mask_ratio,
            "canonical_per_feature_availability": per_feature_availability,
            "canonical_progress_predicates": history_progress.values(),
            "canonical_progress_predicate_provenance": history_progress.provenance(),
            "candidate_count": len(candidates),
            "configured_candidate_count": self.candidate_count,
            "search_budget": self._last_search_budget,
            "adaptive_search_enabled": self.adaptive_search,
            "search_budget_reason": self._last_search_reason,
            "candidate_diversity_enabled": self.enable_candidate_diversity,
            "candidate_generation_intents": self._last_candidate_generation_intents,
            "candidate_sample_failures": self._last_sample_failures,
            "candidate_sample_failure_types": sorted(set(self._last_sample_failure_types)),
            "candidate_retries": self.candidate_retries,
            "candidate_workers": self.candidate_workers,
            "selected_index": selected_index,
            "certified_filter_enabled": self.enable_certified_filter,
            "certified_pruned_count": len(pruned),
            "certified_prune_reasons": sorted(set(pruned.values())),
            "ranker_checkpoint_loaded": self.ranker is not None,
            "oracle_visible_to_controller": False,
            "oracle_used_for_candidate_generation": False,
            "oracle_used_for_filtering": False,
            "tool_names_used_as_features": False,
            "tool_name_use_policy": "coarse_action_type_normalization_only",
            "candidate_scores": [
                {
                    "candidate_index": idx,
                    "candidate_generation_intent": self._last_candidate_generation_intents[idx] if idx < len(self._last_candidate_generation_intents) else "unknown",
                    "score": scored_by_index.get(idx, None),
                    "known_mask_ratio": coverage_by_index.get(idx, {}).get("mean_known_mask_ratio"),
                    "certified_pruned": idx in pruned,
                    "certified_prune_reason": pruned.get(idx, None),
                    "action_type": self._candidate_action_type(candidate),
                    "tool_call_count": len(getattr(candidate, "tool_calls", None) or []),
                    "has_content": bool(getattr(candidate, "content", None)),
                }
                for idx, candidate in enumerate(candidates)
            ],
        }
        with self.controller_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def _write_action_space_trace(
        self,
        candidates: List[_ActionSpaceCandidate],
        *,
        selected_index: int,
        scored_candidates: List[tuple[int, float, dict[str, float]]],
        pruned: dict[int, str],
        history: List[APICompatibleMessage],
    ) -> None:
        if self.controller_trace_path is None:
            return
        self.controller_trace_path.parent.mkdir(parents=True, exist_ok=True)
        scored_by_index = {idx: score for idx, score, _coverage in scored_candidates}
        coverage_by_index = {idx: coverage for idx, _score, coverage in scored_candidates}
        coverage_values = [coverage for _idx, _score, coverage in scored_candidates if coverage]
        per_feature_availability: dict[str, float] = {}
        if coverage_values:
            feature_names = sorted({name for coverage in coverage_values for name in coverage if name != "mean_known_mask_ratio"})
            per_feature_availability = {
                name: sum(float(coverage.get(name, 0.0)) for coverage in coverage_values) / len(coverage_values)
                for name in feature_names
            }
        mean_known_mask_ratio = (
            sum(float(coverage.get("mean_known_mask_ratio", 0.0)) for coverage in coverage_values) / len(coverage_values)
            if coverage_values
            else None
        )
        contract = canonical_contract_manifest()
        history_progress = tau3_progress_predicates(message_history=history)
        row = {
            "event": "tau3_controller_selection",
            "timestamp": time.time(),
            "scoring_mode": "action_space_canonical_ranker" if self.ranker is not None else "action_space_heuristic",
            "canonical_contract": contract["contract"],
            "canonical_feature_schema_hash": contract["feature_schema_hash"],
            "canonical_feature_count": len(canonical_feature_names()),
            "canonical_mean_known_mask_ratio": mean_known_mask_ratio,
            "canonical_per_feature_availability": per_feature_availability,
            "canonical_progress_predicates": history_progress.values(),
            "canonical_progress_predicate_provenance": history_progress.provenance(),
            "candidate_count": len(candidates),
            "configured_candidate_count": len(candidates),
            "search_budget": 1,
            "adaptive_search_enabled": self.adaptive_search,
            "search_budget_reason": "action_space_one_instantiation",
            "candidate_diversity_enabled": False,
            "action_space_controller_enabled": True,
            "candidate_generation_intents": [candidate.intent for candidate in candidates],
            "candidate_sample_failures": 0,
            "candidate_sample_failure_types": [],
            "candidate_retries": self.candidate_retries,
            "candidate_workers": self.candidate_workers,
            "selected_index": selected_index,
            "certified_filter_enabled": self.enable_certified_filter,
            "certified_pruned_count": len(pruned),
            "certified_prune_reasons": sorted(set(pruned.values())),
            "ranker_checkpoint_loaded": self.ranker is not None,
            "oracle_visible_to_controller": False,
            "oracle_used_for_candidate_generation": False,
            "oracle_used_for_filtering": False,
            "tool_names_used_as_features": False,
            "tool_name_use_policy": "coarse_action_type_normalization_only",
            "candidate_scores": [
                {
                    "candidate_index": candidate.index,
                    "candidate_generation_intent": candidate.intent,
                    "score": scored_by_index.get(candidate.index, None),
                    "known_mask_ratio": coverage_by_index.get(candidate.index, {}).get("mean_known_mask_ratio"),
                    "certified_pruned": candidate.index in pruned,
                    "certified_prune_reason": pruned.get(candidate.index, None),
                    "action_type": candidate.action_type,
                    "tool_call_count": 1 if candidate.tool is not None else 0,
                    "has_content": candidate.tool is None,
                }
                for candidate in candidates
            ],
        }
        with self.controller_trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    @staticmethod
    def _candidate_action_type(candidate: AssistantMessage) -> str:
        tool_calls = list(getattr(candidate, "tool_calls", None) or [])
        if not tool_calls:
            raw_data = getattr(candidate, "raw_data", None)
            if isinstance(raw_data, dict):
                meta_intent = str(raw_data.get("cgr_action_intent", "")).lower()
                if meta_intent in ("ask", "respond", "terminate"):
                    return meta_intent
            return "respond" if getattr(candidate, "content", None) else "terminate"
        return Tau3SolverControllerAgent._classify_tool_name(tool_calls[0].name)

    @staticmethod
    def _classify_tool_name(name: str) -> str:
        lowered = name.lower()
        if any(token in lowered for token in ("update", "create", "cancel", "book", "refund", "exchange", "modify", "send", "transfer", "delete")):
            return "write"
        return "read"

    @staticmethod
    def _attach_total_sampling_cost(selected: AssistantMessage, candidates: List[AssistantMessage]) -> AssistantMessage:
        total_cost: Optional[float] = 0.0
        total_prompt = 0
        total_completion = 0
        usage_seen = False
        for candidate in candidates:
            if candidate.cost is None:
                total_cost = None
            elif total_cost is not None:
                total_cost += candidate.cost
            if candidate.usage:
                usage_seen = True
                total_prompt += int(candidate.usage.get("prompt_tokens") or 0)
                total_completion += int(candidate.usage.get("completion_tokens") or 0)
        selected.cost = total_cost
        if usage_seen:
            selected.usage = {"prompt_tokens": total_prompt, "completion_tokens": total_completion}
        return selected


def create_tau3_solver_controller_agent(tools, domain_policy, **kwargs):
    llm_args = deepcopy(kwargs.get("llm_args") or {})
    candidate_count = int(llm_args.pop("cgr_candidate_count", 4))
    candidate_workers = int(llm_args.pop("cgr_candidate_workers", 1))
    candidate_retries = int(llm_args.pop("cgr_candidate_retries", 2))
    candidate_retry_backoff = float(llm_args.pop("cgr_candidate_retry_backoff", 0.5))
    adaptive_search = bool(llm_args.pop("cgr_adaptive_search", False))
    min_candidate_count = int(llm_args.pop("cgr_min_candidate_count", 1))
    enable_candidate_diversity = bool(llm_args.pop("cgr_enable_candidate_diversity", False))
    action_space_controller = bool(llm_args.pop("cgr_action_space_controller", False))
    ranker_checkpoint = llm_args.pop("cgr_ranker_checkpoint", None)
    max_turns = int(llm_args.pop("cgr_max_turns", 200))
    max_tool_budget = int(llm_args.pop("cgr_max_tool_budget", max_turns))
    controller_trace_path = llm_args.pop("cgr_controller_trace_path", None)
    enable_certified_filter = bool(llm_args.pop("cgr_enable_certified_filter", False))
    return Tau3SolverControllerAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=llm_args,
        candidate_count=candidate_count,
        candidate_workers=candidate_workers,
        candidate_retries=candidate_retries,
        candidate_retry_backoff=candidate_retry_backoff,
        adaptive_search=adaptive_search,
        min_candidate_count=min_candidate_count,
        enable_candidate_diversity=enable_candidate_diversity,
        action_space_controller=action_space_controller,
        ranker_checkpoint=ranker_checkpoint,
        max_turns=max_turns,
        max_tool_budget=max_tool_budget,
        controller_trace_path=controller_trace_path,
        enable_certified_filter=enable_certified_filter,
    )


def _candidate_diversity_hint(intent: str) -> str:
    if intent == "prefer_read":
        return (
            "CGR candidate diversification hint: if a safe information-gathering action is useful, "
            "prefer a read/search/list/lookup style tool call. Avoid repeating an exact previous tool call. "
            "If no such action is appropriate, answer normally."
        )
    if intent == "prefer_write":
        return (
            "CGR candidate diversification hint: if the required entity and user authorization are already visible, "
            "prefer one safe state-changing tool call. Do not invent missing identifiers or confirmations; ask or read instead."
        )
    if intent == "prefer_ask_or_respond":
        return (
            "CGR candidate diversification hint: if required information, identity, or confirmation is missing, "
            "prefer asking the user a concise clarification. If the task is complete, respond concisely instead of calling a tool."
        )
    return ""
