from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Optional

from tau2.agent.base_agent import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import APICompatibleMessage, AssistantMessage, Message, MultiToolMessage, SystemMessage, ToolMessage, UserMessage
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


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
    ):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        self.candidate_count = max(1, int(candidate_count))

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
        candidates = self._sample_candidates(state)
        selected = self._select_candidate(candidates, state.messages)
        selected = self._attach_total_sampling_cost(selected, candidates)
        state.messages.append(selected)
        return selected, state

    def _select_candidate(self, candidates: List[AssistantMessage], history: List[APICompatibleMessage]) -> AssistantMessage:
        greedy = candidates[0]
        if greedy.content and not greedy.tool_calls:
            return greedy
        tool_candidates = [candidate for candidate in candidates if candidate.tool_calls and not candidate.content]
        if tool_candidates:
            return max(tool_candidates, key=lambda candidate: self._score_candidate(candidate, history))
        return greedy

    def _sample_candidates(self, state: LLMAgentState) -> List[AssistantMessage]:
        candidates: List[AssistantMessage] = []
        for idx in range(self.candidate_count):
            llm_args = deepcopy(self.llm_args)
            if "seed" in llm_args:
                llm_args["seed"] = int(llm_args["seed"]) + idx
            if idx > 0 and self.candidate_count > 1:
                llm_args["temperature"] = max(float(llm_args.get("temperature", 0.0)), 0.2)
            candidate = generate(
                model=self.llm,
                tools=self.tools,
                messages=state.system_messages + state.messages,
                call_name="cgr_solver_controller_candidate",
                **llm_args,
            )
            candidates.append(candidate)
        return candidates

    def _score_candidate(self, candidate: AssistantMessage, history: List[APICompatibleMessage]) -> tuple[float, str]:
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
        return score, candidate.model_dump_json()

    @staticmethod
    def _has_repeated_exact_tool_call(name: str, arguments: dict, history: List[APICompatibleMessage]) -> bool:
        for item in history:
            if not isinstance(item, AssistantMessage) or not item.tool_calls:
                continue
            for call in item.tool_calls:
                if call.name == name and call.arguments == arguments:
                    return True
        return False

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
    return Tau3SolverControllerAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=kwargs.get("llm"),
        llm_args=llm_args,
        candidate_count=candidate_count,
    )
