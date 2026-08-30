from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random
from typing import Dict, List, Sequence, Tuple

from cgr_exp.harness_contract import ActionFeatures, RuntimeFeatures


WORKFLOW_FAMILIES = [
    "approval",
    "incident",
    "procurement",
    "reproducibility",
    "support",
]


@dataclass(frozen=True)
class WorkflowTool:
    name: str
    action_type: str
    reveals_fact: int | None = None
    completes_requirement: int | None = None
    precondition_fact: int | None = None
    irreversible: bool = False
    policy_risk: float = 0.0
    cost: int = 1


@dataclass(frozen=True)
class WorkflowAction:
    tool_index: int


@dataclass(frozen=True)
class WorkflowState:
    observed_facts: Tuple[bool, ...]
    completed_requirements: Tuple[bool, ...]
    failed_actions: Tuple[int, ...]
    used_actions: Tuple[int, ...]
    remaining_turns: int
    remaining_tool_budget: int
    violations: int = 0

    def key(self) -> Tuple[object, ...]:
        return (
            self.observed_facts,
            self.completed_requirements,
            tuple(sorted(set(self.failed_actions))),
            len(self.used_actions),
            self.remaining_turns,
            self.remaining_tool_budget,
            self.violations,
        )


@dataclass(frozen=True)
class WorkflowTask:
    task_id: str
    family: str
    fact_count: int
    requirement_count: int
    tools: Tuple[WorkflowTool, ...]
    turn_budget: int
    tool_budget: int

    def root_state(self) -> WorkflowState:
        return WorkflowState(
            observed_facts=tuple(False for _ in range(self.fact_count)),
            completed_requirements=tuple(False for _ in range(self.requirement_count)),
            failed_actions=(),
            used_actions=(),
            remaining_turns=self.turn_budget,
            remaining_tool_budget=self.tool_budget,
        )

    def candidate_actions(self, state: WorkflowState) -> List[WorkflowAction]:
        if state.remaining_turns <= 0 or state.remaining_tool_budget <= 0:
            return []
        used = set(state.used_actions)
        return [WorkflowAction(idx) for idx in range(len(self.tools)) if idx not in used]

    def apply(self, state: WorkflowState, action: WorkflowAction) -> WorkflowState:
        tool = self.tools[action.tool_index]
        observed = list(state.observed_facts)
        completed = list(state.completed_requirements)
        failed = list(state.failed_actions)
        used = [*state.used_actions, action.tool_index]
        violations = state.violations
        precondition_ok = tool.precondition_fact is None or state.observed_facts[tool.precondition_fact]
        if not precondition_ok:
            failed.append(action.tool_index)
            violations += 1
        else:
            if tool.reveals_fact is not None:
                observed[tool.reveals_fact] = True
            if tool.completes_requirement is not None:
                completed[tool.completes_requirement] = True
        return WorkflowState(
            observed_facts=tuple(observed),
            completed_requirements=tuple(completed),
            failed_actions=tuple(failed),
            used_actions=tuple(used),
            remaining_turns=max(0, state.remaining_turns - 1),
            remaining_tool_budget=max(0, state.remaining_tool_budget - tool.cost),
            violations=violations,
        )

    def success(self, state: WorkflowState) -> bool:
        return all(state.completed_requirements) and state.violations == 0

    def reward(self, state: WorkflowState) -> float:
        completed = sum(1 for item in state.completed_requirements if item)
        progress = completed / max(1, self.requirement_count)
        return (100.0 if self.success(state) else 0.0) + 10.0 * progress - 8.0 * state.violations - len(state.used_actions)


def generate_workflow_source_tasks(count: int, seed: int = 0, anonymize_tools: bool = True) -> List[WorkflowTask]:
    rng = random.Random(seed)
    tasks = []
    for idx in range(count):
        family = WORKFLOW_FAMILIES[idx % len(WORKFLOW_FAMILIES)]
        fact_count = rng.randint(2, 5)
        requirement_count = rng.randint(2, 5)
        turn_budget = requirement_count + fact_count + rng.randint(1, 4)
        tool_budget = turn_budget
        tools = _make_tools(rng, idx, family, fact_count, requirement_count, anonymize_tools)
        tasks.append(
            WorkflowTask(
                task_id=f"workflow-{seed}-{idx}",
                family=family,
                fact_count=fact_count,
                requirement_count=requirement_count,
                tools=tuple(tools),
                turn_budget=turn_budget,
                tool_budget=tool_budget,
            )
        )
    return tasks


def workflow_runtime_features(task: WorkflowTask, state: WorkflowState) -> RuntimeFeatures:
    completed = sum(1 for item in state.completed_requirements if item)
    observed = sum(1 for item in state.observed_facts if item)
    used_read = sum(1 for idx in state.used_actions if task.tools[idx].action_type == "read")
    used_write = sum(1 for idx in state.used_actions if task.tools[idx].action_type == "write")
    valid = task.candidate_actions(state)
    valid_count = sum(1 for action in valid if workflow_action_features(task, state, action).precondition_satisfied >= 1.0)
    return RuntimeFeatures(
        verified_progress_frac=completed / max(1, task.requirement_count),
        verified_progress_known_mask=1.0,
        remaining_requirement_frac=(task.requirement_count - completed) / max(1, task.requirement_count),
        remaining_requirement_known_mask=1.0,
        observed_fact_frac=observed / max(1, task.fact_count),
        observed_fact_known_mask=1.0,
        remaining_turn_budget_frac=state.remaining_turns / max(1, task.turn_budget),
        remaining_tool_budget_frac=state.remaining_tool_budget / max(1, task.tool_budget),
        constraint_pressure=min(1.0, state.violations / 3.0),
        constraint_pressure_known_mask=1.0,
        valid_action_ratio=valid_count / max(1, len(valid)),
        previous_failure_rate=len(state.failed_actions) / max(1, len(state.used_actions)),
        stagnation_steps_frac=_stagnation_frac(task, state),
        uncertainty_score=1.0 - observed / max(1, task.fact_count),
        read_write_balance=(used_read - used_write) / max(1, len(state.used_actions)),
        irreversible_action_used=1.0 if any(task.tools[idx].irreversible for idx in state.used_actions) else 0.0,
    )


def workflow_action_features(task: WorkflowTask, state: WorkflowState, action: WorkflowAction) -> ActionFeatures:
    tool = task.tools[action.tool_index]
    precondition_ok = tool.precondition_fact is None or state.observed_facts[tool.precondition_fact]
    duplicates = action.tool_index in state.used_actions
    references_observed = tool.precondition_fact is not None and state.observed_facts[tool.precondition_fact]
    references_unverified = tool.precondition_fact is not None and not state.observed_facts[tool.precondition_fact]
    return ActionFeatures(
        action_is_read=1.0 if tool.action_type == "read" else 0.0,
        action_is_write=1.0 if tool.action_type == "write" else 0.0,
        action_is_respond=1.0 if tool.action_type == "respond" else 0.0,
        action_is_ask=1.0 if tool.action_type == "ask" else 0.0,
        action_is_terminate=1.0 if tool.action_type == "terminate" else 0.0,
        argument_completeness=1.0,
        references_observed_entity=1.0 if references_observed else 0.0,
        references_unverified_entity=1.0 if references_unverified else 0.0,
        expected_tool_cost_frac=tool.cost / max(1, task.tool_budget),
        action_irreversible=1.0 if tool.irreversible else 0.0,
        precondition_satisfied=1.0 if precondition_ok else 0.0,
        policy_risk=tool.policy_risk,
        duplicates_previous_action=1.0 if duplicates else 0.0,
    )


def workflow_action_intent_features(task: WorkflowTask, state: WorkflowState, action: WorkflowAction) -> ActionFeatures:
    tool = task.tools[action.tool_index]
    precondition_ok = tool.precondition_fact is None or state.observed_facts[tool.precondition_fact]
    duplicates = action.tool_index in state.used_actions
    references_observed = tool.precondition_fact is not None and state.observed_facts[tool.precondition_fact]
    references_unverified = tool.precondition_fact is not None and not state.observed_facts[tool.precondition_fact]
    return ActionFeatures(
        action_is_read=1.0 if tool.action_type == "read" else 0.0,
        action_is_write=1.0 if tool.action_type == "write" else 0.0,
        action_is_respond=1.0 if tool.action_type == "respond" else 0.0,
        action_is_ask=1.0 if tool.action_type == "ask" else 0.0,
        action_is_terminate=1.0 if tool.action_type == "terminate" else 0.0,
        argument_completeness=0.0,
        references_observed_entity=1.0 if references_observed else 0.0,
        references_unverified_entity=1.0 if references_unverified else 0.0,
        expected_tool_cost_frac=tool.cost / max(1, task.tool_budget),
        action_irreversible=1.0 if tool.irreversible else 0.0,
        precondition_satisfied=1.0 if precondition_ok else 0.0,
        policy_risk=tool.policy_risk + (0.5 if tool.action_type == "write" and not precondition_ok else 0.0),
        duplicates_previous_action=1.0 if duplicates else 0.0,
    )


def source_generator_manifest(seed: int, count: int, anonymize_tools: bool = True) -> Dict[str, object]:
    distribution = _source_generator_distribution(anonymize_tools)
    return {
        "source": "PolicyConstrainedWorkflowGeneratorV1",
        "seed": seed,
        "task_count": count,
        "families": WORKFLOW_FAMILIES,
        "distribution_hash": hashlib.sha256(json.dumps(distribution, sort_keys=True).encode("utf-8")).hexdigest(),
        "anonymized_tool_names": anonymize_tools,
        "tau3_trajectories_used": False,
        "tau3_heldout_test_seen_before_freeze": False,
        "source_generator_distribution_fixed_before_target_test_evaluation": True,
        "parameter_distribution": distribution,
    }


def _source_generator_distribution(anonymize_tools: bool) -> Dict[str, object]:
    return {
        "fact_count": "uniform_int[2,5]",
        "requirement_count": "uniform_int[2,5]",
        "turn_budget": "requirements + facts + uniform_int[1,4]",
        "families": {"schedule": "round_robin", "values": list(WORKFLOW_FAMILIES)},
        "tool_names": "anonymous source_tool_{task}_{index}" if anonymize_tools else "family_action_{index}",
    }


def _make_tools(
    rng: random.Random,
    task_idx: int,
    family: str,
    fact_count: int,
    requirement_count: int,
    anonymize: bool,
) -> List[WorkflowTool]:
    tools: List[WorkflowTool] = []
    for fact in range(fact_count):
        tools.append(
            WorkflowTool(
                name=_tool_name(task_idx, len(tools), family, "read", anonymize),
                action_type="read",
                reveals_fact=fact,
                policy_risk=0.0,
            )
        )
    for req in range(requirement_count):
        needs_fact = rng.randrange(fact_count) if rng.random() < 0.75 else None
        tools.append(
            WorkflowTool(
                name=_tool_name(task_idx, len(tools), family, "write", anonymize),
                action_type="write",
                completes_requirement=req,
                precondition_fact=needs_fact,
                irreversible=rng.random() < 0.35,
                policy_risk=0.1 + 0.4 * rng.random(),
            )
        )
    tools.append(WorkflowTool(name=_tool_name(task_idx, len(tools), family, "ask", anonymize), action_type="ask"))
    tools.append(WorkflowTool(name=_tool_name(task_idx, len(tools), family, "terminate", anonymize), action_type="terminate"))
    return tools


def _tool_name(task_idx: int, tool_idx: int, family: str, action_type: str, anonymize: bool) -> str:
    if anonymize:
        return f"source_tool_{task_idx}_{tool_idx}"
    return f"{family}_{action_type}_{tool_idx}"


def _stagnation_frac(task: WorkflowTask, state: WorkflowState) -> float:
    last_actions = state.used_actions[-3:]
    if not last_actions:
        return 0.0
    progress_tools = {idx for idx, tool in enumerate(task.tools) if tool.reveals_fact is not None or tool.completes_requirement is not None}
    stagnant = sum(1 for idx in last_actions if idx not in progress_tools)
    return stagnant / len(last_actions)
