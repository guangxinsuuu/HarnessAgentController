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

SOURCE_WORKFLOW_FAMILIES = [
    "approval",
    "incident",
    "procurement",
]

TARGET_WORKFLOW_FAMILIES = [
    "academic",
    "banking",
    "farm",
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
    decoy: bool = False


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
    terminal: bool = False

    def key(self) -> Tuple[object, ...]:
        return (
            self.observed_facts,
            self.completed_requirements,
            tuple(sorted(set(self.failed_actions))),
            len(self.used_actions),
            self.remaining_turns,
            self.remaining_tool_budget,
            self.violations,
            self.terminal,
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
        if state.terminal or state.remaining_turns <= 0 or state.remaining_tool_budget <= 0:
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
        terminal = False
        all_requirements_done = all(state.completed_requirements)
        if tool.action_type in ("respond", "terminate"):
            terminal = True
            precondition_ok = all_requirements_done
        elif tool.action_type == "ask":
            precondition_ok = any(not fact for fact in state.observed_facts)
        else:
            precondition_ok = tool.precondition_fact is None or state.observed_facts[tool.precondition_fact]
        if not precondition_ok:
            failed.append(action.tool_index)
            violations += 1
        else:
            if tool.action_type == "ask" and tool.reveals_fact is None:
                for idx, value in enumerate(observed):
                    if not value:
                        observed[idx] = True
                        break
            elif tool.reveals_fact is not None:
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
            terminal=terminal,
        )

    def success(self, state: WorkflowState) -> bool:
        return state.terminal and all(state.completed_requirements) and state.violations == 0

    def reward(self, state: WorkflowState) -> float:
        completed = sum(1 for item in state.completed_requirements if item)
        progress = completed / max(1, self.requirement_count)
        terminal_penalty = 20.0 if state.terminal and not self.success(state) else 0.0
        return (100.0 if self.success(state) else 0.0) + 10.0 * progress - 8.0 * state.violations - terminal_penalty - len(state.used_actions)


def generate_workflow_source_tasks(
    count: int,
    seed: int = 0,
    anonymize_tools: bool = True,
    families: Sequence[str] | None = None,
    difficulty: str = "v1",
) -> List[WorkflowTask]:
    rng = random.Random(seed)
    family_list = list(families or WORKFLOW_FAMILIES)
    if not family_list:
        raise ValueError("at least one workflow family is required")
    if difficulty not in ("v1", "v2"):
        raise ValueError(f"unknown workflow difficulty: {difficulty}")
    tasks = []
    for idx in range(count):
        family = family_list[idx % len(family_list)]
        if difficulty == "v2":
            fact_count = rng.randint(3, 4)
            requirement_count = rng.randint(2, 3)
            turn_budget = requirement_count + fact_count + rng.randint(0, 1)
        else:
            fact_count = rng.randint(2, 5)
            requirement_count = rng.randint(2, 5)
            turn_budget = requirement_count + fact_count + rng.randint(1, 4)
        tool_budget = turn_budget
        tools = _make_tools(rng, idx, family, fact_count, requirement_count, anonymize_tools, difficulty=difficulty)
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
    precondition_ok = _action_precondition_ok(task, state, tool)
    duplicates = action.tool_index in state.used_actions
    references_observed = tool.precondition_fact is not None and state.observed_facts[tool.precondition_fact]
    references_unverified = _references_unverified(state, tool)
    return ActionFeatures(
        action_is_read=1.0 if tool.action_type == "read" else 0.0,
        action_is_write=1.0 if tool.action_type == "write" else 0.0,
        action_is_respond=1.0 if tool.action_type == "respond" else 0.0,
        action_is_ask=1.0 if tool.action_type == "ask" else 0.0,
        action_is_terminate=1.0 if tool.action_type == "terminate" else 0.0,
        argument_completeness=1.0,
        references_observed_entity=1.0 if references_observed else 0.0,
        references_unverified_entity=1.0 if references_unverified else 0.0,
        declared_observation_gain=1.0 if tool.reveals_fact is not None and not state.observed_facts[tool.reveals_fact] else 0.0,
        declared_requirement_gain=1.0 if tool.completes_requirement is not None and not state.completed_requirements[tool.completes_requirement] else 0.0,
        expected_tool_cost_frac=tool.cost / max(1, task.tool_budget),
        action_irreversible=1.0 if tool.irreversible else 0.0,
        precondition_satisfied=1.0 if precondition_ok else 0.0,
        policy_risk=tool.policy_risk,
        duplicates_previous_action=1.0 if duplicates else 0.0,
    )


def workflow_action_intent_features(task: WorkflowTask, state: WorkflowState, action: WorkflowAction) -> ActionFeatures:
    tool = task.tools[action.tool_index]
    precondition_ok = _action_precondition_ok(task, state, tool)
    duplicates = action.tool_index in state.used_actions
    references_observed = tool.precondition_fact is not None and state.observed_facts[tool.precondition_fact]
    references_unverified = _references_unverified(state, tool)
    premature_final = tool.action_type in ("respond", "terminate") and not all(state.completed_requirements)
    return ActionFeatures(
        action_is_read=1.0 if tool.action_type == "read" else 0.0,
        action_is_write=1.0 if tool.action_type == "write" else 0.0,
        action_is_respond=1.0 if tool.action_type == "respond" else 0.0,
        action_is_ask=1.0 if tool.action_type == "ask" else 0.0,
        action_is_terminate=1.0 if tool.action_type == "terminate" else 0.0,
        argument_completeness=0.0,
        references_observed_entity=1.0 if references_observed else 0.0,
        references_unverified_entity=1.0 if references_unverified else 0.0,
        declared_observation_gain=1.0 if tool.reveals_fact is not None and not state.observed_facts[tool.reveals_fact] else 0.0,
        declared_requirement_gain=1.0 if tool.completes_requirement is not None and not state.completed_requirements[tool.completes_requirement] else 0.0,
        expected_tool_cost_frac=tool.cost / max(1, task.tool_budget),
        action_irreversible=1.0 if tool.irreversible else 0.0,
        precondition_satisfied=1.0 if precondition_ok else 0.0,
        policy_risk=tool.policy_risk
        + (0.5 if tool.action_type == "write" and not precondition_ok else 0.0)
        + (0.8 if premature_final else 0.0)
        + (0.3 if tool.action_type == "ask" and not precondition_ok else 0.0),
        duplicates_previous_action=1.0 if duplicates else 0.0,
    )


def source_generator_manifest(
    seed: int,
    count: int,
    anonymize_tools: bool = True,
    families: Sequence[str] | None = None,
    difficulty: str = "v1",
) -> Dict[str, object]:
    family_list = list(families or WORKFLOW_FAMILIES)
    distribution = _source_generator_distribution(anonymize_tools, family_list, difficulty)
    return {
        "source": "PolicyConstrainedWorkflowGeneratorV1",
        "seed": seed,
        "task_count": count,
        "families": family_list,
        "difficulty": difficulty,
        "distribution_hash": hashlib.sha256(json.dumps(distribution, sort_keys=True).encode("utf-8")).hexdigest(),
        "anonymized_tool_names": anonymize_tools,
        "tau3_trajectories_used": False,
        "tau3_heldout_test_seen_before_freeze": False,
        "source_generator_distribution_fixed_before_target_test_evaluation": True,
        "parameter_distribution": distribution,
    }


def _source_generator_distribution(anonymize_tools: bool, families: Sequence[str] | None = None, difficulty: str = "v1") -> Dict[str, object]:
    family_list = list(families or WORKFLOW_FAMILIES)
    common = {
        "fact_count": "uniform_int[2,5]",
        "requirement_count": "uniform_int[2,5]",
        "turn_budget": "requirements + facts + uniform_int[1,4]",
        "families": {"schedule": "round_robin", "values": family_list},
        "tool_names": "anonymous source_tool_{task}_{index}" if anonymize_tools else "family_action_{index}",
    }
    if difficulty == "v1":
        return common
    if difficulty == "v2":
        return {
            **common,
            "fact_count": "uniform_int[3,4]",
            "requirement_count": "uniform_int[2,3]",
            "turn_budget": "requirements + facts + uniform_int[0,1]",
            "chained_read_probability": 0.55,
            "decoy_read_count": 1,
            "decoy_write_count": "max(1, requirement_count // 2)",
            "write_precondition_probability": 0.95,
            "purpose": "harder action-control benchmark with tighter budgets, distractor actions, and multi-step information dependencies",
        }
    raise ValueError(f"unknown workflow difficulty: {difficulty}")


def _make_tools(
    rng: random.Random,
    task_idx: int,
    family: str,
    fact_count: int,
    requirement_count: int,
    anonymize: bool,
    difficulty: str = "v1",
) -> List[WorkflowTool]:
    tools: List[WorkflowTool] = []
    for fact in range(fact_count):
        read_precondition = fact - 1 if difficulty == "v2" and fact > 0 and rng.random() < 0.55 else None
        tools.append(
            WorkflowTool(
                name=_tool_name(task_idx, len(tools), family, "read", anonymize),
                action_type="read",
                reveals_fact=fact,
                precondition_fact=read_precondition,
                policy_risk=0.0,
            )
        )
    if difficulty == "v2":
        for _ in range(1):
            tools.append(
                WorkflowTool(
                    name=_tool_name(task_idx, len(tools), family, "read", anonymize),
                    action_type="read",
                    reveals_fact=None,
                    precondition_fact=rng.randrange(fact_count) if rng.random() < 0.5 else None,
                    policy_risk=0.05,
                    decoy=True,
                )
            )
    for req in range(requirement_count):
        needs_fact_probability = 0.95 if difficulty == "v2" else 0.75
        needs_fact = rng.randrange(fact_count) if rng.random() < needs_fact_probability else None
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
    if difficulty == "v2":
        for _ in range(max(1, requirement_count // 2)):
            tools.append(
                WorkflowTool(
                    name=_tool_name(task_idx, len(tools), family, "write", anonymize),
                    action_type="write",
                    completes_requirement=None,
                    precondition_fact=rng.randrange(fact_count),
                    irreversible=rng.random() < 0.5,
                    policy_risk=0.05 + 0.25 * rng.random(),
                    decoy=True,
                )
            )
    tools.append(WorkflowTool(name=_tool_name(task_idx, len(tools), family, "ask", anonymize), action_type="ask"))
    tools.append(WorkflowTool(name=_tool_name(task_idx, len(tools), family, "respond", anonymize), action_type="respond"))
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


def _action_precondition_ok(task: WorkflowTask, state: WorkflowState, tool: WorkflowTool) -> bool:
    if tool.action_type in ("respond", "terminate"):
        return all(state.completed_requirements)
    if tool.action_type == "ask":
        return any(not fact for fact in state.observed_facts)
    return tool.precondition_fact is None or state.observed_facts[tool.precondition_fact]


def _references_unverified(state: WorkflowState, tool: WorkflowTool) -> bool:
    if tool.precondition_fact is not None:
        return not state.observed_facts[tool.precondition_fact]
    if tool.action_type == "ask":
        return any(not fact for fact in state.observed_facts)
    if tool.action_type in ("respond", "terminate"):
        return False
    return False
