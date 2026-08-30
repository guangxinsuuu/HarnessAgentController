from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence, Tuple

from cgr_exp.harness_contract import CanonicalCandidate, TrainingOnlyLabels
from cgr_exp.workflow_source_generator import (
    WorkflowAction,
    WorkflowState,
    WorkflowTask,
    workflow_action_intent_features,
    workflow_action_features,
    workflow_runtime_features,
)


@dataclass(frozen=True)
class WorkflowTraceRecord:
    task_id: str
    family: str
    state_key: str
    action_index: int
    features: Tuple[float, ...]
    labels: TrainingOnlyLabels


Pair = Tuple[Tuple[float, ...], Tuple[float, ...]]


def collect_workflow_oracle_records(task: WorkflowTask, max_states: int = 200, action_intent_features: bool = False) -> List[WorkflowTraceRecord]:
    root = task.root_state()
    frontier = [root]
    seen = set()
    records: List[WorkflowTraceRecord] = []
    cost_memo: Dict[Tuple[object, ...], float] = {}
    reward_memo: Dict[Tuple[object, ...], float] = {}
    while frontier and len(seen) < max_states:
        state = frontier.pop(0)
        if state.key() in seen:
            continue
        seen.add(state.key())
        state_features = workflow_runtime_features(task, state)
        actions = task.candidate_actions(state)
        ranked = rank_actions_by_oracle(task, state, actions, cost_memo=cost_memo, reward_memo=reward_memo)
        rank_by_action = {action.tool_index: rank for rank, action in enumerate(ranked)}
        for action in actions:
            child = task.apply(state, action)
            labels = _labels_for_child(task, child, rank_by_action[action.tool_index], cost_memo=cost_memo, reward_memo=reward_memo)
            candidate = CanonicalCandidate(
                candidate_id=f"{task.task_id}:{len(records)}",
                state=state_features,
                action=workflow_action_intent_features(task, state, action) if action_intent_features else workflow_action_features(task, state, action),
            )
            records.append(
                WorkflowTraceRecord(
                    task_id=task.task_id,
                    family=task.family,
                    state_key=str(hash(state.key())),
                    action_index=action.tool_index,
                    features=tuple(candidate.feature_vector()),
                    labels=labels,
                )
            )
            if child.remaining_turns > 0 and not task.success(child):
                frontier.append(child)
    return records


def make_workflow_ranking_pairs(records: Sequence[WorkflowTraceRecord], max_pairs_per_state: int = 24) -> List[Pair]:
    by_state: Dict[Tuple[str, str], List[WorkflowTraceRecord]] = {}
    for record in records:
        by_state.setdefault((record.task_id, record.state_key), []).append(record)
    pairs: List[Pair] = []
    for rows in by_state.values():
        local: List[Pair] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left = rows[i]
                right = rows[j]
                if _label_key(left) < _label_key(right):
                    local.append((left.features, right.features))
                elif _label_key(right) < _label_key(left):
                    local.append((right.features, left.features))
        pairs.extend(local[:max_pairs_per_state])
    return pairs


def rank_actions_by_oracle(
    task: WorkflowTask,
    state: WorkflowState,
    actions: Sequence[WorkflowAction],
    cost_memo: Dict[Tuple[object, ...], float] | None = None,
    reward_memo: Dict[Tuple[object, ...], float] | None = None,
) -> List[WorkflowAction]:
    if cost_memo is None:
        cost_memo = {}
    if reward_memo is None:
        reward_memo = {}
    return sorted(actions, key=lambda action: _oracle_action_key(task, state, action, cost_memo, reward_memo))


def best_descendant_reward(task: WorkflowTask, state: WorkflowState, memo: Dict[Tuple[object, ...], float] | None = None) -> float:
    if memo is None:
        memo = {}
    key = state.key()
    if key in memo:
        return memo[key]
    best = task.reward(state)
    for action in task.candidate_actions(state):
        best = max(best, best_descendant_reward(task, task.apply(state, action), memo))
    memo[key] = best
    return best


def cost_to_success(task: WorkflowTask, state: WorkflowState, memo: Dict[Tuple[object, ...], float] | None = None) -> float:
    if memo is None:
        memo = {}
    key = state.key()
    if key in memo:
        return memo[key]
    if task.success(state):
        memo[key] = 0.0
        return 0.0
    if state.remaining_turns <= 0 or state.remaining_tool_budget <= 0:
        memo[key] = math.inf
        return math.inf
    best = math.inf
    for action in task.candidate_actions(state):
        child_cost = cost_to_success(task, task.apply(state, action), memo)
        if math.isfinite(child_cost):
            best = min(best, 1.0 + child_cost)
    memo[key] = best
    return best


def _labels_for_child(
    task: WorkflowTask,
    child: WorkflowState,
    oracle_rank: int,
    cost_memo: Dict[Tuple[object, ...], float] | None = None,
    reward_memo: Dict[Tuple[object, ...], float] | None = None,
) -> TrainingOnlyLabels:
    cost = cost_to_success(task, child, cost_memo)
    best_reward = best_descendant_reward(task, child, reward_memo)
    success = math.isfinite(cost) or task.success(child)
    return TrainingOnlyLabels(
        success=success,
        final_reward=best_reward,
        future_violation_risk=1.0 if child.violations > 0 else 0.0,
        best_descendant_value=best_reward,
        cost_to_success=cost,
        oracle_rank=oracle_rank,
        eventual_outcome="success" if success else "dead_end",
    )


def _oracle_action_key(
    task: WorkflowTask,
    state: WorkflowState,
    action: WorkflowAction,
    cost_memo: Dict[Tuple[object, ...], float],
    reward_memo: Dict[Tuple[object, ...], float],
) -> Tuple[float, float, float, int]:
    child = task.apply(state, action)
    cost = cost_to_success(task, child, cost_memo)
    reward = best_descendant_reward(task, child, reward_memo)
    return (
        0.0 if math.isfinite(cost) or task.success(child) else 1.0,
        cost if math.isfinite(cost) else 1e9,
        -reward,
        action.tool_index,
    )


def _label_key(record: WorkflowTraceRecord) -> Tuple[float, float, float, int]:
    labels = record.labels
    return (
        0.0 if labels.success else 1.0,
        labels.cost_to_success if math.isfinite(labels.cost_to_success) else 1e9,
        -labels.best_descendant_value,
        labels.oracle_rank,
    )
