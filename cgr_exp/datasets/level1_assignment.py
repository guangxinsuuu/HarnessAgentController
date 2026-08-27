from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

from cgr_exp.schemas import AssignAction, CompletionStatus, CostLedger, FeasibleStatus, SearchNode


@dataclass(frozen=True)
class AssignmentTask:
    task_id: str
    durations: Dict[str, int]
    resources: Tuple[str, ...]
    allowed: Dict[str, Tuple[str, ...]]

    def root_node(self) -> SearchNode:
        return make_node(self, assignments={}, parent_id=None, action=None, node_id=f"{self.task_id}:root", incumbent=None)

    def exhaustive_oracle(self) -> float:
        best = float("inf")
        tasks = tuple(self.durations)

        def rec(i: int, loads: Dict[str, int]) -> None:
            nonlocal best
            if max(loads.values() or [0]) >= best:
                return
            if i == len(tasks):
                best = min(best, float(max(loads.values() or [0])))
                return
            task = tasks[i]
            for resource in self.allowed[task]:
                nxt = dict(loads)
                nxt[resource] += self.durations[task]
                rec(i + 1, nxt)

        rec(0, {r: 0 for r in self.resources})
        return best

    def best_descendant_objective(self, node: SearchNode) -> float:
        assignments = dict(node.structured_state["assignments"])
        return self.best_descendant_objective_from_assignments(assignments)

    def best_descendant_objective_from_assignments(self, assignments: Dict[str, str]) -> float:
        loads = {r: 0 for r in self.resources}
        for assigned_task, resource in assignments.items():
            loads[resource] += self.durations[assigned_task]
        tasks = [t for t in self.durations if t not in assignments]
        best = float("inf")

        def rec(i: int, cur_loads: Dict[str, int]) -> None:
            nonlocal best
            if max(cur_loads.values() or [0]) >= best:
                return
            if i == len(tasks):
                best = min(best, float(max(cur_loads.values() or [0])))
                return
            task = tasks[i]
            for resource in self.allowed[task]:
                nxt = dict(cur_loads)
                nxt[resource] += self.durations[task]
                rec(i + 1, nxt)

        rec(0, loads)
        return best

    def lower_bound(self, assignments: Dict[str, str]) -> float:
        loads = {r: 0 for r in self.resources}
        for task, resource in assignments.items():
            loads[resource] += self.durations[task]
        remaining = sum(v for k, v in self.durations.items() if k not in assignments)
        avg = (sum(loads.values()) + remaining) / max(1, len(loads))
        largest = max([v for k, v in self.durations.items() if k not in assignments] or [0])
        return float(max(max(loads.values() or [0]), avg, largest))

    def candidate_actions(self, node: SearchNode, k: int) -> List[AssignAction]:
        assigned = set(node.structured_state["assignments"])
        remaining = [task for task in self.durations if task not in assigned]
        if not remaining:
            return []
        task = min(remaining, key=lambda t: (len(self.allowed[t]), -self.durations[t], t))
        loads = node.structured_state["loads"]
        resources = sorted(self.allowed[task], key=lambda r: (loads[r], r))
        return [AssignAction(task=task, resource=r) for r in resources[:k]]

    def apply(self, node: SearchNode, action: AssignAction, child_index: int = 0) -> SearchNode:
        assignments = dict(node.structured_state["assignments"])
        if action.task not in self.durations or action.resource not in self.resources or action.task in assignments:
            return node.with_updates(feasible_status=FeasibleStatus.INFEASIBLE)
        assignments[action.task] = action.resource
        return make_node(
            self,
            assignments=assignments,
            parent_id=node.node_id,
            action=action,
            node_id=f"{node.node_id}/{child_index}:{action.task}->{action.resource}",
            incumbent=node.incumbent_at_creation,
        )


def generate_assignment_fixture(rng: random.Random, task_id: str, job_count: int, resource_count: int) -> AssignmentTask:
    resources = tuple(f"R{i}" for i in range(resource_count))
    durations = {f"T{i}": rng.randint(1, 9) for i in range(job_count)}
    allowed: Dict[str, Tuple[str, ...]] = {}
    for task in durations:
        count = rng.randint(1, resource_count)
        allowed[task] = tuple(sorted(rng.sample(resources, count)))
    return AssignmentTask(task_id=task_id, durations=durations, resources=resources, allowed=allowed)


def make_node(
    task: AssignmentTask,
    assignments: Dict[str, str],
    parent_id: Optional[str],
    action: Optional[AssignAction],
    node_id: str,
    incumbent: Optional[float],
    uncertain_constraints: Optional[List[str]] = None,
) -> SearchNode:
    loads = {r: 0 for r in task.resources}
    for t, r in assignments.items():
        loads[r] += task.durations[t]
    complete = len(assignments) == len(task.durations)
    lower = task.lower_bound(assignments)
    return SearchNode(
        node_id=node_id,
        parent_id=parent_id,
        depth=len(assignments),
        task_id=task.task_id,
        semantic_state={"rendering": task.task_id},
        structured_state={"assignments": dict(assignments), "loads": loads},
        verified_constraints=[f"assign:{t}:{r}" for t, r in sorted(assignments.items())],
        uncertain_constraints=uncertain_constraints or [],
        lower_bound=lower,
        incumbent_at_creation=incumbent,
        objective_if_complete=float(max(loads.values())) if complete else None,
        feasible_status=FeasibleStatus.FEASIBLE,
        completion_status=CompletionStatus.COMPLETE if complete else CompletionStatus.PARTIAL,
        proposal_action=action,
        observation=None,
        universal_features={
            "depth": float(len(assignments)),
            "branch_factor": 0.0 if complete else float(sum(len(task.allowed[t]) for t in task.durations if t not in assignments)),
        },
        verifier_features={
            "lower_bound": lower,
            "verified_constraint_count": float(len(assignments)),
            "uncertain_constraint_count": float(len(uncertain_constraints or [])),
        },
        semantic_features=None,
        missingness_mask={"lower_bound": False, "semantic_features": True},
        accumulated_expensive_cost=CostLedger(),
        metadata={"schema_version": "phase-a"},
    )
