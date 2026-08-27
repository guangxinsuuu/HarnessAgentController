from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Optional, Tuple

from cgr_exp.schemas import CompletionStatus, CostLedger, FeasibleStatus, SearchNode, ValidationResult


@dataclass(frozen=True)
class ScheduleAction:
    job: str
    machine: str
    action_type: str = "schedule"

    def validate_schema(self) -> ValidationResult:
        errors: List[str] = []
        if not isinstance(self.job, str) or not self.job:
            errors.append("job must be a non-empty string")
        if not isinstance(self.machine, str) or not self.machine:
            errors.append("machine must be a non-empty string")
        return ValidationResult(not errors, errors)

    def to_dict(self) -> Dict[str, str]:
        return {"action_type": self.action_type, "job": self.job, "machine": self.machine}


@dataclass(frozen=True)
class SchedulingTask:
    task_id: str
    durations: Dict[str, int]
    machines: Tuple[str, ...]
    allowed: Dict[str, Tuple[str, ...]]
    release_times: Dict[str, int]
    due_dates: Dict[str, int]
    predecessors: Dict[str, Tuple[str, ...]]

    def root_node(self) -> SearchNode:
        return self._make_node(scheduled=(), parent_id=None, action=None, node_id=f"{self.task_id}:root", incumbent=None)

    def ready_jobs(self, scheduled: Tuple[Dict[str, object], ...] | List[Dict[str, object]]) -> List[str]:
        done = {str(row["job"]) for row in scheduled}
        return [
            job
            for job in self.durations
            if job not in done and all(pred in done for pred in self.predecessors.get(job, ()))
        ]

    def lower_bound(self, scheduled: Tuple[Dict[str, object], ...] | List[Dict[str, object]]) -> float:
        loads = self._machine_loads(scheduled)
        current_makespan = max(loads.values() or [0])
        remaining = sum(duration for job, duration in self.durations.items() if job not in self._scheduled_jobs(scheduled))
        average_work = (sum(loads.values()) + remaining) / max(1, len(self.machines))
        longest_released = max(
            (self.release_times[job] + self.durations[job] for job in self.durations if job not in self._scheduled_jobs(scheduled)),
            default=0,
        )
        return float(max(current_makespan, average_work, longest_released))

    def exhaustive_oracle(self) -> float:
        return self.best_descendant_objective_from_scheduled(())

    def best_descendant_objective(self, node: SearchNode) -> float:
        return self.best_descendant_objective_from_scheduled(tuple(node.structured_state["scheduled"]))

    def best_descendant_objective_from_scheduled(self, scheduled: Tuple[Dict[str, object], ...] | List[Dict[str, object]]) -> float:
        best = float("inf")
        memo: Dict[Tuple[Tuple[str, str, int, int], ...], float] = {}

        def key(cur: Tuple[Dict[str, object], ...]) -> Tuple[Tuple[str, str, int, int], ...]:
            return tuple((str(row["job"]), str(row["machine"]), int(row["start"]), int(row["finish"])) for row in cur)

        def rec(cur: Tuple[Dict[str, object], ...]) -> float:
            nonlocal best
            cur_key = key(cur)
            if cur_key in memo:
                return memo[cur_key]
            if self.lower_bound(cur) >= best:
                memo[cur_key] = float("inf")
                return memo[cur_key]
            if len(cur) == len(self.durations):
                value = self.objective(cur)
                best = min(best, value)
                memo[cur_key] = value
                return value
            local_best = float("inf")
            for job in self.ready_jobs(cur):
                for machine in self.allowed[job]:
                    local_best = min(local_best, rec(cur + (self._scheduled_row(cur, job, machine),)))
            memo[cur_key] = local_best
            return local_best

        rec(tuple(scheduled))
        return best

    def objective(self, scheduled: Tuple[Dict[str, object], ...] | List[Dict[str, object]]) -> float:
        makespan = max((int(row["finish"]) for row in scheduled), default=0)
        tardiness = sum(max(0, int(row["finish"]) - self.due_dates[str(row["job"])]) for row in scheduled)
        return float(makespan + tardiness)

    def candidate_actions(self, node: SearchNode, k: int) -> List[ScheduleAction]:
        scheduled = tuple(node.structured_state["scheduled"])
        ready = sorted(self.ready_jobs(scheduled), key=lambda job: (self.due_dates[job], -self.durations[job], job))
        actions: List[ScheduleAction] = []
        loads = self._machine_loads(scheduled)
        for job in ready:
            machines = sorted(self.allowed[job], key=lambda machine: (loads[machine], machine))
            actions.extend(ScheduleAction(job=job, machine=machine) for machine in machines)
        return actions[:k]

    def apply(self, node: SearchNode, action: ScheduleAction, child_index: int = 0) -> SearchNode:
        scheduled = tuple(node.structured_state["scheduled"])
        if not self.is_valid_action(scheduled, action):
            return node.with_updates(feasible_status=FeasibleStatus.INFEASIBLE)
        row = self._scheduled_row(scheduled, action.job, action.machine)
        return self._make_node(
            scheduled=scheduled + (row,),
            parent_id=node.node_id,
            action=action,
            node_id=f"{node.node_id}/{child_index}:{action.job}->{action.machine}",
            incumbent=node.incumbent_at_creation,
        )

    def is_valid_action(self, scheduled: Tuple[Dict[str, object], ...] | List[Dict[str, object]], action: ScheduleAction) -> bool:
        if action.job not in self.durations or action.machine not in self.machines:
            return False
        if action.machine not in self.allowed[action.job]:
            return False
        return action.job in self.ready_jobs(scheduled)

    def _scheduled_row(self, scheduled: Tuple[Dict[str, object], ...] | List[Dict[str, object]], job: str, machine: str) -> Dict[str, object]:
        loads = self._machine_loads(scheduled)
        predecessor_finish = max(
            (int(row["finish"]) for row in scheduled if str(row["job"]) in self.predecessors.get(job, ())),
            default=0,
        )
        start = max(loads[machine], self.release_times[job], predecessor_finish)
        finish = start + self.durations[job]
        return {"job": job, "machine": machine, "start": start, "finish": finish}

    def _make_node(
        self,
        scheduled: Tuple[Dict[str, object], ...],
        parent_id: Optional[str],
        action: Optional[ScheduleAction],
        node_id: str,
        incumbent: Optional[float],
        uncertain_constraints: Optional[List[str]] = None,
    ) -> SearchNode:
        complete = len(scheduled) == len(self.durations)
        lower = self.lower_bound(scheduled)
        return SearchNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=len(scheduled),
            task_id=self.task_id,
            semantic_state={"family": "scheduling"},
            structured_state={"scheduled": list(scheduled), "machine_loads": self._machine_loads(scheduled)},
            verified_constraints=[f"schedule:{row['job']}:{row['machine']}" for row in scheduled],
            uncertain_constraints=uncertain_constraints or [],
            lower_bound=lower,
            incumbent_at_creation=incumbent,
            objective_if_complete=self.objective(scheduled) if complete else None,
            feasible_status=FeasibleStatus.FEASIBLE,
            completion_status=CompletionStatus.COMPLETE if complete else CompletionStatus.PARTIAL,
            proposal_action=action,
            observation=None,
            universal_features={
                "depth": float(len(scheduled)),
                "branch_factor": float(sum(len(self.allowed[job]) for job in self.ready_jobs(scheduled))),
            },
            verifier_features={
                "lower_bound": lower,
                "verified_constraint_count": float(len(scheduled)),
                "uncertain_constraint_count": float(len(uncertain_constraints or [])),
            },
            semantic_features=None,
            missingness_mask={"lower_bound": False, "semantic_features": True},
            accumulated_expensive_cost=CostLedger(),
            metadata={"family": "scheduling", "schema_version": "level1-scheduling-v1"},
        )

    def _scheduled_jobs(self, scheduled: Tuple[Dict[str, object], ...] | List[Dict[str, object]]) -> set[str]:
        return {str(row["job"]) for row in scheduled}

    def _machine_loads(self, scheduled: Tuple[Dict[str, object], ...] | List[Dict[str, object]]) -> Dict[str, int]:
        loads = {machine: 0 for machine in self.machines}
        for row in scheduled:
            machine = str(row["machine"])
            loads[machine] = max(loads[machine], int(row["finish"]))
        return loads


def generate_scheduling_fixture(rng: random.Random, task_id: str, job_count: int, machine_count: int) -> SchedulingTask:
    machines = tuple(f"M{i}" for i in range(machine_count))
    durations = {f"J{i}": rng.randint(1, 7) for i in range(job_count)}
    release_times = {job: rng.randint(0, 3) for job in durations}
    total_duration = sum(durations.values())
    due_dates = {job: rng.randint(max(2, total_duration // max(1, machine_count)), total_duration + 6) for job in durations}
    allowed: Dict[str, Tuple[str, ...]] = {}
    for job in durations:
        count = rng.randint(1, machine_count)
        allowed[job] = tuple(sorted(rng.sample(machines, count)))
    predecessors: Dict[str, Tuple[str, ...]] = {job: () for job in durations}
    jobs = tuple(durations)
    for i, job in enumerate(jobs[1:], start=1):
        if rng.random() < 0.35:
            predecessors[job] = (jobs[rng.randrange(0, i)],)
    return SchedulingTask(
        task_id=task_id,
        durations=durations,
        machines=machines,
        allowed=allowed,
        release_times=release_times,
        due_dates=due_dates,
        predecessors=predecessors,
    )
