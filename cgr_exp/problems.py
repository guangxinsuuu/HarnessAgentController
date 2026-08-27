from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Action = Tuple[int, int]


@dataclass(frozen=True)
class Problem:
    pid: str
    durations: Tuple[int, ...]
    machine_count: int
    allowed: Tuple[Tuple[int, ...], ...]
    family: str

    @property
    def job_count(self) -> int:
        return len(self.durations)

    def natural_language(self) -> str:
        parts = [
            f"There are {self.job_count} jobs and {self.machine_count} machines.",
            "Assign every job to one allowed machine and minimize makespan.",
        ]
        for j, dur in enumerate(self.durations):
            allowed = ", ".join(f"M{m}" for m in self.allowed[j])
            parts.append(f"Job J{j} takes {dur} time units and may run on {allowed}.")
        return " ".join(parts)


@dataclass(frozen=True)
class State:
    assignments: Tuple[Optional[int], ...]
    loads: Tuple[int, ...]
    depth: int = 0
    last_action: Optional[Action] = None
    uncertain_count: int = 0
    invalid_count: int = 0
    proposer_kind: str = "solver"

    @classmethod
    def root(cls, problem: Problem, proposer_kind: str = "solver") -> "State":
        return cls(
            assignments=tuple(None for _ in range(problem.job_count)),
            loads=tuple(0 for _ in range(problem.machine_count)),
            proposer_kind=proposer_kind,
        )

    def is_complete(self) -> bool:
        return all(a is not None for a in self.assignments)

    def unassigned_jobs(self) -> List[int]:
        return [i for i, m in enumerate(self.assignments) if m is None]

    def apply(self, problem: Problem, action: Action, verified: bool) -> "State":
        job, machine = action
        assignments = list(self.assignments)
        loads = list(self.loads)
        invalid_count = self.invalid_count
        uncertain_count = self.uncertain_count

        if job < 0 or job >= problem.job_count or machine < 0 or machine >= problem.machine_count:
            invalid_count += 1
            return State(tuple(assignments), tuple(loads), self.depth, action, uncertain_count, invalid_count, self.proposer_kind)
        if assignments[job] is not None:
            invalid_count += 1
            return State(tuple(assignments), tuple(loads), self.depth, action, uncertain_count, invalid_count, self.proposer_kind)

        if not verified:
            uncertain_count += 1
        assignments[job] = machine
        loads[machine] += problem.durations[job]
        return State(tuple(assignments), tuple(loads), self.depth + 1, action, uncertain_count, invalid_count, self.proposer_kind)

    def objective(self) -> int:
        return max(self.loads) if self.loads else 0

    def key(self) -> Tuple[Tuple[Optional[int], ...], Tuple[int, ...]]:
        return self.assignments, self.loads


def generate_problem(rng: random.Random, pid: str, family: str, job_count: int, machine_count: int) -> Problem:
    if family == "balanced":
        durations = tuple(rng.randint(1, 9) for _ in range(job_count))
        min_allowed = max(1, machine_count - 1)
    elif family == "sparse":
        durations = tuple(rng.randint(2, 12) for _ in range(job_count))
        min_allowed = 1
    elif family == "heavy_tail":
        durations = tuple(1 + int(rng.expovariate(0.25)) % 20 for _ in range(job_count))
        min_allowed = max(1, machine_count // 2)
    else:
        raise ValueError(f"unknown family: {family}")

    allowed: List[Tuple[int, ...]] = []
    for _ in range(job_count):
        k = rng.randint(min_allowed, machine_count)
        machines = tuple(sorted(rng.sample(range(machine_count), k)))
        allowed.append(machines)
    return Problem(pid=pid, durations=durations, machine_count=machine_count, allowed=tuple(allowed), family=family)


def generate_suite(
    seed: int,
    count: int,
    families: Sequence[str],
    job_range: Tuple[int, int],
    machine_range: Tuple[int, int],
    prefix: str,
) -> List[Problem]:
    rng = random.Random(seed)
    problems: List[Problem] = []
    for i in range(count):
        family = families[i % len(families)]
        n = rng.randint(*job_range)
        m = rng.randint(*machine_range)
        problems.append(generate_problem(rng, f"{prefix}-{i:04d}", family, n, m))
    return problems


def all_valid_actions(problem: Problem, state: State) -> List[Action]:
    actions: List[Action] = []
    for job in state.unassigned_jobs():
        for machine in problem.allowed[job]:
            actions.append((job, machine))
    return actions


def choose_branch_job(problem: Problem, state: State) -> Optional[int]:
    jobs = state.unassigned_jobs()
    if not jobs:
        return None
    return min(jobs, key=lambda j: (len(problem.allowed[j]), -problem.durations[j], j))


def solver_branch_actions(problem: Problem, state: State) -> List[Action]:
    job = choose_branch_job(problem, state)
    if job is None:
        return []
    machines = sorted(problem.allowed[job], key=lambda m: (state.loads[m], m))
    return [(job, m) for m in machines]


def lower_bound(problem: Problem, state: State) -> int:
    remaining = sum(problem.durations[j] for j in state.unassigned_jobs())
    avg_bound = (sum(state.loads) + remaining + problem.machine_count - 1) // problem.machine_count
    largest_remaining = max([problem.durations[j] for j in state.unassigned_jobs()] or [0])
    return max(max(state.loads or (0,)), avg_bound, largest_remaining)


def exact_optimum(problem: Problem) -> int:
    best = 10**9
    root = State.root(problem)

    def rec(state: State) -> None:
        nonlocal best
        lb = lower_bound(problem, state)
        if lb >= best:
            return
        if state.is_complete():
            best = min(best, state.objective())
            return
        for action in solver_branch_actions(problem, state):
            rec(state.apply(problem, action, verified=True))

    rec(root)
    return best


def best_completion_objective(problem: Problem, state: State, memo: Optional[Dict[Tuple[Tuple[Optional[int], ...], Tuple[int, ...]], int]] = None) -> int:
    if memo is None:
        memo = {}
    key = state.key()
    if key in memo:
        return memo[key]
    if state.is_complete():
        memo[key] = state.objective()
        return memo[key]
    best = 10**9
    for action in solver_branch_actions(problem, state):
        child = state.apply(problem, action, verified=True)
        best = min(best, best_completion_objective(problem, child, memo))
    memo[key] = best
    return best

