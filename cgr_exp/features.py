from __future__ import annotations

import math
from typing import List

from .problems import Problem, State, lower_bound


FEATURE_NAMES = [
    "bias",
    "depth_frac",
    "assigned_frac",
    "load_max_norm",
    "load_mean_norm",
    "load_std_norm",
    "lb_norm",
    "remaining_frac",
    "branching_norm",
    "uncertain_frac",
    "invalid_frac",
    "is_llm",
    "last_job_frac",
    "last_machine_frac",
]


def state_features(problem: Problem, state: State, incumbent: int = 10**9) -> List[float]:
    total = max(1, sum(problem.durations))
    loads = list(state.loads)
    mean = sum(loads) / max(1, len(loads))
    var = sum((x - mean) ** 2 for x in loads) / max(1, len(loads))
    unassigned = state.unassigned_jobs()
    branching = sum(len(problem.allowed[j]) for j in unassigned)
    last_job = -1.0
    last_machine = -1.0
    if state.last_action is not None:
        last_job = state.last_action[0] / max(1, problem.job_count - 1)
        last_machine = state.last_action[1] / max(1, problem.machine_count - 1)
    return [
        1.0,
        state.depth / max(1, problem.job_count),
        (problem.job_count - len(unassigned)) / max(1, problem.job_count),
        max(loads or [0]) / total,
        mean / total,
        math.sqrt(var) / total,
        lower_bound(problem, state) / total,
        len(unassigned) / max(1, problem.job_count),
        branching / max(1, problem.job_count * problem.machine_count),
        state.uncertain_count / max(1, problem.job_count),
        state.invalid_count / max(1, problem.job_count),
        1.0 if state.proposer_kind == "llm" else 0.0,
        last_job,
        last_machine,
    ]

