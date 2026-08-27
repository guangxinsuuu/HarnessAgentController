from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, Optional, Tuple

from .problems import Action, Problem, State, lower_bound


@dataclass(frozen=True)
class Verification:
    schema_valid: bool
    verified: bool
    feasible: bool
    lower_bound: Optional[int]
    reason: str


class Verifier:
    """Conservative typed-action verifier.

    False negatives move true constraints to uncertain status and weaken pruning.
    False positives deliberately admit incorrect constraints for robustness tests.
    """

    def __init__(self, false_negative_rate: float = 0.0, false_positive_rate: float = 0.0, seed: int = 0):
        self.false_negative_rate = false_negative_rate
        self.false_positive_rate = false_positive_rate
        self.rng = random.Random(seed)

    def check(self, problem: Problem, state: State, action: Action) -> Verification:
        job, machine = action
        schema_valid = 0 <= job < problem.job_count and 0 <= machine < problem.machine_count
        if not schema_valid:
            return Verification(False, False, False, None, "schema-invalid")
        if state.assignments[job] is not None:
            return Verification(True, True, False, None, "already-assigned")

        truly_allowed = machine in problem.allowed[job]
        if truly_allowed and self.rng.random() < self.false_negative_rate:
            return Verification(True, False, True, None, "withheld-true-constraint")
        if (not truly_allowed) and self.rng.random() < self.false_positive_rate:
            child = state.apply(problem, action, verified=True)
            return Verification(True, True, True, lower_bound(problem, child), "false-positive-admitted")
        if not truly_allowed:
            return Verification(True, True, False, None, "disallowed")

        child = state.apply(problem, action, verified=True)
        return Verification(True, True, True, lower_bound(problem, child), "verified")

