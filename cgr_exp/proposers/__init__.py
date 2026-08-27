from __future__ import annotations

import random
from typing import List, Tuple

from cgr_exp.problems import Action, Problem, State, all_valid_actions, solver_branch_actions


class TypedProposer:
    def propose(self, problem: Problem, state: State, budget_k: int) -> List[Action]:
        raise NotImplementedError


class SolverProposer(TypedProposer):
    def propose(self, problem: Problem, state: State, budget_k: int) -> List[Action]:
        return solver_branch_actions(problem, state)[:budget_k]


class SimulatedLLMProposer(TypedProposer):
    """Typed LLM-proposer simulator for dependency-free experiments."""

    def __init__(
        self,
        seed: int = 0,
        recall: float = 0.85,
        invalid_rate: float = 0.08,
        duplicate_rate: float = 0.05,
        distractor_rate: float = 0.10,
    ):
        self.rng = random.Random(seed)
        self.recall = recall
        self.invalid_rate = invalid_rate
        self.duplicate_rate = duplicate_rate
        self.distractor_rate = distractor_rate

    def propose(self, problem: Problem, state: State, budget_k: int) -> List[Action]:
        valid = all_valid_actions(problem, state)
        self.rng.shuffle(valid)
        actions: List[Action] = []

        for action in valid:
            if len(actions) >= budget_k:
                break
            if self.rng.random() <= self.recall:
                actions.append(action)

        while len(actions) < budget_k and self.rng.random() < self.distractor_rate:
            job = self.rng.randrange(problem.job_count)
            machine = self.rng.randrange(problem.machine_count)
            actions.append((job, machine))

        if self.rng.random() < self.invalid_rate:
            actions.append((problem.job_count + self.rng.randrange(3), self.rng.randrange(problem.machine_count + 2)))
        if actions and self.rng.random() < self.duplicate_rate:
            actions.append(self.rng.choice(actions))

        return actions[:budget_k]


def proposal_quality(problem: Problem, state: State, actions: List[Action]) -> Tuple[float, float, float]:
    valid = set(all_valid_actions(problem, state))
    if not actions:
        return 0.0, 0.0, 0.0
    schema_valid = sum(1 for j, m in actions if 0 <= j < problem.job_count and 0 <= m < problem.machine_count) / len(actions)
    recall = 1.0 if any(a in valid for a in actions) else 0.0
    duplicate = 1.0 - len(set(actions)) / len(actions)
    return recall, schema_valid, duplicate
