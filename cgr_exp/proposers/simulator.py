from __future__ import annotations

import random
from typing import List

from cgr_exp.proposers import SimulatedLLMProposer
from cgr_exp.schemas import AssignAction


class DeterministicSimulatorProposer:
    def __init__(
        self,
        seed: int = 0,
        useful_action_recall: float = 0.8,
        invalid_action_prob: float = 0.05,
        duplicate_prob: float = 0.05,
        semantic_noise_prob: float = 0.05,
        paraphrase_sensitivity: float = 0.0,
    ):
        self.rng = random.Random(seed)
        self.useful_action_recall = useful_action_recall
        self.invalid_action_prob = invalid_action_prob
        self.duplicate_prob = duplicate_prob
        self.semantic_noise_prob = semantic_noise_prob
        self.paraphrase_sensitivity = paraphrase_sensitivity

    def propose(self, task, node, k: int, budget_state=None) -> List[AssignAction]:
        del budget_state
        actions: List[AssignAction] = []
        assigned = set(node.structured_state["assignments"])
        remaining = [t for t in task.durations if t not in assigned]
        if remaining:
            focus = min(remaining, key=lambda t: (len(task.allowed[t]), -task.durations[t], t))
            resources = list(task.allowed[focus])
            self.rng.shuffle(resources)
            for resource in resources:
                if len(actions) >= k:
                    break
                if self.rng.random() <= self.useful_action_recall:
                    actions.append(AssignAction(task=focus, resource=resource))

        while len(actions) < k and remaining and self.rng.random() < self.semantic_noise_prob:
            actions.append(AssignAction(task=self.rng.choice(remaining), resource=self.rng.choice(task.resources)))

        if self.rng.random() < self.invalid_action_prob:
            actions.append(AssignAction(task="", resource="UNKNOWN_RESOURCE"))
        if actions and self.rng.random() < self.duplicate_prob:
            actions.append(self.rng.choice(actions))
        return actions[:k]


__all__ = ["SimulatedLLMProposer", "DeterministicSimulatorProposer"]
