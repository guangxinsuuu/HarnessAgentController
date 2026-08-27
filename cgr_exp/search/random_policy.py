from __future__ import annotations

import random

from cgr_exp.schemas import SearchNode


class RandomPolicy:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def score(self, node: SearchNode) -> float:
        del node
        return self.rng.random()

