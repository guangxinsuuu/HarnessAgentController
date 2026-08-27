from __future__ import annotations

from cgr_exp.schemas import SearchNode


class BestBoundPolicy:
    name = "best_bound"

    def score(self, node: SearchNode) -> float:
        if node.lower_bound is None:
            return float("-inf")
        return -float(node.lower_bound)

