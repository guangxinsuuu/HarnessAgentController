from __future__ import annotations

from cgr_exp.schemas import SearchNode


class DFSPolicy:
    name = "dfs"

    def score(self, node: SearchNode) -> float:
        return float(node.depth)

