from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from cgr_exp.schemas import SearchNode


@dataclass
class Frontier:
    nodes: List[SearchNode] = field(default_factory=list)

    def add(self, node: SearchNode) -> None:
        self.nodes.append(node)

    def extend(self, nodes: Iterable[SearchNode]) -> None:
        self.nodes.extend(nodes)

    def pop_best(self, policy) -> SearchNode:
        if not self.nodes:
            raise IndexError("cannot pop from an empty frontier")
        selected_index = max(range(len(self.nodes)), key=lambda i: (policy.score(self.nodes[i]), self.nodes[i].node_id))
        return self.nodes.pop(selected_index)

    def __len__(self) -> int:
        return len(self.nodes)
