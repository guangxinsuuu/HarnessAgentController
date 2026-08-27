from __future__ import annotations

from typing import Protocol


class RankerProtocol(Protocol):
    def score_nodes(self, nodes, search_context):
        ...

