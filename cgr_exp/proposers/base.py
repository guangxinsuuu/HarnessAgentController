from __future__ import annotations

from typing import Protocol


class TypedProposerProtocol(Protocol):
    def propose(self, task, node, k, budget_state):
        ...

