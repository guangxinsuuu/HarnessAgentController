from __future__ import annotations

from typing import Any, Protocol

from cgr_exp.schemas import SearchNode, TypedAction, VerificationResult


class VerifierProtocol(Protocol):
    def verify(self, task: Any, node: SearchNode, action: TypedAction) -> VerificationResult:
        ...

    def make_child(
        self,
        task: Any,
        parent: SearchNode,
        action: TypedAction,
        result: VerificationResult,
        child_index: int,
    ) -> SearchNode:
        ...
