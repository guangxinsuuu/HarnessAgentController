from __future__ import annotations

from cgr_exp.certification.prune import PruneDecision, should_prune
from cgr_exp.schemas import SearchNode


def conservative_should_prune(node: SearchNode, incumbent: float | None) -> PruneDecision:
    return should_prune(node, incumbent)
