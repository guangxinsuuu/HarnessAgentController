from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from cgr_exp.schemas import FeasibleStatus, SearchNode


@dataclass(frozen=True)
class PruneDecision:
    prune: bool
    reason: str


def should_prune(
    node: SearchNode,
    incumbent: Optional[float],
    learned_score: Optional[float] = None,
    memory_score: Optional[float] = None,
) -> PruneDecision:
    del learned_score, memory_score
    if node.feasible_status == FeasibleStatus.INFEASIBLE:
        return PruneDecision(True, "verified-infeasible")
    if incumbent is None:
        return PruneDecision(False, "no-incumbent")
    if node.lower_bound is not None and node.lower_bound >= incumbent:
        return PruneDecision(True, "lower-bound")
    return PruneDecision(False, "no-certificate")

