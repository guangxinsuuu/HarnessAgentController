from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, FrozenSet, List, Optional, Tuple

from cgr_exp.schemas import CompletionStatus, CostLedger, FeasibleStatus, SearchNode, ValidationResult


@dataclass(frozen=True)
class ChooseSetAction:
    set_id: str
    action_type: str = "choose_set"

    def validate_schema(self) -> ValidationResult:
        return ValidationResult(isinstance(self.set_id, str) and bool(self.set_id), [] if self.set_id else ["set_id required"])

    def to_dict(self):
        return {"action_type": self.action_type, "set_id": self.set_id}


@dataclass(frozen=True)
class SetCoverTask:
    task_id: str
    universe: Tuple[str, ...]
    sets: Dict[str, FrozenSet[str]]
    costs: Dict[str, float]

    def root_node(self) -> SearchNode:
        return self._make_node(chosen=(), parent_id=None, action=None, node_id=f"{self.task_id}:root")

    def covered(self, chosen: Tuple[str, ...]) -> FrozenSet[str]:
        out = set()
        for set_id in chosen:
            out.update(self.sets[set_id])
        return frozenset(out)

    def lower_bound(self, chosen: Tuple[str, ...]) -> float:
        covered = self.covered(chosen)
        uncovered = [u for u in self.universe if u not in covered]
        if not uncovered:
            return sum(self.costs[s] for s in chosen)
        max_cover = max((len(self.sets[s].intersection(uncovered)) for s in self.sets if s not in chosen), default=1)
        min_cost = min((self.costs[s] for s in self.sets if s not in chosen), default=0.0)
        return sum(self.costs[s] for s in chosen) + math.ceil(len(uncovered) / max(1, max_cover)) * min_cost

    def exhaustive_oracle(self) -> float:
        set_ids = tuple(self.sets)
        best = float("inf")

        def rec(i: int, chosen: Tuple[str, ...]) -> None:
            nonlocal best
            lb = self.lower_bound(chosen)
            if lb >= best:
                return
            if set(self.covered(chosen)) == set(self.universe):
                best = min(best, sum(self.costs[s] for s in chosen))
                return
            if i == len(set_ids):
                return
            rec(i + 1, chosen + (set_ids[i],))
            rec(i + 1, chosen)

        rec(0, ())
        return best

    def best_descendant_objective(self, node: SearchNode) -> float:
        return self.best_descendant_objective_from_chosen(tuple(node.structured_state["chosen_sets"]))

    def best_descendant_objective_from_chosen(self, chosen: Tuple[str, ...]) -> float:
        set_ids = tuple(s for s in self.sets if s not in chosen)
        best = float("inf")

        def rec(i: int, cur: Tuple[str, ...]) -> None:
            nonlocal best
            lb = self.lower_bound(cur)
            if lb >= best:
                return
            if set(self.covered(cur)) == set(self.universe):
                best = min(best, sum(self.costs[s] for s in cur))
                return
            if i == len(set_ids):
                return
            rec(i + 1, cur + (set_ids[i],))
            rec(i + 1, cur)

        rec(0, chosen)
        return best

    def candidate_actions(self, node: SearchNode, k: int) -> List[ChooseSetAction]:
        chosen = tuple(node.structured_state["chosen_sets"])
        covered = self.covered(chosen)
        candidates = []
        for set_id, elems in self.sets.items():
            if set_id in chosen:
                continue
            gain = len(elems.difference(covered))
            if gain > 0:
                candidates.append((-(gain / max(self.costs[set_id], 1e-9)), set_id))
        candidates.sort()
        return [ChooseSetAction(set_id=sid) for _, sid in candidates[:k]]

    def apply(self, node: SearchNode, action: ChooseSetAction, child_index: int = 0) -> SearchNode:
        chosen = tuple(node.structured_state["chosen_sets"])
        if action.set_id not in self.sets or action.set_id in chosen:
            return node.with_updates(feasible_status=FeasibleStatus.INFEASIBLE)
        del child_index
        return self._make_node(chosen=chosen + (action.set_id,), parent_id=node.node_id, action=action, node_id=f"{node.node_id}/{action.set_id}")

    def _make_node(self, chosen: Tuple[str, ...], parent_id: Optional[str], action: Optional[ChooseSetAction], node_id: str) -> SearchNode:
        complete = set(self.covered(chosen)) == set(self.universe)
        cost = sum(self.costs[s] for s in chosen)
        lower = self.lower_bound(chosen)
        return SearchNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=len(chosen),
            task_id=self.task_id,
            semantic_state={"family": "setcover"},
            structured_state={"chosen_sets": list(chosen), "covered": sorted(self.covered(chosen))},
            verified_constraints=[f"choose:{s}" for s in chosen],
            uncertain_constraints=[],
            lower_bound=lower,
            incumbent_at_creation=None,
            objective_if_complete=cost if complete else None,
            feasible_status=FeasibleStatus.FEASIBLE,
            completion_status=CompletionStatus.COMPLETE if complete else CompletionStatus.PARTIAL,
            proposal_action=action,
            observation=None,
            universal_features={"depth": float(len(chosen)), "branch_factor": float(len(self.sets) - len(chosen))},
            verifier_features={"lower_bound": lower, "verified_constraint_count": float(len(chosen)), "uncertain_constraint_count": 0.0},
            semantic_features=None,
            missingness_mask={"lower_bound": False, "semantic_features": True},
            accumulated_expensive_cost=CostLedger(),
            metadata={"family": "setcover"},
        )


def generate_setcover_fixture(rng: random.Random, task_id: str, universe_size: int, set_count: int) -> SetCoverTask:
    universe = tuple(f"e{i}" for i in range(universe_size))
    sets: Dict[str, FrozenSet[str]] = {}
    costs: Dict[str, float] = {}
    for i in range(set_count):
        size = rng.randint(1, max(1, universe_size))
        elems = set(rng.sample(universe, size))
        sets[f"S{i}"] = frozenset(elems)
        costs[f"S{i}"] = float(rng.randint(1, 5))
    for elem in universe:
        if not any(elem in s for s in sets.values()):
            sid = rng.choice(tuple(sets))
            sets[sid] = frozenset(set(sets[sid]) | {elem})
    return SetCoverTask(task_id=task_id, universe=universe, sets=sets, costs=costs)
