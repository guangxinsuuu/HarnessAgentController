from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Dict, List, Optional

from cgr_exp.certification.prune import should_prune
from cgr_exp.budgets import BudgetState
from cgr_exp.schemas import CompletionStatus, CostLedger, SearchNode


@dataclass(frozen=True)
class FrontierSearchResult:
    best_objective: Optional[float]
    optimum: float
    false_prune_count: int
    expanded_nodes: int
    generated_children: int
    frontier_peak_size: int
    cost: CostLedger
    trace: List[Dict[str, Any]]
    budget_exhausted: bool = False
    pruned_nodes: int = 0
    certificate_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["cost"] = self.cost.to_dict()
        return data


def run_frontier_search(
    task: Any,
    policy: Any,
    verifier: Any,
    node_budget: int,
    proposal_budget: int,
    seed: int,
    proposer: Any = None,
    proposal_call_budget: Optional[int] = None,
    evaluation_call_budget: Optional[int] = None,
    generated_token_budget: Optional[int] = None,
    certified_pruning: bool = True,
) -> FrontierSearchResult:
    del seed
    start = time.perf_counter()
    optimum = task.exhaustive_oracle()
    incumbent: Optional[float] = None
    root = task.root_node()
    frontier: List[SearchNode] = [root]
    expanded = 0
    generated = 0
    false_prunes = 0
    pruned_nodes = 0
    certificate_count = 0
    budget = BudgetState(proposal_call_budget=proposal_call_budget, evaluation_call_budget=evaluation_call_budget, generated_token_budget=generated_token_budget)
    budget.cost = CostLedger(solver_calls=1)
    trace: List[Dict[str, Any]] = []
    frontier_peak = 1

    budget_exhausted = False
    while frontier and expanded < node_budget:
        survivors: List[SearchNode] = []
        for node in frontier:
            decision = should_prune(node, incumbent) if certified_pruning else None
            if decision is None:
                survivors.append(node)
                continue
            if decision.prune:
                pruned_nodes += 1
                certificate_count += 1
                if incumbent is not None and task.best_descendant_objective(node) < incumbent:
                    false_prunes += 1
                trace.append({"event": "prune", "node_id": node.node_id, "reason": decision.reason})
            else:
                survivors.append(node)
        frontier = survivors
        if not frontier:
            break
        selected_index = max(range(len(frontier)), key=lambda i: (policy.score(frontier[i]), frontier[i].node_id))
        node = frontier.pop(selected_index)
        expanded += 1
        trace.append({"event": "expand", "node": node.to_dict(), "frontier_size": len(frontier), "incumbent": incumbent})

        if node.completion_status == CompletionStatus.COMPLETE:
            value = node.objective_if_complete
            if value is not None and (incumbent is None or value < incumbent):
                incumbent = value
            continue

        if not budget.can_charge_proposal():
            budget_exhausted = True
            trace.append({"event": "budget_exhausted", "reason": "proposal_call_budget"})
            break
        budget.charge_proposal()
        actions = proposer.propose(task, node, proposal_budget, budget) if proposer is not None else task.candidate_actions(node, proposal_budget)
        for idx, action in enumerate(actions):
            verified = verifier.verify(task, node, action)
            budget.charge_verifier()
            if not verified.feasible:
                continue
            child = verifier.make_child(task, node.with_updates(incumbent_at_creation=incumbent), action, verified, idx)
            generated += 1
            if child.completion_status == CompletionStatus.COMPLETE:
                value = child.objective_if_complete
                if value is not None and (incumbent is None or value < incumbent):
                    incumbent = value
            else:
                frontier.append(child)
                frontier_peak = max(frontier_peak, len(frontier))

    elapsed = time.perf_counter() - start
    budget.cost = budget.cost + CostLedger(wall_clock_seconds=elapsed)
    return FrontierSearchResult(incumbent, optimum, false_prunes, expanded, generated, frontier_peak, budget.cost, trace, budget_exhausted, pruned_nodes, certificate_count)


def write_trace_jsonl(path, result: FrontierSearchResult, limit: int = 20) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in result.trace[:limit]:
            f.write(json.dumps(row, sort_keys=True) + "\n")
