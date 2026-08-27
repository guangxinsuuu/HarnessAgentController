from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import List, Optional

from cgr_exp.features import state_features
from cgr_exp.problems import Problem, State, best_completion_objective, exact_optimum, lower_bound
from cgr_exp.proposers import TypedProposer, proposal_quality
from cgr_exp.ranker import LinearRanker
from cgr_exp.verifier import Verifier


@dataclass
class SearchTraceEvent:
    depth: int
    lb: int
    incumbent: int
    frontier: int
    action_count: int
    proposal_recall: float
    schema_valid: float
    duplicate_rate: float


@dataclass
class SearchResult:
    objective: Optional[int]
    optimum: int
    success: bool
    gap: float
    nodes_expanded: int
    proposer_calls: int
    false_prunes: int
    pruned: int
    trace: List[SearchTraceEvent] = field(default_factory=list)


def _priority(
    problem: Problem,
    state: State,
    incumbent: int,
    policy: str,
    ranker: Optional[LinearRanker],
    rng: random.Random,
) -> float:
    if policy == "dfs":
        return state.depth + rng.random() * 1e-6
    if policy == "random":
        return rng.random()
    if policy == "best_bound":
        return -lower_bound(problem, state)
    if policy == "oracle":
        return -best_completion_objective(problem, state)
    if policy == "learned":
        if ranker is None:
            raise ValueError("learned policy requires ranker")
        return ranker.score(state_features(problem, state, incumbent))
    raise ValueError(f"unknown policy: {policy}")


def cgr_search(
    problem: Problem,
    proposer: TypedProposer,
    verifier: Verifier,
    policy: str,
    ranker: Optional[LinearRanker] = None,
    node_budget: int = 500,
    action_budget: int = 8,
    seed: int = 0,
    proposer_kind: str = "solver",
) -> SearchResult:
    rng = random.Random(seed)
    optimum = exact_optimum(problem)
    incumbent = 10**9
    frontier: List[State] = [State.root(problem, proposer_kind=proposer_kind)]
    expanded = 0
    calls = 0
    pruned = 0
    false_prunes = 0
    trace: List[SearchTraceEvent] = []
    seen = set()

    while frontier and expanded < node_budget:
        survivors: List[State] = []
        for st in frontier:
            lb = lower_bound(problem, st)
            if lb >= incumbent:
                pruned += 1
                if best_completion_objective(problem, st) < incumbent:
                    false_prunes += 1
            else:
                survivors.append(st)
        frontier = survivors
        if not frontier:
            break

        best_idx = max(
            range(len(frontier)),
            key=lambda i: _priority(problem, frontier[i], incumbent, policy, ranker, rng),
        )
        state = frontier.pop(best_idx)
        if state.key() in seen:
            continue
        seen.add(state.key())
        expanded += 1

        if state.is_complete():
            incumbent = min(incumbent, state.objective())
            continue

        actions = proposer.propose(problem, state, action_budget)
        calls += 1
        recall, schema_valid, duplicate = proposal_quality(problem, state, actions)
        trace.append(SearchTraceEvent(state.depth, lower_bound(problem, state), incumbent, len(frontier), len(actions), recall, schema_valid, duplicate))

        for action in actions:
            verdict = verifier.check(problem, state, action)
            if not verdict.schema_valid or not verdict.feasible:
                continue
            child = state.apply(problem, action, verified=verdict.verified)
            if child.is_complete():
                incumbent = min(incumbent, child.objective())
                continue
            if verdict.verified and verdict.lower_bound is not None and verdict.lower_bound >= incumbent:
                pruned += 1
                if best_completion_objective(problem, child) < incumbent:
                    false_prunes += 1
                continue
            frontier.append(child)

    objective = None if incumbent == 10**9 else incumbent
    gap = 1.0 if objective is None else max(0.0, (objective - optimum) / max(1, optimum))
    return SearchResult(
        objective=objective,
        optimum=optimum,
        success=objective == optimum,
        gap=gap,
        nodes_expanded=expanded,
        proposer_calls=calls,
        false_prunes=false_prunes,
        pruned=pruned,
        trace=trace,
    )

