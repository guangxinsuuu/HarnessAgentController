from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

from .features import state_features
from .problems import Problem, State, best_completion_objective, lower_bound
from .proposers import TypedProposer
from .ranker import Pair
from .verifier import Verifier


def collect_states(
    problems: Sequence[Problem],
    proposer: TypedProposer,
    verifier_factory,
    node_budget: int,
    action_budget: int,
    seed: int,
    proposer_kind: str,
) -> List[Tuple[Problem, State]]:
    rng = random.Random(seed)
    out: List[Tuple[Problem, State]] = []
    for pidx, problem in enumerate(problems):
        verifier: Verifier = verifier_factory(seed + pidx)
        frontier: List[State] = [State.root(problem, proposer_kind=proposer_kind)]
        incumbent = 10**9
        expanded = 0
        seen = set()
        while frontier and expanded < node_budget:
            frontier = [st for st in frontier if lower_bound(problem, st) < incumbent]
            if not frontier:
                break
            idx = rng.randrange(len(frontier))
            state = frontier.pop(idx)
            if state.key() in seen:
                continue
            seen.add(state.key())
            out.append((problem, state))
            expanded += 1
            if state.is_complete():
                incumbent = min(incumbent, state.objective())
                continue
            for action in proposer.propose(problem, state, action_budget):
                verdict = verifier.check(problem, state, action)
                if not verdict.schema_valid or not verdict.feasible:
                    continue
                child = state.apply(problem, action, verified=verdict.verified)
                if child.is_complete():
                    incumbent = min(incumbent, child.objective())
                else:
                    frontier.append(child)
    return out


def make_ranking_pairs(samples: Sequence[Tuple[Problem, State]], max_pairs_per_problem: int, seed: int) -> List[Pair]:
    rng = random.Random(seed)
    by_problem: Dict[str, List[Tuple[Problem, State]]] = {}
    for problem, state in samples:
        by_problem.setdefault(problem.pid, []).append((problem, state))

    pairs: List[Pair] = []
    for rows in by_problem.values():
        if len(rows) < 2:
            continue
        candidates = []
        memo = {}
        for problem, state in rows:
            value = best_completion_objective(problem, state, memo)
            feat = state_features(problem, state)
            candidates.append((value, feat))
        local_pairs: List[Pair] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                vi, fi = candidates[i]
                vj, fj = candidates[j]
                if vi < vj:
                    local_pairs.append((fi, fj))
                elif vj < vi:
                    local_pairs.append((fj, fi))
        rng.shuffle(local_pairs)
        pairs.extend(local_pairs[:max_pairs_per_problem])
    rng.shuffle(pairs)
    return pairs

