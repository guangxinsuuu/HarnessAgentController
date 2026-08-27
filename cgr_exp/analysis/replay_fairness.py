from __future__ import annotations

import hashlib
import json
from typing import Iterable, Sequence

from cgr_exp.graph.graph_builder import build_graph_edges
from cgr_exp.models.graph_gnn import GraphGNNRanker
from cgr_exp.models.tree_gnn import TreeGNNRanker
from cgr_exp.proposers.replay import ReplayProposer, build_replay_from_task
from cgr_exp.search.best_bound import BestBoundPolicy
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.search.dfs import DFSPolicy
from cgr_exp.search.random_policy import RandomPolicy
from cgr_exp.verification.level1_verifier import Level1AssignmentVerifier
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier
from cgr_exp.verification.level1_setcover_verifier import Level1SetCoverVerifier


class RankerPolicy:
    def __init__(self, name: str, ranker, edge_config):
        self.name = name
        self.ranker = ranker
        self.edge_config = edge_config

    def score(self, node):
        # Controller asks one node at a time; local context contains current node only.
        edges = build_graph_edges([node], self.edge_config)
        return self.ranker.score_nodes([node], {"edges": edges})[node.node_id]


def _policy(name: str, seed: int):
    if name == "dfs":
        return DFSPolicy()
    if name == "random":
        return RandomPolicy(seed=seed)
    if name == "best_bound":
        return BestBoundPolicy()
    if name == "tree_gnn":
        return RankerPolicy("tree_gnn", TreeGNNRanker(), {})
    if name == "graph_gnn":
        return RankerPolicy("graph_gnn", GraphGNNRanker(), {"sibling_failure": True, "shared_constraint": True, "equivalent_state": True})
    raise ValueError(f"unknown policy: {name}")


def replay_hash(replay) -> str:
    serial = {
        node_id: [action.to_dict() for action in actions]
        for node_id, actions in sorted(replay.items())
    }
    return hashlib.sha256(json.dumps(serial, sort_keys=True).encode("utf-8")).hexdigest()


def verifier_for_task(task, seed: int):
    if getattr(task, "sets", None) is not None and getattr(task, "universe", None) is not None:
        del seed
        return Level1SetCoverVerifier()
    if getattr(task, "machines", None) is not None and getattr(task, "predecessors", None) is not None:
        del seed
        return Level1SchedulingVerifier()
    return Level1AssignmentVerifier(seed=seed)


def run_replay_fairness_matrix(task, seed: int, policies: Sequence[str], node_budget: int, proposal_budget: int):
    replay = build_replay_from_task(task, proposal_budget)
    digest = replay_hash(replay)
    rows = []
    for policy_name in policies:
        result = run_frontier_search(
            task=task,
            policy=_policy(policy_name, seed),
            verifier=verifier_for_task(task, seed),
            node_budget=node_budget,
            proposal_budget=proposal_budget,
            seed=seed,
            proposer=ReplayProposer(replay),
        )
        rows.append(
            {
                "policy": policy_name,
                "replay_hash": digest,
                "same_proposal_support": True,
                "best_objective": result.best_objective,
                "optimum": result.optimum,
                "success": result.best_objective == result.optimum,
                "expanded_nodes": result.expanded_nodes,
                "proposal_calls": result.cost.proposal_calls,
                "verifier_calls": result.cost.verifier_calls,
            }
        )
    return rows
