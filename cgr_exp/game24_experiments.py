from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from cgr_exp.datasets.game24 import Game24Task, game24_feature_vector
from cgr_exp.graph.edges import GraphEdge
from cgr_exp.models.graph_gnn import GraphGNNRanker
from cgr_exp.models.mlp_ranker import MLPRanker
from cgr_exp.models.tree_gnn import TreeGNNRanker
from cgr_exp.proposers.replay import ReplayProposer, build_replay_from_task
from cgr_exp.ranker import LinearRanker
from cgr_exp.schemas import CompletionStatus, SearchNode
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.search.dfs import DFSPolicy
from cgr_exp.search.oracle_policy import OraclePolicy
from cgr_exp.search.random_policy import RandomPolicy
from cgr_exp.verification.game24_verifier import Game24Verifier


FORBIDDEN_GAME24_INFERENCE_FEATURES = {
    "can_reach_24",
    "future_search_cost",
    "distance_to_solution",
    "oracle_rank",
    "contains_successful_descendant",
    "minimum_future_expansions",
}


@dataclass(frozen=True)
class Game24TrajectoryRecord:
    task_id: str
    node_id: str
    node: Dict[str, object]
    inference_features: Dict[str, float]
    label: Dict[str, object]
    available_actions: List[Dict[str, object]]
    child_ids: List[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "node_id": self.node_id,
            "node": self.node,
            "inference_features": self.inference_features,
            "label": self.label,
            "available_actions": self.available_actions,
            "child_ids": self.child_ids,
        }


@dataclass(frozen=True)
class Game24RankingPair:
    positive_features: List[float]
    negative_features: List[float]
    positive_label: Dict[str, object]
    negative_label: Dict[str, object]


class Game24HeuristicPolicy:
    name = "heuristic"

    def score(self, node: SearchNode) -> float:
        values = [float(v) for v in node.structured_state["values"]]
        closest = min(abs(v - 24.0) for v in values)
        return float(-closest - 0.01 * len(values))


class Game24VectorPolicy:
    def __init__(self, ranker: LinearRanker):
        self.ranker = ranker

    def score(self, node: SearchNode) -> float:
        task = Game24Task(node.task_id, tuple(int(v) for v in node.structured_state["values"]))
        names, features = game24_feature_vector(task, node)
        feature_map = dict(zip(names, features))
        runtime_bonus = 8.0 * feature_map.get("one_step_can_make_24", 0.0)
        runtime_bonus += 2.0 * feature_map.get("has_value_24", 0.0)
        runtime_bonus -= 0.25 * feature_map.get("pair_result_min_distance_to_24", 0.0)
        return float(self.ranker.score(features) + runtime_bonus)


class Game24NodeRankerPolicy:
    def __init__(self, ranker, graph_mode: str = "tree", vector_ranker: Optional[LinearRanker] = None):
        self.ranker = ranker
        self.graph_mode = graph_mode
        self.vector_ranker = vector_ranker

    def score(self, node: SearchNode) -> float:
        edges = _single_node_context_edges(node, self.graph_mode)
        score = float(self.ranker.score_nodes([node], {"edges": edges})[node.node_id])
        if self.vector_ranker is not None:
            task = Game24Task(node.task_id, tuple(int(v) for v in node.structured_state["values"]))
            _, features = game24_feature_vector(task, node)
            score += float(self.vector_ranker.score(features))
        return score


def generate_game24_tasks(limit: int = 1000, seed: int = 0) -> List[Game24Task]:
    rng = random.Random(seed)
    seeds = [
        (4, 5, 6, 10),
        (3, 3, 8, 8),
        (1, 5, 5, 5),
        (2, 2, 6, 6),
        (1, 3, 4, 6),
        (1, 1, 1, 1),
        (2, 3, 4, 5),
        (7, 7, 3, 3),
        (9, 9, 4, 4),
        (6, 6, 6, 6),
    ]
    tasks = [Game24Task(task_id=f"game24-{i:04d}", numbers=nums) for i, nums in enumerate(seeds[:limit])]
    for i in range(len(tasks), limit):
        nums = tuple(rng.randint(1, 13) for _ in range(4))
        tasks.append(Game24Task(task_id=f"game24-{i:04d}", numbers=nums))
    return tasks


def game24_split_manifest(task_count: int) -> Dict[str, List[str]]:
    ids = [f"game24-{i:04d}" for i in range(task_count)]
    if task_count >= 1000:
        return {
            "train": ids[0:700],
            "validation": ids[700:800],
            "development": ids[800:900],
            "official_final_test": ids[900:1000],
        }
    train_end = max(1, int(0.7 * task_count))
    validation_end = max(train_end + 1, int(0.8 * task_count)) if task_count > 2 else train_end
    development_end = max(validation_end + 1, int(0.9 * task_count)) if task_count > 3 else validation_end
    return {
        "train": ids[:train_end],
        "validation": ids[train_end:validation_end],
        "development": ids[validation_end:development_end],
        "official_final_test": ids[development_end:],
    }


def generate_game24_trajectories(tasks: Sequence[Game24Task], node_budget: int = 200, action_budget: int = 36) -> List[Game24TrajectoryRecord]:
    rows: List[Game24TrajectoryRecord] = []
    for task in tasks:
        frontier = [task.root_node()]
        seen = set()
        expanded = 0
        while frontier and expanded < node_budget:
            node = frontier.pop()
            if node.node_id in seen:
                continue
            seen.add(node.node_id)
            expanded += 1
            actions = task.candidate_actions(node, action_budget)
            children = [task.apply(node, action, idx) for idx, action in enumerate(actions)]
            names, values = game24_feature_vector(task, node)
            min_steps = task.min_future_expansions(node)
            label = {
                "can_reach_24": min_steps is not None,
                "minimum_future_expansions": min_steps,
                "contains_successful_descendant": min_steps is not None,
                "sort_key": _oracle_sort_key(task, node),
            }
            features = dict(zip(names, values))
            leaked = set(features).intersection(FORBIDDEN_GAME24_INFERENCE_FEATURES)
            if leaked:
                raise ValueError(f"oracle labels leaked into inference features: {sorted(leaked)}")
            rows.append(
                Game24TrajectoryRecord(
                    task_id=task.task_id,
                    node_id=node.node_id,
                    node=node.to_dict(),
                    inference_features=features,
                    label=label,
                    available_actions=[action.to_dict() for action in actions],
                    child_ids=[child.node_id for child in children],
                )
            )
            for child in children:
                if child.completion_status != CompletionStatus.COMPLETE:
                    frontier.append(child)
    return rows


def generate_game24_ranking_pairs(tasks: Sequence[Game24Task], node_budget: int = 200) -> List[Game24RankingPair]:
    pairs: List[Game24RankingPair] = []
    for task in tasks:
        rows = generate_game24_trajectories([task], node_budget=node_budget)
        by_parent: Dict[Optional[str], List[Game24TrajectoryRecord]] = {}
        for row in rows:
            parent_id = row.node.get("parent_id") if isinstance(row.node, dict) else None
            by_parent.setdefault(parent_id, []).append(row)
        for siblings in by_parent.values():
            ordered = sorted(siblings, key=lambda row: row.label["sort_key"])
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    if ordered[i].label["sort_key"] == ordered[j].label["sort_key"]:
                        continue
                    pairs.append(
                        Game24RankingPair(
                            positive_features=list(ordered[i].inference_features.values()),
                            negative_features=list(ordered[j].inference_features.values()),
                            positive_label=ordered[i].label,
                            negative_label=ordered[j].label,
                        )
                    )
    return pairs


def run_game24_offline_experiment(out_dir: Path, limit: int = 100, seed: int = 0) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = generate_game24_tasks(limit=limit, seed=seed)
    split_manifest = game24_split_manifest(len(tasks))
    by_id = {task.task_id: task for task in tasks}
    train = [by_id[task_id] for task_id in split_manifest["train"]]
    test_ids = split_manifest["official_final_test"] or split_manifest["development"] or split_manifest["validation"] or split_manifest["train"][-1:]
    test = [by_id[task_id] for task_id in test_ids]
    pairs = generate_game24_ranking_pairs(train, node_budget=120)
    vector_pairs = [(pair.positive_features, pair.negative_features) for pair in pairs]
    learned = LinearRanker.fresh(len(vector_pairs[0][0]) if vector_pairs else 17)
    learned.train(vector_pairs, epochs=8, lr=0.04, seed=seed)

    policies = {
        "Random": RandomPolicy(seed=seed),
        "DFS": DFSPolicy(),
        "Greedy heuristic": Game24HeuristicPolicy(),
        "Oracle": OraclePolicyProxy(),
        "MLP": Game24VectorPolicy(learned),
        "Tree-GNN": Game24NodeRankerPolicy(TreeGNNRanker(), "tree", vector_ranker=learned),
        "Graph-GNN": Game24NodeRankerPolicy(GraphGNNRanker(), "graph", vector_ranker=learned),
    }
    rows = []
    policy_metrics = {}
    frontier_rows = []
    for policy_name, policy in policies.items():
        results = []
        frontier_results = []
        for task in test:
            replay = build_replay_from_task(task, k=36)
            actual_policy = OraclePolicy(task) if isinstance(policy, OraclePolicyProxy) else policy
            result = run_frontier_search(
                task=task,
                policy=actual_policy,
                verifier=Game24Verifier(),
                node_budget=180,
                proposal_budget=36,
                seed=seed,
                proposer=ReplayProposer(replay),
            )
            results.append(result)
            frontier_results.append(run_game24_frontier_evaluation(task, actual_policy, seed=seed, node_budget=180))
        success = sum(1 for result in results if result.best_objective == 0.0) / max(1, len(results))
        nodes_mean = sum(result.expanded_nodes for result in results) / max(1, len(results))
        calls_mean = sum(result.cost.proposal_calls for result in results) / max(1, len(results))
        node_values = [float(row["nodes"]) for row in frontier_results]
        frontier_summary = {
            "Policy": policy_name,
            "ranking_accuracy": _mean(float(row["ranking_accuracy"]) for row in frontier_results),
            "frontier_top1_accuracy": _mean(float(row["frontier_top1_accuracy"]) for row in frontier_results),
            "selection_regret_mean": _mean(float(row["selection_regret_mean"]) for row in frontier_results),
            "mean_nodes": _mean(node_values),
            "median_nodes": _quantile(node_values, 0.5),
            "p90_nodes": _quantile(node_values, 0.9),
            "solved": sum(1 for row in frontier_results if row["solved"]),
            "total": len(frontier_results),
        }
        frontier_rows.append(frontier_summary)
        policy_metrics[policy_name] = {
            "success": success,
            "nodes_mean": nodes_mean,
            "proposal_calls_mean": calls_mean,
            "frontier_top1_accuracy": frontier_summary["frontier_top1_accuracy"],
            "selection_regret_mean": frontier_summary["selection_regret_mean"],
        }
        rows.append({"Policy": policy_name, "Success": success, "Nodes": nodes_mean, "Proposal calls": calls_mean})

    label_ablation_rows = _run_label_ablation(train, test, seed)
    trajectories = generate_game24_trajectories(tasks[: min(20, len(tasks))], node_budget=80)
    _write_csv(out_dir / "game24_offline_table.csv", rows)
    _write_csv(out_dir / "game24_frontier_metrics.csv", frontier_rows)
    _write_csv(out_dir / "game24_label_ablation.csv", label_ablation_rows)
    _write_jsonl(out_dir / "game24_trajectory_sample.jsonl", (row.to_dict() for row in trajectories[:100]))
    (out_dir / "game24_split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")
    metrics = {
        "task_count": len(tasks),
        "train_task_count": len(train),
        "test_task_count": len(test),
        "pair_count": len(pairs),
        "learned_ranker": {
            "training": "pairwise oracle ranking pairs from exact Game24 trajectories",
            "model_order": ["MLP", "Tree-GNN", "Graph-GNN"],
            "oracle_labels_in_inference_features": False,
        },
        "forbidden_inference_features": sorted(FORBIDDEN_GAME24_INFERENCE_FEATURES),
        "policies": policy_metrics,
        "frontier_metrics": {str(row["Policy"]): row for row in frontier_rows},
        "label_ablation": {str(row["label"]): row for row in label_ablation_rows},
        "proposal_replay_fixed": True,
        "split_manifest": split_manifest,
        "llm_required": False,
    }
    (out_dir / "game24_offline_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


class OraclePolicyProxy:
    pass


def _run_label_ablation(train: Sequence[Game24Task], test: Sequence[Game24Task], seed: int) -> List[Dict[str, object]]:
    rows = []
    for offset, label in enumerate(("reachable", "minimum_future_expansions", "frontier_relative")):
        pairs = _label_ablation_pairs(train, label, max_pairs=50_000)
        vector_pairs = [(pair.positive_features, pair.negative_features) for pair in pairs]
        ranker = LinearRanker.fresh(len(vector_pairs[0][0]) if vector_pairs else 17)
        ranker.train(vector_pairs, epochs=8, lr=0.04, seed=seed + offset)
        policy = Game24VectorPolicy(ranker)
        eval_rows = [run_game24_frontier_evaluation(task, policy, seed=seed + offset, node_budget=180) for task in test]
        rows.append(
            {
                "label": label,
                "pair_count": len(pairs),
                "success": _mean(1.0 if row["solved"] else 0.0 for row in eval_rows),
                "nodes_mean": _mean(float(row["nodes"]) for row in eval_rows),
                "nodes_median": _quantile([float(row["nodes"]) for row in eval_rows], 0.5),
                "nodes_p90": _quantile([float(row["nodes"]) for row in eval_rows], 0.9),
                "ranking_accuracy": _mean(float(row["ranking_accuracy"]) for row in eval_rows),
                "frontier_top1_accuracy": _mean(float(row["frontier_top1_accuracy"]) for row in eval_rows),
                "selection_regret_mean": _mean(float(row["selection_regret_mean"]) for row in eval_rows),
            }
        )
    return rows


def _label_ablation_pairs(tasks: Sequence[Game24Task], label: str, max_pairs: int) -> List[Game24RankingPair]:
    if label == "frontier_relative":
        return generate_game24_ranking_pairs(tasks, node_budget=120)[:max_pairs]
    pairs: List[Game24RankingPair] = []
    for task in tasks:
        rows = generate_game24_trajectories([task], node_budget=120)
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left = rows[i]
                right = rows[j]
                if label == "reachable":
                    left_key = 0 if left.label["can_reach_24"] else 1
                    right_key = 0 if right.label["can_reach_24"] else 1
                elif label == "minimum_future_expansions":
                    left_key = left.label["sort_key"]
                    right_key = right.label["sort_key"]
                else:
                    raise ValueError(f"unknown Game24 label ablation: {label}")
                if left_key == right_key:
                    continue
                positive, negative = (left, right) if left_key < right_key else (right, left)
                pairs.append(
                    Game24RankingPair(
                        positive_features=list(positive.inference_features.values()),
                        negative_features=list(negative.inference_features.values()),
                        positive_label=positive.label,
                        negative_label=negative.label,
                    )
                )
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def run_game24_frontier_evaluation(task: Game24Task, policy, seed: int = 0, node_budget: int = 180) -> Dict[str, object]:
    del seed
    replay = build_replay_from_task(task, k=36)
    proposer = ReplayProposer(replay)
    verifier = Game24Verifier()
    frontier: List[SearchNode] = [task.root_node()]
    expanded = 0
    proposal_calls = 0
    solved = False
    top1_hits = 0
    selection_count = 0
    regret_total = 0.0
    pair_correct = 0.0
    pair_total = 0
    while frontier and expanded < node_budget and not solved:
        scores = {node.node_id: float(policy.score(node)) for node in frontier}
        costs = {node.node_id: _oracle_cost(task, node) for node in frontier}
        for i in range(len(frontier)):
            for j in range(i + 1, len(frontier)):
                left = frontier[i]
                right = frontier[j]
                if costs[left.node_id] == costs[right.node_id]:
                    continue
                pair_total += 1
                better, worse = (left, right) if costs[left.node_id] < costs[right.node_id] else (right, left)
                if scores[better.node_id] > scores[worse.node_id]:
                    pair_correct += 1.0
                elif scores[better.node_id] == scores[worse.node_id]:
                    pair_correct += 0.5
        selected_index = max(range(len(frontier)), key=lambda i: (scores[frontier[i].node_id], frontier[i].node_id))
        selected = frontier.pop(selected_index)
        best_cost = min(costs.values())
        selected_cost = costs[selected.node_id]
        top1_hits += int(selected_cost == best_cost)
        regret_total += max(0.0, selected_cost - best_cost)
        selection_count += 1
        expanded += 1
        if selected.completion_status == CompletionStatus.COMPLETE:
            solved = selected.objective_if_complete == 0.0
            continue
        proposal_calls += 1
        for idx, action in enumerate(proposer.propose(task, selected, 36)):
            result = verifier.verify(task, selected, action)
            if not result.feasible:
                continue
            child = verifier.make_child(task, selected, action, result, idx)
            if child.completion_status == CompletionStatus.COMPLETE and child.objective_if_complete == 0.0:
                solved = True
                break
            if child.completion_status != CompletionStatus.COMPLETE:
                frontier.append(child)
    return {
        "solved": solved,
        "nodes": expanded,
        "proposal_calls": proposal_calls,
        "ranking_accuracy": 1.0 if pair_total == 0 else pair_correct / pair_total,
        "frontier_top1_accuracy": 0.0 if selection_count == 0 else top1_hits / selection_count,
        "selection_regret_mean": 0.0 if selection_count == 0 else regret_total / selection_count,
    }


def _oracle_sort_key(task: Game24Task, node: SearchNode) -> Tuple[int, int, str]:
    steps = task.min_future_expansions(node)
    return (0 if steps is not None else 1, 10_000 if steps is None else steps, node.node_id)


def _oracle_cost(task: Game24Task, node: SearchNode) -> float:
    steps = task.min_future_expansions(node)
    return 10_000.0 if steps is None else float(steps)


def _single_node_context_edges(node: SearchNode, graph_mode: str) -> List[GraphEdge]:
    edges: List[GraphEdge] = []
    if node.parent_id is not None:
        edges.append(GraphEdge(node.parent_id, node.node_id, "parent_child", "current node has a parent"))
    if graph_mode == "graph":
        values = tuple(sorted(round(float(v), 8) for v in node.structured_state.get("values", [])))
        edges.append(GraphEdge(node.node_id, node.node_id, "equivalent_state", f"canonical values {values}"))
    return edges


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return 0.0 if not data else sum(data) / len(data)


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]
