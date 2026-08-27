from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cgr_exp.datasets.level1_scheduling import SchedulingTask
from cgr_exp.scheduling_micro import (
    CachedSchedulingOraclePolicy,
    SCHEDULING_FEATURE_NAMES,
    SchedulingHeuristicPolicy,
    SchedulingTraceRecord,
    TinyPairwiseMLPRanker,
    collect_solver_trace_records,
    generate_scheduling_micro_instances,
    scheduling_feature_vector,
)
from cgr_exp.schemas import SearchNode
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.search.random_policy import RandomPolicy
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier


TREE_FEATURE_NAMES = [
    *SCHEDULING_FEATURE_NAMES,
    "tree_parent_bound_delta_norm",
    "tree_last_start_norm",
    "tree_sibling_index_norm",
    "tree_path_machine_switch_frac",
]

GRAPH_FEATURE_NAMES = [
    *TREE_FEATURE_NAMES,
    "graph_ready_resource_competition_norm",
    "graph_same_machine_remaining_frac",
    "graph_single_machine_pressure_frac",
    "graph_successor_unlock_frac",
    "graph_critical_path_remaining_norm",
]


class SchedulingFeaturePolicy:
    name = "scheduling_feature_policy"

    def __init__(self, task: SchedulingTask, ranker: TinyPairwiseMLPRanker, representation: str):
        self.task = task
        self.ranker = ranker
        self.representation = representation
        self.cache: Dict[str, float] = {}

    def score(self, node: SearchNode) -> float:
        if node.node_id not in self.cache:
            self.cache[node.node_id] = self.ranker.score(scheduling_representation_vector(self.task, node, self.representation))
        return self.cache[node.node_id]


def run_scheduling_representation_ablation(
    out_dir: Path,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 120,
    action_budget: int = 8,
    records_per_instance: int = 60,
    rank_epochs: int = 25,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    all_records: List[SchedulingTraceRecord] = []
    for split, tasks in instances.items():
        for task in tasks:
            all_records.extend(collect_solver_trace_records(task, split, node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    task_map = {task.task_id: task for tasks in instances.values() for task in tasks}
    train_records = [record for record in all_records if record.split == "train"]
    validation_records = [record for record in all_records if record.split == "validation"]
    rankers = {}
    ranking_errors = {}
    pair_counts = {}
    for representation in ("mlp", "tree_gnn", "graph_gnn"):
        train_pairs = make_representation_pairs(train_records, task_map, representation=representation, max_pairs_per_task=40, seed=seed)
        validation_pairs = make_representation_pairs(validation_records, task_map, representation=representation, max_pairs_per_task=40, seed=seed + 1)
        ranker = _train_ranker(train_pairs, rank_epochs, seed)
        rankers[representation] = ranker
        ranking_errors[representation] = ranker.ranking_error(validation_pairs)
        pair_counts[representation] = {"train": len(train_pairs), "validation": len(validation_pairs)}
    table_rows = _evaluate_representation_policies(instances["test"], rankers, node_budget=node_budget, action_budget=action_budget, seed=seed)
    metrics = {
        "benchmark": "scheduling_representation_ablation",
        "seed": seed,
        "splits": {split: len(tasks) for split, tasks in instances.items()},
        "job_count": job_count,
        "machine_count": machine_count,
        "node_budget": node_budget,
        "action_budget": action_budget,
        "records_per_instance": records_per_instance,
        "rank_epochs": rank_epochs,
        "trace_record_count": len(all_records),
        "train_trace_record_count": len(train_records),
        "validation_trace_record_count": len(validation_records),
        "pair_counts": pair_counts,
        "validation_ranking_errors": ranking_errors,
        "ranker_type": "tiny_pairwise_mlp_over_representation_features",
        "methods": {row["Policy"]: row for row in table_rows},
        "artifacts": [
            "scheduling_representation_table.csv",
            "scheduling_representation_metrics.json",
            "scheduling_representation_pairs_*.jsonl",
            "scheduling_representation_ranker_*.json",
        ],
    }
    _write_csv(out_dir / "scheduling_representation_table.csv", table_rows)
    for representation in ("mlp", "tree_gnn", "graph_gnn"):
        pairs = make_representation_pairs(train_records, task_map, representation=representation, max_pairs_per_task=40, seed=seed)
        _write_pairs(out_dir / f"scheduling_representation_pairs_{representation}.jsonl", pairs)
        (out_dir / f"scheduling_representation_ranker_{representation}.json").write_text(rankers[representation].to_json(), encoding="utf-8")
    (out_dir / "scheduling_representation_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def scheduling_representation_vector(task: SchedulingTask, node: SearchNode, representation: str) -> List[float]:
    base = scheduling_feature_vector(task, node)
    if representation == "mlp":
        return base
    tree = [*base, *_tree_features(task, node)]
    if representation == "tree_gnn":
        return tree
    if representation == "graph_gnn":
        return [*tree, *_graph_features(task, node)]
    raise ValueError(f"unknown scheduling representation: {representation}")


def make_representation_pairs(
    records: Sequence[SchedulingTraceRecord],
    tasks_by_id: Dict[str, SchedulingTask],
    representation: str,
    max_pairs_per_task: int,
    seed: int,
) -> List[Tuple[List[float], List[float]]]:
    rng = random.Random(seed)
    grouped: Dict[str, List[SchedulingTraceRecord]] = {}
    for record in records:
        if record.task_id in tasks_by_id:
            grouped.setdefault(record.task_id, []).append(record)
    pairs: List[Tuple[List[float], List[float]]] = []
    for task_id, task_records in grouped.items():
        task = tasks_by_id[task_id]
        candidates = []
        ordered = sorted(task_records, key=_ranking_key)
        for i, better in enumerate(ordered):
            for worse in ordered[i + 1 :]:
                if _ranking_key(better) == _ranking_key(worse):
                    continue
                better_node = SearchNode.from_dict(better.state)
                worse_node = SearchNode.from_dict(worse.state)
                candidates.append(
                    (
                        scheduling_representation_vector(task, better_node, representation),
                        scheduling_representation_vector(task, worse_node, representation),
                    )
                )
        rng.shuffle(candidates)
        pairs.extend(candidates[:max_pairs_per_task])
    rng.shuffle(pairs)
    return pairs


def _evaluate_representation_policies(
    tasks: Sequence[SchedulingTask],
    rankers: Dict[str, TinyPairwiseMLPRanker],
    node_budget: int,
    action_budget: int,
    seed: int,
) -> List[Dict[str, object]]:
    policies = {
        "Random": lambda task, offset: RandomPolicy(seed + offset),
        "Hand heuristic": lambda task, offset: SchedulingHeuristicPolicy(),
        "MLP": lambda task, offset: SchedulingFeaturePolicy(task, rankers["mlp"], "mlp"),
        "Tree-GNN": lambda task, offset: SchedulingFeaturePolicy(task, rankers["tree_gnn"], "tree_gnn"),
        "Graph-GNN": lambda task, offset: SchedulingFeaturePolicy(task, rankers["graph_gnn"], "graph_gnn"),
        "Oracle": lambda task, offset: CachedSchedulingOraclePolicy(task),
    }
    rows = []
    for name, factory in policies.items():
        results = []
        for idx, task in enumerate(tasks):
            result = run_frontier_search(
                task=task,
                policy=factory(task, idx),
                verifier=Level1SchedulingVerifier(),
                node_budget=node_budget,
                proposal_budget=action_budget,
                seed=seed + idx,
            )
            results.append(result)
        rows.append(
            {
                "Policy": name,
                "success": _mean(1.0 if result.best_objective == result.optimum else 0.0 for result in results),
                "gap": _mean(0.0 if result.best_objective == result.optimum else _gap(result.best_objective, result.optimum) for result in results),
                "nodes_mean": _mean(float(result.expanded_nodes) for result in results),
                "proposal_calls_mean": _mean(float(result.cost.proposal_calls) for result in results),
                "false_prunes": sum(result.false_prune_count for result in results),
            }
        )
    return rows


def _tree_features(task: SchedulingTask, node: SearchNode) -> List[float]:
    scheduled = tuple(node.structured_state.get("scheduled", []))
    total_duration = max(1.0, float(sum(task.durations.values())))
    parent_scheduled = scheduled[:-1]
    parent_lower = task.lower_bound(parent_scheduled) if scheduled else node.lower_bound
    lower = float(node.lower_bound) if node.lower_bound is not None else total_duration
    parent_delta = 0.0 if parent_lower is None else max(0.0, float(parent_lower) - lower) / total_duration
    last_start = float(scheduled[-1]["start"]) / total_duration if scheduled else 0.0
    sibling_index = _last_child_index(node.node_id) / max(1.0, float(len(task.durations) * len(task.machines)))
    switches = 0
    for prev, cur in zip(scheduled, scheduled[1:]):
        if str(prev["machine"]) != str(cur["machine"]):
            switches += 1
    switch_frac = switches / max(1, len(scheduled) - 1)
    return [parent_delta, last_start, sibling_index, switch_frac]


def _graph_features(task: SchedulingTask, node: SearchNode) -> List[float]:
    scheduled = tuple(node.structured_state.get("scheduled", []))
    scheduled_jobs = {str(row["job"]) for row in scheduled}
    remaining = [job for job in task.durations if job not in scheduled_jobs]
    ready = task.ready_jobs(scheduled)
    total_duration = max(1.0, float(sum(task.durations.values())))
    action = node.proposal_action
    action_job = str(getattr(action, "job", "")) if action is not None else ""
    action_machine = str(getattr(action, "machine", "")) if action is not None else ""
    ready_competition = sum(1 for job in ready for machine in task.allowed[job] if machine == action_machine)
    same_machine_remaining = sum(1 for job in remaining if action_machine in task.allowed[job])
    single_machine_pressure = sum(1 for job in remaining if len(task.allowed[job]) == 1 and action_machine in task.allowed[job])
    successors = [job for job, preds in task.predecessors.items() if action_job in preds and job not in scheduled_jobs]
    unlocked = sum(1 for job in successors if all(pred in scheduled_jobs or pred == action_job for pred in task.predecessors.get(job, ())))
    critical_path = _critical_path_remaining(task, remaining)
    denom = max(1, len(task.durations))
    return [
        ready_competition / max(1, len(ready) * len(task.machines)),
        same_machine_remaining / denom,
        single_machine_pressure / denom,
        unlocked / max(1, len(successors)),
        critical_path / total_duration,
    ]


def _critical_path_remaining(task: SchedulingTask, remaining: Sequence[str]) -> float:
    remaining_set = set(remaining)
    memo: Dict[str, float] = {}

    def visit(job: str) -> float:
        if job in memo:
            return memo[job]
        successors = [other for other in remaining_set if job in task.predecessors.get(other, ())]
        memo[job] = float(task.durations[job]) + max((visit(child) for child in successors), default=0.0)
        return memo[job]

    return max((visit(job) for job in remaining_set), default=0.0)


def _last_child_index(node_id: str) -> float:
    try:
        tail = node_id.rsplit("/", 1)[-1]
        return float(tail.split(":", 1)[0])
    except (IndexError, TypeError, ValueError):
        return 0.0


def _ranking_key(record: SchedulingTraceRecord) -> Tuple[float, int, float]:
    lower = float(record.lower_bound) if record.lower_bound is not None else float("inf")
    return (float(record.best_descendant_objective), -int(record.depth), lower)


def _train_ranker(pairs: Sequence[Tuple[List[float], List[float]]], rank_epochs: int, seed: int) -> TinyPairwiseMLPRanker:
    dim = len(pairs[0][0]) if pairs else len(SCHEDULING_FEATURE_NAMES)
    ranker = TinyPairwiseMLPRanker(dim=dim, hidden_dim=16, seed=seed)
    if pairs:
        ranker.train(pairs, epochs=rank_epochs, lr=0.02, seed=seed)
    return ranker


def _write_pairs(path: Path, pairs: Sequence[Tuple[List[float], List[float]]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for positive, negative in pairs:
            handle.write(json.dumps({"positive_features": positive, "negative_features": negative}, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _gap(best_objective: float | None, optimum: float) -> float:
    if best_objective is None:
        return 1.0
    return max(0.0, (float(best_objective) - float(optimum)) / max(1.0, float(optimum)))


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return 0.0 if not data else sum(data) / len(data)
