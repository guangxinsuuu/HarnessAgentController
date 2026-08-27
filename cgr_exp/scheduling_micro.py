from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cgr_exp.datasets.level1_scheduling import SchedulingTask, generate_scheduling_fixture
from cgr_exp.ranker import LinearRanker, Pair
from cgr_exp.schemas import CompletionStatus, SearchNode
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.search.random_policy import RandomPolicy
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier


SCHEDULING_FEATURE_NAMES = [
    "bias",
    "depth_frac",
    "scheduled_frac",
    "ready_frac",
    "remaining_frac",
    "load_max_norm",
    "load_mean_norm",
    "load_imbalance_norm",
    "lower_bound_norm",
    "incumbent_gap_norm",
    "branching_norm",
    "last_duration_norm",
    "last_release_norm",
    "last_due_norm",
    "last_finish_norm",
    "last_slack_norm",
    "last_tardiness_norm",
    "last_allowed_machine_frac",
    "last_machine_load_norm",
    "precedence_remaining_frac",
]


@dataclass(frozen=True)
class SchedulingTraceRecord:
    task_id: str
    split: str
    node_id: str
    parent_id: str | None
    depth: int
    incumbent: float | None
    lower_bound: float | None
    bound_gap: float | None
    child_count: int
    best_descendant_objective: float
    future_search_cost: float
    eventual_outcome: str
    prune_reason: str
    features: List[float]
    feature_names: List[str]
    state: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "task_id": self.task_id,
            "split": self.split,
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "incumbent": self.incumbent,
            "lower_bound": self.lower_bound,
            "bound_gap": self.bound_gap,
            "child_count": self.child_count,
            "best_descendant_objective": self.best_descendant_objective,
            "future_search_cost": self.future_search_cost,
            "eventual_outcome": self.eventual_outcome,
            "prune_reason": self.prune_reason,
            "features": self.features,
            "feature_names": self.feature_names,
            "state": self.state,
        }


class SchedulingHeuristicPolicy:
    name = "hand_heuristic"

    def score(self, node: SearchNode) -> float:
        lower = float(node.lower_bound) if node.lower_bound is not None else float("inf")
        ready = float(node.universal_features.get("branch_factor", 0.0))
        return -lower + 1e-3 * ready


class SchedulingMLPPolicy:
    name = "mlp"

    def __init__(self, task: SchedulingTask, ranker):
        self.task = task
        self.ranker = ranker
        self.cache: Dict[str, float] = {}

    def score(self, node: SearchNode) -> float:
        if node.node_id not in self.cache:
            self.cache[node.node_id] = self.ranker.score(scheduling_feature_vector(self.task, node))
        return self.cache[node.node_id]


class TinyPairwiseMLPRanker:
    def __init__(self, dim: int, hidden_dim: int = 16, seed: int = 0):
        rng = random.Random(seed)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.mean = [0.0] * dim
        self.scale = [1.0] * dim
        self.w1 = [[rng.uniform(-0.1, 0.1) for _ in range(dim)] for _ in range(hidden_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [rng.uniform(-0.1, 0.1) for _ in range(hidden_dim)]
        self.b2 = 0.0

    def fit_normalizer(self, xs: Sequence[Sequence[float]]) -> None:
        if not xs:
            return
        self.mean = [sum(x[i] for x in xs) / len(xs) for i in range(self.dim)]
        self.scale = []
        for i in range(self.dim):
            var = sum((x[i] - self.mean[i]) ** 2 for x in xs) / len(xs)
            self.scale.append(max(1e-6, math.sqrt(var)))

    def score(self, x: Sequence[float]) -> float:
        z = self._norm(x)
        h = [math.tanh(sum(w * v for w, v in zip(row, z)) + b) for row, b in zip(self.w1, self.b1)]
        return sum(w * v for w, v in zip(self.w2, h)) + self.b2

    def train(self, pairs: Sequence[Pair], epochs: int = 20, lr: float = 0.02, seed: int = 0) -> None:
        if not pairs:
            return
        self.fit_normalizer([x for pair in pairs for x in pair])
        rng = random.Random(seed)
        order = list(range(len(pairs)))
        for _ in range(epochs):
            rng.shuffle(order)
            for idx in order:
                xp, xn = pairs[idx]
                self._train_pair(xp, xn, lr)

    def ranking_error(self, pairs: Sequence[Pair]) -> float:
        if not pairs:
            return 0.0
        return sum(1 for xp, xn in pairs if self.score(xp) <= self.score(xn)) / len(pairs)

    def to_json(self) -> str:
        return json.dumps(
            {
                "ranker": "tiny_pairwise_mlp",
                "dim": self.dim,
                "hidden_dim": self.hidden_dim,
                "mean": self.mean,
                "scale": self.scale,
                "w1": self.w1,
                "b1": self.b1,
                "w2": self.w2,
                "b2": self.b2,
            },
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> "TinyPairwiseMLPRanker":
        obj = json.loads(text)
        ranker = cls(dim=int(obj["dim"]), hidden_dim=int(obj["hidden_dim"]), seed=0)
        ranker.mean = [float(x) for x in obj["mean"]]
        ranker.scale = [float(x) for x in obj["scale"]]
        ranker.w1 = [[float(v) for v in row] for row in obj["w1"]]
        ranker.b1 = [float(v) for v in obj["b1"]]
        ranker.w2 = [float(v) for v in obj["w2"]]
        ranker.b2 = float(obj["b2"])
        return ranker

    def _train_pair(self, xp: Sequence[float], xn: Sequence[float], lr: float) -> None:
        zp, hp, sp = self._forward(xp)
        zn, hn, sn = self._forward(xn)
        margin = sp - sn
        grad = -1.0 / (1.0 + math.exp(max(-40.0, min(40.0, margin))))
        self._backward(zp, hp, grad, lr)
        self._backward(zn, hn, -grad, lr)

    def _forward(self, x: Sequence[float]) -> Tuple[List[float], List[float], float]:
        z = self._norm(x)
        h = [math.tanh(sum(w * v for w, v in zip(row, z)) + b) for row, b in zip(self.w1, self.b1)]
        score = sum(w * v for w, v in zip(self.w2, h)) + self.b2
        return z, h, score

    def _backward(self, z: Sequence[float], h: Sequence[float], grad_score: float, lr: float) -> None:
        old_w2 = list(self.w2)
        for j in range(self.hidden_dim):
            self.w2[j] -= lr * grad_score * h[j]
        self.b2 -= lr * grad_score
        for j in range(self.hidden_dim):
            grad_hidden = grad_score * old_w2[j] * (1.0 - h[j] * h[j])
            for i in range(self.dim):
                self.w1[j][i] -= lr * grad_hidden * z[i]
            self.b1[j] -= lr * grad_hidden

    def _norm(self, x: Sequence[float]) -> List[float]:
        return [(float(x[i]) - self.mean[i]) / self.scale[i] for i in range(self.dim)]


class CachedSchedulingOraclePolicy:
    name = "oracle"

    def __init__(self, task: SchedulingTask):
        self.task = task
        self.cache: Dict[str, float] = {}

    def score(self, node: SearchNode) -> float:
        if node.node_id not in self.cache:
            self.cache[node.node_id] = self.task.best_descendant_objective(node)
        return -self.cache[node.node_id]


def scheduling_feature_vector(task: SchedulingTask, node: SearchNode, incumbent: float | None = None) -> List[float]:
    scheduled = tuple(node.structured_state.get("scheduled", []))
    total_duration = max(1.0, float(sum(task.durations.values())))
    loads = [float(node.structured_state.get("machine_loads", {}).get(machine, 0.0)) for machine in task.machines]
    mean_load = sum(loads) / max(1, len(loads))
    load_imbalance = max(loads or [0.0]) - min(loads or [0.0])
    ready_jobs = task.ready_jobs(scheduled)
    scheduled_jobs = {str(row["job"]) for row in scheduled}
    remaining_jobs = [job for job in task.durations if job not in scheduled_jobs]
    lower_bound = float(node.lower_bound) if node.lower_bound is not None else total_duration
    inc = incumbent if incumbent is not None else node.incumbent_at_creation
    incumbent_gap = 0.0 if inc is None else max(0.0, float(inc) - lower_bound)
    last_duration = 0.0
    last_release = 0.0
    last_due = 0.0
    last_finish = 0.0
    last_slack = 0.0
    last_tardiness = 0.0
    last_allowed_machine_frac = 0.0
    last_machine_load = 0.0
    if node.proposal_action is not None and hasattr(node.proposal_action, "job") and hasattr(node.proposal_action, "machine"):
        job = str(node.proposal_action.job)
        machine = str(node.proposal_action.machine)
        last_duration = float(task.durations.get(job, 0))
        last_release = float(task.release_times.get(job, 0))
        last_due = float(task.due_dates.get(job, 0))
        last_allowed_machine_frac = len(task.allowed.get(job, ())) / max(1, len(task.machines))
        for row in scheduled:
            if str(row["job"]) == job:
                last_finish = float(row["finish"])
                break
        last_slack = (last_due - last_finish) / total_duration
        last_tardiness = max(0.0, last_finish - last_due)
        last_machine_load = float(node.structured_state.get("machine_loads", {}).get(machine, 0.0))
    remaining_precedence = sum(len(task.predecessors.get(job, ())) for job in remaining_jobs)
    return [
        1.0,
        float(node.depth) / max(1, len(task.durations)),
        len(scheduled) / max(1, len(task.durations)),
        len(ready_jobs) / max(1, len(task.durations)),
        len(remaining_jobs) / max(1, len(task.durations)),
        max(loads or [0.0]) / total_duration,
        mean_load / total_duration,
        load_imbalance / total_duration,
        lower_bound / total_duration,
        incumbent_gap / total_duration,
        float(node.universal_features.get("branch_factor", 0.0)) / max(1, len(task.durations) * len(task.machines)),
        last_duration / total_duration,
        last_release / total_duration,
        last_due / total_duration,
        last_finish / total_duration,
        last_slack,
        last_tardiness / total_duration,
        last_allowed_machine_frac,
        last_machine_load / total_duration,
        remaining_precedence / max(1, len(task.durations)),
    ]


def generate_scheduling_micro_instances(
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    seed: int = 0,
    job_count: int = 6,
    machine_count: int = 3,
) -> Dict[str, List[SchedulingTask]]:
    rng = random.Random(seed)
    splits = {"train": train_count, "validation": validation_count, "test": test_count}
    out: Dict[str, List[SchedulingTask]] = {}
    offset = 0
    for split, count in splits.items():
        tasks = []
        for idx in range(count):
            tasks.append(generate_scheduling_fixture(rng, f"sched-{split}-{idx + offset:04d}", job_count=job_count, machine_count=machine_count))
        out[split] = tasks
        offset += count
    return out


def collect_solver_trace_records(
    task: SchedulingTask,
    split: str,
    node_budget: int = 80,
    action_budget: int = 8,
    max_records: int = 20,
) -> List[SchedulingTraceRecord]:
    verifier = Level1SchedulingVerifier()
    root = task.root_node()
    frontier: List[SearchNode] = [root]
    incumbent: float | None = None
    expanded = 0
    records: List[SchedulingTraceRecord] = []
    seen = set()
    while frontier and expanded < node_budget:
        survivors = []
        for node in frontier:
            if incumbent is not None and node.lower_bound is not None and node.lower_bound >= incumbent:
                if len(records) < max_records:
                    records.append(_trace_record(task, split, node, incumbent, 0, "bound_pruned"))
            else:
                survivors.append(node)
        frontier = survivors
        if not frontier:
            break
        node = min(frontier, key=lambda item: (float(item.lower_bound or 0.0), item.node_id))
        frontier.remove(node)
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        expanded += 1
        if node.completion_status == CompletionStatus.COMPLETE:
            value = node.objective_if_complete
            if value is not None and (incumbent is None or value < incumbent):
                incumbent = value
            if len(records) < max_records:
                records.append(_trace_record(task, split, node, incumbent, 0, "complete"))
            continue
        actions = task.candidate_actions(node, action_budget)
        children: List[SearchNode] = []
        for idx, action in enumerate(actions):
            verified = verifier.verify(task, node.with_updates(incumbent_at_creation=incumbent), action)
            if not verified.feasible:
                continue
            child = verifier.make_child(task, node.with_updates(incumbent_at_creation=incumbent), action, verified, idx)
            children.append(child)
            if len(records) < max_records:
                records.append(_trace_record(task, split, child, incumbent, 0, "child_candidate"))
            if child.completion_status == CompletionStatus.COMPLETE:
                value = child.objective_if_complete
                if value is not None and (incumbent is None or value < incumbent):
                    incumbent = value
            else:
                frontier.append(child)
        if len(records) < max_records:
            records.append(_trace_record(task, split, node, incumbent, len(children), "expanded"))
    return records


def make_scheduling_ranking_pairs(records: Sequence[SchedulingTraceRecord], max_pairs_per_task: int = 40, seed: int = 0) -> List[Pair]:
    rng = random.Random(seed)
    by_task_parent: Dict[Tuple[str, str], List[SchedulingTraceRecord]] = {}
    by_task: Dict[str, List[SchedulingTraceRecord]] = {}
    for record in records:
        by_task.setdefault(record.task_id, []).append(record)
        if record.parent_id is not None:
            by_task_parent.setdefault((record.task_id, record.parent_id), []).append(record)
    pairs: List[Pair] = []
    by_task_pairs: Dict[str, List[Pair]] = {}
    for (task_id, _parent_id), rows in by_task_parent.items():
        if len(rows) < 2:
            continue
        local: List[Pair] = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left = rows[i]
                right = rows[j]
                left_key = _ranking_key(left)
                right_key = _ranking_key(right)
                if left_key < right_key:
                    local.append((left.features, right.features))
                elif right_key < left_key:
                    local.append((right.features, left.features))
        by_task_pairs.setdefault(task_id, []).extend(local)
    for task_id, rows in by_task.items():
        local = by_task_pairs.setdefault(task_id, [])
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left = rows[i]
                right = rows[j]
                left_key = _ranking_key(left)
                right_key = _ranking_key(right)
                if left_key < right_key:
                    local.append((left.features, right.features))
                elif right_key < left_key:
                    local.append((right.features, left.features))
    for local in by_task_pairs.values():
        rng.shuffle(local)
        pairs.extend(local[:max_pairs_per_task])
    rng.shuffle(pairs)
    return pairs


def run_scheduling_micro_experiment(
    out_dir: Path,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 120,
    action_budget: int = 8,
    records_per_instance: int = 20,
    shard_size: int = 50,
    rank_epochs: int = 10,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    all_records: List[SchedulingTraceRecord] = []
    for split, tasks in instances.items():
        for task in tasks:
            all_records.extend(collect_solver_trace_records(task, split, node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    train_records = [record for record in all_records if record.split == "train"]
    val_records = [record for record in all_records if record.split == "validation"]
    test_records = [record for record in all_records if record.split == "test"]
    train_pairs = make_scheduling_ranking_pairs(train_records, max_pairs_per_task=40, seed=seed)
    val_pairs = make_scheduling_ranking_pairs(val_records, max_pairs_per_task=40, seed=seed + 1)
    linear_ranker = LinearRanker.fresh(len(SCHEDULING_FEATURE_NAMES))
    linear_ranker.train(train_pairs, epochs=rank_epochs, lr=0.04, seed=seed)
    ranker = TinyPairwiseMLPRanker(len(SCHEDULING_FEATURE_NAMES), hidden_dim=16, seed=seed)
    ranker.train(train_pairs, epochs=rank_epochs, lr=0.02, seed=seed)
    table_rows = _evaluate_scheduling_policies(instances["test"], ranker, node_budget=node_budget, action_budget=action_budget, seed=seed, ranking_records=test_records)
    metrics = {
        "benchmark": "scheduling_micro",
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
        "validation_trace_record_count": len(val_records),
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(val_pairs),
        "validation_ranking_error": ranker.ranking_error(val_pairs),
        "linear_validation_ranking_error": linear_ranker.ranking_error(val_pairs),
        "ranker_type": "tiny_pairwise_mlp",
        "methods": {row["Policy"]: row for row in table_rows},
        "artifacts": [
            "scheduling_micro_instances.json",
            "scheduling_solver_trace_shard_*.jsonl",
            "scheduling_ranking_pairs.jsonl",
            "scheduling_micro_table.csv",
            "scheduling_micro_metrics.json",
            "scheduling_mlp_ranker.json",
        ],
    }
    _write_instances(out_dir / "scheduling_micro_instances.json", instances)
    _write_trace_shards(out_dir, all_records, shard_size)
    _write_pairs(out_dir / "scheduling_ranking_pairs.jsonl", train_pairs)
    _write_csv(out_dir / "scheduling_micro_table.csv", table_rows)
    (out_dir / "scheduling_mlp_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    (out_dir / "scheduling_micro_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _trace_record(task: SchedulingTask, split: str, node: SearchNode, incumbent: float | None, child_count: int, prune_reason: str) -> SchedulingTraceRecord:
    best = task.best_descendant_objective(node)
    lower = node.lower_bound
    gap = None if lower is None or best == float("inf") else max(0.0, best - float(lower))
    outcome = "optimal_reachable" if best < float("inf") else "dead_end"
    return SchedulingTraceRecord(
        task_id=task.task_id,
        split=split,
        node_id=node.node_id,
        parent_id=node.parent_id,
        depth=node.depth,
        incumbent=incumbent,
        lower_bound=lower,
        bound_gap=gap,
        child_count=child_count,
        best_descendant_objective=best,
        future_search_cost=best,
        eventual_outcome=outcome,
        prune_reason=prune_reason,
        features=scheduling_feature_vector(task, node, incumbent=incumbent),
        feature_names=list(SCHEDULING_FEATURE_NAMES),
        state=node.to_dict(),
    )


def _ranking_key(record: SchedulingTraceRecord) -> Tuple[float, int, float]:
    lower = float(record.lower_bound) if record.lower_bound is not None else float("inf")
    return (float(record.best_descendant_objective), -int(record.depth), lower)


def _evaluate_scheduling_policies(
    tasks: Sequence[SchedulingTask],
    ranker: LinearRanker,
    node_budget: int,
    action_budget: int,
    seed: int,
    ranking_records: Sequence[SchedulingTraceRecord] | None = None,
) -> List[Dict[str, object]]:
    policies = {
        "Random": lambda task, offset: RandomPolicy(seed + offset),
        "Hand heuristic": lambda task, offset: SchedulingHeuristicPolicy(),
        "MLP": lambda task, offset: SchedulingMLPPolicy(task, ranker),
        "Oracle": lambda task, offset: CachedSchedulingOraclePolicy(task),
    }
    rows = []
    ranking_by_policy = _source_policy_ranking_diagnostics(ranking_records or [], ranker, seed)
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
        row = {
            "Policy": name,
            "success": _mean(1.0 if result.best_objective == result.optimum else 0.0 for result in results),
            "gap": _mean(0.0 if result.best_objective == result.optimum else _gap(result.best_objective, result.optimum) for result in results),
            "nodes_mean": _mean(float(result.expanded_nodes) for result in results),
            "proposal_calls_mean": _mean(float(result.cost.proposal_calls) for result in results),
            "false_prunes": sum(result.false_prune_count for result in results),
        }
        row.update(ranking_by_policy.get(name, _empty_ranking_metrics()))
        rows.append(row)
    return rows


def _source_policy_ranking_diagnostics(records: Sequence[SchedulingTraceRecord], ranker, seed: int) -> Dict[str, Dict[str, object]]:
    scorers = {
        "Random": lambda record: _stable_random_score(seed, record),
        "Hand heuristic": lambda record: -(float(record.lower_bound) if record.lower_bound is not None else float("inf")) + 1e-3 * float(record.child_count),
        "MLP": lambda record: float(ranker.score(record.features)),
        "Oracle": lambda record: -float(record.best_descendant_objective),
    }
    return {name: _ranking_diagnostics_for_records(records, scorer) for name, scorer in scorers.items()}


def _ranking_diagnostics_for_records(records: Sequence[SchedulingTraceRecord], scorer) -> Dict[str, object]:
    grouped: Dict[Tuple[str, str | None], List[SchedulingTraceRecord]] = {}
    for record in records:
        grouped.setdefault((record.task_id, record.parent_id), []).append(record)
    frontier_count = 0
    top1_hits = 0
    pair_total = 0
    pair_correct = 0
    regrets = []
    for siblings in grouped.values():
        if len(siblings) < 2:
            continue
        frontier_count += 1
        scores = [float(scorer(record)) for record in siblings]
        selected_idx = max(range(len(siblings)), key=lambda idx: (scores[idx], siblings[idx].node_id))
        oracle_values = [float(record.best_descendant_objective) for record in siblings]
        best_value = min(oracle_values)
        if oracle_values[selected_idx] == best_value:
            top1_hits += 1
        regrets.append(max(0.0, oracle_values[selected_idx] - best_value))
        for i in range(len(siblings)):
            for j in range(i + 1, len(siblings)):
                if oracle_values[i] == oracle_values[j]:
                    continue
                pair_total += 1
                better, worse = (i, j) if oracle_values[i] < oracle_values[j] else (j, i)
                pair_correct += int(scores[better] > scores[worse])
    return {
        "frontier_decisions": frontier_count,
        "frontier_top1_accuracy": 0.0 if frontier_count == 0 else top1_hits / frontier_count,
        "pairwise_accuracy": 0.0 if pair_total == 0 else pair_correct / pair_total,
        "selection_regret_mean": _mean(regrets),
    }


def _empty_ranking_metrics() -> Dict[str, object]:
    return {
        "frontier_decisions": 0,
        "frontier_top1_accuracy": 0.0,
        "pairwise_accuracy": 0.0,
        "selection_regret_mean": 0.0,
    }


def _stable_random_score(seed: int, record: SchedulingTraceRecord) -> float:
    payload = f"{seed}:{record.task_id}:{record.parent_id}:{record.node_id}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


def _write_instances(path: Path, instances: Dict[str, List[SchedulingTask]]) -> None:
    payload = {
        split: [
            {
                "task_id": task.task_id,
                "durations": task.durations,
                "machines": list(task.machines),
                "allowed": {job: list(machines) for job, machines in task.allowed.items()},
                "release_times": task.release_times,
                "due_dates": task.due_dates,
                "predecessors": {job: list(preds) for job, preds in task.predecessors.items()},
                "optimum": task.exhaustive_oracle(),
            }
            for task in tasks
        ]
        for split, tasks in instances.items()
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_trace_shards(out_dir: Path, records: Sequence[SchedulingTraceRecord], shard_size: int) -> None:
    shard_size = max(1, shard_size)
    for shard_idx, start in enumerate(range(0, len(records), shard_size)):
        path = out_dir / f"scheduling_solver_trace_shard_{shard_idx:03d}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in records[start : start + shard_size]:
                handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def _write_pairs(path: Path, pairs: Sequence[Pair]) -> None:
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
