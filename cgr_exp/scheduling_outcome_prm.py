from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cgr_exp.datasets.level1_scheduling import ScheduleAction, SchedulingTask
from cgr_exp.scheduling_micro import (
    SCHEDULING_FEATURE_NAMES,
    SchedulingMLPPolicy,
    TinyPairwiseMLPRanker,
    generate_scheduling_micro_instances,
)
from cgr_exp.schemas import CompletionStatus, CostLedger
from cgr_exp.search.controller import FrontierSearchResult


def run_scheduling_outcome_prm(
    source_run_dir: Path,
    out_dir: Path,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 80,
    action_budget: int = 6,
    rank_epochs: int = 25,
    max_pairs_per_task: int = 80,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    train_records = _read_trace_records(source_run_dir / "target_prm_trace_records.jsonl")
    pairs = _make_outcome_pairs(train_records, max_pairs_per_task=max_pairs_per_task, seed=seed + 17)
    ranker = TinyPairwiseMLPRanker(len(SCHEDULING_FEATURE_NAMES), hidden_dim=16, seed=seed + 23)
    ranker.train(pairs, epochs=rank_epochs, lr=0.02, seed=seed + 29)
    cache = _CachedProposalReplay.from_jsonl(source_run_dir / "proposal_cache.jsonl")
    started = time.perf_counter()
    per_instance_rows = []
    for idx, task in enumerate(instances["test"]):
        result = _run_cached_search(task, SchedulingMLPPolicy(task, ranker), cache, node_budget=node_budget, action_budget=action_budget)
        per_instance_rows.append(
            {
                "Policy": "Outcome-PRM",
                "task_id": task.task_id,
                "success": result.best_objective == result.optimum,
                "gap": _gap(result.best_objective, result.optimum),
                "nodes": result.expanded_nodes,
                "proposal_calls": result.cost.proposal_calls,
                "proposal_tokens": result.cost.input_tokens + result.cost.output_tokens,
                "value_calls": 0,
                "value_tokens": 0,
                "total_tokens": result.cost.input_tokens + result.cost.output_tokens,
                "false_prunes": result.false_prune_count,
                "budget_exhausted": result.budget_exhausted,
            }
        )
        if (idx + 1) % 10 == 0 or idx + 1 == len(instances["test"]):
            print(f"[outcome-prm] {idx + 1}/{len(instances['test'])} test episodes", flush=True)
    table_rows = [_summarize_policy_rows(per_instance_rows)]
    metrics = {
        "benchmark": "scheduling_outcome_prm",
        "source_run_dir": str(source_run_dir),
        "seed": seed,
        "splits": {split: len(tasks) for split, tasks in instances.items()},
        "training_states": "target LLM-induced Scheduling states",
        "label_definition": "outcome/global pairwise ranking by exact best_descendant_objective, not frontier-relative sibling ranking",
        "target_trace_record_count": len(train_records),
        "outcome_pair_count": len(pairs),
        "rank_epochs": rank_epochs,
        "node_budget": node_budget,
        "action_budget": action_budget,
        "shared_cached_proposal_replay": True,
        "oracle_labels_in_inference_features": False,
        "wall_clock_seconds": time.perf_counter() - started,
        "methods": {"Outcome-PRM": table_rows[0]},
        "artifacts": [
            "outcome_prm_table.csv",
            "outcome_prm_per_instance.csv",
            "outcome_prm_metrics.json",
            "outcome_prm_ranker.json",
        ],
    }
    _write_csv(out_dir / "outcome_prm_table.csv", table_rows)
    _write_csv(out_dir / "outcome_prm_per_instance.csv", per_instance_rows)
    (out_dir / "outcome_prm_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "outcome_prm_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    return metrics


def _run_cached_search(task: SchedulingTask, policy, cache: "_CachedProposalReplay", node_budget: int, action_budget: int) -> FrontierSearchResult:
    optimum = task.exhaustive_oracle()
    frontier = [task.root_node()]
    incumbent = None
    expanded = 0
    generated = 0
    frontier_peak = 1
    trace = []
    cost = CostLedger()
    while frontier and expanded < node_budget:
        node = max(frontier, key=lambda item: (float(policy.score(item)), item.node_id))
        frontier.remove(node)
        expanded += 1
        if node.completion_status == CompletionStatus.COMPLETE:
            value = node.objective_if_complete
            if value is not None and (incumbent is None or value < incumbent):
                incumbent = value
            continue
        actions, token_cost = cache.actions_for(task, node, action_budget)
        cost = cost + CostLedger(proposal_calls=1, input_tokens=token_cost[0], output_tokens=token_cost[1])
        if not actions:
            continue
        for idx, action in enumerate(actions[:action_budget]):
            if not task.is_valid_action(tuple(node.structured_state.get("scheduled", [])), action):
                continue
            child = task.apply(node.with_updates(incumbent_at_creation=incumbent), action, child_index=idx)
            generated += 1
            if child.completion_status == CompletionStatus.COMPLETE:
                value = child.objective_if_complete
                if value is not None and (incumbent is None or value < incumbent):
                    incumbent = value
            else:
                frontier.append(child)
                frontier_peak = max(frontier_peak, len(frontier))
    return FrontierSearchResult(
        best_objective=incumbent,
        optimum=optimum,
        false_prune_count=0,
        expanded_nodes=expanded,
        generated_children=generated,
        frontier_peak_size=frontier_peak,
        cost=cost,
        trace=trace,
        budget_exhausted=bool(frontier and expanded >= node_budget and incumbent != optimum),
        pruned_nodes=0,
        certificate_count=0,
    )


class _CachedProposalReplay:
    def __init__(self, rows: Dict[Tuple[str, Tuple[str, ...]], Dict[str, object]]):
        self.rows = rows

    @classmethod
    def from_jsonl(cls, path: Path) -> "_CachedProposalReplay":
        rows: Dict[Tuple[str, Tuple[str, ...]], Dict[str, object]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                state = row.get("canonical_state", {})
                if not isinstance(state, dict):
                    continue
                key = (str(row.get("instance_id")), _history_key(state.get("observable_history", [])))
                rows[key] = row
        return cls(rows)

    def actions_for(self, task: SchedulingTask, node, k: int) -> tuple[List[ScheduleAction], tuple[int, int]]:
        row = self.rows.get((task.task_id, _history_key(node.structured_state.get("scheduled", []))))
        if row is None:
            return task.candidate_actions(node, k), (0, 0)
        actions = [
            ScheduleAction(job=str(item.get("job", "")), machine=str(item.get("machine", "")))
            for item in row.get("actions", [])
            if isinstance(item, dict)
        ]
        return actions[:k], (int(row.get("input_tokens", 0)), int(row.get("output_tokens", 0)))


def _history_key(history: object) -> Tuple[Tuple[str, str, int, int], ...]:
    if not isinstance(history, list):
        return ()
    key = []
    for row in history:
        if not isinstance(row, dict):
            continue
        key.append((str(row.get("job", "")), str(row.get("machine", "")), int(row.get("start", 0)), int(row.get("finish", 0))))
    return tuple(key)


def _make_outcome_pairs(records: Sequence[Dict[str, object]], max_pairs_per_task: int, seed: int):
    rng = random.Random(seed)
    by_task: Dict[str, List[Dict[str, object]]] = {}
    for row in records:
        if isinstance(row.get("features"), list) and row.get("best_descendant_objective") is not None:
            by_task.setdefault(str(row["task_id"]), []).append(row)
    pairs = []
    for task_id, rows in by_task.items():
        del task_id
        local = []
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                left = rows[i]
                right = rows[j]
                left_value = float(left["best_descendant_objective"])
                right_value = float(right["best_descendant_objective"])
                if left_value == right_value:
                    continue
                if left_value < right_value:
                    local.append((left["features"], right["features"]))
                else:
                    local.append((right["features"], left["features"]))
        if len(local) > max_pairs_per_task:
            local = rng.sample(local, max_pairs_per_task)
        pairs.extend(local)
    return pairs


def _read_trace_records(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _summarize_policy_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    return {
        "Policy": "Outcome-PRM",
        "success": _mean(1.0 if _as_bool(row["success"]) else 0.0 for row in rows),
        "gap": _mean(float(row["gap"]) for row in rows),
        "nodes_mean": _mean(float(row["nodes"]) for row in rows),
        "proposal_calls_mean": _mean(float(row["proposal_calls"]) for row in rows),
        "value_calls": sum(int(row["value_calls"]) for row in rows),
        "proposal_tokens": sum(int(row["proposal_tokens"]) for row in rows),
        "value_tokens": sum(int(row["value_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "false_prunes": sum(int(row["false_prunes"]) for row in rows),
        "budget_failures": sum(1 for row in rows if _as_bool(row["budget_exhausted"])),
    }


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
