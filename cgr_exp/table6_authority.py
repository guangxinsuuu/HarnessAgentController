from __future__ import annotations

import csv
from dataclasses import replace
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cgr_exp.certification.prune import should_prune
from cgr_exp.datasets.level1_scheduling import ScheduleAction, SchedulingTask
from cgr_exp.scheduling_micro import (
    SCHEDULING_FEATURE_NAMES,
    SchedulingMLPPolicy,
    TinyPairwiseMLPRanker,
    collect_solver_trace_records,
    generate_scheduling_micro_instances,
    make_scheduling_ranking_pairs,
)
from cgr_exp.schemas import CompletionStatus, CostLedger, FeasibleStatus, SearchNode
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier


TABLE6_SETTINGS = [
    ("Clean certified", "clean", 0.0),
    ("Verified FN 1%", "verified_fn", 0.01),
    ("Verified FN 5%", "verified_fn", 0.05),
    ("Verified FN 10%", "verified_fn", 0.10),
    ("Verified FN 20%", "verified_fn", 0.20),
    ("Verified FP 0.5%", "verified_fp", 0.005),
    ("Verified FP 1%", "verified_fp", 0.01),
    ("Verified FP 2%", "verified_fp", 0.02),
    ("Verified FP 5%", "verified_fp", 0.05),
    ("Uncertain-only 1%", "uncertain_only", 0.01),
    ("Uncertain-only 5%", "uncertain_only", 0.05),
    ("Uncertain-only 10%", "uncertain_only", 0.10),
    ("Uncertain-only 20%", "uncertain_only", 0.20),
    ("Learned-score pruning clean", "learned_score_pruning", 0.20),
]


def run_table6_authority_experiment(
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
    corruption_seeds: Sequence[int] = (0, 1, 2),
    settings: Sequence[Tuple[str, str, float]] = tuple(TABLE6_SETTINGS),
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    train_records = []
    validation_records = []
    for task in instances["train"]:
        train_records.extend(collect_solver_trace_records(task, "train", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    for task in instances["validation"]:
        validation_records.extend(collect_solver_trace_records(task, "validation", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    train_pairs = make_scheduling_ranking_pairs(train_records, max_pairs_per_task=40, seed=seed)
    validation_pairs = make_scheduling_ranking_pairs(validation_records, max_pairs_per_task=40, seed=seed + 1)
    ranker = TinyPairwiseMLPRanker(len(SCHEDULING_FEATURE_NAMES), hidden_dim=16, seed=seed)
    ranker.train(train_pairs, epochs=rank_epochs, lr=0.02, seed=seed)

    per_instance_rows = []
    started = time.perf_counter()
    total = len(settings) * len(corruption_seeds) * len(instances["test"])
    done = 0
    for label, kind, severity in settings:
        for corruption_seed in corruption_seeds:
            for task_idx, task in enumerate(instances["test"]):
                corruption = _make_corruption(task, kind, severity, seed + 100_000 + corruption_seed * 1009 + task_idx)
                row = _evaluate_corrupted_task(
                    true_task=task,
                    ranker=ranker,
                    label=label,
                    kind=kind,
                    severity=severity,
                    corruption_seed=corruption_seed,
                    corruption=corruption,
                    node_budget=node_budget,
                    action_budget=action_budget,
                )
                per_instance_rows.append(row)
                done += 1
                if done % max(1, len(instances["test"])) == 0:
                    print(f"[table6-authority] {done}/{total} condition-task rows", flush=True)
    table_rows = _summarize_table6(per_instance_rows)
    elapsed = time.perf_counter() - started
    metrics = {
        "benchmark": "table6_authority_separation",
        "seed": seed,
        "splits": {split: len(tasks) for split, tasks in instances.items()},
        "job_count": job_count,
        "machine_count": machine_count,
        "node_budget": node_budget,
        "action_budget": action_budget,
        "records_per_instance": records_per_instance,
        "rank_epochs": rank_epochs,
        "corruption_seeds": list(corruption_seeds),
        "settings": [{"Condition": label, "kind": kind, "severity": severity} for label, kind, severity in settings],
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "validation_ranking_error": ranker.ranking_error(validation_pairs),
        "authority_boundary": "uncertain constraints may affect ranking only; verified constraints may affect irreversible pruning",
        "semantic_false_prune_audit": "each pruned/rejected state is audited against the uncorrupted true descendant subproblem",
        "oracle_labels_in_inference_features": False,
        "wall_clock_seconds": elapsed,
        "artifacts": [
            "table6_authority.csv",
            "table6_authority_per_instance.csv",
            "table6_authority_metrics.json",
            "table6_authority_ranker.json",
        ],
    }
    _write_csv(out_dir / "table6_authority.csv", table_rows)
    _write_csv(out_dir / "table6_authority_per_instance.csv", per_instance_rows)
    (out_dir / "table6_authority_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "table6_authority_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    return {**metrics, "rows": table_rows}


def _evaluate_corrupted_task(
    true_task: SchedulingTask,
    ranker,
    label: str,
    kind: str,
    severity: float,
    corruption: Dict[str, object],
    corruption_seed: int,
    node_budget: int,
    action_budget: int,
) -> Dict[str, object]:
    cert_task = corruption["cert_task"]
    false_precedences = list(corruption["false_precedences"])
    policy = SchedulingMLPPolicy(true_task, ranker)
    if kind == "uncertain_only":
        policy = _UncertainBiasedPolicy(true_task, ranker, false_precedences)
    learned_prune = kind == "learned_score_pruning"
    result = _run_authority_search(
        true_task=true_task,
        cert_task=cert_task,
        policy=policy,
        false_precedences=false_precedences,
        node_budget=node_budget,
        action_budget=action_budget,
        learned_prune=learned_prune,
        learned_prune_fraction=severity,
    )
    optimum = true_task.exhaustive_oracle()
    generated = result["generated_children"]
    pruned = result["certified_prunes"] + result["verified_infeasible_rejections"] + result["learned_score_prunes"]
    return {
        "Condition": label,
        "corruption_kind": kind,
        "severity": severity,
        "corruption_seed": corruption_seed,
        "task_id": true_task.task_id,
        "verified_precision": corruption["verified_precision"],
        "verified_recall": corruption["verified_recall"],
        "bound_validity_rate": result["bound_validity_rate"],
        "bound_slack": result["bound_slack"],
        "success": result["best_objective"] == optimum,
        "gap": _gap(result["best_objective"], optimum),
        "prune_rate": 0.0 if generated == 0 else pruned / generated,
        "semantic_false_prune": result["semantic_false_prunes"],
        "audited_prune_decisions": result["audited_prune_decisions"],
        "nodes": result["expanded_nodes"],
        "tokens": result["tokens"],
        "proposal_calls": result["proposal_calls"],
        "verified_fn_count": corruption["verified_fn_count"],
        "verified_fp_count": corruption["verified_fp_count"],
        "uncertain_false_count": corruption["uncertain_false_count"],
        "false_precedence_count": len(false_precedences),
        "certified_prunes": result["certified_prunes"],
        "verified_infeasible_rejections": result["verified_infeasible_rejections"],
        "learned_score_prunes": result["learned_score_prunes"],
        "best_objective": "" if result["best_objective"] is None else result["best_objective"],
        "optimum": optimum,
    }


def _run_authority_search(
    true_task: SchedulingTask,
    cert_task: SchedulingTask,
    policy,
    false_precedences: Sequence[Tuple[str, str]],
    node_budget: int,
    action_budget: int,
    learned_prune: bool,
    learned_prune_fraction: float,
) -> Dict[str, object]:
    verifier = Level1SchedulingVerifier()
    frontier: List[SearchNode] = [_with_cert_lower_bound(true_task.root_node(), cert_task)]
    incumbent = None
    expanded = 0
    generated = 0
    proposal_calls = 0
    certified_prunes = 0
    verified_rejections = 0
    learned_prunes = 0
    semantic_false_prunes = 0
    audited = 0
    bound_checks = 0
    valid_bounds = 0
    slacks: List[float] = []
    seen = set()
    descendant_cache: Dict[Tuple[Tuple[str, str, int, int], ...], float] = {}

    def true_descendant_best(node: SearchNode) -> float:
        scheduled = tuple(node.structured_state.get("scheduled", []))
        key = tuple((str(row["job"]), str(row["machine"]), int(row["start"]), int(row["finish"])) for row in scheduled)
        if key not in descendant_cache:
            descendant_cache[key] = true_task.best_descendant_objective(node)
        return descendant_cache[key]

    while frontier and expanded < node_budget:
        survivors = []
        for node in frontier:
            true_best = true_descendant_best(node)
            _record_bound_audit(node, true_best, slacks)
            bound_checks += 1
            if node.lower_bound is not None and float(node.lower_bound) <= true_best + 1e-9:
                valid_bounds += 1
            decision = should_prune(node, incumbent)
            if decision.prune:
                certified_prunes += 1
                audited += 1
                if incumbent is not None and true_best < incumbent:
                    semantic_false_prunes += 1
                continue
            survivors.append(node)
        frontier = survivors
        if learned_prune and len(frontier) > 1:
            scored = [(float(policy.score(node)), node) for node in frontier]
            scored.sort(key=lambda item: (item[0], item[1].node_id))
            prune_count = min(len(scored) - 1, int(round(len(scored) * learned_prune_fraction)))
            pruned_ids = {node.node_id for _, node in scored[:prune_count]}
            kept = []
            for score, node in scored:
                del score
                if node.node_id in pruned_ids:
                    learned_prunes += 1
                    audited += 1
                    true_best = true_descendant_best(node)
                    if incumbent is None or true_best < incumbent:
                        semantic_false_prunes += 1
                else:
                    kept.append(node)
            frontier = kept
        if not frontier:
            break
        node = max(frontier, key=lambda item: (float(policy.score(item)), item.node_id))
        frontier.remove(node)
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        expanded += 1
        if node.completion_status == CompletionStatus.COMPLETE:
            value = node.objective_if_complete
            if value is not None and (incumbent is None or value < incumbent):
                incumbent = value
            continue
        proposal_calls += 1
        actions = true_task.candidate_actions(node, action_budget)
        for idx, action in enumerate(actions):
            generated += 1
            if not _cert_allows_action(cert_task, node, action):
                verified_rejections += 1
                audited += 1
                true_child = true_task.apply(node, action, child_index=idx)
                true_best = true_descendant_best(true_child)
                if incumbent is None or true_best < incumbent:
                    semantic_false_prunes += 1
                continue
            verified = verifier.verify(true_task, node.with_updates(incumbent_at_creation=incumbent), action)
            if not verified.feasible:
                continue
            child = verifier.make_child(true_task, node.with_updates(incumbent_at_creation=incumbent), action, verified, idx)
            child = _with_cert_lower_bound(child, cert_task, false_precedences=false_precedences)
            if child.completion_status == CompletionStatus.COMPLETE:
                value = child.objective_if_complete
                if value is not None and (incumbent is None or value < incumbent):
                    incumbent = value
            else:
                frontier.append(child)
    tokens = proposal_calls * (120 + 16 * action_budget)
    return {
        "best_objective": incumbent,
        "expanded_nodes": expanded,
        "generated_children": generated,
        "proposal_calls": proposal_calls,
        "tokens": tokens,
        "certified_prunes": certified_prunes,
        "verified_infeasible_rejections": verified_rejections,
        "learned_score_prunes": learned_prunes,
        "semantic_false_prunes": semantic_false_prunes,
        "audited_prune_decisions": audited,
        "bound_validity_rate": 1.0 if bound_checks == 0 else valid_bounds / bound_checks,
        "bound_slack": _mean(slacks),
        "cost": CostLedger(proposal_calls=proposal_calls, input_tokens=proposal_calls * 120, output_tokens=proposal_calls * 16 * action_budget),
    }


def _with_cert_lower_bound(node: SearchNode, cert_task: SchedulingTask, false_precedences: Sequence[Tuple[str, str]] = ()) -> SearchNode:
    scheduled = tuple(node.structured_state.get("scheduled", []))
    feasible = FeasibleStatus.FEASIBLE
    if not _schedule_prefix_respects_precedences(scheduled, cert_task.predecessors):
        feasible = FeasibleStatus.INFEASIBLE
    return node.with_updates(
        lower_bound=cert_task.lower_bound(scheduled),
        feasible_status=feasible,
        uncertain_constraints=[f"false-precedence:{a}->{b}" for a, b in false_precedences],
    )


def _cert_allows_action(cert_task: SchedulingTask, node: SearchNode, action: ScheduleAction) -> bool:
    scheduled = tuple(node.structured_state.get("scheduled", []))
    if action.job not in cert_task.durations or action.machine not in cert_task.machines:
        return False
    if action.machine not in cert_task.allowed[action.job]:
        return False
    return action.job in cert_task.ready_jobs(scheduled)


def _make_corruption(task: SchedulingTask, kind: str, severity: float, seed: int) -> Dict[str, object]:
    rng = random.Random(seed)
    true_edges = sorted((pred, job) for job, preds in task.predecessors.items() for pred in preds)
    kept_edges = list(true_edges)
    false_edges: List[Tuple[str, str]] = []
    uncertain_edges: List[Tuple[str, str]] = []
    if kind == "verified_fn" and true_edges:
        removed = {edge for edge in true_edges if rng.random() < severity}
        kept_edges = [edge for edge in true_edges if edge not in removed]
    elif kind == "verified_fp":
        candidates = _false_precedence_candidates(task)
        false_edges = [edge for edge in candidates if rng.random() < severity]
    elif kind == "uncertain_only":
        candidates = _false_precedence_candidates(task)
        uncertain_edges = [edge for edge in candidates if rng.random() < severity]
    pred_map: Dict[str, Tuple[str, ...]] = {job: () for job in task.durations}
    for pred, job in [*kept_edges, *false_edges]:
        pred_map[job] = tuple(sorted({*pred_map.get(job, ()), pred}))
    cert_task = replace(task, predecessors=pred_map)
    true_count = len(true_edges)
    fn_count = true_count - len(kept_edges)
    fp_count = len(false_edges)
    precision_den = len(kept_edges) + fp_count
    recall_den = true_count
    return {
        "cert_task": cert_task,
        "false_precedences": false_edges or uncertain_edges,
        "verified_precision": 1.0 if precision_den == 0 else len(kept_edges) / precision_den,
        "verified_recall": 1.0 if recall_den == 0 else len(kept_edges) / recall_den,
        "verified_fn_count": fn_count,
        "verified_fp_count": fp_count,
        "uncertain_false_count": len(uncertain_edges),
    }


def _false_precedence_candidates(task: SchedulingTask) -> List[Tuple[str, str]]:
    jobs = sorted(task.durations)
    reachable = _precedence_reachability(task.predecessors, jobs)
    candidates = []
    for pred in jobs:
        for job in jobs:
            if pred == job:
                continue
            if (pred, job) in reachable or (job, pred) in reachable:
                continue
            candidates.append((pred, job))
    return candidates


def _precedence_reachability(predecessors: Dict[str, Tuple[str, ...]], jobs: Sequence[str]) -> set[Tuple[str, str]]:
    children: Dict[str, List[str]] = {job: [] for job in jobs}
    for job, preds in predecessors.items():
        for pred in preds:
            children.setdefault(pred, []).append(job)
    out = set()
    for start in jobs:
        stack = list(children.get(start, []))
        while stack:
            cur = stack.pop()
            if (start, cur) in out:
                continue
            out.add((start, cur))
            stack.extend(children.get(cur, []))
    return out


def _schedule_prefix_respects_precedences(scheduled: Sequence[Dict[str, object]], predecessors: Dict[str, Tuple[str, ...]]) -> bool:
    done = set()
    for row in scheduled:
        job = str(row["job"])
        if any(pred not in done for pred in predecessors.get(job, ())):
            return False
        done.add(job)
    return True


def _record_bound_audit(node: SearchNode, true_best: float, slacks: List[float]) -> None:
    if node.lower_bound is None:
        return
    if true_best < float("inf"):
        slacks.append(max(0.0, true_best - float(node.lower_bound)))


class _UncertainBiasedPolicy:
    def __init__(self, task: SchedulingTask, ranker, false_precedences: Sequence[Tuple[str, str]]):
        self.base = SchedulingMLPPolicy(task, ranker)
        self.false_precedences = tuple(false_precedences)

    def score(self, node: SearchNode) -> float:
        score = self.base.score(node)
        scheduled_jobs = [str(row["job"]) for row in node.structured_state.get("scheduled", [])]
        positions = {job: idx for idx, job in enumerate(scheduled_jobs)}
        for pred, job in self.false_precedences:
            if pred in positions and job in positions and positions[pred] < positions[job]:
                score += 0.15
            elif job in positions and pred not in positions:
                score -= 0.15
        return score


def _summarize_table6(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Condition"]), []).append(row)
    out = []
    order = [label for label, _, _ in TABLE6_SETTINGS]
    for condition in [*order, *sorted(set(grouped) - set(order))]:
        data = grouped.get(condition, [])
        if not data:
            continue
        out.append(
            {
                "Condition": condition,
                "Verified precision": _mean(float(row["verified_precision"]) for row in data),
                "Verified recall": _mean(float(row["verified_recall"]) for row in data),
                "Bound validity": _mean(float(row["bound_validity_rate"]) for row in data),
                "Success": _mean(1.0 if _as_bool(row["success"]) else 0.0 for row in data),
                "Gap": _mean(float(row["gap"]) for row in data),
                "Prune rate": _mean(float(row["prune_rate"]) for row in data),
                "Semantic false prune": sum(int(row["semantic_false_prune"]) for row in data),
                "Audited prune decisions": sum(int(row["audited_prune_decisions"]) for row in data),
                "Nodes": _mean(float(row["nodes"]) for row in data),
                "Tokens": sum(int(row["tokens"]) for row in data),
                "Bound slack": _mean(float(row["bound_slack"]) for row in data),
                "Replicates": len(data),
            }
        )
    return out


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
