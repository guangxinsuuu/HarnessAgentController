from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from cgr_exp.datasets.level1_scheduling import SchedulingTask
from cgr_exp.scheduling_micro import (
    SCHEDULING_FEATURE_NAMES,
    SchedulingMLPPolicy,
    TinyPairwiseMLPRanker,
    collect_solver_trace_records,
    generate_scheduling_micro_instances,
    make_scheduling_ranking_pairs,
)
from cgr_exp.scheduling_transfer import SimulatedLLMSchedulingProposer, make_level2_scheduling_episodes
from cgr_exp.search.controller import run_frontier_search
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier


def run_scheduling_certification_experiment(
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
    useful_action_recall: float = 0.88,
    perturbation: str = "mixed",
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
    episodes = make_level2_scheduling_episodes(instances["test"], seed=seed + 6000, perturbation=perturbation)
    table_rows, per_instance_rows = _evaluate_certification(
        episodes=episodes,
        ranker=ranker,
        node_budget=node_budget,
        action_budget=action_budget,
        seed=seed,
        useful_action_recall=useful_action_recall,
    )
    metrics = {
        "benchmark": "scheduling_certification",
        "seed": seed,
        "splits": {split: len(tasks) for split, tasks in instances.items()},
        "job_count": job_count,
        "machine_count": machine_count,
        "node_budget": node_budget,
        "action_budget": action_budget,
        "records_per_instance": records_per_instance,
        "rank_epochs": rank_epochs,
        "ranker_type": "tiny_pairwise_mlp",
        "training_source": "level1_solver_traces_only",
        "level2_perturbation": perturbation,
        "useful_action_recall": useful_action_recall,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "validation_ranking_error": ranker.ranking_error(validation_pairs),
        "methods": {row["Policy"]: row for row in table_rows},
        "certification_rule": "prune iff verified lower_bound >= incumbent or verifier proves infeasible",
        "certification_protocol": {
            "pruning_authority": "verified lower-bound/feasibility logic only",
            "learned_score_authority": "ranking only",
            "learned_score_can_prune": False,
            "uncertain_constraints_can_prune": False,
            "verified_pruning_conditions": [
                "verified lower_bound >= incumbent",
                "verifier proves infeasible",
            ],
            "semantic_false_prune_audit": "each pruned state is checked against the unpruned true descendant subproblem",
            "oracle_labels_in_inference_features": False,
        },
        "artifacts": [
            "scheduling_certification_table.csv",
            "table6_clean_certification.csv",
            "scheduling_certification_per_instance.csv",
            "scheduling_certification_metrics.json",
            "scheduling_certification_ranker.json",
        ],
    }
    _write_csv(out_dir / "scheduling_certification_table.csv", table_rows)
    _write_csv(out_dir / "table6_clean_certification.csv", table_rows)
    _write_csv(out_dir / "scheduling_certification_per_instance.csv", per_instance_rows)
    (out_dir / "scheduling_certification_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    (out_dir / "scheduling_certification_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _evaluate_certification(
    episodes,
    ranker: TinyPairwiseMLPRanker,
    node_budget: int,
    action_budget: int,
    seed: int,
    useful_action_recall: float,
):
    settings = {
        "learned priority only": False,
        "learned priority + certified pruning": True,
    }
    by_setting: Dict[str, List[Dict[str, object]]] = {name: [] for name in settings}
    per_instance_rows = []
    for idx, episode in enumerate(episodes):
        for name, certified_pruning in settings.items():
            proposer = SimulatedLLMSchedulingProposer(episode, seed=seed + idx * 1009, useful_action_recall=useful_action_recall)
            result = run_frontier_search(
                task=episode.hidden_task,
                policy=SchedulingMLPPolicy(episode.hidden_task, ranker),
                verifier=Level1SchedulingVerifier(),
                node_budget=node_budget,
                proposal_budget=action_budget,
                seed=seed + idx,
                proposer=proposer,
                certified_pruning=certified_pruning,
            )
            pruned = int(result.pruned_nodes)
            false_prunes = int(result.false_prune_count)
            row = {
                "Policy": name,
                "task_id": episode.hidden_task_id,
                "rendering_id": episode.rendering_id,
                "perturbation": episode.perturbation,
                "success": result.best_objective == result.optimum,
                "gap": _gap(result.best_objective, result.optimum),
                "expanded_nodes": result.expanded_nodes,
                "generated_children": result.generated_children,
                "proposal_calls": result.cost.proposal_calls,
                "verifier_calls": result.cost.verifier_calls,
                "solver_calls": result.cost.solver_calls,
                "certificate_count": result.certificate_count,
                "pruned_nodes": pruned,
                "certificate_rate": 0.0 if result.generated_children == 0 else result.certificate_count / result.generated_children,
                "false_prunes": false_prunes,
                "wrong_prune_rate": 0.0 if pruned == 0 else false_prunes / pruned,
                "proposal_tokens": proposer.cost.input_tokens + proposer.cost.output_tokens,
                "total_tokens": proposer.cost.input_tokens + proposer.cost.output_tokens,
                "wall_clock_seconds": result.cost.wall_clock_seconds,
            }
            by_setting[name].append(row)
            per_instance_rows.append(row)
    table_rows = []
    for name, rows in by_setting.items():
        pruned = sum(int(row["pruned_nodes"]) for row in rows)
        false_prunes = sum(int(row["false_prunes"]) for row in rows)
        generated = sum(int(row["generated_children"]) for row in rows)
        table_rows.append(
            {
                "Policy": name,
                "success": _mean(1.0 if row["success"] else 0.0 for row in rows),
                "gap": _mean(float(row["gap"]) for row in rows),
                "expanded_nodes_mean": _mean(float(row["expanded_nodes"]) for row in rows),
                "proposal_calls_mean": _mean(float(row["proposal_calls"]) for row in rows),
                "verifier_calls": sum(int(row["verifier_calls"]) for row in rows),
                "solver_calls": sum(int(row["solver_calls"]) for row in rows),
                "certificate_count": sum(int(row["certificate_count"]) for row in rows),
                "pruned_nodes": pruned,
                "certificate_rate": 0.0 if generated == 0 else pruned / generated,
                "false_prunes": false_prunes,
                "wrong_prune_rate": 0.0 if pruned == 0 else false_prunes / pruned,
                "total_tokens": sum(int(row["total_tokens"]) for row in rows),
                "wall_clock_seconds": sum(float(row["wall_clock_seconds"]) for row in rows),
            }
        )
    return table_rows, per_instance_rows


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
