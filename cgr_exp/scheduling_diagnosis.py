from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from cgr_exp.scheduling_micro import (
    SCHEDULING_FEATURE_NAMES,
    SchedulingTraceRecord,
    TinyPairwiseMLPRanker,
    collect_solver_trace_records,
    generate_scheduling_micro_instances,
    make_scheduling_ranking_pairs,
)
from cgr_exp.scheduling_transfer import collect_level2_trace_records, make_level2_scheduling_episodes


def run_scheduling_transfer_diagnosis(
    out_dir: Path,
    seed: int = 0,
    source_count: int = 200,
    source_validation_count: int = 50,
    target_count: int = 100,
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
    source = generate_scheduling_micro_instances(source_count, source_validation_count, 0, seed, job_count, machine_count)
    target = generate_scheduling_micro_instances(0, 0, target_count, seed + 10000, job_count, machine_count)
    source_records = []
    source_validation_records = []
    for task in source["train"]:
        source_records.extend(collect_solver_trace_records(task, "source_train", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    for task in source["validation"]:
        source_validation_records.extend(collect_solver_trace_records(task, "source_validation", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    target_episodes = make_level2_scheduling_episodes(target["test"], seed=seed + 4000, perturbation=perturbation)
    target_records = collect_level2_trace_records(
        target_episodes,
        seed=seed + 5000,
        node_budget=node_budget,
        action_budget=action_budget,
        max_records=records_per_instance,
        useful_action_recall=useful_action_recall,
    )
    source_pairs = make_scheduling_ranking_pairs(source_records, max_pairs_per_task=40, seed=seed)
    source_validation_pairs = make_scheduling_ranking_pairs(source_validation_records, max_pairs_per_task=40, seed=seed + 1)
    target_pairs = make_scheduling_ranking_pairs(target_records, max_pairs_per_task=40, seed=seed + 2)
    ranker = TinyPairwiseMLPRanker(len(SCHEDULING_FEATURE_NAMES), hidden_dim=16, seed=seed)
    ranker.train(source_pairs, epochs=rank_epochs, lr=0.02, seed=seed)
    feature_rows = _feature_distribution_rows(source_records, target_records)
    domain_metrics = _domain_classifier_metrics(source_records, target_records, seed=seed)
    ranking_metrics = {
        "source_validation_pairwise_accuracy": 1.0 - ranker.ranking_error(source_validation_pairs),
        "target_pairwise_accuracy": 1.0 - ranker.ranking_error(target_pairs),
        "target_selection_regret_mean": _selection_regret(target_records, ranker),
        "target_pair_count": len(target_pairs),
        "source_pair_count": len(source_pairs),
        "source_validation_pair_count": len(source_validation_pairs),
    }
    metrics = {
        "benchmark": "scheduling_transfer_diagnosis",
        "seed": seed,
        "source_count": source_count,
        "source_validation_count": source_validation_count,
        "target_count": target_count,
        "job_count": job_count,
        "machine_count": machine_count,
        "node_budget": node_budget,
        "action_budget": action_budget,
        "records_per_instance": records_per_instance,
        "rank_epochs": rank_epochs,
        "useful_action_recall": useful_action_recall,
        "perturbation": perturbation,
        "source_record_count": len(source_records),
        "target_record_count": len(target_records),
        "domain_classifier": domain_metrics,
        "ranking_transfer": ranking_metrics,
        "common_feature_names": list(SCHEDULING_FEATURE_NAMES),
        "artifacts": [
            "scheduling_transfer_feature_distribution.csv",
            "scheduling_transfer_diagnosis_metrics.json",
        ],
    }
    _write_csv(out_dir / "scheduling_transfer_feature_distribution.csv", feature_rows)
    (out_dir / "scheduling_transfer_diagnosis_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _feature_distribution_rows(source_records: Sequence[SchedulingTraceRecord], target_records: Sequence[SchedulingTraceRecord]) -> List[Dict[str, object]]:
    rows = []
    for idx, name in enumerate(SCHEDULING_FEATURE_NAMES):
        src = [record.features[idx] for record in source_records]
        tgt = [record.features[idx] for record in target_records]
        src_stats = _stats(src)
        tgt_stats = _stats(tgt)
        pooled = math.sqrt((src_stats["std"] ** 2 + tgt_stats["std"] ** 2) / 2.0)
        rows.append(
            {
                "feature": name,
                "source_mean": src_stats["mean"],
                "target_mean": tgt_stats["mean"],
                "mean_delta": tgt_stats["mean"] - src_stats["mean"],
                "standardized_delta": 0.0 if pooled == 0.0 else (tgt_stats["mean"] - src_stats["mean"]) / pooled,
                "source_std": src_stats["std"],
                "target_std": tgt_stats["std"],
                "source_q10": src_stats["q10"],
                "target_q10": tgt_stats["q10"],
                "source_q50": src_stats["q50"],
                "target_q50": tgt_stats["q50"],
                "source_q90": src_stats["q90"],
                "target_q90": tgt_stats["q90"],
                "source_missing_rate": src_stats["missing_rate"],
                "target_missing_rate": tgt_stats["missing_rate"],
            }
        )
    return rows


def _domain_classifier_metrics(source_records: Sequence[SchedulingTraceRecord], target_records: Sequence[SchedulingTraceRecord], seed: int) -> Dict[str, object]:
    examples = [(record.features, 0.0) for record in source_records] + [(record.features, 1.0) for record in target_records]
    rng = random.Random(seed)
    rng.shuffle(examples)
    split = max(1, int(0.7 * len(examples)))
    train = examples[:split]
    test = examples[split:]
    dim = len(SCHEDULING_FEATURE_NAMES)
    mean = [sum(x[i] for x, _ in train) / len(train) for i in range(dim)]
    scale = []
    for i in range(dim):
        var = sum((x[i] - mean[i]) ** 2 for x, _ in train) / len(train)
        scale.append(max(1e-6, math.sqrt(var)))
    weights = [0.0] * dim
    bias = 0.0
    for _ in range(60):
        rng.shuffle(train)
        for x, y in train:
            z = [(x[i] - mean[i]) / scale[i] for i in range(dim)]
            pred = _sigmoid(sum(w * v for w, v in zip(weights, z)) + bias)
            grad = pred - y
            for i in range(dim):
                weights[i] -= 0.03 * grad * z[i]
            bias -= 0.03 * grad
    correct = 0
    losses = []
    for x, y in test:
        z = [(x[i] - mean[i]) / scale[i] for i in range(dim)]
        pred = _sigmoid(sum(w * v for w, v in zip(weights, z)) + bias)
        correct += int((pred >= 0.5) == bool(y))
        losses.append(-(y * math.log(max(pred, 1e-9)) + (1.0 - y) * math.log(max(1.0 - pred, 1e-9))))
    top_weights = sorted(
        ({"feature": name, "weight": weight} for name, weight in zip(SCHEDULING_FEATURE_NAMES, weights)),
        key=lambda row: abs(float(row["weight"])),
        reverse=True,
    )[:8]
    return {
        "accuracy": 0.0 if not test else correct / len(test),
        "cross_entropy": _mean(losses),
        "train_examples": len(train),
        "test_examples": len(test),
        "top_feature_weights": top_weights,
    }


def _selection_regret(records: Sequence[SchedulingTraceRecord], ranker: TinyPairwiseMLPRanker) -> float:
    grouped: Dict[str, List[SchedulingTraceRecord]] = {}
    for record in records:
        grouped.setdefault(record.task_id, []).append(record)
    regrets = []
    for task_records in grouped.values():
        by_parent: Dict[str | None, List[SchedulingTraceRecord]] = {}
        for record in task_records:
            by_parent.setdefault(record.parent_id, []).append(record)
        for siblings in by_parent.values():
            if len(siblings) < 2:
                continue
            selected = max(siblings, key=lambda record: ranker.score(record.features))
            best = min(siblings, key=lambda record: record.best_descendant_objective)
            regrets.append(max(0.0, float(selected.best_descendant_objective) - float(best.best_descendant_objective)))
    return _mean(regrets)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    missing = len(values) - len(finite)
    if not finite:
        return {"mean": 0.0, "std": 0.0, "q10": 0.0, "q50": 0.0, "q90": 0.0, "missing_rate": 1.0}
    ordered = sorted(finite)
    mean = sum(ordered) / len(ordered)
    var = sum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "mean": mean,
        "std": math.sqrt(var),
        "q10": _quantile(ordered, 0.1),
        "q50": _quantile(ordered, 0.5),
        "q90": _quantile(ordered, 0.9),
        "missing_rate": missing / len(values),
    }


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return 0.0 if not data else sum(data) / len(data)
