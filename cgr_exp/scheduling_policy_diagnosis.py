from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from cgr_exp.datasets.level1_scheduling import ScheduleAction, SchedulingTask
from cgr_exp.scheduling_micro import TinyPairwiseMLPRanker, generate_scheduling_micro_instances, scheduling_feature_vector


def run_scheduling_policy_diagnosis(
    run_dir: Path,
    out_dir: Path,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    regret_threshold: float = 5.0,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {
        task.task_id: task
        for task in generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)["test"]
    }
    solver_ranker = TinyPairwiseMLPRanker.from_json((run_dir / "scheduling_llm_pilot_ranker.json").read_text(encoding="utf-8"))
    target_ranker = TinyPairwiseMLPRanker.from_json((run_dir / "target_prm_ranker.json").read_text(encoding="utf-8"))
    per_decision_rows = _decision_rows(run_dir / "proposal_cache.jsonl", tasks, {"Target-PRM": target_ranker, "Solver-trained MLP": solver_ranker}, regret_threshold)
    summary_rows = _summary_rows(per_decision_rows)
    high_impact_rows = _summary_rows([row for row in per_decision_rows if _as_bool(row["high_impact"])], suffix=" high-impact")
    failure_rows = _failure_conditioned_rows(run_dir / "scheduling_llm_pilot_per_instance.csv", per_decision_rows)
    _write_csv(out_dir / "policy_diagnosis_decisions.csv", per_decision_rows)
    _write_csv(out_dir / "policy_diagnosis_summary.csv", summary_rows)
    _write_csv(out_dir / "policy_diagnosis_high_impact.csv", high_impact_rows)
    _write_csv(out_dir / "policy_diagnosis_failure_conditioned.csv", failure_rows)
    manifest = {
        "benchmark": "scheduling_policy_diagnosis",
        "source_run_dir": str(run_dir),
        "seed": seed,
        "splits": {"train": train_count, "validation": validation_count, "test": test_count},
        "regret_threshold": regret_threshold,
        "decision_count": len(per_decision_rows),
        "high_impact_decision_count": sum(1 for row in per_decision_rows if _as_bool(row["high_impact"])),
        "apples_to_apples_basis": "same cached parent states and same cached GPT action candidates for both rankers",
        "notes": "diagnoses local child-candidate choices in the shared proposal cache; it is not the same as policy-specific frontier distributions",
        "artifacts": [
            "policy_diagnosis_decisions.csv",
            "policy_diagnosis_summary.csv",
            "policy_diagnosis_high_impact.csv",
            "policy_diagnosis_failure_conditioned.csv",
            "policy_diagnosis_manifest.json",
        ],
    }
    (out_dir / "policy_diagnosis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "summary": summary_rows, "high_impact": high_impact_rows, "failure_conditioned": failure_rows}


def _decision_rows(cache_path: Path, tasks: Dict[str, SchedulingTask], rankers: Dict[str, TinyPairwiseMLPRanker], regret_threshold: float) -> List[Dict[str, object]]:
    rows = []
    with cache_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            cached = json.loads(line)
            task = tasks.get(str(cached.get("instance_id")))
            if task is None:
                continue
            parent = _node_from_cached_state(task, cached)
            actions = [ScheduleAction(job=str(item.get("job", "")), machine=str(item.get("machine", ""))) for item in cached.get("actions", []) if isinstance(item, dict)]
            children = []
            for idx, action in enumerate(actions):
                if task.is_valid_action(tuple(parent.structured_state.get("scheduled", [])), action):
                    child = task.apply(parent, action, child_index=idx)
                    if child.feasible_status.value == "feasible":
                        children.append(child)
            if len(children) < 2:
                continue
            oracle_values = [float(task.best_descendant_objective(child)) for child in children]
            best_value = min(oracle_values)
            value_spread = max(oracle_values) - best_value
            for policy, ranker in rankers.items():
                scores = [float(ranker.score(scheduling_feature_vector(task, child))) for child in children]
                selected_idx = max(range(len(children)), key=lambda i: (scores[i], children[i].node_id))
                regret = max(0.0, oracle_values[selected_idx] - best_value)
                pair_total = 0
                pair_correct = 0
                for i in range(len(children)):
                    for j in range(i + 1, len(children)):
                        if oracle_values[i] == oracle_values[j]:
                            continue
                        pair_total += 1
                        better, worse = (i, j) if oracle_values[i] < oracle_values[j] else (j, i)
                        pair_correct += int(scores[better] > scores[worse])
                rows.append(
                    {
                        "Policy": policy,
                        "task_id": task.task_id,
                        "cache_key": cached.get("cache_key", ""),
                        "depth": parent.depth,
                        "candidate_count": len(children),
                        "oracle_best_value": best_value,
                        "oracle_value_spread": value_spread,
                        "selected_value": oracle_values[selected_idx],
                        "selection_regret": regret,
                        "top1": oracle_values[selected_idx] == best_value,
                        "pairwise_accuracy": 1.0 if pair_total == 0 else pair_correct / pair_total,
                        "pair_count": pair_total,
                        "catastrophic_regret": regret > regret_threshold,
                        "high_impact": value_spread > regret_threshold,
                    }
                )
    return rows


def _node_from_cached_state(task: SchedulingTask, cached: Dict[str, object]):
    state = cached.get("canonical_state", {})
    if not isinstance(state, dict):
        state = {}
    scheduled = tuple(
        {
            "job": str(row.get("job")),
            "machine": str(row.get("machine")),
            "start": int(row.get("start", 0)),
            "finish": int(row.get("finish", 0)),
        }
        for row in state.get("observable_history", [])
        if isinstance(row, dict) and row.get("job") is not None and row.get("machine") is not None
    )
    return task._make_node(scheduled=scheduled, parent_id=None, action=None, node_id=f"{task.task_id}:diag:{cached.get('cache_key', '')}", incumbent=None)


def _summary_rows(rows: Sequence[Dict[str, object]], suffix: str = "") -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["Policy"]), []).append(row)
    out = []
    for policy, data in sorted(grouped.items()):
        regrets = sorted(float(row["selection_regret"]) for row in data)
        out.append(
            {
                "Policy": policy + suffix,
                "decisions": len(data),
                "top1": _mean(1.0 if _as_bool(row["top1"]) else 0.0 for row in data),
                "pairwise_accuracy": _weighted_mean(data, "pairwise_accuracy", "pair_count"),
                "regret_mean": _mean(regrets),
                "regret_p90": _percentile(regrets, 0.90),
                "regret_p95": _percentile(regrets, 0.95),
                "regret_p99": _percentile(regrets, 0.99),
                "regret_max": max(regrets) if regrets else 0.0,
                "catastrophic_regret_rate": _mean(1.0 if _as_bool(row["catastrophic_regret"]) else 0.0 for row in data),
                "high_impact_rate": _mean(1.0 if _as_bool(row["high_impact"]) else 0.0 for row in data),
            }
        )
    return out


def _failure_conditioned_rows(per_instance_path: Path, decision_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    per_instance = _read_csv(per_instance_path)
    by_task: Dict[str, Dict[str, Dict[str, str]]] = {}
    for row in per_instance:
        by_task.setdefault(str(row["task_id"]), {})[str(row["Policy"])] = row
    groups = {
        "both succeed": [],
        "solver only succeeds": [],
        "target only succeeds": [],
        "both fail": [],
    }
    for task_id, rows in by_task.items():
        if "Target-PRM" not in rows or "Solver-trained MLP" not in rows:
            continue
        target = _as_bool(rows["Target-PRM"].get("success"))
        solver = _as_bool(rows["Solver-trained MLP"].get("success"))
        if target and solver:
            groups["both succeed"].append(task_id)
        elif solver and not target:
            groups["solver only succeeds"].append(task_id)
        elif target and not solver:
            groups["target only succeeds"].append(task_id)
        else:
            groups["both fail"].append(task_id)
    by_policy_task: Dict[tuple[str, str], List[Dict[str, object]]] = {}
    for row in decision_rows:
        by_policy_task.setdefault((str(row["Policy"]), str(row["task_id"])), []).append(row)
    out = []
    for group, task_ids in groups.items():
        task_set = set(task_ids)
        for policy in ("Target-PRM", "Solver-trained MLP"):
            data = [row for (row_policy, task_id), policy_rows in by_policy_task.items() if row_policy == policy and task_id in task_set for row in policy_rows]
            regrets = sorted(float(row["selection_regret"]) for row in data)
            out.append(
                {
                    "group": group,
                    "Policy": policy,
                    "task_count": len(task_ids),
                    "decision_count": len(data),
                    "top1": _mean(1.0 if _as_bool(row["top1"]) else 0.0 for row in data),
                    "pairwise_accuracy": _weighted_mean(data, "pairwise_accuracy", "pair_count"),
                    "regret_mean": _mean(regrets),
                    "regret_p95": _percentile(regrets, 0.95),
                    "regret_max": max(regrets) if regrets else 0.0,
                    "catastrophic_regret_rate": _mean(1.0 if _as_bool(row["catastrophic_regret"]) else 0.0 for row in data),
                }
            )
    return out


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _weighted_mean(rows: Sequence[Dict[str, object]], value_field: str, weight_field: str) -> float:
    total = sum(float(row.get(weight_field, 0.0)) for row in rows)
    if total == 0:
        return 0.0
    return sum(float(row.get(value_field, 0.0)) * float(row.get(weight_field, 0.0)) for row in rows) / total


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, max(0, int(math.ceil(q * len(sorted_values)) - 1)))
    return float(sorted_values[idx])


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return 0.0 if not data else sum(data) / len(data)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)
