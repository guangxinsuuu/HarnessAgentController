from __future__ import annotations

from typing import Dict, List, Sequence


def build_transfer_training_plan(
    solver_trace_ids: Sequence[str],
    target_train_trace_ids: Sequence[str],
    target_eval_trace_ids: Sequence[str],
    solver_pair_count: int,
    target_train_pair_count: int,
    target_eval_pair_count: int,
    fewshot_counts: Sequence[int],
    rank_epochs: int,
) -> List[Dict[str, object]]:
    _validate_no_target_leakage(target_train_trace_ids, target_eval_trace_ids)
    rows: List[Dict[str, object]] = [
        {
            "stage": "zero_shot",
            "pair_count": solver_pair_count,
            "epochs": rank_epochs,
            "source_trace_ids": list(solver_trace_ids),
            "target_trace_ids": [],
            "target_train_trace_ids": [],
            "target_eval_trace_ids": list(target_eval_trace_ids),
            "solver_training_pair_count": solver_pair_count,
            "target_training_pair_count": 0,
            "target_evaluation_pair_count": target_eval_pair_count,
        }
    ]
    for n in fewshot_counts:
        count = min(n, target_train_pair_count)
        trace_count = min(n, len(target_train_trace_ids))
        rows.append(
            {
                "stage": f"few_shot_{n}",
                "pair_count": count,
                "epochs": max(2, rank_epochs // 2),
                "source_trace_ids": list(solver_trace_ids),
                "target_trace_ids": list(target_train_trace_ids[:trace_count]),
                "target_train_trace_ids": list(target_train_trace_ids[:trace_count]),
                "target_eval_trace_ids": list(target_eval_trace_ids),
                "solver_training_pair_count": solver_pair_count,
                "target_training_pair_count": count,
                "target_evaluation_pair_count": target_eval_pair_count,
            }
        )
    rows.append(
        {
            "stage": "llm_only",
            "pair_count": target_train_pair_count,
            "epochs": rank_epochs,
            "source_trace_ids": [],
            "target_trace_ids": list(target_train_trace_ids),
            "target_train_trace_ids": list(target_train_trace_ids),
            "target_eval_trace_ids": list(target_eval_trace_ids),
            "solver_training_pair_count": 0,
            "target_training_pair_count": target_train_pair_count,
            "target_evaluation_pair_count": target_eval_pair_count,
        }
    )
    return rows


def _validate_no_target_leakage(target_train_trace_ids: Sequence[str], target_eval_trace_ids: Sequence[str]) -> None:
    overlap = set(target_train_trace_ids).intersection(target_eval_trace_ids)
    if overlap:
        raise ValueError(f"target train/eval trace leakage: {sorted(overlap)[:3]}")
