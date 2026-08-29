from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from cgr_exp.scheduling_transfer import preflight_scheduling_llm_pilot


NATIVE_METHODS = {
    "direct": ("Direct / CoT", "first_valid"),
    "self_consistency": ("Self-Consistency", "majority_job_order_then_first_valid"),
    "best_of_n": ("Best-of-N", "best_verified_objective"),
}


def build_q2_run_preflight(
    out_dir: Path,
    *,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 80,
    action_budget: int = 6,
    records_per_instance: int = 60,
    rank_epochs: int = 25,
    model: str | None = None,
    perturbation: str = "mixed",
    native_dir: Path = Path("runs/q2_native_llm_baselines_100_seed0_sc_v2"),
    learned_dir: Path = Path("runs/q2_learned_search_100_seed0_rest_gate_v2"),
    tot_dir: Path = Path("runs/q2_tot_bfs_100_seed0_tot_gate_v2"),
    sample_count: int = 8,
    native_parallel_episodes: int = 8,
    search_parallel_episodes: int = 6,
    tot_parallel_episodes: int = 6,
    batch_size: int = 4,
    target_prm_train_count: int = 100,
    tot_breadth: int = 5,
    tot_value_samples: int = 3,
    env=None,
) -> Dict[str, object]:
    source = os.environ if env is None else env
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = model or source.get("CGR_LLM_MODEL", "gpt-4o-mini")
    rows = [
        _learned_row(
            learned_dir,
            seed=seed,
            train_count=train_count,
            validation_count=validation_count,
            test_count=test_count,
            job_count=job_count,
            machine_count=machine_count,
            node_budget=node_budget,
            action_budget=action_budget,
            model=model_name,
            perturbation=perturbation,
            batch_size=batch_size,
            parallel_episodes=search_parallel_episodes,
            target_prm_train_count=target_prm_train_count,
            env=source,
        ),
        _native_row(
            native_dir,
            test_count=test_count,
            sample_count=sample_count,
            parallel_episodes=native_parallel_episodes,
        ),
        _tot_row(
            tot_dir,
            test_count=test_count,
            node_budget=node_budget,
            breadth=tot_breadth,
            value_samples=tot_value_samples,
            parallel_episodes=tot_parallel_episodes,
        ),
    ]
    csv_path = out_dir / "q2_run_preflight.csv"
    json_path = out_dir / "q2_run_preflight.json"
    md_path = out_dir / "q2_run_preflight.md"
    _write_csv(csv_path, rows)
    manifest = {
        "artifact": "q2_run_preflight",
        "seed": seed,
        "model": model_name,
        "api_key_env": "CGR_LLM_API_KEY",
        "api_key_present": bool(source.get("CGR_LLM_API_KEY")),
        "test_count": test_count,
        "training_config": {
            "train_count": train_count,
            "validation_count": validation_count,
            "records_per_instance": records_per_instance,
            "rank_epochs": rank_epochs,
        },
        "rows": rows,
        "total_remaining_units": sum(int(row["remaining_units"]) for row in rows),
        "total_worst_case_requests": sum(int(row["worst_case_request_count_upper_bound"]) for row in rows),
        "resume_safe": all(bool(row["resume_safe"]) for row in rows),
        "produced_artifacts": [csv_path.name, json_path.name, md_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_render_markdown(manifest), encoding="utf-8")
    return manifest


def _learned_row(
    learned_dir: Path,
    *,
    seed: int,
    train_count: int,
    validation_count: int,
    test_count: int,
    job_count: int,
    machine_count: int,
    node_budget: int,
    action_budget: int,
    model: str,
    perturbation: str,
    batch_size: int,
    parallel_episodes: int,
    target_prm_train_count: int,
    env,
) -> Dict[str, object]:
    preflight = preflight_scheduling_llm_pilot(
        learned_dir,
        include_target_prm=True,
        include_rest_mcts=True,
        include_solver_certification=True,
        seed=seed,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        job_count=job_count,
        machine_count=machine_count,
        node_budget=node_budget,
        action_budget=action_budget,
        model=model,
        perturbation=perturbation,
        batch_size=batch_size,
        parallel_episodes=parallel_episodes,
        target_prm_train_count=target_prm_train_count,
        env=env,
    )
    return {
        "run_group": "learned_search",
        "output_dir": str(learned_dir),
        "completed_units": preflight["completed_episode_count"],
        "remaining_units": preflight["remaining_episode_count"],
        "unit": "episode",
        "method_count": preflight["method_count"],
        "worst_case_request_count_upper_bound": preflight["worst_case_request_count_upper_bound"],
        "worst_case_token_upper_bound": preflight["estimated_prompt_tokens_upper_bound"],
        "parallel_episodes": preflight["parallel_episodes"],
        "resume_safe": preflight["resume_safe"],
        "notes": "run-scheduling-llm-pilot preflight with Target-PRM, ReST-MCTS-style, solver prior, and certification rows",
    }


def _native_row(
    native_dir: Path,
    *,
    test_count: int,
    sample_count: int,
    parallel_episodes: int,
) -> Dict[str, object]:
    rows = _read_csv(native_dir / "scheduling_native_q2_per_instance.csv")
    samples = _read_jsonl(native_dir / "scheduling_native_q2_samples.jsonl")
    completed = _completed_native_tasks(rows, samples, sample_count)
    requested = [(method_key, task_index) for method_key in NATIVE_METHODS for task_index in range(test_count)]
    remaining = [item for item in requested if item not in completed]
    return {
        "run_group": "native_frozen",
        "output_dir": str(native_dir),
        "completed_units": len(completed),
        "remaining_units": len(remaining),
        "unit": "method_task",
        "method_count": len(NATIVE_METHODS),
        "worst_case_request_count_upper_bound": _native_request_count(remaining, sample_count),
        "worst_case_token_upper_bound": "",
        "parallel_episodes": max(1, int(parallel_episodes)),
        "resume_safe": True,
        "notes": "Direct uses one sample; Self-Consistency and Best-of-N use sample_count complete-schedule samples",
    }


def _tot_row(
    tot_dir: Path,
    *,
    test_count: int,
    node_budget: int,
    breadth: int,
    value_samples: int,
    parallel_episodes: int,
) -> Dict[str, object]:
    rows = _read_csv(tot_dir / "scheduling_tot_q2_per_instance.csv")
    trace = _read_jsonl(tot_dir / "scheduling_tot_q2_trace.jsonl")
    completed = _completed_tot_renderings(rows, trace)
    remaining = max(0, int(test_count) - len(completed))
    return {
        "run_group": "native_tot",
        "output_dir": str(tot_dir),
        "completed_units": len(completed),
        "remaining_units": remaining,
        "unit": "episode",
        "method_count": 1,
        "worst_case_request_count_upper_bound": remaining * int(node_budget) * (1 + int(value_samples)),
        "worst_case_token_upper_bound": "",
        "parallel_episodes": max(1, int(parallel_episodes)),
        "resume_safe": True,
        "notes": f"strict ToT BFS; breadth={breadth}; value_samples={value_samples}; each expanded layer uses proposal plus value/ranking calls",
    }


def _completed_native_tasks(
    rows: Sequence[Mapping[str, object]],
    samples: Sequence[Mapping[str, object]],
    sample_count: int,
) -> set[tuple[str, int]]:
    sample_counts: Dict[tuple[str, str], int] = {}
    for sample in samples:
        key = (str(sample.get("method_key")), str(sample.get("task_id")))
        sample_counts[key] = sample_counts.get(key, 0) + 1
    completed = set()
    for row in rows:
        method_key = str(row.get("method_key", ""))
        if method_key not in NATIVE_METHODS:
            continue
        expected_selection = NATIVE_METHODS[method_key][1]
        task_id = str(row.get("task_id", ""))
        task_index = _to_int(row.get("task_index"), default=-1)
        expected_samples = 1 if method_key == "direct" else max(1, int(sample_count))
        if (
            task_index >= 0
            and str(row.get("selection_rule", "")) == expected_selection
            and sample_counts.get((method_key, task_id), 0) >= expected_samples
        ):
            completed.add((method_key, task_index))
    return completed


def _native_request_count(remaining: Sequence[tuple[str, int]], sample_count: int) -> int:
    total = 0
    for method_key, _task_index in remaining:
        total += 1 if method_key == "direct" else max(1, int(sample_count))
    return total


def _completed_tot_renderings(rows: Sequence[Mapping[str, object]], trace: Sequence[Mapping[str, object]]) -> set[str]:
    row_renderings = {str(row.get("rendering_id")) for row in rows if row.get("rendering_id") not in (None, "")}
    finished = {str(row.get("rendering_id")) for row in trace if str(row.get("event")) == "finish"}
    return row_renderings & finished


def _render_markdown(manifest: Mapping[str, object]) -> str:
    lines = [
        "# Q2 Run Preflight",
        "",
        f"Model: `{manifest['model']}`",
        f"API key present: `{manifest['api_key_present']}`",
        f"Test count: `{manifest['test_count']}`",
        f"Total remaining units: `{manifest['total_remaining_units']}`",
        f"Total worst-case requests: `{manifest['total_worst_case_requests']}`",
        "",
        "| Run group | Completed | Remaining | Worst-case requests | Parallel episodes | Notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in manifest["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["run_group"]),
                    str(row["completed_units"]),
                    str(row["remaining_units"]),
                    str(row["worst_case_request_count_upper_bound"]),
                    str(row["parallel_episodes"]),
                    str(row["notes"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _to_int(value: object, *, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default
