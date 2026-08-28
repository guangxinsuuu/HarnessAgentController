from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from cgr_exp.datasets.level1_scheduling import ScheduleAction, SchedulingTask
from cgr_exp.proposers.llm_adapter import OpenAICompatibleHTTPClient
from cgr_exp.scheduling_micro import generate_scheduling_micro_instances
from cgr_exp.schemas import CompletionStatus
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier


NATIVE_METHODS = {
    "direct": {
        "table_method": "Direct / CoT",
        "training_states": "native direct reasoning",
        "sample_count": 1,
        "temperature": 0.0,
        "selection": "first_valid",
    },
    "self_consistency": {
        "table_method": "Self-Consistency",
        "training_states": "native parallel sampling",
        "sample_count": 8,
        "temperature": 0.7,
        "selection": "majority_job_order_then_objective",
    },
    "best_of_n": {
        "table_method": "Best-of-N",
        "training_states": "native matched-budget sampling",
        "sample_count": 8,
        "temperature": 0.7,
        "selection": "best_verified_objective",
    },
}


def run_scheduling_native_q2_from_env(
    out_dir: Path,
    model: Optional[str] = None,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    timeout: float = 120.0,
    methods: Sequence[str] = ("direct", "self_consistency", "best_of_n"),
    sample_count: int = 8,
    parallel_episodes: int = 1,
    resume: bool = False,
) -> Dict[str, object]:
    client = OpenAICompatibleHTTPClient.from_env()
    client.timeout = timeout
    return run_scheduling_native_q2(
        out_dir,
        client=client,
        model=model or client.model,
        seed=seed,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        job_count=job_count,
        machine_count=machine_count,
        methods=methods,
        sample_count=sample_count,
        parallel_episodes=parallel_episodes,
        resume=resume,
    )


def run_scheduling_native_q2(
    out_dir: Path,
    client,
    model: str,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    methods: Sequence[str] = ("direct", "self_consistency", "best_of_n"),
    sample_count: int = 8,
    parallel_episodes: int = 1,
    resume: bool = False,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_methods = _normalize_methods(methods)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    tasks = instances["test"]
    existing = _read_csv_rows(out_dir / "scheduling_native_q2_per_instance.csv") if resume else []
    completed = {(str(row.get("method_key")), str(row.get("task_id"))) for row in existing}
    jobs = []
    for method_key in selected_methods:
        for idx, task in enumerate(tasks):
            if (method_key, task.task_id) not in completed:
                jobs.append((method_key, idx, task))
    started = time.perf_counter()
    new_rows: List[Dict[str, object]] = []
    workers = max(1, int(parallel_episodes))
    if workers == 1:
        for done, item in enumerate(jobs, start=1):
            row = _run_one_method_task(client, model, item[0], item[1], item[2], sample_count)
            new_rows.append(row)
            print(f"[scheduling-native-q2] {done}/{len(jobs)} rows, method={row['Method']}, tokens={row['total_tokens']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one_method_task, client, model, method_key, idx, task, sample_count) for method_key, idx, task in jobs]
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                new_rows.append(row)
                print(f"[scheduling-native-q2] {done}/{len(jobs)} rows, method={row['Method']}, tokens={row['total_tokens']}", flush=True)
    all_rows = [*_coerce_rows(existing), *new_rows]
    all_rows.sort(key=lambda row: (str(row["method_key"]), int(row["task_index"])))
    table_rows = _summarize_rows(all_rows, selected_methods, expected_count=len(tasks), wall_clock_seconds=time.perf_counter() - started)
    _write_csv(out_dir / "scheduling_native_q2_table.csv", table_rows)
    _write_csv(out_dir / "scheduling_native_q2_per_instance.csv", all_rows)
    for row in table_rows:
        method_dir = out_dir / _method_dir_name(str(row["Method"]))
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "table5_native_row.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    manifest = {
        "benchmark": "scheduling_native_q2",
        "seed": seed,
        "model": model,
        "splits": {split: len(split_tasks) for split, split_tasks in instances.items()},
        "split_manifest": {split: [task.task_id for task in split_tasks] for split, split_tasks in instances.items()},
        "methods": selected_methods,
        "sample_count": sample_count,
        "parallel_episodes": workers,
        "resume": resume,
        "live_network_required": not client.__class__.__name__.startswith("Scripted"),
        "oracle_labels_in_inference_features": False,
        "native_protocol": True,
        "produced_artifacts": [
            "scheduling_native_q2_table.csv",
            "scheduling_native_q2_per_instance.csv",
            "scheduling_native_q2_manifest.json",
        ],
    }
    (out_dir / "scheduling_native_q2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"model": model, "table_rows": table_rows, "per_instance_rows": all_rows, **manifest}


def _run_one_method_task(client, model: str, method_key: str, task_index: int, task: SchedulingTask, default_sample_count: int) -> Dict[str, object]:
    spec = dict(NATIVE_METHODS[method_key])
    if method_key != "direct":
        spec["sample_count"] = max(1, int(default_sample_count))
    samples = []
    for sample_idx in range(int(spec["sample_count"])):
        response = client.generate_scheduling_solution(
            task=task,
            model=model,
            method=method_key,
            sample_index=sample_idx,
            temperature=float(spec["temperature"]),
            top_p=1.0,
            prompt_template_version="scheduling-native-q2-v1",
        )
        sample = _evaluate_sample(task, response, sample_idx)
        samples.append(sample)
    chosen = _select_sample(samples, str(spec["selection"]))
    optimum = task.exhaustive_oracle()
    best_objective = chosen.get("objective")
    success = best_objective == optimum
    return {
        "method_key": method_key,
        "Method": spec["table_method"],
        "Training states": spec["training_states"],
        "task_index": task_index,
        "task_id": task.task_id,
        "success": success,
        "gap": _gap(best_objective, optimum),
        "best_objective": "" if best_objective is None else best_objective,
        "optimum": optimum,
        "llm_calls": len(samples),
        "input_tokens": sum(int(sample["input_tokens"]) for sample in samples),
        "output_tokens": sum(int(sample["output_tokens"]) for sample in samples),
        "total_tokens": sum(int(sample["total_tokens"]) for sample in samples),
        "wall_clock_seconds": sum(float(sample["latency"]) for sample in samples),
        "valid_samples": sum(1 for sample in samples if sample["valid_complete"]),
        "sample_count": len(samples),
        "selection_rule": spec["selection"],
        "chosen_sample_index": chosen.get("sample_index", ""),
        "status": "complete",
        "source_policy": "native",
        "notes": "native end-to-end method; verifier selects/validates complete schedules without oracle future labels",
    }


def _evaluate_sample(task: SchedulingTask, response: Dict[str, object], sample_idx: int) -> Dict[str, object]:
    actions, parse_failures = _parse_actions(str(response.get("raw_output", "")))
    node = task.root_node()
    verifier = Level1SchedulingVerifier()
    valid_prefix = True
    violations: List[str] = []
    for child_idx, action in enumerate(actions):
        result = verifier.verify(task, node, action)
        if not result.feasible:
            valid_prefix = False
            violations.extend(result.violations)
            break
        node = verifier.make_child(task, node, action, result, child_idx)
    valid_complete = valid_prefix and node.completion_status == CompletionStatus.COMPLETE and len(actions) == len(task.durations)
    objective = task.objective(tuple(node.structured_state["scheduled"])) if valid_complete else None
    job_order = tuple(str(row["job"]) for row in node.structured_state.get("scheduled", [])) if valid_complete else ()
    return {
        "sample_index": sample_idx,
        "valid_complete": valid_complete,
        "objective": objective,
        "job_order": job_order,
        "parse_failures": parse_failures,
        "violations": violations,
        "input_tokens": int(response.get("input_tokens", 0)),
        "output_tokens": int(response.get("output_tokens", 0)),
        "total_tokens": int(response.get("input_tokens", 0)) + int(response.get("output_tokens", 0)),
        "latency": float(response.get("latency", 0.0)),
    }


def _parse_actions(raw: str) -> tuple[List[ScheduleAction], int]:
    parse_failures = 0
    actions: List[ScheduleAction] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            parsed = parsed.get("actions", [])
        if not isinstance(parsed, list):
            raise ValueError("expected actions list")
        for item in parsed:
            if not isinstance(item, dict):
                parse_failures += 1
                continue
            if item.get("action_type", "schedule") != "schedule":
                parse_failures += 1
                continue
            actions.append(ScheduleAction(job=item.get("job", ""), machine=item.get("machine", "")))
    except Exception:
        parse_failures += 1
    return actions, parse_failures


def _select_sample(samples: Sequence[Dict[str, object]], selection: str) -> Dict[str, object]:
    valid = [sample for sample in samples if sample["valid_complete"]]
    if not valid:
        return dict(samples[0]) if samples else {"sample_index": ""}
    if selection == "best_verified_objective":
        return min(valid, key=lambda sample: (float(sample["objective"]), int(sample["sample_index"])))
    if selection == "majority_job_order_then_objective":
        counts: Dict[tuple[str, ...], int] = {}
        for sample in valid:
            key = tuple(sample["job_order"])
            counts[key] = counts.get(key, 0) + 1
        return min(valid, key=lambda sample: (-counts[tuple(sample["job_order"])], float(sample["objective"]), int(sample["sample_index"])))
    return min(valid, key=lambda sample: int(sample["sample_index"]))


def _summarize_rows(rows: Sequence[Dict[str, object]], method_keys: Sequence[str], expected_count: int, wall_clock_seconds: float) -> List[Dict[str, object]]:
    out = []
    for method_key in method_keys:
        spec = NATIVE_METHODS[method_key]
        method_rows = [row for row in rows if row["method_key"] == method_key]
        complete = len(method_rows) >= expected_count
        out.append(
            {
                "Method": spec["table_method"],
                "Training states": spec["training_states"],
                "success": _mean(1.0 if _as_bool(row["success"]) else 0.0 for row in method_rows),
                "gap": _mean(float(row["gap"]) for row in method_rows),
                "llm_calls": sum(int(row["llm_calls"]) for row in method_rows),
                "tokens": sum(int(row["total_tokens"]) for row in method_rows),
                "wall_clock_seconds": wall_clock_seconds,
                "status": "complete" if complete else f"partial: {len(method_rows)}/{expected_count}",
                "source_policy": "native",
                "notes": f"{spec['selection']}; {len(method_rows)}/{expected_count} completed",
            }
        )
    return out


def _normalize_methods(methods: Sequence[str]) -> List[str]:
    selected = []
    for method in methods:
        key = method.strip().lower().replace("-", "_")
        if not key:
            continue
        if key not in NATIVE_METHODS:
            raise ValueError(f"unknown native Q2 method: {method}")
        selected.append(key)
    return selected or ["direct"]


def _method_dir_name(method: str) -> str:
    return method.lower().replace(" / ", "_").replace("-", "_").replace(" ", "_")


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
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


def _coerce_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        cur = dict(row)
        for field in ("task_index", "llm_calls", "input_tokens", "output_tokens", "total_tokens", "valid_samples", "sample_count"):
            if field in cur:
                cur[field] = int(float(cur[field]))
        for field in ("gap", "wall_clock_seconds", "optimum"):
            if field in cur and cur[field] != "":
                cur[field] = float(cur[field])
        if cur.get("best_objective") != "":
            cur["best_objective"] = float(cur["best_objective"])
        cur["success"] = _as_bool(cur.get("success"))
        out.append(cur)
    return out


def _gap(best_objective: object, optimum: float) -> float:
    if best_objective is None or best_objective == "":
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
