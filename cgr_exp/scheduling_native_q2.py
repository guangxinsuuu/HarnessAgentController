from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from cgr_exp.datasets.level1_scheduling import ScheduleAction, SchedulingTask
from cgr_exp.proposers.llm_adapter import OpenAICompatibleHTTPClient
from cgr_exp.q2_method_contracts import q2_method_contracts
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
        "official_primitive": "one chain-of-thought reasoning path to a final answer",
    },
    "self_consistency": {
        "table_method": "Self-Consistency",
        "training_states": "native parallel sampling",
        "sample_count": 8,
        "temperature": 0.7,
        "selection": "majority_job_order_then_first_valid",
        "official_primitive": "sample multiple reasoning paths and aggregate final answers",
    },
    "best_of_n": {
        "table_method": "Best-of-N",
        "training_states": "native matched-budget sampling",
        "sample_count": 8,
        "temperature": 0.7,
        "selection": "best_verified_objective",
        "official_primitive": "sample multiple final answers and select with a verifier or reward",
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
    per_instance_path = out_dir / "scheduling_native_q2_per_instance.csv"
    sample_path = out_dir / "scheduling_native_q2_samples.jsonl"
    existing = _read_csv_rows(per_instance_path) if resume else []
    existing_samples = _read_jsonl_rows(sample_path) if resume else []
    if not resume and per_instance_path.exists():
        per_instance_path.unlink()
    if not resume and sample_path.exists():
        sample_path.unlink()
    completed = _completed_native_method_tasks(existing, existing_samples, selected_methods, sample_count)
    if resume:
        selected_method_set = set(selected_methods)
        existing = [
            row
            for row in existing
            if str(row.get("method_key")) not in selected_method_set
            or (str(row.get("method_key")), str(row.get("task_id"))) in completed
        ]
        existing_samples = [
            row
            for row in existing_samples
            if str(row.get("method_key")) not in selected_method_set
            or (str(row.get("method_key")), str(row.get("task_id"))) in completed
        ]
        if existing_samples:
            _write_jsonl(sample_path, existing_samples)
        elif sample_path.exists():
            sample_path.write_text("", encoding="utf-8")
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
            row, sample_rows = _run_one_method_task(client, model, item[0], item[1], item[2], sample_count)
            new_rows.append(row)
            _append_csv_row(per_instance_path, row)
            for sample_row in sample_rows:
                _append_jsonl_row(sample_path, sample_row)
            print(f"[scheduling-native-q2] {done}/{len(jobs)} rows, method={row['Method']}, tokens={row['total_tokens']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_run_one_method_task, client, model, method_key, idx, task, sample_count) for method_key, idx, task in jobs]
            for done, future in enumerate(as_completed(futures), start=1):
                row, sample_rows = future.result()
                new_rows.append(row)
                _append_csv_row(per_instance_path, row)
                for sample_row in sample_rows:
                    _append_jsonl_row(sample_path, sample_row)
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
        "native_protocol": _native_protocol_manifest(selected_methods, sample_count),
        "method_alignment": q2_method_contracts([NATIVE_METHODS[key]["table_method"] for key in selected_methods]),
        "produced_artifacts": [
            "scheduling_native_q2_table.csv",
            "scheduling_native_q2_per_instance.csv",
            "scheduling_native_q2_samples.jsonl",
            "scheduling_native_q2_manifest.json",
        ],
    }
    (out_dir / "scheduling_native_q2_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"model": model, "table_rows": table_rows, "per_instance_rows": all_rows, **manifest}


def _completed_native_method_tasks(
    rows: Sequence[Dict[str, object]],
    sample_rows: Sequence[Dict[str, object]],
    selected_methods: Sequence[str],
    sample_count: int,
) -> set[tuple[str, str]]:
    selected = set(selected_methods)
    sample_counts: Dict[tuple[str, str], int] = {}
    for row in sample_rows:
        key = (str(row.get("method_key")), str(row.get("task_id")))
        sample_counts[key] = sample_counts.get(key, 0) + 1
    completed = set()
    for row in rows:
        method_key = str(row.get("method_key"))
        if method_key not in selected:
            continue
        task_id = str(row.get("task_id"))
        key = (method_key, task_id)
        expected_selection = str(NATIVE_METHODS[method_key]["selection"])
        if str(row.get("selection_rule", "")) == expected_selection and sample_counts.get(key, 0) >= _expected_native_sample_count(method_key, sample_count):
            completed.add(key)
    return completed


def _expected_native_sample_count(method_key: str, sample_count: int) -> int:
    return 1 if method_key == "direct" else max(1, int(sample_count))


def _native_protocol_manifest(method_keys: Sequence[str], sample_count: int) -> Dict[str, object]:
    protocols = {}
    for method_key in method_keys:
        spec = NATIVE_METHODS[method_key]
        effective_sample_count = 1 if method_key == "direct" else max(1, int(sample_count))
        protocols[spec["table_method"]] = {
            "official_primitive": spec["official_primitive"],
            "prompt_template_version": "scheduling-native-q2-v1",
            "sample_count": effective_sample_count,
            "temperature": spec["temperature"],
            "selection_rule": spec["selection"],
            "verifier_role": (
                "validate complete schedules for metrics"
                if method_key != "best_of_n"
                else "filter invalid complete schedules and select minimum verified objective"
            ),
            "oracle_future_labels_exposed": False,
        }
    return {
        "task_adaptation": "complete ScheduleAction JSON sequence for a SchedulingTask",
        "methods": protocols,
        "sample_audit_artifact": "scheduling_native_q2_samples.jsonl",
    }


def _run_one_method_task(client, model: str, method_key: str, task_index: int, task: SchedulingTask, default_sample_count: int) -> tuple[Dict[str, object], List[Dict[str, object]]]:
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
    row = {
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
        "parse_failures": sum(int(sample["parse_failures"]) for sample in samples),
        "sample_count": len(samples),
        "selection_rule": spec["selection"],
        "chosen_sample_index": chosen.get("sample_index", ""),
        "status": "complete",
        "source_policy": "native",
        "notes": "native end-to-end method; verifier selects/validates complete schedules without oracle future labels",
    }
    sample_rows = [_sample_audit_row(task, task_index, method_key, spec["table_method"], sample) for sample in samples]
    return row, sample_rows


def _evaluate_sample(task: SchedulingTask, response: Dict[str, object], sample_idx: int) -> Dict[str, object]:
    raw_output = str(response.get("raw_output", ""))
    actions, parse_failures = _parse_actions(raw_output)
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
        "parsed_action_count": len(actions),
        "violations": violations,
        "raw_output": raw_output,
        "input_tokens": int(response.get("input_tokens", 0)),
        "output_tokens": int(response.get("output_tokens", 0)),
        "total_tokens": int(response.get("input_tokens", 0)) + int(response.get("output_tokens", 0)),
        "latency": float(response.get("latency", 0.0)),
    }


def _sample_audit_row(task: SchedulingTask, task_index: int, method_key: str, method: str, sample: Dict[str, object]) -> Dict[str, object]:
    return {
        "method_key": method_key,
        "Method": method,
        "task_index": task_index,
        "task_id": task.task_id,
        "sample_index": sample["sample_index"],
        "valid_complete": sample["valid_complete"],
        "objective": sample["objective"],
        "job_order": list(sample["job_order"]),
        "parse_failures": sample["parse_failures"],
        "parsed_action_count": sample["parsed_action_count"],
        "violations": sample["violations"],
        "input_tokens": sample["input_tokens"],
        "output_tokens": sample["output_tokens"],
        "total_tokens": sample["total_tokens"],
        "latency": sample["latency"],
        "raw_output": sample["raw_output"],
    }


def _parse_actions(raw: str) -> tuple[List[ScheduleAction], int]:
    parse_failures = 0
    actions: List[ScheduleAction] = []
    try:
        parsed = json.loads(_extract_json_payload(raw))
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


def _extract_json_payload(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            text = match.group(1).strip()
    if text and text[0] not in "[{":
        starts = [idx for idx in (text.find("{"), text.find("[")) if idx >= 0]
        if starts:
            text = text[min(starts) :]
    return text


def _select_sample(samples: Sequence[Dict[str, object]], selection: str) -> Dict[str, object]:
    valid = [sample for sample in samples if sample["valid_complete"]]
    if not valid:
        return dict(samples[0]) if samples else {"sample_index": ""}
    if selection == "best_verified_objective":
        return min(valid, key=lambda sample: (float(sample["objective"]), int(sample["sample_index"])))
    if selection == "majority_job_order_then_first_valid":
        counts: Dict[tuple[str, ...], int] = {}
        for sample in valid:
            key = tuple(sample["job_order"])
            counts[key] = counts.get(key, 0) + 1
        return min(valid, key=lambda sample: (-counts[tuple(sample["job_order"])], int(sample["sample_index"])))
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
                "valid_sample_rate": _mean(float(row["valid_samples"]) / max(1.0, float(row["sample_count"])) for row in method_rows),
                "parse_failures": sum(int(row.get("parse_failures", 0)) for row in method_rows),
                "valid_samples": sum(int(row["valid_samples"]) for row in method_rows),
                "total_samples": sum(int(row["sample_count"]) for row in method_rows),
                "native_runtime_sample_count": sum(int(row["sample_count"]) for row in method_rows),
                "native_runtime_selection_events": len(method_rows),
                "native_runtime_valid_samples": sum(int(row["valid_samples"]) for row in method_rows),
                "native_runtime_parse_failures": sum(int(row.get("parse_failures", 0)) for row in method_rows),
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


def _append_csv_row(path: Path, row: Dict[str, object]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_jsonl_row(path: Path, row: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _coerce_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        cur = dict(row)
        cur.setdefault("parse_failures", 0)
        for field in ("task_index", "llm_calls", "input_tokens", "output_tokens", "total_tokens", "valid_samples", "sample_count", "parse_failures"):
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
