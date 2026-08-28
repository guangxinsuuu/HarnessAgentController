from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from cgr_exp.datasets.level1_scheduling import SchedulingTask
from cgr_exp.scheduling_micro import generate_scheduling_micro_instances


def run_scheduling_structured_q2(
    out_dir: Path,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    cpsat_time_limit_seconds: float = 5.0,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    tasks = instances["test"]
    per_instance_rows: List[Dict[str, object]] = []
    for idx, task in enumerate(tasks):
        optimum = task.exhaustive_oracle()
        per_instance_rows.append(_cpsat_row(task, idx, optimum, cpsat_time_limit_seconds))
        per_instance_rows.append(_est_row(task, idx, optimum))
    table_rows = _summarize(per_instance_rows, expected_count=len(tasks))
    _write_csv(out_dir / "scheduling_structured_q2_table.csv", table_rows)
    _write_csv(out_dir / "scheduling_structured_q2_per_instance.csv", per_instance_rows)
    metrics = {
        "benchmark": "scheduling_structured_q2",
        "seed": seed,
        "splits": {split: len(split_tasks) for split, split_tasks in instances.items()},
        "cpsat_time_limit_seconds": cpsat_time_limit_seconds,
        "llm_calls": 0,
        "tokens": 0,
        "structured_input": True,
        "produced_artifacts": [
            "scheduling_structured_q2_table.csv",
            "scheduling_structured_q2_per_instance.csv",
            "scheduling_structured_q2_metrics.json",
        ],
        "table_rows": table_rows,
    }
    (out_dir / "scheduling_structured_q2_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _cpsat_row(task: SchedulingTask, task_index: int, optimum: float, time_limit_seconds: float) -> Dict[str, object]:
    started = time.perf_counter()
    objective: Optional[float] = None
    status = "dependency_missing"
    try:
        objective, status = _solve_with_cpsat(task, time_limit_seconds)
    except ImportError:
        status = "dependency_missing"
    except Exception as exc:
        status = f"error:{exc.__class__.__name__}"
    elapsed = time.perf_counter() - started
    return _row("CP-SAT", "Structured-input solver", task, task_index, objective, optimum, elapsed, status)


def _solve_with_cpsat(task: SchedulingTask, time_limit_seconds: float) -> Tuple[Optional[float], str]:
    from ortools.sat.python import cp_model  # type: ignore

    model = cp_model.CpModel()
    horizon = sum(task.durations.values()) + max(task.release_times.values() or [0]) + max(task.due_dates.values() or [0])
    job_start = {}
    job_end = {}
    presences_by_job = {}
    intervals_by_machine: Dict[str, List[object]] = {machine: [] for machine in task.machines}
    for job, duration in task.durations.items():
        job_start[job] = model.NewIntVar(0, horizon, f"start_{job}")
        job_end[job] = model.NewIntVar(0, horizon, f"end_{job}")
        model.Add(job_start[job] >= task.release_times[job])
        presences = []
        for machine in task.allowed[job]:
            presence = model.NewBoolVar(f"assign_{job}_{machine}")
            start = model.NewIntVar(0, horizon, f"start_{job}_{machine}")
            end = model.NewIntVar(0, horizon, f"end_{job}_{machine}")
            interval = model.NewOptionalIntervalVar(start, duration, end, presence, f"interval_{job}_{machine}")
            model.Add(job_start[job] == start).OnlyEnforceIf(presence)
            model.Add(job_end[job] == end).OnlyEnforceIf(presence)
            intervals_by_machine[machine].append(interval)
            presences.append(presence)
        model.AddExactlyOne(presences)
        presences_by_job[job] = presences
    for machine, intervals in intervals_by_machine.items():
        if intervals:
            model.AddNoOverlap(intervals)
    for job, preds in task.predecessors.items():
        for pred in preds:
            model.Add(job_start[job] >= job_end[pred])
    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, list(job_end.values()))
    tardiness_vars = []
    for job in task.durations:
        tardiness = model.NewIntVar(0, horizon, f"tardiness_{job}")
        model.Add(tardiness >= job_end[job] - task.due_dates[job])
        model.Add(tardiness >= 0)
        tardiness_vars.append(tardiness)
    model.Minimize(makespan + sum(tardiness_vars))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = 1
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return float(solver.ObjectiveValue()), status_name.lower()
    return None, status_name.lower()


def _est_row(task: SchedulingTask, task_index: int, optimum: float) -> Dict[str, object]:
    started = time.perf_counter()
    scheduled = _earliest_start_list_schedule(task)
    objective = task.objective(scheduled)
    return _row("EST / list scheduling heuristic", "Structured-input heuristic", task, task_index, objective, optimum, time.perf_counter() - started, "complete")


def _earliest_start_list_schedule(task: SchedulingTask) -> Tuple[Dict[str, object], ...]:
    scheduled: Tuple[Dict[str, object], ...] = ()
    while len(scheduled) < len(task.durations):
        ready = task.ready_jobs(scheduled)
        if not ready:
            break
        candidates = []
        for job in ready:
            for machine in task.allowed[job]:
                row = task._scheduled_row(scheduled, job, machine)
                tardiness = max(0, int(row["finish"]) - task.due_dates[job])
                candidates.append((int(row["start"]), int(row["finish"]), tardiness, task.due_dates[job], job, machine, row))
        if not candidates:
            break
        *_, row = min(candidates)
        scheduled = scheduled + (row,)
    return scheduled


def _row(
    method: str,
    transfer_setting: str,
    task: SchedulingTask,
    task_index: int,
    objective: Optional[float],
    optimum: float,
    wall_clock_seconds: float,
    status: str,
) -> Dict[str, object]:
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "task_index": task_index,
        "task_id": task.task_id,
        "success": objective == optimum,
        "gap": _gap(objective, optimum),
        "best_objective": "" if objective is None else objective,
        "optimum": optimum,
        "llm_calls": 0,
        "tokens": 0,
        "wall_clock_seconds": wall_clock_seconds,
        "status": status,
    }


def _summarize(rows: Sequence[Dict[str, object]], expected_count: int) -> List[Dict[str, object]]:
    out = []
    for method in ("CP-SAT", "EST / list scheduling heuristic"):
        method_rows = [row for row in rows if row["Method"] == method]
        completed = [row for row in method_rows if row["best_objective"] != ""]
        success = _mean(1.0 if _as_bool(row["success"]) else 0.0 for row in completed) if completed else ""
        gap = _mean(float(row["gap"]) for row in completed) if completed else ""
        out.append(
            {
                "Method": method,
                "Transfer setting": method_rows[0]["Transfer setting"] if method_rows else "",
                "success": success,
                "gap": gap,
                "llm_calls": 0,
                "tokens": 0,
                "wall_clock_seconds": sum(float(row["wall_clock_seconds"]) for row in method_rows),
                "status": "complete" if len(completed) >= expected_count else f"partial: {len(completed)}/{expected_count}",
                "notes": "structured input calibration reference; no LLM proposal or semantic grounding",
            }
        )
    return out


def _gap(value: Optional[float], optimum: float) -> float:
    if value is None or optimum == 0:
        return 1.0
    return max(0.0, (float(value) - optimum) / max(1.0, abs(optimum)))


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
