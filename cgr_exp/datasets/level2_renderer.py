from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Level2Rendering:
    rendering_id: str
    hidden_task_id: str
    perturbation: str
    text: str
    hidden_task: object
    entity_map: Dict[str, Dict[str, str]]


def render_canonical(task) -> str:
    return f"Assign all tasks in {task.task_id} to allowed resources while minimizing makespan."


def _canonical_text(task, task_names: Dict[str, str] | None = None, resource_names: Dict[str, str] | None = None) -> str:
    task_names = task_names or {t: t for t in task.durations}
    resource_names = resource_names or {r: r for r in task.resources}
    parts = ["Assign every task to one allowed resource and minimize the maximum resource load."]
    for task_id, duration in task.durations.items():
        allowed = ", ".join(resource_names[r] for r in task.allowed[task_id])
        parts.append(f"{task_names[task_id]} takes {duration} units and can use {allowed}.")
    return " ".join(parts)


def render_level2_variants(task) -> list[Level2Rendering]:
    if getattr(task, "task_id", "").startswith("sched-") or task.__class__.__name__ == "SchedulingTask":
        return render_scheduling_level2_variants(task)
    task_identity = {t: t for t in task.durations}
    resource_identity = {r: r for r in task.resources}
    renamed_tasks = {t: f"Job-{i + 1}" for i, t in enumerate(task.durations)}
    renamed_resources = {r: f"Machine-{i + 1}" for i, r in enumerate(task.resources)}
    variants = [
        (
            "canonical",
            _canonical_text(task, task_identity, resource_identity),
            {"tasks": task_identity, "resources": resource_identity},
        ),
        (
            "paraphrase",
            "Plan the workload so the busiest resource finishes as early as possible. " + _canonical_text(task, task_identity, resource_identity),
            {"tasks": task_identity, "resources": resource_identity},
        ),
        (
            "entity_rename",
            _canonical_text(task, renamed_tasks, renamed_resources),
            {"tasks": renamed_tasks, "resources": renamed_resources},
        ),
        (
            "irrelevant_context",
            _canonical_text(task, task_identity, resource_identity) + " The office also tracks coffee inventory, which is irrelevant.",
            {"tasks": task_identity, "resources": resource_identity},
        ),
        (
            "mild_ambiguity",
            _canonical_text(task, task_identity, resource_identity) + " If two choices look similar, prefer the one that keeps loads even.",
            {"tasks": task_identity, "resources": resource_identity},
        ),
    ]
    return [
        Level2Rendering(
            rendering_id=f"{task.task_id}:{name}",
            hidden_task_id=task.task_id,
            perturbation=name,
            text=text,
            hidden_task=task,
            entity_map=entity_map,
        )
        for name, text, entity_map in variants
    ]


def _scheduling_text(task, job_names: Dict[str, str] | None = None, machine_names: Dict[str, str] | None = None) -> str:
    job_names = job_names or {job: job for job in task.durations}
    machine_names = machine_names or {machine: machine for machine in task.machines}
    parts = [
        "Create a schedule that assigns every job to one allowed machine.",
        "Respect release times and precedence constraints.",
        "Minimize makespan plus total tardiness.",
    ]
    for job, duration in task.durations.items():
        allowed = ", ".join(machine_names[machine] for machine in task.allowed[job])
        preds = ", ".join(job_names[pred] for pred in task.predecessors.get(job, ())) or "none"
        parts.append(
            f"{job_names[job]} has duration {duration}, release time {task.release_times[job]}, due date {task.due_dates[job]}, allowed machines {allowed}, predecessors {preds}."
        )
    return " ".join(parts)


def render_scheduling_level2_variants(task) -> list[Level2Rendering]:
    job_identity = {job: job for job in task.durations}
    machine_identity = {machine: machine for machine in task.machines}
    renamed_jobs = {job: f"Work-order-{i + 1}" for i, job in enumerate(task.durations)}
    renamed_machines = {machine: f"Line-{i + 1}" for i, machine in enumerate(task.machines)}
    variants = [
        (
            "canonical",
            _scheduling_text(task, job_identity, machine_identity),
            {"jobs": job_identity, "machines": machine_identity},
        ),
        (
            "paraphrase",
            "A dispatcher needs a feasible production plan with low lateness. " + _scheduling_text(task, job_identity, machine_identity),
            {"jobs": job_identity, "machines": machine_identity},
        ),
        (
            "entity_rename",
            _scheduling_text(task, renamed_jobs, renamed_machines),
            {"jobs": renamed_jobs, "machines": renamed_machines},
        ),
        (
            "irrelevant_context",
            _scheduling_text(task, job_identity, machine_identity) + " The plant cafeteria menu is unrelated to the schedule.",
            {"jobs": job_identity, "machines": machine_identity},
        ),
        (
            "mild_ambiguity",
            _scheduling_text(task, job_identity, machine_identity) + " If several choices look close, prefer one that keeps machine loads balanced.",
            {"jobs": job_identity, "machines": machine_identity},
        ),
    ]
    return [
        Level2Rendering(
            rendering_id=f"{task.task_id}:{name}",
            hidden_task_id=task.task_id,
            perturbation=name,
            text=text,
            hidden_task=task,
            entity_map=entity_map,
        )
        for name, text, entity_map in variants
    ]
