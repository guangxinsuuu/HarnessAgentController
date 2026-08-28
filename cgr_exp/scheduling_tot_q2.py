from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from cgr_exp.budgets import BudgetState
from cgr_exp.datasets.level1_scheduling import SchedulingTask
from cgr_exp.proposers.llm_adapter import OpenAICompatibleHTTPClient, RealLLMProposerAdapter
from cgr_exp.scheduling_micro import generate_scheduling_micro_instances
from cgr_exp.scheduling_transfer import Level2SchedulingEpisode, ScriptedSchedulingClient, make_level2_scheduling_episodes
from cgr_exp.schemas import CompletionStatus, CostLedger, SearchNode
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier


def run_scheduling_tot_q2_from_env(
    out_dir: Path,
    model: Optional[str] = None,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 80,
    action_budget: int = 6,
    breadth: int = 5,
    value_samples: int = 3,
    timeout: float = 120.0,
    perturbation: str = "mixed",
    parallel_episodes: int = 1,
    resume: bool = False,
) -> Dict[str, object]:
    client = OpenAICompatibleHTTPClient.from_env()
    client.timeout = timeout
    return run_scheduling_tot_q2(
        out_dir,
        client=client,
        model=model or client.model,
        seed=seed,
        train_count=train_count,
        validation_count=validation_count,
        test_count=test_count,
        job_count=job_count,
        machine_count=machine_count,
        node_budget=node_budget,
        action_budget=action_budget,
        breadth=breadth,
        value_samples=value_samples,
        perturbation=perturbation,
        parallel_episodes=parallel_episodes,
        resume=resume,
    )


def run_scheduling_tot_q2(
    out_dir: Path,
    client,
    model: str,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 80,
    action_budget: int = 6,
    breadth: int = 5,
    value_samples: int = 3,
    perturbation: str = "mixed",
    parallel_episodes: int = 1,
    resume: bool = False,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    episodes = make_level2_scheduling_episodes(instances["test"], seed=seed + 7000, perturbation=perturbation)
    per_instance_path = out_dir / "scheduling_tot_q2_per_instance.csv"
    existing = _read_csv_rows(per_instance_path) if resume else []
    if not resume and per_instance_path.exists():
        per_instance_path.unlink()
    completed = {str(row.get("rendering_id")) for row in existing}
    pending = [(idx, episode) for idx, episode in enumerate(episodes) if episode.rendering_id not in completed]
    started = time.perf_counter()
    new_rows: List[Dict[str, object]] = []
    workers = max(1, int(parallel_episodes))
    if workers == 1:
        for done, (idx, episode) in enumerate(pending, start=1):
            row = _run_one_tot_episode(client, model, idx, episode, seed, node_budget, action_budget, breadth, value_samples)
            new_rows.append(row)
            _append_csv_row(per_instance_path, row)
            print(f"[scheduling-tot-q2] {done}/{len(pending)} episodes, tokens={row['total_tokens']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one_tot_episode, client, model, idx, episode, seed, node_budget, action_budget, breadth, value_samples): idx
                for idx, episode in pending
            }
            for done, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                new_rows.append(row)
                _append_csv_row(per_instance_path, row)
                print(f"[scheduling-tot-q2] {done}/{len(pending)} episodes, tokens={row['total_tokens']}", flush=True)
    rows = [*_coerce_rows(existing), *new_rows]
    rows.sort(key=lambda row: int(row["task_index"]))
    table_rows = [_summarize(rows, expected_count=len(episodes), wall_clock_seconds=time.perf_counter() - started)]
    _write_csv(out_dir / "scheduling_tot_q2_table.csv", table_rows)
    _write_csv(per_instance_path, rows)
    (out_dir / "table5_native_row.json").write_text(json.dumps(table_rows[0], indent=2, sort_keys=True), encoding="utf-8")
    metrics = {
        "benchmark": "scheduling_tot_q2",
        "seed": seed,
        "model": model,
        "splits": {split: len(split_tasks) for split, split_tasks in instances.items()},
        "node_budget": node_budget,
        "action_budget": action_budget,
        "breadth": breadth,
        "value_samples": value_samples,
        "parallel_episodes": workers,
        "resume": resume,
        "live_network_required": not isinstance(client, ScriptedSchedulingClient),
        "native_tot_protocol": {
            "search": "breadth-first frontier search",
            "method_generate": "propose typed scheduling actions",
            "method_evaluate": "LLM frontier value/ranking",
            "method_select": "greedy top-b by averaged value samples",
        },
        "produced_artifacts": [
            "scheduling_tot_q2_table.csv",
            "scheduling_tot_q2_per_instance.csv",
            "scheduling_tot_q2_manifest.json",
            "table5_native_row.json",
        ],
        "table_rows": table_rows,
    }
    (out_dir / "scheduling_tot_q2_manifest.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _run_one_tot_episode(
    client,
    model: str,
    task_index: int,
    episode: Level2SchedulingEpisode,
    seed: int,
    node_budget: int,
    action_budget: int,
    breadth: int,
    value_samples: int,
) -> Dict[str, object]:
    del seed
    task = episode.hidden_task
    verifier = Level1SchedulingVerifier()
    proposer = RealLLMProposerAdapter(client=client, model=model, prompt_template_version="scheduling-level2-v1", temperature=0.0, top_p=1.0)
    budget = BudgetState()
    budget.cost = CostLedger(solver_calls=1)
    frontier: List[SearchNode] = [task.root_node()]
    incumbent = None
    expanded = 0
    generated = 0
    value_calls = 0
    value_input_tokens = 0
    value_output_tokens = 0
    value_latency = 0.0
    started = time.perf_counter()
    optimum = task.exhaustive_oracle()
    while frontier and expanded < node_budget:
        next_frontier: List[SearchNode] = []
        for node in frontier[: max(1, breadth)]:
            if expanded >= node_budget:
                break
            expanded += 1
            if node.completion_status == CompletionStatus.COMPLETE:
                value = node.objective_if_complete
                if value is not None and (incumbent is None or value < incumbent):
                    incumbent = value
                continue
            for child_idx, action in enumerate(proposer.propose(task, node, action_budget, budget_state=budget)):
                verified = verifier.verify(task, node, action)
                budget.charge_verifier()
                if not verified.feasible:
                    continue
                child = verifier.make_child(task, node.with_updates(incumbent_at_creation=incumbent), action, verified, child_idx)
                generated += 1
                if child.completion_status == CompletionStatus.COMPLETE:
                    value = child.objective_if_complete
                    if value is not None and (incumbent is None or value < incumbent):
                        incumbent = value
                else:
                    next_frontier.append(child)
        if not next_frontier:
            break
        score_sums = [0.0 for _ in next_frontier]
        for _sample in range(max(1, value_samples)):
            response = client.score_frontier_states(task=task, nodes=next_frontier, model=model)
            scores = [float(score) for score in response.get("scores", [])]
            if len(scores) != len(next_frontier):
                scores = [float(len(next_frontier) - idx) for idx in range(len(next_frontier))]
            for idx, score in enumerate(scores):
                score_sums[idx] += score
            value_calls += 1
            value_input_tokens += int(response.get("input_tokens", 0))
            value_output_tokens += int(response.get("output_tokens", 0))
            value_latency += float(response.get("latency", 0.0))
        ranked = sorted(range(len(next_frontier)), key=lambda idx: (score_sums[idx] / max(1, value_samples), next_frontier[idx].node_id), reverse=True)
        frontier = [next_frontier[idx] for idx in ranked[: max(1, breadth)]]
    proposal_calls = len(proposer.call_records)
    proposal_input_tokens = sum(int(record.get("input_tokens", 0)) for record in proposer.call_records)
    proposal_output_tokens = sum(int(record.get("output_tokens", 0)) for record in proposer.call_records)
    proposal_latency = sum(float(record.get("latency", 0.0)) for record in proposer.call_records)
    total_tokens = proposal_input_tokens + proposal_output_tokens + value_input_tokens + value_output_tokens
    return {
        "Method": "Tree of Thoughts / GPT-frontier-rank",
        "Transfer setting": "Online",
        "task_index": task_index,
        "task_id": episode.hidden_task_id,
        "rendering_id": episode.rendering_id,
        "perturbation": episode.perturbation,
        "success": incumbent == optimum,
        "gap": _gap(incumbent, optimum),
        "best_objective": "" if incumbent is None else incumbent,
        "optimum": optimum,
        "nodes": expanded,
        "generated": generated,
        "llm_calls": proposal_calls + value_calls,
        "proposal_calls": proposal_calls,
        "value_calls": value_calls,
        "input_tokens": proposal_input_tokens + value_input_tokens,
        "output_tokens": proposal_output_tokens + value_output_tokens,
        "total_tokens": total_tokens,
        "wall_clock_seconds": time.perf_counter() - started + proposal_latency + value_latency,
        "status": "complete",
        "notes": "ToT BFS propose/value/select; value scores averaged across samples",
    }


def _summarize(rows: Sequence[Dict[str, object]], expected_count: int, wall_clock_seconds: float) -> Dict[str, object]:
    return {
        "Method": "Tree of Thoughts",
        "Training states": "online LLM frontier guidance",
        "success": _mean(1.0 if _as_bool(row["success"]) else 0.0 for row in rows),
        "gap": _mean(float(row["gap"]) for row in rows),
        "llm_calls": sum(int(row["llm_calls"]) for row in rows),
        "tokens": sum(int(row["total_tokens"]) for row in rows),
        "wall_clock_seconds": wall_clock_seconds,
        "status": "complete" if len(rows) >= expected_count else f"partial: {len(rows)}/{expected_count}",
        "source_policy": "native_tot_bfs",
        "notes": "ToT BFS with propose/evaluate/select and averaged LLM value samples",
    }


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


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
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


def _coerce_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    out = []
    for row in rows:
        item = dict(row)
        for key in ("task_index", "nodes", "generated", "llm_calls", "proposal_calls", "value_calls", "input_tokens", "output_tokens", "total_tokens"):
            if key in item and item[key] != "":
                item[key] = int(float(item[key]))
        for key in ("gap", "wall_clock_seconds"):
            if key in item and item[key] != "":
                item[key] = float(item[key])
        out.append(item)
    return out
