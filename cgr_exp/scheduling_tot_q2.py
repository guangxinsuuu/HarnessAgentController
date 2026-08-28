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
from cgr_exp.q2_method_contracts import q2_method_contracts
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
    trace_path = out_dir / "scheduling_tot_q2_trace.jsonl"
    existing = _read_csv_rows(per_instance_path) if resume else []
    existing_trace = _read_jsonl_rows(trace_path) if resume else []
    if not resume and per_instance_path.exists():
        per_instance_path.unlink()
    if not resume and trace_path.exists():
        trace_path.unlink()
    completed = _completed_tot_renderings(existing, existing_trace)
    if resume:
        existing = [row for row in existing if str(row.get("rendering_id")) in completed]
        existing_trace = [row for row in existing_trace if str(row.get("rendering_id")) in completed]
        if existing_trace:
            _write_jsonl(trace_path, existing_trace)
        elif trace_path.exists():
            trace_path.write_text("", encoding="utf-8")
    pending = [(idx, episode) for idx, episode in enumerate(episodes) if episode.rendering_id not in completed]
    started = time.perf_counter()
    new_rows: List[Dict[str, object]] = []
    workers = max(1, int(parallel_episodes))
    if workers == 1:
        for done, (idx, episode) in enumerate(pending, start=1):
            row, trace_events = _run_one_tot_episode(client, model, idx, episode, seed, node_budget, action_budget, breadth, value_samples)
            new_rows.append(row)
            _append_csv_row(per_instance_path, row)
            for event in trace_events:
                _append_jsonl_row(trace_path, event)
            print(f"[scheduling-tot-q2] {done}/{len(pending)} episodes, tokens={row['total_tokens']}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_run_one_tot_episode, client, model, idx, episode, seed, node_budget, action_budget, breadth, value_samples): idx
                for idx, episode in pending
            }
            for done, future in enumerate(as_completed(futures), start=1):
                row, trace_events = future.result()
                new_rows.append(row)
                _append_csv_row(per_instance_path, row)
                for event in trace_events:
                    _append_jsonl_row(trace_path, event)
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
            "official_reference": "princeton-nlp/tree-of-thought-llm BFS",
            "official_entrypoints": ["method_generate", "method_evaluate", "method_select"],
            "task_adaptation": "partial ScheduleAction sequences are thoughts/states",
            "search": "breadth-first frontier search",
            "method_generate": "propose typed scheduling actions",
            "method_evaluate": "LLM frontier value/ranking",
            "method_select": "greedy top-b by averaged value samples",
            "n_select_sample": breadth,
            "n_evaluate_sample": value_samples,
            "trace_artifact": "scheduling_tot_q2_trace.jsonl",
            "oracle_future_labels_exposed": False,
        },
        "method_alignment": q2_method_contracts(["Tree of Thoughts"]),
        "produced_artifacts": [
            "scheduling_tot_q2_table.csv",
            "scheduling_tot_q2_per_instance.csv",
            "scheduling_tot_q2_trace.jsonl",
            "scheduling_tot_q2_manifest.json",
            "table5_native_row.json",
        ],
        "table_rows": table_rows,
    }
    (out_dir / "scheduling_tot_q2_manifest.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def _completed_tot_renderings(rows: Sequence[Dict[str, object]], trace_rows: Sequence[Dict[str, object]]) -> set[str]:
    row_renderings = {str(row.get("rendering_id")) for row in rows}
    finished_trace_renderings = {
        str(row.get("rendering_id"))
        for row in trace_rows
        if str(row.get("event")) == "finish"
    }
    return row_renderings & finished_trace_renderings


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
) -> tuple[Dict[str, object], List[Dict[str, object]]]:
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
    trace_events: List[Dict[str, object]] = []
    level = 0
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
                trace_events.append(_tot_trace_event(episode, task_index, level, "complete", node=node, incumbent=incumbent))
                continue
            before_calls = len(proposer.call_records)
            proposed_actions = proposer.propose(task, node, action_budget, budget_state=budget)
            proposal_record = proposer.call_records[-1] if len(proposer.call_records) > before_calls else {}
            proposal_children = []
            invalid_children = 0
            for child_idx, action in enumerate(proposed_actions):
                verified = verifier.verify(task, node, action)
                budget.charge_verifier()
                if not verified.feasible:
                    invalid_children += 1
                    continue
                child = verifier.make_child(task, node.with_updates(incumbent_at_creation=incumbent), action, verified, child_idx)
                generated += 1
                proposal_children.append(child)
                if child.completion_status == CompletionStatus.COMPLETE:
                    value = child.objective_if_complete
                    if value is not None and (incumbent is None or value < incumbent):
                        incumbent = value
                else:
                    next_frontier.append(child)
            trace_events.append(
                _tot_trace_event(
                    episode,
                    task_index,
                    level,
                    "generate",
                    node=node,
                    record=proposal_record,
                    proposed_actions=[_safe_action(action) for action in proposed_actions],
                    child_ids=[child.node_id for child in proposal_children],
                    invalid_children=invalid_children,
                    incumbent=incumbent,
                )
            )
        if not next_frontier:
            break
        score_sums = [0.0 for _ in next_frontier]
        value_sample_records = []
        for sample_idx in range(max(1, value_samples)):
            response = client.score_frontier_states(task=task, nodes=next_frontier, model=model)
            scores = [float(score) for score in response.get("scores", [])]
            fallback_used = len(scores) != len(next_frontier)
            if len(scores) != len(next_frontier):
                scores = [float(len(next_frontier) - idx) for idx in range(len(next_frontier))]
            for idx, score in enumerate(scores):
                score_sums[idx] += score
            value_calls += 1
            value_input_tokens += int(response.get("input_tokens", 0))
            value_output_tokens += int(response.get("output_tokens", 0))
            value_latency += float(response.get("latency", 0.0))
            value_sample_records.append(
                {
                    "sample_index": sample_idx,
                    "scores": scores,
                    "fallback_used": fallback_used,
                    "input_tokens": int(response.get("input_tokens", 0)),
                    "output_tokens": int(response.get("output_tokens", 0)),
                    "latency": float(response.get("latency", 0.0)),
                    "raw_output": str(response.get("raw_output", "")),
                }
            )
        ranked = sorted(range(len(next_frontier)), key=lambda idx: (score_sums[idx] / max(1, value_samples), next_frontier[idx].node_id), reverse=True)
        frontier = [next_frontier[idx] for idx in ranked[: max(1, breadth)]]
        trace_events.append(
            _tot_trace_event(
                episode,
                task_index,
                level,
                "evaluate_select",
                candidate_ids=[node.node_id for node in next_frontier],
                averaged_scores=[score_sums[idx] / max(1, value_samples) for idx in range(len(next_frontier))],
                selected_ids=[node.node_id for node in frontier],
                value_samples=value_sample_records,
                incumbent=incumbent,
            )
        )
        level += 1
    proposal_calls = len(proposer.call_records)
    proposal_input_tokens = sum(int(record.get("input_tokens", 0)) for record in proposer.call_records)
    proposal_output_tokens = sum(int(record.get("output_tokens", 0)) for record in proposer.call_records)
    proposal_latency = sum(float(record.get("latency", 0.0)) for record in proposer.call_records)
    total_tokens = proposal_input_tokens + proposal_output_tokens + value_input_tokens + value_output_tokens
    row = {
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
    trace_events.append(_tot_trace_event(episode, task_index, level, "finish", row=row))
    return row, trace_events


def _tot_trace_event(episode: Level2SchedulingEpisode, task_index: int, level: int, event: str, **kwargs) -> Dict[str, object]:
    payload = {
        "task_index": task_index,
        "task_id": episode.hidden_task_id,
        "rendering_id": episode.rendering_id,
        "perturbation": episode.perturbation,
        "level": level,
        "event": event,
    }
    payload.update(kwargs)
    if "node" in payload:
        node = payload.pop("node")
        payload["node_id"] = getattr(node, "node_id", "")
        payload["depth"] = getattr(node, "depth", 0)
        payload["partial_schedule"] = list(getattr(node, "structured_state", {}).get("scheduled", []))
    return payload


def _safe_action(action) -> Dict[str, object]:
    return action.to_dict() if hasattr(action, "to_dict") else {"action": str(action)}


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
        item = dict(row)
        for key in ("task_index", "nodes", "generated", "llm_calls", "proposal_calls", "value_calls", "input_tokens", "output_tokens", "total_tokens"):
            if key in item and item[key] != "":
                item[key] = int(float(item[key]))
        for key in ("gap", "wall_clock_seconds"):
            if key in item and item[key] != "":
                item[key] = float(item[key])
        out.append(item)
    return out
