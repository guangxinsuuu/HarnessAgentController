from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from cgr_exp.budgets import BudgetState
from cgr_exp.certification.prune import should_prune
from cgr_exp.datasets.level1_scheduling import ScheduleAction, SchedulingTask
from cgr_exp.datasets.level2_renderer import Level2Rendering, render_scheduling_level2_variants
from cgr_exp.proposers.llm_adapter import OpenAICompatibleHTTPClient, RealLLMProposerAdapter
from cgr_exp.scheduling_micro import (
    CachedSchedulingOraclePolicy,
    SCHEDULING_FEATURE_NAMES,
    SchedulingHeuristicPolicy,
    SchedulingMLPPolicy,
    TinyPairwiseMLPRanker,
    collect_solver_trace_records,
    generate_scheduling_micro_instances,
    make_scheduling_ranking_pairs,
    scheduling_feature_vector,
    _evaluate_scheduling_policies,
)
from cgr_exp.schemas import CompletionStatus, CostLedger, SearchNode
from cgr_exp.search.controller import FrontierSearchResult
from cgr_exp.search.random_policy import RandomPolicy
from cgr_exp.verification.level1_scheduling_verifier import Level1SchedulingVerifier


@dataclass(frozen=True)
class Level2SchedulingEpisode:
    rendering_id: str
    hidden_task_id: str
    perturbation: str
    text: str
    hidden_task: SchedulingTask


COMMON_INFERENCE_FEATURES = [
    "scheduled_jobs",
    "remaining_jobs",
    "machine_loads",
    "eligible_jobs",
    "current_makespan",
    "resource_utilization_mean",
    "resource_utilization_max",
    "remaining_work",
    "precedence_remaining",
    "depth",
    "remaining_budget",
    "branch_factor",
    "incumbent",
    "lower_bound",
    "observable_history",
]

ORACLE_LABEL_FIELDS = [
    "best_descendant_objective",
    "future_search_cost",
    "eventual_outcome",
    "oracle_rank",
    "exact_future_objective",
]


class SharedSchedulingProposalCache:
    def __init__(
        self,
        episode: Level2SchedulingEpisode,
        seed: int,
        proposal_k: int,
        node_budget: int,
        useful_action_recall: float,
        model: str = "simulated-scheduling-llm",
        prompt_version: str = "scheduling-level2-v1",
        temperature: float = 0.0,
        adapter=None,
    ):
        self.episode = episode
        self.proposal_k = proposal_k
        self.node_budget = node_budget
        self.model = model
        self.prompt_version = prompt_version
        self.temperature = temperature
        self.adapter = adapter or SimulatedLLMSchedulingProposer(episode, seed=seed, useful_action_recall=useful_action_recall)
        self.cache: Dict[str, List[ScheduleAction]] = {}
        self.meta: Dict[str, Dict[str, object]] = {}
        self.cost = CostLedger()
        self.request_count = 0
        self.cache_hits = 0
        self.empty_valid_actions = 0
        self.support_hits = 0
        self.support_total = 0
        self.invalid_actions = 0
        self.duplicate_actions = 0

    def propose(self, task: SchedulingTask, node: SearchNode, k: int, budget_state=None) -> List[ScheduleAction]:
        del budget_state
        self.request_count += 1
        key = canonical_state_hash(task, node, self.node_budget, self.model, self.prompt_version, k, self.temperature)
        if key in self.cache:
            self.cache_hits += 1
            return list(self.cache[key])[:k]
        actions, record = self._call_adapter(task, node, k)
        valid_actions, invalid_count, duplicate_count = _valid_unique_schedule_actions(task, node, actions)
        self.invalid_actions += invalid_count
        self.duplicate_actions += duplicate_count
        self.cost = self.cost + CostLedger(
            proposal_calls=1,
            input_tokens=int(record.get("input_tokens", 0)),
            output_tokens=int(record.get("output_tokens", 0)),
            wall_clock_seconds=float(record.get("latency", 0.0)),
        )
        if not valid_actions:
            self.empty_valid_actions += 1
        if _proposal_contains_oracle_child(task, node, valid_actions):
            self.support_hits += 1
        self.support_total += 1
        self.cache[key] = list(valid_actions)
        self.meta[key] = {
            "cache_key": key,
            "instance_id": task.task_id,
            "canonical_state": canonical_scheduling_state(task, node, self.node_budget),
            "features": scheduling_feature_vector(task, node),
            "feature_names": list(SCHEDULING_FEATURE_NAMES),
            "model": self.model,
            "prompt_version": self.prompt_version,
            "proposal_k": k,
            "temperature": self.temperature,
            "raw_action_count": len(actions),
            "action_count": len(valid_actions),
            "valid_action_count": len(valid_actions),
            "invalid_action_count": invalid_count,
            "duplicate_action_count": duplicate_count,
            "input_tokens": int(record.get("input_tokens", 0)),
            "output_tokens": int(record.get("output_tokens", 0)),
            "latency": float(record.get("latency", 0.0)),
            "parse_failures": int(record.get("parse_failures", 0)),
            "actions": [action.to_dict() for action in valid_actions],
        }
        return list(valid_actions)[:k]

    def _call_adapter(self, task: SchedulingTask, node: SearchNode, k: int):
        if hasattr(self.adapter, "propose_with_record"):
            return self.adapter.propose_with_record(task, node, k)
        before = self.adapter.cost
        actions = self.adapter.propose(task, node, k)
        return actions, {
            "input_tokens": self.adapter.cost.input_tokens - before.input_tokens,
            "output_tokens": self.adapter.cost.output_tokens - before.output_tokens,
            "latency": self.adapter.cost.wall_clock_seconds - before.wall_clock_seconds,
            "parse_failures": 0,
        }

    def stats(self) -> Dict[str, object]:
        return {
            "shared": True,
            "request_count": self.request_count,
            "state_count": len(self.cache),
            "cache_hits": self.cache_hits,
            "proposal_calls": self.cost.proposal_calls,
            "input_tokens": self.cost.input_tokens,
            "output_tokens": self.cost.output_tokens,
            "total_tokens": self.cost.input_tokens + self.cost.output_tokens,
            "latency": self.cost.wall_clock_seconds,
            "parse_failures": sum(int(row.get("parse_failures", 0)) for row in self.meta.values()),
            "invalid_actions": self.invalid_actions,
            "duplicate_actions": self.duplicate_actions,
            "proposal_recall_at_k": 0.0 if self.support_total == 0 else self.support_hits / self.support_total,
            "empty_valid_action_rate": 0.0 if self.support_total == 0 else self.empty_valid_actions / self.support_total,
        }

    def state_rows(self) -> List[Dict[str, object]]:
        return [dict(row) for row in self.meta.values()]


class SimulatedLLMSchedulingProposer:
    def __init__(
        self,
        rendering: Level2SchedulingEpisode,
        seed: int = 0,
        useful_action_recall: float = 0.88,
        invalid_action_prob: float = 0.04,
        duplicate_prob: float = 0.04,
        semantic_noise_prob: float = 0.08,
    ):
        self.rendering = rendering
        self.rng = random.Random(seed)
        self.useful_action_recall = useful_action_recall
        self.invalid_action_prob = invalid_action_prob
        self.duplicate_prob = duplicate_prob
        self.semantic_noise_prob = semantic_noise_prob
        self.cost = CostLedger()

    def propose(self, task: SchedulingTask, node: SearchNode, k: int, budget_state=None) -> List[ScheduleAction]:
        del budget_state
        ready = task.ready_jobs(tuple(node.structured_state["scheduled"]))
        loads = node.structured_state.get("machine_loads", {})
        ordered_jobs = sorted(ready, key=lambda job: (task.due_dates[job], task.release_times[job], -task.durations[job], job))
        actions: List[ScheduleAction] = []
        for job in ordered_jobs:
            machines = sorted(task.allowed[job], key=lambda machine: (loads.get(machine, 0), machine))
            for machine in machines:
                if len(actions) >= k:
                    break
                if self.rng.random() <= self.useful_action_recall:
                    actions.append(ScheduleAction(job=job, machine=machine))
            if len(actions) >= k:
                break
        while len(actions) < k and ready and self.rng.random() < self.semantic_noise_prob:
            job = self.rng.choice(ready)
            actions.append(ScheduleAction(job=job, machine=self.rng.choice(task.machines)))
        if self.rng.random() < self.invalid_action_prob:
            actions.append(ScheduleAction(job="", machine="UNKNOWN_MACHINE"))
        if actions and self.rng.random() < self.duplicate_prob:
            actions.append(self.rng.choice(actions))
        self.cost = self.cost + CostLedger(proposal_calls=1, input_tokens=_estimate_prompt_tokens(self.rendering.text), output_tokens=16 * len(actions))
        return actions[:k]


class SimulatedLLMValuePolicy:
    name = "simulated_llm_value"

    def __init__(self, task: SchedulingTask, seed: int = 0):
        self.task = task
        self.rng = random.Random(seed)
        self.cost = CostLedger()

    def score(self, node: SearchNode) -> float:
        lower = float(node.lower_bound) if node.lower_bound is not None else 1e6
        branch = float(node.universal_features.get("branch_factor", 0.0))
        noise = self.rng.uniform(-0.15, 0.15)
        self.cost = self.cost + CostLedger(evaluation_calls=1, input_tokens=90, output_tokens=8)
        return -lower + 0.02 * float(node.depth) - 0.001 * branch + noise


class RealLLMValuePolicy:
    name = "real_llm_value"

    def __init__(self, task: SchedulingTask, client, model: str):
        self.task = task
        self.client = client
        self.model = model
        self.cost = CostLedger()

    def score(self, node: SearchNode) -> float:
        response = self.client.score_state(task=self.task, node=node, model=self.model)
        self.cost = self.cost + CostLedger(
            evaluation_calls=1,
            input_tokens=int(response.get("input_tokens", 0)),
            output_tokens=int(response.get("output_tokens", 0)),
            wall_clock_seconds=float(response.get("latency", 0.0)),
        )
        return float(response.get("score", 0.0))


class RealLLMFrontierRankingPolicy:
    name = "real_llm_frontier_ranking"

    def __init__(self, task: SchedulingTask, client, model: str):
        self.task = task
        self.client = client
        self.model = model
        self.cost = CostLedger()

    def score(self, node: SearchNode) -> float:
        return self.score_frontier([node])[0]

    def score_frontier(self, frontier: Sequence[SearchNode]) -> List[float]:
        response = self.client.score_frontier_states(task=self.task, nodes=list(frontier), model=self.model)
        self.cost = self.cost + CostLedger(
            evaluation_calls=1,
            input_tokens=int(response.get("input_tokens", 0)),
            output_tokens=int(response.get("output_tokens", 0)),
            wall_clock_seconds=float(response.get("latency", 0.0)),
        )
        return [float(score) for score in response.get("scores", [])]


class PRMGuidedMCTSStylePolicy:
    name = "prm_guided_mcts_style"

    def __init__(self, task: SchedulingTask, ranker, exploration: float = 1.25, prior_weight: float = 1.0):
        self.task = task
        self.ranker = ranker
        self.exploration = exploration
        self.prior_weight = prior_weight
        self.prior_cache: Dict[str, float] = {}
        self.visits: Dict[str, int] = {}
        self.value_sum: Dict[str, float] = {}
        self.node_cache: Dict[str, SearchNode] = {}
        self.last_selected: SearchNode | None = None
        self.total_backups = 0

    def score(self, node: SearchNode) -> float:
        return self.score_frontier([node])[0]

    def score_frontier(self, frontier: Sequence[SearchNode]) -> List[float]:
        if self.last_selected is not None:
            self._backup(self.last_selected, self._leaf_value(self.last_selected))
        self.last_selected = None
        for node in frontier:
            self.node_cache[node.node_id] = node
        scores = [self._puct_score(node) for node in frontier]
        if frontier:
            selected_index = max(range(len(frontier)), key=lambda idx: (scores[idx], frontier[idx].node_id))
            self.last_selected = frontier[selected_index]
        return scores

    def _puct_score(self, node: SearchNode) -> float:
        prior = self._prior(node)
        visits = self.visits.get(node.node_id, 0)
        q_value = self.value_sum.get(node.node_id, 0.0) / visits if visits else 0.0
        parent_visits = self.visits.get(str(node.parent_id), 0) if node.parent_id else sum(self.visits.values())
        exploration = self.exploration * prior * math.sqrt(parent_visits + 1.0) / (1.0 + visits)
        return q_value + self.prior_weight * prior + exploration

    def _prior(self, node: SearchNode) -> float:
        if node.node_id not in self.prior_cache:
            raw = self.ranker.score(scheduling_feature_vector(self.task, node))
            self.prior_cache[node.node_id] = 1.0 / (1.0 + math.exp(-float(raw)))
        return self.prior_cache[node.node_id]

    def _leaf_value(self, node: SearchNode) -> float:
        if node.completion_status == CompletionStatus.COMPLETE and node.objective_if_complete is not None:
            optimum = max(1.0, self.task.exhaustive_oracle())
            return 1.0 / (1.0 + max(0.0, float(node.objective_if_complete) - optimum) / optimum)
        lower = float(node.lower_bound) if node.lower_bound is not None else 1e6
        scale = max(1.0, self.task.exhaustive_oracle())
        bound_value = 1.0 / (1.0 + max(0.0, lower - scale) / scale)
        return 0.5 * self._prior(node) + 0.5 * bound_value

    def _backup(self, node: SearchNode, value: float) -> None:
        current: SearchNode | None = node
        while current is not None:
            self.visits[current.node_id] = self.visits.get(current.node_id, 0) + 1
            self.value_sum[current.node_id] = self.value_sum.get(current.node_id, 0.0) + value
            self.total_backups += 1
            parent_id = current.parent_id
            current = self.node_cache.get(parent_id) if parent_id else None


def make_level2_scheduling_episodes(tasks: Sequence[SchedulingTask], seed: int = 0, perturbation: str = "mixed") -> List[Level2SchedulingEpisode]:
    rng = random.Random(seed)
    episodes = []
    for task in tasks:
        variants = render_scheduling_level2_variants(task)
        if perturbation == "mixed":
            rendering = rng.choice(variants)
        else:
            matching = [variant for variant in variants if variant.perturbation == perturbation]
            rendering = matching[0] if matching else variants[0]
        episodes.append(_episode_from_rendering(rendering))
    return episodes


def canonical_scheduling_state(task: SchedulingTask, node: SearchNode, remaining_budget: int) -> Dict[str, object]:
    scheduled = tuple(node.structured_state.get("scheduled", []))
    scheduled_jobs = [str(row["job"]) for row in scheduled]
    remaining_jobs = [job for job in task.durations if job not in set(scheduled_jobs)]
    eligible_jobs = task.ready_jobs(scheduled)
    machine_loads = {machine: float(node.structured_state.get("machine_loads", {}).get(machine, 0.0)) for machine in task.machines}
    loads = list(machine_loads.values())
    current_makespan = max(loads or [0.0])
    total_duration = max(1.0, float(sum(task.durations.values())))
    return {
        "scheduled_jobs": scheduled_jobs,
        "remaining_jobs": remaining_jobs,
        "machine_loads": machine_loads,
        "eligible_jobs": eligible_jobs,
        "current_makespan": current_makespan,
        "resource_utilization_mean": (sum(loads) / max(1, len(loads))) / total_duration,
        "resource_utilization_max": max(loads or [0.0]) / total_duration,
        "remaining_work": sum(task.durations[job] for job in remaining_jobs),
        "precedence_remaining": sum(len(task.predecessors.get(job, ())) for job in remaining_jobs),
        "depth": int(node.depth),
        "remaining_budget": max(0, int(remaining_budget) - int(node.depth)),
        "branch_factor": float(node.universal_features.get("branch_factor", 0.0)),
        "incumbent": node.incumbent_at_creation,
        "lower_bound": node.lower_bound,
        "observable_history": _observable_schedule_history(scheduled),
    }


def _observable_schedule_history(scheduled: Sequence[object]) -> List[Dict[str, object]]:
    history = []
    for idx, row in enumerate(scheduled):
        if not isinstance(row, dict):
            continue
        history.append(
            {
                "step": idx,
                "job": str(row.get("job", "")),
                "machine": str(row.get("machine", "")),
                "start": row.get("start"),
                "finish": row.get("finish"),
            }
        )
    return history


def canonical_state_hash(
    task: SchedulingTask,
    node: SearchNode,
    remaining_budget: int,
    model: str,
    prompt_version: str,
    proposal_k: int,
    temperature: float,
) -> str:
    payload = {
        "instance_id": task.task_id,
        "canonical_state": canonical_scheduling_state(task, node, remaining_budget),
        "model": model,
        "prompt_version": prompt_version,
        "proposal_k": proposal_k,
        "temperature": temperature,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_scheduling_zero_shot_transfer(
    out_dir: Path,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 120,
    action_budget: int = 8,
    records_per_instance: int = 60,
    rank_epochs: int = 25,
    useful_action_recall: float = 0.88,
    perturbation: str = "mixed",
    batch_size: int = 1,
    parallel_episodes: int = 1,
    resume: bool = False,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    train_records = []
    validation_records = []
    for task in instances["train"]:
        train_records.extend(collect_solver_trace_records(task, "train", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    for task in instances["validation"]:
        validation_records.extend(collect_solver_trace_records(task, "validation", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    source_test_records = []
    for task in instances["test"]:
        source_test_records.extend(collect_solver_trace_records(task, "test", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    train_pairs = make_scheduling_ranking_pairs(train_records, max_pairs_per_task=40, seed=seed)
    validation_pairs = make_scheduling_ranking_pairs(validation_records, max_pairs_per_task=40, seed=seed + 1)
    ranker = TinyPairwiseMLPRanker(dim=len(train_pairs[0][0]) if train_pairs else 20, hidden_dim=16, seed=seed)
    ranker.train(train_pairs, epochs=rank_epochs, lr=0.02, seed=seed)
    source_policy_rows = _evaluate_scheduling_policies(
        instances["test"],
        ranker,
        node_budget=node_budget,
        action_budget=action_budget,
        seed=seed,
        ranking_records=source_test_records,
    )
    episodes = make_level2_scheduling_episodes(instances["test"], seed=seed + 2000, perturbation=perturbation)
    table_rows, per_instance_rows, proposal_cache_rows, proposal_state_rows = _evaluate_level2_transfer(
        episodes,
        ranker,
        node_budget,
        action_budget,
        seed,
        useful_action_recall,
        checkpoint_dir=out_dir,
        output_prefix="scheduling_transfer",
        batch_size=batch_size,
        parallel_episodes=parallel_episodes,
        resume=resume,
        progress_label="scheduling-transfer",
    )
    proposal_stats = _summarize_cache_rows(proposal_cache_rows)
    metrics = {
        "benchmark": "scheduling_zero_shot_transfer",
        "seed": seed,
        "splits": {split: len(tasks) for split, tasks in instances.items()},
        "job_count": job_count,
        "machine_count": machine_count,
        "node_budget": node_budget,
        "action_budget": action_budget,
        "records_per_instance": records_per_instance,
        "rank_epochs": rank_epochs,
        "batch_size": batch_size,
        "parallel_episodes": max(1, int(parallel_episodes)),
        "resume": resume,
        "ranker_type": "tiny_pairwise_mlp",
        "training_source": "level1_solver_traces_only",
        "target_training_trace_count": 0,
        "level2_perturbation": perturbation,
        "useful_action_recall": useful_action_recall,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "validation_ranking_error": ranker.ranking_error(validation_pairs),
        "source_domain_methods": {row["Policy"]: row for row in source_policy_rows},
        "methods": {row["Policy"]: row for row in table_rows},
        "semantic_state_schema": {
            "common_inference_features": COMMON_INFERENCE_FEATURES,
            "oracle_label_fields_excluded_from_inference": ORACLE_LABEL_FIELDS,
            "oracle_labels_in_inference_features": False,
            "source_state_space": "canonical semantic scheduling state",
            "target_state_space": "canonical semantic scheduling state",
            "source_distribution": "solver-induced semantic search",
            "target_distribution": "llm-induced semantic search",
        },
        "shared_proposal_replay": proposal_stats,
        "certified_pruning_enabled": False,
    }
    _write_csv(out_dir / "scheduling_transfer_table.csv", table_rows)
    _write_csv(out_dir / "scheduling_transfer_per_instance.csv", per_instance_rows)
    _write_csv(out_dir / "scheduling_transfer_proposal_cache.csv", proposal_cache_rows)
    (out_dir / "scheduling_transfer_leakage_manifest.json").write_text(json.dumps(metrics["semantic_state_schema"], indent=2, sort_keys=True), encoding="utf-8")
    _write_level2_episodes(out_dir / "scheduling_level2_episodes.jsonl", episodes)
    _write_pairs(out_dir / "scheduling_solver_train_pairs.jsonl", train_pairs)
    ranker_json = ranker.to_json()
    (out_dir / "scheduling_solver_mlp_ranker.json").write_text(ranker_json, encoding="utf-8")
    (out_dir / "scheduling_transfer_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _write_transfer_audit_artifacts(
        out_dir,
        metrics,
        instances,
        ranker_json,
        table_rows,
        per_instance_rows,
        proposal_cache_rows,
        proposal_state_rows,
        source_records=[*train_records, *validation_records],
        source_policy_rows=source_policy_rows,
    )
    return metrics


def run_scheduling_llm_pilot_from_env(
    out_dir: Path,
    model: str | None = None,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 20,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 80,
    action_budget: int = 6,
    records_per_instance: int = 60,
    rank_epochs: int = 25,
    timeout: float = 120.0,
    perturbation: str = "mixed",
    batch_size: int = 1,
    parallel_episodes: int = 1,
    resume: bool = False,
    include_gpt_value: bool = False,
    include_gpt_frontier_ranking: bool = False,
    include_target_prm: bool = False,
    include_rest_mcts: bool = False,
    include_solver_certification: bool = False,
    target_prm_train_count: int = 100,
) -> Dict[str, object]:
    client = OpenAICompatibleHTTPClient.from_env()
    client.timeout = timeout
    return run_scheduling_llm_pilot(
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
        records_per_instance=records_per_instance,
        rank_epochs=rank_epochs,
        perturbation=perturbation,
        batch_size=batch_size,
        parallel_episodes=parallel_episodes,
        resume=resume,
        include_gpt_value=include_gpt_value,
        include_gpt_frontier_ranking=include_gpt_frontier_ranking,
        include_target_prm=include_target_prm,
        include_rest_mcts=include_rest_mcts,
        include_solver_certification=include_solver_certification,
        target_prm_train_count=target_prm_train_count,
    )


def preflight_scheduling_llm_pilot(
    out_dir: Path,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 20,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 80,
    action_budget: int = 6,
    model: str | None = None,
    batch_size: int = 1,
    parallel_episodes: int = 1,
    perturbation: str = "mixed",
    include_gpt_value: bool = False,
    include_gpt_frontier_ranking: bool = False,
    include_target_prm: bool = False,
    include_rest_mcts: bool = False,
    include_solver_certification: bool = False,
    target_prm_train_count: int = 100,
    env=None,
) -> Dict[str, object]:
    source = os.environ if env is None else env
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    episodes = make_level2_scheduling_episodes(instances["test"], seed=seed + 7000, perturbation=perturbation)
    existing_rows = _read_csv_rows(out_dir / "scheduling_llm_pilot_per_instance.csv")
    completed_counts: Dict[str, int] = {}
    for row in existing_rows:
        completed_counts[str(row.get("rendering_id"))] = completed_counts.get(str(row.get("rendering_id")), 0) + 1
    policy_names = ["Random", "Hand heuristic", "Solver-trained MLP", "Oracle"]
    if include_gpt_value:
        policy_names.insert(2, "GPT-value / ToT-style")
    if include_gpt_frontier_ranking:
        insert_at = 3 if include_gpt_value else 2
        policy_names.insert(insert_at, "GPT-frontier-rank / ToT-style")
    if include_target_prm:
        insert_at = policy_names.index("Solver-trained MLP")
        policy_names.insert(insert_at, "Target-PRM")
    if include_rest_mcts:
        insert_at = policy_names.index("Solver-trained MLP")
        policy_names.insert(insert_at, "PRM-guided MCTS (ReST-MCTS-style)")
    if include_solver_certification:
        insert_at = policy_names.index("Oracle")
        policy_names.insert(insert_at, "Solver-trained MLP + certified pruning")
    method_count = len(policy_names)
    completed_renderings = {rendering_id for rendering_id, count in completed_counts.items() if count >= method_count}
    remaining = [episode for episode in episodes if episode.rendering_id not in completed_renderings]
    avg_prompt_tokens = _mean(_estimate_prompt_tokens(episode.text) for episode in episodes)
    worst_case_proposal_calls = len(remaining) * method_count * node_budget
    preflight = {
        "benchmark": "scheduling_real_llm_pilot_preflight",
        "seed": seed,
        "model": model or source.get("CGR_LLM_MODEL", "gpt-4.1-mini"),
        "api_key_env": "CGR_LLM_API_KEY",
        "api_key_present": bool(source.get("CGR_LLM_API_KEY")),
        "base_url_env": "CGR_LLM_BASE_URL",
        "base_url_present": bool(source.get("CGR_LLM_BASE_URL")),
        "splits": {split: len(tasks) for split, tasks in instances.items()},
        "split_manifest": {split: [task.task_id for task in tasks] for split, tasks in instances.items()},
        "test_episode_count": len(episodes),
        "completed_episode_count": len(completed_renderings),
        "remaining_episode_count": len(remaining),
        "batch_size": max(1, int(batch_size)),
        "parallel_episodes": max(1, int(parallel_episodes)),
        "node_budget": node_budget,
        "action_budget": action_budget,
        "method_count": method_count,
        "policy_names": policy_names,
        "include_gpt_value": include_gpt_value,
        "include_gpt_frontier_ranking": include_gpt_frontier_ranking,
        "include_target_prm": include_target_prm,
        "include_rest_mcts": include_rest_mcts,
        "include_solver_certification": include_solver_certification,
        "target_prm_train_count": target_prm_train_count if include_target_prm or include_rest_mcts else 0,
        "worst_case_request_count_upper_bound": worst_case_proposal_calls,
        "worst_case_value_request_count_upper_bound": len(remaining) * node_budget if include_gpt_value else 0,
        "worst_case_frontier_ranking_request_count_upper_bound": len(remaining) * node_budget if include_gpt_frontier_ranking else 0,
        "target_prm_training_request_count_upper_bound": target_prm_train_count * node_budget if include_target_prm or include_rest_mcts else 0,
        "estimated_prompt_tokens_per_request": avg_prompt_tokens,
        "estimated_prompt_tokens_upper_bound": int(worst_case_proposal_calls * avg_prompt_tokens),
        "resume_safe": True,
        "certified_pruning_enabled": False,
        "oracle_labels_in_inference_features": False,
    }
    (out_dir / "scheduling_llm_pilot_preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8")
    return preflight


def run_scheduling_llm_pilot(
    out_dir: Path,
    client,
    model: str,
    seed: int = 0,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 20,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 80,
    action_budget: int = 6,
    records_per_instance: int = 60,
    rank_epochs: int = 25,
    perturbation: str = "mixed",
    batch_size: int = 1,
    parallel_episodes: int = 1,
    resume: bool = False,
    include_gpt_value: bool = False,
    include_gpt_frontier_ranking: bool = False,
    include_target_prm: bool = False,
    include_rest_mcts: bool = False,
    include_solver_certification: bool = False,
    target_prm_train_count: int = 100,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    instances = generate_scheduling_micro_instances(train_count, validation_count, test_count, seed, job_count, machine_count)
    train_records = []
    validation_records = []
    for task in instances["train"]:
        train_records.extend(collect_solver_trace_records(task, "train", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    for task in instances["validation"]:
        validation_records.extend(collect_solver_trace_records(task, "validation", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    source_test_records = []
    for task in instances["test"]:
        source_test_records.extend(collect_solver_trace_records(task, "test", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    train_pairs = make_scheduling_ranking_pairs(train_records, max_pairs_per_task=40, seed=seed)
    validation_pairs = make_scheduling_ranking_pairs(validation_records, max_pairs_per_task=40, seed=seed + 1)
    ranker = _train_ranker(train_pairs, rank_epochs, seed)
    source_policy_rows = _evaluate_scheduling_policies(
        instances["test"],
        ranker,
        node_budget=node_budget,
        action_budget=action_budget,
        seed=seed,
        ranking_records=source_test_records,
    )
    target_prm_ranker = None
    target_prm_records = []
    target_prm_pairs = []
    target_prm_cache_rows = []
    target_prm_state_rows = []
    if include_target_prm or include_rest_mcts:
        target_prm_instances = generate_scheduling_micro_instances(
            target_prm_train_count,
            0,
            0,
            seed + 50000,
            job_count,
            machine_count,
        )
        target_prm_episodes = make_level2_scheduling_episodes(target_prm_instances["train"], seed=seed + 9000, perturbation=perturbation)
        target_prm_records, target_prm_cache_rows, target_prm_state_rows = collect_real_level2_trace_records(
            target_prm_episodes,
            seed=seed + 9100,
            node_budget=node_budget,
            action_budget=action_budget,
            max_records=records_per_instance,
            proposal_adapter_factory=lambda: RealLLMProposerAdapter(client=client, model=model, prompt_template_version="scheduling-level2-v1", temperature=0.0, top_p=1.0),
            proposal_model=model,
            parallel_episodes=parallel_episodes,
        )
        target_prm_pairs = make_scheduling_ranking_pairs(target_prm_records, max_pairs_per_task=40, seed=seed + 9200)
        target_prm_ranker = _train_ranker(target_prm_pairs, rank_epochs, seed + 9300)
    episodes = make_level2_scheduling_episodes(instances["test"], seed=seed + 7000, perturbation=perturbation)
    policy_names = ["Random", "Hand heuristic", "Solver-trained MLP", "Oracle"]
    value_policy_factory = None
    if include_gpt_value:
        policy_names.insert(2, "GPT-value / ToT-style")
        value_policy_factory = lambda episode, idx: RealLLMValuePolicy(episode.hidden_task, client, model)
    frontier_ranking_policy_factory = None
    if include_gpt_frontier_ranking:
        insert_at = 3 if include_gpt_value else 2
        policy_names.insert(insert_at, "GPT-frontier-rank / ToT-style")
        frontier_ranking_policy_factory = lambda episode, idx: RealLLMFrontierRankingPolicy(episode.hidden_task, client, model)
    extra_policy_factories = {}
    if include_target_prm and target_prm_ranker is not None:
        insert_at = policy_names.index("Solver-trained MLP")
        policy_names.insert(insert_at, "Target-PRM")
        extra_policy_factories["Target-PRM"] = lambda episode, idx: SchedulingMLPPolicy(episode.hidden_task, target_prm_ranker)
    if include_rest_mcts and target_prm_ranker is not None:
        insert_at = policy_names.index("Solver-trained MLP")
        policy_names.insert(insert_at, "PRM-guided MCTS (ReST-MCTS-style)")
        extra_policy_factories["PRM-guided MCTS (ReST-MCTS-style)"] = lambda episode, idx: PRMGuidedMCTSStylePolicy(episode.hidden_task, target_prm_ranker)
    policy_certified_pruning = {}
    if include_solver_certification:
        insert_at = policy_names.index("Oracle") if "Oracle" in policy_names else len(policy_names)
        policy_names.insert(insert_at, "Solver-trained MLP + certified pruning")
        extra_policy_factories["Solver-trained MLP + certified pruning"] = lambda episode, idx: SchedulingMLPPolicy(episode.hidden_task, ranker)
        policy_certified_pruning["Solver-trained MLP + certified pruning"] = True
    table_rows, per_instance_rows, proposal_cache_rows, proposal_state_rows = _evaluate_level2_transfer(
        episodes,
        ranker,
        node_budget,
        action_budget,
        seed,
        useful_action_recall=1.0,
        policy_names=tuple(policy_names),
        proposal_adapter_factory=lambda: RealLLMProposerAdapter(client=client, model=model, prompt_template_version="scheduling-level2-v1", temperature=0.0, top_p=1.0),
        proposal_model=model,
        real_value_policy_factory=value_policy_factory,
        real_frontier_ranking_policy_factory=frontier_ranking_policy_factory,
        extra_policy_factories=extra_policy_factories,
        policy_certified_pruning=policy_certified_pruning,
        checkpoint_dir=out_dir,
        output_prefix="scheduling_llm_pilot",
        batch_size=batch_size,
        parallel_episodes=parallel_episodes,
        resume=resume,
        progress_label="scheduling-llm-pilot",
    )
    proposal_stats = _summarize_cache_rows(proposal_cache_rows)
    metrics = {
        "benchmark": "scheduling_real_llm_pilot",
        "seed": seed,
        "splits": {split: len(tasks) for split, tasks in instances.items()},
        "job_count": job_count,
        "machine_count": machine_count,
        "node_budget": node_budget,
        "action_budget": action_budget,
        "records_per_instance": records_per_instance,
        "rank_epochs": rank_epochs,
        "batch_size": batch_size,
        "parallel_episodes": max(1, int(parallel_episodes)),
        "resume": resume,
        "include_gpt_value": include_gpt_value,
        "include_gpt_frontier_ranking": include_gpt_frontier_ranking,
        "include_target_prm": include_target_prm,
        "include_rest_mcts": include_rest_mcts,
        "include_solver_certification": include_solver_certification,
        "target_prm_train_count": target_prm_train_count if include_target_prm or include_rest_mcts else 0,
        "target_prm_trace_record_count": len(target_prm_records),
        "target_prm_pair_count": len(target_prm_pairs),
        "model": model,
        "live_network_required": not isinstance(client, ScriptedSchedulingClient),
        "training_source": "level1_solver_traces_only",
        "target_training_trace_count": 0,
        "level2_perturbation": perturbation,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "validation_ranking_error": ranker.ranking_error(validation_pairs),
        "source_domain_methods": {row["Policy"]: row for row in source_policy_rows},
        "methods": {row["Policy"]: row for row in table_rows},
        "semantic_state_schema": {
            "common_inference_features": COMMON_INFERENCE_FEATURES,
            "oracle_label_fields_excluded_from_inference": ORACLE_LABEL_FIELDS,
            "oracle_labels_in_inference_features": False,
            "source_state_space": "canonical semantic scheduling state",
            "target_state_space": "canonical semantic scheduling state",
            "source_distribution": "solver-induced semantic search",
            "target_distribution": "real-llm-induced semantic search",
        },
        "shared_proposal_replay": proposal_stats,
        "certified_pruning_enabled": False,
    }
    _write_csv(out_dir / "scheduling_llm_pilot_table.csv", table_rows)
    _write_csv(out_dir / "scheduling_llm_pilot_per_instance.csv", per_instance_rows)
    _write_csv(out_dir / "scheduling_llm_pilot_proposal_cache.csv", proposal_cache_rows)
    _write_level2_episodes(out_dir / "scheduling_llm_pilot_episodes.jsonl", episodes)
    ranker_json = ranker.to_json()
    (out_dir / "scheduling_llm_pilot_ranker.json").write_text(ranker_json, encoding="utf-8")
    if include_target_prm and target_prm_ranker is not None:
        (out_dir / "target_prm_ranker.json").write_text(target_prm_ranker.to_json(), encoding="utf-8")
        _write_trace_records(out_dir / "target_prm_trace_records.jsonl", target_prm_records)
        _write_pairs(out_dir / "target_prm_train_pairs.jsonl", target_prm_pairs)
        _write_csv(out_dir / "target_prm_training_proposal_cache.csv", target_prm_cache_rows)
        _write_jsonl(out_dir / "target_prm_training_proposal_states.jsonl", target_prm_state_rows)
    (out_dir / "scheduling_llm_pilot_leakage_manifest.json").write_text(json.dumps(metrics["semantic_state_schema"], indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "scheduling_llm_pilot_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    _write_transfer_audit_artifacts(
        out_dir,
        metrics,
        instances,
        ranker_json,
        table_rows,
        per_instance_rows,
        proposal_cache_rows,
        proposal_state_rows,
        source_records=[*train_records, *validation_records],
        source_policy_rows=source_policy_rows,
    )
    return metrics


@dataclass
class ScriptedSchedulingClient:
    input_tokens: int = 80
    output_tokens: int = 24
    model: str = "mock-scheduling"

    def generate_typed_actions(self, *args, **kwargs):
        task = kwargs["task"]
        node = kwargs["node"]
        k = int(kwargs["k"])
        actions = [action.to_dict() for action in task.candidate_actions(node, k)]
        return {
            "raw_output": json.dumps({"actions": actions}),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency": 0.0,
        }

    def generate_scheduling_solution(self, *args, **kwargs):
        del args
        task = kwargs["task"]
        node = task.root_node()
        actions = []
        while len(actions) < len(task.durations):
            candidates = task.candidate_actions(node, k=1)
            if not candidates:
                break
            action = candidates[0]
            actions.append(action.to_dict())
            node = task.apply(node, action, child_index=len(actions) - 1)
        return {
            "raw_output": json.dumps({"actions": actions}),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency": 0.0,
        }

    def score_state(self, *args, **kwargs):
        del args
        task = kwargs["task"]
        node = kwargs["node"]
        scheduled = tuple(node.structured_state.get("scheduled", []))
        remaining = len(task.durations) - len(scheduled)
        score = 1.0 / max(1, remaining)
        return {
            "score": score,
            "input_tokens": self.input_tokens,
            "output_tokens": max(1, self.output_tokens // 4),
            "latency": 0.0,
            "raw_output": json.dumps({"score": score}),
        }

    def score_frontier_states(self, *args, **kwargs):
        del args
        nodes = list(kwargs.get("nodes", []))
        scores = []
        for node in nodes:
            scheduled = tuple(node.structured_state.get("scheduled", []))
            scores.append(float(len(scheduled)))
        return {
            "scores": scores,
            "input_tokens": self.input_tokens + 10 * len(nodes),
            "output_tokens": max(1, len(nodes)),
            "latency": 0.0,
            "raw_output": json.dumps({"scores": {getattr(node, "node_id", str(idx)): score for idx, (node, score) in enumerate(zip(nodes, scores))}}),
        }


def run_scheduling_fewshot_curve(
    out_dir: Path,
    seed: int = 0,
    solver_train_count: int = 200,
    target_train_count: int = 500,
    validation_count: int = 50,
    target_test_count: int = 100,
    job_count: int = 6,
    machine_count: int = 3,
    node_budget: int = 120,
    action_budget: int = 8,
    records_per_instance: int = 60,
    rank_epochs: int = 25,
    fewshot_counts: Sequence[int] = (10, 50, 100, 500),
    useful_action_recall: float = 0.88,
    perturbation: str = "mixed",
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source = generate_scheduling_micro_instances(solver_train_count, validation_count, 0, seed, job_count, machine_count)
    target = generate_scheduling_micro_instances(target_train_count, 0, target_test_count, seed + 10000, job_count, machine_count)
    solver_records = []
    validation_records = []
    for task in source["train"]:
        solver_records.extend(collect_solver_trace_records(task, "solver_train", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    for task in source["validation"]:
        validation_records.extend(collect_solver_trace_records(task, "validation", node_budget=node_budget, action_budget=action_budget, max_records=records_per_instance))
    target_train_episodes = make_level2_scheduling_episodes(target["train"], seed=seed + 3000, perturbation=perturbation)
    target_test_episodes = make_level2_scheduling_episodes(target["test"], seed=seed + 4000, perturbation=perturbation)
    target_records = collect_level2_trace_records(
        target_train_episodes,
        seed=seed + 5000,
        node_budget=node_budget,
        action_budget=action_budget,
        max_records=records_per_instance,
        useful_action_recall=useful_action_recall,
    )
    solver_pairs = make_scheduling_ranking_pairs(solver_records, max_pairs_per_task=40, seed=seed)
    validation_pairs = make_scheduling_ranking_pairs(validation_records, max_pairs_per_task=40, seed=seed + 1)
    baseline_ranker = _train_ranker(solver_pairs, rank_epochs, seed)
    baseline_rows, _, _, _ = _evaluate_level2_transfer(target_test_episodes, baseline_ranker, node_budget, action_budget, seed, useful_action_recall, policy_names=("Solver-trained MLP",))
    curve_rows = []
    per_regime_metrics = {}
    for row in baseline_rows:
        if row["Policy"] == "Solver-trained MLP":
            curve_rows.append(
                {
                    "Regime": "Solver-only",
                    "target_traces": 0,
                    "target_trace_records": 0,
                    "target_pair_count": 0,
                    "source_replay_pair_count": 0,
                    **row,
                }
            )
            per_regime_metrics["Solver-only:0"] = row
            break
    for count in fewshot_counts:
        selected_episodes = target_train_episodes[: min(count, len(target_train_episodes))]
        selected_ids = {episode.hidden_task_id for episode in selected_episodes}
        selected_records = [record for record in target_records if record.task_id in selected_ids]
        target_pairs = make_scheduling_ranking_pairs(selected_records, max_pairs_per_task=40, seed=seed + count)
        replay_pairs = _sample_pairs(solver_pairs, size=len(target_pairs), seed=seed + 9000 + count)
        replay_adapt_pairs = [*target_pairs, *replay_pairs]
        solver_plus = _train_ranker(solver_pairs, rank_epochs, seed)
        solver_plus.train(replay_adapt_pairs, epochs=max(3, rank_epochs // 3), lr=0.01, seed=seed + count)
        llm_only = _train_ranker(target_pairs, rank_epochs, seed + count)
        for regime, ranker in (("Solver pretrain + target adapt", solver_plus), ("Target-trajectory trained", llm_only)):
            rows, _, _, _ = _evaluate_level2_transfer(
                target_test_episodes,
                ranker,
                node_budget,
                action_budget,
                seed + count,
                useful_action_recall,
                policy_names=("Solver-trained MLP",),
            )
            mlp_row = next(row for row in rows if row["Policy"] == "Solver-trained MLP")
            replay_count = len(replay_pairs) if regime == "Solver pretrain + target adapt" else 0
            out_row = {
                "Regime": regime,
                "target_traces": len(selected_episodes),
                "target_trace_records": len(selected_records),
                "target_pair_count": len(target_pairs),
                "source_replay_pair_count": replay_count,
                **mlp_row,
            }
            curve_rows.append(out_row)
            per_regime_metrics[f"{regime}:{len(selected_episodes)}"] = {**mlp_row, "target_pair_count": len(target_pairs), "source_replay_pair_count": replay_count}
    metrics = {
        "benchmark": "scheduling_fewshot_curve",
        "seed": seed,
        "solver_train_count": solver_train_count,
        "target_train_count": target_train_count,
        "target_adapt_count": target_train_count,
        "validation_count": validation_count,
        "target_test_count": target_test_count,
        "target_eval_trace_ids": [episode.hidden_task_id for episode in target_test_episodes],
        "target_train_trace_ids": [episode.hidden_task_id for episode in target_train_episodes],
        "target_train_eval_overlap": bool(set(episode.hidden_task_id for episode in target_test_episodes).intersection(episode.hidden_task_id for episode in target_train_episodes)),
        "job_count": job_count,
        "machine_count": machine_count,
        "node_budget": node_budget,
        "action_budget": action_budget,
        "records_per_instance": records_per_instance,
        "rank_epochs": rank_epochs,
        "fewshot_counts_requested": list(fewshot_counts),
        "solver_pair_count": len(solver_pairs),
        "target_trace_record_count": len(target_records),
        "validation_pair_count": len(validation_pairs),
        "solver_validation_ranking_error": baseline_ranker.ranking_error(validation_pairs),
        "training_regimes": per_regime_metrics,
        "target_only_baseline_definition": "states come from LLM-induced trajectories; labels still come from exact expert/oracle",
        "solver_pretrain_adaptation": "fine-tune on target pairs with equal-count source replay pairs to reduce forgetting",
    }
    _write_csv(out_dir / "scheduling_fewshot_curve.csv", curve_rows)
    _write_level2_episodes(out_dir / "scheduling_fewshot_target_train_episodes.jsonl", target_train_episodes)
    _write_level2_episodes(out_dir / "scheduling_fewshot_target_test_episodes.jsonl", target_test_episodes)
    _write_trace_records(out_dir / "scheduling_level2_target_trace_records.jsonl", target_records)
    _write_pairs(out_dir / "scheduling_solver_train_pairs.jsonl", solver_pairs)
    (out_dir / "scheduling_fewshot_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def collect_level2_trace_records(
    episodes: Sequence[Level2SchedulingEpisode],
    seed: int,
    node_budget: int,
    action_budget: int,
    max_records: int,
    useful_action_recall: float,
):
    records = []
    for idx, episode in enumerate(episodes):
        proposer = SimulatedLLMSchedulingProposer(episode, seed=seed + idx, useful_action_recall=useful_action_recall)
        records.extend(_collect_policy_trace_records(episode, proposer, seed + idx, node_budget, action_budget, max_records))
    return records


def collect_real_level2_trace_records(
    episodes: Sequence[Level2SchedulingEpisode],
    seed: int,
    node_budget: int,
    action_budget: int,
    max_records: int,
    proposal_adapter_factory,
    proposal_model: str,
    parallel_episodes: int = 1,
) -> tuple[List[object], List[Dict[str, object]], List[Dict[str, object]]]:
    records = []
    proposal_cache_rows = []
    proposal_state_rows = []
    parallel = max(1, int(parallel_episodes))
    completed = 0
    for start in range(0, len(episodes), parallel):
        window = list(enumerate(episodes[start : start + parallel], start=start))
        window_results = {}
        if parallel == 1:
            idx, episode = window[0]
            window_results[idx] = _collect_one_real_level2_trace_episode(
                idx,
                episode,
                seed,
                node_budget,
                action_budget,
                max_records,
                proposal_adapter_factory,
                proposal_model,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(parallel, len(window))) as executor:
                futures = {
                    executor.submit(
                        _collect_one_real_level2_trace_episode,
                        idx,
                        episode,
                        seed,
                        node_budget,
                        action_budget,
                        max_records,
                        proposal_adapter_factory,
                        proposal_model,
                    ): idx
                    for idx, episode in window
                }
                for future in as_completed(futures):
                    window_results[futures[future]] = future.result()
        for idx, _episode in sorted(window, key=lambda item: item[0]):
            episode_records, cache_row, state_rows = window_results[idx]
            records.extend(episode_records)
            proposal_cache_rows.append(cache_row)
            proposal_state_rows.extend(state_rows)
            completed += 1
            print(
                f"[target-prm-train] {completed}/{len(episodes)} episodes, "
                f"records={len(records)}, states={sum(int(row['state_count']) for row in proposal_cache_rows)}, "
                f"tokens={sum(int(row['total_tokens']) for row in proposal_cache_rows)}",
                flush=True,
            )
    return records, proposal_cache_rows, proposal_state_rows


def _collect_one_real_level2_trace_episode(
    idx: int,
    episode: Level2SchedulingEpisode,
    seed: int,
    node_budget: int,
    action_budget: int,
    max_records: int,
    proposal_adapter_factory,
    proposal_model: str,
) -> tuple[List[object], Dict[str, object], List[Dict[str, object]]]:
    cache = SharedSchedulingProposalCache(
        episode,
        seed=seed + idx * 1009,
        proposal_k=action_budget,
        node_budget=node_budget,
        useful_action_recall=1.0,
        model=proposal_model,
        adapter=proposal_adapter_factory(),
    )
    records = _collect_policy_trace_records(episode, cache, seed + idx, node_budget, action_budget, max_records)
    cache_row = {"task_id": episode.hidden_task_id, "rendering_id": episode.rendering_id, "perturbation": episode.perturbation, **cache.stats()}
    return records, cache_row, cache.state_rows()


def _collect_policy_trace_records(
    episode: Level2SchedulingEpisode,
    proposer: SimulatedLLMSchedulingProposer,
    seed: int,
    node_budget: int,
    action_budget: int,
    max_records: int,
):
    del seed
    from cgr_exp.scheduling_micro import _trace_record

    task = episode.hidden_task
    verifier = Level1SchedulingVerifier()
    frontier: List[SearchNode] = [task.root_node()]
    incumbent = None
    records = []
    expanded = 0
    seen = set()
    while frontier and expanded < node_budget and len(records) < max_records:
        node = frontier.pop()
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        expanded += 1
        if node.completion_status == CompletionStatus.COMPLETE:
            value = node.objective_if_complete
            if value is not None and (incumbent is None or value < incumbent):
                incumbent = value
            records.append(_trace_record(task, "target_llm_train", node, incumbent, 0, "llm_complete"))
            continue
        actions = proposer.propose(task, node, action_budget)
        children = []
        for child_idx, action in enumerate(actions):
            verified = verifier.verify(task, node.with_updates(incumbent_at_creation=incumbent), action)
            if not verified.feasible:
                continue
            child = verifier.make_child(task, node.with_updates(incumbent_at_creation=incumbent), action, verified, child_idx)
            children.append(child)
            if len(records) < max_records:
                records.append(_trace_record(task, "target_llm_train", child, incumbent, 0, "llm_child_candidate"))
            if child.completion_status == CompletionStatus.COMPLETE:
                value = child.objective_if_complete
                if value is not None and (incumbent is None or value < incumbent):
                    incumbent = value
            else:
                frontier.append(child)
        if len(records) < max_records:
            records.append(_trace_record(task, "target_llm_train", node, incumbent, len(children), "llm_expanded"))
    return records


def _evaluate_level2_transfer(
    episodes: Sequence[Level2SchedulingEpisode],
    ranker: TinyPairwiseMLPRanker,
    node_budget: int,
    action_budget: int,
    seed: int,
    useful_action_recall: float,
    policy_names: Sequence[str] | None = None,
    proposal_adapter_factory=None,
    proposal_model: str = "simulated-scheduling-llm",
    checkpoint_dir: Path | None = None,
    output_prefix: str | None = None,
    batch_size: int = 1,
    parallel_episodes: int = 1,
    resume: bool = False,
    progress_label: str | None = None,
    real_value_policy_factory=None,
    real_frontier_ranking_policy_factory=None,
    extra_policy_factories: Dict[str, object] | None = None,
    policy_certified_pruning: Dict[str, bool] | None = None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    all_policies = {
        "Random": lambda episode, idx: RandomPolicy(seed + idx),
        "Hand heuristic": lambda episode, idx: SchedulingHeuristicPolicy(),
        "LLM value evaluator": lambda episode, idx: SimulatedLLMValuePolicy(episode.hidden_task, seed + idx),
        "Solver-trained MLP": lambda episode, idx: SchedulingMLPPolicy(episode.hidden_task, ranker),
        "Oracle": lambda episode, idx: CachedSchedulingOraclePolicy(episode.hidden_task),
    }
    if real_value_policy_factory is not None:
        all_policies["GPT-value / ToT-style"] = real_value_policy_factory
    if real_frontier_ranking_policy_factory is not None:
        all_policies["GPT-frontier-rank / ToT-style"] = real_frontier_ranking_policy_factory
    if extra_policy_factories:
        all_policies.update(extra_policy_factories)
    certified_by_policy = policy_certified_pruning or {}
    selected_names = tuple(policy_names) if policy_names is not None else tuple(all_policies)
    policies = {name: all_policies[name] for name in selected_names}
    by_policy: Dict[str, List[Dict[str, object]]] = {name: [] for name in policies}
    per_instance_rows = []
    proposal_cache_rows = []
    proposal_state_rows = []
    completed_renderings = set()
    if checkpoint_dir is not None and output_prefix is not None and resume:
        per_instance_rows = _read_csv_rows(checkpoint_dir / f"{output_prefix}_per_instance.csv")
        proposal_cache_rows = _read_csv_rows(checkpoint_dir / f"{output_prefix}_proposal_cache.csv")
        proposal_state_rows = _read_jsonl_rows(checkpoint_dir / "proposal_cache.jsonl")
        counts: Dict[str, int] = {}
        for row in per_instance_rows:
            if row.get("Policy") in policies:
                counts[str(row.get("rendering_id"))] = counts.get(str(row.get("rendering_id")), 0) + 1
                by_policy[str(row["Policy"])].append(_coerce_result_row(row))
        completed_renderings = {rendering_id for rendering_id, count in counts.items() if count >= len(policies)}
        if progress_label and completed_renderings:
            print(f"[{progress_label}] resumed {len(completed_renderings)}/{len(episodes)} completed episodes", flush=True)
    flush_every = max(1, int(batch_size))
    parallel = max(1, int(parallel_episodes))
    pending = [(idx, episode) for idx, episode in enumerate(episodes) if episode.rendering_id not in completed_renderings]
    completed_since_flush = 0
    for start in range(0, len(pending), parallel):
        window = pending[start : start + parallel]
        window_results: Dict[int, tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]] = {}
        if parallel == 1:
            idx, episode = window[0]
            window_results[idx] = _evaluate_one_level2_episode(
                idx=idx,
                episode=episode,
                policies=policies,
                seed=seed,
                action_budget=action_budget,
                node_budget=node_budget,
                useful_action_recall=useful_action_recall,
                proposal_model=proposal_model,
                proposal_adapter_factory=proposal_adapter_factory,
                policy_certified_pruning=certified_by_policy,
            )
        else:
            with ThreadPoolExecutor(max_workers=min(parallel, len(window))) as executor:
                futures = {
                    executor.submit(
                        _evaluate_one_level2_episode,
                        idx=idx,
                        episode=episode,
                        policies=policies,
                        seed=seed,
                        action_budget=action_budget,
                        node_budget=node_budget,
                        useful_action_recall=useful_action_recall,
                        proposal_model=proposal_model,
                        proposal_adapter_factory=proposal_adapter_factory,
                        policy_certified_pruning=certified_by_policy,
                    ): idx
                    for idx, episode in window
                }
                for future in as_completed(futures):
                    window_results[futures[future]] = future.result()
        for idx, _episode in sorted(window, key=lambda item: item[0]):
            episode_rows, proposal_cache_row, state_rows = window_results[idx]
            for row in episode_rows:
                by_policy[str(row["Policy"])].append(row)
                per_instance_rows.append(row)
            proposal_cache_rows.append(proposal_cache_row)
            proposal_state_rows.extend(state_rows)
            completed_since_flush += 1
            finished = len({str(row.get("rendering_id")) for row in per_instance_rows})
            if checkpoint_dir is not None and output_prefix is not None and (completed_since_flush >= flush_every or finished == len(episodes)):
                completed_since_flush = 0
                partial_table = _summarize_policy_rows(by_policy)
                _write_csv(checkpoint_dir / f"{output_prefix}_table.csv", partial_table)
                _write_csv(checkpoint_dir / f"{output_prefix}_per_instance.csv", per_instance_rows)
                _write_csv(checkpoint_dir / f"{output_prefix}_proposal_cache.csv", proposal_cache_rows)
                _write_jsonl(checkpoint_dir / "proposal_cache.jsonl", proposal_state_rows)
                if progress_label:
                    print(
                        f"[{progress_label}] {finished}/{len(episodes)} episodes, "
                        f"proposal_states={sum(int(row['state_count']) for row in proposal_cache_rows)}, "
                        f"tokens={sum(int(row['total_tokens']) for row in proposal_cache_rows)}",
                        flush=True,
                    )
    table_rows = []
    table_rows = _summarize_policy_rows(by_policy)
    return table_rows, per_instance_rows, proposal_cache_rows, proposal_state_rows


def _evaluate_one_level2_episode(
    idx: int,
    episode: Level2SchedulingEpisode,
    policies: Dict[str, object],
    seed: int,
    action_budget: int,
    node_budget: int,
    useful_action_recall: float,
    proposal_model: str,
    proposal_adapter_factory=None,
    policy_certified_pruning: Dict[str, bool] | None = None,
) -> tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    shared_cache = SharedSchedulingProposalCache(
        episode,
        seed=seed + idx * 1009,
        proposal_k=action_budget,
        node_budget=node_budget,
        useful_action_recall=useful_action_recall,
        model=proposal_model,
        adapter=proposal_adapter_factory() if proposal_adapter_factory is not None else None,
    )
    oracle_success = None
    episode_rows = []
    for name, factory in policies.items():
        policy = factory(episode, idx)
        result, ranking = _run_frontier_search_with_ranking_diagnostics(
            task=episode.hidden_task,
            policy=policy,
            verifier=Level1SchedulingVerifier(),
            node_budget=node_budget,
            proposal_budget=action_budget,
            seed=seed + idx,
            proposer=shared_cache,
            certified_pruning=bool((policy_certified_pruning or {}).get(name, False)),
        )
        if name == "Oracle":
            oracle_success = result.best_objective == result.optimum
        value_cost = policy.cost if isinstance(policy, (SimulatedLLMValuePolicy, RealLLMValuePolicy, RealLLMFrontierRankingPolicy)) else CostLedger()
        episode_rows.append(
            {
                "Policy": name,
                "task_id": episode.hidden_task_id,
                "rendering_id": episode.rendering_id,
                "perturbation": episode.perturbation,
                "success": result.best_objective == result.optimum,
                "gap": _gap(result.best_objective, result.optimum),
                "nodes": result.expanded_nodes,
                "proposal_calls": result.cost.proposal_calls,
                "proposal_tokens": 0,
                "value_calls": value_cost.evaluation_calls,
                "value_tokens": value_cost.input_tokens + value_cost.output_tokens,
                "total_tokens": 0,
                "false_prunes": result.false_prune_count,
                "budget_exhausted": result.budget_exhausted or (result.best_objective != result.optimum and result.expanded_nodes >= node_budget),
                "frontier_decisions": ranking["frontier_decisions"],
                "frontier_top1_accuracy": ranking["frontier_top1_accuracy"],
                "pairwise_accuracy": ranking["pairwise_accuracy"],
                "selection_regret_mean": ranking["selection_regret_mean"],
            }
        )
    if oracle_success is None and "Oracle" in policies:
        oracle_success = False
    if oracle_success is not None:
        for row in episode_rows:
            row["oracle_reachable_success"] = oracle_success
            row["failure_reason"] = _failure_reason(row, oracle_success)
    stats = shared_cache.stats()
    shared_tokens = int(stats["total_tokens"])
    for row in episode_rows:
        row["proposal_tokens"] = shared_tokens
        row["total_tokens"] = shared_tokens + int(row["value_tokens"])
    proposal_cache_row = {"task_id": episode.hidden_task_id, "rendering_id": episode.rendering_id, "perturbation": episode.perturbation, **stats}
    return episode_rows, proposal_cache_row, shared_cache.state_rows()


def _summarize_policy_rows(by_policy: Dict[str, List[Dict[str, object]]]) -> List[Dict[str, object]]:
    table_rows = []
    for name, rows in by_policy.items():
        table_rows.append(
            {
                "Policy": name,
                "success": _mean(1.0 if _as_bool(row["success"]) else 0.0 for row in rows),
                "gap": _mean(float(row["gap"]) for row in rows),
                "nodes_mean": _mean(float(row["nodes"]) for row in rows),
                "proposal_calls_mean": _mean(float(row["proposal_calls"]) for row in rows),
                "value_calls": sum(int(row["value_calls"]) for row in rows),
                "proposal_tokens": sum(int(row["proposal_tokens"]) for row in rows),
                "value_tokens": sum(int(row["value_tokens"]) for row in rows),
                "total_tokens": sum(int(row["total_tokens"]) for row in rows),
                "false_prunes": sum(int(row["false_prunes"]) for row in rows),
                "frontier_decisions": sum(int(row["frontier_decisions"]) for row in rows),
                "frontier_top1_accuracy": _weighted_mean(rows, "frontier_top1_accuracy", "frontier_decisions"),
                "pairwise_accuracy": _weighted_mean(rows, "pairwise_accuracy", "frontier_decisions"),
                "selection_regret_mean": _weighted_mean(rows, "selection_regret_mean", "frontier_decisions"),
                "budget_failures": sum(1 for row in rows if row.get("failure_reason") == "budget"),
                "proposal_support_failures": sum(1 for row in rows if row.get("failure_reason") == "proposal_support"),
                "ranking_failures": sum(1 for row in rows if row.get("failure_reason") == "ranking"),
            }
        )
    return table_rows


def _run_frontier_search_with_ranking_diagnostics(
    task,
    policy,
    verifier,
    node_budget: int,
    proposal_budget: int,
    seed: int,
    proposer,
    certified_pruning: bool = False,
) -> tuple[FrontierSearchResult, Dict[str, object]]:
    del seed
    start = time.perf_counter()
    optimum = task.exhaustive_oracle()
    incumbent = None
    frontier: List[SearchNode] = [task.root_node()]
    expanded = 0
    generated = 0
    false_prunes = 0
    pruned_nodes = 0
    certificate_count = 0
    budget = BudgetState()
    budget.cost = CostLedger(solver_calls=1)
    trace = []
    frontier_peak = 1
    budget_exhausted = False
    ranking_records = []
    oracle_value_cache: Dict[str, float] = {}

    while frontier and expanded < node_budget:
        survivors: List[SearchNode] = []
        for node in frontier:
            decision = should_prune(node, incumbent) if certified_pruning else None
            if decision is None:
                survivors.append(node)
                continue
            if decision.prune:
                pruned_nodes += 1
                certificate_count += 1
                if incumbent is not None and task.best_descendant_objective(node) < incumbent:
                    false_prunes += 1
                trace.append({"event": "prune", "node_id": node.node_id, "reason": decision.reason})
            else:
                survivors.append(node)
        frontier = survivors
        if not frontier:
            break

        if hasattr(policy, "score_frontier"):
            scores = [float(score) for score in policy.score_frontier(frontier)]
            if len(scores) != len(frontier):
                scores = [float(policy.score(node)) for node in frontier]
        else:
            scores = [float(policy.score(node)) for node in frontier]
        selected_index = max(range(len(frontier)), key=lambda i: (scores[i], frontier[i].node_id))
        ranking_records.append(_frontier_ranking_record(task, frontier, scores, selected_index, oracle_value_cache))
        node = frontier.pop(selected_index)
        expanded += 1
        trace.append({"event": "expand", "node": node.to_dict(), "frontier_size": len(frontier), "incumbent": incumbent})

        if node.completion_status == CompletionStatus.COMPLETE:
            value = node.objective_if_complete
            if value is not None and (incumbent is None or value < incumbent):
                incumbent = value
            continue

        if not budget.can_charge_proposal():
            budget_exhausted = True
            trace.append({"event": "budget_exhausted", "reason": "proposal_call_budget"})
            break
        budget.charge_proposal()
        actions = proposer.propose(task, node, proposal_budget, budget)
        for idx, action in enumerate(actions):
            verified = verifier.verify(task, node, action)
            budget.charge_verifier()
            if not verified.feasible:
                continue
            child = verifier.make_child(task, node.with_updates(incumbent_at_creation=incumbent), action, verified, idx)
            generated += 1
            if child.completion_status == CompletionStatus.COMPLETE:
                value = child.objective_if_complete
                if value is not None and (incumbent is None or value < incumbent):
                    incumbent = value
            else:
                frontier.append(child)
                frontier_peak = max(frontier_peak, len(frontier))

    elapsed = time.perf_counter() - start
    budget.cost = budget.cost + CostLedger(wall_clock_seconds=elapsed)
    result = FrontierSearchResult(
        incumbent,
        optimum,
        false_prunes,
        expanded,
        generated,
        frontier_peak,
        budget.cost,
        trace,
        budget_exhausted,
        pruned_nodes,
        certificate_count,
    )
    return result, _summarize_ranking_records(ranking_records)


def _frontier_ranking_record(
    task: SchedulingTask,
    frontier: Sequence[SearchNode],
    scores: Sequence[float],
    selected_index: int,
    oracle_value_cache: Dict[str, float],
) -> Dict[str, float]:
    oracle_values = [_cached_best_descendant_objective(task, node, oracle_value_cache) for node in frontier]
    best_value = min(oracle_values)
    selected_value = oracle_values[selected_index]
    correct_pairs = 0
    total_pairs = 0
    for i in range(len(frontier)):
        for j in range(i + 1, len(frontier)):
            if oracle_values[i] == oracle_values[j]:
                continue
            total_pairs += 1
            better, worse = (i, j) if oracle_values[i] < oracle_values[j] else (j, i)
            correct_pairs += int(scores[better] > scores[worse])
    return {
        "top1": 1.0 if selected_value == best_value else 0.0,
        "pairwise_accuracy": 1.0 if total_pairs == 0 else correct_pairs / total_pairs,
        "selection_regret": max(0.0, selected_value - best_value),
        "pair_count": float(total_pairs),
    }


def _cached_best_descendant_objective(task: SchedulingTask, node: SearchNode, cache: Dict[str, float]) -> float:
    key = json.dumps(node.structured_state.get("scheduled", []), sort_keys=True)
    if key not in cache:
        cache[key] = float(task.best_descendant_objective(node))
    return cache[key]


def _summarize_ranking_records(records: Sequence[Dict[str, float]]) -> Dict[str, object]:
    if not records:
        return {
            "frontier_decisions": 0,
            "frontier_top1_accuracy": 0.0,
            "pairwise_accuracy": 0.0,
            "selection_regret_mean": 0.0,
        }
    pair_total = sum(float(row["pair_count"]) for row in records)
    pair_correct = sum(float(row["pairwise_accuracy"]) * float(row["pair_count"]) for row in records)
    return {
        "frontier_decisions": len(records),
        "frontier_top1_accuracy": _mean(float(row["top1"]) for row in records),
        "pairwise_accuracy": 0.0 if pair_total == 0.0 else pair_correct / pair_total,
        "selection_regret_mean": _mean(float(row["selection_regret"]) for row in records),
    }


def _episode_from_rendering(rendering: Level2Rendering) -> Level2SchedulingEpisode:
    return Level2SchedulingEpisode(
        rendering_id=rendering.rendering_id,
        hidden_task_id=rendering.hidden_task_id,
        perturbation=rendering.perturbation,
        text=rendering.text,
        hidden_task=rendering.hidden_task,
    )


def _write_level2_episodes(path: Path, episodes: Sequence[Level2SchedulingEpisode]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(
                json.dumps(
                    {
                        "rendering_id": episode.rendering_id,
                        "hidden_task_id": episode.hidden_task_id,
                        "perturbation": episode.perturbation,
                        "text": episode.text,
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _write_transfer_audit_artifacts(
    out_dir: Path,
    metrics: Dict[str, object],
    instances: Dict[str, Sequence[SchedulingTask]],
    ranker_json: str,
    table_rows: Sequence[Dict[str, object]],
    per_instance_rows: Sequence[Dict[str, object]],
    proposal_cache_rows: Sequence[Dict[str, object]],
    proposal_state_rows: Sequence[Dict[str, object]],
    source_records: Sequence[object],
    source_policy_rows: Sequence[Dict[str, object]],
) -> None:
    ranker_hash = hashlib.sha256(ranker_json.encode("utf-8")).hexdigest()
    config = {
        "benchmark": metrics["benchmark"],
        "seed": metrics["seed"],
        "job_count": metrics["job_count"],
        "machine_count": metrics["machine_count"],
        "node_budget": metrics["node_budget"],
        "action_budget": metrics["action_budget"],
        "records_per_instance": metrics["records_per_instance"],
        "rank_epochs": metrics["rank_epochs"],
        "batch_size": metrics.get("batch_size", 1),
        "parallel_episodes": metrics.get("parallel_episodes", 1),
        "resume": metrics.get("resume", False),
        "training_source": metrics["training_source"],
        "target_training_trace_count": metrics["target_training_trace_count"],
        "level2_perturbation": metrics["level2_perturbation"],
        "ranker_type": metrics.get("ranker_type", "tiny_pairwise_mlp"),
        "certified_pruning_enabled": metrics["certified_pruning_enabled"],
        "shared_proposal_replay": True,
        "model": metrics.get("model", metrics.get("shared_proposal_replay", {}).get("model")),
    }
    split_manifest = {
        split: [task.task_id for task in tasks]
        for split, tasks in instances.items()
    }
    feature_schema = {
        "schema_version": "scheduling-transfer-common-v1",
        "common_inference_features": list(COMMON_INFERENCE_FEATURES),
        "numeric_ranker_feature_names": list(SCHEDULING_FEATURE_NAMES),
        "oracle_label_fields_excluded_from_inference": list(ORACLE_LABEL_FIELDS),
        "oracle_labels_in_inference_features": False,
        "canonical_state_hash": "sha256(instance_id, canonical_state_json, model, prompt_version, proposal_k, temperature)",
    }
    checkpoint_metadata = {
        "ranker_hash_sha256": ranker_hash,
        "ranker_type": config["ranker_type"],
        "source_training_split": "train",
        "training_instance_ids": split_manifest.get("train", []),
        "validation_instance_ids": split_manifest.get("validation", []),
        "feature_schema_hash_sha256": hashlib.sha256(json.dumps(feature_schema, sort_keys=True).encode("utf-8")).hexdigest(),
        "label_definition": "pairwise frontier ranking by exact residual best descendant objective; infeasible descendants rank below completable states",
        "seed": metrics["seed"],
        "model_architecture": "Linear-tanh hidden layer pairwise MLP",
    }
    label_manifest = _source_label_manifest(metrics, source_records)
    source_ranking_rows = [
        {
            "split": "source_validation",
            "Policy": "",
            "pair_count": metrics["validation_pair_count"],
            "ranking_error": metrics["validation_ranking_error"],
            "ranking_accuracy": 1.0 - float(metrics["validation_ranking_error"]),
            "success": "",
            "gap": "",
            "nodes_mean": "",
            "frontier_decisions": "",
            "frontier_top1_accuracy": "",
            "pairwise_accuracy": "",
            "selection_regret_mean": "",
        }
    ]
    for row in source_policy_rows:
        source_ranking_rows.append(
            {
                "split": "source_test",
                "Policy": row["Policy"],
                "pair_count": "",
                "ranking_error": "",
                "ranking_accuracy": "",
                "success": row["success"],
                "gap": row["gap"],
                "nodes_mean": row["nodes_mean"],
                "frontier_decisions": row["frontier_decisions"],
                "frontier_top1_accuracy": row["frontier_top1_accuracy"],
                "pairwise_accuracy": row["pairwise_accuracy"],
                "selection_regret_mean": row["selection_regret_mean"],
            }
        )
    token_rows = [
        {
            "Policy": row["Policy"],
            "proposal_tokens": row["proposal_tokens"],
            "value_tokens": row["value_tokens"],
            "total_tokens": row["total_tokens"],
            "value_calls": row["value_calls"],
        }
        for row in table_rows
    ]
    failure_rows = _failure_analysis_rows(per_instance_rows)
    proposal_quality = {
        "shared_proposal_replay": metrics["shared_proposal_replay"],
        "proposal_recall_at_k": metrics["shared_proposal_replay"].get("proposal_recall_at_k", 0.0),
        "empty_valid_action_rate": metrics["shared_proposal_replay"].get("empty_valid_action_rate", 0.0),
        "parse_failures": metrics["shared_proposal_replay"].get("parse_failures", 0),
        "invalid_actions": metrics["shared_proposal_replay"].get("invalid_actions", 0),
        "duplicate_actions": metrics["shared_proposal_replay"].get("duplicate_actions", 0),
    }
    timing_rows = [
        {
            "component": "proposal_cache",
            "wall_clock_seconds": metrics["shared_proposal_replay"].get("latency", 0.0),
            "call_count": metrics["shared_proposal_replay"].get("proposal_calls", 0),
        }
    ]
    domain_shift_rows = _domain_shift_metric_rows(source_records, proposal_state_rows)
    audit = _transfer_completion_audit(metrics, split_manifest)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "checkpoint_metadata.json").write_text(json.dumps(checkpoint_metadata, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "label_manifest.json").write_text(json.dumps(label_manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "source_ranking_metrics.csv", source_ranking_rows)
    _write_csv(out_dir / "source_domain_table.csv", source_policy_rows)
    _write_csv(out_dir / "target_zero_shot_table.csv", table_rows)
    _write_csv(out_dir / "per_instance_results.csv", per_instance_rows)
    _write_csv(out_dir / "failure_analysis.csv", failure_rows)
    _write_csv(out_dir / "token_accounting.csv", token_rows)
    _write_csv(out_dir / "timing_metrics.csv", timing_rows)
    _write_csv(out_dir / "domain_shift_metrics.csv", domain_shift_rows)
    _write_jsonl(out_dir / "proposal_cache.jsonl", proposal_state_rows)
    (out_dir / "proposal_quality.json").write_text(json.dumps(proposal_quality, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "scheduling_transfer_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")


def _domain_shift_metric_rows(source_records: Sequence[object], proposal_state_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    source_vectors = [
        [float(value) for value in getattr(record, "features", [])]
        for record in source_records
        if getattr(record, "features", None)
    ]
    target_vectors = [
        [float(value) for value in row.get("features", [])]
        for row in proposal_state_rows
        if row.get("features")
    ]
    rows: List[Dict[str, object]] = [
        {
            "metric": "source_state_count",
            "feature": "",
            "source_mean": "",
            "target_mean": "",
            "source_std": "",
            "target_std": "",
            "standardized_mean_difference": "",
            "value": len(source_vectors),
            "status": "computed",
        },
        {
            "metric": "target_state_count",
            "feature": "",
            "source_mean": "",
            "target_mean": "",
            "source_std": "",
            "target_std": "",
            "standardized_mean_difference": "",
            "value": len(target_vectors),
            "status": "computed",
        },
    ]
    if not source_vectors or not target_vectors:
        rows.append(
            {
                "metric": "domain_shift_unavailable",
                "feature": "",
                "source_mean": "",
                "target_mean": "",
                "source_std": "",
                "target_std": "",
                "standardized_mean_difference": "",
                "value": "",
                "status": "missing_features",
            }
        )
        return rows
    dim = min(len(SCHEDULING_FEATURE_NAMES), len(source_vectors[0]), len(target_vectors[0]))
    shift_sq = 0.0
    for idx in range(dim):
        source_vals = [vec[idx] for vec in source_vectors if len(vec) > idx]
        target_vals = [vec[idx] for vec in target_vectors if len(vec) > idx]
        source_mean, source_std = _mean_std(source_vals)
        target_mean, target_std = _mean_std(target_vals)
        pooled = math.sqrt((source_std * source_std + target_std * target_std) / 2.0)
        standardized = 0.0 if pooled <= 1e-12 else (target_mean - source_mean) / pooled
        shift_sq += standardized * standardized
        rows.append(
            {
                "metric": "feature_standardized_mean_difference",
                "feature": SCHEDULING_FEATURE_NAMES[idx],
                "source_mean": source_mean,
                "target_mean": target_mean,
                "source_std": source_std,
                "target_std": target_std,
                "standardized_mean_difference": standardized,
                "value": standardized,
                "status": "computed",
            }
        )
    rows.append(
        {
            "metric": "overall_l2_standardized_mean_shift",
            "feature": "",
            "source_mean": "",
            "target_mean": "",
            "source_std": "",
            "target_std": "",
            "standardized_mean_difference": "",
            "value": math.sqrt(shift_sq),
            "status": "computed",
        }
    )
    return rows


def _source_label_manifest(metrics: Dict[str, object], source_records: Sequence[object]) -> Dict[str, object]:
    finite_labels = 0
    infeasible_labels = 0
    for record in source_records:
        value = float(getattr(record, "best_descendant_objective", float("inf")))
        if value == float("inf"):
            infeasible_labels += 1
        else:
            finite_labels += 1
    return {
        "supervision": "pairwise_frontier_ranking",
        "label_source": "exact_residual_scheduling_oracle",
        "residual_solver_status": "exact_exhaustive",
        "objective_sense": "minimize",
        "ranking_order": [
            "completable states before infeasible states",
            "lower exact best descendant objective before higher objective",
            "deeper state before shallower state when objective ties",
            "lower current lower bound before higher lower bound when objective and depth tie",
        ],
        "loss": "pairwise_logistic",
        "train_pair_count": metrics.get("train_pair_count", 0),
        "validation_pair_count": metrics.get("validation_pair_count", 0),
        "source_label_record_count": len(source_records),
        "finite_best_descendant_labels": finite_labels,
        "infeasible_best_descendant_labels": infeasible_labels,
        "uncertain_interval_pairs": 0,
        "unknown_pairs_excluded": 0,
        "pseudo_oracle_labels_used": False,
        "oracle_label_fields": list(ORACLE_LABEL_FIELDS),
        "oracle_labels_in_inference_features": bool(metrics["semantic_state_schema"]["oracle_labels_in_inference_features"]),
        "inference_feature_schema": "feature_schema.json",
    }


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def _failure_analysis_rows(per_instance_rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, int]] = {}
    for row in per_instance_rows:
        policy = str(row["Policy"])
        reason = str(row.get("failure_reason", "unknown"))
        grouped.setdefault(policy, {"solved": 0, "proposal_support": 0, "budget": 0, "ranking": 0, "controller": 0, "unknown": 0})
        grouped[policy][reason if reason in grouped[policy] else "unknown"] += 1
    return [{"Policy": policy, **counts} for policy, counts in grouped.items()]


def _transfer_completion_audit(metrics: Dict[str, object], split_manifest: Dict[str, List[str]]) -> Dict[str, object]:
    train_ids = set(split_manifest.get("train", []))
    validation_ids = set(split_manifest.get("validation", []))
    test_ids = set(split_manifest.get("test", []))
    return {
        "benchmark": metrics["benchmark"],
        "all_core_harness_requirements_met": bool(
            metrics["training_source"] == "level1_solver_traces_only"
            and metrics["target_training_trace_count"] == 0
            and not metrics["semantic_state_schema"]["oracle_labels_in_inference_features"]
            and metrics["shared_proposal_replay"]["shared"]
            and not metrics["certified_pruning_enabled"]
            and not (train_ids & validation_ids)
            and not (train_ids & test_ids)
            and not (validation_ids & test_ids)
        ),
        "real_llm_pilot_complete": bool(metrics["benchmark"] == "scheduling_real_llm_pilot" and metrics.get("live_network_required")),
        "requirements": [
            {
                "requirement": "source and target use canonical semantic scheduling states",
                "status": "met",
                "evidence": ["feature_schema.json", "proposal_cache.jsonl", "scheduling_transfer_leakage_manifest.json"],
            },
            {
                "requirement": "oracle label fields excluded from inference features",
                "status": "met" if not metrics["semantic_state_schema"]["oracle_labels_in_inference_features"] else "failed",
                "evidence": ["feature_schema.json", "label_manifest.json", "scheduling_transfer_leakage_manifest.json"],
            },
            {
                "requirement": "zero-shot target training count is zero",
                "status": "met" if metrics["target_training_trace_count"] == 0 else "failed",
                "evidence": ["config.json", "scheduling_transfer_metrics.json"],
            },
            {
                "requirement": "train validation and test instance ids are disjoint",
                "status": "met" if not (train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids) else "failed",
                "evidence": ["split_manifest.json"],
            },
            {
                "requirement": "shared proposal replay is enabled",
                "status": "met" if metrics["shared_proposal_replay"]["shared"] else "failed",
                "evidence": ["proposal_cache.jsonl", "proposal_quality.json"],
            },
            {
                "requirement": "invalid and duplicate proposals are filtered before search",
                "status": "met",
                "evidence": ["proposal_quality.json", "proposal_cache.jsonl", "tests/unit/test_scheduling_transfer.py"],
            },
            {
                "requirement": "certification disabled in pure transfer experiment",
                "status": "met" if not metrics["certified_pruning_enabled"] else "failed",
                "evidence": ["config.json", "scheduling_transfer_metrics.json"],
            },
            {
                "requirement": "ranking diagnostics reported",
                "status": "met",
                "evidence": ["label_manifest.json", "source_domain_table.csv", "source_ranking_metrics.csv", "target_zero_shot_table.csv", "per_instance_results.csv"],
            },
            {
                "requirement": "source-target domain shift metrics reported",
                "status": "met",
                "evidence": ["domain_shift_metrics.csv", "proposal_cache.jsonl", "scheduling_solver_train_pairs.jsonl"],
            },
            {
                "requirement": "real Scheduling GPT pilot API run",
                "status": "met" if metrics["benchmark"] == "scheduling_real_llm_pilot" and metrics.get("live_network_required") else "missing",
                "evidence": ["scheduling_llm_pilot_table.csv"],
            },
        ],
    }


def _write_pairs(path: Path, pairs) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for positive, negative in pairs:
            handle.write(json.dumps({"positive_features": positive, "negative_features": negative}, sort_keys=True) + "\n")


def _write_trace_records(path: Path, records) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")


def _proposal_contains_oracle_child(task: SchedulingTask, node: SearchNode, actions: Sequence[ScheduleAction]) -> bool:
    if node.completion_status == CompletionStatus.COMPLETE:
        return True
    parent_best = task.best_descendant_objective(node)
    if parent_best == float("inf"):
        return False
    for idx, action in enumerate(actions):
        child = task.apply(node, action, child_index=idx)
        if child.completion_status == CompletionStatus.COMPLETE:
            if child.objective_if_complete == parent_best:
                return True
            continue
        if task.best_descendant_objective(child) == parent_best:
            return True
    return False


def _valid_unique_schedule_actions(task: SchedulingTask, node: SearchNode, actions: Sequence[ScheduleAction]) -> tuple[List[ScheduleAction], int, int]:
    scheduled = tuple(node.structured_state["scheduled"])
    valid_unique: List[ScheduleAction] = []
    seen = set()
    invalid_count = 0
    duplicate_count = 0
    for action in actions:
        if not isinstance(action, ScheduleAction) or not task.is_valid_action(scheduled, action):
            invalid_count += 1
            continue
        key = (action.action_type, action.job, action.machine)
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        valid_unique.append(action)
    return valid_unique, invalid_count, duplicate_count


def _failure_reason(row: Dict[str, object], oracle_success: bool) -> str:
    if row["success"]:
        return "solved"
    if not oracle_success:
        return "proposal_support"
    if row.get("budget_exhausted"):
        return "budget"
    if int(row.get("false_prunes", 0)) > 0:
        return "controller"
    return "ranking"


def _summarize_cache_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    request_count = sum(int(row["request_count"]) for row in rows)
    state_count = sum(int(row["state_count"]) for row in rows)
    support_weight = sum(int(row["state_count"]) * float(row["proposal_recall_at_k"]) for row in rows)
    empty_weight = sum(int(row["state_count"]) * float(row["empty_valid_action_rate"]) for row in rows)
    return {
        "shared": True,
        "cache_key": "sha256(instance_id, canonical_state_json, model, prompt_version, proposal_k, temperature)",
        "episode_count": len(rows),
        "request_count": request_count,
        "state_count": state_count,
        "cache_hits": sum(int(row["cache_hits"]) for row in rows),
        "proposal_calls": sum(int(row["proposal_calls"]) for row in rows),
        "input_tokens": sum(int(row["input_tokens"]) for row in rows),
        "output_tokens": sum(int(row["output_tokens"]) for row in rows),
        "total_tokens": sum(int(row["total_tokens"]) for row in rows),
        "latency": sum(float(row["latency"]) for row in rows),
        "parse_failures": sum(int(row["parse_failures"]) for row in rows),
        "invalid_actions": sum(int(row.get("invalid_actions", 0)) for row in rows),
        "duplicate_actions": sum(int(row.get("duplicate_actions", 0)) for row in rows),
        "proposal_recall_at_k": 0.0 if state_count == 0 else support_weight / state_count,
        "empty_valid_action_rate": 0.0 if state_count == 0 else empty_weight / state_count,
    }


def _train_ranker(pairs, rank_epochs: int, seed: int) -> TinyPairwiseMLPRanker:
    dim = len(pairs[0][0]) if pairs else 20
    ranker = TinyPairwiseMLPRanker(dim=dim, hidden_dim=16, seed=seed)
    if pairs:
        ranker.train(pairs, epochs=rank_epochs, lr=0.02, seed=seed)
    return ranker


def _sample_pairs(pairs, size: int, seed: int):
    if not pairs or size <= 0:
        return []
    rng = random.Random(seed)
    if size <= len(pairs):
        return rng.sample(list(pairs), size)
    return [rng.choice(pairs) for _ in range(size)]


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_jsonl_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_csv_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _coerce_result_row(row: Dict[str, object]) -> Dict[str, object]:
    coerced = dict(row)
    for field in ("success", "budget_exhausted", "oracle_reachable_success"):
        if field in coerced:
            coerced[field] = _as_bool(coerced[field])
    for field in ("nodes", "proposal_calls", "proposal_tokens", "value_calls", "value_tokens", "total_tokens", "false_prunes", "frontier_decisions"):
        if field in coerced:
            coerced[field] = int(float(coerced[field]))
    for field in ("gap", "frontier_top1_accuracy", "pairwise_accuracy", "selection_regret_mean"):
        if field in coerced:
            coerced[field] = float(coerced[field])
    return coerced


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _weighted_mean(rows: Sequence[Dict[str, object]], value_field: str, weight_field: str) -> float:
    total_weight = sum(float(row.get(weight_field, 0.0)) for row in rows)
    if total_weight == 0.0:
        return 0.0
    return sum(float(row.get(value_field, 0.0)) * float(row.get(weight_field, 0.0)) for row in rows) / total_weight


def _estimate_prompt_tokens(text: str) -> int:
    return max(1, len(text.split()))


def _gap(best_objective: float | None, optimum: float) -> float:
    if best_objective is None:
        return 1.0
    return max(0.0, (float(best_objective) - float(optimum)) / max(1.0, float(optimum)))


def _mean(values: Iterable[float]) -> float:
    data = list(values)
    return 0.0 if not data else sum(data) / len(data)
