from __future__ import annotations

import copy
from dataclasses import dataclass
import csv
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import random
import re
import signal
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from cgr_exp.harness_contract import ActionFeatures, CanonicalCandidate, RuntimeFeatures, canonical_contract_manifest, canonical_feature_names
from cgr_exp.harness_ranker import GenericHarnessMLPRanker, GenericHarnessRanker, load_generic_harness_ranker, train_generic_harness_ranker
from cgr_exp.proposers.llm_adapter import OpenAICompatibleHTTPClient
from cgr_exp.workflow_source_generator import SOURCE_WORKFLOW_FAMILIES, generate_workflow_source_tasks, source_generator_manifest


READ_VERBS = ("look", "inventory", "examine")
HELP_VERBS = ("help",)
MOVE_VERBS = ("go",)
OPEN_VERBS = ("open", "close")
TAKE_VERBS = ("take", "pick")
PUT_VERBS = ("put", "place", "insert", "move")
USE_VERBS = ("toggle", "heat", "cool", "clean", "slice", "pour", "turn", "use")
TERMINATE_VERBS = ("done", "finish", "stop")
TRANSFORM_VERBS = ("cool", "heat", "clean", "slice")
ALFWORLD_ADAPTER_VERSION = "v2_relational_goal_action"
ALFWORLD_DENSE_LABEL_REPLAY = "replay-prefix"
ALFWORLD_DENSE_LABEL_WALKTHROUGH = "walkthrough"
ALFWORLD_DENSE_LABEL_PLANNER = "planner"
ALFWORLD_DENSE_LABEL_SOURCES = {ALFWORLD_DENSE_LABEL_REPLAY, ALFWORLD_DENSE_LABEL_WALKTHROUGH, ALFWORLD_DENSE_LABEL_PLANNER}
ALFWORLD_FRONTIER_STRATEGY_PRIORITY = "priority"
ALFWORLD_FRONTIER_STRATEGY_SCORE_DIVERSE = "score-diverse"
ALFWORLD_FRONTIER_STRATEGIES = {ALFWORLD_FRONTIER_STRATEGY_PRIORITY, ALFWORLD_FRONTIER_STRATEGY_SCORE_DIVERSE}
ALFWORLD_RANKER_MODEL_LINEAR = "linear"
ALFWORLD_RANKER_MODEL_MLP = "mlp"
ALFWORLD_RANKER_MODELS = {ALFWORLD_RANKER_MODEL_LINEAR, ALFWORLD_RANKER_MODEL_MLP}
ALFWORLD_PLANNER_LABEL_TIMEOUT_SECONDS = 8.0
ALFWORLD_CONTRACT_SIGN_EXPECTATIONS = {
    "declared_requirement_gain": "positive",
    "declared_observation_gain": "positive",
    "completes_goal_placement": "positive",
    "precondition_satisfied": "positive",
    "policy_risk": "negative",
    "duplicates_previous_action": "negative",
}


@dataclass(frozen=True)
class AlfworldEpisodeResult:
    policy: str
    episode: int
    success: int
    final_score: float
    steps: int
    mean_admissible_count: float
    opportunity_rate: float
    invalid_actions: int
    wall_clock_seconds: float
    tokens: int = 0
    llm_calls: int = 0


@dataclass(frozen=True)
class AlfworldStepTrace:
    policy: str
    episode: int
    step: int
    command: str
    objective: str
    command_type: str
    score: float | str
    second_best_score: float | str
    score_margin: float | str
    expert_available: int
    diagnostic_command: str
    diagnostic_source: str
    matches_diagnostic: int
    admissible_count: int
    verified_progress_frac: float
    remaining_requirement_frac: float
    observed_fact_frac: float
    uncertainty_score: float
    declared_observation_gain: float
    declared_requirement_gain: float
    references_goal_object: float
    references_goal_receptacle: float
    matches_unsatisfied_transform: float
    goal_object_held: float
    goal_receptacle_observed: float
    placement_ready: float
    completes_goal_placement: float
    precondition_satisfied: float
    policy_risk: float
    duplicates_previous_action: float
    top_ranked_commands: str
    candidate_score_summary: str
    success_after_step: int


@dataclass(frozen=True)
class AlfworldDenseActionLabel:
    episode: int
    step: int
    command: str
    cost_to_success: int
    reachable: int
    final_score: float
    rollout_steps: int
    label_source: str
    label_status: str = "ok"
    pairwise_usable: int = 1
    oracle_rank: int = 0
    best_cost_to_success: int = 0
    cost_gap_to_best: int = 0


@dataclass(frozen=True)
class AlfworldGoalSpec:
    object_terms: frozenset[str]
    receptacle_terms: frozenset[str]
    transform_terms: frozenset[str]
    raw_terms: frozenset[str]


class AlfworldLLMShortlistCache:
    def __init__(
        self,
        client: Any,
        *,
        model: str,
        k: int = 4,
        temperature: float = 0.0,
        top_p: float = 1.0,
        cache_path: Path | None = None,
    ):
        self.client = client
        self.model = model
        self.k = k
        self.temperature = temperature
        self.top_p = top_p
        self.cache_path = cache_path
        self.records: list[dict[str, object]] = []
        self._cache: dict[str, dict[str, object]] = {}
        self._used_keys: set[str] = set()
        if self.cache_path is not None and self.cache_path.exists():
            self._load_disk_cache(self.cache_path)

    def _load_disk_cache(self, path: Path) -> None:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = str(row.get("cache_key", ""))
                if not key:
                    continue
                ranked_indices = row.get("ranked_indices", [])
                if not isinstance(ranked_indices, list):
                    continue
                self._cache[key] = {
                    "ranked_indices": ranked_indices,
                    "input_tokens": int(row.get("input_tokens", 0)),
                    "output_tokens": int(row.get("output_tokens", 0)),
                    "latency": float(row.get("latency", 0.0)),
                    "raw_output": str(row.get("raw_output", "")),
                    "cache_hit": False,
                }

    def _cache_key(self, payload: Mapping[str, object]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _append_disk_cache(self, key: str, row: Mapping[str, object]) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        persisted = dict(row)
        persisted["cache_key"] = key
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(persisted, sort_keys=True) + "\n")

    def shortlist(
        self,
        *,
        objective: str,
        observation: str,
        history: Sequence[str],
        admissible: Sequence[str],
    ) -> dict[str, object]:
        key_payload = {
            "model": self.model,
            "k": self.k,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "objective": objective,
            "observation": observation,
            "history": list(history),
            "admissible": list(admissible),
        }
        key = self._cache_key(key_payload)
        self._used_keys.add(key)
        if key in self._cache:
            cached = dict(self._cache[key])
            cached["cache_hit"] = True
            cached["latency"] = 0.0
            return cached
        response = self.client.rank_alfworld_actions(
            objective=objective,
            observation=observation,
            history=list(history),
            admissible=list(admissible),
            k=self.k,
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        ranked_indices = _clean_ranked_indices(response.get("ranked_indices", []), len(admissible))
        row = {
            "ranked_indices": ranked_indices[: max(1, self.k)],
            "input_tokens": int(response.get("input_tokens", 0)),
            "output_tokens": int(response.get("output_tokens", 0)),
            "latency": float(response.get("latency", 0.0)),
            "raw_output": str(response.get("raw_output", "")),
            "cache_hit": False,
        }
        self._cache[key] = dict(row)
        self.records.append(row)
        self._append_disk_cache(key, row)
        return row

    @property
    def tokens(self) -> int:
        return sum(
            int(self._cache[key].get("input_tokens", 0)) + int(self._cache[key].get("output_tokens", 0))
            for key in self._used_keys
            if key in self._cache
        )

    @property
    def calls(self) -> int:
        return len([key for key in self._used_keys if key in self._cache])

    @property
    def new_calls(self) -> int:
        return len(self.records)


class ScriptedAlfworldShortlistClient:
    def __init__(self, *, input_tokens: int = 0, output_tokens: int = 0):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def rank_alfworld_actions(self, *args, **kwargs) -> dict[str, object]:
        objective = str(kwargs.get("objective", ""))
        observation = str(kwargs.get("observation", ""))
        history = list(kwargs.get("history", []))
        admissible = [str(item) for item in kwargs.get("admissible", [])]
        scored = [
            (_hand_score(command, observation, objective, history), -idx, idx)
            for idx, command in enumerate(admissible)
        ]
        scored.sort(reverse=True)
        return {
            "ranked_indices": [idx for _score, _tie, idx in scored],
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency": 0.0,
            "raw_output": json.dumps({"ranked_indices": [idx for _score, _tie, idx in scored]}),
        }


def run_alfworld_transfer(
    out_dir: Path,
    *,
    seed: int = 0,
    split: str = "eval_out_of_distribution",
    config_path: Path | None = None,
    episodes: int = 20,
    max_steps: int = 50,
    source_train_count: int = 200,
    source_epochs: int = 20,
    source_max_states_per_task: int = 200,
    source_difficulty: str = "v2",
    ranker_checkpoint: Path | None = None,
    policies: Sequence[str] = ("Random", "Hand heuristic", "Source-trained controller"),
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None and cache_dir is None:
        _assert_alfworld_available()
    ranker, source_manifest = _load_or_train_ranker(
        ranker_checkpoint=ranker_checkpoint,
        seed=seed,
        source_train_count=source_train_count,
        source_epochs=source_epochs,
        source_max_states_per_task=source_max_states_per_task,
        source_difficulty=source_difficulty,
    )
    all_results: list[AlfworldEpisodeResult] = []
    step_trace_rows: list[dict[str, object]] = []
    for policy in policies:
        env = _make_alfworld_env(
            split=split,
            config_path=config_path,
            env_factory=env_factory,
            training_method=_alfworld_env_method_for_policy(policy),
        )
        for episode in range(episodes):
            all_results.append(
                _run_episode(
                    env,
                    policy=policy,
                    ranker=ranker,
                    seed=seed * 100000 + episode,
                    episode=episode,
                    max_steps=max_steps,
                    step_trace_rows=step_trace_rows,
                )
            )
    rows = _with_step_trace_diagnostics(
        [_aggregate_policy(policy, [row for row in all_results if row.policy == policy]) for policy in policies],
        step_trace_rows,
    )
    manifest = {
        "benchmark": "alfworld_admissible_action_transfer",
        "seed": seed,
        "split": split,
        "episodes": episodes,
        "max_steps": max_steps,
        "policies": list(policies),
        "uses_admissible_commands": True,
        "meta_commands_filtered_from_frontier": True,
        "zero_gain_write_filtered_from_frontier": True,
        "repeated_zero_gain_read_filtered_from_frontier": True,
        "target_trajectory_training_used": False,
        "alfworld_train_trajectories_used": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "alfworld_adapter_version": ALFWORLD_ADAPTER_VERSION,
        "relational_goal_action_features": True,
        "object_identity_feature_used": False,
        "controller_checkpoint_frozen": True,
        "contract": canonical_contract_manifest(),
        "source_controller": source_manifest,
        "rows": rows,
    }
    _write_csv(out_dir / "alfworld_transfer_table.csv", rows)
    _write_csv(out_dir / "alfworld_transfer_per_episode.csv", [row.__dict__ for row in all_results])
    _write_csv(out_dir / "alfworld_transfer_step_trace.csv", step_trace_rows)
    (out_dir / "alfworld_transfer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_alfworld_dense_transfer(
    out_dir: Path,
    *,
    seed: int = 0,
    eval_split: str = "eval_out_of_distribution",
    config_path: Path | None = None,
    train_episodes: int = 100,
    eval_episodes: int = 20,
    max_steps: int = 50,
    epochs: int = 10,
    max_pairs_per_state: int = 24,
    max_frontier_actions: int = 16,
    frontier_strategy: str = ALFWORLD_FRONTIER_STRATEGY_PRIORITY,
    ranker_model: str = ALFWORLD_RANKER_MODEL_LINEAR,
    rank_hidden_dim: int = 32,
    rank_lr: float = 0.03,
    aggregation_rounds: int = 0,
    aggregation_episodes: int = 0,
    counterfactual_rollout_steps: int = 20,
    planner_timeout_seconds: float = ALFWORLD_PLANNER_LABEL_TIMEOUT_SECONDS,
    max_episode_label_seconds: float = 0.0,
    label_source: str = ALFWORLD_DENSE_LABEL_REPLAY,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    ranker_checkpoint: Path | None = None,
    policies: Sequence[str] = ("Random", "Hand heuristic", "ALFWorld dense-ranker", "ALFWorld planner diagnostic"),
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None:
        _assert_alfworld_available()
    if ranker_checkpoint is not None:
        ranker = load_generic_harness_ranker(ranker_checkpoint.read_text(encoding="utf-8"))
        train_manifest = {
            "trained_in_this_run": False,
            "checkpoint": str(ranker_checkpoint),
            "ranker_model": _alfworld_ranker_model_name(ranker),
            "rank_hidden_dim": getattr(ranker, "hidden_dim", 0),
            "counterfactual_planner_cost_to_go": False,
            "label_source": "loaded_checkpoint",
        }
        train_rows: list[dict[str, object]] = []
        action_label_rows: list[dict[str, object]] = []
    else:
        ranker, train_manifest, train_rows, action_label_rows = train_alfworld_dense_ranker(
            seed=seed,
            config_path=config_path,
            train_episodes=train_episodes,
            max_steps=max_steps,
            epochs=epochs,
            max_pairs_per_state=max_pairs_per_state,
            max_frontier_actions=max_frontier_actions,
            frontier_strategy=frontier_strategy,
            ranker_model=ranker_model,
            rank_hidden_dim=rank_hidden_dim,
            rank_lr=rank_lr,
            aggregation_rounds=aggregation_rounds,
            aggregation_episodes=aggregation_episodes,
            counterfactual_rollout_steps=counterfactual_rollout_steps,
            planner_timeout_seconds=planner_timeout_seconds,
            max_episode_label_seconds=max_episode_label_seconds,
            label_source=label_source,
            perturbation_prob=perturbation_prob,
            warmup_random_steps=warmup_random_steps,
            env_factory=env_factory,
        )
    all_results: list[AlfworldEpisodeResult] = []
    step_trace_rows: list[dict[str, object]] = []
    for policy in policies:
        env = _make_alfworld_env(
            split=eval_split,
            config_path=config_path,
            env_factory=env_factory,
            training_method=_alfworld_env_method_for_policy(policy),
        )
        for episode in range(eval_episodes):
            all_results.append(
                _run_episode(
                    env,
                    policy=policy,
                    ranker=ranker,
                    seed=seed * 100000 + episode,
                    episode=episode,
                    max_steps=max_steps,
                    step_trace_rows=step_trace_rows,
                )
            )
    rows = _with_step_trace_diagnostics(
        [_aggregate_policy(policy, [row for row in all_results if row.policy == policy]) for policy in policies],
        step_trace_rows,
    )
    manifest = {
        "benchmark": "alfworld_dense_frontier_transfer",
        "seed": seed,
        "train_split": "train",
        "eval_split": eval_split,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "max_steps": max_steps,
        "max_frontier_actions": max_frontier_actions,
        "frontier_strategy": _normalize_alfworld_frontier_strategy(frontier_strategy),
        "rank_lr": rank_lr,
        "policies": list(policies),
        "uses_admissible_commands": True,
        "meta_commands_filtered_from_frontier": True,
        "zero_gain_write_filtered_from_frontier": True,
        "repeated_zero_gain_read_filtered_from_frontier": True,
        "training_source": "loaded_checkpoint" if ranker_checkpoint is not None else _alfworld_dense_training_source(label_source),
        "label_source": str(train_manifest.get("label_source", _alfworld_dense_label_source_name(label_source))),
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "alfworld_adapter_version": ALFWORLD_ADAPTER_VERSION,
        "relational_goal_action_features": True,
        "object_identity_feature_used": False,
        "controller_checkpoint_frozen_before_eval": True,
        "contract": canonical_contract_manifest(),
        "train": train_manifest,
        "rows": rows,
    }
    (out_dir / "alfworld_dense_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    _write_csv(out_dir / "alfworld_dense_transfer_table.csv", rows)
    _write_csv(out_dir / "alfworld_dense_transfer_per_episode.csv", [row.__dict__ for row in all_results])
    _write_csv(out_dir / "alfworld_dense_transfer_step_trace.csv", step_trace_rows)
    _write_csv(out_dir / "alfworld_dense_training_trace.csv", train_rows)
    _write_csv(out_dir / "alfworld_dense_action_labels.csv", action_label_rows)
    (out_dir / "alfworld_dense_transfer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_alfworld_llm_shortlist_transfer_from_env(
    out_dir: Path,
    *,
    model: str | None = None,
    timeout: float = 120.0,
    seed: int = 0,
    eval_split: str = "eval_in_distribution",
    config_path: Path | None = None,
    ranker_checkpoint: Path | None = None,
    episodes: int = 20,
    max_steps: int = 50,
    shortlist_k: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
    shortlist_cache_path: Path | None = None,
    policies: Sequence[str] = (
        "Frozen LLM top-1",
        "Frozen LLM shortlist + Random",
        "Frozen LLM shortlist + Dense-ranker",
        "Frozen LLM shortlist + Planner diagnostic",
    ),
) -> dict[str, object]:
    client = OpenAICompatibleHTTPClient.from_env()
    client.timeout = timeout
    if model is not None:
        client.model = model
    return run_alfworld_llm_shortlist_transfer(
        out_dir,
        client=client,
        model=model or client.model,
        seed=seed,
        eval_split=eval_split,
        config_path=config_path,
        ranker_checkpoint=ranker_checkpoint,
        episodes=episodes,
        max_steps=max_steps,
        shortlist_k=shortlist_k,
        temperature=temperature,
        top_p=top_p,
        shortlist_cache_path=shortlist_cache_path,
        policies=policies,
    )


def run_alfworld_llm_shortlist_transfer(
    out_dir: Path,
    *,
    client: Any,
    model: str,
    seed: int = 0,
    eval_split: str = "eval_in_distribution",
    config_path: Path | None = None,
    ranker_checkpoint: Path | None = None,
    episodes: int = 20,
    max_steps: int = 50,
    shortlist_k: int = 4,
    temperature: float = 0.0,
    top_p: float = 1.0,
    shortlist_cache_path: Path | None = None,
    policies: Sequence[str] = (
        "Frozen LLM top-1",
        "Frozen LLM shortlist + Random",
        "Frozen LLM shortlist + Dense-ranker",
        "Frozen LLM shortlist + Planner diagnostic",
    ),
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None:
        _assert_alfworld_available()
    if ranker_checkpoint is None:
        raise RuntimeError("run-alfworld-llm-shortlist-transfer requires a frozen ALFWorld dense ranker checkpoint")
    ranker = load_generic_harness_ranker(ranker_checkpoint.read_text(encoding="utf-8"))
    if shortlist_cache_path is None:
        shortlist_cache_path = out_dir / "alfworld_llm_shortlist_cache.jsonl"
    shortlist_cache_loaded = shortlist_cache_path.exists()
    progress_path = out_dir / "alfworld_llm_shortlist_progress.json"
    shortlister = AlfworldLLMShortlistCache(
        client,
        model=model,
        k=shortlist_k,
        temperature=temperature,
        top_p=top_p,
        cache_path=shortlist_cache_path,
    )
    all_results: list[AlfworldEpisodeResult] = []
    step_trace_rows: list[dict[str, object]] = []
    total_policy_episodes = len(policies) * episodes
    completed_policy_episodes = 0
    _write_alfworld_llm_shortlist_progress(
        progress_path,
        status="running",
        completed_policy_episodes=completed_policy_episodes,
        total_policy_episodes=total_policy_episodes,
        policy="",
        episode=-1,
        shortlister=shortlister,
        shortlist_cache_path=shortlist_cache_path,
    )
    for policy in policies:
        env = _make_alfworld_env(
            split=eval_split,
            config_path=config_path,
            env_factory=env_factory,
            training_method=_alfworld_env_method_for_policy(policy),
        )
        for episode in range(episodes):
            result = _run_llm_shortlist_episode(
                env,
                policy=policy,
                ranker=ranker,
                shortlister=shortlister,
                seed=seed * 100000 + episode,
                episode=episode,
                max_steps=max_steps,
                step_trace_rows=step_trace_rows,
            )
            all_results.append(result)
            completed_policy_episodes += 1
            _write_alfworld_llm_shortlist_partial_outputs(out_dir, policies, all_results, step_trace_rows)
            _write_alfworld_llm_shortlist_progress(
                progress_path,
                status="running",
                completed_policy_episodes=completed_policy_episodes,
                total_policy_episodes=total_policy_episodes,
                policy=policy,
                episode=episode,
                shortlister=shortlister,
                shortlist_cache_path=shortlist_cache_path,
            )
            print(
                f"[alfworld-llm-shortlist] {completed_policy_episodes}/{total_policy_episodes} policy-episodes, "
                f"policy={policy}, episode={episode}, success={result.success}, "
                f"new_llm_calls={shortlister.new_calls}, cache_keys_used={shortlister.calls}, "
                f"tokens={shortlister.tokens}",
                flush=True,
            )
    rows = _with_step_trace_diagnostics(
        [_aggregate_policy(policy, [row for row in all_results if row.policy == policy]) for policy in policies],
        step_trace_rows,
    )
    manifest = {
        "benchmark": "alfworld_llm_shortlist_controller_transfer",
        "seed": seed,
        "eval_split": eval_split,
        "episodes": episodes,
        "max_steps": max_steps,
        "shortlist_k": shortlist_k,
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "policies": list(policies),
        "ranker_checkpoint": str(ranker_checkpoint),
        "uses_admissible_commands": True,
        "llm_semantic_shortlist_used": True,
        "llm_calls_are_one_per_state": True,
        "llm_rollout_trajectories_used_for_training": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "alfworld_adapter_version": ALFWORLD_ADAPTER_VERSION,
        "relational_goal_action_features": True,
        "object_identity_feature_used": False,
        "controller_checkpoint_frozen_before_eval": True,
        "actual_unique_llm_calls": shortlister.calls,
        "actual_unique_llm_tokens": shortlister.tokens,
        "new_llm_calls": shortlister.new_calls,
        "shortlist_cache_path": str(shortlist_cache_path),
        "shortlist_cache_loaded": shortlist_cache_loaded,
        "progress_path": str(progress_path),
        "contract": canonical_contract_manifest(),
        "rows": rows,
    }
    _write_csv(out_dir / "alfworld_llm_shortlist_table.csv", rows)
    _write_csv(out_dir / "alfworld_llm_shortlist_per_episode.csv", [row.__dict__ for row in all_results])
    _write_csv(out_dir / "alfworld_llm_shortlist_step_trace.csv", step_trace_rows)
    _write_csv(out_dir / "alfworld_llm_shortlist_calls.csv", shortlister.records)
    (out_dir / "alfworld_llm_shortlist.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_alfworld_llm_shortlist_progress(
        progress_path,
        status="complete",
        completed_policy_episodes=completed_policy_episodes,
        total_policy_episodes=total_policy_episodes,
        policy="",
        episode=-1,
        shortlister=shortlister,
        shortlist_cache_path=shortlist_cache_path,
    )
    return manifest


def audit_alfworld_dense_labels(
    out_dir: Path,
    *,
    seed: int = 0,
    config_path: Path | None = None,
    cache_dir: Path | None = None,
    episodes: int = 5,
    max_steps: int = 20,
    max_pairs_per_state: int = 24,
    max_frontier_actions: int = 16,
    frontier_strategy: str = ALFWORLD_FRONTIER_STRATEGY_PRIORITY,
    ranker_model: str = ALFWORLD_RANKER_MODEL_LINEAR,
    rank_hidden_dim: int = 32,
    rank_lr: float = 0.03,
    aggregation_rounds: int = 0,
    aggregation_episodes: int = 0,
    counterfactual_rollout_steps: int = 20,
    planner_timeout_seconds: float = ALFWORLD_PLANNER_LABEL_TIMEOUT_SECONDS,
    max_episode_label_seconds: float = 0.0,
    label_source: str = ALFWORLD_DENSE_LABEL_REPLAY,
    perturbation_prob: float = 0.0,
    warmup_random_steps: int = 0,
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None and cache_dir is None:
        _assert_alfworld_available()
    if cache_dir is not None:
        train_manifest, trace_rows, action_label_rows = _load_alfworld_dense_audit_cache(
            cache_dir,
            fallback_label_source=label_source,
            fallback_max_pairs_per_state=max_pairs_per_state,
            fallback_max_frontier_actions=max_frontier_actions,
            fallback_frontier_strategy=frontier_strategy,
            fallback_counterfactual_rollout_steps=counterfactual_rollout_steps,
            fallback_planner_timeout_seconds=planner_timeout_seconds,
            fallback_max_episode_label_seconds=max_episode_label_seconds,
            fallback_perturbation_prob=perturbation_prob,
            fallback_warmup_random_steps=warmup_random_steps,
        )
    else:
        _ranker, train_manifest, trace_rows, action_label_rows = train_alfworld_dense_ranker(
            seed=seed,
            config_path=config_path,
            train_episodes=episodes,
            max_steps=max_steps,
            epochs=0,
            max_pairs_per_state=max_pairs_per_state,
            max_frontier_actions=max_frontier_actions,
            frontier_strategy=frontier_strategy,
            ranker_model=ranker_model,
            rank_hidden_dim=rank_hidden_dim,
            rank_lr=rank_lr,
            aggregation_rounds=aggregation_rounds,
            aggregation_episodes=aggregation_episodes,
            counterfactual_rollout_steps=counterfactual_rollout_steps,
            planner_timeout_seconds=planner_timeout_seconds,
            max_episode_label_seconds=max_episode_label_seconds,
            label_source=label_source,
            perturbation_prob=perturbation_prob,
            warmup_random_steps=warmup_random_steps,
            env_factory=env_factory,
        )
    summary = _summarize_dense_label_rows(action_label_rows, train_manifest=train_manifest)
    manifest = {
        "benchmark": "alfworld_dense_frontier_label_audit",
        "seed": seed,
        "split": "train",
        "episodes": episodes,
        "max_steps": max_steps,
        "training_source": _alfworld_dense_training_source(label_source),
        "label_source": _alfworld_dense_label_source_name(label_source),
        "meta_commands_filtered_from_frontier": True,
        "zero_gain_write_filtered_from_frontier": True,
        "repeated_zero_gain_read_filtered_from_frontier": True,
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "counterfactual_planner_cost_to_go": bool(train_manifest.get("counterfactual_planner_cost_to_go", False)),
        "frontier_strategy": str(train_manifest.get("frontier_strategy", frontier_strategy)),
        "alfworld_adapter_version": ALFWORLD_ADAPTER_VERSION,
        "relational_goal_action_features": True,
        "object_identity_feature_used": False,
        "contract": canonical_contract_manifest(),
        "label_summary": summary,
        "collection": train_manifest,
        "audit_source": "training_cache" if cache_dir is not None else "fresh_collection",
        "cache_dir": "" if cache_dir is None else str(cache_dir),
    }
    _write_csv(out_dir / "alfworld_dense_label_audit.csv", [summary])
    _write_csv(out_dir / "alfworld_dense_training_trace.csv", trace_rows)
    _write_csv(out_dir / "alfworld_dense_action_labels.csv", action_label_rows)
    (out_dir / "alfworld_dense_label_audit.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_alfworld_dense_train(
    out_dir: Path,
    *,
    seed: int = 0,
    config_path: Path | None = None,
    train_episodes: int = 100,
    max_steps: int = 50,
    epochs: int = 10,
    max_pairs_per_state: int = 24,
    max_frontier_actions: int = 16,
    frontier_strategy: str = ALFWORLD_FRONTIER_STRATEGY_PRIORITY,
    ranker_model: str = ALFWORLD_RANKER_MODEL_LINEAR,
    rank_hidden_dim: int = 32,
    rank_lr: float = 0.03,
    aggregation_rounds: int = 0,
    aggregation_episodes: int = 0,
    counterfactual_rollout_steps: int = 20,
    planner_timeout_seconds: float = ALFWORLD_PLANNER_LABEL_TIMEOUT_SECONDS,
    max_episode_label_seconds: float = 0.0,
    label_source: str = ALFWORLD_DENSE_LABEL_REPLAY,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    resume: bool = False,
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None:
        _assert_alfworld_available()
    if not resume:
        for path in (
            out_dir / "alfworld_dense_pair_cache.jsonl",
            out_dir / "alfworld_dense_state_label_cache.jsonl",
            out_dir / "alfworld_dense_training_trace.csv",
            out_dir / "alfworld_dense_action_labels.csv",
            out_dir / "alfworld_dense_cache_manifest.json",
        ):
            if path.exists():
                path.unlink()
    ranker, train_manifest, train_rows, action_label_rows = train_alfworld_dense_ranker(
        seed=seed,
        config_path=config_path,
        train_episodes=train_episodes,
        max_steps=max_steps,
        epochs=epochs,
        max_pairs_per_state=max_pairs_per_state,
        max_frontier_actions=max_frontier_actions,
        frontier_strategy=frontier_strategy,
        ranker_model=ranker_model,
        rank_hidden_dim=rank_hidden_dim,
        rank_lr=rank_lr,
        aggregation_rounds=aggregation_rounds,
        aggregation_episodes=aggregation_episodes,
        counterfactual_rollout_steps=counterfactual_rollout_steps,
        planner_timeout_seconds=planner_timeout_seconds,
        max_episode_label_seconds=max_episode_label_seconds,
        label_source=label_source,
        perturbation_prob=perturbation_prob,
        warmup_random_steps=warmup_random_steps,
        cache_dir=out_dir,
        resume=resume,
        env_factory=env_factory,
    )
    manifest = {
        "benchmark": "alfworld_dense_frontier_controller_train",
        "seed": seed,
        "train_split": "train",
        "train_episodes": train_episodes,
        "max_steps": max_steps,
        "training_source": _alfworld_dense_training_source(label_source),
        "label_source": str(train_manifest.get("label_source", _alfworld_dense_label_source_name(label_source))),
        "counterfactual_planner_cost_to_go": bool(train_manifest.get("counterfactual_planner_cost_to_go", False)),
        "frontier_strategy": str(train_manifest.get("frontier_strategy", frontier_strategy)),
        "meta_commands_filtered_from_frontier": True,
        "zero_gain_write_filtered_from_frontier": True,
        "repeated_zero_gain_read_filtered_from_frontier": True,
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "alfworld_adapter_version": ALFWORLD_ADAPTER_VERSION,
        "relational_goal_action_features": True,
        "object_identity_feature_used": False,
        "contract": canonical_contract_manifest(),
        "train": train_manifest,
        "resume": resume,
        "ranker_checkpoint": str(out_dir / "alfworld_dense_ranker.json"),
    }
    (out_dir / "alfworld_dense_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    _write_csv(out_dir / "alfworld_dense_training_trace.csv", train_rows)
    _write_csv(out_dir / "alfworld_dense_action_labels.csv", action_label_rows)
    (out_dir / "alfworld_dense_train.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_alfworld_expert_transfer(
    out_dir: Path,
    *,
    seed: int = 0,
    eval_split: str = "eval_out_of_distribution",
    config_path: Path | None = None,
    train_episodes: int = 200,
    eval_episodes: int = 20,
    max_steps: int = 50,
    epochs: int = 20,
    max_pairs_per_state: int = 24,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    ranker_checkpoint: Path | None = None,
    policies: Sequence[str] = ("Random", "Hand heuristic", "ALFWorld expert-ranker"),
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None:
        _assert_alfworld_available()
    if ranker_checkpoint is not None:
        ranker = GenericHarnessRanker.from_json(ranker_checkpoint.read_text(encoding="utf-8"))
        train_manifest = {
            "trained_in_this_run": False,
            "checkpoint": str(ranker_checkpoint),
            "counterfactual_planner_cost_to_go": False,
        }
        train_rows: list[dict[str, object]] = []
    else:
        ranker, train_manifest, train_rows = train_alfworld_expert_ranker(
            seed=seed,
            config_path=config_path,
            train_episodes=train_episodes,
            max_steps=max_steps,
            epochs=epochs,
            max_pairs_per_state=max_pairs_per_state,
            perturbation_prob=perturbation_prob,
            warmup_random_steps=warmup_random_steps,
            env_factory=env_factory,
        )
    all_results: list[AlfworldEpisodeResult] = []
    step_trace_rows: list[dict[str, object]] = []
    for policy in policies:
        env = _make_alfworld_env(
            split=eval_split,
            config_path=config_path,
            env_factory=env_factory,
            training_method=_alfworld_env_method_for_policy(policy),
        )
        for episode in range(eval_episodes):
            all_results.append(
                _run_episode(
                    env,
                    policy=policy,
                    ranker=ranker,
                    seed=seed * 100000 + episode,
                    episode=episode,
                    max_steps=max_steps,
                    step_trace_rows=step_trace_rows,
                )
            )
    rows = _with_step_trace_diagnostics(
        [_aggregate_policy(policy, [row for row in all_results if row.policy == policy]) for policy in policies],
        step_trace_rows,
    )
    manifest = {
        "benchmark": "alfworld_expert_frontier_transfer",
        "seed": seed,
        "train_split": "train",
        "eval_split": eval_split,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "max_steps": max_steps,
        "policies": list(policies),
        "uses_admissible_commands": True,
        "meta_commands_filtered_from_frontier": True,
        "zero_gain_write_filtered_from_frontier": True,
        "repeated_zero_gain_read_filtered_from_frontier": True,
        "training_source": "alfworld_train_expert_plan_frontier_labels",
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "alfworld_adapter_version": ALFWORLD_ADAPTER_VERSION,
        "relational_goal_action_features": True,
        "object_identity_feature_used": False,
        "controller_checkpoint_frozen_before_eval": True,
        "contract": canonical_contract_manifest(),
        "train": train_manifest,
        "rows": rows,
    }
    (out_dir / "alfworld_expert_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    _write_csv(out_dir / "alfworld_expert_transfer_table.csv", rows)
    _write_csv(out_dir / "alfworld_expert_transfer_per_episode.csv", [row.__dict__ for row in all_results])
    _write_csv(out_dir / "alfworld_expert_transfer_step_trace.csv", step_trace_rows)
    _write_csv(out_dir / "alfworld_expert_training_trace.csv", train_rows)
    (out_dir / "alfworld_expert_transfer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_alfworld_expert_train(
    out_dir: Path,
    *,
    seed: int = 0,
    config_path: Path | None = None,
    train_episodes: int = 200,
    max_steps: int = 50,
    epochs: int = 20,
    max_pairs_per_state: int = 24,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None:
        _assert_alfworld_available()
    ranker, train_manifest, train_rows = train_alfworld_expert_ranker(
        seed=seed,
        config_path=config_path,
        train_episodes=train_episodes,
        max_steps=max_steps,
        epochs=epochs,
        max_pairs_per_state=max_pairs_per_state,
        perturbation_prob=perturbation_prob,
        warmup_random_steps=warmup_random_steps,
        env_factory=env_factory,
    )
    manifest = {
        "benchmark": "alfworld_expert_frontier_controller_train",
        "seed": seed,
        "train_split": "train",
        "train_episodes": train_episodes,
        "max_steps": max_steps,
        "training_source": "alfworld_train_expert_plan_frontier_labels",
        "meta_commands_filtered_from_frontier": True,
        "zero_gain_write_filtered_from_frontier": True,
        "repeated_zero_gain_read_filtered_from_frontier": True,
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "alfworld_adapter_version": ALFWORLD_ADAPTER_VERSION,
        "relational_goal_action_features": True,
        "object_identity_feature_used": False,
        "contract": canonical_contract_manifest(),
        "train": train_manifest,
        "ranker_checkpoint": str(out_dir / "alfworld_expert_ranker.json"),
    }
    (out_dir / "alfworld_expert_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    _write_csv(out_dir / "alfworld_expert_training_trace.csv", train_rows)
    (out_dir / "alfworld_expert_train.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def train_alfworld_expert_ranker(
    *,
    seed: int = 0,
    config_path: Path | None = None,
    train_episodes: int = 200,
    max_steps: int = 50,
    epochs: int = 20,
    max_pairs_per_state: int = 24,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    env_factory: Callable[..., Any] | None = None,
) -> tuple[GenericHarnessRanker, dict[str, object], list[dict[str, object]]]:
    rng = random.Random(seed)
    env = _make_alfworld_env(split="train", config_path=config_path, env_factory=env_factory, training_method="dagger")
    pairs = []
    trace_rows: list[dict[str, object]] = []
    expert_labeled_states = 0
    skipped_states = 0
    for episode in range(train_episodes):
        obs, info = env.reset()
        observation = _first(obs)
        objective = _objective_from_info(info, observation)
        history: list[str] = [f"obs:{observation}"]
        episode_pairs = 0
        final_score = 0.0
        for step in range(max_steps):
            admissible = _contract_filtered_admissible_commands(
                _admissible_from_info(info),
                observation=observation,
                objective=objective,
                history=history,
                max_steps=max_steps,
            )
            expert = _expert_action_from_info(info)
            if not admissible:
                break
            if expert in admissible and len(admissible) >= 2:
                positives = [
                    alfworld_candidate_to_canonical(
                        candidate_id=f"alfworld-train:{episode}:{step}:expert",
                        observation=observation,
                        objective=objective,
                        history=history,
                        admissible_commands=admissible,
                        command=expert,
                        max_steps=max_steps,
                    ).feature_vector()
                ]
                negatives = [command for command in admissible if command != expert]
                rng.shuffle(negatives)
                for command in negatives[:max_pairs_per_state]:
                    negative = alfworld_candidate_to_canonical(
                        candidate_id=f"alfworld-train:{episode}:{step}:neg",
                        observation=observation,
                        objective=objective,
                        history=history,
                        admissible_commands=admissible,
                        command=command,
                        max_steps=max_steps,
                    ).feature_vector()
                    pairs.append((positives[0], negative))
                    episode_pairs += 1
                expert_labeled_states += 1
            else:
                skipped_states += 1
            command = _training_rollout_command(
                expert=expert,
                admissible=admissible,
                rng=rng,
                step=step,
                perturbation_prob=perturbation_prob,
                warmup_random_steps=warmup_random_steps,
            )
            next_obs, scores, dones, info = env.step([command])
            final_score = float(_first(scores, 0.0))
            observation = str(_first(next_obs, ""))
            history.extend([f"act:{command}", f"obs:{observation}"])
            if bool(_first(dones, False)):
                break
        trace_rows.append({"episode": episode, "pairs": episode_pairs, "steps": _action_count(history), "final_score": final_score})
    if not pairs:
        raise RuntimeError("ALFWorld expert training produced zero ranking pairs; check expert_plan and admissible_commands availability.")
    ranker = GenericHarnessRanker.create(len(canonical_feature_names()), seed=seed)
    ranker.train(pairs, epochs=epochs, lr=0.03, seed=seed)
    manifest = {
        "train_pair_count": len(pairs),
        "expert_labeled_states": expert_labeled_states,
        "skipped_states": skipped_states,
        "epochs": epochs,
        "max_pairs_per_state": max_pairs_per_state,
        "perturbation_prob": perturbation_prob,
        "warmup_random_steps": warmup_random_steps,
        "expert_source": "ALFWorld train split extra.expert_plan",
        "counterfactual_planner_cost_to_go": False,
    }
    return ranker, manifest, trace_rows


def train_alfworld_dense_ranker(
    *,
    seed: int = 0,
    config_path: Path | None = None,
    train_episodes: int = 100,
    max_steps: int = 50,
    epochs: int = 10,
    max_pairs_per_state: int = 24,
    max_frontier_actions: int = 16,
    frontier_strategy: str = ALFWORLD_FRONTIER_STRATEGY_PRIORITY,
    ranker_model: str = ALFWORLD_RANKER_MODEL_LINEAR,
    rank_hidden_dim: int = 32,
    rank_lr: float = 0.03,
    aggregation_rounds: int = 0,
    aggregation_episodes: int = 0,
    counterfactual_rollout_steps: int = 20,
    planner_timeout_seconds: float = ALFWORLD_PLANNER_LABEL_TIMEOUT_SECONDS,
    max_episode_label_seconds: float = 0.0,
    label_source: str = ALFWORLD_DENSE_LABEL_REPLAY,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    cache_dir: Path | None = None,
    resume: bool = False,
    env_factory: Callable[..., Any] | None = None,
) -> tuple[GenericHarnessRanker, dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    label_source = _normalize_alfworld_dense_label_source(label_source)
    rng = random.Random(seed)
    walkthrough_cache: dict[str, list[str]] = {}
    pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    trace_rows: list[dict[str, object]] = []
    action_label_rows: list[dict[str, object]] = []
    completed_episodes: set[int] = set()
    planner_label_cache: dict[str, AlfworldDenseActionLabel] = {}
    planner_label_cache_path: Path | None = None
    planner_label_cache_loaded_entries = 0
    planner_label_cache_hits = 0
    planner_label_cache_misses = 0
    dense_feature_cache_manifest_compatible = True
    resume_feature_cache_manifest_compatible = True
    skipped_incompatible_dense_pairs = 0
    rebuilt_dense_pairs_from_state_cache = 0
    if cache_dir is not None and resume:
        cached = _load_alfworld_dense_training_cache(
            cache_dir,
            rebuild_feature_pairs_from_state_cache=True,
            max_pairs_per_state=max_pairs_per_state,
            seed=seed,
        )
        pairs.extend(cached["pairs"])
        trace_rows.extend(cached["trace_rows"])
        action_label_rows.extend(cached["action_label_rows"])
        completed_episodes.update(cached["completed_episodes"])
        dense_feature_cache_manifest_compatible = bool(cached["cache_manifest_compatible"])
        resume_feature_cache_manifest_compatible = dense_feature_cache_manifest_compatible
        skipped_incompatible_dense_pairs = int(cached["skipped_incompatible_pairs"])
        rebuilt_dense_pairs_from_state_cache = int(cached.get("rebuilt_feature_pairs_from_state_cache", 0))
    if cache_dir is not None and label_source == ALFWORLD_DENSE_LABEL_PLANNER:
        planner_label_cache_path = cache_dir / "alfworld_dense_planner_label_cache.jsonl"
        planner_label_cache = _load_alfworld_planner_label_cache(planner_label_cache_path)
        planner_label_cache_loaded_entries = len(planner_label_cache)
    train_env = None
    label_env = None
    needs_collection = any(episode not in completed_episodes for episode in range(train_episodes))
    if needs_collection:
        train_env = _make_alfworld_env(split="train", config_path=config_path, env_factory=env_factory, training_method="dagger")
        if label_source == ALFWORLD_DENSE_LABEL_REPLAY:
            label_env = _make_alfworld_env(split="train", config_path=config_path, env_factory=env_factory, training_method="dagger")
        elif label_source == ALFWORLD_DENSE_LABEL_PLANNER:
            label_env = _make_alfworld_planner_env(env_factory=env_factory)
    labeled_states = 0
    skipped_states = 0
    finite_action_labels = 0
    infinite_action_labels = 0
    label_status_counts: dict[str, int] = {}
    expert_monotone_checks = 0
    expert_monotone_failures = 0
    planner_path_cost_monotone_checks = 0
    planner_path_cost_monotone_failures = 0
    planner_path_cost_monotone_inconclusive = 0
    planner_path_cost_monotone_failure_examples: list[dict[str, object]] = []
    planner_path_cost_monotone_inconclusive_examples: list[dict[str, object]] = []
    walkthrough_policy_steps = 0
    walkthrough_policy_misses = 0
    planner_policy_steps = 0
    planner_policy_misses = 0
    episode_label_timeouts = 0
    for episode in range(train_episodes):
        if episode in completed_episodes:
            if train_env is not None:
                train_env.reset()
            continue
        if train_env is None:
            raise RuntimeError("ALFWorld dense training cache state is inconsistent; collection is required but no environment was created.")
        obs, info = train_env.reset()
        observation = _first(obs)
        objective = _objective_from_info(info, observation)
        gamefile = _gamefile_from_info(info)
        history: list[str] = [f"obs:{observation}"]
        prefix: list[str] = []
        episode_pairs = 0
        episode_pair_rows: list[dict[str, object]] = []
        episode_state_label_rows: list[dict[str, object]] = []
        episode_labels = 0
        final_score = 0.0
        episode_start = time.perf_counter()
        skip_reason = ""
        pending_planner_monotone: dict[str, object] | None = None
        for step in range(max_steps):
            if max_episode_label_seconds > 0.0 and (time.perf_counter() - episode_start) >= max_episode_label_seconds:
                skip_reason = "episode_label_timeout"
                episode_label_timeouts += 1
                break
            admissible = _contract_filtered_admissible_commands(
                _admissible_from_info(info),
                observation=observation,
                objective=objective,
                history=history,
                max_steps=max_steps,
            )
            expert = _expert_action_from_info(info)
            if not admissible:
                break
            planner_next = None
            if label_source == ALFWORLD_DENSE_LABEL_PLANNER:
                if expert in admissible:
                    planner_next = expert
            frontier_reference = planner_next if planner_next is not None else expert
            frontier = _frontier_sample(
                admissible,
                expert=frontier_reference,
                max_frontier_actions=max_frontier_actions,
                strategy=frontier_strategy,
                rng=rng,
                observation=observation,
                objective=objective,
                history=history,
                max_steps=max_steps,
            )
            if label_source == ALFWORLD_DENSE_LABEL_WALKTHROUGH:
                labels = _label_frontier_by_walkthrough_suffix(
                    gamefile=gamefile,
                    prefix=prefix,
                    commands=frontier,
                    max_rollout_steps=counterfactual_rollout_steps,
                    walkthrough_cache=walkthrough_cache,
                )
            elif label_source == ALFWORLD_DENSE_LABEL_PLANNER:
                if label_env is None:
                    raise RuntimeError("planner ALFWorld label source requires a PDDL planner environment")
                labels = _label_frontier_by_planner_cost(
                    label_env,
                    gamefile=gamefile,
                    prefix=prefix,
                    commands=frontier,
                    max_rollout_steps=counterfactual_rollout_steps,
                    timeout_seconds=planner_timeout_seconds,
                    label_cache=planner_label_cache,
                    cache_path=planner_label_cache_path,
                )
                planner_label_cache_hits += sum(1 for label in labels if label.label_status.endswith("_cache_hit"))
                planner_label_cache_misses += sum(1 for label in labels if not label.label_status.endswith("_cache_hit"))
            else:
                if label_env is None:
                    raise RuntimeError("replay-prefix ALFWorld label source requires a label environment")
                labels = _label_frontier_by_expert_rollout(
                    label_env,
                    gamefile=gamefile,
                    prefix=prefix,
                    commands=frontier,
                    max_rollout_steps=counterfactual_rollout_steps,
                )
            labels = _annotate_alfworld_dense_label_ranks(labels)
            labels_by_command = {label.command: label for label in labels}
            monotone_reference = frontier_reference
            if monotone_reference in labels_by_command and labels_by_command[monotone_reference].reachable:
                expert_monotone_checks += 1
                best_cost = min(label.cost_to_success for label in labels)
                if labels_by_command[monotone_reference].cost_to_success > best_cost:
                    expert_monotone_failures += 1
            if label_source == ALFWORLD_DENSE_LABEL_PLANNER and pending_planner_monotone is not None:
                censored_labels = [label for label in labels if not _alfworld_dense_label_pairwise_usable(label)]
                reachable_labels = [label for label in labels if label.reachable and _alfworld_dense_label_pairwise_usable(label)]
                if censored_labels:
                    planner_path_cost_monotone_inconclusive += 1
                    if len(planner_path_cost_monotone_inconclusive_examples) < 20:
                        planner_path_cost_monotone_inconclusive_examples.append(
                            {
                                "episode": int(pending_planner_monotone.get("episode", episode)),
                                "step": int(pending_planner_monotone.get("step", step - 1)),
                                "command": str(pending_planner_monotone.get("command", "")),
                                "reason": "successor_state_has_censored_labels",
                                "censored_statuses": sorted({str(label.label_status) for label in censored_labels}),
                            }
                        )
                elif reachable_labels and int(pending_planner_monotone.get("reachable", 0)):
                    planner_path_cost_monotone_checks += 1
                    current_best_cost = min(label.cost_to_success for label in reachable_labels)
                    previous_cost = int(pending_planner_monotone["cost_to_success"])
                    expected_previous_cost = current_best_cost + 1
                    if previous_cost < expected_previous_cost:
                        planner_path_cost_monotone_inconclusive += 1
                        if len(planner_path_cost_monotone_inconclusive_examples) < 20:
                            planner_path_cost_monotone_inconclusive_examples.append(
                                {
                                    "episode": int(pending_planner_monotone.get("episode", episode)),
                                    "step": int(pending_planner_monotone.get("step", step - 1)),
                                    "command": str(pending_planner_monotone.get("command", "")),
                                    "reason": "successor_sample_missing_better_continuation_or_env_mismatch",
                                    "selected_cost_to_success": previous_cost,
                                    "next_state_best_action_cost": current_best_cost,
                                    "expected_selected_cost": expected_previous_cost,
                                }
                            )
                    elif previous_cost > expected_previous_cost:
                        planner_path_cost_monotone_failures += 1
                        if len(planner_path_cost_monotone_failure_examples) < 20:
                            planner_path_cost_monotone_failure_examples.append(
                                {
                                    "episode": int(pending_planner_monotone.get("episode", episode)),
                                    "step": int(pending_planner_monotone.get("step", step - 1)),
                                    "command": str(pending_planner_monotone.get("command", "")),
                                    "selected_cost_to_success": previous_cost,
                                    "next_state_best_action_cost": current_best_cost,
                                    "expected_selected_cost": expected_previous_cost,
                                }
                            )
                else:
                    planner_path_cost_monotone_inconclusive += 1
                    if len(planner_path_cost_monotone_inconclusive_examples) < 20:
                        planner_path_cost_monotone_inconclusive_examples.append(
                            {
                                "episode": int(pending_planner_monotone.get("episode", episode)),
                                "step": int(pending_planner_monotone.get("step", step - 1)),
                                "command": str(pending_planner_monotone.get("command", "")),
                                "reason": "successor_state_has_no_reachable_usable_label",
                            }
                        )
            row_labels = []
            for label in labels:
                action_label = AlfworldDenseActionLabel(
                    episode=episode,
                    step=step,
                    command=label.command,
                    cost_to_success=label.cost_to_success,
                    reachable=label.reachable,
                    final_score=label.final_score,
                    rollout_steps=label.rollout_steps,
                    label_source=label.label_source,
                    label_status=label.label_status,
                    pairwise_usable=label.pairwise_usable,
                    oracle_rank=label.oracle_rank,
                    best_cost_to_success=label.best_cost_to_success,
                    cost_gap_to_best=label.cost_gap_to_best,
                )
                action_label_rows.append(action_label.__dict__)
                row_labels.append(label)
                finite_action_labels += 1 if label.reachable else 0
                infinite_action_labels += 0 if label.reachable else 1
                label_status_counts[label.label_status] = label_status_counts.get(label.label_status, 0) + 1
            episode_labels += len(labels)
            state_pairs = _dense_labels_to_pairs(
                labels=row_labels,
                observation=observation,
                objective=objective,
                history=history,
                admissible_commands=admissible,
                max_steps=max_steps,
                max_pairs=max_pairs_per_state,
                rng=rng,
                episode=episode,
                step=step,
            )
            if state_pairs:
                labeled_states += 1
                pairs.extend(state_pairs)
                episode_pair_rows.extend(
                    {
                        "episode": episode,
                        "positive": list(positive),
                        "negative": list(negative),
                        **_alfworld_dense_pair_cache_metadata(),
                    }
                    for positive, negative in state_pairs
                )
                episode_pairs += len(state_pairs)
            else:
                skipped_states += 1
            rollout_source = ""
            if label_source == ALFWORLD_DENSE_LABEL_WALKTHROUGH:
                command, used_walkthrough = _walkthrough_training_rollout_command(
                    gamefile=gamefile,
                    prefix=prefix,
                    expert=expert,
                    admissible=admissible,
                    rng=rng,
                    step=step,
                    perturbation_prob=perturbation_prob,
                    warmup_random_steps=warmup_random_steps,
                    walkthrough_cache=walkthrough_cache,
                )
                rollout_source = "walkthrough" if used_walkthrough else "walkthrough_fallback"
                walkthrough_policy_steps += 1
                walkthrough_policy_misses += 0 if used_walkthrough else 1
            elif label_source == ALFWORLD_DENSE_LABEL_PLANNER:
                if label_env is None:
                    raise RuntimeError("planner ALFWorld rollout requires a PDDL planner environment")
                planner_policy_steps += 1
                if step < warmup_random_steps or rng.random() < perturbation_prob:
                    command = rng.choice(list(admissible))
                    rollout_source = "random"
                    planner_policy_misses += 1
                else:
                    command = _best_reachable_label_command(row_labels, admissible=admissible)
                    if command is None:
                        if expert in admissible:
                            command = str(expert)
                            rollout_source = "expert_fallback"
                            planner_policy_misses += 1
                        else:
                            command = rng.choice(list(admissible))
                            rollout_source = "random_fallback"
                            planner_policy_misses += 1
                    else:
                        rollout_source = "best_reachable_label"
            else:
                command = _training_rollout_command(
                    expert=expert,
                    admissible=admissible,
                    rng=rng,
                    step=step,
                    perturbation_prob=perturbation_prob,
                    warmup_random_steps=warmup_random_steps,
                )
                rollout_source = "expert_or_random"
            episode_state_label_rows.append(
                _alfworld_dense_state_label_cache_row(
                    episode=episode,
                    step=step,
                    observation=observation,
                    objective=objective,
                    history=history,
                    admissible_commands=admissible,
                    max_steps=max_steps,
                    labels=row_labels,
                    label_source=label_source,
                    selected_rollout_command=command,
                    selected_rollout_source=rollout_source,
                )
            )
            next_obs, scores, dones, info = train_env.step([command])
            prefix.append(command)
            final_score = float(_first(scores, 0.0))
            selected_label = labels_by_command.get(command)
            if label_source == ALFWORLD_DENSE_LABEL_PLANNER and selected_label is not None and selected_label.reachable:
                if bool(_first(dones, False)) and final_score > 0:
                    planner_path_cost_monotone_checks += 1
                    if int(selected_label.cost_to_success) != 1:
                        planner_path_cost_monotone_failures += 1
                        if len(planner_path_cost_monotone_failure_examples) < 20:
                            planner_path_cost_monotone_failure_examples.append(
                                {
                                    "episode": episode,
                                    "step": step,
                                    "command": command,
                                    "selected_cost_to_success": int(selected_label.cost_to_success),
                                    "next_state_best_action_cost": 0,
                                    "expected_selected_cost": 1,
                                }
                            )
                    pending_planner_monotone = None
                else:
                    pending_planner_monotone = {
                        "episode": episode,
                        "step": step,
                        "command": command,
                        "cost_to_success": int(selected_label.cost_to_success),
                        "reachable": int(selected_label.reachable),
                    }
            else:
                pending_planner_monotone = None
            observation = str(_first(next_obs, ""))
            history.extend([f"act:{command}", f"obs:{observation}"])
            if bool(_first(dones, False)):
                break
        label_seconds = time.perf_counter() - episode_start
        trace_rows.append(
            {
                "episode": episode,
                "aggregation_round": "",
                "rollout_policy": "base_training_policy",
                "pairs": episode_pairs,
                "action_labels": episode_labels,
                "steps": _action_count(history),
                "final_score": final_score,
                "label_seconds": label_seconds,
                "skip_reason": skip_reason,
                "planner_path_cost_monotone_checks": planner_path_cost_monotone_checks,
                "planner_path_cost_monotone_failures": planner_path_cost_monotone_failures,
                "planner_path_cost_monotone_inconclusive": planner_path_cost_monotone_inconclusive,
            }
        )
        if cache_dir is not None:
            _append_alfworld_dense_training_cache(
                cache_dir,
                pair_rows=episode_pair_rows,
                state_label_rows=episode_state_label_rows,
                trace_rows=trace_rows,
                action_label_rows=action_label_rows,
            )
        print(
            "[alfworld-dense-train] "
            f"{episode + 1}/{train_episodes} episodes, "
            f"states={labeled_states}, pairs={len(pairs)}, "
            f"finite_labels={finite_action_labels}, inf_labels={infinite_action_labels}",
            flush=True,
        )
    if not pairs:
        raise RuntimeError("ALFWorld dense training produced zero ranking pairs; check replay-prefix labels and expert availability.")
    if cache_dir is not None:
        dense_feature_cache_manifest_compatible = _alfworld_dense_cache_manifest_compatible(cache_dir / "alfworld_dense_cache_manifest.json")
    ranker_model = _normalize_alfworld_ranker_model(ranker_model)
    ranker = _fit_alfworld_dense_ranker(
        pairs,
        ranker_model=ranker_model,
        rank_hidden_dim=rank_hidden_dim,
        epochs=epochs,
        rank_lr=rank_lr,
        seed=seed,
    )
    aggregation_round_summaries: list[dict[str, object]] = []
    aggregation_rounds = max(0, int(aggregation_rounds))
    aggregation_episodes = int(aggregation_episodes if aggregation_episodes > 0 else train_episodes)
    for round_idx in range(aggregation_rounds):
        summary = _collect_alfworld_dense_ranker_aggregation_round(
            seed=seed,
            round_idx=round_idx,
            episode_offset=train_episodes + round_idx * aggregation_episodes,
            episodes=aggregation_episodes,
            completed_episodes=completed_episodes,
            config_path=config_path,
            max_steps=max_steps,
            max_pairs_per_state=max_pairs_per_state,
            max_frontier_actions=max_frontier_actions,
            frontier_strategy=frontier_strategy,
            counterfactual_rollout_steps=counterfactual_rollout_steps,
            planner_timeout_seconds=planner_timeout_seconds,
            max_episode_label_seconds=max_episode_label_seconds,
            label_source=label_source,
            ranker=ranker,
            pairs=pairs,
            trace_rows=trace_rows,
            action_label_rows=action_label_rows,
            cache_dir=cache_dir,
            planner_label_cache=planner_label_cache,
            planner_label_cache_path=planner_label_cache_path,
            env_factory=env_factory,
        )
        aggregation_round_summaries.append(summary)
        labeled_states += int(summary.get("labeled_states", 0))
        skipped_states += int(summary.get("skipped_states", 0))
        finite_action_labels += int(summary.get("finite_action_labels", 0))
        infinite_action_labels += int(summary.get("infinite_action_labels", 0))
        episode_label_timeouts += int(summary.get("episode_label_timeouts", 0))
        planner_label_cache_hits += int(summary.get("planner_label_cache_hits", 0))
        planner_label_cache_misses += int(summary.get("planner_label_cache_misses", 0))
        for key, value in dict(summary.get("label_status_counts", {})).items():
            label_status_counts[str(key)] = label_status_counts.get(str(key), 0) + int(value)
        if int(summary.get("added_pairs", 0)) > 0:
            ranker = _fit_alfworld_dense_ranker(
                pairs,
                ranker_model=ranker_model,
                rank_hidden_dim=rank_hidden_dim,
                epochs=epochs,
                rank_lr=rank_lr,
                seed=seed + round_idx + 1,
            )
    pre_projection_train_diagnostics = _alfworld_pairwise_ranker_diagnostics(ranker, pairs)
    ranker_pre_projection_diagnostics = _alfworld_ranker_weight_diagnostics(ranker)
    ranker_projection = _project_alfworld_ranker_contract_signs(ranker)
    post_projection_train_diagnostics = _alfworld_pairwise_ranker_diagnostics(ranker, pairs)
    ranker_diagnostics = _alfworld_ranker_weight_diagnostics(ranker)
    total_finite_action_labels = sum(1 for row in action_label_rows if int(row.get("reachable", 0)) == 1)
    total_infinite_action_labels = sum(1 for row in action_label_rows if int(row.get("reachable", 0)) != 1)
    total_labeled_states = len(
        {
            (int(row.get("episode", -1)), int(row.get("step", -1)))
            for row in action_label_rows
            if int(row.get("episode", -1)) >= 0 and int(row.get("step", -1)) >= 0
        }
    )
    total_label_status_counts: dict[str, int] = {}
    for row in action_label_rows:
        status = str(row.get("label_status", "")).strip()
        if status:
            total_label_status_counts[status] = total_label_status_counts.get(status, 0) + 1
    pairwise_usable_action_labels = sum(1 for row in action_label_rows if _alfworld_dense_label_row_pairwise_usable(row))
    censored_action_labels = len(action_label_rows) - pairwise_usable_action_labels
    oracle_top_usable_action_labels = sum(
        1
        for row in action_label_rows
        if _alfworld_dense_label_row_pairwise_usable(row) and int(row.get("oracle_rank", 0)) == 1
    )
    usable_cost_gaps = [
        int(row.get("cost_gap_to_best", 0))
        for row in action_label_rows
        if _alfworld_dense_label_row_pairwise_usable(row)
    ]
    final_dense_state_label_cache_rows = (
        len(_load_alfworld_dense_state_label_cache(cache_dir / "alfworld_dense_state_label_cache.jsonl")) if cache_dir is not None else 0
    )
    manifest = {
        "train_pair_count": len(pairs),
        "dense_labeled_states": total_labeled_states,
        "new_dense_labeled_states": labeled_states,
        "skipped_states": skipped_states,
        "finite_action_labels": total_finite_action_labels,
        "new_finite_action_labels": finite_action_labels,
        "infinite_action_labels": total_infinite_action_labels,
        "new_infinite_action_labels": infinite_action_labels,
        "pairwise_usable_action_labels": pairwise_usable_action_labels,
        "censored_action_labels": censored_action_labels,
        "censored_action_label_rate": 0.0 if not action_label_rows else censored_action_labels / len(action_label_rows),
        "oracle_top_usable_action_labels": oracle_top_usable_action_labels,
        "oracle_top_usable_action_frac": 0.0
        if pairwise_usable_action_labels == 0
        else oracle_top_usable_action_labels / pairwise_usable_action_labels,
        "usable_mean_cost_gap_to_best": 0.0 if not usable_cost_gaps else sum(usable_cost_gaps) / len(usable_cost_gaps),
        "usable_max_cost_gap_to_best": "" if not usable_cost_gaps else max(usable_cost_gaps),
        "epochs": epochs,
        "ranker_model": ranker_model,
        "rank_hidden_dim": rank_hidden_dim if ranker_model == ALFWORLD_RANKER_MODEL_MLP else 0,
        "rank_lr": rank_lr,
        "aggregation_rounds": aggregation_rounds,
        "aggregation_episodes": aggregation_episodes,
        "aggregation_added_pairs": sum(int(row.get("added_pairs", 0)) for row in aggregation_round_summaries),
        "aggregation_added_states": sum(int(row.get("labeled_states", 0)) for row in aggregation_round_summaries),
        "aggregation_rollout_policy": "current_dense_ranker",
        "aggregation_round_summaries": aggregation_round_summaries,
        "max_pairs_per_state": max_pairs_per_state,
        "max_frontier_actions": max_frontier_actions,
        "frontier_strategy": frontier_strategy,
        "frontier_reference_priority": "expert_plan_if_available_then_frontier_sampling_score",
        "resume_requested": resume,
        "resumed_completed_episodes": len(completed_episodes),
        "resume_feature_cache_manifest_compatible": resume_feature_cache_manifest_compatible,
        "dense_feature_cache_manifest_compatible": dense_feature_cache_manifest_compatible,
        "skipped_incompatible_dense_pairs": skipped_incompatible_dense_pairs,
        "dense_state_label_cache": str(cache_dir / "alfworld_dense_state_label_cache.jsonl") if cache_dir is not None else "",
        "dense_state_label_cache_rows": final_dense_state_label_cache_rows,
        "rebuilt_dense_pairs_from_state_cache": rebuilt_dense_pairs_from_state_cache,
        "dense_pair_cache": str(cache_dir / "alfworld_dense_pair_cache.jsonl") if cache_dir is not None else "",
        "counterfactual_rollout_steps": counterfactual_rollout_steps,
        "planner_timeout_seconds": planner_timeout_seconds,
        "max_episode_label_seconds": max_episode_label_seconds,
        "planner_label_cache": str(planner_label_cache_path or ""),
        "planner_label_cache_loaded_entries": planner_label_cache_loaded_entries,
        "planner_label_cache_entries": len(planner_label_cache),
        "new_planner_label_cache_hits": planner_label_cache_hits,
        "new_planner_label_cache_misses": planner_label_cache_misses,
        "planner_label_cache_hits": planner_label_cache_hits,
        "planner_label_cache_misses": planner_label_cache_misses,
        "perturbation_prob": perturbation_prob,
        "warmup_random_steps": warmup_random_steps,
        "label_source": _alfworld_dense_label_source_name(label_source),
        "counterfactual_planner_cost_to_go": label_source == ALFWORLD_DENSE_LABEL_PLANNER,
        "walkthrough_suffix_cost_proxy": label_source == ALFWORLD_DENSE_LABEL_WALKTHROUGH,
        "walkthrough_policy_steps": walkthrough_policy_steps,
        "walkthrough_policy_misses": walkthrough_policy_misses,
        "walkthrough_policy_miss_rate": 0.0 if walkthrough_policy_steps == 0 else walkthrough_policy_misses / walkthrough_policy_steps,
        "planner_policy_steps": planner_policy_steps,
        "planner_policy_misses": planner_policy_misses,
        "planner_policy_miss_rate": 0.0 if planner_policy_steps == 0 else planner_policy_misses / planner_policy_steps,
        "planner_path_cost_monotone_checks": planner_path_cost_monotone_checks,
        "planner_path_cost_monotone_failures": planner_path_cost_monotone_failures,
        "planner_path_cost_monotone_inconclusive": planner_path_cost_monotone_inconclusive,
        "planner_path_cost_monotone_failure_rate": 0.0
        if planner_path_cost_monotone_checks == 0
        else planner_path_cost_monotone_failures / planner_path_cost_monotone_checks,
        "planner_path_cost_monotone_failure_examples": planner_path_cost_monotone_failure_examples,
        "planner_path_cost_monotone_inconclusive_examples": planner_path_cost_monotone_inconclusive_examples,
        "episode_label_timeouts": episode_label_timeouts,
        "new_label_status_counts": dict(sorted(label_status_counts.items())),
        "label_status_counts": dict(sorted(total_label_status_counts.items())),
        "expert_monotone_checks": expert_monotone_checks,
        "expert_monotone_failures": expert_monotone_failures,
        "expert_monotone_failure_rate": 0.0 if expert_monotone_checks == 0 else expert_monotone_failures / expert_monotone_checks,
        "ranker_direct_feature_weights": ranker_diagnostics["direct_feature_weights"],
        "ranker_pre_projection_train_diagnostics": pre_projection_train_diagnostics,
        "ranker_post_projection_train_diagnostics": post_projection_train_diagnostics,
        "ranker_pre_projection_contract_sign_violations": ranker_pre_projection_diagnostics["contract_sign_violations"],
        "ranker_pre_projection_contract_sign_violation_count": ranker_pre_projection_diagnostics["contract_sign_violation_count"],
        "ranker_contract_sign_projection": ranker_projection,
        "ranker_contract_sign_violations": ranker_diagnostics["contract_sign_violations"],
        "ranker_contract_sign_violation_count": ranker_diagnostics["contract_sign_violation_count"],
        "ranker_contract_sign_tolerance": ranker_diagnostics["contract_sign_tolerance"],
    }
    return ranker, manifest, trace_rows, action_label_rows


def alfworld_candidate_to_canonical(
    *,
    candidate_id: str,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible_commands: Sequence[str],
    command: str,
    max_steps: int,
) -> CanonicalCandidate:
    state = alfworld_runtime_features(
        observation=observation,
        objective=objective,
        history=history,
        admissible_commands=admissible_commands,
        max_steps=max_steps,
    )
    action = alfworld_action_features(
        command=command,
        observation=observation,
        objective=objective,
        history=history,
        max_steps=max_steps,
    )
    return CanonicalCandidate(candidate_id=candidate_id, state=state, action=action)


def _fit_alfworld_dense_ranker(
    pairs: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
    *,
    ranker_model: str,
    rank_hidden_dim: int,
    epochs: int,
    rank_lr: float,
    seed: int,
) -> GenericHarnessRanker | GenericHarnessMLPRanker:
    ranker_model = _normalize_alfworld_ranker_model(ranker_model)
    if ranker_model == ALFWORLD_RANKER_MODEL_MLP:
        ranker: GenericHarnessRanker | GenericHarnessMLPRanker = GenericHarnessMLPRanker(
            len(canonical_feature_names()),
            hidden_dim=rank_hidden_dim,
            seed=seed,
        )
    else:
        ranker = GenericHarnessRanker.create(len(canonical_feature_names()), seed=seed)
    ranker.train(pairs, epochs=epochs, lr=rank_lr, seed=seed)
    return ranker


def _collect_alfworld_dense_ranker_aggregation_round(
    *,
    seed: int,
    round_idx: int,
    episode_offset: int,
    episodes: int,
    completed_episodes: set[int],
    config_path: Path | None,
    max_steps: int,
    max_pairs_per_state: int,
    max_frontier_actions: int,
    frontier_strategy: str,
    counterfactual_rollout_steps: int,
    planner_timeout_seconds: float,
    max_episode_label_seconds: float,
    label_source: str,
    ranker: GenericHarnessRanker | GenericHarnessMLPRanker,
    pairs: list[tuple[tuple[float, ...], tuple[float, ...]]],
    trace_rows: list[dict[str, object]],
    action_label_rows: list[dict[str, object]],
    cache_dir: Path | None,
    planner_label_cache: dict[str, AlfworldDenseActionLabel],
    planner_label_cache_path: Path | None,
    env_factory: Callable[..., Any] | None,
) -> dict[str, object]:
    if episodes <= 0:
        return {
            "round": round_idx,
            "episodes": 0,
            "skipped_completed_episodes": 0,
            "added_pairs": 0,
            "labeled_states": 0,
            "skipped_states": 0,
            "finite_action_labels": 0,
            "infinite_action_labels": 0,
            "episode_label_timeouts": 0,
            "planner_label_cache_hits": 0,
            "planner_label_cache_misses": 0,
            "label_status_counts": {},
        }

    rng = random.Random(seed * 1000003 + round_idx)
    train_env = _make_alfworld_env(split="train", config_path=config_path, env_factory=env_factory, training_method="dagger")
    if label_source == ALFWORLD_DENSE_LABEL_PLANNER:
        label_env = _make_alfworld_planner_env(env_factory=env_factory)
    elif label_source == ALFWORLD_DENSE_LABEL_REPLAY:
        label_env = _make_alfworld_env(split="train", config_path=config_path, env_factory=env_factory, training_method="dagger")
    else:
        label_env = None
    walkthrough_cache: dict[str, list[str]] = {}
    added_pairs = 0
    labeled_states = 0
    skipped_states = 0
    finite_action_labels = 0
    infinite_action_labels = 0
    skipped_completed = 0
    episode_label_timeouts = 0
    planner_label_cache_hits = 0
    planner_label_cache_misses = 0
    label_status_counts: dict[str, int] = {}

    for local_episode in range(episodes):
        episode = episode_offset + local_episode
        if episode in completed_episodes:
            train_env.reset()
            skipped_completed += 1
            continue
        obs, info = train_env.reset()
        observation = _first(obs)
        objective = _objective_from_info(info, observation)
        gamefile = _gamefile_from_info(info)
        history: list[str] = [f"obs:{observation}"]
        prefix: list[str] = []
        episode_pairs = 0
        episode_labels = 0
        episode_pair_rows: list[dict[str, object]] = []
        episode_state_label_rows: list[dict[str, object]] = []
        final_score = 0.0
        skip_reason = ""
        episode_start = time.perf_counter()
        for step in range(max_steps):
            if max_episode_label_seconds > 0.0 and (time.perf_counter() - episode_start) >= max_episode_label_seconds:
                skip_reason = "aggregation_episode_label_timeout"
                episode_label_timeouts += 1
                break
            admissible = _contract_filtered_admissible_commands(
                _admissible_from_info(info),
                observation=observation,
                objective=objective,
                history=history,
                max_steps=max_steps,
            )
            if not admissible:
                break
            frontier = _frontier_sample(
                admissible,
                expert=None,
                max_frontier_actions=max_frontier_actions,
                strategy=frontier_strategy,
                rng=rng,
                observation=observation,
                objective=objective,
                history=history,
                max_steps=max_steps,
            )
            if label_source == ALFWORLD_DENSE_LABEL_WALKTHROUGH:
                labels = _label_frontier_by_walkthrough_suffix(
                    gamefile=gamefile,
                    prefix=prefix,
                    commands=frontier,
                    max_rollout_steps=counterfactual_rollout_steps,
                    walkthrough_cache=walkthrough_cache,
                )
            elif label_source == ALFWORLD_DENSE_LABEL_PLANNER:
                if label_env is None:
                    raise RuntimeError("planner ALFWorld aggregation requires a PDDL planner environment")
                labels = _label_frontier_by_planner_cost(
                    label_env,
                    gamefile=gamefile,
                    prefix=prefix,
                    commands=frontier,
                    max_rollout_steps=counterfactual_rollout_steps,
                    timeout_seconds=planner_timeout_seconds,
                    label_cache=planner_label_cache,
                    cache_path=planner_label_cache_path,
                )
                planner_label_cache_hits += sum(1 for label in labels if label.label_status.endswith("_cache_hit"))
                planner_label_cache_misses += sum(1 for label in labels if not label.label_status.endswith("_cache_hit"))
            else:
                if label_env is None:
                    raise RuntimeError("replay-prefix ALFWorld aggregation requires a label environment")
                labels = _label_frontier_by_expert_rollout(
                    label_env,
                    gamefile=gamefile,
                    prefix=prefix,
                    commands=frontier,
                    max_rollout_steps=counterfactual_rollout_steps,
                )
            labels = _annotate_alfworld_dense_label_ranks(labels)
            row_labels = []
            for label in labels:
                action_label = AlfworldDenseActionLabel(
                    episode=episode,
                    step=step,
                    command=label.command,
                    cost_to_success=label.cost_to_success,
                    reachable=label.reachable,
                    final_score=label.final_score,
                    rollout_steps=label.rollout_steps,
                    label_source=label.label_source,
                    label_status=label.label_status,
                    pairwise_usable=label.pairwise_usable,
                    oracle_rank=label.oracle_rank,
                    best_cost_to_success=label.best_cost_to_success,
                    cost_gap_to_best=label.cost_gap_to_best,
                )
                action_label_rows.append(action_label.__dict__)
                row_labels.append(label)
                finite_action_labels += 1 if label.reachable else 0
                infinite_action_labels += 0 if label.reachable else 1
                label_status_counts[label.label_status] = label_status_counts.get(label.label_status, 0) + 1
            episode_labels += len(labels)
            state_pairs = _dense_labels_to_pairs(
                labels=row_labels,
                observation=observation,
                objective=objective,
                history=history,
                admissible_commands=admissible,
                max_steps=max_steps,
                max_pairs=max_pairs_per_state,
                rng=rng,
                episode=episode,
                step=step,
            )
            if state_pairs:
                labeled_states += 1
                pairs.extend(state_pairs)
                episode_pairs += len(state_pairs)
                added_pairs += len(state_pairs)
                episode_pair_rows.extend(
                    {
                        "episode": episode,
                        "positive": list(positive),
                        "negative": list(negative),
                        **_alfworld_dense_pair_cache_metadata(),
                    }
                    for positive, negative in state_pairs
                )
            else:
                skipped_states += 1
            command, _selection = _select_command_with_trace(
                "ALFWorld dense-ranker",
                observation,
                objective,
                history,
                admissible,
                ranker,
                rng,
                max_steps,
            )
            episode_state_label_rows.append(
                _alfworld_dense_state_label_cache_row(
                    episode=episode,
                    step=step,
                    observation=observation,
                    objective=objective,
                    history=history,
                    admissible_commands=admissible,
                    max_steps=max_steps,
                    labels=row_labels,
                    label_source=label_source,
                    selected_rollout_command=command,
                    selected_rollout_source=f"aggregation_ranker_round_{round_idx}",
                )
            )
            next_obs, scores, dones, info = train_env.step([command])
            prefix.append(command)
            final_score = float(_first(scores, 0.0))
            observation = str(_first(next_obs, ""))
            history.extend([f"act:{command}", f"obs:{observation}"])
            if bool(_first(dones, False)):
                break
        trace_rows.append(
            {
                "episode": episode,
                "aggregation_round": round_idx,
                "rollout_policy": "current_dense_ranker",
                "pairs": episode_pairs,
                "action_labels": episode_labels,
                "steps": _action_count(history),
                "final_score": final_score,
                "label_seconds": time.perf_counter() - episode_start,
                "skip_reason": skip_reason,
                "planner_path_cost_monotone_checks": "",
                "planner_path_cost_monotone_failures": "",
                "planner_path_cost_monotone_inconclusive": "",
            }
        )
        completed_episodes.add(episode)
        if cache_dir is not None:
            _append_alfworld_dense_training_cache(
                cache_dir,
                pair_rows=episode_pair_rows,
                state_label_rows=episode_state_label_rows,
                trace_rows=trace_rows,
                action_label_rows=action_label_rows,
            )
        print(
            "[alfworld-dense-aggregate] "
            f"round={round_idx + 1}, {local_episode + 1}/{episodes} episodes, "
            f"states={labeled_states}, added_pairs={added_pairs}, "
            f"finite_labels={finite_action_labels}, inf_labels={infinite_action_labels}",
            flush=True,
        )

    return {
        "round": round_idx,
        "episodes": episodes,
        "episode_offset": episode_offset,
        "skipped_completed_episodes": skipped_completed,
        "added_pairs": added_pairs,
        "labeled_states": labeled_states,
        "skipped_states": skipped_states,
        "finite_action_labels": finite_action_labels,
        "infinite_action_labels": infinite_action_labels,
        "episode_label_timeouts": episode_label_timeouts,
        "planner_label_cache_hits": planner_label_cache_hits,
        "planner_label_cache_misses": planner_label_cache_misses,
        "label_status_counts": dict(sorted(label_status_counts.items())),
    }


def alfworld_runtime_features(
    *,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible_commands: Sequence[str],
    max_steps: int,
) -> RuntimeFeatures:
    goal = _alfworld_goal_spec(objective)
    history_text = " ".join(history).lower()
    observed_terms = _observed_terms(observation=observation, history=history)
    progress = _relational_progress(goal=goal, observation=observation, history=history)
    failed = sum(1 for item in history if item.startswith("obs:") and ("can't" in item.lower() or "nothing happens" in item.lower()))
    read_count = sum(1 for item in history if item.startswith("act:") and _command_type(item[4:].strip()) == "read")
    write_count = sum(1 for item in history if item.startswith("act:") and _command_type(item[4:].strip()) == "write")
    relevant_observed = (goal.object_terms | goal.receptacle_terms) & observed_terms
    expected_facts = max(1, len(goal.object_terms | goal.receptacle_terms))
    observed_fact_frac = min(1.0, len(relevant_observed) / expected_facts)
    missing_goal_terms = (goal.object_terms | goal.receptacle_terms) - observed_terms
    return RuntimeFeatures(
        verified_progress_frac=progress,
        verified_progress_known_mask=1.0,
        remaining_requirement_frac=1.0 - progress,
        remaining_requirement_known_mask=1.0,
        observed_fact_frac=observed_fact_frac,
        observed_fact_known_mask=1.0,
        remaining_turn_budget_frac=max(0.0, (max_steps - _action_count(history)) / max(1, max_steps)),
        remaining_tool_budget_frac=max(0.0, (max_steps - _action_count(history)) / max(1, max_steps)),
        constraint_pressure=min(1.0, failed / max(1, len(history))),
        constraint_pressure_known_mask=1.0,
        valid_action_ratio=1.0,
        previous_failure_rate=failed / max(1, _action_count(history)),
        stagnation_steps_frac=_stagnation_frac(history),
        uncertainty_score=min(1.0, len(missing_goal_terms) / expected_facts),
        read_write_balance=(read_count - write_count) / max(1, _action_count(history)),
        irreversible_action_used=1.0 if write_count else 0.0,
    )


def alfworld_action_features(
    *,
    command: str,
    observation: str,
    objective: str,
    history: Sequence[str],
    max_steps: int,
) -> ActionFeatures:
    action_type = _command_type(command)
    repeats = any(item == f"act:{command}" for item in history)
    goal = _alfworld_goal_spec(objective)
    command_terms = _object_tokens(command.lower())
    observed_terms = _observed_terms(observation=observation, history=history)
    held_terms = _held_terms(observation=observation, history=history)
    target_related = bool(command_terms & (goal.object_terms | goal.receptacle_terms | goal.transform_terms))
    references_observed = bool(command_terms & observed_terms)
    recent_move_streak = _recent_action_type_streak(history, "move")
    recent_read_streak = _recent_action_type_streak(history, "read")
    relation = _action_goal_relation(
        command=command,
        action_type=action_type,
        goal=goal,
        observed_terms=observed_terms,
        held_terms=held_terms,
        repeats=repeats,
        recent_move_streak=recent_move_streak,
        recent_read_streak=recent_read_streak,
    )
    return ActionFeatures(
        action_is_read=1.0 if action_type == "read" else 0.0,
        action_is_write=1.0 if action_type == "write" else 0.0,
        action_is_respond=0.0,
        action_is_ask=0.0,
        action_is_terminate=1.0 if action_type == "terminate" else 0.0,
        argument_completeness=1.0 if command.strip() else 0.0,
        references_observed_entity=1.0 if references_observed else 0.0,
        references_unverified_entity=1.0 if command_terms and target_related and not references_observed else 0.0,
        references_goal_object=relation["references_goal_object"],
        references_goal_receptacle=relation["references_goal_receptacle"],
        matches_unsatisfied_transform=relation["matches_unsatisfied_transform"],
        goal_object_held=relation["goal_object_held"],
        goal_receptacle_observed=relation["goal_receptacle_observed"],
        placement_ready=relation["placement_ready"],
        completes_goal_placement=relation["completes_goal_placement"],
        declared_observation_gain=relation["observation_gain"],
        declared_requirement_gain=relation["requirement_gain"],
        expected_tool_cost_frac=1.0 / max(1, max_steps),
        action_irreversible=1.0 if action_type == "write" else 0.0,
        precondition_satisfied=relation["precondition_satisfied"],
        policy_risk=relation["policy_risk"],
        duplicates_previous_action=1.0 if repeats else 0.0,
    )


def _frontier_sample(
    admissible: Sequence[str],
    *,
    expert: str | None,
    max_frontier_actions: int,
    strategy: str = ALFWORLD_FRONTIER_STRATEGY_PRIORITY,
    rng: random.Random,
    observation: str = "",
    objective: str = "",
    history: Sequence[str] = (),
    max_steps: int = 50,
) -> list[str]:
    strategy = _normalize_alfworld_frontier_strategy(strategy)
    commands = list(dict.fromkeys(str(command) for command in admissible))
    if max_frontier_actions <= 0 or len(commands) <= max_frontier_actions:
        return commands
    selected: list[str] = []
    if expert in commands:
        selected.append(str(expert))

    rest = [command for command in commands if command not in selected]
    scored = [
        (
            _frontier_sampling_score(
                command,
                observation=observation,
                objective=objective,
                history=history,
                max_steps=max_steps,
            ),
            rng.random(),
            command,
        )
        for command in rest
    ]
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    if strategy == ALFWORLD_FRONTIER_STRATEGY_SCORE_DIVERSE:
        return _frontier_score_diverse_sample(
            scored=scored,
            selected=selected,
            max_frontier_actions=max_frontier_actions,
        )

    # Labeling every ALFWorld action with a planner is expensive. Keep the
    # planner/expert action, then preserve action-type diversity and high-value
    # goal-related negatives so dense labels are informative rather than random.
    for _score, _tie, command in scored:
        if len(selected) >= max_frontier_actions:
            break
        if command in selected:
            continue
        if _frontier_priority_candidate(
            command,
            observation=observation,
            objective=objective,
            history=history,
            max_steps=max_steps,
        ):
            selected.append(command)

    selected_types = {_command_type(command) for command in selected}
    for _score, _tie, command in scored:
        if len(selected) >= max_frontier_actions:
            break
        command_type = _command_type(command)
        if command_type not in selected_types:
            selected.append(command)
            selected_types.add(command_type)

    for _score, _tie, command in scored:
        if len(selected) >= max_frontier_actions:
            break
        if command not in selected:
            selected.append(command)
    return selected


def _normalize_alfworld_frontier_strategy(strategy: str) -> str:
    normalized = str(strategy or ALFWORLD_FRONTIER_STRATEGY_PRIORITY).strip().lower().replace("_", "-")
    if normalized not in ALFWORLD_FRONTIER_STRATEGIES:
        raise ValueError(f"unknown ALFWorld frontier strategy: {strategy}")
    return normalized


def _normalize_alfworld_ranker_model(model: str) -> str:
    normalized = str(model or ALFWORLD_RANKER_MODEL_LINEAR).strip().lower().replace("_", "-")
    if normalized not in ALFWORLD_RANKER_MODELS:
        raise ValueError(f"unknown ALFWorld ranker model: {model}")
    return normalized


def _alfworld_ranker_model_name(ranker: object) -> str:
    if isinstance(ranker, GenericHarnessMLPRanker):
        return ALFWORLD_RANKER_MODEL_MLP
    return ALFWORLD_RANKER_MODEL_LINEAR


def _append_unique_command(selected: list[str], command: str, *, max_frontier_actions: int) -> bool:
    if len(selected) >= max_frontier_actions or command in selected:
        return False
    selected.append(command)
    return True


def _frontier_score_diverse_sample(
    *,
    scored: Sequence[tuple[float, float, str]],
    selected: list[str],
    max_frontier_actions: int,
) -> list[str]:
    if len(selected) >= max_frontier_actions:
        return selected[:max_frontier_actions]
    if not scored:
        return selected

    ranked = list(scored)
    for idx in (0, len(ranked) - 1, len(ranked) // 2):
        _append_unique_command(selected, ranked[idx][2], max_frontier_actions=max_frontier_actions)

    type_best: dict[str, tuple[float, float, str]] = {}
    type_worst: dict[str, tuple[float, float, str]] = {}
    for item in ranked:
        command_type = _command_type(item[2])
        type_best.setdefault(command_type, item)
        type_worst[command_type] = item

    for command_type in sorted(type_best):
        _append_unique_command(selected, type_best[command_type][2], max_frontier_actions=max_frontier_actions)
    for command_type in sorted(type_worst):
        _append_unique_command(selected, type_worst[command_type][2], max_frontier_actions=max_frontier_actions)

    for _score, _tie, command in ranked:
        _append_unique_command(selected, command, max_frontier_actions=max_frontier_actions)
    return selected[:max_frontier_actions]


def _frontier_priority_candidate(
    command: str,
    *,
    observation: str,
    objective: str,
    history: Sequence[str],
    max_steps: int,
) -> bool:
    action = alfworld_action_features(
        command=command,
        observation=observation,
        objective=objective,
        history=history,
        max_steps=max_steps,
    )
    return any(
        value > 0.0
        for value in (
            action.references_goal_object,
            action.references_goal_receptacle,
            action.matches_unsatisfied_transform,
            action.declared_observation_gain,
            action.declared_requirement_gain,
            action.completes_goal_placement,
        )
    )


def _frontier_sampling_score(
    command: str,
    *,
    observation: str,
    objective: str,
    history: Sequence[str],
    max_steps: int,
) -> float:
    action = alfworld_action_features(
        command=command,
        observation=observation,
        objective=objective,
        history=history,
        max_steps=max_steps,
    )
    return (
        _hand_score(command, observation, objective, history)
        + 3.0 * action.declared_requirement_gain
        + 2.0 * action.declared_observation_gain
        + 1.5 * action.references_goal_object
        + 1.2 * action.references_goal_receptacle
        + 1.0 * action.completes_goal_placement
        + 0.8 * action.precondition_satisfied
        - 1.5 * action.duplicates_previous_action
        - 0.8 * action.policy_risk
    )


def _label_frontier_by_expert_rollout(
    env: Any,
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    commands: Sequence[str],
    max_rollout_steps: int,
) -> list[AlfworldDenseActionLabel]:
    labels = []
    replayed_state: Any | None = None
    try:
        replayed_state, _observation, _info, _score, _done = _replay_prefix(env, gamefile=gamefile, prefix=prefix)
    except Exception:
        replayed_state = None
    for command in commands:
        cost: int
        reachable: bool
        score: float
        rollout_steps: int
        state_copy = _copy_env_state(replayed_state)
        if state_copy is not None:
            cost, reachable, score, rollout_steps = _expert_rollout_cost_from_current_state(
                state_copy,
                candidate=command,
                max_rollout_steps=max_rollout_steps,
            )
        else:
            cost, reachable, score, rollout_steps = _expert_rollout_cost_from_prefix(
                env,
                gamefile=gamefile,
                prefix=prefix,
                candidate=command,
                max_rollout_steps=max_rollout_steps,
            )
        labels.append(
            AlfworldDenseActionLabel(
                episode=-1,
                step=-1,
                command=command,
                cost_to_success=cost,
                reachable=1 if reachable else 0,
                final_score=score,
                rollout_steps=rollout_steps,
                label_source="replay_prefix_plus_expert_rollout_cost_proxy",
            )
        )
    return labels


def _label_frontier_by_planner_cost(
    env: Any,
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    commands: Sequence[str],
    max_rollout_steps: int,
    timeout_seconds: float = ALFWORLD_PLANNER_LABEL_TIMEOUT_SECONDS,
    label_cache: dict[str, AlfworldDenseActionLabel] | None = None,
    cache_path: Path | None = None,
) -> list[AlfworldDenseActionLabel]:
    labels = []
    unreachable_cost = max_rollout_steps + len(prefix) + len(commands) + 2
    cache_keys = [
        _alfworld_planner_label_cache_key(
            gamefile=gamefile,
            prefix=prefix,
            command=command,
            max_rollout_steps=max_rollout_steps,
        )
        for command in commands
    ]
    if label_cache is not None and commands and all(key in label_cache for key in cache_keys):
        return [
            AlfworldDenseActionLabel(
                episode=-1,
                step=-1,
                command=label_cache[key].command,
                cost_to_success=label_cache[key].cost_to_success,
                reachable=label_cache[key].reachable,
                final_score=label_cache[key].final_score,
                rollout_steps=label_cache[key].rollout_steps,
                label_source=label_cache[key].label_source,
                label_status=f"{label_cache[key].label_status}_cache_hit",
            )
            for key in cache_keys
        ]
    replayed_state: Any | None = None
    replayed_admissible: list[str] | None = None
    replayed_done = False
    try:
        replayed_state, _observation, info, _score, replayed_done = _replay_prefix(env, gamefile=gamefile, prefix=prefix)
        replayed_admissible = _filtered_admissible_commands(_admissible_from_info(info))
    except Exception:
        replayed_state = None
        replayed_admissible = None
    for command, cache_key in zip(commands, cache_keys):
        if label_cache is not None and cache_key in label_cache:
            cached = label_cache[cache_key]
            labels.append(
                AlfworldDenseActionLabel(
                    episode=-1,
                    step=-1,
                    command=cached.command,
                    cost_to_success=cached.cost_to_success,
                    reachable=cached.reachable,
                    final_score=cached.final_score,
                    rollout_steps=cached.rollout_steps,
                    label_source=cached.label_source,
                    label_status=f"{cached.label_status}_cache_hit",
                )
            )
            continue
        try:
            state_copy = _copy_env_state(replayed_state)
            if state_copy is not None and replayed_admissible is not None:
                cost, reachable, score, next_done, rollout_steps, label_status = _run_with_timeout(
                    timeout_seconds,
                    lambda: _planner_cost_for_candidate_from_state(
                        state_copy,
                        command=command,
                        current_admissible=replayed_admissible,
                        done=replayed_done,
                        unreachable_cost=unreachable_cost,
                    ),
                )
            else:
                cost, reachable, score, next_done, rollout_steps, label_status = _run_with_timeout(
                    timeout_seconds,
                    lambda: _planner_cost_for_candidate(
                        env,
                        gamefile=gamefile,
                        prefix=prefix,
                        command=command,
                        unreachable_cost=unreachable_cost,
                    ),
                )
            label = AlfworldDenseActionLabel(
                episode=-1,
                step=-1,
                command=command,
                cost_to_success=max(1, int(cost)),
                reachable=1 if reachable else 0,
                final_score=1.0 if next_done and score > 0 else 0.0,
                rollout_steps=int(rollout_steps),
                label_source="textworld_pddl_replan_cost_to_go",
                label_status=label_status,
            )
        except TimeoutError:
            label = AlfworldDenseActionLabel(
                episode=-1,
                step=-1,
                command=command,
                cost_to_success=unreachable_cost,
                reachable=0,
                final_score=0.0,
                rollout_steps=0,
                label_source="textworld_pddl_replan_cost_to_go",
                label_status="planner_timeout",
            )
        except ValueError:
            label = AlfworldDenseActionLabel(
                episode=-1,
                step=-1,
                command=command,
                cost_to_success=unreachable_cost,
                reachable=0,
                final_score=0.0,
                rollout_steps=0,
                label_source="textworld_pddl_replan_cost_to_go",
                label_status="candidate_not_admissible_after_replay",
            )
        except Exception:
            label = AlfworldDenseActionLabel(
                episode=-1,
                step=-1,
                command=command,
                cost_to_success=unreachable_cost,
                reachable=0,
                final_score=0.0,
                rollout_steps=0,
                label_source="textworld_pddl_replan_cost_to_go",
                label_status="planner_error",
            )
        labels.append(label)
        if label_cache is not None:
            label_cache[cache_key] = label
        if cache_path is not None:
            _append_alfworld_planner_label_cache(cache_path, cache_key, label)
    return labels


def _planner_cost_for_candidate_from_state(
    env: Any,
    *,
    command: str,
    current_admissible: Sequence[str],
    done: bool,
    unreachable_cost: int,
) -> tuple[int, bool, float, bool, int, str]:
    if done:
        return 1, True, 1.0, True, 0, "already_done_snapshot"
    if command not in current_admissible:
        raise ValueError("candidate is not admissible in copied planner state")
    _next_observation, score, next_done, next_info = _step_any(env, command)
    if next_done and score > 0:
        return 1, True, score, True, 1, "candidate_solves_snapshot"
    remaining_plan = _planner_policy_commands_from_info(next_info)
    suffix_steps, suffix_reachable, suffix_score, suffix_done = _rollout_planner_suffix_to_success(
        env,
        remaining_plan,
        max_suffix_steps=max(0, unreachable_cost - 1),
    )
    reachable = bool(suffix_reachable)
    cost = 1 + suffix_steps if reachable else unreachable_cost
    return cost, reachable, suffix_score if reachable else score, suffix_done, cost if reachable else 0, "planner_replanned_snapshot" if reachable else "planner_no_plan_snapshot"


def _alfworld_planner_label_cache_key(
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    command: str,
    max_rollout_steps: int,
) -> str:
    payload = {
        "version": "alfworld-planner-label-v2",
        "gamefile": str(gamefile or ""),
        "prefix": [_normalize_command(item) for item in prefix],
        "command": _normalize_command(command),
        "max_rollout_steps": int(max_rollout_steps),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _load_alfworld_planner_label_cache(path: Path | None) -> dict[str, AlfworldDenseActionLabel]:
    if path is None or not path.exists():
        return {}
    cache: dict[str, AlfworldDenseActionLabel] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                label_row = row["label"]
                cache[str(row["key"])] = AlfworldDenseActionLabel(
                    episode=-1,
                    step=-1,
                    command=str(label_row["command"]),
                    cost_to_success=int(label_row["cost_to_success"]),
                    reachable=int(label_row["reachable"]),
                    final_score=float(label_row["final_score"]),
                    rollout_steps=int(label_row["rollout_steps"]),
                    label_source=str(label_row["label_source"]),
                    label_status=str(label_row.get("label_status", "ok")),
                    pairwise_usable=int(label_row.get("pairwise_usable", 1)),
                    oracle_rank=int(label_row.get("oracle_rank", 0)),
                    best_cost_to_success=int(label_row.get("best_cost_to_success", 0)),
                    cost_gap_to_best=int(label_row.get("cost_gap_to_best", 0)),
                )
            except Exception:
                continue
    return cache


def _append_alfworld_planner_label_cache(path: Path, key: str, label: AlfworldDenseActionLabel) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"key": key, "label": label.__dict__}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _planner_cost_for_candidate(
    env: Any,
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    command: str,
    unreachable_cost: int,
) -> tuple[int, bool, float, bool, int, str]:
    target, _observation, info, _score, done = _replay_prefix(env, gamefile=gamefile, prefix=prefix)
    if done:
        return 1, True, 1.0, True, 0, "already_done"
    current_admissible = _filtered_admissible_commands(_admissible_from_info(info))
    if command not in current_admissible:
        raise ValueError("candidate is not admissible in replayed planner state")
    _next_observation, score, next_done, next_info = _step_any(target, command)
    if next_done and score > 0:
        return 1, True, score, True, 1, "candidate_solves"
    remaining_plan = _planner_policy_commands_from_info(next_info)
    suffix_steps, suffix_reachable, suffix_score, suffix_done = _rollout_planner_suffix_to_success(
        target,
        remaining_plan,
        max_suffix_steps=max(0, unreachable_cost - 1),
    )
    reachable = bool(suffix_reachable)
    cost = 1 + suffix_steps if reachable else unreachable_cost
    return cost, reachable, suffix_score if reachable else score, suffix_done, cost if reachable else 0, "planner_replanned" if reachable else "planner_no_plan"


def _rollout_planner_suffix_to_success(
    env: Any,
    commands: Sequence[str],
    *,
    max_suffix_steps: int,
) -> tuple[int, bool, float, bool]:
    steps = 0
    score = 0.0
    done = False
    for command in commands[:max_suffix_steps]:
        _observation, score, done, info = _step_any(env, command)
        steps += 1
        if done and score > 0:
            return steps, True, score, done
        if done:
            return steps, False, score, done
    return steps, False, score, done


def _label_frontier_by_walkthrough_suffix(
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    commands: Sequence[str],
    max_rollout_steps: int,
    walkthrough_cache: dict[str, list[str]],
) -> list[AlfworldDenseActionLabel]:
    walkthrough = _walkthrough_for_gamefile(gamefile, cache=walkthrough_cache)
    prefix_norm = [_normalize_command(command) for command in prefix]
    walkthrough_norm = [_normalize_command(command) for command in walkthrough]
    aligned = prefix_norm == walkthrough_norm[: len(prefix_norm)]
    remaining = walkthrough_norm[len(prefix_norm) :] if aligned else []
    remaining_lookup = {command: idx for idx, command in enumerate(remaining)}
    unreachable_cost = 1 + max_rollout_steps + max(1, len(remaining))

    labels = []
    for command in commands:
        command_norm = _normalize_command(command)
        idx = remaining_lookup.get(command_norm)
        reachable = aligned and idx is not None
        cost = len(remaining) + int(idx) if reachable else unreachable_cost
        labels.append(
            AlfworldDenseActionLabel(
                episode=-1,
                step=-1,
                command=command,
                cost_to_success=max(1, cost),
                reachable=1 if reachable else 0,
                final_score=1.0 if reachable and idx == 0 and len(remaining) == 1 else 0.0,
                rollout_steps=1 + int(idx) if reachable else 0,
                label_source="walkthrough_suffix_cost_proxy",
            )
        )
    return labels


def _walkthrough_for_gamefile(gamefile: str | None, *, cache: dict[str, list[str]]) -> list[str]:
    if not gamefile:
        return []
    if gamefile in cache:
        return cache[gamefile]
    try:
        body = json.loads(Path(gamefile).read_text(encoding="utf-8"))
        walkthrough = body.get("walkthrough", [])
        commands = [str(command) for command in walkthrough if str(command).strip()]
    except Exception:
        commands = []
    cache[gamefile] = commands
    return commands


def _normalize_command(command: str) -> str:
    return " ".join(str(command).lower().strip().split())


def _normalize_alfworld_dense_label_source(label_source: str) -> str:
    normalized = str(label_source).strip().lower().replace("_", "-")
    aliases = {
        "replay": ALFWORLD_DENSE_LABEL_REPLAY,
        "rollout": ALFWORLD_DENSE_LABEL_REPLAY,
        "expert-rollout": ALFWORLD_DENSE_LABEL_REPLAY,
        "walkthrough-suffix": ALFWORLD_DENSE_LABEL_WALKTHROUGH,
        "walkthrough-suffix-cost": ALFWORLD_DENSE_LABEL_WALKTHROUGH,
        "pddl": ALFWORLD_DENSE_LABEL_PLANNER,
        "pddl-planner": ALFWORLD_DENSE_LABEL_PLANNER,
        "planner-cost": ALFWORLD_DENSE_LABEL_PLANNER,
        "textworld-planner": ALFWORLD_DENSE_LABEL_PLANNER,
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in ALFWORLD_DENSE_LABEL_SOURCES:
        raise ValueError(f"unknown ALFWorld dense label source: {label_source}")
    return normalized


def _alfworld_dense_label_source_name(label_source: str) -> str:
    normalized = _normalize_alfworld_dense_label_source(label_source)
    if normalized == ALFWORLD_DENSE_LABEL_PLANNER:
        return "textworld_pddl_replan_cost_to_go"
    if normalized == ALFWORLD_DENSE_LABEL_WALKTHROUGH:
        return "walkthrough_suffix_cost_proxy"
    return "replay_prefix_plus_expert_rollout_cost_proxy"


def _alfworld_dense_training_source(label_source: str) -> str:
    normalized = _normalize_alfworld_dense_label_source(label_source)
    if normalized == ALFWORLD_DENSE_LABEL_PLANNER:
        return "alfworld_train_textworld_pddl_planner_frontier_labels"
    if normalized == ALFWORLD_DENSE_LABEL_WALKTHROUGH:
        return "alfworld_train_walkthrough_suffix_frontier_labels"
    return "alfworld_train_replay_prefix_expert_rollout_frontier_labels"


def _copy_env_state(env: Any | None) -> Any | None:
    if env is None:
        return None
    if not hasattr(env, "copy"):
        try:
            return copy.deepcopy(env)
        except Exception:
            return None
    try:
        return env.copy()
    except Exception:
        try:
            return copy.deepcopy(env)
        except Exception:
            return None


def _run_with_timeout(timeout_seconds: float, fn: Callable[[], Any]) -> Any:
    if timeout_seconds <= 0:
        return fn()
    if "fork" in mp.get_all_start_methods():
        ctx = mp.get_context("fork")
        parent, child = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_run_timeout_child, args=(child, fn))
        process.start()
        child.close()
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1.0)
            parent.close()
            raise TimeoutError(f"ALFWorld planner candidate exceeded {timeout_seconds:.3f}s")
        if parent.poll():
            status, payload = parent.recv()
            parent.close()
            if status == "ok":
                return payload
            exc_type, message = payload
            if exc_type == "ValueError":
                raise ValueError(message)
            if exc_type == "TimeoutError":
                raise TimeoutError(message)
            raise RuntimeError(f"{exc_type}: {message}")
        parent.close()
        if process.exitcode == 0:
            return None
        raise RuntimeError(f"ALFWorld planner candidate worker exited with code {process.exitcode}")

    if not hasattr(signal, "SIGALRM"):
        return fn()

    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"ALFWorld planner candidate exceeded {timeout_seconds:.3f}s")

    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return fn()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)


def _run_timeout_child(conn: Any, fn: Callable[[], Any]) -> None:
    try:
        conn.send(("ok", fn()))
    except BaseException as exc:
        conn.send(("err", (exc.__class__.__name__, str(exc))))
    finally:
        conn.close()


def _dense_labels_to_pairs(
    *,
    labels: Sequence[AlfworldDenseActionLabel],
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible_commands: Sequence[str],
    max_steps: int,
    max_pairs: int,
    rng: random.Random,
    episode: int,
    step: int,
) -> list[tuple[tuple[float, ...], tuple[float, ...]]]:
    ordered = sorted((label for label in labels if _alfworld_dense_label_pairwise_usable(label)), key=lambda label: (label.cost_to_success, label.command))
    candidate_features = {
        label.command: alfworld_candidate_to_canonical(
            candidate_id=f"alfworld-dense:{episode}:{step}:{idx}",
            observation=observation,
            objective=objective,
            history=history,
            admissible_commands=admissible_commands,
            command=label.command,
            max_steps=max_steps,
        ).feature_vector()
        for idx, label in enumerate(ordered)
    }
    raw_pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    for better_idx, better in enumerate(ordered):
        for worse in ordered[better_idx + 1 :]:
            if better.cost_to_success < worse.cost_to_success:
                raw_pairs.append((candidate_features[better.command], candidate_features[worse.command]))
    rng.shuffle(raw_pairs)
    if max_pairs > 0:
        raw_pairs = raw_pairs[:max_pairs]
    return raw_pairs


def _annotate_alfworld_dense_label_ranks(labels: Sequence[AlfworldDenseActionLabel]) -> list[AlfworldDenseActionLabel]:
    usable = [label for label in labels if _alfworld_dense_label_pairwise_usable(label)]
    if not usable:
        return [
            AlfworldDenseActionLabel(
                episode=label.episode,
                step=label.step,
                command=label.command,
                cost_to_success=label.cost_to_success,
                reachable=label.reachable,
                final_score=label.final_score,
                rollout_steps=label.rollout_steps,
                label_source=label.label_source,
                label_status=label.label_status,
                pairwise_usable=0,
                oracle_rank=0,
                best_cost_to_success=0,
                cost_gap_to_best=0,
            )
            for label in labels
        ]
    best_cost = min(label.cost_to_success for label in usable)
    cost_ranks = {cost: idx + 1 for idx, cost in enumerate(sorted({label.cost_to_success for label in usable}))}
    annotated = []
    for label in labels:
        pairwise_usable = 1 if _alfworld_dense_label_pairwise_usable(label) else 0
        annotated.append(
            AlfworldDenseActionLabel(
                episode=label.episode,
                step=label.step,
                command=label.command,
                cost_to_success=label.cost_to_success,
                reachable=label.reachable,
                final_score=label.final_score,
                rollout_steps=label.rollout_steps,
                label_source=label.label_source,
                label_status=label.label_status,
                pairwise_usable=pairwise_usable,
                oracle_rank=cost_ranks.get(label.cost_to_success, 0) if pairwise_usable else 0,
                best_cost_to_success=best_cost,
                cost_gap_to_best=max(0, int(label.cost_to_success) - int(best_cost)) if pairwise_usable else 0,
            )
        )
    return annotated


def _alfworld_dense_label_pairwise_usable(label: AlfworldDenseActionLabel) -> bool:
    status = str(label.label_status or "").strip().lower()
    if not status:
        return True
    censored_markers = (
        "timeout",
        "planner_error",
        "candidate_not_admissible_after_replay",
    )
    return not any(marker in status for marker in censored_markers)


def _annotate_alfworld_dense_label_row_groups(action_label_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = {}
    group_order: list[tuple[int, int]] = []
    for row in action_label_rows:
        key = (int(row.get("episode", -1)), int(row.get("step", -1)))
        if key not in grouped:
            group_order.append(key)
        grouped.setdefault(key, []).append(row)

    annotated_rows: list[dict[str, object]] = []
    for key in group_order:
        labels = [
            AlfworldDenseActionLabel(
                episode=int(row.get("episode", -1)),
                step=int(row.get("step", -1)),
                command=str(row.get("command", "")),
                cost_to_success=int(row.get("cost_to_success", 0)),
                reachable=int(row.get("reachable", 0)),
                final_score=float(row.get("final_score", 0.0)),
                rollout_steps=int(row.get("rollout_steps", 0)),
                label_source=str(row.get("label_source", "")),
                label_status=str(row.get("label_status", "")),
                pairwise_usable=int(row.get("pairwise_usable", 1)),
                oracle_rank=int(row.get("oracle_rank", 0)),
                best_cost_to_success=int(row.get("best_cost_to_success", 0)),
                cost_gap_to_best=int(row.get("cost_gap_to_best", 0)),
            )
            for row in grouped[key]
        ]
        annotated = _annotate_alfworld_dense_label_ranks(labels)
        for row, label in zip(grouped[key], annotated):
            merged = dict(row)
            merged["pairwise_usable"] = label.pairwise_usable
            merged["oracle_rank"] = label.oracle_rank
            merged["best_cost_to_success"] = label.best_cost_to_success
            merged["cost_gap_to_best"] = label.cost_gap_to_best
            annotated_rows.append(merged)
    return annotated_rows


def _alfworld_dense_label_row_pairwise_usable(row: Mapping[str, object]) -> bool:
    return _alfworld_dense_label_pairwise_usable(
        AlfworldDenseActionLabel(
            episode=int(row.get("episode", -1)),
            step=int(row.get("step", -1)),
            command=str(row.get("command", "")),
            cost_to_success=int(row.get("cost_to_success", 0)),
            reachable=int(row.get("reachable", 0)),
            final_score=float(row.get("final_score", 0.0)),
            rollout_steps=int(row.get("rollout_steps", 0)),
            label_source=str(row.get("label_source", "")),
            label_status=str(row.get("label_status", "")),
            pairwise_usable=int(row.get("pairwise_usable", 1)),
            oracle_rank=int(row.get("oracle_rank", 0)),
            best_cost_to_success=int(row.get("best_cost_to_success", 0)),
            cost_gap_to_best=int(row.get("cost_gap_to_best", 0)),
        )
    )


def _alfworld_ranker_weight_diagnostics(ranker: GenericHarnessRanker, *, tolerance: float = 0.02) -> dict[str, object]:
    if not hasattr(ranker, "weights"):
        return {
            "direct_feature_weights": {},
            "contract_sign_violations": [],
            "contract_sign_violation_count": 0,
            "contract_sign_tolerance": tolerance,
        }
    names = canonical_feature_names()
    direct_feature_names = list(ALFWORLD_CONTRACT_SIGN_EXPECTATIONS)
    weights = {
        name: float(ranker.weights[names.index(name)])
        for name in direct_feature_names
        if name in names
    }
    violations = []
    for name, expected in ALFWORLD_CONTRACT_SIGN_EXPECTATIONS.items():
        value = weights.get(name)
        if value is None:
            continue
        if expected == "positive" and value < -tolerance:
            violations.append(f"{name}<0")
        if expected == "negative" and value > tolerance:
            violations.append(f"{name}>0")
    return {
        "direct_feature_weights": weights,
        "contract_sign_violations": violations,
        "contract_sign_violation_count": len(violations),
        "contract_sign_tolerance": tolerance,
    }


def _project_alfworld_ranker_contract_signs(ranker: GenericHarnessRanker) -> dict[str, object]:
    if not hasattr(ranker, "weights"):
        return {
            "applied": False,
            "scope": "direct_contract_action_features",
            "reason": "ranker_has_no_direct_linear_weights",
            "projected_weight_count": 0,
            "projected_features": [],
        }
    names = canonical_feature_names()
    projected_features: list[str] = []
    for feature, expected in ALFWORLD_CONTRACT_SIGN_EXPECTATIONS.items():
        if feature not in names:
            continue
        idx = names.index(feature)
        value = float(ranker.weights[idx])
        if expected == "positive" and value < 0.0:
            ranker.weights[idx] = 0.0
            projected_features.append(f"{feature}<0")
        elif expected == "negative" and value > 0.0:
            ranker.weights[idx] = 0.0
            projected_features.append(f"{feature}>0")
    return {
        "applied": True,
        "scope": "direct_contract_action_features",
        "projected_weight_count": len(projected_features),
        "projected_features": projected_features,
    }


def _alfworld_pairwise_ranker_diagnostics(
    ranker: GenericHarnessRanker,
    pairs: Sequence[tuple[tuple[float, ...], tuple[float, ...]]],
) -> dict[str, object]:
    if not pairs:
        return {
            "pair_count": 0,
            "ranking_accuracy": 0.0,
            "ranking_error": 0.0,
            "mean_margin": 0.0,
            "min_margin": 0.0,
            "p10_margin": 0.0,
            "median_margin": 0.0,
        }
    margins = sorted(float(ranker.score(positive) - ranker.score(negative)) for positive, negative in pairs)
    wrong = sum(1 for margin in margins if margin <= 0.0)
    return {
        "pair_count": len(pairs),
        "ranking_accuracy": 1.0 - wrong / len(pairs),
        "ranking_error": wrong / len(pairs),
        "mean_margin": sum(margins) / len(margins),
        "min_margin": margins[0],
        "p10_margin": margins[min(len(margins) - 1, max(0, int(0.1 * (len(margins) - 1))))],
        "median_margin": margins[len(margins) // 2],
    }


def _summarize_dense_label_rows(action_label_rows: Sequence[Mapping[str, object]], *, train_manifest: Mapping[str, object]) -> dict[str, object]:
    by_state: dict[tuple[int, int], list[int]] = {}
    usable_by_state: dict[tuple[int, int], list[int]] = {}
    status_counts: dict[str, int] = {}
    finite = 0
    infinite = 0
    pairwise_usable = 0
    costs: list[int] = []
    usable_cost_gaps: list[int] = []
    oracle_top_usable_labels = 0
    for row in action_label_rows:
        key = (int(row["episode"]), int(row["step"]))
        cost = int(row["cost_to_success"])
        reachable = int(row["reachable"])
        status = str(row.get("label_status", "ok") or "ok")
        status_counts[status] = status_counts.get(status, 0) + 1
        by_state.setdefault(key, []).append(cost)
        costs.append(cost)
        if _alfworld_dense_label_row_pairwise_usable(row):
            pairwise_usable += 1
            usable_by_state.setdefault(key, []).append(cost)
            usable_cost_gaps.append(int(row.get("cost_gap_to_best", max(0, cost - min(by_state[key])))))
            if int(row.get("oracle_rank", 0)) == 1:
                oracle_top_usable_labels += 1
        if reachable:
            finite += 1
        else:
            infinite += 1
    varied_states = sum(1 for values in by_state.values() if len(set(values)) >= 2)
    usable_varied_states = sum(1 for values in usable_by_state.values() if len(set(values)) >= 2)
    usable_top_tie_states = sum(1 for key, values in usable_by_state.items() if sum(1 for row in action_label_rows if (int(row["episode"]), int(row["step"])) == key and _alfworld_dense_label_row_pairwise_usable(row) and int(row.get("oracle_rank", 0)) == 1) > 1)
    comparable_states = len(by_state)
    usable_comparable_states = len(usable_by_state)
    summary = {
        "states": comparable_states,
        "action_labels": len(action_label_rows),
        "finite_action_labels": finite,
        "infinite_action_labels": infinite,
        "reachable_action_frac": 0.0 if not action_label_rows else finite / len(action_label_rows),
        "pairwise_usable_action_labels": pairwise_usable,
        "censored_action_labels": len(action_label_rows) - pairwise_usable,
        "censored_action_label_rate": 0.0 if not action_label_rows else (len(action_label_rows) - pairwise_usable) / len(action_label_rows),
        "oracle_top_usable_action_labels": oracle_top_usable_labels,
        "oracle_top_usable_action_frac": 0.0 if pairwise_usable == 0 else oracle_top_usable_labels / pairwise_usable,
        "usable_mean_cost_gap_to_best": 0.0 if not usable_cost_gaps else sum(usable_cost_gaps) / len(usable_cost_gaps),
        "usable_max_cost_gap_to_best": "" if not usable_cost_gaps else max(usable_cost_gaps),
        "states_with_cost_variation": varied_states,
        "cost_variation_rate": 0.0 if comparable_states == 0 else varied_states / comparable_states,
        "usable_states": usable_comparable_states,
        "usable_states_with_cost_variation": usable_varied_states,
        "usable_states_with_oracle_top_tie": usable_top_tie_states,
        "usable_cost_variation_rate": 0.0 if usable_comparable_states == 0 else usable_varied_states / usable_comparable_states,
        "unique_cost_values": len(set(costs)),
        "min_cost": "" if not costs else min(costs),
        "max_cost": "" if not costs else max(costs),
        "train_pair_count": int(train_manifest.get("train_pair_count", 0)),
        "expert_monotone_checks": int(train_manifest.get("expert_monotone_checks", 0)),
        "expert_monotone_failures": int(train_manifest.get("expert_monotone_failures", 0)),
        "expert_monotone_failure_rate": float(train_manifest.get("expert_monotone_failure_rate", 0.0)),
        "planner_path_cost_monotone_checks": int(train_manifest.get("planner_path_cost_monotone_checks", 0)),
        "planner_path_cost_monotone_failures": int(train_manifest.get("planner_path_cost_monotone_failures", 0)),
        "planner_path_cost_monotone_inconclusive": int(train_manifest.get("planner_path_cost_monotone_inconclusive", 0)),
        "planner_path_cost_monotone_failure_rate": float(train_manifest.get("planner_path_cost_monotone_failure_rate", 0.0)),
        "planner_path_cost_monotone_failure_example_count": len(train_manifest.get("planner_path_cost_monotone_failure_examples", [])),
        "planner_path_cost_monotone_inconclusive_example_count": len(train_manifest.get("planner_path_cost_monotone_inconclusive_examples", [])),
        "label_source": str(train_manifest.get("label_source", "")),
        "counterfactual_planner_cost_to_go": bool(train_manifest.get("counterfactual_planner_cost_to_go", False)),
        "planner_label_cache_loaded_entries": int(train_manifest.get("planner_label_cache_loaded_entries", 0)),
        "planner_label_cache_entries": int(train_manifest.get("planner_label_cache_entries", 0)),
        "new_planner_label_cache_hits": int(train_manifest.get("new_planner_label_cache_hits", 0)),
        "new_planner_label_cache_misses": int(train_manifest.get("new_planner_label_cache_misses", 0)),
        "planner_label_cache_hits": int(train_manifest.get("planner_label_cache_hits", 0)),
        "planner_label_cache_misses": int(train_manifest.get("planner_label_cache_misses", 0)),
        "dense_state_label_cache_rows": int(train_manifest.get("dense_state_label_cache_rows", 0)),
        "rebuilt_dense_pairs_from_state_cache": int(train_manifest.get("rebuilt_dense_pairs_from_state_cache", 0)),
        "ranker_contract_sign_violation_count": int(train_manifest.get("ranker_contract_sign_violation_count", 0)),
        "ranker_contract_sign_violations": "|".join(str(item) for item in train_manifest.get("ranker_contract_sign_violations", [])),
        "ranker_contract_sign_tolerance": float(train_manifest.get("ranker_contract_sign_tolerance", 0.0)),
    }
    for status, count in sorted(status_counts.items()):
        safe_status = re.sub(r"[^a-zA-Z0-9_]+", "_", status).strip("_") or "unknown"
        summary[f"label_status_{safe_status}"] = count
    summary["label_status_unique"] = len(status_counts)
    return summary


def _expert_rollout_cost_from_prefix(
    env: Any,
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    candidate: str,
    max_rollout_steps: int,
) -> tuple[int, bool, float, int]:
    try:
        target, observation, info, score, done = _replay_prefix(env, gamefile=gamefile, prefix=prefix)
        return _expert_rollout_cost_from_current_state(target, candidate=candidate, max_rollout_steps=max_rollout_steps)
    except Exception:
        return 1 + max_rollout_steps + 1, False, 0.0, 0


def _expert_rollout_cost_from_current_state(
    env: Any,
    *,
    candidate: str,
    max_rollout_steps: int,
) -> tuple[int, bool, float, int]:
    try:
        observation, score, done, info = _step_any(env, candidate)
        if done and score > 0:
            return 1, True, score, 1
        for step in range(max_rollout_steps):
            expert = _expert_action_from_info(info)
            admissible = _filtered_admissible_commands(_admissible_from_info(info))
            if expert not in admissible:
                return 1 + max_rollout_steps + 1, False, score, step + 1
            observation, score, done, info = _step_any(env, str(expert))
            if done and score > 0:
                return step + 2, True, score, step + 2
        return 1 + max_rollout_steps + 1, False, score, max_rollout_steps + 1
    except Exception:
        return 1 + max_rollout_steps + 1, False, 0.0, 0


def _replay_prefix(env: Any, *, gamefile: str | None, prefix: Sequence[str]) -> tuple[Any, str, Mapping[str, Any], float, bool]:
    target = _loadable_env(env)
    if gamefile is not None and hasattr(target, "load"):
        target.load(gamefile)
    observation, info = _reset_any(target)
    score = 0.0
    done = False
    for command in prefix:
        observation, score, done, info = _step_any(target, command)
        if done:
            break
    return target, observation, info, score, done


def _loadable_env(env: Any) -> Any:
    batch_env = getattr(env, "batch_env", None)
    envs = getattr(batch_env, "envs", None)
    if envs:
        return envs[0]
    return env


def _reset_any(env: Any) -> tuple[str, Mapping[str, Any]]:
    result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        observation = str(_first(obs))
        if isinstance(info, Mapping):
            return observation, info
        return observation, {}
    if isinstance(result, Mapping):
        return str(result.get("feedback", result.get("raw", ""))), result
    return str(result), {}


def _step_any(env: Any, command: str) -> tuple[str, float, bool, Mapping[str, Any]]:
    try:
        result = env.step(command)
    except Exception:
        result = env.step([command])
    if len(result) == 4:
        obs, score, done, info = result
        return str(_first(obs)), float(_first(score, score)), bool(_first(done, done)), info if isinstance(info, Mapping) else {}
    if len(result) == 3:
        state, score, done = result
        if isinstance(state, Mapping):
            return str(state.get("feedback", "")), float(score), bool(done), state
        if isinstance(state, tuple) and len(state) == 2:
            obs, info = state
            return str(obs), float(score), bool(done), info if isinstance(info, Mapping) else {}
    raise RuntimeError("unsupported ALFWorld step return shape")


def _load_or_train_ranker(
    *,
    ranker_checkpoint: Path | None,
    seed: int,
    source_train_count: int,
    source_epochs: int,
    source_max_states_per_task: int,
    source_difficulty: str,
) -> tuple[GenericHarnessRanker, dict[str, object]]:
    if ranker_checkpoint is not None:
        ranker = GenericHarnessRanker.from_json(ranker_checkpoint.read_text(encoding="utf-8"))
        return ranker, {"checkpoint": str(ranker_checkpoint), "trained_in_this_run": False}
    tasks = generate_workflow_source_tasks(
        source_train_count,
        seed=seed,
        anonymize_tools=True,
        families=SOURCE_WORKFLOW_FAMILIES,
        difficulty=source_difficulty,
    )
    ranker, pairs = train_generic_harness_ranker(
        tasks,
        epochs=source_epochs,
        seed=seed,
        action_intent_training=True,
        max_states_per_task=source_max_states_per_task,
    )
    return ranker, {
        "trained_in_this_run": True,
        "source": "structured_tool_workflow_v2",
        "train_pair_count": len(pairs),
        "source_generator": source_generator_manifest(seed, source_train_count, families=SOURCE_WORKFLOW_FAMILIES, difficulty=source_difficulty),
    }


def _make_alfworld_env(*, split: str, config_path: Path | None, env_factory: Callable[..., Any] | None, training_method: str = "dqn") -> Any:
    if env_factory is not None:
        return env_factory(split=split, config_path=config_path)
    try:
        import alfworld.agents.environment as environment
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ALFWorld is not installed. Install alfworld/textworld and run ALFWorld setup before using run-alfworld-transfer."
        ) from exc
    if config_path is not None:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise RuntimeError("ALFWorld config loading requires PyYAML.") from exc
        with config_path.open(encoding="utf-8") as reader:
            config = yaml.safe_load(reader)
    else:
        config = _default_alfworld_textworld_config(training_method=training_method)
    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    try:
        env_cls = environment.get_environment(env_type)
    except AttributeError:
        env_cls = getattr(environment, env_type)
    return env_cls(config, train_eval=split).init_env(batch_size=1)


def _make_alfworld_planner_env(*, env_factory: Callable[..., Any] | None = None) -> Any:
    if env_factory is not None:
        return env_factory(split="train", config_path=None)
    try:
        from textworld.core import EnvInfos
        from textworld.envs.pddl.pddl import PddlEnv
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    except ModuleNotFoundError as exc:
        raise RuntimeError("TextWorld PDDL planner support is not installed.") from exc
    base = PddlEnv(EnvInfos(won=True, admissible_commands=True, policy_commands=True, extras=["gamefile", "walkthrough"]))
    env = AlfredDemangler(env=base, shuffle=False)
    return AlfredInfos(env=env)


def _alfworld_env_method_for_policy(policy: str) -> str:
    return "dagger" if _is_alfworld_planner_diagnostic(policy) else "dqn"


def _is_alfworld_planner_diagnostic(policy: str) -> bool:
    return policy in {"ALFWorld expert-plan diagnostic", "ALFWorld planner diagnostic", "Frozen LLM shortlist + Planner diagnostic"}


def _default_alfworld_textworld_config(*, training_method: str = "dqn") -> dict[str, Any]:
    data_root = os.environ.get("ALFWORLD_DATA")
    if not data_root:
        data_root = str(Path("data/alfworld").resolve())
    root = Path(data_root)
    return {
        "general": {"training_method": training_method},
        "env": {
            "type": "AlfredTWEnv",
            "goal_desc_human_anns_prob": 0.0,
            "task_types": [1, 2, 3, 4, 5, 6],
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "dataset": {
            "data_path": str(root / "json_2.1.1" / "train"),
            "eval_id_data_path": str(root / "json_2.1.1" / "valid_seen"),
            "eval_ood_data_path": str(root / "json_2.1.1" / "valid_unseen"),
            "num_train_games": 0,
            "num_eval_games": 0,
        },
        "logic": {
            "domain": str(root / "logic" / "alfred.pddl"),
            "grammar": str(root / "logic" / "alfred.twl2"),
        },
        "rl": {"training": {"max_nb_steps_per_episode": 100}},
        "dagger": {"training": {"max_nb_steps_per_episode": 100}},
    }


def _assert_alfworld_available() -> None:
    if importlib.util.find_spec("alfworld") is None or importlib.util.find_spec("textworld") is None:
        raise RuntimeError(
            "ALFWorld/TextWorld is not installed. Install alfworld/textworld and run ALFWorld setup before using run-alfworld-transfer."
        )


def _run_episode(
    env: Any,
    *,
    policy: str,
    ranker: GenericHarnessRanker,
    seed: int,
    episode: int,
    max_steps: int,
    step_trace_rows: list[dict[str, object]] | None = None,
) -> AlfworldEpisodeResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    obs, info = env.reset()
    observation = _first(obs)
    objective = _objective_from_info(info, observation)
    gamefile = _gamefile_from_info(info)
    history: list[str] = [f"obs:{observation}"]
    prefix: list[str] = []
    walkthrough_cache: dict[str, list[str]] = {}
    final_score = 0.0
    invalid_actions = 0
    admissible_counts = []
    opportunities = 0
    steps = 0
    done = False
    for step in range(max_steps):
        admissible = _contract_filtered_admissible_commands(
            _admissible_from_info(info),
            observation=observation,
            objective=objective,
            history=history,
            max_steps=max_steps,
        )
        if not admissible:
            break
        admissible_counts.append(len(admissible))
        opportunities += 1 if len(admissible) >= 2 else 0
        diagnostic = _alfworld_diagnostic_action_with_source(
            info,
            gamefile=gamefile,
            prefix=prefix,
            admissible=admissible,
            walkthrough_cache=walkthrough_cache,
        )
        diagnostic_action = str(diagnostic.get("command", ""))
        command, selection = _select_command_with_trace(
            policy,
            observation,
            objective,
            history,
            admissible,
            ranker,
            rng,
            max_steps,
            expert=diagnostic_action or None,
        )
        selection = dict(selection)
        selection["diagnostic_command"] = diagnostic_action
        selection["diagnostic_source"] = str(diagnostic.get("source", "none"))
        selection["matches_diagnostic"] = 1 if diagnostic_action and command == diagnostic_action else 0
        if command not in admissible:
            invalid_actions += 1
        next_obs, scores, dones, info = env.step([command])
        final_score = float(_first(scores, 0.0))
        done = bool(_first(dones, False))
        if step_trace_rows is not None:
            step_trace_rows.append(
                _step_trace_row(
                    policy=policy,
                    episode=episode,
                    step=step,
                    command=command,
                    selection=selection,
                    observation=observation,
                    objective=objective,
                    history=history,
                    admissible=admissible,
                    max_steps=max_steps,
                    success_after_step=1 if done and final_score > 0 else 0,
                )
            )
        observation = str(_first(next_obs, ""))
        gamefile = _gamefile_from_info(info) or gamefile
        prefix.append(command)
        history.extend([f"act:{command}", f"obs:{observation}"])
        steps = step + 1
        if done:
            break
    return AlfworldEpisodeResult(
        policy=policy,
        episode=episode,
        success=1 if done and final_score > 0 else 0,
        final_score=final_score,
        steps=steps,
        mean_admissible_count=0.0 if not admissible_counts else sum(admissible_counts) / len(admissible_counts),
        opportunity_rate=0.0 if not admissible_counts else opportunities / len(admissible_counts),
        invalid_actions=invalid_actions,
        wall_clock_seconds=time.perf_counter() - start,
    )


def _run_llm_shortlist_episode(
    env: Any,
    *,
    policy: str,
    ranker: GenericHarnessRanker | GenericHarnessMLPRanker,
    shortlister: AlfworldLLMShortlistCache,
    seed: int,
    episode: int,
    max_steps: int,
    step_trace_rows: list[dict[str, object]] | None = None,
) -> AlfworldEpisodeResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    obs, info = env.reset()
    observation = _first(obs)
    objective = _objective_from_info(info, observation)
    gamefile = _gamefile_from_info(info)
    history: list[str] = [f"obs:{observation}"]
    prefix: list[str] = []
    walkthrough_cache: dict[str, list[str]] = {}
    final_score = 0.0
    invalid_actions = 0
    admissible_counts = []
    opportunities = 0
    steps = 0
    done = False
    token_total = 0
    call_total = 0
    for step in range(max_steps):
        admissible = _contract_filtered_admissible_commands(
            _admissible_from_info(info),
            observation=observation,
            objective=objective,
            history=history,
            max_steps=max_steps,
        )
        if not admissible:
            break
        admissible_counts.append(len(admissible))
        opportunities += 1 if len(admissible) >= 2 else 0
        diagnostic = _alfworld_diagnostic_action_with_source(
            info,
            gamefile=gamefile,
            prefix=prefix,
            admissible=admissible,
            walkthrough_cache=walkthrough_cache,
        )
        shortlist_record = shortlister.shortlist(
            objective=objective,
            observation=observation,
            history=history,
            admissible=admissible,
        )
        shortlist_indices = list(shortlist_record.get("ranked_indices", []))
        shortlist_commands = [
            admissible[idx]
            for idx in shortlist_indices[: shortlister.k]
            if isinstance(idx, int) and 0 <= idx < len(admissible)
        ]
        token_total += int(shortlist_record.get("input_tokens", 0)) + int(shortlist_record.get("output_tokens", 0))
        call_total += 1
        command, selection = _select_llm_shortlist_command(
            policy=policy,
            observation=observation,
            objective=objective,
            history=history,
            admissible=admissible,
            shortlist_indices=shortlist_indices,
            ranker=ranker,
            rng=rng,
            max_steps=max_steps,
            diagnostic_action=str(diagnostic.get("command", "")),
        )
        selection = dict(selection)
        selection["diagnostic_command"] = str(diagnostic.get("command", ""))
        selection["diagnostic_source"] = str(diagnostic.get("source", "none"))
        selection["matches_diagnostic"] = 1 if selection["diagnostic_command"] and command == selection["diagnostic_command"] else 0
        selection["diagnostic_in_shortlist"] = 1 if selection["diagnostic_command"] and selection["diagnostic_command"] in shortlist_commands else 0
        selection["llm_shortlist"] = " | ".join(shortlist_commands)
        selection["llm_shortlist_size"] = len(shortlist_commands)
        selection["llm_cache_hit"] = 1 if shortlist_record.get("cache_hit") else 0
        selection["llm_input_tokens"] = int(shortlist_record.get("input_tokens", 0))
        selection["llm_output_tokens"] = int(shortlist_record.get("output_tokens", 0))
        if command not in admissible:
            invalid_actions += 1
        next_obs, scores, dones, info = env.step([command])
        final_score = float(_first(scores, 0.0))
        done = bool(_first(dones, False))
        if step_trace_rows is not None:
            row = _step_trace_row(
                policy=policy,
                episode=episode,
                step=step,
                command=command,
                selection=selection,
                observation=observation,
                objective=objective,
                history=history,
                admissible=admissible,
                max_steps=max_steps,
                success_after_step=1 if done and final_score > 0 else 0,
            )
            row.update(
                {
                    "llm_shortlist": selection["llm_shortlist"],
                    "llm_shortlist_size": selection["llm_shortlist_size"],
                    "diagnostic_in_shortlist": selection["diagnostic_in_shortlist"],
                    "llm_cache_hit": selection["llm_cache_hit"],
                    "llm_input_tokens": selection["llm_input_tokens"],
                    "llm_output_tokens": selection["llm_output_tokens"],
                }
            )
            step_trace_rows.append(row)
        observation = str(_first(next_obs, ""))
        gamefile = _gamefile_from_info(info) or gamefile
        prefix.append(command)
        history.extend([f"act:{command}", f"obs:{observation}"])
        steps = step + 1
        if done:
            break
    return AlfworldEpisodeResult(
        policy=policy,
        episode=episode,
        success=1 if done and final_score > 0 else 0,
        final_score=final_score,
        steps=steps,
        mean_admissible_count=0.0 if not admissible_counts else sum(admissible_counts) / len(admissible_counts),
        opportunity_rate=0.0 if not admissible_counts else opportunities / len(admissible_counts),
        invalid_actions=invalid_actions,
        wall_clock_seconds=time.perf_counter() - start,
        tokens=token_total,
        llm_calls=call_total,
    )


def _select_command(
    policy: str,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible: Sequence[str],
    ranker: GenericHarnessRanker,
    rng: random.Random,
    max_steps: int,
    expert: str | None = None,
) -> str:
    return _select_command_with_trace(policy, observation, objective, history, admissible, ranker, rng, max_steps, expert=expert)[0]


def _select_command_with_trace(
    policy: str,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible: Sequence[str],
    ranker: GenericHarnessRanker,
    rng: random.Random,
    max_steps: int,
    expert: str | None = None,
) -> tuple[str, dict[str, object]]:
    if policy == "Random":
        command = rng.choice(list(admissible))
        return command, {"score": "", "second_best_score": "", "score_margin": "", "expert_available": 0, "top_ranked_commands": "", "candidate_score_summary": ""}
    if _is_alfworld_planner_diagnostic(policy):
        if expert in admissible:
            return str(expert), {"score": "", "second_best_score": "", "score_margin": "", "expert_available": 1, "top_ranked_commands": "", "candidate_score_summary": ""}
        return rng.choice(list(admissible)), {"score": "", "second_best_score": "", "score_margin": "", "expert_available": 0, "top_ranked_commands": "", "candidate_score_summary": ""}
    if policy == "Hand heuristic":
        scored = [(_hand_score(command, observation, objective, history), -idx, command) for idx, command in enumerate(admissible)]
        scored.sort(reverse=True)
        scored = _prefer_non_stagnating_selection(
            scored=scored,
            observation=observation,
            objective=objective,
            history=history,
            admissible=admissible,
            max_steps=max_steps,
        )
        score = scored[0][0]
        second = scored[1][0] if len(scored) >= 2 else ""
        return scored[0][2], {
            "score": score,
            "second_best_score": second,
            "score_margin": "" if second == "" else score - float(second),
            "expert_available": 0,
            "top_ranked_commands": _ranked_command_summary(scored),
            "candidate_score_summary": _candidate_score_summary(scored),
        }
    if policy in {"Source-trained controller", "ALFWorld expert-ranker", "ALFWorld dense-ranker"}:
        scored = []
        for idx, command in enumerate(admissible):
            candidate = alfworld_candidate_to_canonical(
                candidate_id=f"alfworld:{len(history)}:{idx}",
                observation=observation,
                objective=objective,
                history=history,
                admissible_commands=admissible,
                command=command,
                max_steps=max_steps,
            )
            scored.append((ranker.score(candidate.feature_vector()), -idx, command))
        scored.sort(reverse=True)
        scored = _prefer_non_stagnating_selection(
            scored=scored,
            observation=observation,
            objective=objective,
            history=history,
            admissible=admissible,
            max_steps=max_steps,
        )
        score = scored[0][0]
        second = scored[1][0] if len(scored) >= 2 else ""
        return scored[0][2], {
            "score": score,
            "second_best_score": second,
            "score_margin": "" if second == "" else score - float(second),
            "expert_available": 0,
            "top_ranked_commands": _ranked_command_summary(scored),
            "candidate_score_summary": _candidate_score_summary(scored),
        }
    raise ValueError(f"unknown ALFWorld policy: {policy}")


def _select_llm_shortlist_command(
    *,
    policy: str,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible: Sequence[str],
    shortlist_indices: Sequence[int],
    ranker: GenericHarnessRanker | GenericHarnessMLPRanker,
    rng: random.Random,
    max_steps: int,
    diagnostic_action: str = "",
) -> tuple[str, dict[str, object]]:
    clean_indices: list[int] = []
    seen: set[int] = set()
    for item in shortlist_indices:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < len(admissible) and idx not in seen:
            clean_indices.append(idx)
            seen.add(idx)
    shortlist = [admissible[idx] for idx in clean_indices]
    if not shortlist:
        shortlist = list(admissible)
    if policy == "Frozen LLM top-1":
        return shortlist[0], {
            "score": "",
            "second_best_score": "",
            "score_margin": "",
            "expert_available": 0,
            "top_ranked_commands": " | ".join(shortlist[:5]),
            "candidate_score_summary": "",
        }
    if policy == "Frozen LLM shortlist + Random":
        command = rng.choice(shortlist)
        return command, {
            "score": "",
            "second_best_score": "",
            "score_margin": "",
            "expert_available": 0,
            "top_ranked_commands": " | ".join(shortlist[:5]),
            "candidate_score_summary": "",
        }
    if policy == "Frozen LLM shortlist + Planner diagnostic":
        if diagnostic_action in shortlist:
            command = diagnostic_action
        else:
            command = shortlist[0]
        return command, {
            "score": "",
            "second_best_score": "",
            "score_margin": "",
            "expert_available": 1 if diagnostic_action in shortlist else 0,
            "top_ranked_commands": " | ".join(shortlist[:5]),
            "candidate_score_summary": "",
        }
    if policy == "Frozen LLM shortlist + Dense-ranker":
        scored = []
        for idx, command in enumerate(shortlist):
            candidate = alfworld_candidate_to_canonical(
                candidate_id=f"alfworld-llm-shortlist:{len(history)}:{idx}",
                observation=observation,
                objective=objective,
                history=history,
                admissible_commands=admissible,
                command=command,
                max_steps=max_steps,
            )
            scored.append((ranker.score(candidate.feature_vector()), -idx, command))
        scored.sort(reverse=True)
        scored = _prefer_non_stagnating_selection(
            scored=scored,
            observation=observation,
            objective=objective,
            history=history,
            admissible=shortlist,
            max_steps=max_steps,
        )
        score = scored[0][0]
        second = scored[1][0] if len(scored) >= 2 else ""
        return scored[0][2], {
            "score": score,
            "second_best_score": second,
            "score_margin": "" if second == "" else score - float(second),
            "expert_available": 0,
            "top_ranked_commands": _ranked_command_summary(scored),
            "candidate_score_summary": _candidate_score_summary(scored),
        }
    raise ValueError(f"unknown ALFWorld LLM-shortlist policy: {policy}")


def _prefer_non_stagnating_selection(
    *,
    scored: list[tuple[float, int, str]],
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible: Sequence[str],
    max_steps: int,
) -> list[tuple[float, int, str]]:
    if len(scored) <= 1 or not _is_stagnating_noop(scored[0][2], observation=observation, objective=objective, history=history, admissible=admissible, max_steps=max_steps):
        return scored
    for idx in range(1, len(scored)):
        _score, _tie_breaker, command = scored[idx]
        if not _is_stagnating_noop(command, observation=observation, objective=objective, history=history, admissible=admissible, max_steps=max_steps):
            return [scored[idx], *scored[:idx], *scored[idx + 1 :]]
    return scored


def _ranked_command_summary(scored: Sequence[tuple[float, int, str]], *, limit: int = 5) -> str:
    return " | ".join(command for _score, _idx, command in list(scored)[:limit])


def _candidate_score_summary(scored: Sequence[tuple[float, int, str]], *, limit: int = 8) -> str:
    return " | ".join(f"{score:.4f}:{command}" for score, _idx, command in list(scored)[:limit])


def _is_stagnating_noop(
    command: str,
    *,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible: Sequence[str],
    max_steps: int,
) -> bool:
    last_action = next((item[4:].strip() for item in reversed(history) if item.startswith("act:")), "")
    previous_actions = {item[4:].strip() for item in history if item.startswith("act:")}
    candidate = alfworld_candidate_to_canonical(
        candidate_id="alfworld-stagnation-check",
        observation=observation,
        objective=objective,
        history=history,
        admissible_commands=admissible,
        command=command,
        max_steps=max_steps,
    )
    if candidate.action.action_is_write == 1.0 and candidate.action.declared_observation_gain == 0.0 and candidate.action.declared_requirement_gain == 0.0:
        return True
    if command != last_action and command not in previous_actions:
        return False
    return candidate.action.declared_observation_gain == 0.0 and candidate.action.declared_requirement_gain == 0.0


def _step_trace_row(
    *,
    policy: str,
    episode: int,
    step: int,
    command: str,
    selection: Mapping[str, object],
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible: Sequence[str],
    max_steps: int,
    success_after_step: int,
) -> dict[str, object]:
    candidate = alfworld_candidate_to_canonical(
        candidate_id=f"alfworld-trace:{episode}:{step}",
        observation=observation,
        objective=objective,
        history=history,
        admissible_commands=admissible,
        command=command,
        max_steps=max_steps,
    )
    trace = AlfworldStepTrace(
        policy=policy,
        episode=episode,
        step=step,
        command=command,
        objective=objective,
        command_type=_command_type(command),
        score=selection.get("score", ""),
        second_best_score=selection.get("second_best_score", ""),
        score_margin=selection.get("score_margin", ""),
        expert_available=int(selection.get("expert_available", 0)),
        diagnostic_command=str(selection.get("diagnostic_command", "")),
        diagnostic_source=str(selection.get("diagnostic_source", "")),
        matches_diagnostic=int(selection.get("matches_diagnostic", 0)),
        admissible_count=len(admissible),
        verified_progress_frac=candidate.state.verified_progress_frac,
        remaining_requirement_frac=candidate.state.remaining_requirement_frac,
        observed_fact_frac=candidate.state.observed_fact_frac,
        uncertainty_score=candidate.state.uncertainty_score,
        declared_observation_gain=candidate.action.declared_observation_gain,
        declared_requirement_gain=candidate.action.declared_requirement_gain,
        references_goal_object=candidate.action.references_goal_object,
        references_goal_receptacle=candidate.action.references_goal_receptacle,
        matches_unsatisfied_transform=candidate.action.matches_unsatisfied_transform,
        goal_object_held=candidate.action.goal_object_held,
        goal_receptacle_observed=candidate.action.goal_receptacle_observed,
        placement_ready=candidate.action.placement_ready,
        completes_goal_placement=candidate.action.completes_goal_placement,
        precondition_satisfied=candidate.action.precondition_satisfied,
        policy_risk=candidate.action.policy_risk,
        duplicates_previous_action=candidate.action.duplicates_previous_action,
        top_ranked_commands=str(selection.get("top_ranked_commands", "")),
        candidate_score_summary=str(selection.get("candidate_score_summary", "")),
        success_after_step=success_after_step,
    )
    return trace.__dict__


def _expert_action_from_info(info: Mapping[str, Any]) -> str | None:
    for key in ("extra.expert_plan", "expert_plan"):
        value = info.get(key)
        if value:
            plan = _first(value, value)
            action = _first(plan, plan)
            if action:
                return str(action)
    extras = info.get("extras")
    if isinstance(extras, Mapping):
        value = extras.get("expert_plan")
        if value:
            plan = _first(value, value)
            action = _first(plan, plan)
            if action:
                return str(action)
    return None


def _alfworld_diagnostic_action_from_info(
    info: Mapping[str, Any],
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    admissible: Sequence[str],
    walkthrough_cache: dict[str, list[str]],
) -> str | None:
    command = _alfworld_diagnostic_action_with_source(
        info,
        gamefile=gamefile,
        prefix=prefix,
        admissible=admissible,
        walkthrough_cache=walkthrough_cache,
    ).get("command")
    return str(command) if command else None


def _alfworld_diagnostic_action_with_source(
    info: Mapping[str, Any],
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    admissible: Sequence[str],
    walkthrough_cache: dict[str, list[str]],
) -> dict[str, str]:
    expert = _expert_action_from_info(info)
    matched = _match_admissible_command(expert, admissible)
    if matched is not None:
        return {"command": matched, "source": "expert_plan"}
    for command in _planner_policy_commands_from_info(info):
        matched = _match_admissible_command(command, admissible)
        if matched is not None:
            return {"command": matched, "source": "policy_commands"}
    walkthrough_next = _walkthrough_next_command(gamefile=gamefile, prefix=prefix, walkthrough_cache=walkthrough_cache)
    matched = _match_admissible_command(walkthrough_next, admissible)
    if matched is not None:
        return {"command": matched, "source": "walkthrough"}
    return {"command": "", "source": "none"}


def _match_admissible_command(command: str | None, admissible: Sequence[str]) -> str | None:
    if command is None:
        return None
    if command in admissible:
        return str(command)
    target = _normalize_command(command)
    for candidate in admissible:
        if _normalize_command(candidate) == target:
            return str(candidate)
    return None


def _training_rollout_command(
    *,
    expert: str | None,
    admissible: Sequence[str],
    rng: random.Random,
    step: int,
    perturbation_prob: float,
    warmup_random_steps: int,
) -> str:
    if not admissible:
        raise ValueError("cannot roll out ALFWorld state without admissible commands")
    if step < warmup_random_steps or rng.random() < perturbation_prob or expert not in admissible:
        return rng.choice(list(admissible))
    return str(expert)


def _walkthrough_training_rollout_command(
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    expert: str | None,
    admissible: Sequence[str],
    rng: random.Random,
    step: int,
    perturbation_prob: float,
    warmup_random_steps: int,
    walkthrough_cache: dict[str, list[str]],
) -> tuple[str, bool]:
    if not admissible:
        raise ValueError("cannot roll out ALFWorld state without admissible commands")
    if step < warmup_random_steps or rng.random() < perturbation_prob:
        return rng.choice(list(admissible)), False
    next_command = _walkthrough_next_command(gamefile=gamefile, prefix=prefix, walkthrough_cache=walkthrough_cache)
    if next_command in admissible:
        return str(next_command), True
    if expert in admissible:
        return str(expert), False
    return rng.choice(list(admissible)), False


def _planner_training_rollout_command(
    env: Any,
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    expert: str | None,
    admissible: Sequence[str],
    rng: random.Random,
    step: int,
    perturbation_prob: float,
    warmup_random_steps: int,
    timeout_seconds: float = ALFWORLD_PLANNER_LABEL_TIMEOUT_SECONDS,
) -> str:
    if not admissible:
        raise ValueError("cannot roll out ALFWorld state without admissible commands")
    if step < warmup_random_steps or rng.random() < perturbation_prob:
        return rng.choice(list(admissible))
    try:
        planner_next = _run_with_timeout(
            timeout_seconds,
            lambda: _planner_next_command(env, gamefile=gamefile, prefix=prefix, admissible=admissible),
        )
    except TimeoutError:
        planner_next = None
    if planner_next is not None:
        return planner_next
    if expert in admissible:
        return str(expert)
    return rng.choice(list(admissible))


def _best_reachable_label_command(labels: Sequence[AlfworldDenseActionLabel], *, admissible: Sequence[str]) -> str | None:
    admissible_set = set(admissible)
    reachable = [label for label in labels if label.reachable and label.command in admissible_set]
    if not reachable:
        return None
    best = min(reachable, key=lambda label: (label.cost_to_success, label.command))
    return best.command


def _planner_next_command(env: Any, *, gamefile: str | None, prefix: Sequence[str], admissible: Sequence[str]) -> str | None:
    try:
        _target, _observation, info, _score, done = _replay_prefix(env, gamefile=gamefile, prefix=prefix)
        if done:
            return None
        for command in _planner_policy_commands_from_info(info):
            if command in admissible:
                return command
    except Exception:
        return None
    return None


def _planner_policy_commands_from_info(info: Mapping[str, Any]) -> list[str]:
    values: Any = info.get("policy_commands")
    if values is None:
        values = info.get("extra.policy_commands")
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, Sequence):
        commands = []
        for value in values:
            if isinstance(value, str):
                commands.append(value)
            elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
                commands.extend(str(item) for item in value if str(item).strip())
        return [command for command in commands if command.strip()]
    return []


def _walkthrough_next_command(*, gamefile: str | None, prefix: Sequence[str], walkthrough_cache: dict[str, list[str]]) -> str | None:
    walkthrough = _walkthrough_for_gamefile(gamefile, cache=walkthrough_cache)
    prefix_norm = [_normalize_command(command) for command in prefix]
    walkthrough_norm = [_normalize_command(command) for command in walkthrough]
    if prefix_norm != walkthrough_norm[: len(prefix_norm)]:
        return None
    if len(prefix_norm) >= len(walkthrough):
        return None
    return walkthrough[len(prefix_norm)]


def _hand_score(command: str, observation: str, objective: str, history: Sequence[str]) -> float:
    action_type = _command_type(command)
    repeats = any(item == f"act:{command}" for item in history)
    score = -2.0 if repeats else 0.0
    if action_type == "help":
        score -= 5.0
    elif action_type == "read":
        score += 2.5
    elif action_type == "write":
        score += 2.0
    elif action_type == "move":
        score += 1.0
    elif action_type == "terminate":
        score -= 4.0
    if _object_tokens(command.lower()) & _object_tokens(objective.lower()):
        score += 1.5
    if _object_tokens(command.lower()) & _object_tokens(observation.lower()):
        score += 0.5
    return score


def _command_type(command: str) -> str:
    lowered = command.lower().strip()
    if lowered.startswith(HELP_VERBS):
        return "help"
    if lowered.startswith(TERMINATE_VERBS):
        return "terminate"
    if lowered.startswith(READ_VERBS):
        return "read"
    if lowered.startswith(MOVE_VERBS):
        return "move"
    if lowered.startswith(OPEN_VERBS):
        return "read"
    if lowered.startswith(TAKE_VERBS) or lowered.startswith(PUT_VERBS) or lowered.startswith(USE_VERBS):
        return "write"
    return "write"


def _alfworld_goal_spec(objective: str) -> AlfworldGoalSpec:
    lowered = objective.lower().replace("_", " ")
    raw_terms = _object_tokens(lowered)
    transform_terms = frozenset(term for term in TRANSFORM_VERBS if re.search(rf"\b{re.escape(term)}\w*\b", lowered))
    object_terms: set[str] = set()
    receptacle_terms: set[str] = set()

    put_match = re.search(r"\bput\b\s+(?P<object>.+?)\s+\b(?:in|on|into|onto)\b\s+(?P<receptacle>.+)$", lowered)
    if put_match:
        object_terms.update(_goal_phrase_terms(put_match.group("object"), drop_transforms=True))
        receptacle_terms.update(_goal_phrase_terms(put_match.group("receptacle"), drop_transforms=False))
    else:
        examine_match = re.search(r"\b(?:examine|look at)\b\s+(?P<object>.+?)\s+\b(?:with|under|using)\b\s+(?P<receptacle>.+)$", lowered)
        if examine_match:
            object_terms.update(_goal_phrase_terms(examine_match.group("object"), drop_transforms=True))
            receptacle_terms.update(_goal_phrase_terms(examine_match.group("receptacle"), drop_transforms=False))
        else:
            pickup_match = re.search(r"\b(?:pick up|take|get)\b\s+(?P<object>.+)$", lowered)
            if pickup_match:
                object_terms.update(_goal_phrase_terms(pickup_match.group("object"), drop_transforms=True))

    if not object_terms:
        object_terms.update(term for term in raw_terms if term not in _goal_function_words() and term not in transform_terms)
    object_terms.difference_update(receptacle_terms)
    return AlfworldGoalSpec(
        object_terms=frozenset(object_terms),
        receptacle_terms=frozenset(receptacle_terms),
        transform_terms=transform_terms,
        raw_terms=frozenset(raw_terms),
    )


def _goal_phrase_terms(phrase: str, *, drop_transforms: bool) -> set[str]:
    terms = _object_tokens(phrase)
    terms.difference_update(_goal_function_words())
    if drop_transforms:
        terms.difference_update(TRANSFORM_VERBS)
    return terms


def _goal_function_words() -> set[str]:
    return {
        "put",
        "place",
        "pick",
        "take",
        "get",
        "find",
        "examine",
        "look",
        "with",
        "under",
        "using",
        "into",
        "onto",
        "two",
        "some",
        "one",
        "all",
        "cleaned",
        "heated",
        "cooled",
        "sliced",
    }


def _observed_terms(*, observation: str, history: Sequence[str]) -> set[str]:
    texts = [_strip_task_text(observation)]
    texts.extend(_strip_task_text(item[4:]) for item in history if item.startswith("obs:"))
    return _object_tokens(" ".join(texts))


def _held_terms(*, observation: str, history: Sequence[str]) -> set[str]:
    held = set()
    for action in [item[4:].strip().lower() for item in history if item.startswith("act:")]:
        if action.startswith(TAKE_VERBS):
            held.update(_command_object_terms(action))
        if action.startswith(PUT_VERBS):
            held.difference_update(_command_object_terms(action))
    holding_text = " ".join(
        _strip_task_text(text)
        for text in [observation, *(item[4:] for item in history if item.startswith("obs:"))]
        if "holding" in text.lower() or "carrying" in text.lower() or "inventory" in text.lower()
    )
    held.update(_object_tokens(holding_text))
    held.difference_update({"holding", "carrying", "inventory", "nothing"})
    return held


def _relational_progress(*, goal: AlfworldGoalSpec, observation: str, history: Sequence[str]) -> float:
    observed_terms = _observed_terms(observation=observation, history=history)
    held_terms = _held_terms(observation=observation, history=history)
    actions = [item[4:].strip().lower() for item in history if item.startswith("act:")]
    completed = 0.0
    total = 0.0

    if goal.object_terms:
        total += 1.0
        if goal.object_terms & observed_terms:
            completed += 0.5
        if goal.object_terms & held_terms:
            completed += 0.5

    if goal.transform_terms:
        total += 1.0
        for action in actions:
            action_terms = _object_tokens(action)
            if action.startswith(tuple(goal.transform_terms)) and (not goal.object_terms or action_terms & goal.object_terms):
                completed += 1.0
                break

    if goal.receptacle_terms:
        total += 1.0
        if goal.receptacle_terms & observed_terms:
            completed += 0.5
        for action in actions:
            action_terms = _object_tokens(action)
            if action.startswith(PUT_VERBS) and action_terms & goal.receptacle_terms and (not goal.object_terms or action_terms & goal.object_terms):
                completed += 0.5
                break

    if total == 0.0:
        return _progress_proxy(objective=" ".join(goal.raw_terms), observation=observation, history=history)
    return min(1.0, completed / total)


def _action_goal_relation(
    *,
    command: str,
    action_type: str,
    goal: AlfworldGoalSpec,
    observed_terms: set[str],
    held_terms: set[str],
    repeats: bool,
    recent_move_streak: int = 0,
    recent_read_streak: int = 0,
) -> dict[str, float]:
    lowered = command.lower().strip()
    command_terms = _object_tokens(lowered)
    action_object_terms = _command_object_terms(lowered)
    action_receptacle_terms = _command_receptacle_terms(lowered)
    goal_object_ref = bool(command_terms & goal.object_terms) if goal.object_terms else False
    goal_receptacle_ref = bool(command_terms & goal.receptacle_terms) if goal.receptacle_terms else False
    transform_ref = bool(goal.transform_terms and any(lowered.startswith(term) for term in goal.transform_terms))
    object_observed = bool(goal.object_terms & observed_terms) if goal.object_terms else True
    object_held = bool(goal.object_terms & held_terms) if goal.object_terms else False
    receptacle_observed = bool(goal.receptacle_terms & observed_terms) if goal.receptacle_terms else True
    action_object_goal_ref = bool(action_object_terms & goal.object_terms) if goal.object_terms else False
    action_receptacle_goal_ref = bool(action_receptacle_terms & goal.receptacle_terms) if goal.receptacle_terms else False
    placement_ready = action_type == "write" and action_object_goal_ref and action_receptacle_goal_ref and object_held and receptacle_observed
    productive_repeated_goal_navigation = action_type == "move" and object_held and goal_receptacle_ref

    observation_gain = 0.0
    requirement_gain = 0.0
    precondition = 1.0
    risk = 0.9 if action_type == "help" else (0.1 if action_type in {"read", "move"} else (0.35 if action_type == "write" else 0.0))

    if repeats:
        risk += 0.25
        if action_type in {"read", "help"}:
            risk += 0.35
    if action_type in {"read", "move"}:
        missing_object = goal.object_terms and not object_observed
        missing_receptacle = goal.receptacle_terms and not receptacle_observed
        if not repeats and (missing_object or missing_receptacle):
            if goal_object_ref or goal_receptacle_ref:
                observation_gain = 1.0
            elif lowered in {"look", "inventory"}:
                observation_gain = 0.5
        if action_type == "move" and object_held and goal_receptacle_ref:
            requirement_gain = max(requirement_gain, 0.5)

    if action_type == "help":
        precondition = 0.0
    elif lowered.startswith(TAKE_VERBS):
        requirement_gain = 1.0 if goal_object_ref and object_observed and not object_held and not repeats else 0.0
        precondition = 1.0 if object_observed else 0.0
    elif lowered.startswith(PUT_VERBS):
        requirement_gain = 1.0 if placement_ready and not repeats else 0.0
        precondition = 1.0 if (object_held or not goal.object_terms) and (receptacle_observed or not goal.receptacle_terms) else 0.0
        if goal.object_terms and goal.receptacle_terms and action_object_goal_ref and object_held and not action_receptacle_goal_ref:
            risk += 0.4
    elif lowered.startswith(USE_VERBS):
        requirement_gain = 1.0 if (transform_ref or bool(command_terms & goal.transform_terms)) and (object_held or object_observed) and not repeats else 0.0
        precondition = 1.0 if object_held or object_observed or goal_object_ref else 0.0
    elif action_type == "terminate":
        precondition = 1.0 if _goal_ready_to_finish(goal=goal, observed_terms=observed_terms, held_terms=held_terms) else 0.0
        risk = 0.8 if precondition == 0.0 else 0.2

    if action_type == "write" and precondition == 0.0:
        risk += 0.3
    zero_gain = observation_gain == 0.0 and requirement_gain == 0.0
    if action_type == "move":
        if zero_gain and not goal_receptacle_ref:
            risk += 0.12
        if zero_gain and recent_move_streak >= 2:
            risk += 0.35
        elif zero_gain and recent_move_streak == 1:
            risk += 0.18
        if object_held and goal.receptacle_terms and not action_receptacle_goal_ref and not goal_receptacle_ref:
            risk += 0.25
        if object_held and (action_receptacle_goal_ref or goal_receptacle_ref):
            risk = min(risk, 0.08)
    if action_type == "read" and zero_gain and recent_read_streak >= 2:
        risk += 0.25
    if lowered.startswith(TAKE_VERBS) and requirement_gain > 0.0 and precondition == 1.0:
        risk = min(risk, 0.08)
    if lowered.startswith(USE_VERBS) and requirement_gain > 0.0 and precondition == 1.0:
        risk = min(risk, 0.08)
    if placement_ready and requirement_gain > 0.0:
        risk = min(risk, 0.03)
    return {
        "references_goal_object": 1.0 if action_object_goal_ref or goal_object_ref else 0.0,
        "references_goal_receptacle": 1.0 if action_receptacle_goal_ref or goal_receptacle_ref else 0.0,
        "matches_unsatisfied_transform": 1.0 if transform_ref else 0.0,
        "goal_object_held": 1.0 if object_held else 0.0,
        "goal_receptacle_observed": 1.0 if receptacle_observed else 0.0,
        "placement_ready": 1.0 if placement_ready else 0.0,
        "completes_goal_placement": 1.0 if placement_ready and not repeats else 0.0,
        "observation_gain": 0.0 if repeats else min(1.0, observation_gain),
        "requirement_gain": 0.0 if repeats and not productive_repeated_goal_navigation else min(1.0, requirement_gain),
        "precondition_satisfied": min(1.0, max(0.0, precondition)),
        "policy_risk": min(1.0, max(0.0, risk)),
    }


def _goal_ready_to_finish(*, goal: AlfworldGoalSpec, observed_terms: set[str], held_terms: set[str]) -> bool:
    object_ok = not goal.object_terms or bool(goal.object_terms & (observed_terms | held_terms))
    receptacle_ok = not goal.receptacle_terms or bool(goal.receptacle_terms & observed_terms)
    return object_ok and receptacle_ok


def _command_object_terms(command: str) -> set[str]:
    lowered = command.lower().strip()
    take_match = re.search(r"\b(?:take|pick up|pick)\b\s+(?P<object>.+?)\s+\bfrom\b\s+.+$", lowered)
    if take_match:
        return _object_tokens(take_match.group("object"))
    put_match = re.search(r"\b(?:put|place|insert|move)\b\s+(?P<object>.+?)\s+\b(?:in|on|into|onto|to)\b\s+.+$", lowered)
    if put_match:
        return _object_tokens(put_match.group("object"))
    transform_match = re.search(r"\b(?:cool|heat|clean|slice|toggle|pour|turn|use)\b\s+(?P<object>.+?)(?:\s+\b(?:with|using|in|on)\b\s+.+)?$", lowered)
    if transform_match:
        return _object_tokens(transform_match.group("object"))
    return _object_tokens(lowered)


def _command_receptacle_terms(command: str) -> set[str]:
    lowered = command.lower().strip()
    put_match = re.search(r"\b(?:put|place|insert|move)\b\s+.+?\s+\b(?:in|on|into|onto|to)\b\s+(?P<receptacle>.+)$", lowered)
    if put_match:
        return _object_tokens(put_match.group("receptacle"))
    take_match = re.search(r"\b(?:take|pick up|pick)\b\s+.+?\s+\bfrom\b\s+(?P<receptacle>.+)$", lowered)
    if take_match:
        return _object_tokens(take_match.group("receptacle"))
    go_match = re.search(r"\bgo\b\s+\bto\b\s+(?P<receptacle>.+)$", lowered)
    if go_match:
        return _object_tokens(go_match.group("receptacle"))
    return set()


def _object_tokens(text: str) -> set[str]:
    stop = {
        "the",
        "a",
        "an",
        "to",
        "from",
        "in",
        "on",
        "with",
        "and",
        "then",
        "you",
        "are",
        "is",
        "at",
        "of",
        "your",
        "task",
        "this",
        "that",
        "there",
        "here",
        "room",
        "rooms",
    }
    return {token.strip(".,:;!?()[]") for token in text.lower().split() if len(token.strip(".,:;!?()[]")) > 2 and token.strip(".,:;!?()[]") not in stop}


def _progress_proxy(*, objective: str, observation: str, history: Sequence[str]) -> float:
    objective_tokens = _object_tokens(objective)
    if not objective_tokens:
        return 0.0
    seen = _object_tokens(" ".join(history) + " " + observation)
    return min(1.0, len(objective_tokens & seen) / len(objective_tokens))


def _stagnation_frac(history: Sequence[str]) -> float:
    actions = [item[4:] for item in history if item.startswith("act:")]
    if not actions:
        return 0.0
    tail = actions[-3:]
    return len(tail) / 3.0 if len(set(tail)) == 1 else 0.0


def _recent_action_type_streak(history: Sequence[str], action_type: str) -> int:
    streak = 0
    for item in reversed(history):
        if not item.startswith("act:"):
            continue
        if _command_type(item[4:].strip()) != action_type:
            break
        streak += 1
    return streak


def _action_count(history: Sequence[str]) -> int:
    return sum(1 for item in history if item.startswith("act:"))


def _admissible_from_info(info: Mapping[str, Any]) -> list[str]:
    commands = info.get("admissible_commands", [])
    if isinstance(commands, (list, tuple)) and all(isinstance(command, str) for command in commands):
        return [str(command) for command in commands]
    commands = _first(commands, commands)
    if isinstance(commands, str):
        return [commands]
    return [str(command) for command in commands]


def _filtered_admissible_commands(commands: Sequence[str]) -> list[str]:
    deduped = list(dict.fromkeys(str(command).strip() for command in commands if str(command).strip()))
    if len(deduped) <= 1:
        return deduped
    task_actions = [command for command in deduped if _command_type(command) != "help"]
    return task_actions or deduped


def _contract_filtered_admissible_commands(
    commands: Sequence[str],
    *,
    observation: str,
    objective: str,
    history: Sequence[str],
    max_steps: int,
) -> list[str]:
    filtered = _filtered_admissible_commands(commands)
    if len(filtered) <= 1:
        return filtered
    candidates = [
        (
            command,
            alfworld_candidate_to_canonical(
                candidate_id=f"alfworld-filter:{idx}",
                observation=observation,
                objective=objective,
                history=history,
                admissible_commands=filtered,
                command=command,
                max_steps=max_steps,
            ),
        )
        for idx, command in enumerate(filtered)
    ]
    kept = []
    for command, candidate in candidates:
        action = candidate.action
        zero_gain = action.declared_observation_gain == 0.0 and action.declared_requirement_gain == 0.0
        drop_zero_gain_write = action.action_is_write == 1.0 and zero_gain
        drop_repeated_zero_gain_read = action.action_is_read == 1.0 and action.duplicates_previous_action == 1.0 and zero_gain
        if not drop_zero_gain_write and not drop_repeated_zero_gain_read:
            kept.append(command)
    return kept if kept else filtered


def _gamefile_from_info(info: Mapping[str, Any]) -> str | None:
    value = info.get("extra.gamefile")
    if not value:
        return None
    gamefile = str(_first(value, value)).strip()
    if not gamefile or gamefile.lower() in {"none", "null"}:
        return None
    return gamefile


def _objective_from_info(info: Mapping[str, Any], observation: str) -> str:
    for key in ("objective", "goal", "task_desc"):
        value = info.get(key)
        if value:
            return str(_first(value, value))
    parsed = _task_text_from_observation(observation)
    return parsed or observation


def _task_text_from_observation(observation: str) -> str:
    marker = "your task is to:"
    lowered = observation.lower()
    idx = lowered.find(marker)
    if idx < 0:
        return ""
    task = observation[idx + len(marker) :].strip()
    for stop in ("\n\n", "\n"):
        if stop in task:
            task = task.split(stop, 1)[0].strip()
    return task.strip(" .")


def _strip_task_text(observation: str) -> str:
    lowered = observation.lower()
    for marker in ("your task is to:", "your task is to"):
        idx = lowered.find(marker)
        if idx >= 0:
            return observation[:idx]
    return observation


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


def _clean_ranked_indices(indices: object, candidate_count: int) -> list[int]:
    fallback = list(range(candidate_count))
    if not isinstance(indices, Sequence) or isinstance(indices, (str, bytes, bytearray)):
        return fallback
    clean: list[int] = []
    seen: set[int] = set()
    for item in indices:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < candidate_count and idx not in seen:
            clean.append(idx)
            seen.add(idx)
    clean.extend(idx for idx in fallback if idx not in seen)
    return clean


def _aggregate_policy(policy: str, rows: Sequence[AlfworldEpisodeResult]) -> dict[str, object]:
    total = len(rows)
    return {
        "policy": policy,
        "success": 0.0 if total == 0 else sum(row.success for row in rows) / total,
        "final_score": 0.0 if total == 0 else sum(row.final_score for row in rows) / total,
        "steps": 0.0 if total == 0 else sum(row.steps for row in rows) / total,
        "mean_admissible_count": 0.0 if total == 0 else sum(row.mean_admissible_count for row in rows) / total,
        "opportunity_rate": 0.0 if total == 0 else sum(row.opportunity_rate for row in rows) / total,
        "invalid_actions": 0 if total == 0 else sum(row.invalid_actions for row in rows),
        "wall_clock_seconds": 0.0 if total == 0 else sum(row.wall_clock_seconds for row in rows),
        "episodes": total,
        "tokens": 0 if total == 0 else sum(row.tokens for row in rows),
        "llm_calls": 0 if total == 0 else sum(row.llm_calls for row in rows),
    }


def _with_step_trace_diagnostics(rows: list[dict[str, object]], step_trace_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_policy: dict[str, dict[str, float]] = {}
    for row in step_trace_rows:
        policy = str(row.get("policy", ""))
        stats = by_policy.setdefault(policy, {"steps": 0.0, "available": 0.0, "matched": 0.0, "in_shortlist": 0.0, "shortlist_size": 0.0})
        stats["steps"] += 1.0
        if str(row.get("diagnostic_command", "")).strip():
            stats["available"] += 1.0
        try:
            matched = int(row.get("matches_diagnostic", 0))
        except (TypeError, ValueError):
            matched = 0
        try:
            in_shortlist = int(row.get("diagnostic_in_shortlist", 0))
        except (TypeError, ValueError):
            in_shortlist = 0
        try:
            shortlist_size = float(row.get("llm_shortlist_size", 0.0))
        except (TypeError, ValueError):
            shortlist_size = 0.0
        stats["matched"] += 1.0 if matched else 0.0
        stats["in_shortlist"] += 1.0 if in_shortlist else 0.0
        stats["shortlist_size"] += shortlist_size
    for row in rows:
        stats = by_policy.get(str(row.get("policy", "")), {"steps": 0.0, "available": 0.0, "matched": 0.0, "in_shortlist": 0.0, "shortlist_size": 0.0})
        steps = stats["steps"]
        available = stats["available"]
        row["diagnostic_available_rate"] = 0.0 if steps == 0.0 else available / steps
        row["diagnostic_match_rate"] = 0.0 if available == 0.0 else stats["matched"] / available
        row["proposal_recall_at_k"] = 0.0 if available == 0.0 else stats["in_shortlist"] / available
        row["mean_llm_shortlist_size"] = 0.0 if steps == 0.0 else stats["shortlist_size"] / steps
    return rows


def _write_alfworld_llm_shortlist_progress(
    path: Path,
    *,
    status: str,
    completed_policy_episodes: int,
    total_policy_episodes: int,
    policy: str,
    episode: int,
    shortlister: AlfworldLLMShortlistCache,
    shortlist_cache_path: Path,
) -> None:
    path.write_text(
        json.dumps(
            {
                "status": status,
                "completed_policy_episodes": completed_policy_episodes,
                "total_policy_episodes": total_policy_episodes,
                "last_policy": policy,
                "last_episode": episode,
                "new_llm_calls": shortlister.new_calls,
                "cache_keys_used": shortlister.calls,
                "tokens": shortlister.tokens,
                "shortlist_cache_path": str(shortlist_cache_path),
                "partial_table_path": str(path.parent / "alfworld_llm_shortlist_partial_table.csv"),
                "partial_per_episode_path": str(path.parent / "alfworld_llm_shortlist_partial_per_episode.csv"),
                "updated_at_unix": time.time(),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _write_alfworld_llm_shortlist_partial_outputs(
    out_dir: Path,
    policies: Sequence[str],
    all_results: Sequence[AlfworldEpisodeResult],
    step_trace_rows: Sequence[Mapping[str, object]],
) -> None:
    completed_policies = [policy for policy in policies if any(row.policy == policy for row in all_results)]
    rows = _with_step_trace_diagnostics(
        [_aggregate_policy(policy, [row for row in all_results if row.policy == policy]) for policy in completed_policies],
        step_trace_rows,
    )
    _write_csv(out_dir / "alfworld_llm_shortlist_partial_table.csv", rows)
    _write_csv(out_dir / "alfworld_llm_shortlist_partial_per_episode.csv", [row.__dict__ for row in all_results])
    _write_csv(out_dir / "alfworld_llm_shortlist_partial_step_trace.csv", step_trace_rows)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _alfworld_dense_cache_manifest() -> dict[str, object]:
    contract = canonical_contract_manifest()
    return {
        "cache": "alfworld_dense_feature_pair_cache",
        "version": 1,
        "alfworld_adapter_version": ALFWORLD_ADAPTER_VERSION,
        "feature_schema_hash": contract["feature_schema_hash"],
        "feature_dim": len(canonical_feature_names()),
    }


def _alfworld_dense_cache_manifest_compatible(path: Path) -> bool:
    if not path.exists():
        return False
    expected = _alfworld_dense_cache_manifest()
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    required_keys = ("cache", "version", "alfworld_adapter_version", "feature_schema_hash", "feature_dim")
    return all(actual.get(key) == expected.get(key) for key in required_keys)


def _alfworld_dense_pair_cache_metadata() -> dict[str, object]:
    manifest = _alfworld_dense_cache_manifest()
    return {
        "alfworld_adapter_version": manifest["alfworld_adapter_version"],
        "feature_schema_hash": manifest["feature_schema_hash"],
        "feature_dim": manifest["feature_dim"],
    }


def _alfworld_dense_pair_cache_row_compatible(row: Mapping[str, object]) -> bool:
    metadata = _alfworld_dense_pair_cache_metadata()
    return (
        row.get("alfworld_adapter_version") == metadata["alfworld_adapter_version"]
        and row.get("feature_schema_hash") == metadata["feature_schema_hash"]
        and int(row.get("feature_dim", -1)) == int(metadata["feature_dim"])
    )


def _load_alfworld_dense_training_cache(
    cache_dir: Path,
    *,
    require_feature_compatibility: bool = True,
    rebuild_feature_pairs_from_state_cache: bool = False,
    max_pairs_per_state: int = 24,
    seed: int = 0,
) -> dict[str, object]:
    manifest_path = cache_dir / "alfworld_dense_cache_manifest.json"
    cache_compatible = _alfworld_dense_cache_manifest_compatible(manifest_path)
    pair_path = cache_dir / "alfworld_dense_pair_cache.jsonl"
    pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    skipped_incompatible_pairs = 0
    can_load_feature_cache = cache_compatible or not require_feature_compatibility
    if can_load_feature_cache and pair_path.exists():
        with pair_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if require_feature_compatibility and not _alfworld_dense_pair_cache_row_compatible(row):
                    skipped_incompatible_pairs += 1
                    continue
                pairs.append(
                    (
                        tuple(float(value) for value in row["positive"]),
                        tuple(float(value) for value in row["negative"]),
                    )
                )
    elif pair_path.exists():
        skipped_incompatible_pairs = sum(1 for line in pair_path.read_text(encoding="utf-8").splitlines() if line.strip())
    trace_rows = _read_csv_rows(cache_dir / "alfworld_dense_training_trace.csv")
    action_label_rows = _read_csv_rows(cache_dir / "alfworld_dense_action_labels.csv")
    action_label_rows = _annotate_alfworld_dense_label_row_groups(action_label_rows)
    state_label_rows = _load_alfworld_dense_state_label_cache(cache_dir / "alfworld_dense_state_label_cache.jsonl")
    rebuilt_feature_pairs_from_state_cache = 0
    if (
        require_feature_compatibility
        and not pairs
        and rebuild_feature_pairs_from_state_cache
        and state_label_rows
    ):
        rebuilt_pairs, rebuilt_action_label_rows = _rebuild_alfworld_dense_pairs_from_state_cache(
            state_label_rows,
            max_pairs_per_state=max_pairs_per_state,
            seed=seed,
        )
        pairs = rebuilt_pairs
        action_label_rows = rebuilt_action_label_rows
        rebuilt_feature_pairs_from_state_cache = len(rebuilt_pairs)
    completed_episodes = {
        int(row["episode"])
        for row in trace_rows
        if str(row.get("episode", "")).strip()
    } if (cache_compatible and skipped_incompatible_pairs == 0) or rebuilt_feature_pairs_from_state_cache > 0 else set()
    return {
        "pairs": pairs,
        "trace_rows": trace_rows if cache_compatible or not require_feature_compatibility or rebuilt_feature_pairs_from_state_cache > 0 else [],
        "action_label_rows": action_label_rows if cache_compatible or not require_feature_compatibility or rebuilt_feature_pairs_from_state_cache > 0 else [],
        "completed_episodes": completed_episodes,
        "cache_manifest_compatible": cache_compatible,
        "skipped_incompatible_pairs": skipped_incompatible_pairs,
        "state_label_cache_rows": len(state_label_rows),
        "rebuilt_feature_pairs_from_state_cache": rebuilt_feature_pairs_from_state_cache,
    }


def _load_alfworld_dense_audit_cache(
    cache_dir: Path,
    *,
    fallback_label_source: str,
    fallback_max_pairs_per_state: int,
    fallback_max_frontier_actions: int,
    fallback_frontier_strategy: str,
    fallback_counterfactual_rollout_steps: int,
    fallback_planner_timeout_seconds: float,
    fallback_max_episode_label_seconds: float,
    fallback_perturbation_prob: float,
    fallback_warmup_random_steps: int,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    cached = _load_alfworld_dense_training_cache(cache_dir, require_feature_compatibility=False)
    pairs = cached["pairs"]
    trace_rows = cached["trace_rows"]
    action_label_rows = cached["action_label_rows"]
    if not action_label_rows:
        raise RuntimeError(f"ALFWorld dense audit cache has no action labels: {cache_dir}")

    train_json = cache_dir / "alfworld_dense_train.json"
    if train_json.exists():
        try:
            body = json.loads(train_json.read_text(encoding="utf-8"))
            train_manifest = dict(body.get("train", {}))
        except Exception:
            train_manifest = {}
    else:
        train_manifest = {}

    label_source = str(train_manifest.get("label_source", _alfworld_dense_label_source_name(fallback_label_source)))
    train_manifest.setdefault("train_pair_count", len(pairs))
    train_manifest.setdefault("max_pairs_per_state", fallback_max_pairs_per_state)
    train_manifest.setdefault("max_frontier_actions", fallback_max_frontier_actions)
    train_manifest.setdefault("frontier_strategy", fallback_frontier_strategy)
    train_manifest.setdefault("counterfactual_rollout_steps", fallback_counterfactual_rollout_steps)
    train_manifest.setdefault("planner_timeout_seconds", fallback_planner_timeout_seconds)
    train_manifest.setdefault("max_episode_label_seconds", fallback_max_episode_label_seconds)
    train_manifest.setdefault("planner_label_cache", str(cache_dir / "alfworld_dense_planner_label_cache.jsonl"))
    planner_label_cache_entries = len(_load_alfworld_planner_label_cache(cache_dir / "alfworld_dense_planner_label_cache.jsonl"))
    train_manifest.setdefault("planner_label_cache_loaded_entries", planner_label_cache_entries)
    train_manifest.setdefault("planner_label_cache_entries", planner_label_cache_entries)
    train_manifest.setdefault("new_planner_label_cache_hits", 0)
    train_manifest.setdefault("new_planner_label_cache_misses", 0)
    train_manifest.setdefault("planner_label_cache_hits", 0)
    train_manifest.setdefault("planner_label_cache_misses", 0)
    train_manifest.setdefault("perturbation_prob", fallback_perturbation_prob)
    train_manifest.setdefault("warmup_random_steps", fallback_warmup_random_steps)
    train_manifest.setdefault("label_source", label_source)
    train_manifest.setdefault("counterfactual_planner_cost_to_go", label_source == "textworld_pddl_replan_cost_to_go")
    train_manifest.setdefault("dense_pair_cache", str(cache_dir / "alfworld_dense_pair_cache.jsonl"))
    train_manifest.setdefault("dense_state_label_cache", str(cache_dir / "alfworld_dense_state_label_cache.jsonl"))
    train_manifest.setdefault("dense_state_label_cache_rows", int(cached.get("state_label_cache_rows", 0)))
    train_manifest.setdefault("rebuilt_dense_pairs_from_state_cache", int(cached.get("rebuilt_feature_pairs_from_state_cache", 0)))
    train_manifest.setdefault("resumed_completed_episodes", len(cached["completed_episodes"]))
    return train_manifest, list(trace_rows), list(action_label_rows)


def _alfworld_dense_state_label_cache_row(
    *,
    episode: int,
    step: int,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible_commands: Sequence[str],
    max_steps: int,
    labels: Sequence[AlfworldDenseActionLabel],
    label_source: str,
    selected_rollout_command: str = "",
    selected_rollout_source: str = "",
) -> dict[str, object]:
    return {
        "state_label_cache_version": 1,
        "episode": episode,
        "step": step,
        "observation": observation,
        "objective": objective,
        "history": list(history),
        "admissible_commands": list(admissible_commands),
        "max_steps": max_steps,
        "label_source": _alfworld_dense_label_source_name(label_source),
        "selected_rollout_command": selected_rollout_command,
        "selected_rollout_source": selected_rollout_source,
        "labels": [
            {
                **label.__dict__,
                "episode": episode,
                "step": step,
            }
            for label in labels
        ],
    }


def _load_alfworld_dense_state_label_cache(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(row.get("state_label_cache_version", 0)) != 1:
                continue
            rows.append(row)
    return rows


def _state_cache_row_to_labels(row: Mapping[str, object]) -> list[AlfworldDenseActionLabel]:
    labels: list[AlfworldDenseActionLabel] = []
    episode = int(row.get("episode", -1))
    step = int(row.get("step", -1))
    for raw in row.get("labels", []):
        if not isinstance(raw, Mapping):
            continue
        labels.append(
            AlfworldDenseActionLabel(
                episode=episode,
                step=step,
                command=str(raw.get("command", "")),
                cost_to_success=int(raw.get("cost_to_success", 0)),
                reachable=int(raw.get("reachable", 0)),
                final_score=float(raw.get("final_score", 0.0)),
                rollout_steps=int(raw.get("rollout_steps", 0)),
                label_source=str(raw.get("label_source", row.get("label_source", ""))),
                label_status=str(raw.get("label_status", "ok")),
                pairwise_usable=int(raw.get("pairwise_usable", 1)),
                oracle_rank=int(raw.get("oracle_rank", 0)),
                best_cost_to_success=int(raw.get("best_cost_to_success", 0)),
                cost_gap_to_best=int(raw.get("cost_gap_to_best", 0)),
            )
        )
    return _annotate_alfworld_dense_label_ranks(labels)


def _rebuild_alfworld_dense_pairs_from_state_cache(
    state_label_rows: Sequence[Mapping[str, object]],
    *,
    max_pairs_per_state: int,
    seed: int,
) -> tuple[list[tuple[tuple[float, ...], tuple[float, ...]]], list[dict[str, object]]]:
    pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    action_label_rows: list[dict[str, object]] = []
    for row in state_label_rows:
        episode = int(row.get("episode", -1))
        step = int(row.get("step", -1))
        labels = _state_cache_row_to_labels(row)
        if not labels:
            continue
        row_pairs = _dense_labels_to_pairs(
            labels=labels,
            observation=str(row.get("observation", "")),
            objective=str(row.get("objective", "")),
            history=[str(item) for item in row.get("history", [])],
            admissible_commands=[str(item) for item in row.get("admissible_commands", [])],
            max_steps=int(row.get("max_steps", 0)),
            max_pairs=max_pairs_per_state,
            rng=random.Random(seed + episode * 1009 + step),
            episode=episode,
            step=step,
        )
        pairs.extend(row_pairs)
        action_label_rows.extend(label.__dict__ for label in labels)
    return pairs, action_label_rows


def _append_alfworld_dense_training_cache(
    cache_dir: Path,
    *,
    pair_rows: Sequence[Mapping[str, object]],
    state_label_rows: Sequence[Mapping[str, object]] = (),
    trace_rows: Sequence[Mapping[str, object]],
    action_label_rows: Sequence[Mapping[str, object]],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "alfworld_dense_cache_manifest.json").write_text(json.dumps(_alfworld_dense_cache_manifest(), indent=2, sort_keys=True), encoding="utf-8")
    pair_path = cache_dir / "alfworld_dense_pair_cache.jsonl"
    if pair_rows:
        with pair_path.open("a", encoding="utf-8") as handle:
            for row in pair_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    state_label_path = cache_dir / "alfworld_dense_state_label_cache.jsonl"
    if state_label_rows:
        with state_label_path.open("a", encoding="utf-8") as handle:
            for row in state_label_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
    _write_csv(cache_dir / "alfworld_dense_training_trace.csv", trace_rows)
    _write_csv(cache_dir / "alfworld_dense_action_labels.csv", action_label_rows)
