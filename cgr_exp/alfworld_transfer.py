from __future__ import annotations

from dataclasses import dataclass
import csv
import importlib.util
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from cgr_exp.harness_contract import ActionFeatures, CanonicalCandidate, RuntimeFeatures, canonical_contract_manifest, canonical_feature_names
from cgr_exp.harness_ranker import GenericHarnessRanker, train_generic_harness_ranker
from cgr_exp.workflow_source_generator import SOURCE_WORKFLOW_FAMILIES, generate_workflow_source_tasks, source_generator_manifest


READ_VERBS = ("look", "inventory", "examine")
MOVE_VERBS = ("go",)
OPEN_VERBS = ("open", "close")
TAKE_VERBS = ("take", "pick")
PUT_VERBS = ("put", "place", "insert")
USE_VERBS = ("toggle", "heat", "cool", "clean", "slice", "pour", "turn", "use")
TERMINATE_VERBS = ("done", "finish", "stop")


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
    if env_factory is None:
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
    for policy in policies:
        env = _make_alfworld_env(split=split, config_path=config_path, env_factory=env_factory)
        for episode in range(episodes):
            all_results.append(
                _run_episode(
                    env,
                    policy=policy,
                    ranker=ranker,
                    seed=seed * 100000 + episode,
                    episode=episode,
                    max_steps=max_steps,
                )
            )
    rows = [_aggregate_policy(policy, [row for row in all_results if row.policy == policy]) for policy in policies]
    manifest = {
        "benchmark": "alfworld_admissible_action_transfer",
        "seed": seed,
        "split": split,
        "episodes": episodes,
        "max_steps": max_steps,
        "policies": list(policies),
        "uses_admissible_commands": True,
        "target_trajectory_training_used": False,
        "alfworld_train_trajectories_used": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "controller_checkpoint_frozen": True,
        "contract": canonical_contract_manifest(),
        "source_controller": source_manifest,
        "rows": rows,
    }
    _write_csv(out_dir / "alfworld_transfer_table.csv", rows)
    _write_csv(out_dir / "alfworld_transfer_per_episode.csv", [row.__dict__ for row in all_results])
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
    counterfactual_rollout_steps: int = 20,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    ranker_checkpoint: Path | None = None,
    policies: Sequence[str] = ("Random", "Hand heuristic", "ALFWorld dense-ranker"),
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
            counterfactual_rollout_steps=counterfactual_rollout_steps,
            perturbation_prob=perturbation_prob,
            warmup_random_steps=warmup_random_steps,
            env_factory=env_factory,
        )
    all_results: list[AlfworldEpisodeResult] = []
    for policy in policies:
        env = _make_alfworld_env(split=eval_split, config_path=config_path, env_factory=env_factory)
        for episode in range(eval_episodes):
            all_results.append(
                _run_episode(
                    env,
                    policy=policy,
                    ranker=ranker,
                    seed=seed * 100000 + episode,
                    episode=episode,
                    max_steps=max_steps,
                )
            )
    rows = [_aggregate_policy(policy, [row for row in all_results if row.policy == policy]) for policy in policies]
    manifest = {
        "benchmark": "alfworld_dense_frontier_transfer",
        "seed": seed,
        "train_split": "train",
        "eval_split": eval_split,
        "train_episodes": train_episodes,
        "eval_episodes": eval_episodes,
        "max_steps": max_steps,
        "policies": list(policies),
        "uses_admissible_commands": True,
        "training_source": "alfworld_train_replay_prefix_expert_rollout_frontier_labels",
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "controller_checkpoint_frozen_before_eval": True,
        "contract": canonical_contract_manifest(),
        "train": train_manifest,
        "rows": rows,
    }
    (out_dir / "alfworld_dense_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    _write_csv(out_dir / "alfworld_dense_transfer_table.csv", rows)
    _write_csv(out_dir / "alfworld_dense_transfer_per_episode.csv", [row.__dict__ for row in all_results])
    _write_csv(out_dir / "alfworld_dense_training_trace.csv", train_rows)
    _write_csv(out_dir / "alfworld_dense_action_labels.csv", action_label_rows)
    (out_dir / "alfworld_dense_transfer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def audit_alfworld_dense_labels(
    out_dir: Path,
    *,
    seed: int = 0,
    config_path: Path | None = None,
    episodes: int = 5,
    max_steps: int = 20,
    max_pairs_per_state: int = 24,
    max_frontier_actions: int = 16,
    counterfactual_rollout_steps: int = 20,
    perturbation_prob: float = 0.0,
    warmup_random_steps: int = 0,
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None:
        _assert_alfworld_available()
    _ranker, train_manifest, trace_rows, action_label_rows = train_alfworld_dense_ranker(
        seed=seed,
        config_path=config_path,
        train_episodes=episodes,
        max_steps=max_steps,
        epochs=0,
        max_pairs_per_state=max_pairs_per_state,
        max_frontier_actions=max_frontier_actions,
        counterfactual_rollout_steps=counterfactual_rollout_steps,
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
        "training_source": "alfworld_train_replay_prefix_expert_rollout_frontier_labels",
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "counterfactual_planner_cost_to_go": False,
        "label_source": "replay_prefix_plus_expert_rollout_cost_proxy",
        "contract": canonical_contract_manifest(),
        "label_summary": summary,
        "collection": train_manifest,
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
    counterfactual_rollout_steps: int = 20,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    env_factory: Callable[..., Any] | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if env_factory is None:
        _assert_alfworld_available()
    ranker, train_manifest, train_rows, action_label_rows = train_alfworld_dense_ranker(
        seed=seed,
        config_path=config_path,
        train_episodes=train_episodes,
        max_steps=max_steps,
        epochs=epochs,
        max_pairs_per_state=max_pairs_per_state,
        max_frontier_actions=max_frontier_actions,
        counterfactual_rollout_steps=counterfactual_rollout_steps,
        perturbation_prob=perturbation_prob,
        warmup_random_steps=warmup_random_steps,
        env_factory=env_factory,
    )
    manifest = {
        "benchmark": "alfworld_dense_frontier_controller_train",
        "seed": seed,
        "train_split": "train",
        "train_episodes": train_episodes,
        "max_steps": max_steps,
        "training_source": "alfworld_train_replay_prefix_expert_rollout_frontier_labels",
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "contract": canonical_contract_manifest(),
        "train": train_manifest,
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
    for policy in policies:
        env = _make_alfworld_env(split=eval_split, config_path=config_path, env_factory=env_factory)
        for episode in range(eval_episodes):
            all_results.append(
                _run_episode(
                    env,
                    policy=policy,
                    ranker=ranker,
                    seed=seed * 100000 + episode,
                    episode=episode,
                    max_steps=max_steps,
                )
            )
    rows = [_aggregate_policy(policy, [row for row in all_results if row.policy == policy]) for policy in policies]
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
        "training_source": "alfworld_train_expert_plan_frontier_labels",
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
        "controller_checkpoint_frozen_before_eval": True,
        "contract": canonical_contract_manifest(),
        "train": train_manifest,
        "rows": rows,
    }
    (out_dir / "alfworld_expert_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    _write_csv(out_dir / "alfworld_expert_transfer_table.csv", rows)
    _write_csv(out_dir / "alfworld_expert_transfer_per_episode.csv", [row.__dict__ for row in all_results])
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
        "llm_rollout_trajectories_used": False,
        "target_eval_trajectories_used_for_training": False,
        "oracle_labels_in_inference_features": False,
        "command_string_used_as_learned_feature": False,
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
            admissible = _admissible_from_info(info)
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
    counterfactual_rollout_steps: int = 20,
    perturbation_prob: float = 0.15,
    warmup_random_steps: int = 5,
    env_factory: Callable[..., Any] | None = None,
) -> tuple[GenericHarnessRanker, dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    train_env = _make_alfworld_env(split="train", config_path=config_path, env_factory=env_factory, training_method="dagger")
    label_env = _make_alfworld_env(split="train", config_path=config_path, env_factory=env_factory, training_method="dagger")
    pairs: list[tuple[tuple[float, ...], tuple[float, ...]]] = []
    trace_rows: list[dict[str, object]] = []
    action_label_rows: list[dict[str, object]] = []
    labeled_states = 0
    skipped_states = 0
    finite_action_labels = 0
    infinite_action_labels = 0
    expert_monotone_checks = 0
    expert_monotone_failures = 0
    for episode in range(train_episodes):
        obs, info = train_env.reset()
        observation = _first(obs)
        objective = _objective_from_info(info, observation)
        gamefile = _gamefile_from_info(info)
        history: list[str] = [f"obs:{observation}"]
        prefix: list[str] = []
        episode_pairs = 0
        episode_labels = 0
        final_score = 0.0
        for step in range(max_steps):
            admissible = _admissible_from_info(info)
            expert = _expert_action_from_info(info)
            if not admissible:
                break
            frontier = _frontier_sample(admissible, expert=expert, max_frontier_actions=max_frontier_actions, rng=rng)
            labels = _label_frontier_by_expert_rollout(
                label_env,
                gamefile=gamefile,
                prefix=prefix,
                commands=frontier,
                max_rollout_steps=counterfactual_rollout_steps,
            )
            labels_by_command = {label.command: label for label in labels}
            if expert in labels_by_command and labels_by_command[expert].reachable:
                expert_monotone_checks += 1
                best_cost = min(label.cost_to_success for label in labels)
                if labels_by_command[expert].cost_to_success > best_cost:
                    expert_monotone_failures += 1
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
                )
                action_label_rows.append(action_label.__dict__)
                row_labels.append(label)
                finite_action_labels += 1 if label.reachable else 0
                infinite_action_labels += 0 if label.reachable else 1
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
                episode_labels += len(labels)
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
                "pairs": episode_pairs,
                "action_labels": episode_labels,
                "steps": _action_count(history),
                "final_score": final_score,
            }
        )
        print(
            "[alfworld-dense-train] "
            f"{episode + 1}/{train_episodes} episodes, "
            f"states={labeled_states}, pairs={len(pairs)}, "
            f"finite_labels={finite_action_labels}, inf_labels={infinite_action_labels}"
        )
    if not pairs:
        raise RuntimeError("ALFWorld dense training produced zero ranking pairs; check replay-prefix labels and expert availability.")
    ranker = GenericHarnessRanker.create(len(canonical_feature_names()), seed=seed)
    ranker.train(pairs, epochs=epochs, lr=0.03, seed=seed)
    manifest = {
        "train_pair_count": len(pairs),
        "dense_labeled_states": labeled_states,
        "skipped_states": skipped_states,
        "finite_action_labels": finite_action_labels,
        "infinite_action_labels": infinite_action_labels,
        "epochs": epochs,
        "max_pairs_per_state": max_pairs_per_state,
        "max_frontier_actions": max_frontier_actions,
        "counterfactual_rollout_steps": counterfactual_rollout_steps,
        "perturbation_prob": perturbation_prob,
        "warmup_random_steps": warmup_random_steps,
        "label_source": "replay_prefix_plus_expert_rollout_cost_proxy",
        "counterfactual_planner_cost_to_go": False,
        "expert_monotone_checks": expert_monotone_checks,
        "expert_monotone_failures": expert_monotone_failures,
        "expert_monotone_failure_rate": 0.0 if expert_monotone_checks == 0 else expert_monotone_failures / expert_monotone_checks,
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


def alfworld_runtime_features(
    *,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible_commands: Sequence[str],
    max_steps: int,
) -> RuntimeFeatures:
    text = f"{objective} {observation}".lower()
    history_text = " ".join(history).lower()
    observed_entities = _object_tokens(text)
    progress = _progress_proxy(objective=objective, observation=observation, history=history)
    failed = sum(1 for item in history if item.startswith("obs:") and ("can't" in item.lower() or "nothing happens" in item.lower()))
    read_count = sum(1 for item in history if item.startswith("act:") and _command_type(item[4:].strip()) == "read")
    write_count = sum(1 for item in history if item.startswith("act:") and _command_type(item[4:].strip()) == "write")
    return RuntimeFeatures(
        verified_progress_frac=progress,
        verified_progress_known_mask=1.0,
        remaining_requirement_frac=1.0 - progress,
        remaining_requirement_known_mask=1.0,
        observed_fact_frac=min(1.0, len(observed_entities) / 12.0),
        observed_fact_known_mask=1.0,
        remaining_turn_budget_frac=max(0.0, (max_steps - _action_count(history)) / max(1, max_steps)),
        remaining_tool_budget_frac=max(0.0, (max_steps - _action_count(history)) / max(1, max_steps)),
        constraint_pressure=min(1.0, failed / max(1, len(history))),
        constraint_pressure_known_mask=1.0,
        valid_action_ratio=1.0,
        previous_failure_rate=failed / max(1, _action_count(history)),
        stagnation_steps_frac=_stagnation_frac(history),
        uncertainty_score=1.0 if not observed_entities else max(0.0, 1.0 - len(observed_entities) / 12.0),
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
    objective_tokens = _object_tokens(objective.lower())
    command_tokens = _object_tokens(command.lower())
    observation_tokens = _object_tokens(observation.lower())
    target_related = bool(command_tokens & objective_tokens)
    references_observed = bool(command_tokens & observation_tokens) or target_related
    return ActionFeatures(
        action_is_read=1.0 if action_type == "read" else 0.0,
        action_is_write=1.0 if action_type == "write" else 0.0,
        action_is_respond=0.0,
        action_is_ask=0.0,
        action_is_terminate=1.0 if action_type == "terminate" else 0.0,
        argument_completeness=1.0 if command.strip() else 0.0,
        references_observed_entity=1.0 if references_observed else 0.0,
        references_unverified_entity=1.0 if command_tokens and not references_observed else 0.0,
        declared_observation_gain=1.0 if action_type == "read" and not repeats else 0.0,
        declared_requirement_gain=1.0 if action_type == "write" and target_related and not repeats else 0.0,
        expected_tool_cost_frac=1.0 / max(1, max_steps),
        action_irreversible=1.0 if action_type == "write" else 0.0,
        precondition_satisfied=1.0,
        policy_risk=0.1 if action_type == "read" else (0.3 if action_type == "write" else 0.0),
        duplicates_previous_action=1.0 if repeats else 0.0,
    )


def _frontier_sample(admissible: Sequence[str], *, expert: str | None, max_frontier_actions: int, rng: random.Random) -> list[str]:
    commands = list(dict.fromkeys(str(command) for command in admissible))
    if max_frontier_actions <= 0 or len(commands) <= max_frontier_actions:
        return commands
    selected: list[str] = []
    if expert in commands:
        selected.append(str(expert))
    rest = [command for command in commands if command not in selected]
    rng.shuffle(rest)
    selected.extend(rest[: max(0, max_frontier_actions - len(selected))])
    return selected


def _label_frontier_by_expert_rollout(
    env: Any,
    *,
    gamefile: str | None,
    prefix: Sequence[str],
    commands: Sequence[str],
    max_rollout_steps: int,
) -> list[AlfworldDenseActionLabel]:
    labels = []
    for command in commands:
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
    ordered = sorted(labels, key=lambda label: (label.cost_to_success, label.command))
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


def _summarize_dense_label_rows(action_label_rows: Sequence[Mapping[str, object]], *, train_manifest: Mapping[str, object]) -> dict[str, object]:
    by_state: dict[tuple[int, int], list[int]] = {}
    finite = 0
    infinite = 0
    costs: list[int] = []
    for row in action_label_rows:
        key = (int(row["episode"]), int(row["step"]))
        cost = int(row["cost_to_success"])
        reachable = int(row["reachable"])
        by_state.setdefault(key, []).append(cost)
        costs.append(cost)
        if reachable:
            finite += 1
        else:
            infinite += 1
    varied_states = sum(1 for values in by_state.values() if len(set(values)) >= 2)
    comparable_states = len(by_state)
    return {
        "states": comparable_states,
        "action_labels": len(action_label_rows),
        "finite_action_labels": finite,
        "infinite_action_labels": infinite,
        "reachable_action_frac": 0.0 if not action_label_rows else finite / len(action_label_rows),
        "states_with_cost_variation": varied_states,
        "cost_variation_rate": 0.0 if comparable_states == 0 else varied_states / comparable_states,
        "unique_cost_values": len(set(costs)),
        "min_cost": "" if not costs else min(costs),
        "max_cost": "" if not costs else max(costs),
        "train_pair_count": int(train_manifest.get("train_pair_count", 0)),
        "expert_monotone_checks": int(train_manifest.get("expert_monotone_checks", 0)),
        "expert_monotone_failures": int(train_manifest.get("expert_monotone_failures", 0)),
        "expert_monotone_failure_rate": float(train_manifest.get("expert_monotone_failure_rate", 0.0)),
        "label_source": str(train_manifest.get("label_source", "")),
        "counterfactual_planner_cost_to_go": bool(train_manifest.get("counterfactual_planner_cost_to_go", False)),
    }


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
        observation, score, done, info = _step_any(target, candidate)
        if done and score > 0:
            return 1, True, score, 1
        for step in range(max_rollout_steps):
            expert = _expert_action_from_info(info)
            admissible = _admissible_from_info(info)
            if expert not in admissible:
                return 1 + max_rollout_steps + 1, False, score, step + 1
            observation, score, done, info = _step_any(target, str(expert))
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
    obs, info = env.reset()
    observation = str(_first(obs))
    if isinstance(info, Mapping):
        return observation, info
    return observation, {}


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


def _run_episode(env: Any, *, policy: str, ranker: GenericHarnessRanker, seed: int, episode: int, max_steps: int) -> AlfworldEpisodeResult:
    rng = random.Random(seed)
    start = time.perf_counter()
    obs, info = env.reset()
    observation = _first(obs)
    objective = _objective_from_info(info, observation)
    history: list[str] = [f"obs:{observation}"]
    final_score = 0.0
    invalid_actions = 0
    admissible_counts = []
    opportunities = 0
    steps = 0
    done = False
    for step in range(max_steps):
        admissible = _admissible_from_info(info)
        if not admissible:
            break
        admissible_counts.append(len(admissible))
        opportunities += 1 if len(admissible) >= 2 else 0
        command = _select_command(policy, observation, objective, history, admissible, ranker, rng, max_steps)
        if command not in admissible:
            invalid_actions += 1
        next_obs, scores, dones, info = env.step([command])
        final_score = float(_first(scores, 0.0))
        done = bool(_first(dones, False))
        observation = str(_first(next_obs, ""))
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


def _select_command(
    policy: str,
    observation: str,
    objective: str,
    history: Sequence[str],
    admissible: Sequence[str],
    ranker: GenericHarnessRanker,
    rng: random.Random,
    max_steps: int,
) -> str:
    if policy == "Random":
        return rng.choice(list(admissible))
    if policy == "Hand heuristic":
        return max(admissible, key=lambda command: (_hand_score(command, observation, objective, history), -admissible.index(command)))
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
        return max(scored)[2]
    raise ValueError(f"unknown ALFWorld policy: {policy}")


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


def _hand_score(command: str, observation: str, objective: str, history: Sequence[str]) -> float:
    action_type = _command_type(command)
    repeats = any(item == f"act:{command}" for item in history)
    score = -2.0 if repeats else 0.0
    if action_type == "read":
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


def _gamefile_from_info(info: Mapping[str, Any]) -> str | None:
    value = info.get("extra.gamefile")
    if not value:
        return None
    return str(_first(value, value))


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


def _first(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return value


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
        "tokens": 0,
        "llm_calls": 0,
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
