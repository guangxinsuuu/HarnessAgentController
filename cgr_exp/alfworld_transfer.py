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

from cgr_exp.harness_contract import ActionFeatures, CanonicalCandidate, RuntimeFeatures, canonical_contract_manifest
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


def _make_alfworld_env(*, split: str, config_path: Path | None, env_factory: Callable[..., Any] | None) -> Any:
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
        config = _default_alfworld_textworld_config()
    env_type = config.get("env", {}).get("type", "AlfredTWEnv")
    try:
        env_cls = environment.get_environment(env_type)
    except AttributeError:
        env_cls = getattr(environment, env_type)
    return env_cls(config, train_eval=split).init_env(batch_size=1)


def _default_alfworld_textworld_config() -> dict[str, Any]:
    data_root = os.environ.get("ALFWORLD_DATA")
    if not data_root:
        data_root = str(Path("data/alfworld").resolve())
    root = Path(data_root)
    return {
        "general": {"training_method": "dqn"},
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
    if policy == "Source-trained controller":
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
    commands = _first(commands, commands)
    return [str(command) for command in commands]


def _objective_from_info(info: Mapping[str, Any], observation: str) -> str:
    for key in ("objective", "goal", "task_desc"):
        value = info.get(key)
        if value:
            return str(_first(value, value))
    return observation


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
