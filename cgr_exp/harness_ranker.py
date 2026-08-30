from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from cgr_exp.harness_contract import audit_canonical_contract_manifest, canonical_contract_manifest, canonical_feature_names
from cgr_exp.workflow_oracle import collect_workflow_oracle_records, make_workflow_ranking_pairs, rank_actions_by_oracle
from cgr_exp.workflow_source_generator import (
    SOURCE_WORKFLOW_FAMILIES,
    TARGET_WORKFLOW_FAMILIES,
    WorkflowAction,
    WorkflowState,
    WorkflowTask,
    generate_workflow_source_tasks,
    source_generator_manifest,
    workflow_action_intent_features,
    workflow_action_features,
    workflow_runtime_features,
)


Pair = Tuple[Tuple[float, ...], Tuple[float, ...]]


@dataclass
class GenericHarnessRanker:
    weights: List[float]
    bias: float = 0.0

    @classmethod
    def create(cls, dim: int, seed: int = 0) -> "GenericHarnessRanker":
        rng = random.Random(seed)
        return cls([rng.uniform(-0.01, 0.01) for _ in range(dim)], 0.0)

    def score(self, features: Sequence[float]) -> float:
        return self.bias + sum(weight * float(value) for weight, value in zip(self.weights, features))

    def train(self, pairs: Sequence[Pair], epochs: int = 20, lr: float = 0.05, seed: int = 0) -> None:
        rng = random.Random(seed)
        pairs = list(pairs)
        for _epoch in range(epochs):
            rng.shuffle(pairs)
            for positive, negative in pairs:
                diff = [float(p) - float(n) for p, n in zip(positive, negative)]
                margin = self.bias + sum(weight * value for weight, value in zip(self.weights, diff))
                grad = 1.0 / (1.0 + math.exp(max(-50.0, min(50.0, margin))))
                for idx, value in enumerate(diff):
                    self.weights[idx] += lr * grad * value

    def ranking_error(self, pairs: Sequence[Pair]) -> float:
        if not pairs:
            return 0.0
        wrong = sum(1 for positive, negative in pairs if self.score(positive) <= self.score(negative))
        return wrong / len(pairs)

    def to_json(self) -> str:
        return json.dumps({"weights": self.weights, "bias": self.bias, "feature_names": canonical_feature_names()}, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "GenericHarnessRanker":
        data = json.loads(text)
        weights = [float(value) for value in data["weights"]]
        expected_dim = len(canonical_feature_names())
        if len(weights) != expected_dim:
            raise ValueError(f"ranker weight dimension mismatch: {len(weights)} != {expected_dim}")
        feature_names = data.get("feature_names")
        if feature_names is not None and list(feature_names) != canonical_feature_names():
            raise ValueError("ranker feature schema mismatch")
        return cls(weights, float(data.get("bias", 0.0)))


def train_generic_harness_ranker(
    tasks: Sequence[WorkflowTask],
    epochs: int = 20,
    seed: int = 0,
    action_intent_training: bool = False,
    max_states_per_task: int = 200,
) -> tuple[GenericHarnessRanker, List[Pair]]:
    records = []
    for task in tasks:
        records.extend(collect_workflow_oracle_records(task, max_states=max_states_per_task, action_intent_features=action_intent_training))
    pairs = make_workflow_ranking_pairs(records)
    ranker = GenericHarnessRanker.create(len(canonical_feature_names()), seed=seed)
    ranker.train(pairs, epochs=epochs, lr=0.03, seed=seed)
    return ranker, pairs


def evaluate_workflow_ranker(
    tasks: Sequence[WorkflowTask],
    ranker: GenericHarnessRanker,
    max_states_per_task: int = 200,
    contract_coverage: str = "full",
    action_intent_training: bool = False,
) -> Dict[str, object]:
    total = 0
    top1 = 0
    random_top1 = 0.0
    by_family: Dict[str, Dict[str, float]] = {}
    for task in tasks:
        frontier = [task.root_state()]
        seen = set()
        while frontier and len(seen) < max_states_per_task:
            state = frontier.pop(0)
            if state.key() in seen:
                continue
            seen.add(state.key())
            actions = task.candidate_actions(state)
            if len(actions) < 2:
                continue
            oracle_best = rank_actions_by_oracle(task, state, actions)[0].tool_index
            state_features = _apply_contract_coverage(workflow_runtime_features(task, state), contract_coverage)
            scored = []
            for action in actions:
                action_features = workflow_action_intent_features(task, state, action) if action_intent_training else workflow_action_features(task, state, action)
                vec = [*state_features.vector(), *action_features.vector(), *(s * a for s in state_features.vector() for a in action_features.vector())]
                scored.append((ranker.score(vec), action))
                child = task.apply(state, action)
                if child.remaining_turns > 0 and not task.success(child):
                    frontier.append(child)
            selected = max(scored, key=lambda item: (item[0], -item[1].tool_index))[1].tool_index
            total += 1
            top1 += 1 if selected == oracle_best else 0
            random_top1 += 1.0 / len(actions)
            family = by_family.setdefault(task.family, {"total": 0.0, "top1": 0.0})
            family["total"] += 1.0
            family["top1"] += 1.0 if selected == oracle_best else 0.0
    family_rows = {
        family: {"top1_accuracy": 0.0 if stats["total"] == 0 else stats["top1"] / stats["total"], "decisions": int(stats["total"])}
        for family, stats in sorted(by_family.items())
    }
    return {
        "decision_count": total,
        "top1_accuracy": 0.0 if total == 0 else top1 / total,
        "random_expected_top1": 0.0 if total == 0 else random_top1 / total,
        "contract_coverage": contract_coverage,
        "by_family": family_rows,
    }


def workflow_known_mask_coverage(tasks: Sequence[WorkflowTask], max_states_per_task: int = 200) -> Dict[str, object]:
    total = 0
    ratio_sum = 0.0
    per_feature: Dict[str, float] = {}
    for task in tasks:
        frontier = [task.root_state()]
        seen = set()
        while frontier and len(seen) < max_states_per_task:
            state = frontier.pop(0)
            if state.key() in seen:
                continue
            seen.add(state.key())
            features = workflow_runtime_features(task, state)
            masks = features.known_mask_values()
            total += 1
            ratio_sum += features.known_mask_ratio()
            for name, value in masks.items():
                per_feature[name] = per_feature.get(name, 0.0) + value
            for action in task.candidate_actions(state):
                child = task.apply(state, action)
                if child.remaining_turns > 0 and not task.success(child):
                    frontier.append(child)
    return {
        "state_count": total,
        "mean_known_mask_ratio": 0.0 if total == 0 else ratio_sum / total,
        "per_feature_availability": {name: value / total for name, value in sorted(per_feature.items())} if total else {},
    }


def _apply_contract_coverage(features, coverage: str):
    if coverage == "full":
        return features
    if coverage == "partial":
        return type(features)(
            verified_progress_frac=0.0,
            verified_progress_known_mask=0.0,
            remaining_requirement_frac=0.0,
            remaining_requirement_known_mask=0.0,
            observed_fact_frac=features.observed_fact_frac,
            observed_fact_known_mask=features.observed_fact_known_mask,
            remaining_turn_budget_frac=features.remaining_turn_budget_frac,
            remaining_tool_budget_frac=features.remaining_tool_budget_frac,
            constraint_pressure=features.constraint_pressure,
            constraint_pressure_known_mask=features.constraint_pressure_known_mask,
            valid_action_ratio=features.valid_action_ratio,
            previous_failure_rate=features.previous_failure_rate,
            stagnation_steps_frac=features.stagnation_steps_frac,
            uncertainty_score=max(features.uncertainty_score, 0.5),
            read_write_balance=features.read_write_balance,
            irreversible_action_used=features.irreversible_action_used,
        )
    if coverage == "black_box":
        return type(features)(
            verified_progress_frac=0.0,
            verified_progress_known_mask=0.0,
            remaining_requirement_frac=0.0,
            remaining_requirement_known_mask=0.0,
            observed_fact_frac=0.0,
            observed_fact_known_mask=0.0,
            remaining_turn_budget_frac=features.remaining_turn_budget_frac,
            remaining_tool_budget_frac=features.remaining_tool_budget_frac,
            constraint_pressure=0.0,
            constraint_pressure_known_mask=0.0,
            valid_action_ratio=features.valid_action_ratio,
            previous_failure_rate=features.previous_failure_rate,
            stagnation_steps_frac=features.stagnation_steps_frac,
            uncertainty_score=1.0,
            read_write_balance=features.read_write_balance,
            irreversible_action_used=features.irreversible_action_used,
        )
    raise ValueError(f"unknown contract coverage: {coverage}")


def run_workflow_source_validation(
    out_dir: Path,
    train_count: int = 80,
    validation_count: int = 30,
    seed: int = 0,
    epochs: int = 20,
    max_states_per_task: int = 200,
    action_intent_training: bool = False,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_tasks = generate_workflow_source_tasks(train_count, seed=seed, anonymize_tools=True)
    validation_tasks = generate_workflow_source_tasks(validation_count, seed=seed + 10000, anonymize_tools=True)
    named_tool_validation_tasks = generate_workflow_source_tasks(validation_count, seed=seed + 10000, anonymize_tools=False)
    ranker, train_pairs = train_generic_harness_ranker(
        train_tasks,
        epochs=epochs,
        seed=seed,
        action_intent_training=action_intent_training,
        max_states_per_task=max_states_per_task,
    )
    validation_records = []
    for task in validation_tasks:
        validation_records.extend(collect_workflow_oracle_records(task, max_states=max_states_per_task, action_intent_features=action_intent_training))
    validation_pairs = make_workflow_ranking_pairs(validation_records)
    eval_metrics = evaluate_workflow_ranker(validation_tasks, ranker, max_states_per_task=max_states_per_task, action_intent_training=action_intent_training)
    train_mask_coverage = workflow_known_mask_coverage(train_tasks, max_states_per_task=max_states_per_task)
    validation_mask_coverage = workflow_known_mask_coverage(validation_tasks, max_states_per_task=max_states_per_task)
    named_tool_eval_metrics = evaluate_workflow_ranker(named_tool_validation_tasks, ranker, max_states_per_task=max_states_per_task, action_intent_training=action_intent_training)
    partial_contract_eval_metrics = evaluate_workflow_ranker(
        validation_tasks,
        ranker,
        max_states_per_task=max_states_per_task,
        contract_coverage="partial",
        action_intent_training=action_intent_training,
    )
    black_box_contract_eval_metrics = evaluate_workflow_ranker(
        validation_tasks,
        ranker,
        max_states_per_task=max_states_per_task,
        contract_coverage="black_box",
        action_intent_training=action_intent_training,
    )
    tool_identity_top1_delta = abs(float(eval_metrics["top1_accuracy"]) - float(named_tool_eval_metrics["top1_accuracy"]))
    contract_coverage_rows = [
        {
            "condition": "full_contract",
            "top1_accuracy": eval_metrics["top1_accuracy"],
            "random_expected_top1": eval_metrics["random_expected_top1"],
            "decision_count": eval_metrics["decision_count"],
        },
        {
            "condition": "partial_contract_missing_progress_requirements",
            "top1_accuracy": partial_contract_eval_metrics["top1_accuracy"],
            "random_expected_top1": partial_contract_eval_metrics["random_expected_top1"],
            "decision_count": partial_contract_eval_metrics["decision_count"],
        },
        {
            "condition": "black_box_contract_missing_semantic_state",
            "top1_accuracy": black_box_contract_eval_metrics["top1_accuracy"],
            "random_expected_top1": black_box_contract_eval_metrics["random_expected_top1"],
            "decision_count": black_box_contract_eval_metrics["decision_count"],
        },
    ]
    contract_manifest = canonical_contract_manifest()
    manifest = {
        "benchmark": "canonical_harness_source_workflow_validation",
        "seed": seed,
        "train_count": train_count,
        "validation_count": validation_count,
        "epochs": epochs,
        "max_states_per_task": max_states_per_task,
        "source_action_abstraction": "available_action_intent" if action_intent_training else "instantiated_action",
        "action_intent_training": action_intent_training,
        "train_pair_count": len(train_pairs),
        "validation_pair_count": len(validation_pairs),
        "validation_ranking_error": ranker.ranking_error(validation_pairs),
        "validation_top1_accuracy": eval_metrics["top1_accuracy"],
        "validation_random_expected_top1": eval_metrics["random_expected_top1"],
        "canonical_feature_coverage": {
            "train": train_mask_coverage,
            "validation": validation_mask_coverage,
        },
        "tool_identity_generalization": {
            "anonymous_tool_validation_top1_accuracy": eval_metrics["top1_accuracy"],
            "semantic_tool_name_validation_top1_accuracy": named_tool_eval_metrics["top1_accuracy"],
            "top1_absolute_delta": tool_identity_top1_delta,
            "ready": tool_identity_top1_delta <= 1e-12,
            "stress_protocol": "same validation task seed with anonymous vs semantic tool names; canonical features exclude tool identity",
        },
        "contract_coverage_stress": {
            "ready": all(int(row["decision_count"]) == int(eval_metrics["decision_count"]) for row in contract_coverage_rows),
            "full_contract_top1_accuracy": eval_metrics["top1_accuracy"],
            "partial_contract_top1_accuracy": partial_contract_eval_metrics["top1_accuracy"],
            "black_box_contract_top1_accuracy": black_box_contract_eval_metrics["top1_accuracy"],
            "partial_contract_top1_delta_from_full": float(eval_metrics["top1_accuracy"]) - float(partial_contract_eval_metrics["top1_accuracy"]),
            "black_box_contract_top1_delta_from_full": float(eval_metrics["top1_accuracy"]) - float(black_box_contract_eval_metrics["top1_accuracy"]),
            "stress_protocol": "evaluate the frozen source-trained controller under explicit known-mask degradation without changing labels, tasks, or ranker weights",
        },
        "contract": contract_manifest,
        "contract_audit": audit_canonical_contract_manifest(contract_manifest),
        "source_generator": source_generator_manifest(seed, train_count),
        "controller_checkpoint_frozen": True,
        "adapter_changes_checkpoint_stays_frozen": True,
        "tau3_train_trajectories_used": False,
        "oracle_labels_in_inference_features": False,
        "learned_score_authorizes_pruning": False,
        "tool_names_used_as_features": False,
        "families": eval_metrics["by_family"],
    }
    (out_dir / "generic_harness_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    (out_dir / "canonical_harness_source_validation.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(
        out_dir / "canonical_harness_source_validation.csv",
        [
            {
                "controller": "GenericHarnessRanker",
                "train_pairs": len(train_pairs),
                "validation_pairs": len(validation_pairs),
                "ranking_error": manifest["validation_ranking_error"],
                "top1_accuracy": manifest["validation_top1_accuracy"],
                "random_expected_top1": manifest["validation_random_expected_top1"],
            }
        ],
    )
    _write_csv(
        out_dir / "canonical_harness_tool_identity_stress.csv",
        [
            {
                "condition": "anonymous_tool_names",
                "top1_accuracy": eval_metrics["top1_accuracy"],
                "random_expected_top1": eval_metrics["random_expected_top1"],
                "decision_count": eval_metrics["decision_count"],
            },
            {
                "condition": "semantic_tool_names",
                "top1_accuracy": named_tool_eval_metrics["top1_accuracy"],
                "random_expected_top1": named_tool_eval_metrics["random_expected_top1"],
                "decision_count": named_tool_eval_metrics["decision_count"],
            },
        ],
    )
    _write_csv(out_dir / "canonical_harness_contract_coverage_stress.csv", contract_coverage_rows)
    _write_csv(
        out_dir / "canonical_harness_feature_coverage.csv",
        [
            {
                "split": split,
                "state_count": coverage["state_count"],
                "mean_known_mask_ratio": coverage["mean_known_mask_ratio"],
                **{
                    f"availability_{name}": value
                    for name, value in dict(coverage["per_feature_availability"]).items()
                },
            }
            for split, coverage in [("train", train_mask_coverage), ("validation", validation_mask_coverage)]
        ],
    )
    return manifest


def run_structured_workflow_transfer(
    out_dir: Path,
    train_count: int = 200,
    validation_count: int = 50,
    test_count: int = 100,
    seed: int = 0,
    epochs: int = 20,
    node_budget: int = 80,
    max_states_per_task: int = 200,
    action_intent_training: bool = True,
    source_families: Sequence[str] = SOURCE_WORKFLOW_FAMILIES,
    target_families: Sequence[str] = TARGET_WORKFLOW_FAMILIES,
    source_difficulty: str = "v2",
    target_difficulty: str = "v2",
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    source_families = tuple(source_families)
    target_families = tuple(target_families)
    if set(source_families) & set(target_families):
        raise ValueError("source and target workflow families must be disjoint for transfer evaluation")
    train_tasks = generate_workflow_source_tasks(train_count, seed=seed, anonymize_tools=True, families=source_families, difficulty=source_difficulty)
    validation_tasks = generate_workflow_source_tasks(validation_count, seed=seed + 10000, anonymize_tools=True, families=source_families, difficulty=source_difficulty)
    target_tasks = generate_workflow_source_tasks(test_count, seed=seed + 20000, anonymize_tools=True, families=target_families, difficulty=target_difficulty)
    ranker, train_pairs = train_generic_harness_ranker(
        train_tasks,
        epochs=epochs,
        seed=seed,
        action_intent_training=action_intent_training,
        max_states_per_task=max_states_per_task,
    )
    validation_metrics = evaluate_workflow_ranker(
        validation_tasks,
        ranker,
        max_states_per_task=max_states_per_task,
        action_intent_training=action_intent_training,
    )
    rows = []
    per_instance_rows = []
    for policy in ("Random", "Hand heuristic", "Source-trained controller", "Oracle"):
        results = []
        for idx, task in enumerate(target_tasks):
            result = _run_workflow_policy(
                task,
                policy=policy,
                ranker=ranker,
                seed=seed * 100000 + idx,
                node_budget=node_budget,
                action_intent_training=action_intent_training,
            )
            results.append(result)
            per_instance_rows.append({"policy": policy, **result})
        rows.append(_aggregate_workflow_results(policy, results))
    manifest = {
        "benchmark": "structured_tool_workflow_transfer",
        "seed": seed,
        "train_count": train_count,
        "validation_count": validation_count,
        "test_count": test_count,
        "epochs": epochs,
        "node_budget": node_budget,
        "max_states_per_task": max_states_per_task,
        "action_intent_training": action_intent_training,
        "source_families": list(source_families),
        "target_families": list(target_families),
        "source_difficulty": source_difficulty,
        "target_difficulty": target_difficulty,
        "source_target_family_overlap": sorted(set(source_families) & set(target_families)),
        "train_pair_count": len(train_pairs),
        "source_validation_top1_accuracy": validation_metrics["top1_accuracy"],
        "source_validation_random_expected_top1": validation_metrics["random_expected_top1"],
        "source_generator": source_generator_manifest(seed, train_count, families=source_families, difficulty=source_difficulty),
        "target_generator": source_generator_manifest(seed + 20000, test_count, families=target_families, difficulty=target_difficulty),
        "contract": canonical_contract_manifest(),
        "controller_checkpoint_frozen": True,
        "target_trajectory_training_used": False,
        "oracle_labels_in_inference_features": False,
        "tool_names_used_as_features": False,
        "rows": rows,
    }
    (out_dir / "structured_workflow_ranker.json").write_text(ranker.to_json(), encoding="utf-8")
    (out_dir / "structured_workflow_transfer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "structured_workflow_transfer_table.csv", rows)
    _write_csv(out_dir / "structured_workflow_transfer_per_instance.csv", per_instance_rows)
    _write_csv(
        out_dir / "structured_workflow_source_validation.csv",
        [
            {
                "controller": "GenericHarnessRanker",
                "train_pairs": len(train_pairs),
                "top1_accuracy": validation_metrics["top1_accuracy"],
                "random_expected_top1": validation_metrics["random_expected_top1"],
                "decision_count": validation_metrics["decision_count"],
            }
        ],
    )
    return manifest


def _run_workflow_policy(
    task: WorkflowTask,
    *,
    policy: str,
    ranker: GenericHarnessRanker,
    seed: int,
    node_budget: int,
    action_intent_training: bool,
) -> Dict[str, object]:
    rng = random.Random(seed)
    state = task.root_state()
    nodes = 0
    decisions = 0
    opportunity = 0
    while nodes < node_budget and task.candidate_actions(state):
        actions = task.candidate_actions(state)
        decisions += 1
        opportunity += 1 if len(actions) >= 2 else 0
        action = _select_workflow_action(task, state, actions, policy=policy, ranker=ranker, rng=rng, action_intent_training=action_intent_training)
        state = task.apply(state, action)
        nodes += 1
        if state.terminal:
            break
    optimal_reward = _oracle_rollout_reward(task, node_budget=node_budget)
    reward = task.reward(state)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "success": 1 if task.success(state) else 0,
        "reward": reward,
        "optimal_reward": optimal_reward,
        "gap": max(0.0, optimal_reward - reward) / max(1.0, abs(optimal_reward)),
        "nodes": nodes,
        "decisions": decisions,
        "opportunity_rate": 0.0 if decisions == 0 else opportunity / decisions,
        "violations": state.violations,
        "completed_requirements": sum(1 for item in state.completed_requirements if item),
        "requirement_count": task.requirement_count,
        "terminal": 1 if state.terminal else 0,
    }


def _select_workflow_action(
    task: WorkflowTask,
    state: WorkflowState,
    actions: Sequence[WorkflowAction],
    *,
    policy: str,
    ranker: GenericHarnessRanker,
    rng: random.Random,
    action_intent_training: bool,
) -> WorkflowAction:
    if policy == "Random":
        return rng.choice(list(actions))
    if policy == "Oracle":
        return rank_actions_by_oracle(task, state, actions)[0]
    if policy == "Hand heuristic":
        return max(actions, key=lambda action: (_hand_heuristic_workflow_score(task, state, action), -action.tool_index))
    if policy == "Source-trained controller":
        state_features = workflow_runtime_features(task, state)
        scored = []
        for action in actions:
            action_features = workflow_action_intent_features(task, state, action) if action_intent_training else workflow_action_features(task, state, action)
            vec = [*state_features.vector(), *action_features.vector(), *(s * a for s in state_features.vector() for a in action_features.vector())]
            scored.append((ranker.score(vec), action))
        return max(scored, key=lambda item: (item[0], -item[1].tool_index))[1]
    raise ValueError(f"unknown workflow policy: {policy}")


def _hand_heuristic_workflow_score(task: WorkflowTask, state: WorkflowState, action: WorkflowAction) -> float:
    tool = task.tools[action.tool_index]
    runtime = workflow_runtime_features(task, state)
    completed = runtime.remaining_requirement_frac <= 0.0
    if tool.action_type in ("respond", "terminate"):
        return 10.0 if completed else -10.0
    if tool.action_type == "read":
        return 3.0 if runtime.uncertainty_score > 0.0 else -2.0
    if tool.action_type == "ask":
        return 2.0 if runtime.uncertainty_score > 0.0 and runtime.observed_fact_frac >= 0.5 else -3.0
    if tool.action_type == "write":
        return 4.0 if runtime.observed_fact_frac >= 0.6 else -4.0
    return 0.0


def _oracle_rollout_reward(task: WorkflowTask, node_budget: int) -> float:
    state = task.root_state()
    nodes = 0
    while nodes < node_budget and task.candidate_actions(state):
        action = rank_actions_by_oracle(task, state, task.candidate_actions(state))[0]
        state = task.apply(state, action)
        nodes += 1
        if state.terminal:
            break
    return task.reward(state)


def _aggregate_workflow_results(policy: str, results: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    total = len(results)
    family_stats: Dict[str, Dict[str, float]] = {}
    for row in results:
        family = str(row["family"])
        stats = family_stats.setdefault(family, {"count": 0.0, "success": 0.0, "gap": 0.0})
        stats["count"] += 1.0
        stats["success"] += float(row["success"])
        stats["gap"] += float(row["gap"])
    return {
        "policy": policy,
        "success": 0.0 if total == 0 else sum(float(row["success"]) for row in results) / total,
        "gap": 0.0 if total == 0 else sum(float(row["gap"]) for row in results) / total,
        "reward": 0.0 if total == 0 else sum(float(row["reward"]) for row in results) / total,
        "nodes": 0.0 if total == 0 else sum(float(row["nodes"]) for row in results) / total,
        "violations": 0.0 if total == 0 else sum(float(row["violations"]) for row in results) / total,
        "opportunity_rate": 0.0 if total == 0 else sum(float(row["opportunity_rate"]) for row in results) / total,
        "tokens": 0,
        "llm_calls": 0,
        "test_count": total,
        "by_family": json.dumps(
            {
                family: {
                    "success": stats["success"] / stats["count"],
                    "gap": stats["gap"] / stats["count"],
                    "count": int(stats["count"]),
                }
                for family, stats in sorted(family_stats.items())
            },
            sort_keys=True,
        ),
    }


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    import csv

    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
