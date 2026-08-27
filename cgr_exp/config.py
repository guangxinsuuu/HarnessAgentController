from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict

from cgr_exp.experiments import ExperimentConfig


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return asdict(config)


def load_json_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_dict(raw: Dict[str, Any], base: ExperimentConfig) -> ExperimentConfig:
    validate_basic_config(raw)
    flat = flatten_config(raw)
    return ExperimentConfig(**{**asdict(base), **flat})


def load_and_validate_config(path: Path, base: ExperimentConfig) -> ExperimentConfig:
    return config_from_dict(load_json_config(path), base)


def flatten_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if "seed" in raw:
        flat["seed"] = raw["seed"]
    dataset = raw.get("dataset", {})
    if "train_instances" in dataset:
        flat["train_count"] = dataset["train_instances"]
    if "test_instances" in dataset:
        flat["test_count"] = dataset["test_instances"]
    proposer = raw.get("proposer", {})
    if "k" in proposer:
        flat["action_budget"] = proposer["k"]
    if "useful_action_recall" in proposer:
        flat["llm_recall"] = proposer["useful_action_recall"]
    if "invalid_action_prob" in proposer:
        flat["llm_invalid_rate"] = proposer["invalid_action_prob"]
    if "duplicate_prob" in proposer:
        flat["llm_duplicate_rate"] = proposer["duplicate_prob"]
    verification = raw.get("verification", {})
    if "false_negative_rate" in verification:
        flat["false_negative_rate"] = verification["false_negative_rate"]
    if "false_positive_rate" in verification:
        flat["false_positive_rate"] = verification["false_positive_rate"]
    search = raw.get("search", {})
    if "node_budget" in search:
        flat["node_budget"] = search["node_budget"]
    training = raw.get("training", {})
    if "target_adaptation_trajectories" in training:
        values = training["target_adaptation_trajectories"]
        if isinstance(values, int):
            flat["fewshot_counts"] = (values,) if values > 0 else ()
        else:
            flat["fewshot_counts"] = tuple(v for v in values if v > 0)
    return flat


def validate_basic_config(raw: Dict[str, Any]) -> None:
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")
    if "seed" in raw and not isinstance(raw["seed"], int):
        raise ValueError("seed must be an integer")
    for section in ("dataset", "proposer", "verification", "search", "training", "graph", "memory"):
        if section in raw and not isinstance(raw[section], dict):
            raise ValueError(f"{section} must be an object")
    dataset = raw.get("dataset", {})
    _positive_int(dataset, "train_instances")
    _nonnegative_int(dataset, "val_instances")
    _positive_int(dataset, "test_instances")
    proposer = raw.get("proposer", {})
    _positive_int(proposer, "k")
    _probability(proposer, "useful_action_recall")
    _probability(proposer, "invalid_action_prob")
    _probability(proposer, "duplicate_prob")
    verification = raw.get("verification", {})
    _probability(verification, "false_negative_rate")
    _probability(verification, "false_positive_rate")
    _probability(verification, "uncertain_noise_rate")
    search = raw.get("search", {})
    _positive_int(search, "node_budget")
    _positive_int(search, "proposal_call_budget")
    training = raw.get("training", {})
    if "target_adaptation_trajectories" in training:
        values = training["target_adaptation_trajectories"]
        if isinstance(values, int):
            values = [values]
        if not isinstance(values, list) or not all(isinstance(v, int) and v >= 0 for v in values):
            raise ValueError("training.target_adaptation_trajectories must be a non-negative integer or list of non-negative integers")
    memory = raw.get("memory", {})
    _nonnegative_int(memory, "top_r")


def _probability(section: Dict[str, Any], key: str) -> None:
    if key in section and (not isinstance(section[key], (int, float)) or not 0.0 <= float(section[key]) <= 1.0):
        raise ValueError(f"{key} must be a probability in [0, 1]")


def _positive_int(section: Dict[str, Any], key: str) -> None:
    if key in section and (not isinstance(section[key], int) or section[key] <= 0):
        raise ValueError(f"{key} must be a positive integer")


def _nonnegative_int(section: Dict[str, Any], key: str) -> None:
    if key in section and (not isinstance(section[key], int) or section[key] < 0):
        raise ValueError(f"{key} must be a non-negative integer")
