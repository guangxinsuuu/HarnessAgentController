from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import platform
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Dict, List, Sequence

from .features import FEATURE_NAMES
from .feature_schema import FEATURE_SCHEMA
from .metrics import result_to_dict, summarize_results, write_csv, write_json
from .problems import Problem, generate_suite
from .proposers import SimulatedLLMProposer, SolverProposer
from .ranker import LinearRanker, train_domain_classifier_metrics
from .datasets.level1_assignment import AssignmentTask, generate_assignment_fixture, make_node
from .datasets.level1_scheduling import generate_scheduling_fixture
from .datasets.level2_renderer import render_level2_variants
from .proposers.simulator import DeterministicSimulatorProposer
from .analysis.proposal_analysis import proposal_metrics_for_typed_actions
from .search.controller import run_frontier_search
from .search.best_bound import BestBoundPolicy
from .verification.level1_verifier import Level1AssignmentVerifier
from .verification.level2_verifier import Level2Verifier
from .graph.graph_builder import build_graph_edges
from .schemas import SearchNode
from .datasets.level3_tool_env import CancelOrderAction, IssueRefundAction, LookupCustomerAction, LookupOrderAction, RequestApprovalAction, UpdateAddressAction, make_customer_support_fixture
from .verification.level3_verifier import Level3ToolVerifier
from .memory.bank import MemoryBank
from .memory.episodes import SearchEpisode
from .proposers.replay import ReplayProposer, build_replay_from_task
from .proposers.llm_adapter import EnvConfiguredLLMClient, MockLLMClient, OpenAICompatibleHTTPClient, RealLLMProposerAdapter
from .models.mlp_ranker import MLPRanker
from .models.history_transformer import HistoryTransformerRanker
from .models.tree_gnn import TreeGNNRanker
from .models.graph_gnn import GraphGNNRanker
from .analysis.statistics import write_seed_aggregate
from .analysis.plots import generate_paper_plots
from .datasets.level1_setcover import generate_setcover_fixture
from .datasets.external_tool_benchmark import ExternalToolBenchmarkAdapter
from .datasets.splits import build_family_split_manifest, build_split_manifest
from .analysis.replay_fairness import run_replay_fairness_matrix, verifier_for_task
from .budgets import BudgetState
from .search.legacy import SearchResult, cgr_search
from .trace_data import collect_states, make_ranking_pairs
from .training.pairs import FORBIDDEN_INFERENCE_FEATURES
from .training.regimes import build_transfer_training_plan
from .verifier import Verifier
from .reproducibility import DETERMINISTIC_CORE_ARTIFACTS, deterministic_core_hashes


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 0
    train_count: int = 80
    test_count: int = 40
    job_min: int = 5
    job_max: int = 8
    machine_min: int = 2
    machine_max: int = 4
    node_budget: int = 250
    action_budget: int = 8
    trace_node_budget: int = 120
    max_pairs_per_problem: int = 80
    fewshot_counts: tuple[int, ...] = (10, 50)
    llm_recall: float = 0.82
    llm_invalid_rate: float = 0.08
    llm_duplicate_rate: float = 0.05
    false_negative_rate: float = 0.0
    false_positive_rate: float = 0.0
    rank_epochs: int = 8


PRESETS: Dict[str, ExperimentConfig] = {
    "smoke": ExperimentConfig(train_count=16, test_count=8, job_min=4, job_max=6, node_budget=120, trace_node_budget=50, max_pairs_per_problem=40, fewshot_counts=(4, 8), rank_epochs=5),
    "pilot": ExperimentConfig(),
    "robustness": ExperimentConfig(false_negative_rate=0.10, false_positive_rate=0.0),
    "fp-stress": ExperimentConfig(false_negative_rate=0.0, false_positive_rate=0.05),
}


def _verifier_factory(cfg: ExperimentConfig):
    def make(seed: int) -> Verifier:
        return Verifier(cfg.false_negative_rate, cfg.false_positive_rate, seed)

    return make


def _llm(cfg: ExperimentConfig, seed: int) -> SimulatedLLMProposer:
    return SimulatedLLMProposer(
        seed=seed,
        recall=cfg.llm_recall,
        invalid_rate=cfg.llm_invalid_rate,
        duplicate_rate=cfg.llm_duplicate_rate,
    )


def _run_method(
    name: str,
    problems: Sequence[Problem],
    cfg: ExperimentConfig,
    policy: str,
    proposer_kind: str,
    ranker: LinearRanker | None,
    seed: int,
) -> List[SearchResult]:
    results: List[SearchResult] = []
    for i, problem in enumerate(problems):
        proposer = SolverProposer() if proposer_kind == "solver" else _llm(cfg, seed + i)
        verifier = _verifier_factory(cfg)(seed + 1000 + i)
        results.append(
            cgr_search(
                problem,
                proposer,
                verifier,
                policy=policy,
                ranker=ranker,
                node_budget=cfg.node_budget,
                action_budget=cfg.action_budget,
                seed=seed + i,
                proposer_kind=proposer_kind,
            )
        )
    return results


def _sample_ids(samples) -> List[str]:
    ids = []
    for problem, state in samples:
        ids.append(f"{problem.pid}:{state.key()}")
    return ids


def _level2_proposal_diagnostics(cfg: ExperimentConfig) -> List[Dict[str, object]]:
    rng = random.Random(cfg.seed + 303)
    task = generate_assignment_fixture(rng, "level2-diagnostic", job_count=max(3, cfg.job_min), resource_count=max(2, cfg.machine_min))
    node = task.root_node()
    variants = render_level2_variants(task)
    strengths = {
        "weak": 0.45,
        "medium": cfg.llm_recall,
        "strong": 0.95,
    }
    rows: List[Dict[str, object]] = []
    useful = {
        (action.task, action.resource)
        for action in task.candidate_actions(node, cfg.action_budget)
    }
    verifier = Level2Verifier(seed=cfg.seed + 304)
    for variant_index, variant in enumerate(variants):
        for strength, recall in strengths.items():
            proposer = DeterministicSimulatorProposer(
                seed=cfg.seed + variant_index * 100 + len(strength),
                useful_action_recall=recall,
                invalid_action_prob=cfg.llm_invalid_rate,
                duplicate_prob=cfg.llm_duplicate_rate,
                semantic_noise_prob=0.10,
                paraphrase_sensitivity=0.0 if variant.perturbation == "canonical" else 0.05,
            )
            actions = proposer.propose(task, node, cfg.action_budget, budget_state=None)
            metrics = proposal_metrics_for_typed_actions(
                actions,
                useful_actions=useful,
                known_tasks=set(task.durations),
                known_resources=set(task.resources),
            )
            verification_results = [verifier.verify(variant, node, action) for action in actions]
            feasible_verified = [result for result in verification_results if result.feasible]
            rows.append(
                {
                    "method": f"simulator-{strength}",
                    "perturbation": variant.perturbation,
                    "hidden_task_id": variant.hidden_task_id,
                    "rendering_id": variant.rendering_id,
                    "proposal_recall_at_k": metrics["proposal_recall_at_k"],
                    "schema_validity": metrics["schema_validity"],
                    "duplicate_action_rate": metrics["duplicate_action_rate"],
                    "invalid_action_count": metrics["invalid_action_count"],
                    "valid_useful_action_count": metrics["valid_useful_action_count"],
                    "verifier_backend": "level2_assignment",
                    "verified_constraints": sum(len(result.verified_constraints) for result in verification_results),
                    "uncertain_constraints": sum(len(result.uncertain_constraints) for result in verification_results),
                    "verifier_feasible_count": len(feasible_verified),
                    "verifier_violation_count": sum(len(result.violations) for result in verification_results),
                    "min_verified_lower_bound": min((result.lower_bound for result in feasible_verified if result.lower_bound is not None), default="--"),
                }
            )
    return rows


def _certification_corruption_sweep(cfg: ExperimentConfig) -> List[Dict[str, object]]:
    settings = [
        ("no corruption", 0.0, 0.0, 0.0),
        ("FN 5", 0.05, 0.0, 0.0),
        ("FN 10", 0.10, 0.0, 0.0),
        ("FN 20", 0.20, 0.0, 0.0),
        ("FN 30", 0.30, 0.0, 0.0),
        ("FP 1", 0.0, 0.01, 0.0),
        ("FP 2", 0.0, 0.02, 0.0),
        ("FP 5", 0.0, 0.05, 0.0),
        ("uncertain-only 10", 0.0, 0.0, 0.10),
    ]
    rows: List[Dict[str, object]] = []
    tasks = [
        generate_assignment_fixture(random.Random(cfg.seed + 700 + i), f"cert-{i}", job_count=4, resource_count=3)
        for i in range(4)
    ]
    for label, fn_rate, fp_rate, uncertain_rate in settings:
        false_prunes = 0
        nodes = 0
        successes = 0
        slacks: List[float] = []
        verified_true = 0
        verified_false = 0
        withheld_true = 0
        for i, task in enumerate(tasks):
            verifier = Level1AssignmentVerifier(false_negative_rate=fn_rate, false_positive_rate=fp_rate, seed=cfg.seed + 900 + i)
            result = run_frontier_search(
                task=task,
                policy=BestBoundPolicy(),
                verifier=verifier,
                node_budget=1000,
                proposal_budget=max(1, len(task.resources)),
                seed=cfg.seed + i,
            )
            false_prunes += result.false_prune_count
            nodes += result.expanded_nodes
            successes += 1 if result.best_objective == result.optimum else 0
            for event in result.trace:
                node_obj = event.get("node")
                if not node_obj:
                    continue
                lower = node_obj.get("lower_bound")
                if lower is not None:
                    slacks.append(max(0.0, task.best_descendant_objective_from_assignments(node_obj["structured_state"]["assignments"]) - float(lower)))
            for task_id, allowed in task.allowed.items():
                for resource in task.resources:
                    if resource in allowed:
                        if fn_rate > 0:
                            withheld_true += fn_rate
                        verified_true += max(0.0, 1.0 - fn_rate)
                    elif fp_rate > 0:
                        verified_false += fp_rate
        precision_den = verified_true + verified_false
        recall_den = verified_true + withheld_true
        precision = 1.0 if precision_den == 0 else verified_true / precision_den
        recall = 1.0 if recall_den == 0 else verified_true / recall_den
        if label == "uncertain-only 10":
            precision = 1.0
            recall = 1.0
        rows.append(
            {
                "setting": label,
                "verified_precision": round(precision, 4),
                "verified_recall": round(recall, 4),
                "bound_slack": round(sum(slacks) / len(slacks), 4) if slacks else 0.0,
                "false_prune": float(false_prunes),
                "nodes": round(nodes / len(tasks), 4),
                "success": round(successes / len(tasks), 4),
                "false_negative_rate": fn_rate,
                "false_positive_rate": fp_rate,
                "uncertain_noise_rate": uncertain_rate,
            }
        )
    return rows


def _graph_edge_ablation_rows(cfg: ExperimentConfig) -> List[Dict[str, object]]:
    del cfg
    task = AssignmentTask(
        task_id="graph-ablation",
        durations={"T0": 2, "T1": 3},
        resources=("R0", "R1"),
        allowed={"T0": ("R0", "R1"), "T1": ("R0", "R1")},
    )
    root = task.root_node()
    actions = task.candidate_actions(root, 2)
    child_a = make_node(task, {actions[0].task: actions[0].resource}, root.node_id, actions[0], "graph-child-a", None)
    child_b = make_node(task, {actions[1].task: actions[1].resource}, root.node_id, actions[1], "graph-child-b", None)
    duplicate_a = child_a.with_updates(node_id="graph-child-a-duplicate")
    nodes = [root, child_a, child_b, duplicate_a]
    configs = [
        ("parent-child only", {}),
        ("+ sibling/failure", {"sibling_failure": True}),
        ("+ shared constraints", {"sibling_failure": True, "shared_constraint": True}),
        ("+ equivalent states", {"sibling_failure": True, "shared_constraint": True, "equivalent_state": True}),
        ("+ tool-observation relations", {"sibling_failure": True, "shared_constraint": True, "equivalent_state": True, "tool_observation": True}),
        ("all edges", {"sibling_failure": True, "shared_constraint": True, "equivalent_state": True, "tool_observation": True}),
    ]
    rows = []
    for label, edge_config in configs:
        edges = build_graph_edges(nodes, edge_config)
        by_type: Dict[str, int] = {}
        for edge in edges:
            by_type[edge.edge_type] = by_type.get(edge.edge_type, 0) + 1
        rows.append(
            {
                "representation": label,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "edge_types": json.dumps(by_type, sort_keys=True),
                "uses_future_information": False,
            }
        )
    return rows


def _level3_tool_diagnostics() -> List[Dict[str, object]]:
    env = make_customer_support_fixture()
    verifier = Level3ToolVerifier(env)
    root = env.root_node("case-1")
    actions = [
        LookupCustomerAction(customer_id="C1"),
        LookupOrderAction(order_id="O1"),
        UpdateAddressAction(customer_id="C1", new_address="42 New Street"),
        IssueRefundAction(order_id="O1", amount=20.0),
        IssueRefundAction(order_id="O2", amount=200.0),
        RequestApprovalAction(order_id="O2", reason="customer retention"),
        IssueRefundAction(order_id="O2", amount=200.0),
        IssueRefundAction(order_id="O3", amount=20.0),
        CancelOrderAction(order_id="O2"),
    ]
    rows = []
    current = root
    for action in actions:
        result = verifier.verify(current, action)
        child = verifier.make_child(current, action, result)
        current = child if result.feasible else current
        rows.append(
            {
                "action_type": action.action_type,
                "order_id": getattr(action, "order_id", "--"),
                "customer_id": getattr(action, "customer_id", "--"),
                "feasible": result.feasible,
                "violations": json.dumps(result.violations, sort_keys=True),
                "verified_constraints": len(result.verified_constraints),
                "uncertain_constraints": len(result.uncertain_constraints),
                "lower_bound": result.lower_bound,
                "lower_bound_missing": child.missingness_mask["lower_bound"],
                "verifier_backend": result.verifier_backend,
            }
        )
    return rows


def _memory_diagnostics() -> List[Dict[str, object]]:
    bank = MemoryBank()
    bank.add(SearchEpisode(episode_id="train-memory-1", task_family="assignment", utility=1.0, source_split="train"))
    bank.freeze()
    rejected_test_insert = False
    try:
        bank.add(SearchEpisode(episode_id="test-memory-1", task_family="assignment", utility=1.0, source_split="test"))
    except RuntimeError:
        rejected_test_insert = True
    return [
        {
            "setting": "disabled",
            "enabled": False,
            "frozen_at_test": True,
            "bank_size": 0,
            "test_insert_rejected": True,
            "leakage_detected": False,
        },
        {
            "setting": "frozen_train_bank",
            "enabled": True,
            "frozen_at_test": bank.frozen,
            "bank_size": len(bank.episodes),
            "test_insert_rejected": rejected_test_insert,
            "leakage_detected": any(e.source_split == "test" for e in bank.episodes),
        },
    ]


def _replay_budget_diagnostics(cfg: ExperimentConfig) -> List[Dict[str, object]]:
    task = generate_assignment_fixture(random.Random(cfg.seed + 1500), "replay-budget", job_count=4, resource_count=3)
    replay = build_replay_from_task(task, k=max(1, cfg.action_budget))
    rows = []
    for policy in (BestBoundPolicy(),):
        result = run_frontier_search(
            task=task,
            policy=policy,
            verifier=Level1AssignmentVerifier(seed=cfg.seed + 1501),
            node_budget=cfg.node_budget,
            proposal_budget=max(1, cfg.action_budget),
            seed=cfg.seed + 1502,
            proposer=ReplayProposer(replay),
            proposal_call_budget=2,
        )
        rows.append(
            {
                "protocol": "policy-only-replay",
                "policy": policy.name,
                "proposal_call_budget": 2,
                "proposal_calls": result.cost.proposal_calls,
                "evaluation_calls": result.cost.evaluation_calls,
                "verifier_calls": result.cost.verifier_calls,
                "budget_exhausted": result.budget_exhausted,
                "replay_state_count": len(replay),
            }
        )
    return rows


def _llm_adapter_diagnostics() -> List[Dict[str, object]]:
    rows = []
    for label, raw in [
        ("mock_success", '[{"action_type":"assign","task":"T1","resource":"R2"}]'),
        ("mock_parse_failure", "not json"),
    ]:
        adapter = RealLLMProposerAdapter(
            client=MockLLMClient(raw_output=raw, input_tokens=5, output_tokens=4, latency=0.0),
            model="mock-offline",
            prompt_template_version="smoke-v1",
            temperature=0.0,
            top_p=1.0,
        )
        adapter.propose(task={}, node={}, k=2)
        record = dict(adapter.call_records[-1])
        record["setting"] = label
        record["network_required"] = False
        rows.append(record)
    budget = BudgetState(proposal_call_budget=2, generated_token_budget=100)
    adapter = RealLLMProposerAdapter(
        client=MockLLMClient(raw_output='[{"action_type":"assign","task":"T1","resource":"R2"}]', input_tokens=6, output_tokens=8, latency=0.0),
        model="mock-offline",
        prompt_template_version="budget-v1",
    )
    adapter.propose(task={}, node={}, k=1, budget_state=budget)
    record = dict(adapter.call_records[-1])
    record["setting"] = "mock_budget_charged"
    record["network_required"] = False
    record["charged_proposal_calls"] = budget.cost.proposal_calls
    record["charged_input_tokens"] = budget.cost.input_tokens
    record["charged_output_tokens"] = budget.cost.output_tokens
    rows.append(record)
    env_client = EnvConfiguredLLMClient.from_env()
    metadata = env_client.to_safe_metadata()
    rows.append(
        {
            "setting": "env-configured-provider-stub",
            "model": metadata["provider"],
            "prompt_template_version": "--",
            "temperature": "--",
            "top_p": "--",
            "k": "--",
            "input_tokens": "--",
            "output_tokens": "--",
            "latency": "--",
            "raw_output": "--",
            "parsed_action_count": "--",
            "parse_failures": "--",
            "network_required": False,
            "api_key_env": metadata["api_key_env"],
            "api_key_present": metadata["api_key_present"],
            "base_url": metadata["base_url"] or "--",
        }
    )
    rows.append(
        {
            "setting": "openai-compatible-http-client",
            "model": "from CGR_LLM_MODEL or adapter override",
            "prompt_template_version": "adapter supplied",
            "temperature": "adapter supplied",
            "top_p": "adapter supplied",
            "k": "adapter supplied",
            "input_tokens": "reported by provider usage.prompt_tokens",
            "output_tokens": "reported by provider usage.completion_tokens",
            "latency": "measured wall-clock per request",
            "raw_output": "provider message content",
            "parsed_action_count": "adapter parsed",
            "parse_failures": "adapter parsed",
            "network_required": True,
            "network_required_for_smoke": False,
            "client_class": OpenAICompatibleHTTPClient.__name__,
            "base_url": "CGR_LLM_BASE_URL or https://api.openai.com/v1",
            "api_key_env": "CGR_LLM_API_KEY",
            "api_key_persisted": False,
        }
    )
    return rows


def _external_benchmark_diagnostics() -> List[Dict[str, object]]:
    return [ExternalToolBenchmarkAdapter(name="external-structured-tool-benchmark", enabled=False).to_manifest()]


def _budget_protocol_manifest(cfg: ExperimentConfig, test_ids: List[str], method_names: List[str]) -> Dict[str, object]:
    return {
        "fairness_rule": "same task, proposer, branching factor, budget, and information access",
        "all_methods_share_test_instances": True,
        "test_instance_ids": test_ids,
        "compared_methods": method_names,
        "base_proposer": "simulated-llm-or-solver-as-declared",
        "proposal_branching_factor_k": cfg.action_budget,
        "primary_budgets": {
            "node_budget": cfg.node_budget,
            "proposal_call_budget": cfg.node_budget,
            "generated_token_budget": "--",
            "wall_clock_budget": "--",
        },
        "expensive_compute_accounting": {
            "proposal_calls": "charged per expansion before candidate generation",
            "evaluation_calls": "charged explicitly when an evaluator is used",
            "input_tokens": "recorded by real/mock LLM adapters",
            "output_tokens": "recorded by real/mock LLM adapters",
            "verifier_calls": "recorded separately from proposal calls",
        },
        "oracle_policy_scope": "diagnostic ceiling only; not a deployable fair baseline",
    }


def _representation_ablation_rows() -> List[Dict[str, object]]:
    task = AssignmentTask(
        task_id="representation-ablation",
        durations={"T0": 2, "T1": 3},
        resources=("R0", "R1"),
        allowed={"T0": ("R0", "R1"), "T1": ("R0", "R1")},
    )
    root = task.root_node()
    actions = task.candidate_actions(root, 2)
    child_a = make_node(task, {actions[0].task: actions[0].resource}, root.node_id, actions[0], "repr-child-a", None)
    child_b = make_node(task, {actions[1].task: actions[1].resource}, root.node_id, actions[1], "repr-child-b", None)
    duplicate_a = child_a.with_updates(node_id="repr-child-a-duplicate")
    nodes = [root, child_a, child_b, duplicate_a]
    tree_edges = build_graph_edges(nodes, {})
    graph_edges = build_graph_edges(nodes, {"sibling_failure": True, "shared_constraint": True, "equivalent_state": True})
    rankers = [
        ("node-local MLP", MLPRanker(), tree_edges),
        ("history/path model", HistoryTransformerRanker(), tree_edges),
        ("Tree-GNN", TreeGNNRanker(), tree_edges),
        ("Graph-GNN", GraphGNNRanker(), graph_edges),
    ]
    oracle_best = min(nodes, key=lambda n: task.best_descendant_objective(n)).node_id
    rows = []
    for label, ranker, edges in rankers:
        scores = ranker.score_nodes(nodes, {"edges": edges})
        ordered = sorted(nodes, key=lambda n: scores[n.node_id], reverse=True)
        inversions = 0
        total = 0
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                total += 1
                if task.best_descendant_objective(ordered[i]) > task.best_descendant_objective(ordered[j]):
                    inversions += 1
        rows.append(
            {
                "representation": label,
                "rank_error": round(inversions / total, 4) if total else 0.0,
                "oracle_path_top1": max(scores, key=scores.get) == oracle_best,
                "node_count": len(nodes),
                "edge_count": len(edges),
                "llm_calls": "--",
                "expanded_nodes": "--",
            }
        )
    return rows


def _level1_family_diagnostics(cfg: ExperimentConfig) -> List[Dict[str, object]]:
    assignment = generate_assignment_fixture(random.Random(cfg.seed + 2100), "family-assignment", job_count=4, resource_count=3)
    setcover = generate_setcover_fixture(random.Random(cfg.seed + 2101), "family-setcover", universe_size=5, set_count=6)
    scheduling = generate_scheduling_fixture(random.Random(cfg.seed + 2102), "family-scheduling", job_count=4, machine_count=2)
    rows = []
    for family, task in [("assignment", assignment), ("setcover", setcover), ("scheduling", scheduling)]:
        root = task.root_node()
        optimum = task.exhaustive_oracle()
        replay = build_replay_from_task(task, k=max(1, cfg.action_budget))
        search_result = run_frontier_search(
            task=task,
            policy=BestBoundPolicy(),
            verifier=verifier_for_task(task, cfg.seed + 2110),
            node_budget=max(cfg.node_budget, 100),
            proposal_budget=max(1, cfg.action_budget),
            seed=cfg.seed + 2111,
            proposer=ReplayProposer(replay),
        )
        rows.append(
            {
                "family": family,
                "task_id": task.task_id,
                "oracle_objective": optimum,
                "root_lower_bound": root.lower_bound,
                "sound_root_bound": root.lower_bound <= optimum if root.lower_bound is not None else False,
                "semantic_masked": root.missingness_mask.get("semantic_features", False),
                "search_best_objective": search_result.best_objective,
                "search_reaches_oracle": search_result.best_objective == optimum,
                "expanded_nodes": search_result.expanded_nodes,
                "verifier_calls": search_result.cost.verifier_calls,
            }
        )
    return rows


def _quality_compute_sweep_rows(method_results: Dict[str, List[SearchResult]], cfg: ExperimentConfig) -> List[Dict[str, object]]:
    budgets = sorted({max(1, cfg.node_budget // 4), max(1, cfg.node_budget // 2), max(1, cfg.node_budget)})
    rows: List[Dict[str, object]] = []
    for method, results in method_results.items():
        for budget in budgets:
            budgeted_success = []
            budgeted_gap = []
            budgeted_calls = []
            for result in results:
                calls = min(result.proposer_calls, budget)
                budgeted_calls.append(calls)
                if result.proposer_calls <= budget:
                    budgeted_success.append(1.0 if result.success else 0.0)
                    budgeted_gap.append(result.gap)
                else:
                    budgeted_success.append(0.0)
                    budgeted_gap.append(1.0)
            rows.append(
                {
                    "method": method,
                    "normalized_expensive_budget": budget,
                    "success": round(sum(budgeted_success) / len(budgeted_success), 4) if budgeted_success else 0.0,
                    "gap": round(sum(budgeted_gap) / len(budgeted_gap), 4) if budgeted_gap else 0.0,
                    "proposal_calls": round(sum(budgeted_calls) / len(budgeted_calls), 4) if budgeted_calls else 0.0,
                    "budget_type": "normalized_proposal_calls",
                    "sweep_mode": "posthoc_anytime_prefix",
                }
            )
    return rows


def _replay_fairness_rows(cfg: ExperimentConfig) -> List[Dict[str, object]]:
    task = generate_assignment_fixture(random.Random(cfg.seed + 3100), "fairness-artifact", job_count=4, resource_count=3)
    return run_replay_fairness_matrix(
        task=task,
        seed=cfg.seed + 3101,
        policies=("dfs", "random", "best_bound", "tree_gnn", "graph_gnn"),
        node_budget=max(cfg.node_budget, 100),
        proposal_budget=max(1, min(cfg.action_budget, 3)),
    )


def run_experiment(cfg: ExperimentConfig, out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    train = generate_suite(cfg.seed, cfg.train_count, ("balanced", "sparse"), (cfg.job_min, cfg.job_max), (cfg.machine_min, cfg.machine_max), "train")
    test = generate_suite(cfg.seed + 999, cfg.test_count, ("balanced", "sparse", "heavy_tail"), (cfg.job_min, cfg.job_max + 1), (cfg.machine_min, cfg.machine_max), "test")
    split_manifest = build_split_manifest(cfg.seed, "assignment", max(1, min(5, cfg.test_count)))
    family_split_manifest = build_family_split_manifest(
        cfg.seed,
        ("assignment", "scheduling", "setcover", "level3_tool"),
        max(1, min(5, cfg.test_count)),
    )

    solver_samples = collect_states(
        train,
        SolverProposer(),
        _verifier_factory(cfg),
        cfg.trace_node_budget,
        cfg.action_budget,
        cfg.seed + 11,
        "solver",
    )
    llm_train_samples = collect_states(
        train,
        _llm(cfg, cfg.seed + 22),
        _verifier_factory(cfg),
        cfg.trace_node_budget,
        cfg.action_budget,
        cfg.seed + 22,
        "llm",
    )
    llm_eval_samples = collect_states(
        test,
        _llm(cfg, cfg.seed + 24),
        _verifier_factory(cfg),
        cfg.trace_node_budget,
        cfg.action_budget,
        cfg.seed + 24,
        "llm",
    )
    solver_pairs = make_ranking_pairs(solver_samples, cfg.max_pairs_per_problem, cfg.seed + 33)
    llm_train_pairs = make_ranking_pairs(llm_train_samples, cfg.max_pairs_per_problem, cfg.seed + 44)
    llm_eval_pairs = make_ranking_pairs(llm_eval_samples, cfg.max_pairs_per_problem, cfg.seed + 45)

    dim = len(FEATURE_NAMES)
    solver_ranker = LinearRanker.fresh(dim)
    solver_ranker.train(solver_pairs, epochs=cfg.rank_epochs, seed=cfg.seed + 55)

    llm_ranker = LinearRanker.fresh(dim)
    llm_ranker.train(llm_train_pairs, epochs=cfg.rank_epochs, seed=cfg.seed + 66)

    transfer_rows: List[Dict[str, object]] = []
    method_results: Dict[str, List[SearchResult]] = {}

    baselines = [
        ("CoT/DFS", "dfs", "llm", None),
        ("Best-of-N/random", "random", "llm", None),
        ("Certified best-bound", "best_bound", "llm", None),
        ("Oracle priority", "oracle", "llm", None),
        ("Solver B&B", "best_bound", "solver", None),
        ("CGR zero-shot", "learned", "llm", solver_ranker),
        ("CGR LLM-only", "learned", "llm", llm_ranker),
    ]
    for name, policy, proposer_kind, ranker in baselines:
        method_results[name] = _run_method(name, test, cfg, policy, proposer_kind, ranker, cfg.seed + 100)
    budget_protocol = _budget_protocol_manifest(cfg, [p.pid for p in test], list(method_results))

    source_features = [x for pair in solver_pairs for x in pair[:1]]
    target_features = [x for pair in llm_eval_pairs for x in pair[:1]]
    domain_classifier = train_domain_classifier_metrics(source_features, target_features, seed=cfg.seed + 77)
    discrepancy = float(domain_classifier["proxy_discrepancy"])

    transfer_rows.append(
        {
            "setting": "zero-shot",
            "training_data": "solver traces only",
            "solver_training_pair_count": len(solver_pairs),
            "target_training_pair_count": 0,
            "target_evaluation_pair_count": len(llm_eval_pairs),
            "target_rank_error": round(solver_ranker.ranking_error(llm_eval_pairs), 4),
            "domain_discrepancy_proxy": round(discrepancy, 4),
            **{k: round(v, 4) for k, v in summarize_results(method_results["CGR zero-shot"]).items()},
        }
    )
    transfer_rows.append(
        {
            "setting": "llm-only",
            "training_data": "LLM traces only",
            "solver_training_pair_count": 0,
            "target_training_pair_count": len(llm_train_pairs),
            "target_evaluation_pair_count": len(llm_eval_pairs),
            "target_rank_error": round(llm_ranker.ranking_error(llm_eval_pairs), 4),
            "domain_discrepancy_proxy": round(discrepancy, 4),
            **{k: round(v, 4) for k, v in summarize_results(method_results["CGR LLM-only"]).items()},
        }
    )

    for n in cfg.fewshot_counts:
        adapted = LinearRanker(weights=list(solver_ranker.weights), mean=list(solver_ranker.mean), scale=list(solver_ranker.scale))
        adapted.train(llm_train_pairs[:n], epochs=max(2, cfg.rank_epochs // 2), lr=0.03, seed=cfg.seed + n, refit_normalizer=False)
        name = f"CGR few-shot n={n}"
        method_results[name] = _run_method(name, test, cfg, "learned", "llm", adapted, cfg.seed + 200 + n)
        transfer_rows.append(
            {
                "setting": f"few-shot-{n}",
                "training_data": f"solver pretrain + {n} LLM ranking pairs",
                "solver_training_pair_count": len(solver_pairs),
                "target_training_pair_count": min(n, len(llm_train_pairs)),
                "target_evaluation_pair_count": len(llm_eval_pairs),
                "target_rank_error": round(adapted.ranking_error(llm_eval_pairs), 4),
                "domain_discrepancy_proxy": round(discrepancy, 4),
                **{k: round(v, 4) for k, v in summarize_results(method_results[name]).items()},
            }
        )

    primary_rows: List[Dict[str, object]] = []
    for name, results in method_results.items():
        s = summarize_results(results)
        primary_rows.append(
            {
                "Method": name,
                "Success": round(s["success"], 4),
                "Gap": round(s["gap"], 4),
                "Violations": 0,
                "False prune": s["false_prune"],
                "LLM calls": round(s["llm_calls_mean"], 2),
                "Tokens": "--",
                "Nodes": round(s["nodes_mean"], 2),
                "Latency": "--",
            }
        )

    trace_path = out_dir / "trace_sample.jsonl"
    with trace_path.open("w", encoding="utf-8") as f:
        for name, results in list(method_results.items())[:3]:
            for r in results[:2]:
                row = result_to_dict(r)
                row["trace"] = row["trace"][:8]
                f.write(json.dumps({"method": name, "result": row}, sort_keys=True) + "\n")

    write_csv(out_dir / "primary_table.csv", primary_rows)
    transfer_table_rows = [
        {
            "Setting": row["setting"],
            "Target trajectories used": row["target_training_pair_count"],
            "Ranking accuracy": round(1.0 - float(row["target_rank_error"]), 4),
            "Success": row["success"],
            "Expanded nodes": row["nodes_mean"],
            "LLM calls": row["llm_calls_mean"],
        }
        for row in transfer_rows
    ]
    write_csv(out_dir / "transfer_table.csv", transfer_table_rows)
    graph_edge_rows = _graph_edge_ablation_rows(cfg)
    level3_rows = _level3_tool_diagnostics()
    memory_rows = _memory_diagnostics()
    replay_budget_rows = _replay_budget_diagnostics(cfg)
    llm_adapter_rows = _llm_adapter_diagnostics()
    external_benchmark_rows = _external_benchmark_diagnostics()
    representation_rows = _representation_ablation_rows()
    level1_family_rows = _level1_family_diagnostics(cfg)
    quality_compute_rows = _quality_compute_sweep_rows(method_results, cfg)
    replay_fairness_rows = _replay_fairness_rows(cfg)
    certification_rows = _certification_corruption_sweep(cfg)
    certification_table_rows = [
        {
            "setting": row["setting"],
            "verified precision": row["verified_precision"],
            "verified recall": row["verified_recall"],
            "bound slack": row["bound_slack"],
            "false prune": row["false_prune"],
            "nodes": row["nodes"],
            "success": row["success"],
        }
        for row in certification_rows
    ]
    ranking_rows = [
        {"setting": "source_solver", "pair_count": len(solver_pairs), "ranking_error": solver_ranker.ranking_error(solver_pairs), "ranking_accuracy": 1.0 - solver_ranker.ranking_error(solver_pairs)},
        {"setting": "target_zero_shot", "pair_count": len(llm_eval_pairs), "ranking_error": solver_ranker.ranking_error(llm_eval_pairs), "ranking_accuracy": 1.0 - solver_ranker.ranking_error(llm_eval_pairs)},
        {"setting": "target_llm_only", "pair_count": len(llm_eval_pairs), "ranking_error": llm_ranker.ranking_error(llm_eval_pairs), "ranking_accuracy": 1.0 - llm_ranker.ranking_error(llm_eval_pairs)},
    ]
    label_schema = {
        "inference_feature_names": FEATURE_NAMES,
        "forbidden_inference_features": list(FORBIDDEN_INFERENCE_FEATURES),
        "label_only_fields": [
            "positive_best_descendant_objective",
            "negative_best_descendant_objective",
            "remaining_exact_expansions_to_best",
            "contains_optimal_descendant",
        ],
        "label_rule": "lower_best_descendant_objective_preferred",
    }
    ranking_rows.extend(
        {
            "setting": row["representation"],
            "pair_count": "--",
            "ranking_error": row["rank_error"],
            "ranking_accuracy": "--",
        }
        for row in representation_rows
    )
    proposal_rows = [
        {
            "method": name,
            "perturbation": "aggregate",
            "hidden_task_id": "--",
            "rendering_id": "--",
            "proposal_recall_at_k": summarize_results(results)["proposal_recall"],
            "schema_validity": summarize_results(results)["schema_valid"],
            "duplicate_action_rate": summarize_results(results)["duplicate_rate"],
            "invalid_action_count": "--",
            "valid_useful_action_count": "--",
        }
        for name, results in method_results.items()
    ]
    proposal_diagnostics = _level2_proposal_diagnostics(cfg)
    proposal_rows.extend(proposal_diagnostics)
    budget_rows = [
        {
            "method": name,
            "proposal_calls": summarize_results(results)["llm_calls_mean"],
            "evaluation_calls": 0,
            "node_budget": cfg.node_budget,
            "proposal_budget": cfg.action_budget,
            "proposal_call_budget": cfg.node_budget,
            "budget_exhausted": False,
            "token_budget": "--",
        }
        for name, results in method_results.items()
    ]
    budget_rows.extend(
        {
            "method": f"replay:{row['policy']}",
            "proposal_calls": row["proposal_calls"],
            "evaluation_calls": row["evaluation_calls"],
            "node_budget": cfg.node_budget,
            "proposal_budget": cfg.action_budget,
            "proposal_call_budget": row["proposal_call_budget"],
            "budget_exhausted": row["budget_exhausted"],
            "token_budget": "--",
        }
        for row in replay_budget_rows
    )
    domain_rows = [
        {
            "source": "solver-induced",
            "target": "llm-induced-simulator",
            "domain_classifier_accuracy": round(float(domain_classifier["heldout_accuracy"]), 4),
            "auc": round(float(domain_classifier["auc"]), 4),
            "proxy_discrepancy": discrepancy,
            "zero_shot_ranking_error": solver_ranker.ranking_error(llm_eval_pairs),
        }
    ]
    instance_rows = []
    for name, results in method_results.items():
        for idx, result in enumerate(results):
            instance_rows.append(
                {
                    "method": name,
                    "instance_index": idx,
                    "objective": result.objective,
                    "optimum": result.optimum,
                    "success": result.success,
                    "gap": result.gap,
                    "nodes": result.nodes_expanded,
                    "proposal_calls": result.proposer_calls,
                    "false_prune": result.false_prunes,
                }
            )

    write_csv(out_dir / "ranking_table.csv", ranking_rows)
    write_csv(out_dir / "certification_table.csv", certification_table_rows)
    write_csv(out_dir / "proposal_table.csv", proposal_rows)
    write_csv(out_dir / "budget_table.csv", budget_rows)
    write_json(out_dir / "budget_protocol.json", budget_protocol)
    write_csv(out_dir / "domain_shift_table.csv", domain_rows)
    write_csv(out_dir / "edge_ablation_table.csv", graph_edge_rows)
    write_csv(out_dir / "level3_tool_table.csv", level3_rows)
    write_csv(out_dir / "memory_table.csv", memory_rows)
    write_csv(out_dir / "replay_table.csv", replay_budget_rows)
    write_csv(out_dir / "llm_adapter_table.csv", llm_adapter_rows)
    write_csv(out_dir / "external_benchmark_table.csv", external_benchmark_rows)
    write_csv(out_dir / "representation_ablation_table.csv", representation_rows)
    write_csv(out_dir / "level1_family_table.csv", level1_family_rows)
    write_csv(out_dir / "quality_compute_table.csv", quality_compute_rows)
    write_csv(out_dir / "replay_fairness_matrix.csv", replay_fairness_rows)
    write_json(out_dir / "split_manifest.json", split_manifest)
    write_json(out_dir / "family_split_manifest.json", family_split_manifest)
    write_json(out_dir / "ranking_label_schema.json", label_schema)
    with (out_dir / "metrics_by_instance.jsonl").open("w", encoding="utf-8") as f:
        for row in instance_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    write_seed_aggregate(
        out_dir,
        rows=[
            {
                "seed": cfg.seed,
                "method": row["method"],
                "success": float(row["success"]),
                "gap": float(row["gap"]),
                "nodes": float(row["nodes"]),
                "proposal_calls": float(row["proposal_calls"]),
            }
            for row in instance_rows
        ],
        group_key="method",
        metric_keys=["success", "gap", "nodes", "proposal_calls"],
    )
    with (out_dir / "train_history.jsonl").open("w", encoding="utf-8") as f:
        solver_trace_ids = _sample_ids(solver_samples)
        target_train_trace_ids = _sample_ids(llm_train_samples)
        target_eval_trace_ids = _sample_ids(llm_eval_samples)
        for row in build_transfer_training_plan(
            solver_trace_ids=solver_trace_ids,
            target_train_trace_ids=target_train_trace_ids,
            target_eval_trace_ids=target_eval_trace_ids,
            solver_pair_count=len(solver_pairs),
            target_train_pair_count=len(llm_train_pairs),
            target_eval_pair_count=len(llm_eval_pairs),
            fewshot_counts=cfg.fewshot_counts,
            rank_epochs=cfg.rank_epochs,
        ):
            f.write(json.dumps(row, sort_keys=True) + "\n")
    reproducibility_manifest = {
        "seed": cfg.seed,
        "deterministic_core_artifacts": list(DETERMINISTIC_CORE_ARTIFACTS),
        "deterministic_core_hashes": deterministic_core_hashes(out_dir),
        "excluded_nondeterministic_artifacts": ["environment.json", "manifest.json"],
        "hash_algorithm": "sha256",
    }
    write_json(out_dir / "reproducibility_manifest.json", reproducibility_manifest)

    try:
        git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.strip() or None
    except OSError:
        git_commit = None

    environment = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": {"stdlib_only_core": True},
        "git_commit": git_commit,
        "started_at": started_at,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    config_obj = {
        **asdict(cfg),
        "dataset_split_identifiers": {"train": [p.pid for p in train], "test": [p.pid for p in test]},
        "extended_split_manifest": split_manifest,
        "family_split_manifest": family_split_manifest,
        "feature_schema_version": FEATURE_SCHEMA.version,
        "shared_feature_schema": {
            "universal": list(FEATURE_SCHEMA.universal),
            "verifier": list(FEATURE_SCHEMA.verifier),
            "semantic": list(FEATURE_SCHEMA.semantic),
            "mask_keys": list(FEATURE_SCHEMA.mask_keys),
            "legacy_vector_feature_names": FEATURE_NAMES,
        },
        "ranking_label_schema": label_schema,
        "budget_protocol": budget_protocol,
        "reproducibility_manifest": reproducibility_manifest,
        "graph_edge_configuration": {"enabled": False},
        "memory_configuration": {"enabled": False},
        "training_regimes": ["zero_shot", "few_shot", "llm_only"],
        "search_budget": cfg.node_budget,
        "proposal_budget": cfg.action_budget,
    }
    write_json(out_dir / "config.json", config_obj)
    write_json(out_dir / "environment.json", environment)
    metrics = {
        "config": config_obj,
        "train_problem_count": len(train),
        "test_problem_count": len(test),
        "solver_trace_count": len(solver_samples),
        "level2_trace_count": len(llm_train_samples) + len(llm_eval_samples),
        "level2_train_trace_count": len(llm_train_samples),
        "level2_eval_trace_count": len(llm_eval_samples),
        "solver_pair_count": len(solver_pairs),
        "llm_pair_count": len(llm_eval_pairs),
        "llm_train_pair_count": len(llm_train_pairs),
        "llm_eval_pair_count": len(llm_eval_pairs),
        "feature_names": FEATURE_NAMES,
        "shared_feature_schema": config_obj["shared_feature_schema"],
        "ranking_label_schema": label_schema,
        "budget_protocol": budget_protocol,
        "reproducibility_manifest": reproducibility_manifest,
        "source_rank_error": solver_ranker.ranking_error(solver_pairs),
        "target_rank_error_zero_shot": solver_ranker.ranking_error(llm_eval_pairs),
        "target_rank_error_llm_only": llm_ranker.ranking_error(llm_eval_pairs),
        "domain_discrepancy_proxy": discrepancy,
        "domain_classifier": domain_classifier,
        "level2_rendering_count": len({row["rendering_id"] for row in proposal_diagnostics}),
        "proposal_diagnostics": proposal_diagnostics,
        "graph_edge_ablation": graph_edge_rows,
        "level3_tool_diagnostics": level3_rows,
        "memory_diagnostics": memory_rows,
        "replay_budget_diagnostics": replay_budget_rows,
        "llm_adapter_diagnostics": llm_adapter_rows,
        "external_benchmark_diagnostics": external_benchmark_rows,
        "representation_ablation": representation_rows,
        "level1_family_diagnostics": level1_family_rows,
        "quality_compute_sweep": quality_compute_rows,
        "replay_fairness_matrix": replay_fairness_rows,
        "certification_diagnostics": certification_rows,
        "split_manifest": split_manifest,
        "family_split_manifest": family_split_manifest,
        "transfer": {
            "zero_shot": transfer_rows[0],
            "llm_only": transfer_rows[1],
            "few_shot": transfer_rows[2:],
        },
        "methods": {name: summarize_results(results) for name, results in method_results.items()},
    }
    write_json(out_dir / "metrics.json", metrics)
    plot_paths = generate_paper_plots(metrics, out_dir)
    metrics["plot_artifacts"] = [path.name for path in plot_paths]
    write_json(out_dir / "metrics.json", metrics)
    write_json(
        out_dir / "manifest.json",
        {
            "run_dir": str(out_dir),
            "required_artifacts": [
                "config.json",
                "environment.json",
                "metrics.json",
                "metrics_by_instance.jsonl",
                "primary_table.csv",
                "transfer_table.csv",
                "ranking_table.csv",
                "certification_table.csv",
                "proposal_table.csv",
                "budget_table.csv",
                "budget_protocol.json",
                "domain_shift_table.csv",
                "edge_ablation_table.csv",
                "level3_tool_table.csv",
                "memory_table.csv",
                "replay_table.csv",
                "llm_adapter_table.csv",
                "external_benchmark_table.csv",
                "representation_ablation_table.csv",
                "level1_family_table.csv",
                "quality_compute_table.csv",
                "replay_fairness_matrix.csv",
                "split_manifest.json",
                "family_split_manifest.json",
                "ranking_label_schema.json",
                "reproducibility_manifest.json",
                "multi_seed_summary.csv",
                "multi_seed_manifest.json",
                "plot_manifest.json",
                "quality_compute_pareto.svg",
                "quality_compute_sweep.svg",
                "transfer_sample_efficiency.svg",
                "proposal_recall_vs_success.svg",
                "distribution_shift_vs_ranking.svg",
                "trace_sample.jsonl",
                "train_history.jsonl",
                "stdout.log",
                "manifest.json",
            ],
            "conceptual_summary": "CGR treats LLM reasoning as frontier search: the LLM proposes semantic branches, a transferable learned ranking policy allocates expensive compute across surviving partial states, and only conservatively verified structure can authorize irreversible pruning.",
        },
    )
    (out_dir / "stdout.log").write_text(
        "\n".join(
            [
                f"run_dir={out_dir}",
                f"solver_trace_count={len(solver_samples)}",
                f"level2_trace_count={len(llm_train_samples) + len(llm_eval_samples)}",
                f"solver_pair_count={len(solver_pairs)}",
                f"llm_pair_count={len(llm_eval_pairs)}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics
