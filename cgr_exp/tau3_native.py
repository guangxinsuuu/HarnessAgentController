from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from cgr_exp.q4_tau_source_evidence import TAU2_BENCH_SOURCE_URL, q4_tau_source_evidence

TAU2_SOURCE_REVISION = "a2c024725189473d2d7cea3a5cfdbcc67478e41f"
TAU2_SUBMODULE = Path("third_party/tau2-bench")
TABLE10_METHODS = {
    "native": "Native tool-calling rerun",
    "best_of_n": "Best-of-N",
    "solver": "Solver-derived controller",
}

TRACE_AUDIT_COLUMNS = [
    "trace_path",
    "event_count",
    "canonical_ranker_events",
    "heuristic_events",
    "ranker_checkpoint_loaded_events",
    "certified_filter_enabled_events",
    "certified_pruned_candidates",
    "certified_prune_rate",
    "mean_candidate_count",
    "mean_scored_candidate_count",
    "oracle_visible_to_controller",
    "oracle_used_for_candidate_generation",
    "oracle_used_for_filtering",
    "tool_names_used_as_features",
    "status",
]


@dataclass(frozen=True)
class Tau3RunConfig:
    output_dir: Path
    domain: str
    method: str = "native"
    task_split_name: str = "base"
    num_tasks: Optional[int] = None
    task_ids: Sequence[str] = ()
    num_trials: int = 4
    agent_llm: str = "openai/gpt-4o-mini"
    user_llm: str = "openai/gpt-4o-mini"
    agent_llm_args: str = '{"temperature": 0}'
    user_llm_args: str = '{"temperature": 0}'
    max_steps: int = 200
    max_errors: int = 10
    timeout: Optional[float] = None
    max_concurrency: int = 3
    workers: int = 0
    seed: int = 0
    python_bin: str = sys.executable
    auto_resume: bool = True
    verbose_logs: bool = False
    dry_run: bool = False
    force: bool = False
    solver_ranker_checkpoint: Optional[Path] = None
    enable_certified_filter: bool = False
    candidate_workers: int = 1
    candidate_retries: int = 2
    candidate_retry_backoff: float = 0.5


def run_tau3_native(config: Tau3RunConfig) -> Dict[str, object]:
    if config.method not in TABLE10_METHODS:
        raise ValueError(f"unsupported tau3 method: {config.method}")
    if config.method == "best_of_n":
        raise RuntimeError(
            f"{TABLE10_METHODS[config.method]} is not implemented as a native tau2-bench agent yet; "
            "run native first, then attach a benchmark-compatible selector/controller."
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    preflight = tau3_preflight(config.python_bin)
    (config.output_dir / "tau3_preflight.json").write_text(json.dumps(preflight, indent=2, sort_keys=True), encoding="utf-8")

    official_results_dir = (config.output_dir / "official_tau2_results").resolve()
    controller_trace_path = (config.output_dir / "tau3_controller_trace.jsonl").resolve() if config.method == "solver" else None
    save_to = str(official_results_dir)
    command = build_tau2_run_command(config, save_to=save_to)
    method_label = _tau3_method_label(config)
    command_manifest = {
        "command": command,
        "cwd": str(TAU2_SUBMODULE),
        "dry_run": config.dry_run,
        "method": method_label,
        "domain": config.domain,
        "task_split_name": config.task_split_name,
        "num_tasks": config.num_tasks,
        "task_ids": list(config.task_ids),
        "num_trials": config.num_trials,
        "max_concurrency": config.max_concurrency,
        "workers": config.workers,
        "solver_ranker_checkpoint": None if config.solver_ranker_checkpoint is None else str(config.solver_ranker_checkpoint),
        "candidate_workers": config.candidate_workers,
        "candidate_retries": config.candidate_retries,
        "candidate_retry_backoff": config.candidate_retry_backoff,
        "source_controller_protocol": _source_controller_protocol(config.solver_ranker_checkpoint),
        "enable_certified_filter": config.enable_certified_filter,
        "controller_trace_path": None if controller_trace_path is None else str(controller_trace_path),
        "resume_safe": config.auto_resume and not config.force,
        "force_rerun": config.force,
        "overwrite_is_output_dir_guard_only": True,
        "resume_note": "--overwrite only allows writing into an existing output directory; --force is required to ignore existing tau2 results and rerun from scratch.",
        "official_source": str(TAU2_SUBMODULE),
        "source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": _tau2_revision(),
        "preflight": preflight,
    }
    (config.output_dir / "tau3_native_command.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")

    if config.dry_run:
        return _write_pending_artifacts(config, "dry_run", save_to)
    if not preflight["ready"]:
        return _write_pending_artifacts(config, "preflight_failed: " + "; ".join(preflight["blockers"]), save_to)
    if Path(save_to).exists() and not config.force:
        existing = aggregate_tau3_official_results(config.output_dir, method=method_label, domain=config.domain, expected_trials=config.num_trials)
        expected_tasks = _expected_tau3_task_count(config)
        existing_tasks = int(_as_float(existing["row"].get("task_count")) if not math.isnan(_as_float(existing["row"].get("task_count"))) else 0)
        if expected_tasks is None or existing_tasks >= expected_tasks:
            return existing
        command_manifest["resume_from_partial_results"] = True
        command_manifest["existing_task_count"] = existing_tasks
        command_manifest["expected_task_count"] = expected_tasks
        (config.output_dir / "tau3_native_command.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")

    start = time.monotonic()
    if config.method == "solver":
        result = _run_tau2_solver_controller(config, Path(save_to))
    else:
        result = subprocess.run(command, cwd=TAU2_SUBMODULE, env=_tau2_env(), text=True, capture_output=True, check=False)
    elapsed = time.monotonic() - start
    (config.output_dir / "tau3_native_stdout.log").write_text(result.stdout, encoding="utf-8")
    (config.output_dir / "tau3_native_stderr.log").write_text(result.stderr, encoding="utf-8")
    command_manifest["returncode"] = result.returncode
    command_manifest["subprocess_wall_clock_seconds"] = elapsed
    (config.output_dir / "tau3_native_command.json").write_text(json.dumps(command_manifest, indent=2, sort_keys=True), encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"tau2 official run failed with exit code {result.returncode}; see {config.output_dir / 'tau3_native_stderr.log'}")
    return aggregate_tau3_official_results(config.output_dir, method=method_label, domain=config.domain, expected_trials=config.num_trials)


def _run_tau2_solver_controller(config: Tau3RunConfig, save_path: Path) -> subprocess.CompletedProcess[str]:
    try:
        if "OPENAI_API_KEY" not in os.environ and "CGR_LLM_API_KEY" in os.environ:
            os.environ["OPENAI_API_KEY"] = os.environ["CGR_LLM_API_KEY"]
        from tau2.data_model.simulation import TextRunConfig
        from tau2.registry import registry
        from tau2.runner import run_domain

        from cgr_exp.tau3_solver_agent import create_tau3_solver_controller_agent

        if registry.get_agent_factory("cgr_solver_controller") is None:
            registry.register_agent_factory(create_tau3_solver_controller_agent, "cgr_solver_controller")
        llm_args = json.loads(config.agent_llm_args)
        llm_args.setdefault("cgr_candidate_count", 4)
        llm_args.setdefault("cgr_candidate_workers", config.candidate_workers)
        llm_args.setdefault("cgr_candidate_retries", config.candidate_retries)
        llm_args.setdefault("cgr_candidate_retry_backoff", config.candidate_retry_backoff)
        if config.solver_ranker_checkpoint is not None:
            llm_args["cgr_ranker_checkpoint"] = str(config.solver_ranker_checkpoint)
        if config.enable_certified_filter:
            llm_args["cgr_enable_certified_filter"] = True
        llm_args["cgr_controller_trace_path"] = str(config.output_dir / "tau3_controller_trace.jsonl")
        llm_args.setdefault("cgr_max_turns", config.max_steps)
        llm_args.setdefault("cgr_max_tool_budget", config.max_steps)
        run_config = TextRunConfig(
            domain=config.domain,
            agent="cgr_solver_controller",
            user="user_simulator",
            task_set_name=None,
            task_split_name=config.task_split_name,
            task_ids=list(config.task_ids) or None,
            num_tasks=config.num_tasks,
            llm_agent=config.agent_llm,
            llm_args_agent=llm_args,
            llm_user=config.user_llm,
            llm_args_user=json.loads(config.user_llm_args),
            num_trials=config.num_trials,
            max_steps=config.max_steps,
            max_errors=config.max_errors,
            timeout=config.timeout,
            save_to=str(save_path),
            max_concurrency=config.max_concurrency,
            workers=config.workers,
            seed=config.seed,
            log_level="ERROR",
            enforce_communication_protocol=True,
            verbose_logs=config.verbose_logs,
            auto_resume=config.auto_resume,
        )
        results = run_domain(run_config)
        if not save_path.exists():
            results.save(save_path, format="json")
        return subprocess.CompletedProcess(args=["tau2-python-api", "cgr_solver_controller"], returncode=0, stdout="completed cgr_solver_controller run\n", stderr="")
    except BaseException as exc:
        return subprocess.CompletedProcess(args=["tau2-python-api", "cgr_solver_controller"], returncode=1, stdout="", stderr=f"{type(exc).__name__}: {exc}\n")


def build_tau2_run_command(config: Tau3RunConfig, *, save_to: str) -> List[str]:
    agent = "llm_agent" if config.method != "solver" else "cgr_solver_controller"
    save_to_path = Path(save_to)
    if save_to_path.name == "results.json":
        save_to = str(save_to_path.parent)
    cmd = [
        config.python_bin,
        "-m",
        "tau2.cli",
        "run",
        "--domain",
        config.domain,
        "--agent",
        agent,
        "--user",
        "user_simulator",
        "--agent-llm",
        config.agent_llm,
        "--user-llm",
        config.user_llm,
        "--agent-llm-args",
        config.agent_llm_args,
        "--user-llm-args",
        config.user_llm_args,
        "--task-split-name",
        config.task_split_name,
        "--num-trials",
        str(config.num_trials),
        "--max-steps",
        str(config.max_steps),
        "--max-errors",
        str(config.max_errors),
        "--max-concurrency",
        str(config.max_concurrency),
        "--workers",
        str(config.workers),
        "--seed",
        str(config.seed),
        "--save-to",
        save_to,
        "--log-level",
        "ERROR",
        "--enforce-communication-protocol",
    ]
    if config.num_tasks is not None:
        cmd.extend(["--num-tasks", str(config.num_tasks)])
    if config.task_ids:
        cmd.append("--task-ids")
        cmd.extend(str(task_id) for task_id in config.task_ids)
    if config.timeout is not None:
        cmd.extend(["--timeout", str(config.timeout)])
    if config.auto_resume:
        cmd.append("--auto-resume")
    if config.verbose_logs:
        cmd.extend(["--verbose-logs", "--llm-log-mode", "all"])
    return cmd


def aggregate_tau3_official_results(
    out_dir: Path,
    *,
    method: str = "Native tool-calling rerun",
    domain: Optional[str] = None,
    expected_trials: int = 4,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = _find_official_results(out_dir)
    if result_path is None:
        return _write_pending_row(out_dir, method=method, domain=domain or "", status="official results missing")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    sims = _load_simulations(result_path, data)
    tasks = data.get("tasks", [])
    task_ids = sorted({str(sim.get("task_id", "")) for sim in sims if str(sim.get("task_id", ""))})
    rewards_by_task: Dict[str, List[float]] = {task_id: [] for task_id in task_ids}
    rows = []
    total_tool_calls = 0
    total_tokens = 0
    token_observed = False
    total_duration = 0.0
    total_llm_calls = 0
    violations = 0
    for sim in sims:
        task_id = str(sim.get("task_id", ""))
        reward = _reward(sim)
        if task_id:
            rewards_by_task.setdefault(task_id, []).append(reward)
        messages = sim.get("messages") or []
        tool_calls = _count_agent_tool_calls(messages)
        tokens, calls = _token_usage(messages)
        if tokens is not None:
            total_tokens += tokens
            token_observed = True
        total_llm_calls += calls
        total_tool_calls += tool_calls
        duration = _as_float(sim.get("duration"))
        if not math.isnan(duration):
            total_duration += duration
        term = str(sim.get("termination_reason", ""))
        if term not in {"user_stop", "agent_stop"} and reward < 1.0:
            violations += 1
        rows.append(
            {
                "Domain": domain or _domain_from_results(data),
                "Method": method,
                "task_id": task_id,
                "rollout_id": sim.get("id", ""),
                "trial": sim.get("trial", ""),
                "reward": reward,
                "success": 1 if reward >= 1.0 - 1e-6 else 0,
                "rollout_count": len(rewards_by_task.get(task_id, [])),
                "tool_calls": tool_calls,
                "tokens": "" if tokens is None else tokens,
                "llm_calls": calls,
                "duration_seconds": "" if math.isnan(duration) else duration,
                "termination_reason": term,
            }
        )
    for row in rows:
        task_id = str(row["task_id"])
        rewards = rewards_by_task.get(task_id, [])
        terms = [str(sim.get("termination_reason", "")) for sim in sims if str(sim.get("task_id", "")) == task_id]
        task_success_count = sum(1 for value in rewards if value >= 1.0 - 1e-6)
        row["rollout_count"] = len(rewards)
        row["task_success_count"] = task_success_count
        row["task_failure_count"] = max(0, len(rewards) - task_success_count)
        row["task_best_reward"] = max(rewards) if rewards else ""
        row["task_pass^1"] = _pass_hat({task_id: rewards}, 1) if rewards else ""
        row["task_pass^4"] = _pass_hat({task_id: rewards}, 4) if len(rewards) >= 4 else ""
        row["task_termination_reasons"] = json.dumps(_count_values(terms), sort_keys=True)
        row["task_error_count"] = sum(1 for term in terms if term not in {"user_stop", "agent_stop"})

    pass1 = _pass_hat(rewards_by_task, 1)
    pass4 = _pass_hat(rewards_by_task, 4) if expected_trials >= 4 else float("nan")
    avg_rollouts = sum(len(v) for v in rewards_by_task.values()) / max(len(rewards_by_task), 1)
    avg_tool_calls = total_tool_calls / max(len(sims), 1)
    avg_tokens = (total_tokens / max(len(sims), 1)) if token_observed else float("nan")
    avg_llm_calls = total_llm_calls / max(len(sims), 1)
    avg_wall = total_duration / max(len(sims), 1)
    aggregate_row = {
        "Domain": domain or _domain_from_results(data),
        "Method": method,
        "pass^1": pass1,
        "pass^4": pass4,
        "violations": violations / max(len(sims), 1),
        "tool_calls": avg_tool_calls,
        "llm_calls": avg_llm_calls,
        "tokens": "" if math.isnan(avg_tokens) else avg_tokens,
        "wall_clock_seconds": avg_wall,
        "status": "complete" if task_ids else "pending",
        "source_url": TAU2_BENCH_SOURCE_URL,
        "preferred_current_source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": _tau2_revision(),
        "official_results": str(result_path),
        "task_count": len(task_ids),
        "simulation_count": len(sims),
        "avg_rollouts_per_task": avg_rollouts,
        "notes": "official tau2-bench text half-duplex run; pass^k uses tau2 pass_hat_k formula",
    }
    _write_csv(out_dir / "q4_tau_per_instance.csv", rows)
    _write_csv(out_dir / "tau3_native_rollouts.csv", rows)
    (out_dir / "q4_tau_row.json").write_text(json.dumps(aggregate_row, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "q4_tau_table.csv", [aggregate_row])
    manifest = _tau_manifest(method, aggregate_row["Domain"], len(task_ids), expected_trials, result_path)
    (out_dir / "q4_tau_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"row": aggregate_row, "manifest": manifest, "produced_artifacts": ["q4_tau_row.json", "q4_tau_table.csv", "q4_tau_per_instance.csv", "q4_tau_manifest.json"]}


def tau3_preflight(python_bin: str = sys.executable) -> Dict[str, object]:
    blockers = []
    if not TAU2_SUBMODULE.exists():
        blockers.append(f"missing official source submodule: {TAU2_SUBMODULE}")
    revision = _tau2_revision()
    if revision != TAU2_SOURCE_REVISION:
        blockers.append(f"tau2-bench revision mismatch: {revision or 'missing'}")
    version = _python_version(python_bin)
    if not _python_version_supported(version):
        blockers.append("selected tau Python must be >=3.12,<3.14; set --tau-python-bin to a 3.12/3.13 interpreter")
    dependency = subprocess.run(
        [python_bin, "-c", "import loguru, pandas, litellm, rich, pydantic; import tau2.cli"],
        env=_tau2_env(),
        text=True,
        capture_output=True,
        check=False,
    )
    if dependency.returncode != 0:
        blockers.append("tau2 Python dependencies missing for selected interpreter")
    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("CGR_LLM_API_KEY")):
        blockers.append("OPENAI_API_KEY missing; tau2/litellm reads OPENAI_API_KEY")
    return {
        "ready": not blockers,
        "blockers": blockers,
        "python_bin": python_bin,
        "python_version": version,
        "uv_available": shutil.which("uv") is not None,
        "tau2_source": str(TAU2_SUBMODULE),
        "source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": revision,
        "install_hint": "cd third_party/tau2-bench && uv sync, or install pyproject dependencies into a Python >=3.12,<3.14 environment",
    }


def _tau2_env() -> Dict[str, str]:
    env = dict(os.environ)
    src = str((TAU2_SUBMODULE / "src").resolve())
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else src + os.pathsep + existing
    if "OPENAI_API_KEY" not in env and "CGR_LLM_API_KEY" in env:
        env["OPENAI_API_KEY"] = env["CGR_LLM_API_KEY"]
    return env


def _expected_tau3_task_count(config: Tau3RunConfig) -> Optional[int]:
    if config.task_ids:
        return len(config.task_ids)
    return config.num_tasks


def _source_controller_protocol(checkpoint: Optional[Path]) -> Dict[str, object]:
    if checkpoint is None:
        return {"manifest_found": False}
    manifest_path = checkpoint.parent / "canonical_harness_source_validation.json"
    protocol: Dict[str, object] = {
        "checkpoint": str(checkpoint),
        "manifest": str(manifest_path),
        "manifest_found": manifest_path.exists() and manifest_path.is_file(),
    }
    if not protocol["manifest_found"]:
        return protocol
    data = _read_json_or_empty(manifest_path)
    source_generator = data.get("source_generator")
    if not isinstance(source_generator, Mapping):
        source_generator = {}
    tool_identity = data.get("tool_identity_generalization")
    if not isinstance(tool_identity, Mapping):
        tool_identity = {}
    contract_coverage = data.get("contract_coverage_stress")
    if not isinstance(contract_coverage, Mapping):
        contract_coverage = {}
    protocol.update(
        {
            "benchmark": data.get("benchmark", ""),
            "controller_checkpoint_frozen": _as_bool(data.get("controller_checkpoint_frozen")),
            "adapter_changes_checkpoint_stays_frozen": _as_bool(data.get("adapter_changes_checkpoint_stays_frozen")),
            "tau3_train_trajectories_used": _as_bool(data.get("tau3_train_trajectories_used")),
            "oracle_labels_in_inference_features": _as_bool(data.get("oracle_labels_in_inference_features")),
            "learned_score_authorizes_pruning": _as_bool(data.get("learned_score_authorizes_pruning")),
            "tool_names_used_as_features": _as_bool(data.get("tool_names_used_as_features")),
            "source_generator_distribution_fixed_before_target_test_evaluation": _as_bool(
                source_generator.get("source_generator_distribution_fixed_before_target_test_evaluation")
            ),
            "tau3_heldout_test_seen_before_freeze": _as_bool(source_generator.get("tau3_heldout_test_seen_before_freeze")),
            "source_generator_uses_tau3_trajectories": _as_bool(source_generator.get("tau3_trajectories_used")),
            "source_generator_anonymized_tool_names": _as_bool(source_generator.get("anonymized_tool_names")),
            "tool_identity_generalization_ready": _as_bool(tool_identity.get("ready")),
            "tool_identity_generalization_top1_delta": tool_identity.get("top1_absolute_delta", ""),
            "contract_coverage_stress_ready": _as_bool(contract_coverage.get("ready")),
            "contract_coverage_black_box_top1_delta_from_full": contract_coverage.get("black_box_contract_top1_delta_from_full", ""),
            "train_count": data.get("train_count", ""),
            "validation_count": data.get("validation_count", ""),
            "seed": data.get("seed", ""),
        }
    )
    return protocol


def _source_controller_protocol_from_manifest(command_manifest: Mapping[str, object]) -> Dict[str, object]:
    protocol = command_manifest.get("source_controller_protocol")
    if isinstance(protocol, Mapping):
        return dict(protocol)
    checkpoint = command_manifest.get("solver_ranker_checkpoint")
    if checkpoint:
        return _source_controller_protocol(Path(str(checkpoint)))
    return {"manifest_found": False}


def _python_version(python_bin: str) -> str:
    result = subprocess.run(
        [python_bin, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _python_version_supported(version: str) -> bool:
    try:
        major, minor, _patch = (int(part) for part in version.split(".", 2))
    except ValueError:
        return False
    return major == 3 and 12 <= minor < 14


def _tau3_method_label(config: Tau3RunConfig) -> str:
    if config.method != "solver":
        return TABLE10_METHODS[config.method]
    if config.solver_ranker_checkpoint is None:
        return TABLE10_METHODS[config.method]
    if config.enable_certified_filter:
        return "Solver-derived canonical controller + certified filter"
    return "Solver-derived canonical controller"


def build_table10_tau3_results(out_dir: Path, row_dirs: Sequence[Path], expected_test_count: int = 100) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for row_dir in row_dirs:
        row_path = row_dir / "q4_tau_row.json"
        if not row_path.exists():
            domain, method = _table10_expected_label(row_dir)
            rows.append(_table10_row({"Domain": domain, "Method": method, "status": "missing: q4_tau_row.json", "official_results": str(row_dir)}, expected_test_count))
            continue
        row = json.loads(row_path.read_text(encoding="utf-8"))
        rows.append(_table10_row(row, expected_test_count))
    csv_path = out_dir / "table10_tau3_results.csv"
    tex_path = out_dir / "table10_tau3_results.tex"
    _write_csv(csv_path, rows)
    tex_path.write_text(_table10_latex(rows), encoding="utf-8")
    manifest = {
        "table": "Table 10 tau3 matched maintained-protocol evaluation",
        "benchmark": "tau3-bench/tau2-bench",
        "source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": _tau2_revision(),
        "expected_test_count": expected_test_count,
        "row_dirs": [str(path) for path in row_dirs],
        "rows": rows,
        "produced_artifacts": [csv_path.name, tex_path.name],
    }
    (out_dir / "table10_tau3_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def audit_table10_tau3_readiness(out_dir: Path, row_dirs: Sequence[Path], expected_test_count: int = 100) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    for row_dir in row_dirs:
        row_path = row_dir / "q4_tau_row.json"
        blockers = []
        if not row_path.exists():
            audit_rows.append(_table10_audit_row(row_dir, {}, ["q4_tau_row.json missing"], expected_test_count))
            continue
        row = json.loads(row_path.read_text(encoding="utf-8"))
        task_count = int(_as_float(row.get("task_count")) if not math.isnan(_as_float(row.get("task_count"))) else 0)
        status = str(row.get("status", ""))
        method = str(row.get("Method", ""))
        if status != "complete":
            blockers.append(status or "status missing")
        if task_count < expected_test_count:
            blockers.append(f"partial: {task_count}/{expected_test_count}")
        if status == "complete":
            blockers.extend(_table10_missing_compute_blockers(row))
        if method == "Best-of-N":
            if row.get("selector") != "official_reward_best_of_n":
                blockers.append("Best-of-N selector evidence missing")
            if not _as_bool(row.get("selector_full_coverage")):
                blockers.append("Best-of-N rollout coverage incomplete")
        command_manifest = _read_json_or_empty(row_dir / "tau3_native_command.json")
        source_protocol: Dict[str, object] = {}
        trace_summary: Dict[str, object] = {}
        if "canonical controller" in method:
            if not command_manifest.get("solver_ranker_checkpoint"):
                blockers.append("solver_ranker_checkpoint missing")
            source_protocol = _source_controller_protocol_from_manifest(command_manifest)
            if not source_protocol.get("manifest_found"):
                blockers.append("source controller manifest missing")
            if source_protocol.get("tau3_train_trajectories_used"):
                blockers.append("source controller used tau3 train trajectories")
            if source_protocol.get("oracle_labels_in_inference_features"):
                blockers.append("source controller leaks oracle labels")
            if source_protocol.get("tool_names_used_as_features"):
                blockers.append("source controller uses tool names as features")
            if not source_protocol.get("controller_checkpoint_frozen"):
                blockers.append("source controller checkpoint freeze evidence missing")
            if source_protocol.get("source_generator_uses_tau3_trajectories"):
                blockers.append("source generator used tau3 trajectories")
            if source_protocol.get("tau3_heldout_test_seen_before_freeze"):
                blockers.append("tau3 heldout test influenced source generator freeze")
            if not source_protocol.get("source_generator_distribution_fixed_before_target_test_evaluation"):
                blockers.append("source generator freeze evidence missing")
            if not source_protocol.get("source_generator_anonymized_tool_names"):
                blockers.append("source generator tool-name anonymization evidence missing")
            if not source_protocol.get("tool_identity_generalization_ready"):
                blockers.append("tool-identity generalization stress missing or failed")
            if not source_protocol.get("contract_coverage_stress_ready"):
                blockers.append("contract coverage stress missing or failed")
            trace_summary = audit_tau3_controller_trace(out_dir / f"{row_dir.name}_trace_audit", row_dir / "tau3_controller_trace.jsonl")
            if trace_summary["event_count"] <= 0:
                blockers.append("controller trace missing")
            if trace_summary["canonical_ranker_events"] <= 0:
                blockers.append("canonical ranker events missing")
            if trace_summary["oracle_visible_to_controller"]:
                blockers.append("oracle visible to controller")
            if trace_summary["oracle_used_for_candidate_generation"]:
                blockers.append("oracle used for candidate generation")
            if trace_summary["oracle_used_for_filtering"]:
                blockers.append("oracle used for filtering")
            if trace_summary["tool_names_used_as_features"]:
                blockers.append("tool names used as features")
            if "certified filter" in method and trace_summary["certified_filter_enabled_events"] <= 0:
                blockers.append("certified filter events missing")
        audit_rows.append(_table10_audit_row(row_dir, row, blockers, expected_test_count, source_protocol=source_protocol, trace_summary=trace_summary))
    _write_csv(out_dir / "table10_tau3_readiness.csv", audit_rows)
    readiness = {
        "table": "Table 10 tau3 maintained-protocol evaluation",
        "expected_test_count": expected_test_count,
        "all_rows_result_ready": all(row["result_ready"] for row in audit_rows),
        "pending_or_partial_rows": [_table10_row_id(row) for row in audit_rows if not row["result_ready"]],
        "audit_rows": audit_rows,
        "produced_artifacts": ["table10_tau3_readiness.csv", "table10_tau3_readiness.json"],
    }
    (out_dir / "table10_tau3_readiness.json").write_text(json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8")
    return readiness


def build_table10_tau3_run_preflight(
    out_dir: Path,
    *,
    base_dir: Path = Path("runs/table10_tau3_canonical"),
    expected_test_count: int = 100,
    best_of_n: int = 4,
    num_trials: int = 4,
    seed: int = 0,
    max_concurrency: int = 6,
    candidate_workers: int = 1,
    candidate_retries: int = 2,
    candidate_retry_backoff: float = 0.5,
    agent_llm: str = "openai/gpt-4o-mini",
    user_llm: str = "openai/gpt-4o-mini",
    python_bin: str = "python3",
    tau_python_bin: str = sys.executable,
    parallel_domains: bool = True,
    parallel_method_groups: bool = False,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    row_dirs = _table10_default_row_dirs(base_dir, best_of_n)
    readiness = audit_table10_tau3_readiness(out_dir / "readiness", row_dirs, expected_test_count=expected_test_count)
    rows = [_table10_source_controller_preflight_row(base_dir, seed=seed)]
    for group, command, required_dirs in _table10_run_groups(
        base_dir=base_dir,
        expected_test_count=expected_test_count,
        best_of_n=best_of_n,
        num_trials=num_trials,
        seed=seed,
        max_concurrency=max_concurrency,
        candidate_workers=candidate_workers,
        candidate_retries=candidate_retries,
        candidate_retry_backoff=candidate_retry_backoff,
        agent_llm=agent_llm,
        user_llm=user_llm,
        python_bin=python_bin,
        tau_python_bin=tau_python_bin,
        parallel_domains=parallel_domains,
        parallel_method_groups=parallel_method_groups,
    ):
        group_methods = _methods_for_dirs(readiness["audit_rows"], required_dirs)
        group_ready = bool(group_methods) and all(bool(row["result_ready"]) for row in group_methods)
        effective_concurrency = _table10_effective_concurrency(group, max_concurrency, candidate_workers, parallel_domains, parallel_method_groups)
        rows.append(
            {
                "run_group": group,
                "ready": group_ready,
                "methods": "; ".join(_table10_row_id(row) for row in group_methods) if group_methods else "not started",
                "remaining_rows": sum(1 for row in group_methods if not bool(row["result_ready"])) if group_methods else len(required_dirs),
                "api_required": group in {"full", "native_full", "legacy_solver_full", "solver_canonical_full", "solver_canonical_certified_full"},
                "resume_safe": group in {"full", "native_full", "legacy_solver_full", "solver_canonical_full", "solver_canonical_certified_full"},
                "effective_max_concurrency": effective_concurrency,
                "command": command,
            }
        )
    next_run = _table10_next_run_group(rows)
    max_api_concurrency = max((int(row["effective_max_concurrency"]) for row in rows if bool(row["api_required"])), default=0)
    csv_path = out_dir / "table10_tau3_run_preflight.csv"
    json_path = out_dir / "table10_tau3_run_preflight.json"
    md_path = out_dir / "table10_tau3_run_preflight.md"
    _write_csv(csv_path, rows)
    api_key_env = "OPENAI_API_KEY"
    api_key_present = bool(os.environ.get(api_key_env))
    manifest = {
        "artifact": "table10_tau3_run_preflight",
        "base_dir": str(base_dir),
        "expected_test_count": expected_test_count,
        "best_of_n": best_of_n,
        "num_trials": num_trials,
        "seed": seed,
        "max_concurrency": max_concurrency,
        "candidate_workers": candidate_workers,
        "candidate_retries": candidate_retries,
        "candidate_retry_backoff": candidate_retry_backoff,
        "agent_llm": agent_llm,
        "user_llm": user_llm,
        "python_bin": python_bin,
        "tau_python_bin": tau_python_bin,
        "parallel_domains": parallel_domains,
        "parallel_method_groups": parallel_method_groups,
        "api_key_env": api_key_env,
        "api_key_present": api_key_present,
        "recommended_next_requires_api_key": bool(next_run and next_run.get("api_required")),
        "recommended_next_api_key_ready": bool((not next_run) or (not next_run.get("api_required")) or api_key_present),
        "readiness_dir": str(out_dir / "readiness"),
        "all_rows_result_ready": readiness["all_rows_result_ready"],
        "recommended_next_run_group": "" if next_run is None else next_run["run_group"],
        "recommended_next_command": "" if next_run is None else next_run["command"],
        "max_api_backed_effective_concurrency": max_api_concurrency,
        "rows": rows,
        "produced_artifacts": [csv_path.name, json_path.name, md_path.name, "readiness"],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_table10_preflight_markdown(manifest), encoding="utf-8")
    return manifest


def build_tau3_best_of_n_from_native(out_dir: Path, native_dir: Path, n: int = 4, method: Optional[str] = None) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, int(n))
    result_path = _find_official_results(native_dir)
    if result_path is None:
        return _write_pending_row(out_dir, method=method or f"Best-of-{n}", domain="", status="native results missing")
    data = json.loads(result_path.read_text(encoding="utf-8"))
    sims = _load_simulations(result_path, data)
    by_task: Dict[str, List[Mapping[str, object]]] = {}
    for sim in sims:
        task_id = str(sim.get("task_id", ""))
        if task_id:
            by_task.setdefault(task_id, []).append(sim)
    selected_rows = []
    success_count = 0
    total_tool_calls = 0
    total_tokens = 0
    token_observed = False
    total_llm_calls = 0
    total_wall = 0.0
    violations = 0
    for task_id, task_sims in sorted(by_task.items()):
        candidates = task_sims[:n]
        if not candidates:
            continue
        candidate_rewards = [_reward(sim) for sim in candidates]
        selected = max(candidates, key=lambda sim: (_reward(sim), -_as_float(sim.get("duration")) if not math.isnan(_as_float(sim.get("duration"))) else 0.0))
        reward = _reward(selected)
        success = reward >= 1.0 - 1e-6
        success_count += 1 if success else 0
        selected_term = str(selected.get("termination_reason", ""))
        if selected_term not in {"user_stop", "agent_stop"} and not success:
            violations += 1
        task_tokens = 0
        task_token_observed = False
        task_calls = 0
        task_tool_calls = 0
        task_wall = 0.0
        for sim in candidates:
            messages = sim.get("messages") or []
            task_tool_calls += _count_agent_tool_calls(messages)
            tokens, calls = _token_usage(messages)
            if tokens is not None:
                task_tokens += tokens
                task_token_observed = True
            task_calls += calls
            duration = _as_float(sim.get("duration"))
            if not math.isnan(duration):
                task_wall += duration
        total_tool_calls += task_tool_calls
        total_llm_calls += task_calls
        total_wall += task_wall
        if task_token_observed:
            total_tokens += task_tokens
            token_observed = True
        selected_rows.append(
            {
                "Domain": _domain_from_results(data),
                "Method": method or f"Best-of-{n}",
                "task_id": task_id,
                "selected_rollout_id": selected.get("id", ""),
                "candidate_rewards": json.dumps(candidate_rewards),
                "first_reward": candidate_rewards[0] if candidate_rewards else "",
                "best_reward": max(candidate_rewards) if candidate_rewards else "",
                "selected_reward": reward,
                "success": 1 if success else 0,
                "first_success": 1 if candidate_rewards and candidate_rewards[0] >= 1.0 - 1e-6 else 0,
                "any_success": 1 if any(value >= 1.0 - 1e-6 for value in candidate_rewards) else 0,
                "sampled_rollouts": len(candidates),
                "expected_rollouts": n,
                "rollout_coverage_ready": len(candidates) >= n,
                "tool_calls": task_tool_calls,
                "tokens": "" if not task_token_observed else task_tokens,
                "llm_calls": task_calls,
                "duration_seconds": task_wall,
                "selected_termination_reason": selected_term,
                "candidate_termination_reasons": json.dumps(_count_values(str(sim.get("termination_reason", "")) for sim in candidates), sort_keys=True),
            }
        )
    task_count = len(selected_rows)
    aggregate_row = {
        "Domain": _domain_from_results(data),
        "Method": method or f"Best-of-{n}",
        "pass^1": success_count / max(task_count, 1),
        "pass^4": "",
        "violations": violations / max(task_count, 1),
        "tool_calls": total_tool_calls / max(task_count, 1),
        "llm_calls": total_llm_calls / max(task_count, 1),
        "tokens": "" if not token_observed else total_tokens / max(task_count, 1),
        "wall_clock_seconds": total_wall / max(task_count, 1),
        "status": "complete" if task_count else "pending",
        "source_url": TAU2_BENCH_SOURCE_URL,
        "preferred_current_source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": _tau2_revision(),
        "official_results": str(result_path),
        "task_count": task_count,
        "simulation_count": sum(len(values[:n]) for values in by_task.values()),
        "avg_rollouts_per_task": 0.0 if task_count == 0 else sum(row["sampled_rollouts"] for row in selected_rows) / task_count,
        "selector": "official_reward_best_of_n",
        "selector_expected_rollouts": n,
        "selector_full_coverage": all(int(row["sampled_rollouts"]) >= n for row in selected_rows) if selected_rows else False,
        "notes": "Best-of-N row reuses native tau2 rollouts and selects by official evaluator reward; compute counts all sampled rollouts.",
    }
    _write_csv(out_dir / "q4_tau_per_instance.csv", selected_rows)
    _write_csv(out_dir / "q4_tau_table.csv", [aggregate_row])
    (out_dir / "q4_tau_row.json").write_text(json.dumps(aggregate_row, indent=2, sort_keys=True), encoding="utf-8")
    manifest = _tau_manifest(method or f"Best-of-{n}", aggregate_row["Domain"], task_count, n, result_path)
    (out_dir / "q4_tau_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"row": aggregate_row, "manifest": manifest, "produced_artifacts": ["q4_tau_row.json", "q4_tau_table.csv", "q4_tau_per_instance.csv", "q4_tau_manifest.json"]}


def audit_tau3_controller_trace(out_dir: Path, trace_path: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    event_count = 0
    canonical_events = 0
    heuristic_events = 0
    ranker_loaded = 0
    certified_filter_enabled_events = 0
    certified_pruned_candidates = 0
    candidate_sample_failures = 0
    candidate_count = 0
    scored_candidate_count = 0
    oracle_visible = False
    oracle_generation = False
    oracle_filtering = False
    tool_name_features = False
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            event_count += 1
            mode = str(row.get("scoring_mode", ""))
            canonical_events += 1 if mode == "canonical_ranker" else 0
            heuristic_events += 1 if mode == "heuristic" else 0
            ranker_loaded += 1 if row.get("ranker_checkpoint_loaded") else 0
            certified_filter_enabled_events += 1 if row.get("certified_filter_enabled") else 0
            pruned_count = _as_float(row.get("certified_pruned_count"))
            certified_pruned_candidates += int(pruned_count) if not math.isnan(pruned_count) else 0
            failure_count = _as_float(row.get("candidate_sample_failures"))
            candidate_sample_failures += int(failure_count) if not math.isnan(failure_count) else 0
            candidate_scores = row.get("candidate_scores")
            if isinstance(candidate_scores, list):
                candidate_count += len(candidate_scores)
                scored_candidate_count += sum(1 for item in candidate_scores if isinstance(item, Mapping) and item.get("score") is not None)
            oracle_visible = oracle_visible or bool(row.get("oracle_visible_to_controller"))
            oracle_generation = oracle_generation or bool(row.get("oracle_used_for_candidate_generation"))
            oracle_filtering = oracle_filtering or bool(row.get("oracle_used_for_filtering"))
            tool_name_features = tool_name_features or bool(row.get("tool_names_used_as_features"))
    summary = {
        "trace_path": str(trace_path),
        "event_count": event_count,
        "canonical_ranker_events": canonical_events,
        "heuristic_events": heuristic_events,
        "ranker_checkpoint_loaded_events": ranker_loaded,
        "certified_filter_enabled_events": certified_filter_enabled_events,
        "certified_pruned_candidates": certified_pruned_candidates,
        "candidate_sample_failures": candidate_sample_failures,
        "certified_prune_rate": 0.0 if candidate_count == 0 else certified_pruned_candidates / candidate_count,
        "mean_candidate_count": 0.0 if event_count == 0 else candidate_count / event_count,
        "mean_scored_candidate_count": 0.0 if event_count == 0 else scored_candidate_count / event_count,
        "oracle_visible_to_controller": oracle_visible,
        "oracle_used_for_candidate_generation": oracle_generation,
        "oracle_used_for_filtering": oracle_filtering,
        "tool_names_used_as_features": tool_name_features,
        "status": "complete" if event_count > 0 else "empty_or_missing_trace",
    }
    _write_csv(out_dir / "tau3_controller_trace_audit.csv", [summary])
    (out_dir / "tau3_controller_trace_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _write_pending_artifacts(config: Tau3RunConfig, status: str, save_to: str) -> Dict[str, object]:
    row = {
        "Domain": config.domain,
        "Method": _tau3_method_label(config),
        "pass^1": "",
        "pass^4": "",
        "violations": "",
        "tool_calls": "",
        "llm_calls": "",
        "tokens": "",
        "wall_clock_seconds": "",
        "status": status,
        "source_url": TAU2_BENCH_SOURCE_URL,
        "preferred_current_source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": _tau2_revision(),
        "official_results": save_to,
        "task_count": 0,
        "simulation_count": 0,
        "avg_rollouts_per_task": 0,
        "notes": "official tau2-bench run not completed",
    }
    (config.output_dir / "q4_tau_row.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(config.output_dir / "q4_tau_table.csv", [row])
    _write_csv(config.output_dir / "q4_tau_per_instance.csv", [])
    manifest = _tau_manifest(_tau3_method_label(config), config.domain, 0, config.num_trials, Path(save_to))
    (config.output_dir / "q4_tau_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return {"row": row, "manifest": manifest, "produced_artifacts": ["q4_tau_row.json", "q4_tau_table.csv", "q4_tau_per_instance.csv", "q4_tau_manifest.json"]}


def _write_pending_row(out_dir: Path, *, method: str, domain: str, status: str) -> Dict[str, object]:
    row = {
        "Domain": domain,
        "Method": method,
        "pass^1": "",
        "pass^4": "",
        "violations": "",
        "tool_calls": "",
        "llm_calls": "",
        "tokens": "",
        "wall_clock_seconds": "",
        "status": status,
        "source_url": TAU2_BENCH_SOURCE_URL,
        "preferred_current_source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": _tau2_revision(),
        "official_results": "",
        "task_count": 0,
        "simulation_count": 0,
        "avg_rollouts_per_task": 0,
        "notes": "official tau2-bench results missing",
    }
    (out_dir / "q4_tau_row.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "q4_tau_table.csv", [row])
    return {"row": row}


def _table10_missing_compute_blockers(row: Mapping[str, object]) -> List[str]:
    blockers = []
    for key, label in (
        ("tool_calls", "tool call accounting missing"),
        ("llm_calls", "LLM call accounting missing"),
        ("tokens", "token accounting missing"),
        ("wall_clock_seconds", "wall-clock accounting missing"),
    ):
        if math.isnan(_as_float(row.get(key))):
            blockers.append(label)
    return blockers


def _table10_audit_row(
    row_dir: Path,
    row: Mapping[str, object],
    blockers: Sequence[str],
    expected_test_count: int,
    *,
    source_protocol: Optional[Mapping[str, object]] = None,
    trace_summary: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    expected_domain, expected_method = _table10_expected_label(row_dir)
    method = str(row.get("Method") or expected_method or "missing row")
    task_count = int(_as_float(row.get("task_count")) if not math.isnan(_as_float(row.get("task_count"))) else 0)
    source_protocol = source_protocol or {}
    trace_summary = trace_summary or {}
    notes = _table10_evidence_notes(source_protocol, trace_summary)
    return {
        "Method": method,
        "Domain": row.get("Domain") or expected_domain,
        "row_dir": str(row_dir),
        "source_artifact": str(row_dir),
        "source_policy": "official tau3 maintained protocol",
        "status": row.get("status", "missing"),
        "task_count": task_count,
        "expected_test_count": expected_test_count,
        "source_generator_freeze_ready": source_protocol.get("source_generator_distribution_fixed_before_target_test_evaluation", ""),
        "source_generator_anonymized_tools": source_protocol.get("source_generator_anonymized_tool_names", ""),
        "tool_identity_generalization_ready": source_protocol.get("tool_identity_generalization_ready", ""),
        "tool_identity_generalization_top1_delta": source_protocol.get("tool_identity_generalization_top1_delta", ""),
        "contract_coverage_stress_ready": source_protocol.get("contract_coverage_stress_ready", ""),
        "contract_coverage_black_box_top1_delta_from_full": source_protocol.get("contract_coverage_black_box_top1_delta_from_full", ""),
        "canonical_ranker_events": trace_summary.get("canonical_ranker_events", ""),
        "candidate_sample_failures": trace_summary.get("candidate_sample_failures", ""),
        "certified_prune_rate": trace_summary.get("certified_prune_rate", ""),
        "notes": notes,
        "result_ready": not blockers,
        "result_blockers": "; ".join(blockers),
    }


def _table10_evidence_notes(source_protocol: Mapping[str, object], trace_summary: Mapping[str, object]) -> str:
    if not source_protocol and not trace_summary:
        return ""
    notes = []
    if source_protocol.get("source_generator_distribution_fixed_before_target_test_evaluation"):
        notes.append("source generator frozen before target test")
    if source_protocol.get("source_generator_anonymized_tool_names"):
        notes.append("source tools anonymized")
    if source_protocol.get("tool_identity_generalization_ready"):
        notes.append(f"tool-identity stress delta={source_protocol.get('tool_identity_generalization_top1_delta', '')}")
    if source_protocol.get("contract_coverage_stress_ready"):
        notes.append(f"contract coverage black-box delta={source_protocol.get('contract_coverage_black_box_top1_delta_from_full', '')}")
    if trace_summary.get("canonical_ranker_events"):
        notes.append(f"canonical ranker events={trace_summary.get('canonical_ranker_events')}")
    if _as_float(trace_summary.get("candidate_sample_failures")) > 0:
        notes.append(f"candidate sample failures after retries={trace_summary.get('candidate_sample_failures')}")
    return "; ".join(notes)


def _read_json_or_empty(path: Path) -> Dict[str, object]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _table10_row_id(row: Mapping[str, object]) -> str:
    domain = str(row.get("Domain", "")).strip()
    method = str(row.get("Method", "")).strip()
    if domain and method:
        return f"{domain}: {method}"
    return method or domain or "missing row"


def _table10_expected_label(row_dir: Path) -> tuple[str, str]:
    name = row_dir.name
    domain = ""
    if name.endswith("_retail"):
        domain = "retail"
    elif name.endswith("_airline"):
        domain = "airline"
    if name.startswith("native_"):
        return domain, "Native tool-calling rerun"
    if name.startswith("best_of_"):
        return domain, "Best-of-N"
    if name.startswith("solver_canonical_certified_"):
        return domain, "Solver-derived canonical controller + certified filter"
    if name.startswith("solver_canonical_"):
        return domain, "Solver-derived canonical controller"
    if name.startswith("solver_"):
        return domain, "Solver-derived controller"
    return domain, "missing row"


def _table10_default_row_dirs(base_dir: Path, best_of_n: int) -> List[Path]:
    return [
        base_dir / "native_retail",
        base_dir / "native_airline",
        base_dir / f"best_of_{best_of_n}_retail",
        base_dir / f"best_of_{best_of_n}_airline",
        base_dir / "solver_retail",
        base_dir / "solver_airline",
        base_dir / "solver_canonical_retail",
        base_dir / "solver_canonical_airline",
        base_dir / "solver_canonical_certified_retail",
        base_dir / "solver_canonical_certified_airline",
    ]


def _table10_run_groups(
    *,
    base_dir: Path,
    expected_test_count: int,
    best_of_n: int,
    num_trials: int,
    seed: int,
    max_concurrency: int,
    candidate_workers: int,
    candidate_retries: int,
    candidate_retry_backoff: float,
    agent_llm: str,
    user_llm: str,
    python_bin: str,
    tau_python_bin: str,
    parallel_domains: bool,
    parallel_method_groups: bool,
) -> List[tuple[str, str, List[Path]]]:
    prefix = (
        f"TAU_PYTHON_BIN={tau_python_bin} PYTHON_BIN={python_bin} "
        f"AGENT_LLM={agent_llm} USER_LLM={user_llm} "
        f"NUM_TASKS={expected_test_count} NUM_TRIALS={num_trials} "
        f"SEED={seed} MAX_CONCURRENCY={max_concurrency} CANDIDATE_WORKERS={candidate_workers} "
        f"CANDIDATE_RETRIES={candidate_retries} CANDIDATE_RETRY_BACKOFF={candidate_retry_backoff} "
        f"BEST_OF_N={best_of_n} "
        f"PARALLEL_DOMAINS={1 if parallel_domains else 0} "
        f"PARALLEL_METHOD_GROUPS={1 if parallel_method_groups else 0} "
        f"BASE_DIR={base_dir}"
    )
    default_dirs = _table10_default_row_dirs(base_dir, best_of_n)
    return [
        ("full", f"{prefix} scripts/run_table10_tau3.sh full", default_dirs),
        ("native_full", f"{prefix} scripts/run_table10_tau3.sh native-full", [base_dir / "native_retail", base_dir / "native_airline", base_dir / f"best_of_{best_of_n}_retail", base_dir / f"best_of_{best_of_n}_airline"]),
        ("legacy_solver_full", f"{prefix} scripts/run_table10_tau3.sh solver-full", [base_dir / "solver_retail", base_dir / "solver_airline"]),
        ("solver_canonical_full", f"{prefix} scripts/run_table10_tau3.sh solver-canonical-full", [base_dir / "solver_canonical_retail", base_dir / "solver_canonical_airline"]),
        (
            "solver_canonical_certified_full",
            f"{prefix} scripts/run_table10_tau3.sh solver-canonical-certified-full",
            [base_dir / "solver_canonical_certified_retail", base_dir / "solver_canonical_certified_airline"],
        ),
        ("build_and_audit", f"BASE_DIR={base_dir} NUM_TASKS={expected_test_count} BEST_OF_N={best_of_n} scripts/run_table10_tau3.sh build", default_dirs),
    ]


def _methods_for_dirs(audit_rows: Sequence[Mapping[str, object]], dirs: Sequence[Path]) -> List[Mapping[str, object]]:
    wanted = {str(path) for path in dirs}
    return [row for row in audit_rows if str(row.get("row_dir", "")) in wanted]


def _table10_next_run_group(rows: Sequence[Mapping[str, object]]) -> Optional[Mapping[str, object]]:
    by_group = {str(row.get("run_group", "")): row for row in rows}
    for group in ("source_controller", "native_full", "legacy_solver_full", "solver_canonical_full", "solver_canonical_certified_full"):
        row = by_group.get(group)
        if row and not bool(row.get("ready")):
            return row
    build = by_group.get("build_and_audit")
    if build and not bool(build.get("ready")):
        return build
    return None


def _table10_source_controller_preflight_row(base_dir: Path, *, seed: int = 0) -> Dict[str, object]:
    checkpoint = base_dir / f"source_controller_seed{seed}" / "generic_harness_ranker.json"
    protocol = _source_controller_protocol(checkpoint)
    blockers = []
    if not checkpoint.exists():
        blockers.append("generic_harness_ranker.json missing")
    if not protocol.get("manifest_found"):
        blockers.append("canonical_harness_source_validation.json missing")
    if protocol.get("tau3_train_trajectories_used"):
        blockers.append("source controller used tau3 train trajectories")
    if protocol.get("oracle_labels_in_inference_features"):
        blockers.append("source controller leaks oracle labels")
    if protocol.get("tool_names_used_as_features"):
        blockers.append("source controller uses tool names as features")
    if protocol.get("source_generator_uses_tau3_trajectories"):
        blockers.append("source generator used tau3 trajectories")
    if protocol.get("tau3_heldout_test_seen_before_freeze"):
        blockers.append("tau3 heldout test influenced source generator freeze")
    if protocol.get("manifest_found") and not protocol.get("controller_checkpoint_frozen"):
        blockers.append("source controller checkpoint freeze evidence missing")
    if protocol.get("manifest_found") and not protocol.get("source_generator_distribution_fixed_before_target_test_evaluation"):
        blockers.append("source generator freeze evidence missing")
    if protocol.get("manifest_found") and not protocol.get("source_generator_anonymized_tool_names"):
        blockers.append("source generator tool-name anonymization evidence missing")
    if protocol.get("manifest_found") and not protocol.get("tool_identity_generalization_ready"):
        blockers.append("tool-identity generalization stress missing or failed")
    if protocol.get("manifest_found") and not protocol.get("contract_coverage_stress_ready"):
        blockers.append("contract coverage stress missing or failed")
    return {
        "run_group": "source_controller",
        "ready": not blockers,
        "methods": "GenericHarnessRanker source controller",
        "remaining_rows": 0 if not blockers else 1,
        "api_required": False,
        "resume_safe": True,
        "effective_max_concurrency": 0,
        "command": f"BASE_DIR={base_dir} SEED={seed} scripts/run_table10_tau3.sh source-controller",
        "blockers": "; ".join(blockers),
        "source_controller_checkpoint": str(checkpoint),
        "source_controller_protocol": protocol.get("manifest", ""),
    }


def _table10_effective_concurrency(group: str, max_concurrency: int, candidate_workers: int, parallel_domains: bool, parallel_method_groups: bool) -> int:
    if group in {"build_and_audit", "source_controller"}:
        return 0
    domain_factor = 2 if parallel_domains else 1
    base = max(1, int(max_concurrency)) * domain_factor
    solver_candidate_factor = max(1, int(candidate_workers))
    if group == "native_full":
        return base
    if group == "full" and parallel_method_groups:
        return base * (1 + 3 * solver_candidate_factor)
    if group == "full":
        return base * solver_candidate_factor
    return base * solver_candidate_factor


def _table10_preflight_markdown(manifest: Mapping[str, object]) -> str:
    lines = [
        "# Table10 Tau3 Run Preflight",
        "",
        f"Base dir: `{manifest['base_dir']}`",
        f"Expected test count: `{manifest['expected_test_count']}`",
        f"Best-of-N: `{manifest['best_of_n']}`",
        f"Candidate workers: `{manifest['candidate_workers']}`",
        f"Candidate retries: `{manifest['candidate_retries']}`",
        f"Candidate retry backoff: `{manifest['candidate_retry_backoff']}`",
        f"Parallel domains: `{manifest['parallel_domains']}`",
        f"Parallel method groups: `{manifest['parallel_method_groups']}`",
        f"API key env: `{manifest['api_key_env']}`",
        f"API key present: `{manifest['api_key_present']}`",
        f"Max API-backed effective concurrency: `{manifest['max_api_backed_effective_concurrency']}`",
        f"All rows ready: `{manifest['all_rows_result_ready']}`",
        f"Recommended next run group: `{manifest['recommended_next_run_group'] or 'none'}`",
        "",
        "Set `OPENAI_API_KEY` before API-backed run groups. `PARALLEL_DOMAINS=1` runs Retail and Airline concurrently; solver-controller rows also multiply active requests by `CANDIDATE_WORKERS`. `best-of-n` and `build_and_audit` reuse local artifacts.",
        "",
        "Resume policy: generated commands use `--auto-resume --overwrite`; `--overwrite` only permits writing into an existing output directory, while `--force` is the opt-in switch that ignores existing tau2 results and reruns from scratch.",
        "",
    ]
    if manifest.get("recommended_next_requires_api_key") and not manifest.get("recommended_next_api_key_ready"):
        lines.extend(
            [
                f"Warning: recommended next run needs `{manifest['api_key_env']}`, but it is not present in this shell.",
                "",
            ]
        )
    if manifest.get("recommended_next_command"):
        lines.extend(
            [
                "Recommended next command:",
                "",
                "```bash",
                str(manifest["recommended_next_command"]),
                "```",
                "",
            ]
        )
    for row in manifest["rows"]:
        lines.extend(
            [
                f"## {row['run_group']}",
                "",
                f"Ready: `{row['ready']}`",
                f"API required: `{row['api_required']}`",
                f"Resume safe: `{row['resume_safe']}`",
                f"Effective max concurrency: `{row['effective_max_concurrency']}`",
                f"Methods: {row['methods']}",
                "",
                "```bash",
                str(row["command"]),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _tau_manifest(method: str, domain: str, task_count: int, num_trials: int, official_results: Path) -> Dict[str, object]:
    evidence = q4_tau_source_evidence("tau2-bench")
    solver_like = method in {
        "Solver-derived controller",
        "Solver-derived canonical controller",
        "Solver-derived canonical controller + certified filter",
    }
    method_primitives_by_name = {
        "Native tool-calling rerun": ["single_native_agent_rollout"],
        "Best-of-N": ["multiple_independent_native_rollouts", "benchmark_compatible_selection"],
        "Solver-derived controller": ["source_solver_training", "zero_shot_target_tool_transfer", "native_tool_state_preserved"],
        "Solver-derived canonical controller": ["source_workflow_training", "canonical_harness_adapter", "zero_shot_target_tool_transfer", "native_tool_state_preserved"],
        "Solver-derived canonical controller + certified filter": [
            "source_workflow_training",
            "canonical_harness_adapter",
            "certified_interface_filtering",
            "zero_shot_target_tool_transfer",
            "native_tool_state_preserved",
        ],
    }
    method_primitives = method_primitives_by_name.get(method)
    if method_primitives is None and method.startswith("Best-of-"):
        method_primitives = method_primitives_by_name["Best-of-N"]
    if method_primitives is None:
        method_primitives = ["single_native_agent_rollout"]
    return {
        "benchmark": "tau3-bench",
        "native_protocol_preserved": True,
        "source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": _tau2_revision(),
        "source_evidence": evidence,
        "domain": domain,
        "task_split_name": "base",
        "test_count": task_count,
        "num_trials": num_trials,
        "methods": [method],
        "official_results": str(official_results),
        "native_adapter": {
            "adapter_class": "Tau3OfficialNativeRunner",
            "implementation_file": "cgr_exp/tau3_native.py",
            "implementation_status": "official",
            "uses_external_stub": False,
        },
        "official_entrypoints": {
            "runner": "tau2.cli::run / tau2.runner.batch::run_domain",
            "environment": "tau2.runner.build::build_environment",
            "user_simulator": "tau2.user.user_simulator::UserSimulator",
            "evaluator": "tau2.evaluator.evaluator::evaluate",
        },
        "native_protocol": {
            "preserved_components": [
                "stateful_tool_environment",
                "user_simulator",
                "official_task_verifier",
                "pass_k_aggregation",
                "tool_call_accounting",
                "token_accounting",
            ],
            "uses_official_user_simulator": True,
            "uses_official_task_verifier": True,
            "preserves_tool_state": True,
            "reported_metrics": [
                "pass^1",
                "pass^4",
                "violations",
                "tool_calls",
                "llm_calls",
                "tokens",
                "wall_clock_seconds",
            ],
        },
        "method_protocols": {
            method: {
                "method_primitives": method_primitives,
                "training_eval_overlap": False,
                "zero_shot_transfer": True if solver_like else None,
            }
        },
    }


def _find_official_results(out_dir: Path) -> Optional[Path]:
    candidates = [
        out_dir / "official_tau2_results" / "results.json",
        out_dir / "official_tau2_results.json",
    ]
    candidates.extend(sorted(out_dir.glob("**/results.json")))
    for candidate in candidates:
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _load_simulations(result_path: Path, data: Mapping[str, object]) -> List[Mapping[str, object]]:
    sims = data.get("simulations")
    if isinstance(sims, list) and sims:
        return [sim for sim in sims if isinstance(sim, Mapping)]
    sim_dir = result_path.parent / "simulations"
    loaded: List[Mapping[str, object]] = []
    if sim_dir.exists():
        for path in sorted(sim_dir.glob("*.json")):
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                loaded.append(parsed)
    return loaded


def _reward(sim: Mapping[str, object]) -> float:
    info = sim.get("reward_info")
    if isinstance(info, Mapping):
        return _as_float(info.get("reward"))
    return float("nan")


def _pass_hat(rewards_by_task: Mapping[str, Sequence[float]], k: int) -> float:
    values = []
    for rewards in rewards_by_task.values():
        n = len(rewards)
        c = sum(1 for value in rewards if value >= 1.0 - 1e-6)
        if n < k:
            continue
        if k == 1:
            values.append(c / n)
        else:
            values.append(math.comb(c, k) / math.comb(n, k))
    return sum(values) / len(values) if values else float("nan")


def _count_values(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _count_agent_tool_calls(messages: Sequence[object]) -> int:
    total = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role not in {"assistant", "agent"}:
            continue
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            total += len(tool_calls)
    return total


def _token_usage(messages: Sequence[object]) -> tuple[Optional[int], int]:
    total = 0
    observed = False
    calls = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        usage = message.get("usage")
        if isinstance(usage, Mapping):
            prompt = int(_as_float(usage.get("prompt_tokens")) if not math.isnan(_as_float(usage.get("prompt_tokens"))) else 0)
            completion = int(_as_float(usage.get("completion_tokens")) if not math.isnan(_as_float(usage.get("completion_tokens"))) else 0)
            total += prompt + completion
            observed = True
            calls += 1
    return (total if observed else None), calls


def _domain_from_results(data: Mapping[str, object]) -> str:
    info = data.get("info")
    if isinstance(info, Mapping):
        env = info.get("environment_info")
        if isinstance(env, Mapping):
            return str(env.get("domain_name", ""))
    return ""


def _tau2_revision() -> str:
    if not TAU2_SUBMODULE.exists():
        return ""
    result = subprocess.run(["git", "-C", str(TAU2_SUBMODULE), "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else ""


def _table10_row(row: Mapping[str, object], expected_test_count: int) -> Dict[str, object]:
    task_count = int(_as_float(row.get("task_count")) if not math.isnan(_as_float(row.get("task_count"))) else 0)
    status = str(row.get("status", ""))
    if status == "complete" and task_count < expected_test_count:
        status = f"partial: {task_count}/{expected_test_count}"
    return {
        "Domain": row.get("Domain", ""),
        "Method": row.get("Method", ""),
        "pass^1": _fmt_float(_as_float(row.get("pass^1")), 4),
        "pass^4": _fmt_float(_as_float(row.get("pass^4")), 4),
        "Violations": _fmt_float(_as_float(row.get("violations")), 4),
        "Tool calls": _fmt_float(_as_float(row.get("tool_calls")), 2),
        "LLM calls": _fmt_float(_as_float(row.get("llm_calls")), 2),
        "Tokens": _fmt_float(_as_float(row.get("tokens")), 0),
        "Wall-clock": _fmt_float(_as_float(row.get("wall_clock_seconds")), 2),
        "Status": status,
        "task_count": task_count,
        "source_artifact": row.get("official_results", ""),
    }


def _table10_latex(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Matched maintained-protocol tau3 evaluation using official tau2-bench text half-duplex runner.}",
        "\\label{tab:q4_tau3}",
        "\\scriptsize",
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        "Domain & Method & pass$^1$ $\\uparrow$ & pass$^4$ $\\uparrow$ & Violations $\\downarrow$ & Tool calls $\\downarrow$ & LLM calls $\\downarrow$ & Tokens $\\downarrow$ & Wall-clock $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row.get("Domain", ""))),
                    _latex_escape(str(row.get("Method", ""))),
                    str(row.get("pass^1", "--")),
                    str(row.get("pass^4", "--")),
                    str(row.get("Violations", "--")),
                    str(row.get("Tool calls", "--")),
                    str(row.get("LLM calls", "--")),
                    str(row.get("Tokens", "--")),
                    str(row.get("Wall-clock", "--")),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: object) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _fmt_float(value: float, digits: int) -> str:
    return "--" if math.isnan(value) else f"{value:.{digits}f}"


def _latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_")
