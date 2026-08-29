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
    command_manifest = {
        "command": command,
        "cwd": str(TAU2_SUBMODULE),
        "dry_run": config.dry_run,
        "method": TABLE10_METHODS[config.method],
        "domain": config.domain,
        "task_split_name": config.task_split_name,
        "num_tasks": config.num_tasks,
        "task_ids": list(config.task_ids),
        "num_trials": config.num_trials,
        "max_concurrency": config.max_concurrency,
        "workers": config.workers,
        "solver_ranker_checkpoint": None if config.solver_ranker_checkpoint is None else str(config.solver_ranker_checkpoint),
        "enable_certified_filter": config.enable_certified_filter,
        "controller_trace_path": None if controller_trace_path is None else str(controller_trace_path),
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
        return aggregate_tau3_official_results(config.output_dir, method=TABLE10_METHODS[config.method], domain=config.domain, expected_trials=config.num_trials)

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
    return aggregate_tau3_official_results(config.output_dir, method=TABLE10_METHODS[config.method], domain=config.domain, expected_trials=config.num_trials)


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
        row["rollout_count"] = len(rewards_by_task.get(task_id, []))

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


def build_table10_tau3_results(out_dir: Path, row_dirs: Sequence[Path], expected_test_count: int = 100) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for row_dir in row_dirs:
        row_path = row_dir / "q4_tau_row.json"
        if not row_path.exists():
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


def audit_tau3_controller_trace(out_dir: Path, trace_path: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    event_count = 0
    canonical_events = 0
    heuristic_events = 0
    ranker_loaded = 0
    certified_filter_enabled_events = 0
    certified_pruned_candidates = 0
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
        "Method": TABLE10_METHODS[config.method],
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
    manifest = _tau_manifest(TABLE10_METHODS[config.method], config.domain, 0, config.num_trials, Path(save_to))
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


def _tau_manifest(method: str, domain: str, task_count: int, num_trials: int, official_results: Path) -> Dict[str, object]:
    evidence = q4_tau_source_evidence("tau2-bench")
    method_primitives = {
        "Native tool-calling rerun": ["single_native_agent_rollout"],
        "Best-of-N": ["multiple_independent_native_rollouts", "benchmark_compatible_selection"],
        "Solver-derived controller": ["source_solver_training", "zero_shot_target_tool_transfer", "native_tool_state_preserved"],
    }.get(method, ["single_native_agent_rollout"])
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
            "reported_metrics": ["pass^1", "pass^4", "violations", "tool_calls", "tokens"],
        },
        "method_protocols": {
            method: {
                "method_primitives": method_primitives,
                "training_eval_overlap": False,
                "zero_shot_transfer": True if method == "Solver-derived controller" else None,
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
        "\\small",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Domain & Method & pass$^1$ $\\uparrow$ & pass$^4$ $\\uparrow$ & Violations $\\downarrow$ & Tool calls $\\downarrow$ & Tokens $\\downarrow$ \\\\",
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
                    str(row.get("Tokens", "--")),
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


def _fmt_float(value: float, digits: int) -> str:
    return "--" if math.isnan(value) else f"{value:.{digits}f}"


def _latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_")
