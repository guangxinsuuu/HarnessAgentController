from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from cgr_exp.q2_method_contracts import q2_method_contract, q2_method_contracts


TABLE5_METHODS = [
    ("CP-SAT", "Structured-input solver", None, "structured"),
    ("EST / list scheduling heuristic", "Structured-input heuristic", None, "structured"),
    ("Direct / CoT", "Frozen", None, "native"),
    ("Self-Consistency", "Frozen", None, "native"),
    ("Best-of-N", "Frozen", None, "native"),
    ("Tree of Thoughts", "Online", None, "native"),
    ("Target-PRM priority", "In-domain trained", "Target-PRM", "scheduling"),
    ("PRM-guided MCTS (ReST-MCTS-style)", "In-domain trained", "PRM-guided MCTS (ReST-MCTS-style)", "scheduling"),
    ("Solver-trained priority", "Zero-shot transfer", "Solver-trained MLP", "scheduling"),
    ("Solver-trained priority + certified pruning", "Zero-shot transfer", "Solver-trained MLP + certified pruning", "scheduling"),
]

TABLE5_SECTIONS = [
    (
        "Task-specific optimization references -- exact structured input",
        {"CP-SAT", "EST / list scheduling heuristic"},
    ),
    (
        "Frozen and online LLM inference",
        {"Direct / CoT", "Self-Consistency", "Best-of-N", "Tree of Thoughts"},
    ),
    (
        "Target-domain learned search",
        {"Target-PRM priority", "PRM-guided MCTS (ReST-MCTS-style)"},
    ),
    (
        "Ours -- solver-to-LLM zero-shot transfer",
        {"Solver-trained priority", "Solver-trained priority + certified pruning"},
    ),
]


def build_table5_q2_results(
    out_dir: Path,
    solver_prior_dir: Path = Path("runs/scheduling_llm_pilot_real_100_seed0"),
    expected_test_count: int = 100,
    gpt_frontier_rank_dir: Optional[Path] = None,
    target_prm_dir: Optional[Path] = None,
    direct_dir: Optional[Path] = None,
    self_consistency_dir: Optional[Path] = None,
    best_of_n_dir: Optional[Path] = None,
    rest_mcts_dir: Optional[Path] = None,
    solver_certification_dir: Optional[Path] = None,
    structured_dir: Optional[Path] = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    method_dirs: Dict[str, Optional[Path]] = {
        "CP-SAT": structured_dir,
        "EST / list scheduling heuristic": structured_dir,
        "Direct / CoT": direct_dir,
        "Self-Consistency": self_consistency_dir,
        "Best-of-N": best_of_n_dir,
        "Tree of Thoughts": gpt_frontier_rank_dir,
        "Target-PRM priority": target_prm_dir,
        "PRM-guided MCTS (ReST-MCTS-style)": rest_mcts_dir,
        "Solver-trained priority": solver_prior_dir,
        "Solver-trained priority + certified pruning": solver_certification_dir,
    }
    rows: List[Dict[str, object]] = []
    for method, transfer_setting, source_policy, source_kind in TABLE5_METHODS:
        source_dir = method_dirs[method]
        row = _pending_row(method, transfer_setting, source_dir)
        if source_dir is not None and source_kind == "structured":
            loaded = _row_from_structured_artifacts(method, transfer_setting, source_dir)
            if loaded is not None:
                row = loaded
        elif source_dir is not None and source_policy is not None:
            loaded = _row_from_scheduling_artifacts(method, transfer_setting, source_policy, source_dir, expected_test_count)
            if loaded is not None:
                row = loaded
        elif source_dir is not None:
            loaded = _row_from_native_artifacts(method, transfer_setting, source_dir)
            if loaded is not None:
                row = loaded
        row["expected_test_count"] = expected_test_count
        row.update(_contract_fields(method))
        rows.append(row)
    csv_path = out_dir / "table5_q2_results.csv"
    tex_path = out_dir / "table5_q2_results.tex"
    manifest_path = out_dir / "table5_q2_manifest.json"
    _write_csv(csv_path, rows)
    tex_path.write_text(_latex_table(rows), encoding="utf-8")
    manifest = {
        "table": "Table 4",
        "legacy_artifact_prefix": "table5_q2",
        "question": "Q2 end-to-end quality-compute comparison",
        "strict_pending_policy": True,
        "solver_prior_dir": str(solver_prior_dir),
        "expected_test_count": expected_test_count,
        "gpt_frontier_rank_dir": None if gpt_frontier_rank_dir is None else str(gpt_frontier_rank_dir),
        "target_prm_dir": None if target_prm_dir is None else str(target_prm_dir),
        "structured_dir": None if structured_dir is None else str(structured_dir),
        "rows": rows,
        "method_alignment": q2_method_contracts([row["Method"] for row in rows]),
        "sections": [{"title": title, "methods": [method for method, *_rest in TABLE5_METHODS if method in methods]} for title, methods in TABLE5_SECTIONS],
        "produced_artifacts": [csv_path.name, tex_path.name, manifest_path.name],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def audit_table5_q2_readiness(
    out_dir: Path,
    solver_prior_dir: Path = Path("runs/scheduling_llm_pilot_real_100_seed0"),
    expected_test_count: int = 100,
    gpt_frontier_rank_dir: Optional[Path] = None,
    target_prm_dir: Optional[Path] = None,
    direct_dir: Optional[Path] = None,
    self_consistency_dir: Optional[Path] = None,
    best_of_n_dir: Optional[Path] = None,
    rest_mcts_dir: Optional[Path] = None,
    solver_certification_dir: Optional[Path] = None,
    structured_dir: Optional[Path] = None,
) -> Dict[str, object]:
    manifest = build_table5_q2_results(
        out_dir,
        solver_prior_dir=solver_prior_dir,
        expected_test_count=expected_test_count,
        gpt_frontier_rank_dir=gpt_frontier_rank_dir,
        target_prm_dir=target_prm_dir,
        direct_dir=direct_dir,
        self_consistency_dir=self_consistency_dir,
        best_of_n_dir=best_of_n_dir,
        rest_mcts_dir=rest_mcts_dir,
        solver_certification_dir=solver_certification_dir,
        structured_dir=structured_dir,
    )
    rows = list(manifest["rows"])
    audit_rows = [_audit_row(row) for row in rows]
    audit_csv = out_dir / "table5_q2_readiness.csv"
    audit_json = out_dir / "table5_q2_readiness.json"
    _write_csv(audit_csv, audit_rows)
    summary = {
        "table": "Table 4",
        "legacy_artifact_prefix": "table5_q2",
        "expected_test_count": expected_test_count,
        "all_rows_result_ready": all(_as_bool(row["result_ready"]) for row in audit_rows),
        "all_rows_compute_ready": all(_as_bool(row["compute_ready"]) for row in audit_rows),
        "pending_or_partial_rows": [row["Method"] for row in audit_rows if not _as_bool(row["result_ready"])],
        "compute_qualified_rows": [row["Method"] for row in audit_rows if not _as_bool(row["compute_ready"])],
        "audit_rows": audit_rows,
        "produced_artifacts": [*manifest["produced_artifacts"], audit_csv.name, audit_json.name],
    }
    audit_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "readiness": summary}


def _audit_row(row: Mapping[str, object]) -> Dict[str, object]:
    status = str(row.get("Status", "pending"))
    notes = str(row.get("notes", ""))
    tokens = str(row.get("Tokens", "--"))
    result_blockers = []
    compute_blockers = []
    if status != "complete":
        result_blockers.append(status)
    if "artifact missing" in notes:
        result_blockers.append("artifact missing")
    missing_artifacts = _missing_required_artifacts(row)
    if missing_artifacts:
        result_blockers.append("missing artifacts: " + ", ".join(missing_artifacts))
    invalid_artifacts = _invalid_required_artifacts(row)
    if invalid_artifacts:
        result_blockers.append("invalid artifacts: " + ", ".join(invalid_artifacts))
    if tokens == "--":
        compute_blockers.append("missing token accounting")
    if "shared-cache fallback accounting" in notes:
        compute_blockers.append("shared-cache fallback accounting")
    if status != "complete":
        compute_blockers.append("result not complete")
    return {
        "Method": row["Method"],
        "Claim scope": row.get("Claim scope", ""),
        "Required wording": row.get("Required wording", ""),
        "Status": status,
        "result_ready": not result_blockers,
        "compute_ready": not compute_blockers,
        "result_blockers": "; ".join(result_blockers),
        "compute_blockers": "; ".join(compute_blockers),
        "token_notes": notes,
    }


def _missing_required_artifacts(row: Mapping[str, object]) -> List[str]:
    method = str(row.get("Method", ""))
    status = str(row.get("Status", "pending"))
    if status != "complete":
        return []
    artifact_value = str(row.get("source_artifact", "")).strip()
    if not artifact_value:
        return ["source artifact"]
    artifact = Path(artifact_value)
    source_dir = _artifact_run_root(method, artifact)
    required_by_method = {
        "CP-SAT": ["scheduling_structured_q2_metrics.json"],
        "EST / list scheduling heuristic": ["scheduling_structured_q2_metrics.json"],
        "Direct / CoT": ["scheduling_native_q2_samples.jsonl", "scheduling_native_q2_manifest.json"],
        "Self-Consistency": ["scheduling_native_q2_samples.jsonl", "scheduling_native_q2_manifest.json"],
        "Best-of-N": ["scheduling_native_q2_samples.jsonl", "scheduling_native_q2_manifest.json"],
        "Tree of Thoughts": ["scheduling_tot_q2_trace.jsonl", "scheduling_tot_q2_manifest.json"],
        "Target-PRM priority": [
            "scheduling_llm_pilot_metrics.json",
            "target_prm_ranker.json",
            "target_prm_trace_records.jsonl",
            "target_prm_train_pairs.jsonl",
        ],
        "PRM-guided MCTS (ReST-MCTS-style)": [
            "scheduling_llm_pilot_metrics.json",
            "target_prm_ranker.json",
            "target_prm_trace_records.jsonl",
            "target_prm_train_pairs.jsonl",
        ],
        "Solver-trained priority": ["scheduling_llm_pilot_metrics.json", "scheduling_llm_pilot_ranker.json"],
        "Solver-trained priority + certified pruning": ["scheduling_llm_pilot_metrics.json", "scheduling_llm_pilot_ranker.json"],
    }
    missing = []
    for filename in required_by_method.get(method, []):
        path = source_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


def _artifact_run_root(method: str, artifact: Path) -> Path:
    source_dir = artifact.parent
    if method in {"Direct / CoT", "Self-Consistency", "Best-of-N"} and artifact.name == "table5_native_row.json":
        if source_dir.name == _native_method_dir_name(method):
            return source_dir.parent
    return source_dir


def _invalid_required_artifacts(row: Mapping[str, object]) -> List[str]:
    method = str(row.get("Method", ""))
    status = str(row.get("Status", "pending"))
    if status != "complete":
        return []
    artifact_value = str(row.get("source_artifact", "")).strip()
    if not artifact_value:
        return []
    source_dir = _artifact_run_root(method, Path(artifact_value))
    invalid = []
    if method in {"CP-SAT", "EST / list scheduling heuristic"}:
        structured_manifest, manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_structured_q2_metrics.json")
        if manifest_error:
            invalid.append(f"scheduling_structured_q2_metrics.json {manifest_error}")
        if structured_manifest:
            if structured_manifest.get("structured_input") is not True:
                invalid.append("scheduling_structured_q2_metrics.json must declare structured_input true")
            if _as_float(structured_manifest.get("llm_calls")) != 0:
                invalid.append("structured reference must have llm_calls = 0")
            if _as_float(structured_manifest.get("tokens")) != 0:
                invalid.append("structured reference must have tokens = 0")
            alignment = structured_manifest.get("method_alignment")
            if not isinstance(alignment, Mapping) or method not in alignment:
                invalid.append(f"scheduling_structured_q2_metrics.json missing Q2 method alignment for {method}")
    if method in {"Direct / CoT", "Self-Consistency", "Best-of-N"}:
        native_manifest, manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_native_q2_manifest.json")
        if manifest_error:
            invalid.append(f"scheduling_native_q2_manifest.json {manifest_error}")
        if native_manifest:
            protocol = native_manifest.get("native_protocol")
            if not isinstance(protocol, Mapping):
                invalid.append("scheduling_native_q2_manifest.json missing native_protocol")
            else:
                methods = protocol.get("methods")
                if not isinstance(methods, Mapping) or method not in methods:
                    invalid.append(f"scheduling_native_q2_manifest.json missing native protocol for {method}")
                elif method == "Best-of-N":
                    verifier_role = str(methods[method].get("verifier_role", ""))
                    if "select minimum verified objective" not in verifier_role:
                        invalid.append("Best-of-N manifest does not disclose verifier objective selection")
        sample_path = source_dir / "scheduling_native_q2_samples.jsonl"
        if sample_path.exists() and sample_path.stat().st_size > 0:
            samples, error = _read_jsonl_for_audit(sample_path)
            if error:
                invalid.append(f"scheduling_native_q2_samples.jsonl {error}")
                return invalid
            method_samples = [sample for sample in samples if str(sample.get("Method")) == method]
            if not method_samples:
                invalid.append("scheduling_native_q2_samples.jsonl missing method rows")
            expected_count = _expected_count_from_row(row)
            if expected_count is not None:
                task_ids = {str(sample.get("task_id")) for sample in method_samples if sample.get("task_id") not in (None, "")}
                if len(task_ids) < expected_count:
                    invalid.append(f"scheduling_native_q2_samples.jsonl partial task coverage: {len(task_ids)}/{expected_count}")
    if method == "Tree of Thoughts":
        tot_manifest, manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_tot_q2_manifest.json")
        if manifest_error:
            invalid.append(f"scheduling_tot_q2_manifest.json {manifest_error}")
        if tot_manifest:
            protocol = tot_manifest.get("native_tot_protocol")
            if not isinstance(protocol, Mapping):
                invalid.append("scheduling_tot_q2_manifest.json missing native_tot_protocol")
            else:
                entrypoints = protocol.get("official_entrypoints")
                expected_entrypoints = ["method_generate", "method_evaluate", "method_select"]
                if entrypoints != expected_entrypoints:
                    invalid.append("Tree of Thoughts manifest missing official BFS entrypoints")
                if protocol.get("oracle_future_labels_exposed") is not False:
                    invalid.append("Tree of Thoughts manifest must exclude oracle future labels")
                if str(protocol.get("search")) != "breadth-first frontier search":
                    invalid.append("Tree of Thoughts manifest must declare breadth-first frontier search")
        trace_path = source_dir / "scheduling_tot_q2_trace.jsonl"
        if trace_path.exists() and trace_path.stat().st_size > 0:
            trace, error = _read_jsonl_for_audit(trace_path)
            if error:
                invalid.append(f"scheduling_tot_q2_trace.jsonl {error}")
                return invalid
            finish_events = [event for event in trace if str(event.get("event")) == "finish"]
            if not finish_events:
                invalid.append("scheduling_tot_q2_trace.jsonl missing finish event")
            expected_count = _expected_count_from_row(row)
            if expected_count is not None:
                renderings = {str(event.get("rendering_id")) for event in finish_events if event.get("rendering_id") not in (None, "")}
                if len(renderings) < expected_count:
                    invalid.append(f"scheduling_tot_q2_trace.jsonl partial finish coverage: {len(renderings)}/{expected_count}")
    if method in {"Target-PRM priority", "PRM-guided MCTS (ReST-MCTS-style)"}:
        learned_manifest, manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_llm_pilot_metrics.json")
        if manifest_error:
            invalid.append(f"scheduling_llm_pilot_metrics.json {manifest_error}")
        if learned_manifest:
            alignment = learned_manifest.get("q2_method_alignment")
            if not isinstance(alignment, Mapping) or method not in alignment:
                invalid.append(f"scheduling_llm_pilot_metrics.json missing Q2 method alignment for {method}")
            protocols = learned_manifest.get("learned_search_protocol")
            protocol_by_policy = protocols.get("policies") if isinstance(protocols, Mapping) else None
            source_policy = str(row.get("source_policy", ""))
            if not isinstance(protocol_by_policy, Mapping) or source_policy not in protocol_by_policy:
                invalid.append(f"scheduling_llm_pilot_metrics.json missing learned search protocol for {source_policy}")
            elif method == "PRM-guided MCTS (ReST-MCTS-style)" and protocol_by_policy[source_policy].get("full_iterative_self_training") is not False:
                invalid.append("ReST-MCTS-style manifest must not claim full iterative self-training")
        for filename in ("target_prm_trace_records.jsonl", "target_prm_train_pairs.jsonl"):
            path = source_dir / filename
            if path.exists() and path.stat().st_size > 0:
                rows, error = _read_jsonl_for_audit(path)
                if error:
                    invalid.append(f"{filename} {error}")
                elif not rows:
                    invalid.append(f"{filename} empty jsonl")
    if method in {"Solver-trained priority", "Solver-trained priority + certified pruning"}:
        learned_manifest, manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_llm_pilot_metrics.json")
        if manifest_error:
            invalid.append(f"scheduling_llm_pilot_metrics.json {manifest_error}")
        if learned_manifest:
            alignment = learned_manifest.get("q2_method_alignment")
            if not isinstance(alignment, Mapping) or method not in alignment:
                invalid.append(f"scheduling_llm_pilot_metrics.json missing Q2 method alignment for {method}")
            schema = learned_manifest.get("semantic_state_schema")
            if not isinstance(schema, Mapping) or schema.get("oracle_labels_in_inference_features") is not False:
                invalid.append("scheduling_llm_pilot_metrics.json must exclude oracle labels from inference features")
            if method == "Solver-trained priority":
                protocols = learned_manifest.get("learned_search_protocol")
                protocol_by_policy = protocols.get("policies") if isinstance(protocols, Mapping) else None
                if not isinstance(protocol_by_policy, Mapping) or "Solver-trained MLP" not in protocol_by_policy:
                    invalid.append("scheduling_llm_pilot_metrics.json missing solver-trained MLP protocol")
            if method == "Solver-trained priority + certified pruning":
                protocols = learned_manifest.get("learned_search_protocol")
                protocol_by_policy = protocols.get("policies") if isinstance(protocols, Mapping) else None
                cert_protocol = protocol_by_policy.get("Solver-trained MLP + certified pruning") if isinstance(protocol_by_policy, Mapping) else None
                if not isinstance(cert_protocol, Mapping):
                    invalid.append("scheduling_llm_pilot_metrics.json missing certified pruning protocol")
                else:
                    if cert_protocol.get("learned_score_authorizes_pruning") is not False:
                        invalid.append("certified pruning protocol must not let learned score authorize pruning")
                    if str(cert_protocol.get("pruning_authority")) != "verified lower-bound/feasibility logic only":
                        invalid.append("certified pruning protocol must declare verified pruning authority")
    return invalid


def _expected_count_from_row(row: Mapping[str, object]) -> Optional[int]:
    status = str(row.get("Status", ""))
    if status != "complete":
        return None
    explicit_expected_count = _as_float(row.get("expected_test_count"))
    if explicit_expected_count > 0:
        return int(explicit_expected_count)
    source_dir = _artifact_run_root(str(row.get("Method", "")), Path(str(row.get("source_artifact", ""))))
    manifest_paths = [
        source_dir / "scheduling_native_q2_manifest.json",
        source_dir / "scheduling_tot_q2_manifest.json",
    ]
    for path in manifest_paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        splits = data.get("splits")
        if isinstance(splits, Mapping):
            test_count = _as_float(splits.get("test"))
            if test_count > 0:
                return int(test_count)
    return None


def _read_json_manifest_for_audit(path: Path) -> tuple[Dict[str, object], str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}, ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"malformed json at line {exc.lineno}"
    if not isinstance(data, dict):
        return {}, "must contain a JSON object"
    return data, ""


def _pending_row(method: str, transfer_setting: str, source_dir: Optional[Path]) -> Dict[str, object]:
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "Success": "--",
        "Gap": "--",
        "LLM calls": "--",
        "Tokens": "--",
        "Wall-clock": "--",
        "Status": "pending",
        "source_artifact": "" if source_dir is None else str(source_dir),
        "source_policy": "",
        "notes": "artifact missing or native runner pending",
    }


def _contract_fields(method: str) -> Dict[str, object]:
    contract = q2_method_contract(method)
    return {
        "Claim scope": str(contract.get("claim_scope", "")),
        "Official full reproduction": bool(contract.get("official_reproduction", False)),
        "Required wording": str(contract.get("required_table_wording", "")),
    }


def _row_from_scheduling_artifacts(
    method: str,
    transfer_setting: str,
    source_policy: str,
    source_dir: Path,
    expected_test_count: int,
) -> Optional[Dict[str, object]]:
    table_path = source_dir / "scheduling_llm_pilot_table.csv"
    per_instance_path = source_dir / "scheduling_llm_pilot_per_instance.csv"
    if not table_path.exists():
        return None
    table_rows = _read_csv(table_path)
    source_row = next((row for row in table_rows if row.get("Policy") == source_policy), None)
    if source_row is None:
        return None
    per_instance_rows = [row for row in _read_csv(per_instance_path) if row.get("Policy") == source_policy]
    instance_count = len(per_instance_rows)
    complete = instance_count >= expected_test_count
    proposal_calls = _as_float(source_row.get("proposal_calls_mean")) * instance_count
    value_calls = _as_float(source_row.get("value_calls"))
    total_calls = proposal_calls + value_calls
    token_field = "on_demand_total_tokens" if source_row.get("on_demand_total_tokens") not in (None, "") else "total_tokens"
    token_source = "on-demand per-policy accounting" if token_field == "on_demand_total_tokens" else "shared-cache fallback accounting"
    wall_clock = _wall_clock_seconds(source_dir)
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "Success": _fmt_float(_as_float(source_row.get("success")), 4),
        "Gap": _fmt_float(_as_float(source_row.get("gap")), 5),
        "LLM calls": _fmt_count(total_calls) if instance_count else "--",
        "Tokens": _fmt_tokens(_as_float(source_row.get(token_field))),
        "Wall-clock": _fmt_seconds(wall_clock),
        "Status": "complete" if complete else f"partial: {instance_count}/{expected_test_count}",
        "source_artifact": str(table_path),
        "source_policy": source_policy,
        "notes": (
            f"controller-row extracted from scheduling_llm_pilot artifacts; tokens use {token_source}"
            if complete
            else "partial checkpoint; do not promote to main result"
        ),
    }


def _row_from_native_artifacts(method: str, transfer_setting: str, source_dir: Path) -> Optional[Dict[str, object]]:
    table_path = source_dir / "table5_native_row.json"
    if table_path.exists():
        data = json.loads(table_path.read_text(encoding="utf-8"))
    else:
        method_row_path = source_dir / _native_method_dir_name(method) / "table5_native_row.json"
        if method_row_path.exists():
            data = json.loads(method_row_path.read_text(encoding="utf-8"))
            table_path = method_row_path
        else:
            native_rows = _read_csv(source_dir / "scheduling_native_q2_table.csv")
            data = next((row for row in native_rows if row.get("Method") == method), None)
            if data is None:
                return None
            table_path = source_dir / "scheduling_native_q2_table.csv"
    declared_method = str(data.get("Method", method))
    if declared_method and declared_method != method:
        return None
    source_policy = str(data.get("source_policy", "native"))
    if method == "Tree of Thoughts" and source_policy != "native_tot_bfs":
        return None
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "Success": _fmt_float(_as_float(data.get("success", data.get("Success"))), 4),
        "Gap": _fmt_float(_as_float(data.get("gap", data.get("Gap"))), 5),
        "LLM calls": _fmt_count(_as_float(data.get("llm_calls"))),
        "Tokens": _fmt_tokens(_as_float(data.get("tokens"))),
        "Wall-clock": _fmt_seconds(_as_float(data.get("wall_clock_seconds"))),
        "Status": str(data.get("status", "complete")),
        "source_artifact": str(table_path),
        "source_policy": source_policy,
        "notes": str(data.get("notes", "native protocol row")),
    }


def _native_method_dir_name(method: str) -> str:
    return method.lower().replace(" / ", "_").replace("-", "_").replace(" ", "_")


def _row_from_structured_artifacts(method: str, transfer_setting: str, source_dir: Path) -> Optional[Dict[str, object]]:
    table_path = source_dir / "scheduling_structured_q2_table.csv"
    if not table_path.exists():
        return None
    data = next((row for row in _read_csv(table_path) if row.get("Method") == method), None)
    if data is None:
        return None
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "Success": "--" if data.get("success", "") == "" else _fmt_float(_as_float(data.get("success")), 4),
        "Gap": "--" if data.get("gap", "") == "" else _fmt_float(_as_float(data.get("gap")), 5),
        "LLM calls": "0",
        "Tokens": "0",
        "Wall-clock": _fmt_seconds(_as_float(data.get("wall_clock_seconds"))),
        "Status": str(data.get("status", "complete")),
        "source_artifact": str(table_path),
        "source_policy": "structured",
        "notes": str(data.get("notes", "structured input reference")),
    }


def _wall_clock_seconds(source_dir: Path) -> Optional[float]:
    rows = _read_csv(source_dir / "timing_metrics.csv")
    if not rows:
        return None
    return sum(_as_float(row.get("wall_clock_seconds")) for row in rows)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _read_jsonl_for_audit(path: Path) -> tuple[List[Dict[str, object]], str]:
    try:
        return _read_jsonl(path), ""
    except json.JSONDecodeError as exc:
        return [], f"malformed jsonl at line {exc.lineno}"


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _latex_table(rows: List[Dict[str, object]]) -> str:
    row_by_method = {str(row["Method"]): row for row in rows}
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Q2 end-to-end quality--compute comparison on real-LLM Scheduling. Task-specific optimization methods receive exact structured input and are calibration references. Target-PRM and ReST-MCTS-style rows use target-domain LLM-induced supervision; Solver prior rows use source solver-induced supervision only.}",
        "\\label{tab:q2_results}",
        "\\small",
        "\\begin{tabular}{llrrrrrl}",
        "\\toprule",
        "Method & Transfer setting & Success $\\uparrow$ & Gap $\\downarrow$ & LLM calls $\\downarrow$ & Tokens $\\downarrow$ & Wall-clock $\\downarrow$ & Status \\\\",
        "\\midrule",
    ]
    for section_idx, (section_title, methods) in enumerate(TABLE5_SECTIONS):
        if section_idx:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{_latex_escape(section_title)}}}}} \\\\")
        for method, *_rest in TABLE5_METHODS:
            if method not in methods:
                continue
            row = row_by_method[method]
            lines.append(
                " & ".join(
                    [
                        _latex_escape(str(row["Method"])),
                        _latex_escape(str(row["Transfer setting"])),
                        str(row["Success"]),
                        str(row["Gap"]),
                        str(row["LLM calls"]),
                        str(row["Tokens"]),
                        str(row["Wall-clock"]),
                        _latex_escape(str(row["Status"])),
                    ]
                )
                + " \\\\"
            )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\vspace{1mm}",
            "\\footnotesize{Best-of-$N$ is verifier-selected; ReST-MCTS-style reuses the PRM-guided MCTS search component rather than the full iterative self-training pipeline. Rows whose notes report shared-cache fallback accounting should not be used as final end-to-end compute claims.}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def _as_float(value: object) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _fmt_float(value: float, digits: int) -> str:
    return f"{value:.{digits}f}"


def _fmt_count(value: float) -> str:
    return str(int(round(value)))


def _fmt_tokens(value: float) -> str:
    if value <= 0:
        return "--"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(int(round(value)))


def _fmt_seconds(value: Optional[float]) -> str:
    if value is None or value <= 0:
        return "--"
    if value >= 3600:
        return f"{value / 3600:.2f}h"
    if value >= 60:
        return f"{value / 60:.1f}m"
    return f"{value:.1f}s"


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
    )
