from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from cgr_exp.q4_tau_source_evidence import (
    Q4_TAU_METHOD_REQUIREMENTS,
    TAU2_BENCH_SOURCE_URL,
    TAU_BENCH_SOURCE_URL,
    q4_tau_source_evidence_for_url,
    q4_tau_source_evidence_summary,
)
from cgr_exp.q4_tau_source_ref_verifier import q4_tau_source_ref_verification_summary

TAU_SOURCE_URL = TAU_BENCH_SOURCE_URL
TAU2_SOURCE_URL = TAU2_BENCH_SOURCE_URL

TAU_Q4_METHODS = [
    ("Native agent rerun", "native benchmark protocol"),
    ("Best-of-N", "native sampling plus benchmark-compatible selection"),
    ("Value-guided search", "online value-guided tool-agent search"),
    ("Target-PRM", "target tool-agent trajectories"),
    ("Solver-derived controller", "solver-derived zero-shot transfer"),
]

TAU_NATIVE_PROTOCOL_REQUIRED = {
    "stateful_tool_environment",
    "user_simulator",
    "official_task_verifier",
    "pass_k_aggregation",
    "tool_call_accounting",
    "token_accounting",
}

TAU_METHOD_PROTOCOL_REQUIRED = {
    "Native agent rerun": {"single_native_agent_rollout"},
    "Best-of-N": {"multiple_independent_native_rollouts", "benchmark_compatible_selection"},
    "Value-guided search": {"online_value_guidance", "native_tool_state_preserved"},
    "Target-PRM": {"target_tool_agent_trajectory_training", "disjoint_train_test_tasks", "process_or_step_supervision"},
    "Solver-derived controller": {"source_solver_training", "zero_shot_target_tool_transfer", "native_tool_state_preserved"},
}


def build_q4_tau_results(
    out_dir: Path,
    expected_test_count: int = 100,
    native_dir: Optional[Path] = None,
    best_of_n_dir: Optional[Path] = None,
    value_guided_dir: Optional[Path] = None,
    target_prm_dir: Optional[Path] = None,
    solver_controller_dir: Optional[Path] = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    method_dirs = {
        "Native agent rerun": native_dir,
        "Best-of-N": best_of_n_dir,
        "Value-guided search": value_guided_dir,
        "Target-PRM": target_prm_dir,
        "Solver-derived controller": solver_controller_dir,
    }
    rows = []
    for method, transfer_setting in TAU_Q4_METHODS:
        source_dir = method_dirs[method]
        row = _pending_row(method, transfer_setting, source_dir, expected_test_count)
        if source_dir is not None:
            loaded = _row_from_tau_artifacts(method, transfer_setting, source_dir, expected_test_count)
            if loaded is not None:
                row = loaded
        rows.append(row)
    csv_path = out_dir / "q4_tau_results.csv"
    tex_path = out_dir / "q4_tau_results.tex"
    manifest_path = out_dir / "q4_tau_manifest.json"
    _write_csv(csv_path, rows)
    tex_path.write_text(_latex_table(rows), encoding="utf-8")
    manifest = {
        "table": "Q4 tau-bench",
        "question": "stateful tool-agent evaluation using native tau/tau2 protocol",
        "expected_test_count": expected_test_count,
        "source_url": TAU_SOURCE_URL,
        "preferred_current_source_url": TAU2_SOURCE_URL,
        "source_evidence": q4_tau_source_evidence_summary(),
        "source_ref_verification": q4_tau_source_ref_verification_summary(),
        "method_requirements": Q4_TAU_METHOD_REQUIREMENTS,
        "strict_pending_policy": True,
        "rows": rows,
        "produced_artifacts": [csv_path.name, tex_path.name, manifest_path.name],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def audit_q4_tau_readiness(
    out_dir: Path,
    expected_test_count: int = 100,
    native_dir: Optional[Path] = None,
    best_of_n_dir: Optional[Path] = None,
    value_guided_dir: Optional[Path] = None,
    target_prm_dir: Optional[Path] = None,
    solver_controller_dir: Optional[Path] = None,
) -> Dict[str, object]:
    manifest = build_q4_tau_results(
        out_dir,
        expected_test_count=expected_test_count,
        native_dir=native_dir,
        best_of_n_dir=best_of_n_dir,
        value_guided_dir=value_guided_dir,
        target_prm_dir=target_prm_dir,
        solver_controller_dir=solver_controller_dir,
    )
    audit_rows = [_audit_row(row) for row in manifest["rows"]]
    audit_csv = out_dir / "q4_tau_readiness.csv"
    audit_json = out_dir / "q4_tau_readiness.json"
    _write_csv(audit_csv, audit_rows)
    readiness = {
        "table": "Q4 tau-bench",
        "expected_test_count": expected_test_count,
        "all_rows_result_ready": all(_as_bool(row["result_ready"]) for row in audit_rows),
        "pending_or_partial_rows": [row["Method"] for row in audit_rows if not _as_bool(row["result_ready"])],
        "audit_rows": audit_rows,
        "produced_artifacts": [*manifest["produced_artifacts"], audit_csv.name, audit_json.name],
    }
    audit_json.write_text(json.dumps(readiness, indent=2, sort_keys=True), encoding="utf-8")
    return {**manifest, "readiness": readiness}


def _pending_row(method: str, transfer_setting: str, source_dir: Optional[Path], expected_test_count: int) -> Dict[str, object]:
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "pass^1": "--",
        "pass^4": "--",
        "Violations": "--",
        "Tool calls": "--",
        "Tokens": "--",
        "Wall-clock": "--",
        "Status": "pending",
        "source_artifact": "" if source_dir is None else str(source_dir),
        "source_url": TAU_SOURCE_URL,
        "preferred_current_source_url": TAU2_SOURCE_URL,
        "source_revision": "",
        "expected_test_count": expected_test_count,
        "notes": "native tau/tau2 artifact missing; external adapter is currently an explicit stub",
    }


def _row_from_tau_artifacts(method: str, transfer_setting: str, source_dir: Path, expected_test_count: int) -> Optional[Dict[str, object]]:
    row_path = source_dir / "q4_tau_row.json"
    table_path = source_dir / "q4_tau_table.csv"
    if row_path.exists() and row_path.stat().st_size > 0:
        data = json.loads(row_path.read_text(encoding="utf-8"))
        artifact_path = row_path
    elif table_path.exists() and table_path.stat().st_size > 0:
        data = next((row for row in _read_csv(table_path) if row.get("Method") == method), None)
        if data is None:
            return None
        artifact_path = table_path
    else:
        return None
    if str(data.get("Method", method)) != method:
        return None
    instance_count = len(_task_ids(source_dir / "q4_tau_per_instance.csv", method))
    status = str(data.get("status", data.get("Status", "complete" if instance_count >= expected_test_count else f"partial: {instance_count}/{expected_test_count}")))
    manifest, _manifest_error = _read_json_manifest(source_dir / "q4_tau_manifest.json")
    source_revision = str(data.get("source_revision", manifest.get("source_revision", "")))
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "pass^1": _fmt_float(_as_float(data.get("pass^1", data.get("pass1"))), 4),
        "pass^4": _fmt_float(_as_float(data.get("pass^4", data.get("pass4"))), 4),
        "Violations": _fmt_float(_as_float(data.get("violations", data.get("Violations"))), 4),
        "Tool calls": _fmt_count(_as_float(data.get("tool_calls", data.get("Tool calls")))),
        "Tokens": _fmt_count(_as_float(data.get("tokens", data.get("Tokens")))),
        "Wall-clock": _fmt_seconds(_as_float(data.get("wall_clock_seconds", data.get("Wall-clock")))),
        "Status": status,
        "source_artifact": str(artifact_path),
        "source_url": str(data.get("source_url", TAU_SOURCE_URL)),
        "preferred_current_source_url": str(data.get("preferred_current_source_url", TAU2_SOURCE_URL)),
        "source_revision": source_revision,
        "expected_test_count": expected_test_count,
        "notes": str(data.get("notes", "native tau/tau2 protocol artifact")),
    }


def _audit_row(row: Mapping[str, object]) -> Dict[str, object]:
    method = str(row["Method"])
    status = str(row.get("Status", "pending"))
    artifact_value = str(row.get("source_artifact", "")).strip()
    blockers = []
    if status != "complete":
        blockers.append(status)
    if not artifact_value:
        blockers.append("artifact missing")
    else:
        source_dir = Path(artifact_value).parent if Path(artifact_value).suffix else Path(artifact_value)
        blockers.extend(_tau_artifact_blockers(method, source_dir, int(_as_float(row.get("expected_test_count")))))
    return {
        "Method": method,
        "source_artifact": row.get("source_artifact", ""),
        "source_url": row.get("source_url", ""),
        "preferred_current_source_url": row.get("preferred_current_source_url", ""),
        "source_revision": row.get("source_revision", ""),
        "Status": status,
        "result_ready": not blockers,
        "result_blockers": "; ".join(blockers),
        "notes": row.get("notes", ""),
    }


def _tau_artifact_blockers(method: str, source_dir: Path, expected_test_count: int) -> List[str]:
    blockers = []
    manifest, manifest_error = _read_json_manifest(source_dir / "q4_tau_manifest.json")
    if manifest_error:
        blockers.append(f"q4_tau_manifest.json {manifest_error}")
    if not manifest:
        blockers.append("q4_tau_manifest.json missing")
    else:
        if manifest.get("benchmark") not in {"tau-bench", "tau2-bench", "tau3-bench"}:
            blockers.append("q4_tau_manifest.json must declare tau-bench/tau2-bench/tau3-bench")
        if manifest.get("native_protocol_preserved") is not True:
            blockers.append("q4_tau_manifest.json must preserve native protocol")
        if manifest.get("source_url") not in {TAU_SOURCE_URL, TAU2_SOURCE_URL}:
            blockers.append("q4_tau_manifest.json source_url is not a tau source")
        blockers.extend(_source_evidence_blockers(manifest))
        protocol = manifest.get("native_protocol")
        blockers.extend(_native_protocol_blockers(protocol))
        method_protocols = manifest.get("method_protocols")
        blockers.extend(_method_protocol_blockers(method, method_protocols))
        methods = manifest.get("methods")
        if not isinstance(methods, list) or method not in {str(item) for item in methods}:
            blockers.append(f"q4_tau_manifest.json missing method {method}")
    task_ids = _task_ids(source_dir / "q4_tau_per_instance.csv", method)
    if len(task_ids) < expected_test_count:
        blockers.append(f"q4_tau_per_instance.csv partial task coverage: {len(task_ids)}/{expected_test_count}")
    row_blockers = _aggregate_row_blockers(method, source_dir)
    blockers.extend(row_blockers)
    return blockers


def _source_evidence_blockers(manifest: Mapping[str, object]) -> List[str]:
    source_url = str(manifest.get("source_url", ""))
    expected = q4_tau_source_evidence_for_url(source_url)
    if not expected:
        return ["q4_tau_manifest.json source_url has no pinned Q4 evidence"]
    source_revision = str(manifest.get("source_revision", ""))
    expected_revision = str(expected.get("source_revision", ""))
    blockers = []
    if source_revision != expected_revision:
        blockers.append("q4_tau_manifest.json source_revision must match pinned Q4 source evidence")
    artifact_evidence = manifest.get("source_evidence")
    if not isinstance(artifact_evidence, Mapping):
        blockers.append("q4_tau_manifest.json missing source_evidence")
        return blockers
    if artifact_evidence.get("source_url") != expected.get("source_url"):
        blockers.append("q4_tau_manifest.json source_evidence source_url mismatch")
    if artifact_evidence.get("source_revision") != expected_revision:
        blockers.append("q4_tau_manifest.json source_evidence source_revision mismatch")
    artifact_refs = {str(item) for item in artifact_evidence.get("official_code_refs", [])} if isinstance(artifact_evidence.get("official_code_refs"), list) else set()
    expected_refs = {str(item) for item in expected.get("official_code_refs", [])} if isinstance(expected.get("official_code_refs"), list) else set()
    if not expected_refs.issubset(artifact_refs):
        blockers.append("q4_tau_manifest.json source_evidence missing pinned official refs")
    return blockers


def _native_protocol_blockers(protocol: object) -> List[str]:
    if not isinstance(protocol, Mapping):
        return ["q4_tau_manifest.json missing native_protocol"]
    declared = {str(item) for item in protocol.get("preserved_components", [])} if isinstance(protocol.get("preserved_components"), list) else set()
    missing = sorted(TAU_NATIVE_PROTOCOL_REQUIRED - declared)
    blockers = [f"q4_tau_manifest.json native_protocol missing preserved components: {', '.join(missing)}"] if missing else []
    if protocol.get("uses_official_user_simulator") is not True:
        blockers.append("q4_tau_manifest.json native_protocol must use official user simulator")
    if protocol.get("uses_official_task_verifier") is not True:
        blockers.append("q4_tau_manifest.json native_protocol must use official task verifier")
    if protocol.get("preserves_tool_state") is not True:
        blockers.append("q4_tau_manifest.json native_protocol must preserve tool state")
    if "pass^1" not in {str(item) for item in protocol.get("reported_metrics", [])}:
        blockers.append("q4_tau_manifest.json native_protocol must report pass^1")
    if "pass^4" not in {str(item) for item in protocol.get("reported_metrics", [])}:
        blockers.append("q4_tau_manifest.json native_protocol must report pass^4")
    return blockers


def _method_protocol_blockers(method: str, method_protocols: object) -> List[str]:
    if not isinstance(method_protocols, Mapping):
        return ["q4_tau_manifest.json missing method_protocols"]
    protocol = method_protocols.get(method)
    if not isinstance(protocol, Mapping):
        return [f"q4_tau_manifest.json missing method protocol for {method}"]
    declared = {str(item) for item in protocol.get("method_primitives", [])} if isinstance(protocol.get("method_primitives"), list) else set()
    missing = sorted(TAU_METHOD_PROTOCOL_REQUIRED.get(method, set()) - declared)
    blockers = [f"q4_tau_manifest.json method protocol for {method} missing primitives: {', '.join(missing)}"] if missing else []
    if method in {"Target-PRM", "Solver-derived controller"} and protocol.get("training_eval_overlap") is not False:
        blockers.append(f"q4_tau_manifest.json method protocol for {method} must declare no training/eval overlap")
    if method == "Solver-derived controller" and protocol.get("zero_shot_transfer") is not True:
        blockers.append("q4_tau_manifest.json Solver-derived controller protocol must declare zero-shot transfer")
    if method == "Target-PRM" and protocol.get("zero_shot_transfer") is not False:
        blockers.append("q4_tau_manifest.json Target-PRM protocol must not claim zero-shot transfer")
    return blockers


def _aggregate_row_blockers(method: str, source_dir: Path) -> List[str]:
    data = _aggregate_row(source_dir, method)
    if data is None:
        return ["q4_tau_row.json or q4_tau_table.csv missing aggregate row"]
    blockers = []
    for key in ("pass^1", "pass^4", "violations", "tool_calls", "tokens", "wall_clock_seconds"):
        if key not in data and key.replace("^", "") not in data and key.title() not in data:
            blockers.append(f"aggregate row missing {key}")
    if _as_float(data.get("pass^4", data.get("pass4"))) < _as_float(data.get("pass^1", data.get("pass1"))):
        blockers.append("aggregate row pass^4 must be at least pass^1")
    return blockers


def _aggregate_row(source_dir: Path, method: str) -> Optional[Mapping[str, object]]:
    row_path = source_dir / "q4_tau_row.json"
    table_path = source_dir / "q4_tau_table.csv"
    if row_path.exists() and row_path.stat().st_size > 0:
        data = json.loads(row_path.read_text(encoding="utf-8"))
        return data if isinstance(data, Mapping) and str(data.get("Method", method)) == method else None
    if table_path.exists() and table_path.stat().st_size > 0:
        return next((row for row in _read_csv(table_path) if row.get("Method") == method), None)
    return None


def _task_ids(path: Path, method: str) -> List[str]:
    task_ids = set()
    for row in _read_csv(path):
        if str(row.get("Method", method)) != method:
            continue
        task_id = str(row.get("task_id", row.get("instance_id", "")))
        if task_id:
            task_ids.add(task_id)
    return sorted(task_ids)


def _read_json_manifest(path: Path) -> tuple[Dict[str, object], str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}, ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"malformed json at line {exc.lineno}"
    if not isinstance(data, dict):
        return {}, "must contain a JSON object"
    return data, ""


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Stateful tool-agent evaluation using the native tau/tau2 protocol. Rows remain pending until native benchmark artifacts are provided.}",
        "\\label{tab:q4_tau}",
        "\\small",
        "\\begin{tabular}{lrrrrrl}",
        "\\toprule",
        "Method & pass$^1$ $\\uparrow$ & pass$^4$ $\\uparrow$ & Violations $\\downarrow$ & Tool calls $\\downarrow$ & Tokens $\\downarrow$ & Status \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    _latex_escape(str(row["Method"])),
                    str(row["pass^1"]),
                    str(row["pass^4"]),
                    str(row["Violations"]),
                    str(row["Tool calls"]),
                    str(row["Tokens"]),
                    _latex_escape(str(row["Status"])),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])
    return "\n".join(lines)


def _fmt_float(value: float, digits: int) -> str:
    return "--" if value != value else f"{value:.{digits}f}"


def _fmt_count(value: float) -> str:
    return "--" if value != value else str(int(round(value)))


def _fmt_seconds(value: float) -> str:
    if value != value:
        return "--"
    return f"{value:.2f}s"


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
    return str(value).lower() in {"true", "1", "yes"}


def _latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_")
