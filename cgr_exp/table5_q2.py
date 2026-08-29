from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from cgr_exp.q2_method_contracts import q2_method_contract, q2_method_contracts
from cgr_exp.q2_source_evidence import Q2_SOURCE_EVIDENCE
from cgr_exp.q2_source_ref_verifier import q2_source_ref_verification_summary


TABLE5_METHODS = [
    ("CP-SAT", "Structured-input solver", None, "structured"),
    ("EST / list scheduling heuristic", "Structured-input heuristic", None, "structured"),
    ("Direct / CoT", "Frozen", None, "native"),
    ("Self-Consistency", "Frozen", None, "native"),
    ("Best-of-N", "Frozen", None, "native"),
    ("Tree of Thoughts", "Online", None, "native"),
    ("Target-PRM priority", "In-domain trained", "Target-PRM", "scheduling"),
    ("Solver-PRM priority", "Zero-shot transfer", "Solver-PRM", "scheduling"),
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
        "Source PRM control",
        {"Solver-PRM priority"},
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
    solver_prm_dir: Optional[Path] = None,
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
        "Solver-PRM priority": solver_prm_dir,
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
        "solver_prm_dir": None if solver_prm_dir is None else str(solver_prm_dir),
        "structured_dir": None if structured_dir is None else str(structured_dir),
        "rows": rows,
        "method_alignment": q2_method_contracts([row["Method"] for row in rows]),
        "source_evidence": _source_evidence_for_rows(rows),
        "source_ref_verification": q2_source_ref_verification_summary(),
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
    solver_prm_dir: Optional[Path] = None,
    direct_dir: Optional[Path] = None,
    self_consistency_dir: Optional[Path] = None,
    best_of_n_dir: Optional[Path] = None,
    rest_mcts_dir: Optional[Path] = None,
    solver_certification_dir: Optional[Path] = None,
    structured_dir: Optional[Path] = None,
    q2_local_primitives: Optional[Path] = None,
) -> Dict[str, object]:
    manifest = build_table5_q2_results(
        out_dir,
        solver_prior_dir=solver_prior_dir,
        expected_test_count=expected_test_count,
        gpt_frontier_rank_dir=gpt_frontier_rank_dir,
        target_prm_dir=target_prm_dir,
        solver_prm_dir=solver_prm_dir,
        direct_dir=direct_dir,
        self_consistency_dir=self_consistency_dir,
        best_of_n_dir=best_of_n_dir,
        rest_mcts_dir=rest_mcts_dir,
        solver_certification_dir=solver_certification_dir,
        structured_dir=structured_dir,
    )
    rows = list(manifest["rows"])
    audit_rows = [_audit_row(row) for row in rows]
    _apply_local_primitive_audit(audit_rows, q2_local_primitives)
    _apply_split_consistency_audit(rows, audit_rows, expected_test_count)
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
        "source_evidence": _source_evidence_for_rows(rows),
        "source_ref_verification": q2_source_ref_verification_summary(),
        "q2_local_primitives": None if q2_local_primitives is None else str(q2_local_primitives),
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
    source_blockers = _source_evidence_blockers_for_row(row)
    if source_blockers:
        result_blockers.append("source evidence: " + ", ".join(source_blockers))
    if tokens == "--":
        compute_blockers.append("missing token accounting")
    if "shared-cache fallback accounting" in notes:
        compute_blockers.append("shared-cache fallback accounting")
    if status != "complete":
        compute_blockers.append("result not complete")
    return {
        "Method": row["Method"],
        "Reference": row.get("Reference", ""),
        "Source URL": row.get("Source URL", ""),
        "Claim scope": row.get("Claim scope", ""),
        "Required wording": row.get("Required wording", ""),
        "source_artifact": row.get("source_artifact", ""),
        "source_policy": row.get("source_policy", ""),
        "Status": status,
        "result_ready": not result_blockers,
        "compute_ready": not compute_blockers,
        "result_blockers": "; ".join(result_blockers),
        "compute_blockers": "; ".join(compute_blockers),
        "token_notes": notes,
    }


def _source_evidence_blockers_for_row(row: Mapping[str, object]) -> List[str]:
    method = str(row.get("Method", ""))
    evidence = Q2_SOURCE_EVIDENCE.get(method)
    if not isinstance(evidence, Mapping):
        return []
    blockers: List[str] = []
    source_url = str(evidence.get("source_url", ""))
    if not source_url:
        blockers.append("missing source_url")
    if source_url != str(row.get("Source URL", "")):
        blockers.append("source_url mismatch")
    source_revision = str(evidence.get("source_revision", ""))
    source_revision_url = str(evidence.get("source_revision_url", ""))
    if source_url.startswith("https://github.com/"):
        if not source_revision:
            blockers.append("missing source_revision")
        if source_revision and len(source_revision) != 40:
            blockers.append("source_revision must be a full 40-character git SHA")
        if source_revision and source_revision not in source_revision_url:
            blockers.append("source_revision_url missing source_revision")
    if source_url.startswith("https://arxiv.org/"):
        if not source_revision:
            blockers.append("missing source_revision")
        if source_revision and not source_revision.startswith("arXiv:"):
            blockers.append("arxiv source_revision must start with arXiv:")
        if source_revision and "v" not in source_revision.rsplit(".", 1)[-1]:
            blockers.append("arxiv source_revision must include explicit version")
        if source_revision and source_revision.removeprefix("arXiv:") not in source_revision_url:
            blockers.append("source_revision_url missing arxiv version")
    source_files = evidence.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        blockers.append("missing source_files")
    official_primitives = evidence.get("official_primitives")
    if not isinstance(official_primitives, list) or not official_primitives:
        blockers.append("missing official_primitives")
    official_code_refs = evidence.get("official_code_refs")
    if not isinstance(official_code_refs, list) or not official_code_refs:
        blockers.append("missing official_code_refs")
    else:
        refs = [str(ref) for ref in official_code_refs]
        if source_url.startswith("https://github.com/") and not any("github:" in ref or "github@" in ref or "raw.githubusercontent.com" in ref for ref in refs):
            blockers.append("github-backed source missing github/raw code refs")
        if source_url.startswith("https://github.com/") and not any("::" in ref for ref in refs):
            blockers.append("github-backed source missing symbol-level code refs")
        if source_url.startswith("https://github.com/") and source_revision and not any(source_revision in ref for ref in refs):
            blockers.append("github-backed source code refs are not pinned to source_revision")
        if source_url.startswith("https://arxiv.org/") and source_revision and not any(source_revision in ref for ref in refs):
            blockers.append("paper source code refs are not pinned to source_revision")
    anchors = evidence.get("official_source_anchors")
    if not isinstance(anchors, list) or not anchors:
        blockers.append("missing official_source_anchors")
    promotion_gates = evidence.get("local_promotion_gates")
    if not isinstance(promotion_gates, list) or not promotion_gates:
        blockers.append("missing local_promotion_gates")
    return blockers


def _artifact_source_evidence_blockers(manifest: Mapping[str, object], method: str, artifact_name: str) -> List[str]:
    expected = Q2_SOURCE_EVIDENCE.get(method)
    if not isinstance(expected, Mapping):
        return []
    observed_all = manifest.get("source_evidence")
    if not isinstance(observed_all, Mapping):
        return [f"{artifact_name} missing source_evidence for {method}"]
    observed = observed_all.get(method)
    if not isinstance(observed, Mapping):
        return [f"{artifact_name} missing source_evidence for {method}"]
    blockers = []
    for key in ("source_url", "source_revision", "source_revision_url"):
        if str(observed.get(key, "")) != str(expected.get(key, "")):
            blockers.append(f"{artifact_name} source_evidence {key} mismatch for {method}")
    expected_refs = {str(ref) for ref in expected.get("official_code_refs", [])}
    observed_refs = {str(ref) for ref in observed.get("official_code_refs", [])} if isinstance(observed.get("official_code_refs"), list) else set()
    if expected_refs - observed_refs:
        blockers.append(f"{artifact_name} source_evidence missing official_code_refs for {method}")
    return blockers


def _apply_local_primitive_audit(audit_rows: List[Dict[str, object]], q2_local_primitives: Optional[Path]) -> None:
    if q2_local_primitives is None:
        return
    local_rows = _read_csv(q2_local_primitives)
    if not local_rows:
        for audit_row in audit_rows:
            _append_result_blocker(audit_row, "local primitive audit missing")
        return
    by_method = {str(row.get("Method", "")): row for row in local_rows}
    for audit_row in audit_rows:
        method = str(audit_row.get("Method", ""))
        local = by_method.get(method)
        if local is None:
            _append_result_blocker(audit_row, "local primitive audit missing method row")
            continue
        if str(local.get("local_primitives_ready", "")).lower() != "true":
            details = []
            missing_paths = str(local.get("missing_paths", "")).strip()
            missing_anchors = str(local.get("missing_anchors", "")).strip()
            missing_primitive_mappings = str(local.get("missing_primitive_mappings", "")).strip()
            missing_primitive_anchors = str(local.get("missing_primitive_anchors", "")).strip()
            if missing_paths:
                details.append(f"missing paths: {missing_paths}")
            if missing_anchors:
                details.append(f"missing anchors: {missing_anchors}")
            if missing_primitive_mappings:
                details.append(f"missing primitive mappings: {missing_primitive_mappings}")
            if missing_primitive_anchors:
                details.append(f"missing primitive anchors: {missing_primitive_anchors}")
            suffix = "; " + "; ".join(details) if details else ""
            _append_result_blocker(audit_row, "local primitive audit failed" + suffix)


def _append_result_blocker(audit_row: Dict[str, object], blocker: str) -> None:
    existing = str(audit_row.get("result_blockers", "")).strip()
    audit_row["result_blockers"] = "; ".join(part for part in [existing, blocker] if part)
    audit_row["result_ready"] = False
    audit_row["compute_ready"] = False
    compute_existing = str(audit_row.get("compute_blockers", "")).strip()
    audit_row["compute_blockers"] = "; ".join(part for part in [compute_existing, "result not complete"] if part)


def _apply_split_consistency_audit(rows: List[Mapping[str, object]], audit_rows: List[Dict[str, object]], expected_test_count: int) -> None:
    signatures: Dict[str, tuple[str, ...]] = {}
    for row in rows:
        if str(row.get("Status", "pending")) != "complete":
            continue
        task_ids = _task_ids_for_row(row)
        if not task_ids:
            continue
        method = str(row["Method"])
        signatures[method] = tuple(task_ids)
    if not signatures:
        return
    reference_method, reference_signature = next(iter(signatures.items()))
    has_split_mismatch = len(set(signatures.values())) > 1
    for audit_row in audit_rows:
        method = str(audit_row["Method"])
        signature = signatures.get(method)
        if signature is None:
            continue
        blockers = []
        if len(signature) < expected_test_count:
            blockers.append(f"split task coverage: {len(signature)}/{expected_test_count}")
        if has_split_mismatch and signature == reference_signature:
            blockers.append("split mismatch across complete rows")
        elif signature != reference_signature:
            blockers.append(f"split mismatch vs {reference_method}")
        if blockers:
            existing = str(audit_row.get("result_blockers", ""))
            audit_row["result_blockers"] = "; ".join(part for part in [existing, *blockers] if part)
            audit_row["result_ready"] = False
            audit_row["compute_ready"] = False
            compute_existing = str(audit_row.get("compute_blockers", ""))
            audit_row["compute_blockers"] = "; ".join(part for part in [compute_existing, "result not complete"] if part)


def _task_ids_for_row(row: Mapping[str, object]) -> List[str]:
    method = str(row.get("Method", ""))
    artifact_value = str(row.get("source_artifact", "")).strip()
    if not artifact_value:
        return []
    source_dir = _artifact_run_root(method, Path(artifact_value))
    if method in {"CP-SAT", "EST / list scheduling heuristic"}:
        return _unique_task_ids(_read_csv(source_dir / "scheduling_structured_q2_per_instance.csv"), method_key="Method", method_value=method)
    if method in {"Direct / CoT", "Self-Consistency", "Best-of-N"}:
        return _unique_task_ids(_read_csv(source_dir / "scheduling_native_q2_per_instance.csv"), method_key="Method", method_value=method)
    if method == "Tree of Thoughts":
        return _unique_task_ids(_read_csv(source_dir / "scheduling_tot_q2_per_instance.csv"))
    source_policy = str(row.get("source_policy", ""))
    return _unique_task_ids(_read_csv(source_dir / "scheduling_llm_pilot_per_instance.csv"), method_key="Policy", method_value=source_policy)


def _target_prm_eval_overlap(source_dir: Path, row: Mapping[str, object]) -> List[str]:
    records, error = _read_jsonl_for_audit(source_dir / "target_prm_trace_records.jsonl")
    if error or not records:
        return []
    train_ids = {str(record.get("task_id")) for record in records if record.get("task_id") not in (None, "")}
    eval_ids = set(_task_ids_for_row(row))
    if not eval_ids:
        metrics, _manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_llm_pilot_metrics.json")
        split_manifest = metrics.get("split_manifest") if isinstance(metrics, Mapping) else None
        if isinstance(split_manifest, Mapping):
            eval_ids = {str(task_id) for task_id in split_manifest.get("test", [])}
    return sorted(train_ids & eval_ids)


def _unique_task_ids(rows: List[Mapping[str, object]], method_key: str = "", method_value: str = "") -> List[str]:
    task_ids = set()
    for row in rows:
        if method_key and str(row.get(method_key)) != method_value:
            continue
        task_id = str(row.get("task_id", row.get("instance_id", "")))
        if task_id:
            task_ids.add(task_id)
    return sorted(task_ids)


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
            "target_prm_label_manifest.json",
        ],
        "Solver-PRM priority": [
            "scheduling_llm_pilot_metrics.json",
            "solver_prm_ranker.json",
            "solver_prm_trace_records.jsonl",
            "solver_prm_train_pairs.jsonl",
            "solver_prm_label_manifest.json",
        ],
        "PRM-guided MCTS (ReST-MCTS-style)": [
            "scheduling_llm_pilot_metrics.json",
            "target_prm_ranker.json",
            "target_prm_trace_records.jsonl",
            "target_prm_train_pairs.jsonl",
            "target_prm_label_manifest.json",
        ],
        "Solver-trained priority": [
            "scheduling_llm_pilot_metrics.json",
            "scheduling_llm_pilot_ranker.json",
            "split_manifest.json",
            "checkpoint_metadata.json",
            "feature_schema.json",
        ],
        "Solver-trained priority + certified pruning": [
            "scheduling_llm_pilot_metrics.json",
            "scheduling_llm_pilot_ranker.json",
            "split_manifest.json",
            "checkpoint_metadata.json",
            "feature_schema.json",
        ],
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
            else:
                invalid.extend(_method_alignment_blockers(alignment, method, "scheduling_structured_q2_metrics.json"))
            invalid.extend(_artifact_source_evidence_blockers(structured_manifest, method, "scheduling_structured_q2_metrics.json"))
            table_rows = _read_csv(source_dir / "scheduling_structured_q2_table.csv")
            structured_row = next((candidate for candidate in table_rows if candidate.get("Method") == method), {})
            invalid.extend(_structured_runtime_evidence_blockers(method, structured_row))
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
                else:
                    invalid.extend(_native_protocol_blockers(method, methods[method]))
            alignment = native_manifest.get("method_alignment")
            if not isinstance(alignment, Mapping) or method not in alignment:
                invalid.append(f"scheduling_native_q2_manifest.json missing Q2 method alignment for {method}")
            else:
                invalid.extend(_method_alignment_blockers(alignment, method, "scheduling_native_q2_manifest.json"))
            invalid.extend(_artifact_source_evidence_blockers(native_manifest, method, "scheduling_native_q2_manifest.json"))
            native_table_rows = _read_csv(source_dir / "scheduling_native_q2_table.csv")
            native_table_row = next((item for item in native_table_rows if str(item.get("Method")) == method), row)
            invalid.extend(_native_runtime_count_blockers(method, native_table_row, methods.get(method, {}) if isinstance(methods, Mapping) else {}, row))
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
                native_manifest, _manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_native_q2_manifest.json")
                protocol = native_manifest.get("native_protocol") if isinstance(native_manifest, Mapping) else None
                methods = protocol.get("methods") if isinstance(protocol, Mapping) else None
                method_protocol = methods.get(method) if isinstance(methods, Mapping) else {}
                invalid.extend(_native_sample_count_blockers(method, method_samples, task_ids, method_protocol))
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
                method_evaluate = str(protocol.get("method_evaluate", ""))
                if "value" not in method_evaluate and "vote" not in method_evaluate:
                    invalid.append("Tree of Thoughts manifest method_evaluate must declare value/vote frontier evaluation")
                if "greedy top-b" not in str(protocol.get("method_select", "")):
                    invalid.append("Tree of Thoughts manifest method_select must declare greedy top-b selection")
                if _as_float(protocol.get("n_select_sample")) <= 0:
                    invalid.append("Tree of Thoughts manifest must declare positive n_select_sample")
                if _as_float(protocol.get("n_evaluate_sample")) <= 0:
                    invalid.append("Tree of Thoughts manifest must declare positive n_evaluate_sample")
            alignment = tot_manifest.get("method_alignment")
            if not isinstance(alignment, Mapping) or method not in alignment:
                invalid.append(f"scheduling_tot_q2_manifest.json missing Q2 method alignment for {method}")
            else:
                invalid.extend(_method_alignment_blockers(alignment, method, "scheduling_tot_q2_manifest.json"))
            invalid.extend(_artifact_source_evidence_blockers(tot_manifest, method, "scheduling_tot_q2_manifest.json"))
        table_rows = _read_csv(source_dir / "scheduling_tot_q2_table.csv")
        source_policy_row = next((item for item in table_rows if str(item.get("Method")) == method), table_rows[0] if table_rows else row)
        if _as_float(source_policy_row.get("tot_generate_events")) <= 0:
            invalid.append("Tree of Thoughts table row must record positive generate events")
        if _as_float(source_policy_row.get("tot_evaluate_events")) <= 0:
            invalid.append("Tree of Thoughts table row must record positive evaluate events")
        if _as_float(source_policy_row.get("tot_select_events")) <= 0:
            invalid.append("Tree of Thoughts table row must record positive select events")
        if _as_float(source_policy_row.get("tot_value_sample_count")) <= 0:
            invalid.append("Tree of Thoughts table row must record positive value sample count")
        trace_path = source_dir / "scheduling_tot_q2_trace.jsonl"
        if trace_path.exists() and trace_path.stat().st_size > 0:
            trace, error = _read_jsonl_for_audit(trace_path)
            if error:
                invalid.append(f"scheduling_tot_q2_trace.jsonl {error}")
                return invalid
            finish_events = [event for event in trace if str(event.get("event")) == "finish"]
            if not finish_events:
                invalid.append("scheduling_tot_q2_trace.jsonl missing finish event")
            invalid.extend(_tot_trace_protocol_blockers(trace))
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
            else:
                invalid.extend(_method_alignment_blockers(alignment, method, "scheduling_llm_pilot_metrics.json"))
            invalid.extend(_artifact_source_evidence_blockers(learned_manifest, method, "scheduling_llm_pilot_metrics.json"))
            protocols = learned_manifest.get("learned_search_protocol")
            protocol_by_policy = protocols.get("policies") if isinstance(protocols, Mapping) else None
            source_policy = str(row.get("source_policy", ""))
            if not isinstance(protocol_by_policy, Mapping) or source_policy not in protocol_by_policy:
                invalid.append(f"scheduling_llm_pilot_metrics.json missing learned search protocol for {source_policy}")
            else:
                policy_protocol = protocol_by_policy[source_policy]
                if method == "Target-PRM priority":
                    if policy_protocol.get("zero_shot_transfer") is not False:
                        invalid.append("Target-PRM protocol must not claim zero-shot transfer")
                    if "target LLM-induced" not in str(policy_protocol.get("training_distribution", "")):
                        invalid.append("Target-PRM protocol must declare target LLM-induced training distribution")
                if method == "PRM-guided MCTS (ReST-MCTS-style)":
                    if policy_protocol.get("full_iterative_self_training") is not False:
                        invalid.append("ReST-MCTS-style manifest must not claim full iterative self-training")
                    if "target LLM-induced" not in str(policy_protocol.get("training_distribution", "")):
                        invalid.append("ReST-MCTS-style protocol must use the target PRM training distribution")
                    declared_components = policy_protocol.get("mcts_components")
                    required_components = {"selection", "expansion", "simulation/evaluation", "backpropagation"}
                    if not isinstance(declared_components, list) or required_components - {str(item) for item in declared_components}:
                        invalid.append("ReST-MCTS-style protocol must declare MCTS selection, expansion, simulation/evaluation, and backpropagation")
                    if "PRM" not in str(policy_protocol.get("process_reward_guidance", "")):
                        invalid.append("ReST-MCTS-style protocol must declare PRM process reward guidance")
                    if "PRM_prior" not in str(policy_protocol.get("selection_score", "")):
                        invalid.append("ReST-MCTS-style protocol must declare PRM-guided PUCT/UCB selection score")
                    table_rows = _read_csv(source_dir / "scheduling_llm_pilot_table.csv")
                    policy_name = str(row.get("source_policy", ""))
                    source_policy_row = next((item for item in table_rows if str(item.get("Policy")) == policy_name), {})
                    if _as_float(source_policy_row.get("mcts_selection_calls")) <= 0:
                        invalid.append("ReST-MCTS-style table row must record positive MCTS selection calls")
                    if _as_float(source_policy_row.get("mcts_expansion_observations")) <= 0:
                        invalid.append("ReST-MCTS-style table row must record positive MCTS expansion observations")
                    if _as_float(source_policy_row.get("mcts_prior_evaluations")) <= 0:
                        invalid.append("ReST-MCTS-style table row must record positive PRM prior evaluations")
                    if _as_float(source_policy_row.get("mcts_leaf_value_evaluations")) <= 0:
                        invalid.append("ReST-MCTS-style table row must record positive MCTS simulation/evaluation calls")
                    if _as_float(source_policy_row.get("mcts_backup_updates")) <= 0:
                        invalid.append("ReST-MCTS-style table row must record positive MCTS backpropagation updates")
                    diagnostics_raw = str(source_policy_row.get("policy_diagnostics", ""))
                    if "PRM_prior" not in diagnostics_raw:
                        invalid.append("ReST-MCTS-style diagnostics must record the PRM-guided PUCT/UCB selection score")
        label_manifest, label_error = _read_json_manifest_for_audit(source_dir / "target_prm_label_manifest.json")
        if label_error:
            invalid.append(f"target_prm_label_manifest.json {label_error}")
        if label_manifest:
            if str(label_manifest.get("supervision")) != "target_process_frontier_ranking":
                invalid.append("target_prm_label_manifest.json must declare target_process_frontier_ranking supervision")
            if str(label_manifest.get("label_source")) != "exact_residual_scheduling_oracle_on_target_llm_states":
                invalid.append("target_prm_label_manifest.json must declare exact residual labels on target LLM states")
            if label_manifest.get("terminal_outcome_only") is not False:
                invalid.append("target_prm_label_manifest.json must not be terminal outcome only")
            if label_manifest.get("oracle_labels_in_inference_features") is not False:
                invalid.append("target_prm_label_manifest.json must exclude oracle labels from inference features")
            if _as_float(label_manifest.get("target_pair_count")) <= 0:
                invalid.append("target_prm_label_manifest.json must record positive target pair count")
            if _as_float(label_manifest.get("process_label_record_count")) <= 0:
                invalid.append("target_prm_label_manifest.json must record positive process label count")
            if label_manifest.get("uses_intermediate_states") is not True:
                invalid.append("target_prm_label_manifest.json must declare intermediate state supervision")
            if str(label_manifest.get("pair_sampling_version")) != "process_first_v2":
                invalid.append("target_prm_label_manifest.json must come from process_first_v2 pair sampling")
            if "process-first" not in str(label_manifest.get("pair_sampling", "")):
                invalid.append("target_prm_label_manifest.json must declare process-first pair sampling")
            if _as_float(label_manifest.get("finite_best_descendant_labels")) <= 0:
                invalid.append("target_prm_label_manifest.json must record finite best-descendant labels")
            if _as_float(label_manifest.get("max_record_depth")) <= 0:
                invalid.append("target_prm_label_manifest.json must include non-root intermediate states")
            if _as_float(label_manifest.get("frontier_parent_group_count")) <= 0:
                invalid.append("target_prm_label_manifest.json must record frontier parent groups")
            if _as_float(label_manifest.get("sibling_pair_candidate_count")) <= 0:
                invalid.append("target_prm_label_manifest.json must record process sibling pair candidates")
        for filename in ("target_prm_trace_records.jsonl", "target_prm_train_pairs.jsonl"):
            path = source_dir / filename
            if path.exists() and path.stat().st_size > 0:
                rows, error = _read_jsonl_for_audit(path)
                if error:
                    invalid.append(f"{filename} {error}")
                elif not rows:
                    invalid.append(f"{filename} empty jsonl")
        overlap = _target_prm_eval_overlap(source_dir, row)
        if overlap:
            invalid.append("target_prm_trace_records.jsonl overlaps held-out evaluation tasks: " + ", ".join(overlap[:5]))
    if method == "Solver-PRM priority":
        invalid.extend(_solver_prior_audit_blockers(source_dir))
        learned_manifest, manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_llm_pilot_metrics.json")
        if manifest_error:
            invalid.append(f"scheduling_llm_pilot_metrics.json {manifest_error}")
        if learned_manifest:
            alignment = learned_manifest.get("q2_method_alignment")
            if not isinstance(alignment, Mapping) or method not in alignment:
                invalid.append(f"scheduling_llm_pilot_metrics.json missing Q2 method alignment for {method}")
            else:
                invalid.extend(_method_alignment_blockers(alignment, method, "scheduling_llm_pilot_metrics.json"))
            schema = learned_manifest.get("semantic_state_schema")
            if not isinstance(schema, Mapping) or schema.get("oracle_labels_in_inference_features") is not False:
                invalid.append("scheduling_llm_pilot_metrics.json must exclude oracle labels from inference features")
            protocols = learned_manifest.get("learned_search_protocol")
            protocol_by_policy = protocols.get("policies") if isinstance(protocols, Mapping) else None
            solver_prm_protocol = protocol_by_policy.get("Solver-PRM") if isinstance(protocol_by_policy, Mapping) else None
            if not isinstance(solver_prm_protocol, Mapping):
                invalid.append("scheduling_llm_pilot_metrics.json missing Solver-PRM protocol")
            else:
                if solver_prm_protocol.get("zero_shot_transfer") is not True:
                    invalid.append("Solver-PRM protocol must declare zero-shot transfer")
                if str(solver_prm_protocol.get("training_distribution")) != "solver-induced Scheduling source trajectories only":
                    invalid.append("Solver-PRM protocol must declare source-only solver training")
                if solver_prm_protocol.get("target_llm_trajectory_training") is not False:
                    invalid.append("Solver-PRM protocol must declare no target LLM trajectory training")
        label_manifest, label_error = _read_json_manifest_for_audit(source_dir / "solver_prm_label_manifest.json")
        if label_error:
            invalid.append(f"solver_prm_label_manifest.json {label_error}")
        if label_manifest:
            if str(label_manifest.get("supervision")) != "solver_process_frontier_ranking":
                invalid.append("solver_prm_label_manifest.json must declare solver_process_frontier_ranking supervision")
            if str(label_manifest.get("label_source")) != "exact_residual_scheduling_oracle_on_solver_states":
                invalid.append("solver_prm_label_manifest.json must declare exact residual labels on solver states")
            if label_manifest.get("target_llm_trajectory_training") is not False:
                invalid.append("solver_prm_label_manifest.json must declare no target LLM trajectory training")
            if label_manifest.get("zero_shot_transfer") is not True:
                invalid.append("solver_prm_label_manifest.json must declare zero-shot transfer")
            if label_manifest.get("terminal_outcome_only") is not False:
                invalid.append("solver_prm_label_manifest.json must not be terminal outcome only")
            if label_manifest.get("oracle_labels_in_inference_features") is not False:
                invalid.append("solver_prm_label_manifest.json must exclude oracle labels from inference features")
            if _as_float(label_manifest.get("target_pair_count")) <= 0:
                invalid.append("solver_prm_label_manifest.json must record positive pair count")
            if _as_float(label_manifest.get("process_label_record_count")) <= 0:
                invalid.append("solver_prm_label_manifest.json must record positive process label count")
        for filename in ("solver_prm_trace_records.jsonl", "solver_prm_train_pairs.jsonl"):
            path = source_dir / filename
            if path.exists() and path.stat().st_size > 0:
                rows, error = _read_jsonl_for_audit(path)
                if error:
                    invalid.append(f"{filename} {error}")
                elif not rows:
                    invalid.append(f"{filename} empty jsonl")
    if method in {"Solver-trained priority", "Solver-trained priority + certified pruning"}:
        invalid.extend(_solver_prior_audit_blockers(source_dir))
        learned_manifest, manifest_error = _read_json_manifest_for_audit(source_dir / "scheduling_llm_pilot_metrics.json")
        if manifest_error:
            invalid.append(f"scheduling_llm_pilot_metrics.json {manifest_error}")
        if learned_manifest:
            alignment = learned_manifest.get("q2_method_alignment")
            if not isinstance(alignment, Mapping) or method not in alignment:
                invalid.append(f"scheduling_llm_pilot_metrics.json missing Q2 method alignment for {method}")
            else:
                invalid.extend(_method_alignment_blockers(alignment, method, "scheduling_llm_pilot_metrics.json"))
            schema = learned_manifest.get("semantic_state_schema")
            if not isinstance(schema, Mapping) or schema.get("oracle_labels_in_inference_features") is not False:
                invalid.append("scheduling_llm_pilot_metrics.json must exclude oracle labels from inference features")
            if method == "Solver-trained priority":
                protocols = learned_manifest.get("learned_search_protocol")
                protocol_by_policy = protocols.get("policies") if isinstance(protocols, Mapping) else None
                solver_protocol = protocol_by_policy.get("Solver-trained MLP") if isinstance(protocol_by_policy, Mapping) else None
                if not isinstance(solver_protocol, Mapping):
                    invalid.append("scheduling_llm_pilot_metrics.json missing solver-trained MLP protocol")
                else:
                    if solver_protocol.get("zero_shot_transfer") is not True:
                        invalid.append("Solver-trained MLP protocol must declare zero-shot transfer")
                    if str(solver_protocol.get("training_distribution")) != "solver-induced Scheduling source trajectories only":
                        invalid.append("Solver-trained MLP protocol must declare source-only solver training")
            if method == "Solver-trained priority + certified pruning":
                protocols = learned_manifest.get("learned_search_protocol")
                protocol_by_policy = protocols.get("policies") if isinstance(protocols, Mapping) else None
                cert_protocol = protocol_by_policy.get("Solver-trained MLP + certified pruning") if isinstance(protocol_by_policy, Mapping) else None
                if not isinstance(cert_protocol, Mapping):
                    invalid.append("scheduling_llm_pilot_metrics.json missing certified pruning protocol")
                else:
                    if cert_protocol.get("zero_shot_transfer") is not True:
                        invalid.append("certified pruning protocol must declare zero-shot transfer")
                    if str(cert_protocol.get("training_distribution")) != "same solver-induced source controller as Solver-trained MLP":
                        invalid.append("certified pruning protocol must reuse source-only solver controller")
                    if cert_protocol.get("learned_score_authorizes_pruning") is not False:
                        invalid.append("certified pruning protocol must not let learned score authorize pruning")
                    if str(cert_protocol.get("pruning_authority")) != "verified lower-bound/feasibility logic only":
                        invalid.append("certified pruning protocol must declare verified pruning authority")
    return invalid


def _method_alignment_blockers(alignment: Mapping[str, object], method: str, artifact_name: str) -> List[str]:
    method_alignment = alignment.get(method)
    if not isinstance(method_alignment, Mapping):
        return [f"{artifact_name} missing Q2 method alignment for {method}"]
    blockers: List[str] = []
    contract = q2_method_contract(method)
    for key in ("claim_scope", "source_url", "required_table_wording"):
        expected = str(contract.get(key, ""))
        observed = str(method_alignment.get(key, ""))
        if observed != expected:
            blockers.append(f"{artifact_name} method alignment {key} mismatch for {method}: expected {expected!r}, observed {observed!r}")
    if bool(method_alignment.get("official_reproduction", False)) != bool(contract.get("official_reproduction", False)):
        blockers.append(f"{artifact_name} method alignment official_reproduction mismatch for {method}")
    declared = method_alignment.get("source_primitives")
    if not isinstance(declared, list):
        blockers.append(f"{artifact_name} missing source_primitives for {method}")
        declared = []
    declared_set = {str(item) for item in declared}
    required = {str(item) for item in contract.get("source_primitives", [])}
    missing = sorted(required - declared_set)
    if missing:
        blockers.append(f"{artifact_name} source_primitives incomplete for {method}: " + ", ".join(missing))
    declared_non_claims = method_alignment.get("non_claims")
    if not isinstance(declared_non_claims, list):
        blockers.append(f"{artifact_name} missing non_claims for {method}")
    else:
        required_non_claims = {str(item) for item in contract.get("non_claims", [])}
        missing_non_claims = sorted(required_non_claims - {str(item) for item in declared_non_claims})
        if missing_non_claims:
            blockers.append(f"{artifact_name} non_claims incomplete for {method}: " + ", ".join(missing_non_claims))
    return blockers


def _structured_runtime_evidence_blockers(method: str, row: Mapping[str, object]) -> List[str]:
    blockers: List[str] = []
    if not row:
        return [f"scheduling_structured_q2_table.csv missing structured runtime row for {method}"]
    if method == "CP-SAT":
        if str(row.get("structured_runtime_solver", "")) != "ortools_cp_sat":
            blockers.append("CP-SAT structured row must declare structured_runtime_solver=ortools_cp_sat")
        for key in (
            "structured_runtime_integer_variables",
            "structured_runtime_interval_variables",
            "structured_runtime_exactly_one_constraints",
            "structured_runtime_no_overlap_constraints",
            "structured_runtime_objective_terms",
        ):
            if _as_float(row.get(key)) <= 0:
                blockers.append(f"CP-SAT structured row missing positive {key}")
    if method == "EST / list scheduling heuristic":
        if str(row.get("structured_runtime_heuristic", "")) != "earliest_start_list_scheduling":
            blockers.append("EST structured row must declare structured_runtime_heuristic=earliest_start_list_scheduling")
        for key in (
            "structured_runtime_dispatch_steps",
            "structured_runtime_ready_list_selections",
            "structured_runtime_candidate_evaluations",
        ):
            if _as_float(row.get(key)) <= 0:
                blockers.append(f"EST structured row missing positive {key}")
        if not str(row.get("structured_runtime_tie_break_rule", "")).strip():
            blockers.append("EST structured row missing structured_runtime_tie_break_rule")
    return blockers


def _native_protocol_blockers(method: str, method_protocol: object) -> List[str]:
    if not isinstance(method_protocol, Mapping):
        return [f"scheduling_native_q2_manifest.json native protocol for {method} must be an object"]
    invalid = []
    expected_selection = {
        "Direct / CoT": "first_valid",
        "Self-Consistency": "majority_job_order_then_first_valid",
        "Best-of-N": "best_verified_objective",
    }[method]
    if str(method_protocol.get("selection_rule", "")) != expected_selection:
        invalid.append(f"{method} manifest must declare selection_rule={expected_selection}")
    sample_count = int(max(0, _as_float(method_protocol.get("sample_count"))))
    if method == "Direct / CoT" and sample_count != 1:
        invalid.append("Direct / CoT manifest must declare exactly one reasoning sample")
    if method in {"Self-Consistency", "Best-of-N"} and sample_count <= 1:
        invalid.append(f"{method} manifest must declare multiple independent samples")
    if method == "Best-of-N":
        verifier_role = str(method_protocol.get("verifier_role", ""))
        if "select minimum verified objective" not in verifier_role:
            invalid.append("Best-of-N manifest does not disclose verifier objective selection")
    if method_protocol.get("oracle_future_labels_exposed") is not False:
        invalid.append(f"{method} manifest must exclude oracle future labels")
    if method_protocol.get("intermediate_reasoning_requested") is not True:
        invalid.append(f"{method} manifest must declare intermediate reasoning requested")
    if "reasoning" not in str(method_protocol.get("reasoning_location", "")):
        invalid.append(f"{method} manifest must declare reasoning audit location")
    return invalid


def _native_sample_count_blockers(
    method: str,
    method_samples: List[Mapping[str, object]],
    task_ids: set[str],
    method_protocol: object,
) -> List[str]:
    if not task_ids:
        return []
    if not isinstance(method_protocol, Mapping):
        return []
    declared = int(max(0, _as_float(method_protocol.get("sample_count"))))
    expected = 1 if method == "Direct / CoT" else declared
    if expected <= 0:
        return []
    counts = {task_id: 0 for task_id in task_ids}
    for sample in method_samples:
        task_id = str(sample.get("task_id", ""))
        if task_id in counts:
            counts[task_id] += 1
    min_count = min(counts.values()) if counts else 0
    max_count = max(counts.values()) if counts else 0
    if method == "Direct / CoT" and (min_count != 1 or max_count != 1):
        return [f"scheduling_native_q2_samples.jsonl Direct / CoT must have exactly one sample per task; observed range {min_count}-{max_count}"]
    if method in {"Self-Consistency", "Best-of-N"} and min_count < expected:
        return [f"scheduling_native_q2_samples.jsonl {method} sample coverage per task below declared sample_count: {min_count}/{expected}"]
    missing_reasoning_audit = [
        sample
        for sample in method_samples
        if "reasoning_present" not in sample or "reasoning_step_count" not in sample
    ]
    if missing_reasoning_audit:
        return [f"scheduling_native_q2_samples.jsonl {method} missing reasoning audit fields"]
    return []


def _native_runtime_count_blockers(
    method: str,
    table_row: Mapping[str, object],
    method_protocol: object,
    readiness_row: Mapping[str, object],
) -> List[str]:
    if not isinstance(method_protocol, Mapping):
        return []
    expected_tasks = _expected_count_from_row(readiness_row)
    if expected_tasks is None:
        return []
    declared = int(max(0, _as_float(method_protocol.get("sample_count"))))
    expected_samples = expected_tasks if method == "Direct / CoT" else expected_tasks * declared
    observed_samples = int(max(0, _as_float(table_row.get("native_runtime_sample_count", table_row.get("total_samples")))))
    observed_selections = int(max(0, _as_float(table_row.get("native_runtime_selection_events"))))
    invalid = []
    if observed_samples < expected_samples:
        invalid.append(f"{method} table row must record native runtime samples: {observed_samples}/{expected_samples}")
    if observed_selections < expected_tasks:
        invalid.append(f"{method} table row must record one native selection event per task: {observed_selections}/{expected_tasks}")
    if "native_runtime_reasoning_samples" not in table_row:
        invalid.append(f"{method} table row must record native reasoning sample audit count")
    if method == "Best-of-N":
        valid_samples = int(max(0, _as_float(table_row.get("native_runtime_valid_samples", table_row.get("valid_samples")))))
        if valid_samples <= 0:
            invalid.append("Best-of-N table row must record verifier-observed valid samples")
    return invalid


def _tot_trace_protocol_blockers(trace: List[Mapping[str, object]]) -> List[str]:
    events = {str(event.get("event")) for event in trace}
    invalid = []
    if "generate" not in events:
        invalid.append("scheduling_tot_q2_trace.jsonl missing generate events")
    has_evaluate = "evaluate_select" in events or "evaluate" in events
    has_select = "evaluate_select" in events or "select" in events
    if not has_evaluate:
        invalid.append("scheduling_tot_q2_trace.jsonl missing evaluate events")
    if not has_select:
        invalid.append("scheduling_tot_q2_trace.jsonl missing select events")
    evaluate_events = [event for event in trace if str(event.get("event")) in {"evaluate_select", "evaluate"}]
    if evaluate_events and not any(isinstance(event.get("value_samples"), list) and event.get("value_samples") for event in evaluate_events):
        invalid.append("scheduling_tot_q2_trace.jsonl missing value sample records")
    for event in evaluate_events:
        candidate_ids = event.get("candidate_ids")
        selected_ids = event.get("selected_ids")
        averaged_scores = event.get("averaged_scores")
        value_samples = event.get("value_samples")
        if not isinstance(candidate_ids, list) or not candidate_ids:
            invalid.append("scheduling_tot_q2_trace.jsonl evaluate event missing candidate_ids")
            continue
        if not isinstance(selected_ids, list) or not selected_ids:
            invalid.append("scheduling_tot_q2_trace.jsonl evaluate event missing selected_ids")
        if not isinstance(averaged_scores, list) or len(averaged_scores) != len(candidate_ids):
            invalid.append("scheduling_tot_q2_trace.jsonl evaluate event averaged_scores must match candidate_ids")
        if isinstance(value_samples, list):
            for sample in value_samples:
                scores = sample.get("scores") if isinstance(sample, Mapping) else None
                if not isinstance(scores, list) or len(scores) != len(candidate_ids):
                    invalid.append("scheduling_tot_q2_trace.jsonl value sample scores must match candidate_ids")
                    break
    return invalid


def _solver_prior_audit_blockers(source_dir: Path) -> List[str]:
    blockers: List[str] = []
    split_manifest, split_error = _read_json_manifest_for_audit(source_dir / "split_manifest.json")
    if split_error:
        blockers.append(f"split_manifest.json {split_error}")
    if split_manifest:
        train_ids = {str(task_id) for task_id in split_manifest.get("train", [])}
        validation_ids = {str(task_id) for task_id in split_manifest.get("validation", [])}
        test_ids = {str(task_id) for task_id in split_manifest.get("test", [])}
        if train_ids & test_ids:
            blockers.append("split_manifest.json train/test overlap")
        if validation_ids & test_ids:
            blockers.append("split_manifest.json validation/test overlap")
    checkpoint, checkpoint_error = _read_json_manifest_for_audit(source_dir / "checkpoint_metadata.json")
    if checkpoint_error:
        blockers.append(f"checkpoint_metadata.json {checkpoint_error}")
    if checkpoint:
        if str(checkpoint.get("source_training_split")) != "train":
            blockers.append("checkpoint_metadata.json source_training_split must be train")
        training_ids = {str(task_id) for task_id in checkpoint.get("training_instance_ids", [])}
        test_ids = {str(task_id) for task_id in split_manifest.get("test", [])} if split_manifest else set()
        if training_ids & test_ids:
            blockers.append("checkpoint_metadata.json training_instance_ids overlap held-out test")
    feature_schema, feature_error = _read_json_manifest_for_audit(source_dir / "feature_schema.json")
    if feature_error:
        blockers.append(f"feature_schema.json {feature_error}")
    if feature_schema and feature_schema.get("oracle_labels_in_inference_features") is not False:
        blockers.append("feature_schema.json must exclude oracle labels from inference features")
    return blockers


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
        "Reference": str(contract.get("reference", "")),
        "Source URL": str(contract.get("source_url", "")),
        "Claim scope": str(contract.get("claim_scope", "")),
        "Official full reproduction": bool(contract.get("official_reproduction", False)),
        "Required wording": str(contract.get("required_table_wording", "")),
    }


def _source_evidence_for_rows(rows: Iterable[Mapping[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(row["Method"]): dict(Q2_SOURCE_EVIDENCE[str(row["Method"])]) for row in rows if str(row["Method"]) in Q2_SOURCE_EVIDENCE}


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
    wall_clock = (
        _as_float(source_row.get("on_demand_wall_clock_seconds"))
        if source_row.get("on_demand_wall_clock_seconds") not in (None, "")
        else _wall_clock_seconds(source_dir)
    )
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
    row = {
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
    if method == "Tree of Thoughts":
        for key in ("tot_generate_events", "tot_evaluate_events", "tot_select_events", "tot_value_sample_count"):
            row[key] = data.get(key, "")
    if method in {"Direct / CoT", "Self-Consistency", "Best-of-N"}:
        for key in (
            "native_runtime_sample_count",
            "native_runtime_selection_events",
            "native_runtime_valid_samples",
            "native_runtime_parse_failures",
            "native_runtime_reasoning_samples",
            "total_samples",
            "valid_samples",
        ):
            row[key] = data.get(key, "")
    return row


def _native_method_dir_name(method: str) -> str:
    return method.lower().replace(" / ", "_").replace("-", "_").replace(" ", "_")


def _row_from_structured_artifacts(method: str, transfer_setting: str, source_dir: Path) -> Optional[Dict[str, object]]:
    table_path = source_dir / "scheduling_structured_q2_table.csv"
    if not table_path.exists():
        return None
    data = next((row for row in _read_csv(table_path) if row.get("Method") == method), None)
    if data is None:
        return None
    row = {
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
    for key in (
        "structured_runtime_solver",
        "structured_runtime_integer_variables",
        "structured_runtime_boolean_variables",
        "structured_runtime_interval_variables",
        "structured_runtime_exactly_one_constraints",
        "structured_runtime_no_overlap_constraints",
        "structured_runtime_precedence_constraints",
        "structured_runtime_release_constraints",
        "structured_runtime_objective_terms",
        "structured_runtime_heuristic",
        "structured_runtime_dispatch_steps",
        "structured_runtime_ready_list_selections",
        "structured_runtime_candidate_evaluations",
        "structured_runtime_tie_break_rule",
    ):
        if key in data:
            row[key] = data.get(key, "")
    return row


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
    fieldnames = list(rows[0])
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
