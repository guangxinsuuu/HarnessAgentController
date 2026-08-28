from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from cgr_exp.q4_tau_source_evidence import Q4_TAU_SOURCE_EVIDENCE


DEFAULT_Q2_READINESS = Path("runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv")
DEFAULT_Q4_READINESS = Path("runs/q4_tau_table_current_audit/q4_tau_readiness.csv")
DEFAULT_CERT_READINESS = Path("runs/certification_tables_readiness/certification_tables_readiness.csv")
DEFAULT_Q2_MATRIX = Path("runs/q2_implementation_matrix/q2_implementation_matrix.csv")


def build_current_experiment_status(
    out_dir: Path,
    *,
    q2_readiness: Path = DEFAULT_Q2_READINESS,
    q4_readiness: Path = DEFAULT_Q4_READINESS,
    certification_readiness: Path = DEFAULT_CERT_READINESS,
    q2_implementation_matrix: Path = DEFAULT_Q2_MATRIX,
    audit_command: str = "PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all",
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    q2_rows = _read_csv(q2_readiness)
    q4_rows = _read_csv(q4_readiness)
    cert_rows = _read_csv(certification_readiness)
    q2_matrix_rows = _read_csv(q2_implementation_matrix)
    markdown = _render_markdown(
        q2_rows=q2_rows,
        q4_rows=q4_rows,
        cert_rows=cert_rows,
        q2_matrix_rows=q2_matrix_rows,
        audit_command=audit_command,
        q2_readiness=q2_readiness,
        q4_readiness=q4_readiness,
        certification_readiness=certification_readiness,
        q2_implementation_matrix=q2_implementation_matrix,
    )
    md_path = out_dir / "current_experiment_status.md"
    json_path = out_dir / "current_experiment_status.json"
    md_path.write_text(markdown, encoding="utf-8")
    manifest = {
        "artifact": "current_experiment_status",
        "audit_command": audit_command,
        "input_artifacts": {
            "q2_readiness": str(q2_readiness),
            "q4_readiness": str(q4_readiness),
            "certification_readiness": str(certification_readiness),
            "q2_implementation_matrix": str(q2_implementation_matrix),
        },
        "q2_rows": _row_count(q2_rows),
        "q2_ready_rows": _ready_count(q2_rows),
        "q4_rows": _row_count(q4_rows),
        "q4_ready_rows": _ready_count(q4_rows),
        "certification_rows": _row_count(cert_rows),
        "certification_ready_rows": _ready_count(cert_rows),
        "produced_artifacts": [md_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _render_markdown(
    *,
    q2_rows: List[Dict[str, str]],
    q4_rows: List[Dict[str, str]],
    cert_rows: List[Dict[str, str]],
    q2_matrix_rows: List[Dict[str, str]],
    audit_command: str,
    q2_readiness: Path,
    q4_readiness: Path,
    certification_readiness: Path,
    q2_implementation_matrix: Path,
) -> str:
    q2_status = _status_map(q2_rows, method_key="Method", ready_key="result_ready", blocker_key="result_blockers")
    q2_impl = {row.get("Method", ""): row for row in q2_matrix_rows}
    lines = [
        "# Current Experiment Promotion Status",
        "",
        "This file is generated from machine-readable readiness artifacts. Do not",
        "promote a paper-table row unless the relevant readiness row is ready.",
        "",
        "Audit command:",
        "",
        "```bash",
        audit_command,
        "```",
        "",
        "## Q2 Real-LLM Scheduling",
        "",
        f"Machine audit: `{q2_readiness}`.",
        "",
        "| Row | Source artifact | Implementation status | Current promotion status | Compute status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for method in _q2_order(q2_rows, q2_matrix_rows):
        audit = q2_status.get(method, {})
        impl = q2_impl.get(method, {})
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(method),
                    _escape(_source_artifact_status(audit)),
                    _escape(_implementation_status(impl)),
                    _escape(_promotion_status(audit)),
                    _escape(_compute_status(audit)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Implementation/source matrix: `{q2_implementation_matrix}`.",
            "",
            "## Q4 tau/tau2 Tool-Agent Evaluation",
            "",
            f"Machine audit: `{q4_readiness}`.",
            "",
            _q4_summary(q4_rows),
            "",
            "| Row | Source artifact | Current promotion status | Notes |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in q4_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(row.get("Method", "")),
                    _escape(_source_artifact_status(row)),
                    _escape(_promotion_status(row)),
                    _escape(row.get("notes", "")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Pinned Q4 source evidence:",
            "",
        ]
    )
    for name, evidence in sorted(Q4_TAU_SOURCE_EVIDENCE.items()):
        lines.append(f"- {name}: `{evidence['source_revision']}`")
    lines.extend(
        [
            "",
            "## Certification Tables",
            "",
            f"Machine audit: `{certification_readiness}`.",
            "",
            "| Table | Source artifact | Current promotion status |",
            "| --- | --- | --- |",
        ]
    )
    for row in cert_rows:
        table = row.get("Table", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape(table),
                    _escape(_source_artifact_status(row)),
                    _escape(_promotion_status(row)),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Required certification authority contract:",
            "",
            "- Verified lower-bound or feasibility logic is the only pruning authority.",
            "- Learned scores may rank frontier states only.",
            "- Uncertain constraints may not authorize irreversible pruning.",
            "- Oracle labels are not inference features.",
            "- Pruned/rejected states are audited against the uncorrupted descendant subproblem.",
            "",
            "## Immediate Rerun Order",
            "",
            "1. Rerun structured Q2 if CP-SAT is blocked by stale source evidence.",
            "2. Rerun learned-search Q2 with Target-PRM, ReST-MCTS-style, Solver prior, and Solver prior + certification in the same current artifact.",
            "3. Rerun native Q2 Direct / CoT, Self-Consistency, and Best-of-N.",
            "4. Rerun strict native ToT BFS with `run-scheduling-tot-q2`.",
            "5. Rerun clean certification and grounding robustness with current code if their readiness rows are blocked.",
            "6. Keep Q4 tau/tau2 rows pending until a native tau/tau2 adapter exists.",
            "",
            "Only rows marked ready by the relevant readiness CSV should be promoted into paper tables.",
            "",
        ]
    )
    return "\n".join(lines)


def _q2_order(q2_rows: Iterable[Mapping[str, str]], q2_matrix_rows: Iterable[Mapping[str, str]]) -> List[str]:
    names = [row.get("Method", "") for row in q2_rows if row.get("Method", "")]
    for row in q2_matrix_rows:
        method = row.get("Method", "")
        if method and method not in names:
            names.append(method)
    return names


def _status_map(rows: Iterable[Mapping[str, str]], *, method_key: str, ready_key: str, blocker_key: str) -> Dict[str, Dict[str, str]]:
    mapped: Dict[str, Dict[str, str]] = {}
    for row in rows:
        method = str(row.get(method_key, ""))
        mapped[method] = {
            "ready": str(row.get(ready_key, "")),
            "blockers": str(row.get(blocker_key, "")),
            "status": str(row.get("Status", "")),
            "compute_ready": str(row.get("compute_ready", "")),
            "compute_blockers": str(row.get("compute_blockers", "")),
            "source_artifact": str(row.get("source_artifact", "")),
            "source_policy": str(row.get("source_policy", "")),
        }
    return mapped


def _implementation_status(row: Mapping[str, str]) -> str:
    if not row:
        return "No implementation matrix entry."
    present = str(row.get("Local code present", "")).lower() == "true"
    scope = row.get("Implemented scope", "")
    return ("Implemented. " if present else "Missing local code. ") + scope


def _promotion_status(row: Mapping[str, str]) -> str:
    ready = str(row.get("ready", row.get("result_ready", ""))).lower() == "true"
    blockers = str(row.get("blockers", row.get("result_blockers", ""))).strip()
    if ready:
        return "Ready in current audit."
    if blockers:
        return f"Blocked: {blockers}"
    status = str(row.get("status", row.get("Status", ""))).strip()
    return f"Blocked: {status}" if status else "Blocked: readiness evidence missing"


def _source_artifact_status(row: Mapping[str, str]) -> str:
    artifact = str(row.get("source_artifact", "")).strip()
    policy = str(row.get("source_policy", "")).strip()
    if not artifact:
        return "missing"
    if policy:
        return f"{artifact} ({policy})"
    return artifact


def _compute_status(row: Mapping[str, str]) -> str:
    ready = str(row.get("compute_ready", "")).lower() == "true"
    blockers = str(row.get("compute_blockers", "")).strip()
    if ready:
        return "Ready."
    if blockers:
        return f"Blocked: {blockers}"
    return "Blocked: compute evidence missing"


def _q4_summary(rows: List[Dict[str, str]]) -> str:
    if not rows:
        return "No Q4 readiness artifact was found."
    if _ready_count(rows) == 0:
        return "All Q4 tau/tau2 rows remain pending; the repository still has no native tau/tau2 runner artifact in the audited directories."
    return f"{_ready_count(rows)}/{len(rows)} Q4 rows are ready in the current audit."


def _row_count(rows: Iterable[Mapping[str, str]]) -> int:
    return len(list(rows))


def _ready_count(rows: Iterable[Mapping[str, str]]) -> int:
    count = 0
    for row in rows:
        value = str(row.get("result_ready", row.get("ready", ""))).lower()
        if value == "true":
            count += 1
    return count


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
