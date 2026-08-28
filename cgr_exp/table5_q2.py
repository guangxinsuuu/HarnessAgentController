from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

from cgr_exp.q2_method_contracts import q2_method_contracts


TABLE5_METHODS = [
    ("CP-SAT", "Structured-input solver", None, "structured"),
    ("EST / list scheduling heuristic", "Structured-input heuristic", None, "structured"),
    ("Direct / CoT", "Frozen", None, "native"),
    ("Self-Consistency", "Frozen", None, "native"),
    ("Best-of-N", "Frozen", None, "native"),
    ("Tree of Thoughts", "Online", "GPT-frontier-rank / ToT-style", "scheduling"),
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
            if loaded is None:
                loaded = _row_from_native_artifacts(method, transfer_setting, source_dir)
            if loaded is not None:
                row = loaded
        elif source_dir is not None:
            loaded = _row_from_native_artifacts(method, transfer_setting, source_dir)
            if loaded is not None:
                row = loaded
        rows.append(row)
    csv_path = out_dir / "table5_q2_results.csv"
    tex_path = out_dir / "table5_q2_results.tex"
    manifest_path = out_dir / "table5_q2_manifest.json"
    _write_csv(csv_path, rows)
    tex_path.write_text(_latex_table(rows), encoding="utf-8")
    manifest = {
        "table": "Table 5",
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
        "table": "Table 5",
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
    if tokens == "--":
        compute_blockers.append("missing token accounting")
    if "shared-cache fallback accounting" in notes:
        compute_blockers.append("shared-cache fallback accounting")
    if status != "complete":
        compute_blockers.append("result not complete")
    return {
        "Method": row["Method"],
        "Status": status,
        "result_ready": not result_blockers,
        "compute_ready": not compute_blockers,
        "result_blockers": "; ".join(result_blockers),
        "compute_blockers": "; ".join(compute_blockers),
        "token_notes": notes,
    }


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
        native_rows = _read_csv(source_dir / "scheduling_native_q2_table.csv")
        data = next((row for row in native_rows if row.get("Method") == method), None)
        if data is None:
            return None
        table_path = source_dir / "scheduling_native_q2_table.csv"
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
        "source_policy": str(data.get("source_policy", "native")),
        "notes": str(data.get("notes", "native protocol row")),
    }


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
