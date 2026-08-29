from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


Q2_READINESS_DEFAULT = Path("runs/q2_table_current_seed0_protocol_v2_audit/table5_q2_readiness.csv")
CERTIFICATION_READINESS_DEFAULT = Path("runs/certification_tables_readiness/certification_tables_readiness.csv")


COMMANDS_BY_GROUP = {
    "learned_search": """python3 -m cgr_exp.cli run-scheduling-llm-pilot \\
  --output-dir runs/q2_learned_search_100_seed0_rest_gate_v2 \\
  --seed 0 \\
  --train-count 200 \\
  --validation-count 50 \\
  --test-count 100 \\
  --job-count 6 \\
  --machine-count 3 \\
  --node-budget 80 \\
  --action-budget 6 \\
  --records-per-instance 60 \\
  --rank-epochs 25 \\
  --model gpt-4o-mini \\
  --timeout 300 \\
  --perturbation mixed \\
  --batch-size 4 \\
  --parallel-episodes 6 \\
  --include-target-prm \\
  --include-rest-mcts \\
  --include-solver-certification \\
  --target-prm-train-count 100 \\
  --resume""",
    "native_frozen": """python3 -m cgr_exp.cli run-scheduling-native-q2 \\
  --output-dir runs/q2_native_llm_baselines_100_seed0_sc_v2 \\
  --seed 0 \\
  --train-count 200 \\
  --validation-count 50 \\
  --test-count 100 \\
  --job-count 6 \\
  --machine-count 3 \\
  --model gpt-4o-mini \\
  --timeout 300 \\
  --methods direct,self_consistency,best_of_n \\
  --sample-count 8 \\
  --parallel-episodes 8 \\
  --resume""",
    "native_tot": """python3 -m cgr_exp.cli run-scheduling-tot-q2 \\
  --output-dir runs/q2_tot_bfs_100_seed0_tot_gate_v2 \\
  --seed 0 \\
  --train-count 200 \\
  --validation-count 50 \\
  --test-count 100 \\
  --job-count 6 \\
  --machine-count 3 \\
  --node-budget 80 \\
  --action-budget 6 \\
  --breadth 5 \\
  --value-samples 3 \\
  --model gpt-4o-mini \\
  --timeout 300 \\
  --parallel-episodes 6 \\
  --perturbation mixed \\
  --resume""",
}

METHOD_GROUPS = {
    "Direct / CoT": "native_frozen",
    "Self-Consistency": "native_frozen",
    "Best-of-N": "native_frozen",
    "Tree of Thoughts": "native_tot",
    "Target-PRM priority": "learned_search",
    "PRM-guided MCTS (ReST-MCTS-style)": "learned_search",
    "Solver-trained priority": "learned_search",
    "Solver-trained priority + certified pruning": "learned_search",
}

GROUP_LABELS = {
    "learned_search": "Learned-search Q2 rows",
    "native_frozen": "Native frozen LLM baselines",
    "native_tot": "Strict native ToT BFS",
}


def build_q2_remaining_runs(
    out_dir: Path,
    *,
    q2_readiness: Path = Q2_READINESS_DEFAULT,
    certification_readiness: Path = CERTIFICATION_READINESS_DEFAULT,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    q2_rows = _read_csv(q2_readiness)
    cert_rows = _read_csv(certification_readiness)
    queue_rows = _queue_rows(q2_rows)
    ready_rows = _ready_rows(q2_rows, cert_rows)
    csv_path = out_dir / "q2_remaining_runs.csv"
    md_path = out_dir / "q2_remaining_runs.md"
    json_path = out_dir / "q2_remaining_runs.json"
    _write_csv(csv_path, queue_rows)
    md_path.write_text(_render_markdown(queue_rows, ready_rows, q2_readiness, certification_readiness), encoding="utf-8")
    manifest = {
        "artifact": "q2_remaining_runs",
        "q2_readiness": str(q2_readiness),
        "certification_readiness": str(certification_readiness),
        "remaining_group_count": len({row["run_group"] for row in queue_rows}),
        "remaining_method_count": len(queue_rows),
        "ready_count": len(ready_rows),
        "rows": queue_rows,
        "ready_rows": ready_rows,
        "produced_artifacts": [csv_path.name, md_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _queue_rows(q2_rows: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Mapping[str, str]]] = {}
    for row in q2_rows:
        method = str(row.get("Method", ""))
        group = METHOD_GROUPS.get(method)
        if group is None or _as_bool(row.get("result_ready")) and _as_bool(row.get("compute_ready")):
            continue
        grouped.setdefault(group, []).append(row)
    queue = []
    for group in ("learned_search", "native_frozen", "native_tot"):
        rows = grouped.get(group, [])
        if not rows:
            continue
        blockers = sorted({str(row.get("result_blockers", "")).strip() for row in rows if str(row.get("result_blockers", "")).strip()})
        compute_blockers = sorted({str(row.get("compute_blockers", "")).strip() for row in rows if str(row.get("compute_blockers", "")).strip()})
        source_artifacts = sorted({str(row.get("source_artifact", "")).strip() for row in rows if str(row.get("source_artifact", "")).strip()})
        queue.append(
            {
                "run_group": group,
                "label": GROUP_LABELS[group],
                "methods": "; ".join(str(row.get("Method", "")) for row in rows),
                "source_artifacts_seen": "; ".join(source_artifacts),
                "result_blockers": " | ".join(blockers),
                "compute_blockers": " | ".join(compute_blockers),
                "command": COMMANDS_BY_GROUP[group],
                "promotion_rule": "rerun with current code; do not patch old metrics by hand; promote only if result_ready=True and compute_ready=True",
            }
        )
    return queue


def _ready_rows(q2_rows: Iterable[Mapping[str, str]], cert_rows: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    ready = []
    for row in q2_rows:
        if _as_bool(row.get("result_ready")) and _as_bool(row.get("compute_ready")):
            ready.append(
                {
                    "scope": "Q2",
                    "name": str(row.get("Method", "")),
                    "source_artifact": str(row.get("source_artifact", "")),
                }
            )
    for row in cert_rows:
        if _as_bool(row.get("result_ready")):
            ready.append(
                {
                    "scope": "Certification",
                    "name": str(row.get("Table", "")),
                    "source_artifact": str(row.get("source_artifact", "")),
                }
            )
    return ready


def _render_markdown(
    queue_rows: List[Mapping[str, str]],
    ready_rows: List[Mapping[str, str]],
    q2_readiness: Path,
    certification_readiness: Path,
) -> str:
    lines = [
        "# Q2 Remaining Runs",
        "",
        "This artifact is generated from readiness CSVs. It lists only reruns needed",
        "for Q2 Scheduling promotion and keeps historical artifacts out of the paper",
        "table unless the current audit accepts them.",
        "",
        f"Q2 readiness: `{q2_readiness}`",
        f"Certification readiness: `{certification_readiness}`",
        "",
        "## Already Ready",
        "",
    ]
    if ready_rows:
        for row in ready_rows:
            lines.append(f"- {row['scope']}: {row['name']} from `{row['source_artifact']}`")
    else:
        lines.append("- None.")
    lines.extend(["", "## Remaining Runs", ""])
    if not queue_rows:
        lines.append("No Q2 reruns are required by the current readiness inputs.")
    for row in queue_rows:
        lines.extend(
            [
                f"### {row['label']}",
                "",
                f"Methods: {row['methods']}",
                "",
                f"Seen artifacts: `{row['source_artifacts_seen'] or 'none'}`",
                "",
                f"Result blockers: {row['result_blockers'] or 'none'}",
                "",
                f"Compute blockers: {row['compute_blockers'] or 'none'}",
                "",
                "```bash",
                row["command"],
                "```",
                "",
                f"Promotion rule: {row['promotion_rule']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Check",
            "",
            "```bash",
            "CERTIFICATION_DIR=runs/q2_structured_cert_reference_100_seed0_budget200_protocol_v2 \\",
            "AUTHORITY_DIR=runs/table7_grounding_robustness_100_seed0_budget200_protocol_v2 \\",
            "PYTHON_BIN=.venv/bin/python scripts/run_paper_audits.sh all",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


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


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}
