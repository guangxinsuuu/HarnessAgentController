from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


CERTIFICATION_PROTOCOL_REQUIRED = {
    "pruning_authority": "verified lower-bound/feasibility logic only",
    "learned_score_authority": "ranking only",
    "learned_score_can_prune": False,
    "uncertain_constraints_can_prune": False,
    "oracle_labels_in_inference_features": False,
}


def audit_certification_tables(
    out_dir: Path,
    *,
    certification_dir: Optional[Path] = None,
    authority_dir: Optional[Path] = None,
    expected_test_count: int = 100,
    authority_source_audit: Optional[Path] = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _clean_certification_row(certification_dir, expected_test_count),
        _grounding_robustness_row(authority_dir, expected_test_count),
    ]
    _apply_authority_source_audit(rows, authority_source_audit)
    csv_path = out_dir / "certification_tables_readiness.csv"
    json_path = out_dir / "certification_tables_readiness.json"
    _write_csv(csv_path, rows)
    manifest = {
        "audit": "certification_tables_readiness",
        "expected_test_count": expected_test_count,
        "all_rows_ready": all(row["result_ready"] for row in rows),
        "pending_or_blocked_rows": [row["Table"] for row in rows if not row["result_ready"]],
        "protocol_required": CERTIFICATION_PROTOCOL_REQUIRED,
        "authority_source_audit": None if authority_source_audit is None else str(authority_source_audit),
        "rows": rows,
        "produced_artifacts": [csv_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _apply_authority_source_audit(rows: List[Dict[str, object]], authority_source_audit: Optional[Path]) -> None:
    if authority_source_audit is None:
        return
    audit_rows = _read_csv(authority_source_audit)
    if not audit_rows:
        for row in rows:
            _append_blocker(row, "certification authority source audit missing")
        return
    failed = [
        f"{audit_row.get('Check', '')}: {audit_row.get('blockers', '')}"
        for audit_row in audit_rows
        if str(audit_row.get("authority_source_ready", "")).lower() != "true"
    ]
    if failed:
        for row in rows:
            _append_blocker(row, "certification authority source audit failed: " + "; ".join(failed))


def _append_blocker(row: Dict[str, object], blocker: str) -> None:
    existing = str(row.get("result_blockers", "")).strip()
    row["result_blockers"] = "; ".join(part for part in [existing, blocker] if part)
    row["result_ready"] = False


def _clean_certification_row(source_dir: Optional[Path], expected_test_count: int) -> Dict[str, object]:
    blockers = []
    if source_dir is None:
        blockers.append("artifact missing")
    else:
        blockers.extend(_required_files_blockers(source_dir, ["scheduling_certification_metrics.json", "table6_clean_certification.csv", "scheduling_certification_per_instance.csv"]))
        metrics, error = _read_json(source_dir / "scheduling_certification_metrics.json")
        if error:
            blockers.append(f"scheduling_certification_metrics.json {error}")
        if metrics:
            if metrics.get("benchmark") != "scheduling_certification":
                blockers.append("metrics benchmark must be scheduling_certification")
            blockers.extend(_protocol_blockers(metrics.get("certification_protocol")))
            methods = metrics.get("methods")
            if not isinstance(methods, Mapping) or "learned priority only" not in methods or "learned priority + certified pruning" not in methods:
                blockers.append("metrics missing clean priority/pruning rows")
            else:
                priority = methods["learned priority only"]
                certified = methods["learned priority + certified pruning"]
                if _as_float(priority.get("pruned_nodes")) != 0.0:
                    blockers.append("priority-only row must not prune")
                if _as_float(certified.get("wrong_prune_rate")) != 0.0:
                    blockers.append("certified-pruning row must have zero observed wrong prune")
        coverage = _unique_ids(source_dir / "scheduling_certification_per_instance.csv", "task_id") if source_dir is not None else []
        if len(coverage) < expected_test_count:
            blockers.append(f"scheduling_certification_per_instance.csv partial task coverage: {len(coverage)}/{expected_test_count}")
    return {
        "Table": "Table 6 clean certification",
        "source_artifact": "" if source_dir is None else str(source_dir),
        "expected_test_count": expected_test_count,
        "result_ready": not blockers,
        "result_blockers": "; ".join(blockers),
    }


def _grounding_robustness_row(source_dir: Optional[Path], expected_test_count: int) -> Dict[str, object]:
    blockers = []
    if source_dir is None:
        blockers.append("artifact missing")
    else:
        blockers.extend(_required_files_blockers(source_dir, ["table6_authority_metrics.json", "table7_grounding_robustness.csv", "table6_authority_per_instance.csv"]))
        metrics, error = _read_json(source_dir / "table6_authority_metrics.json")
        if error:
            blockers.append(f"table6_authority_metrics.json {error}")
        if metrics:
            if metrics.get("benchmark") != "table6_authority_separation":
                blockers.append("metrics benchmark must be table6_authority_separation")
            blockers.extend(_protocol_blockers(metrics.get("certification_protocol")))
            conditions = {str(item) for item in metrics.get("table7_robustness_conditions", [])} if isinstance(metrics.get("table7_robustness_conditions"), list) else set()
            required_conditions = {"Clean certified", "Verified FN 1%", "Verified FN 20%", "Verified FP 0.5%", "Verified FP 5%", "Uncertain-only 20%", "Learned-score pruning clean"}
            missing = sorted(required_conditions - conditions)
            if missing:
                blockers.append(f"table7 robustness conditions missing: {', '.join(missing)}")
        table_rows = _read_csv(source_dir / "table7_grounding_robustness.csv") if source_dir is not None else []
        if table_rows:
            blockers.extend(_table7_authority_runtime_blockers(table_rows))
        task_ids = _unique_ids(source_dir / "table6_authority_per_instance.csv", "task_id") if source_dir is not None else []
        if len(task_ids) < expected_test_count:
            blockers.append(f"table6_authority_per_instance.csv partial task coverage: {len(task_ids)}/{expected_test_count}")
    return {
        "Table": "Table 7 grounding robustness",
        "source_artifact": "" if source_dir is None else str(source_dir),
        "expected_test_count": expected_test_count,
        "result_ready": not blockers,
        "result_blockers": "; ".join(blockers),
    }


def _protocol_blockers(protocol: object) -> List[str]:
    if not isinstance(protocol, Mapping):
        return ["missing certification_protocol"]
    blockers = []
    for key, expected in CERTIFICATION_PROTOCOL_REQUIRED.items():
        if protocol.get(key) != expected:
            blockers.append(f"certification_protocol {key} must be {expected}")
    return blockers


def _table7_authority_runtime_blockers(rows: List[Mapping[str, str]]) -> List[str]:
    blockers: List[str] = []
    by_condition = {str(row.get("Condition", "")): row for row in rows}
    clean = by_condition.get("Clean certified")
    if clean and _as_float(clean.get("Semantic false prune")) != 0.0:
        blockers.append("Clean certified row must have zero semantic false prune")
    learned = by_condition.get("Learned-score pruning clean")
    if learned and _as_float(learned.get("Learned-score prunes")) <= 0:
        blockers.append("Learned-score pruning clean row must trigger positive learned-score prunes")
    uncertain_rows = [row for row in rows if str(row.get("Condition", "")).startswith("Uncertain-only")]
    for row in uncertain_rows:
        condition = str(row.get("Condition", "Uncertain-only"))
        if _as_float(row.get("Verified FP count")) != 0.0:
            blockers.append(f"{condition} row must not convert uncertain constraints into verified false positives")
        if _as_float(row.get("Verified infeasible rejections")) != 0.0:
            blockers.append(f"{condition} row must not reject actions using uncertain-only constraints")
        if _as_float(row.get("Learned-score prunes")) != 0.0:
            blockers.append(f"{condition} row must not use learned-score pruning")
    return blockers


def _required_files_blockers(source_dir: Path, names: Iterable[str]) -> List[str]:
    return [f"{name} missing" for name in names if not (source_dir / name).exists()]


def _read_json(path: Path) -> tuple[Dict[str, object], str]:
    if not path.exists() or path.stat().st_size == 0:
        return {}, ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {}, f"malformed json at line {exc.lineno}"
    if not isinstance(data, dict):
        return {}, "must contain a JSON object"
    return data, ""


def _unique_ids(path: Path, column: str) -> List[str]:
    return sorted({str(row.get(column, "")) for row in _read_csv(path) if str(row.get(column, ""))})


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


def _as_float(value: object) -> float:
    try:
        if value in (None, ""):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")
