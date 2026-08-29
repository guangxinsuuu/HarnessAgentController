from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


CERTIFICATION_AUTHORITY_SOURCE_CHECKS: Dict[str, Dict[str, object]] = {
    "learned_score_has_no_pruning_authority": {
        "path": "cgr_exp/certification/prune.py",
        "required_anchors": [
            "learned_score",
            "memory_score",
            "del learned_score, memory_score",
        ],
        "forbidden_anchors": [
            "if learned_score",
            "if memory_score",
            "learned_score >",
            "learned_score <",
            "memory_score >",
            "memory_score <",
        ],
    },
    "verified_infeasibility_can_prune": {
        "path": "cgr_exp/certification/prune.py",
        "required_anchors": [
            "FeasibleStatus.INFEASIBLE",
            "verified-infeasible",
        ],
        "forbidden_anchors": [],
    },
    "verified_lower_bound_can_prune": {
        "path": "cgr_exp/certification/prune.py",
        "required_anchors": [
            "node.lower_bound is not None",
            "node.lower_bound >= incumbent",
            "lower-bound",
        ],
        "forbidden_anchors": [],
    },
}


def audit_certification_authority_source(out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_audit_row(name, spec) for name, spec in CERTIFICATION_AUTHORITY_SOURCE_CHECKS.items()]
    csv_path = out_dir / "certification_authority_source_audit.csv"
    json_path = out_dir / "certification_authority_source_audit.json"
    _write_csv(csv_path, rows)
    manifest = {
        "audit": "certification_authority_source",
        "all_authority_source_ready": all(bool(row["authority_source_ready"]) for row in rows),
        "rows": rows,
        "produced_artifacts": [csv_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _audit_row(name: str, spec: Mapping[str, object]) -> Dict[str, object]:
    path = Path(str(spec["path"]))
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    required = [str(anchor) for anchor in spec.get("required_anchors", [])]
    forbidden = [str(anchor) for anchor in spec.get("forbidden_anchors", [])]
    required_locations = {anchor: _anchor_locations(anchor, path, text) for anchor in required}
    forbidden_locations = {anchor: _anchor_locations(anchor, path, text) for anchor in forbidden}
    missing_required = [anchor for anchor, locations in required_locations.items() if not locations]
    present_forbidden = {
        anchor: locations
        for anchor, locations in forbidden_locations.items()
        if locations
    }
    blockers = []
    if not path.exists():
        blockers.append("source path missing")
    if missing_required:
        blockers.append("missing required anchors: " + ", ".join(missing_required))
    if present_forbidden:
        blockers.append("forbidden authority anchors present: " + ", ".join(present_forbidden))
    return {
        "Check": name,
        "source_path": str(path),
        "required_anchors": "; ".join(required),
        "required_anchor_locations": json.dumps(required_locations, sort_keys=True),
        "forbidden_anchors": "; ".join(forbidden),
        "forbidden_anchor_locations": json.dumps(present_forbidden, sort_keys=True),
        "authority_source_ready": not blockers,
        "blockers": "; ".join(blockers),
    }


def _anchor_locations(anchor: str, path: Path, text: str) -> List[str]:
    if not text:
        return []
    return [
        f"{path}:{idx}"
        for idx, line in enumerate(text.splitlines(), start=1)
        if anchor in line
    ]


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
