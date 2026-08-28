from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


TAU_BENCH_SOURCE_URL = "https://github.com/sierra-research/tau-bench"
TAU2_BENCH_SOURCE_URL = "https://github.com/sierra-research/tau2-bench"

Q4_TAU_SOURCE_EVIDENCE: Dict[str, Dict[str, object]] = {
    "tau-bench": {
        "source_url": TAU_BENCH_SOURCE_URL,
        "source_revision": "59a200c6d575d595120f1cb70fea53cef0632f6b",
        "source_revision_url": "https://github.com/sierra-research/tau-bench/commit/59a200c6d575d595120f1cb70fea53cef0632f6b",
        "source_files": ["README.md", "tau_bench/run.py", "tau_bench/evaluation.py"],
        "official_code_refs": [
            "raw: https://raw.githubusercontent.com/sierra-research/tau-bench/59a200c6d575d595120f1cb70fea53cef0632f6b/README.md::tau-bench protocol",
            "github@59a200c6d575d595120f1cb70fea53cef0632f6b: sierra-research/tau-bench/tau_bench/run.py::run",
            "github@59a200c6d575d595120f1cb70fea53cef0632f6b: sierra-research/tau-bench/tau_bench/evaluation.py::evaluate",
        ],
        "official_source_anchors": [
            "README: tau-bench emulates dynamic conversations between a simulated user and a tool-using language agent",
            "README current warning: original repo tasks are outdated and users should prefer tau2/tau3 for latest fixed tasks",
        ],
        "official_primitives": [
            "stateful user-agent-tool interaction",
            "official user simulator",
            "official task verifier",
            "pass-k style success aggregation",
            "tool-call accounting",
        ],
        "local_promotion_gates": [
            "native_protocol_preserved true",
            "official user simulator and task verifier used",
            "per-task pass^1/pass^4/tool-call/token metrics",
            "full held-out task coverage",
        ],
        "status_note": "legacy tau-bench tasks; use only with explicit version disclosure",
    },
    "tau2-bench": {
        "source_url": TAU2_BENCH_SOURCE_URL,
        "source_revision": "a2c024725189473d2d7cea3a5cfdbcc67478e41f",
        "source_revision_url": "https://github.com/sierra-research/tau2-bench/commit/a2c024725189473d2d7cea3a5cfdbcc67478e41f",
        "source_files": ["README.md", "tau2/run.py", "tau2/evaluation"],
        "official_code_refs": [
            "raw: https://raw.githubusercontent.com/sierra-research/tau2-bench/a2c024725189473d2d7cea3a5cfdbcc67478e41f/README.md::tau2/tau3 protocol",
            "github@a2c024725189473d2d7cea3a5cfdbcc67478e41f: sierra-research/tau2-bench/tau2/run.py::run",
            "github@a2c024725189473d2d7cea3a5cfdbcc67478e41f: sierra-research/tau2-bench/tau2/evaluation::evaluate",
        ],
        "official_source_anchors": [
            "README: tau2/tau3 is the current successor benchmark for dual-control conversational agents",
            "README: current repository contains fixed airline/retail tasks and additional domains/modalities relative to original tau-bench",
        ],
        "official_primitives": [
            "native stateful conversation environment",
            "official simulated user",
            "official task evaluator",
            "domain tool APIs and policies",
            "pass-k style success aggregation",
            "tool-call and cost accounting",
        ],
        "local_promotion_gates": [
            "benchmark field declares tau2-bench or tau3-bench",
            "native_protocol_preserved true",
            "official simulator/evaluator preserved",
            "tool state preserved across the dialogue",
            "full held-out task coverage",
        ],
        "status_note": "preferred current source for new Q4 tool-agent runs",
    },
}

Q4_TAU_METHOD_REQUIREMENTS: Dict[str, List[str]] = {
    "Native agent rerun": ["single native rollout", "official simulator", "official verifier"],
    "Best-of-N": ["multiple independent native rollouts", "benchmark-compatible selection"],
    "Value-guided search": ["online value guidance", "native tool state preserved"],
    "Target-PRM": ["target tool-agent trajectory training", "disjoint train/test tasks", "process or step supervision"],
    "Solver-derived controller": ["source solver training", "zero-shot target tool transfer", "native tool state preserved"],
}


def q4_tau_source_evidence(benchmark: str) -> Dict[str, object]:
    return dict(Q4_TAU_SOURCE_EVIDENCE[benchmark])


def q4_tau_source_evidence_for_url(source_url: str) -> Dict[str, object]:
    for evidence in Q4_TAU_SOURCE_EVIDENCE.values():
        if evidence["source_url"] == source_url:
            return dict(evidence)
    return {}


def q4_tau_source_evidence_summary() -> Dict[str, object]:
    rows = [_audit_source_evidence_row(name, evidence) for name, evidence in Q4_TAU_SOURCE_EVIDENCE.items()]
    return {
        "artifact": "q4_tau_source_evidence",
        "all_sources_ready": all(row["source_ready"] for row in rows),
        "benchmarks": sorted(Q4_TAU_SOURCE_EVIDENCE),
        "method_requirements": Q4_TAU_METHOD_REQUIREMENTS,
        "rows": rows,
    }


def audit_q4_tau_source_evidence(out_dir: Path) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = q4_tau_source_evidence_summary()
    csv_path = out_dir / "q4_tau_source_evidence_audit.csv"
    json_path = out_dir / "q4_tau_source_evidence_audit.json"
    _write_csv(csv_path, manifest["rows"])
    manifest = {**manifest, "produced_artifacts": [csv_path.name, json_path.name]}
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _audit_source_evidence_row(benchmark: str, evidence: Mapping[str, object]) -> Dict[str, object]:
    blockers = []
    source_url = str(evidence.get("source_url", ""))
    source_revision = str(evidence.get("source_revision", ""))
    source_revision_url = str(evidence.get("source_revision_url", ""))
    source_files = _strings(evidence.get("source_files"))
    code_refs = _strings(evidence.get("official_code_refs"))
    source_anchors = _strings(evidence.get("official_source_anchors"))
    primitives = _strings(evidence.get("official_primitives"))
    gates = _strings(evidence.get("local_promotion_gates"))
    if not source_url.startswith("https://github.com/sierra-research/"):
        blockers.append("source_url must be an official Sierra benchmark repo")
    if len(source_revision) != 40:
        blockers.append("source_revision must be a full 40-character git SHA")
    if source_revision and source_revision not in source_revision_url:
        blockers.append("source_revision_url missing source_revision")
    if not source_files:
        blockers.append("missing source_files")
    if not code_refs:
        blockers.append("missing official_code_refs")
    if source_revision and not any(source_revision in ref for ref in code_refs):
        blockers.append("official_code_refs must be pinned to source_revision")
    if not any("::" in ref for ref in code_refs):
        blockers.append("official_code_refs must include symbol or anchor refs")
    if not source_anchors:
        blockers.append("missing official_source_anchors")
    if not primitives:
        blockers.append("missing official_primitives")
    if not gates:
        blockers.append("missing local_promotion_gates")
    return {
        "Benchmark": benchmark,
        "source_url": source_url,
        "source_revision": source_revision,
        "source_revision_url": source_revision_url,
        "source_files": "; ".join(source_files),
        "official_code_refs": "; ".join(code_refs),
        "official_source_anchors": "; ".join(source_anchors),
        "official_primitives": "; ".join(primitives),
        "local_promotion_gates": "; ".join(gates),
        "status_note": str(evidence.get("status_note", "")),
        "source_ready": not blockers,
        "blockers": "; ".join(blockers),
    }


def _strings(value: object) -> List[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
