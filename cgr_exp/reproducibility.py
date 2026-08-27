from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Iterable


DETERMINISTIC_CORE_ARTIFACTS = (
    "primary_table.csv",
    "transfer_table.csv",
    "ranking_table.csv",
    "certification_table.csv",
    "proposal_table.csv",
    "budget_table.csv",
    "budget_protocol.json",
    "domain_shift_table.csv",
    "trace_sample.jsonl",
    "train_history.jsonl",
    "family_split_manifest.json",
    "ranking_label_schema.json",
    "level1_family_table.csv",
)


def artifact_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def deterministic_core_hashes(run_dir: Path, artifacts: Iterable[str] = DETERMINISTIC_CORE_ARTIFACTS) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for name in artifacts:
        path = run_dir / name
        if path.exists():
            hashes[name] = artifact_sha256(path)
    return hashes
