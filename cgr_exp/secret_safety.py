from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


SECRET_PATTERNS = [
    ("openai_project_key", re.compile(r"sk-proj-[A-Za-z0-9_-]{20,}")),
    ("openai_secret_key", re.compile(r"sk-[A-Za-z0-9_-]{20,}")),
]

ALLOWED_PLACEHOLDERS = {"sk-...", "sk-proj-..."}


def audit_secret_safety(out_dir: Path, *, root: Path = Path(".")) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _scan_rows(root)
    csv_path = out_dir / "secret_safety_audit.csv"
    json_path = out_dir / "secret_safety_audit.json"
    _write_csv(csv_path, rows)
    manifest = {
        "artifact": "secret_safety_audit",
        "root": str(root),
        "pattern_names": [name for name, _pattern in SECRET_PATTERNS],
        "finding_count": len(rows),
        "secrets_ready": len(rows) == 0,
        "rows": rows,
        "produced_artifacts": [csv_path.name, json_path.name],
    }
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _scan_rows(root: Path) -> List[Dict[str, object]]:
    rows = []
    for path in _tracked_files(root):
        full_path = root / path
        try:
            text = full_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            claimed_spans: List[tuple[int, int]] = []
            for pattern_name, pattern in SECRET_PATTERNS:
                for match in pattern.finditer(line):
                    if any(_overlaps(match.span(), span) for span in claimed_spans):
                        continue
                    value = match.group(0)
                    if value in ALLOWED_PLACEHOLDERS:
                        continue
                    claimed_spans.append(match.span())
                    rows.append(
                        {
                            "path": path,
                            "line": line_number,
                            "pattern": pattern_name,
                            "redacted_match": _redact(value),
                        }
                    )
    return rows


def _tracked_files(root: Path) -> List[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return [
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.is_file() and ".git" not in path.parts and "runs" not in path.parts
        ]
    return [line for line in result.stdout.splitlines() if line]


def _redact(value: str) -> str:
    if len(value) <= 12:
        return "***"
    return f"{value[:6]}...{value[-4:]}"


def _overlaps(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    fieldnames = ["path", "line", "pattern", "redacted_match"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
