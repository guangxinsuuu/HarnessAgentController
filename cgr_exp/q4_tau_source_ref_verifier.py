from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping

from cgr_exp.q4_tau_source_evidence import Q4_TAU_SOURCE_EVIDENCE


_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def verify_q4_tau_source_refs(
    out_dir: Path,
    *,
    online: bool = False,
    timeout: float = 10.0,
    urlopen: Callable[..., object] | None = None,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = q4_tau_source_ref_verification_summary(online=online, timeout=timeout, urlopen=urlopen)
    csv_path = out_dir / "q4_tau_source_ref_verification.csv"
    json_path = out_dir / "q4_tau_source_ref_verification.json"
    _write_csv(csv_path, manifest["rows"])
    manifest = {**manifest, "produced_artifacts": [csv_path.name, json_path.name]}
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def q4_tau_source_ref_verification_summary(
    *,
    online: bool = False,
    timeout: float = 10.0,
    urlopen: Callable[..., object] | None = None,
) -> Dict[str, object]:
    opener = urlopen or urllib.request.urlopen
    rows = [
        _verify_source_refs(benchmark, evidence, online=online, timeout=timeout, urlopen=opener)
        for benchmark, evidence in Q4_TAU_SOURCE_EVIDENCE.items()
    ]
    return {
        "artifact": "q4_tau_source_ref_verification",
        "online_checked": online,
        "timeout": timeout,
        "all_static_refs_ready": all(row["static_refs_ready"] for row in rows),
        "all_online_refs_ready": all(row["online_refs_ready"] for row in rows) if online else None,
        "benchmarks": sorted(Q4_TAU_SOURCE_EVIDENCE),
        "rows": rows,
    }


def _verify_source_refs(
    benchmark: str,
    evidence: Mapping[str, object],
    *,
    online: bool,
    timeout: float,
    urlopen: Callable[..., object],
) -> Dict[str, object]:
    source_url = str(evidence.get("source_url", ""))
    source_revision = str(evidence.get("source_revision", ""))
    source_revision_url = str(evidence.get("source_revision_url", ""))
    code_refs = [str(ref) for ref in evidence.get("official_code_refs", [])] if isinstance(evidence.get("official_code_refs"), list) else []
    raw_urls = _raw_urls(code_refs)
    urls_to_check = [url for url in [source_revision_url, *raw_urls] if url]
    blockers: List[str] = []

    if not source_url.startswith("https://github.com/sierra-research/"):
        blockers.append("source_url must be an official Sierra benchmark repo")
    if not _GIT_SHA_RE.match(source_revision):
        blockers.append("source_revision must be lowercase 40-character SHA")
    if source_revision and source_revision not in source_revision_url:
        blockers.append("source_revision_url must include source_revision")
    if source_revision and not all(source_revision in ref for ref in code_refs):
        blockers.append("official_code_refs must be pinned to source_revision")
    if raw_urls and source_revision and not all(source_revision in url for url in raw_urls):
        blockers.append("raw github URLs must be pinned to source_revision")
    if not any("::" in ref for ref in code_refs):
        blockers.append("official_code_refs must include symbol or anchor refs")

    online_errors: List[str] = []
    if online:
        for url in urls_to_check:
            error = _check_url(url, timeout=timeout, urlopen=urlopen)
            if error:
                online_errors.append(f"{url}: {error}")

    return {
        "Benchmark": benchmark,
        "source_url": source_url,
        "source_revision": source_revision,
        "source_revision_url": source_revision_url,
        "raw_url_count": len(raw_urls),
        "checked_url_count": len(urls_to_check) if online else 0,
        "pinned_ref_count": sum(1 for ref in code_refs if source_revision and source_revision in ref),
        "symbol_ref_count": sum(1 for ref in code_refs if "::" in ref),
        "static_refs_ready": not blockers,
        "online_checked": online,
        "online_refs_ready": not online_errors if online else None,
        "blockers": "; ".join(blockers),
        "online_errors": "; ".join(online_errors),
    }


def _raw_urls(code_refs: Iterable[str]) -> List[str]:
    urls = []
    for ref in code_refs:
        if not ref.startswith("raw: "):
            continue
        target = ref.removeprefix("raw: ").split("::", 1)[0].strip()
        if target:
            urls.append(target)
    return urls


def _check_url(url: str, *, timeout: float, urlopen: Callable[..., object]) -> str:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "cgr-q4-source-verifier/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if int(status) >= 400:
                return f"http_status_{status}"
            return ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return exc.__class__.__name__


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
