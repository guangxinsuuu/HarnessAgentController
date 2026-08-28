from __future__ import annotations

import csv
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping

from cgr_exp.q2_source_evidence import Q2_EXTERNAL_SOURCE_METHODS, Q2_SOURCE_EVIDENCE


_ARXIV_REVISION_RE = re.compile(r"^arXiv:\d{4}\.\d{4,5}v\d+$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def verify_q2_source_refs(
    out_dir: Path,
    *,
    online: bool = False,
    timeout: float = 10.0,
    urlopen: Callable[..., object] | None = None,
) -> Dict[str, object]:
    """Verify that Q2 external-method evidence is pinned to inspectable sources."""

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = q2_source_ref_verification_summary(online=online, timeout=timeout, urlopen=urlopen)
    csv_path = out_dir / "q2_source_ref_verification.csv"
    json_path = out_dir / "q2_source_ref_verification.json"
    _write_csv(csv_path, manifest["rows"])
    manifest = {**manifest, "produced_artifacts": [csv_path.name, json_path.name]}
    json_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def q2_source_ref_verification_summary(
    *,
    online: bool = False,
    timeout: float = 10.0,
    urlopen: Callable[..., object] | None = None,
) -> Dict[str, object]:
    opener = urlopen or urllib.request.urlopen
    rows = [_verify_method_refs(method, online=online, timeout=timeout, urlopen=opener) for method in Q2_EXTERNAL_SOURCE_METHODS]
    return {
        "artifact": "q2_source_ref_verification",
        "online_checked": online,
        "timeout": timeout,
        "all_static_refs_ready": all(row["static_refs_ready"] for row in rows),
        "all_online_refs_ready": all(row["online_refs_ready"] for row in rows) if online else None,
        "external_methods": Q2_EXTERNAL_SOURCE_METHODS,
        "rows": rows,
    }


def _verify_method_refs(
    method: str,
    *,
    online: bool,
    timeout: float,
    urlopen: Callable[..., object],
) -> Dict[str, object]:
    evidence = Q2_SOURCE_EVIDENCE.get(method, {})
    source_url = str(evidence.get("source_url", ""))
    source_revision = str(evidence.get("source_revision", ""))
    source_revision_url = str(evidence.get("source_revision_url", ""))
    code_refs = [str(ref) for ref in evidence.get("official_code_refs", [])] if isinstance(evidence.get("official_code_refs"), list) else []
    raw_targets = _raw_targets(code_refs)
    raw_urls = [url for url, _anchor in raw_targets]
    urls_to_check = [url for url in [source_revision_url, *raw_urls] if url]
    blockers: List[str] = []

    if not source_url:
        blockers.append("missing source_url")
    if not source_revision:
        blockers.append("missing source_revision")
    if not source_revision_url:
        blockers.append("missing source_revision_url")

    if source_url.startswith("https://github.com/"):
        if source_revision and not _GIT_SHA_RE.match(source_revision):
            blockers.append("github source_revision must be lowercase 40-character SHA")
        if source_revision and source_revision not in source_revision_url:
            blockers.append("github source_revision_url must include source_revision")
        if source_revision and not any(source_revision in ref for ref in code_refs):
            blockers.append("github official_code_refs must be pinned to source_revision")
        if raw_urls and source_revision and not all(source_revision in url for url in raw_urls):
            blockers.append("raw github URLs must be pinned to source_revision")
        if not any("::" in ref for ref in code_refs):
            blockers.append("github official_code_refs must include symbol-level anchors")

    if source_url.startswith("https://arxiv.org/"):
        if source_revision and not _ARXIV_REVISION_RE.match(source_revision):
            blockers.append("arxiv source_revision must include explicit version, e.g. arXiv:2201.11903v6")
        arxiv_id = source_revision.removeprefix("arXiv:")
        if source_revision and arxiv_id not in source_revision_url:
            blockers.append("arxiv source_revision_url must include the exact versioned id")
        if source_revision and not any(source_revision in ref for ref in code_refs):
            blockers.append("paper official_code_refs must cite the exact arxiv version")

    online_errors: List[str] = []
    raw_anchor_matches = 0
    if online:
        for url in [source_revision_url] if source_revision_url else []:
            error = _check_url(url, timeout=timeout, urlopen=urlopen)
            if error:
                online_errors.append(f"{url}: {error}")
        for url, anchor in raw_targets:
            error = _check_raw_url(url, anchor=anchor, timeout=timeout, urlopen=urlopen)
            if error:
                online_errors.append(f"{url}: {error}")
            else:
                raw_anchor_matches += 1

    return {
        "Method": method,
        "source_url": source_url,
        "source_revision": source_revision,
        "source_revision_url": source_revision_url,
        "raw_url_count": len(raw_urls),
        "checked_url_count": len(urls_to_check) if online else 0,
        "raw_anchor_count": len(raw_targets),
        "raw_anchor_match_count": raw_anchor_matches if online else 0,
        "pinned_ref_count": sum(1 for ref in code_refs if source_revision and source_revision in ref),
        "symbol_ref_count": sum(1 for ref in code_refs if "::" in ref),
        "static_refs_ready": not blockers,
        "online_checked": online,
        "online_refs_ready": not online_errors if online else None,
        "blockers": "; ".join(blockers),
        "online_errors": "; ".join(online_errors),
    }


def _raw_urls(code_refs: Iterable[str]) -> List[str]:
    return [url for url, _anchor in _raw_targets(code_refs)]


def _raw_targets(code_refs: Iterable[str]) -> List[tuple[str, str]]:
    urls = []
    for ref in code_refs:
        if not ref.startswith("raw: "):
            continue
        raw_ref = ref.removeprefix("raw: ")
        target, anchor = raw_ref.split("::", 1) if "::" in raw_ref else (raw_ref, "")
        target = target.strip()
        anchor = anchor.strip()
        if target:
            urls.append((target, anchor))
    return urls


def _check_url(url: str, *, timeout: float, urlopen: Callable[..., object]) -> str:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "cgr-q2-source-verifier/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if int(status) >= 400:
                return f"http_status_{status}"
            return ""
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return exc.__class__.__name__


def _check_raw_url(url: str, *, anchor: str, timeout: float, urlopen: Callable[..., object]) -> str:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "cgr-q2-source-verifier/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if int(status) >= 400:
                return f"http_status_{status}"
            if not anchor:
                return "missing raw content anchor"
            body = response.read()
            text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
            if anchor not in text:
                return f"missing raw content anchor {anchor!r}"
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
