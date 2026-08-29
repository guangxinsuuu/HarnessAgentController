from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from cgr_exp.datasets.external_tool_benchmark import ExternalToolBenchmarkAdapter
from cgr_exp.q4_tau_source_evidence import Q4_TAU_METHOD_REQUIREMENTS, q4_tau_source_evidence_summary
from cgr_exp.table_q4_tau import TAU_METHOD_PROTOCOL_REQUIRED, TAU_NATIVE_PROTOCOL_REQUIRED, TAU_Q4_METHODS


Q4_REQUIRED_ARTIFACTS = [
    "q4_tau_manifest.json",
    "q4_tau_per_instance.csv",
    "q4_tau_row.json or q4_tau_table.csv",
]


def build_q4_tau_adapter_contract(out_dir: Path, *, expected_test_count: int = 100) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [_method_row(method, transfer_setting, expected_test_count) for method, transfer_setting in TAU_Q4_METHODS]
    adapter = ExternalToolBenchmarkAdapter(name="tau/tau2 native adapter")
    manifest = {
        "artifact": "q4_tau_adapter_contract",
        "expected_test_count": expected_test_count,
        "adapter_status": adapter.to_manifest(),
        "source_evidence": q4_tau_source_evidence_summary(),
        "native_protocol_required": sorted(TAU_NATIVE_PROTOCOL_REQUIRED),
        "method_protocol_required": {method: sorted(required) for method, required in TAU_METHOD_PROTOCOL_REQUIRED.items()},
        "required_artifacts": Q4_REQUIRED_ARTIFACTS,
        "rows": rows,
        "promotion_rule": "do not report Q4 rows until a native tau/tau2 artifact preserves the official simulator, tool state, verifier, pass-k aggregation, tool-call accounting, and token accounting",
        "produced_artifacts": [
            "q4_tau_adapter_contract.csv",
            "q4_tau_adapter_contract.md",
            "q4_tau_adapter_contract.json",
        ],
    }
    _write_csv(out_dir / "q4_tau_adapter_contract.csv", rows)
    (out_dir / "q4_tau_adapter_contract.md").write_text(_render_markdown(manifest), encoding="utf-8")
    (out_dir / "q4_tau_adapter_contract.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _method_row(method: str, transfer_setting: str, expected_test_count: int) -> Dict[str, object]:
    return {
        "Method": method,
        "Transfer setting": transfer_setting,
        "expected_test_count": expected_test_count,
        "native_protocol_required": "; ".join(sorted(TAU_NATIVE_PROTOCOL_REQUIRED)),
        "method_primitives_required": "; ".join(sorted(TAU_METHOD_PROTOCOL_REQUIRED[method])),
        "method_requirements": "; ".join(Q4_TAU_METHOD_REQUIREMENTS[method]),
        "required_artifacts": "; ".join(Q4_REQUIRED_ARTIFACTS),
        "implementation_status": "blocked: native tau/tau2 adapter is an explicit stub",
        "promotion_gate": "q4_tau_readiness result_ready=True",
    }


def _render_markdown(manifest: Mapping[str, object]) -> str:
    lines = [
        "# Q4 tau/tau2 Adapter Contract",
        "",
        "This contract is generated from the pinned tau/tau2 source evidence and",
        "the Q4 table readiness gates. It is intentionally strict: synthetic",
        "tool-agent traces or local mock environments must not populate Q4 rows.",
        "",
        f"Expected held-out tasks: `{manifest['expected_test_count']}`",
        "",
        "## Native Protocol Required",
        "",
    ]
    for item in manifest["native_protocol_required"]:
        lines.append(f"- `{item}`")
    lines.extend(
        [
            "",
            "## Required Artifacts",
            "",
        ]
    )
    for item in manifest["required_artifacts"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Method Rows", ""])
    for row in manifest["rows"]:
        lines.extend(
            [
                f"### {row['Method']}",
                "",
                f"Transfer setting: {row['Transfer setting']}",
                "",
                f"Required method primitives: {row['method_primitives_required']}",
                "",
                f"Promotion gate: `{row['promotion_gate']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Promotion Rule",
            "",
            str(manifest["promotion_rule"]),
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
