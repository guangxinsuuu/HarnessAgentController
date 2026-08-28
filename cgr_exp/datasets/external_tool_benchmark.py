from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ExternalToolBenchmarkAdapter:
    name: str
    enabled: bool = False
    source_url: str = "https://github.com/sierra-research/tau-bench"
    preferred_current_source_url: str = "https://github.com/sierra-research/tau2-bench"
    native_protocol_reference: str = "tau-bench native stateful tool-agent evaluation protocol"
    implementation_scope: str = "explicit stub; no local native tau-bench runner implemented"

    def is_available(self) -> bool:
        return False

    def load_tasks(self, split: str):
        del split
        raise RuntimeError("external structured tool benchmark adapter is a stub and is not required for smoke or CI")

    def to_manifest(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "status": "stub",
            "enabled": self.enabled,
            "available": self.is_available(),
            "source_url": self.source_url,
            "preferred_current_source_url": self.preferred_current_source_url,
            "native_protocol_reference": self.native_protocol_reference,
            "implementation_scope": self.implementation_scope,
            "result_ready": False,
            "claim_scope": "external_benchmark_pending",
            "requires_network_for_smoke": False,
        }
