from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class ExternalToolBenchmarkAdapter:
    name: str
    enabled: bool = False

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
            "requires_network_for_smoke": False,
        }
