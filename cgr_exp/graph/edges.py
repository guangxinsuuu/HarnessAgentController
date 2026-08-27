from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass(frozen=True)
class GraphEdge:
    source: str
    target: str
    edge_type: str
    reason: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)

