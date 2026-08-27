from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchEpisode:
    episode_id: str
    task_family: str
    utility: float
    source_split: str = "train"
    task_embedding: tuple[float, ...] = ()
    state_embedding_mean: tuple[float, ...] = ()
    state_embedding_diag_variance: tuple[float, ...] = ()
