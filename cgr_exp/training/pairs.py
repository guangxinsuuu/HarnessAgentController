from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Dict, List, Sequence, Tuple

from cgr_exp.features import FEATURE_NAMES, state_features
from cgr_exp.problems import Problem, State, best_completion_objective
from cgr_exp.ranker import Pair
from cgr_exp.trace_data import make_ranking_pairs


FORBIDDEN_INFERENCE_FEATURES = (
    "contains_optimal_descendant",
    "future_best_objective",
    "future_remaining_cost",
    "best_descendant_objective",
    "remaining_exact_expansions_to_best",
)


@dataclass(frozen=True)
class RankingPairRecord:
    positive_features: List[float]
    negative_features: List[float]
    feature_names: List[str]
    label_metadata: Dict[str, object]
    trace_ids: Tuple[str, str]
    source_split: str

    def to_pair(self) -> Pair:
        return (self.positive_features, self.negative_features)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def make_labeled_ranking_pairs(
    samples: Sequence[Tuple[Problem, State]],
    max_pairs_per_problem: int,
    seed: int,
    source_split: str,
) -> List[RankingPairRecord]:
    _assert_no_forbidden_inference_features(FEATURE_NAMES)
    rng = random.Random(seed)
    by_problem: Dict[str, List[Tuple[Problem, State]]] = {}
    for problem, state in samples:
        by_problem.setdefault(problem.pid, []).append((problem, state))

    records: List[RankingPairRecord] = []
    for rows in by_problem.values():
        if len(rows) < 2:
            continue
        candidates = []
        memo = {}
        for problem, state in rows:
            value = best_completion_objective(problem, state, memo)
            candidates.append((problem, state, value, state_features(problem, state)))
        local: List[RankingPairRecord] = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                left = candidates[i]
                right = candidates[j]
                if left[2] == right[2]:
                    continue
                pos, neg = (left, right) if left[2] < right[2] else (right, left)
                local.append(
                    RankingPairRecord(
                        positive_features=list(pos[3]),
                        negative_features=list(neg[3]),
                        feature_names=list(FEATURE_NAMES),
                        label_metadata={
                            "label_rule": "lower_best_descendant_objective_preferred",
                            "positive_best_descendant_objective": pos[2],
                            "negative_best_descendant_objective": neg[2],
                        },
                        trace_ids=(f"{pos[0].pid}:{pos[1].key()}", f"{neg[0].pid}:{neg[1].key()}"),
                        source_split=source_split,
                    )
                )
        rng.shuffle(local)
        records.extend(local[:max_pairs_per_problem])
    rng.shuffle(records)
    return records


def _assert_no_forbidden_inference_features(feature_names: Sequence[str]) -> None:
    forbidden = set(FORBIDDEN_INFERENCE_FEATURES).intersection(feature_names)
    if forbidden:
        raise ValueError(f"oracle/future label fields cannot be inference features: {sorted(forbidden)}")


__all__ = ["FORBIDDEN_INFERENCE_FEATURES", "RankingPairRecord", "make_labeled_ranking_pairs", "make_ranking_pairs"]
