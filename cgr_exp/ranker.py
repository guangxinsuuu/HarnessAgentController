from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from typing import Dict, Iterable, List, Sequence, Tuple


Pair = Tuple[List[float], List[float]]


@dataclass
class LinearRanker:
    weights: List[float]
    mean: List[float]
    scale: List[float]

    @classmethod
    def fresh(cls, dim: int) -> "LinearRanker":
        return cls([0.0] * dim, [0.0] * dim, [1.0] * dim)

    def _norm(self, x: Sequence[float]) -> List[float]:
        return [(x[i] - self.mean[i]) / self.scale[i] for i in range(len(x))]

    def score(self, x: Sequence[float]) -> float:
        z = self._norm(x)
        return sum(w * v for w, v in zip(self.weights, z))

    def fit_normalizer(self, xs: Sequence[Sequence[float]]) -> None:
        if not xs:
            return
        dim = len(xs[0])
        self.mean = [sum(x[i] for x in xs) / len(xs) for i in range(dim)]
        self.scale = []
        for i in range(dim):
            var = sum((x[i] - self.mean[i]) ** 2 for x in xs) / len(xs)
            self.scale.append(max(1e-6, math.sqrt(var)))

    def train(self, pairs: Sequence[Pair], epochs: int = 8, lr: float = 0.05, l2: float = 1e-4, seed: int = 0, refit_normalizer: bool = True) -> None:
        if not pairs:
            return
        if refit_normalizer:
            xs = [x for pair in pairs for x in pair]
            self.fit_normalizer(xs)
        rng = random.Random(seed)
        order = list(range(len(pairs)))
        for _ in range(epochs):
            rng.shuffle(order)
            for idx in order:
                xp, xn = pairs[idx]
                dp = self._norm(xp)
                dn = self._norm(xn)
                diff = [a - b for a, b in zip(dp, dn)]
                margin = sum(w * d for w, d in zip(self.weights, diff))
                grad_factor = -1.0 / (1.0 + math.exp(max(-40.0, min(40.0, margin))))
                for i in range(len(self.weights)):
                    self.weights[i] -= lr * (grad_factor * diff[i] + l2 * self.weights[i])

    def ranking_error(self, pairs: Sequence[Pair]) -> float:
        if not pairs:
            return 0.0
        bad = sum(1 for xp, xn in pairs if self.score(xp) <= self.score(xn))
        return bad / len(pairs)

    def to_json(self) -> str:
        return json.dumps({"weights": self.weights, "mean": self.mean, "scale": self.scale})

    @classmethod
    def from_json(cls, text: str) -> "LinearRanker":
        obj = json.loads(text)
        return cls(obj["weights"], obj["mean"], obj["scale"])


def _domain_pairs(source: Sequence[List[float]], target: Sequence[List[float]]) -> List[Pair]:
    pairs: List[Pair] = []
    n = min(len(source), len(target))
    for i in range(n):
        pairs.append((target[i], source[i]))
    return pairs


def _pairwise_auc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> float:
    total = 0
    wins = 0.0
    for pos in positive_scores:
        for neg in negative_scores:
            total += 1
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return 0.5 if total == 0 else wins / total


def train_domain_classifier_metrics(source: Sequence[List[float]], target: Sequence[List[float]], seed: int = 0) -> Dict[str, float | str | int]:
    """Train a balanced domain classifier and report held-out discrepancy.

    The proxy is the balanced proxy A-distance form
    ``2 * (2 * heldout_accuracy - 1)``, clipped to ``[0, 2]``. Values near
    0 mean source/target are difficult to distinguish; values near 2 mean a
    linear classifier separates them easily.
    """

    pairs = _domain_pairs(source, target)
    if not pairs:
        return {
            "heldout_accuracy": 0.5,
            "auc": 0.5,
            "proxy_discrepancy": 0.0,
            "train_pair_count": 0,
            "heldout_pair_count": 0,
            "proxy_formula": "2 * (2 * heldout_accuracy - 1), clipped to [0, 2]",
        }
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)
    split = max(1, int(0.7 * len(order)))
    if split == len(order) and len(order) > 1:
        split = len(order) - 1
    train_pairs = [pairs[i] for i in order[:split]]
    heldout_pairs = [pairs[i] for i in order[split:]] or train_pairs
    clf = LinearRanker.fresh(len(pairs[0][0]))
    clf.train(train_pairs, epochs=8, lr=0.03, seed=seed)
    positives = [target_features for target_features, _ in heldout_pairs]
    negatives = [source_features for _, source_features in heldout_pairs]
    positive_scores = [clf.score(x) for x in positives]
    negative_scores = [clf.score(x) for x in negatives]
    threshold = (sum(positive_scores) / len(positive_scores) + sum(negative_scores) / len(negative_scores)) / 2.0
    correct = sum(1 for score in positive_scores if score >= threshold)
    correct += sum(1 for score in negative_scores if score < threshold)
    accuracy = correct / max(1, len(positive_scores) + len(negative_scores))
    auc = _pairwise_auc(positive_scores, negative_scores)
    proxy = max(0.0, min(2.0, 2.0 * (2.0 * accuracy - 1.0)))
    return {
        "heldout_accuracy": accuracy,
        "auc": auc,
        "proxy_discrepancy": proxy,
        "train_pair_count": len(train_pairs),
        "heldout_pair_count": len(heldout_pairs),
        "proxy_formula": "2 * (2 * heldout_accuracy - 1), clipped to [0, 2]",
    }


def train_domain_classifier(source: Sequence[List[float]], target: Sequence[List[float]], seed: int = 0) -> float:
    return float(train_domain_classifier_metrics(source, target, seed=seed)["proxy_discrepancy"])
