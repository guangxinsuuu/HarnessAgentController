from __future__ import annotations

import math


def pairwise_logistic_loss(score_positive: float, score_negative: float) -> float:
    return math.log1p(math.exp(-(score_positive - score_negative)))

