def memory_gate(similarity: float, confidence: float) -> float:
    if similarity <= 0.0 or confidence <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (similarity + confidence) / 2.0))


def gated_fusion(base_score: float, gate: float, episodic_prior: float) -> float:
    return base_score + max(0.0, min(1.0, gate)) * episodic_prior
