def false_prune_rate(false_prunes: int, pruned: int) -> float:
    return 0.0 if pruned == 0 else false_prunes / pruned

